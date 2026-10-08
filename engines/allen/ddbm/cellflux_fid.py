import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torchmetrics.image.fid import FrechetInceptionDistance

from . import dist_util, logger
from .karras_diffusion import (
    ECSI_SAMPLER_OPTIONS,
    ECSI_SAMPLERS,
    ECSI_SECOND_ORDER_STEP_GUARD,
    karras_sample,
)

CELLFLUX_FID_PIXEL_PROTOCOL = "uint8_real_round_generated_floor"


def _safe_retained_png_path(molecule, target_key):
    molecule_path = Path(str(molecule))
    target_key = str(target_key)
    if molecule_path.is_absolute() or ".." in molecule_path.parts:
        raise ValueError(f"Unsafe molecule path: {molecule!r}")
    if Path(target_key).name != target_key or target_key in {"", ".", ".."}:
        raise ValueError(f"Unsafe CellFlux target key: {target_key!r}")
    return molecule_path / f"{target_key}.png"


def save_cellflux_fid_png_batch(images, molecules, target_keys, image_root):
    """Persist the exact BCHW uint8 pixels consumed by matched FID."""
    if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected BCHW uint8 RGB images, got {images.dtype} {tuple(images.shape)}")
    if len(images) != len(molecules) or len(images) != len(target_keys):
        raise ValueError("Image, molecule, and target-key batch lengths differ")
    image_root = Path(image_root)
    records = []
    for image, molecule, target_key in zip(images, molecules, target_keys):
        relative_path = _safe_retained_png_path(molecule, target_key)
        output_path = image_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite generated image: {output_path}")
        Image.fromarray(image.permute(1, 2, 0).contiguous().numpy(), mode="RGB").save(output_path)
        records.append(
            {
                "relative_path": relative_path.as_posix(),
                "target_key": str(target_key),
                "png_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            }
        )
    return records


def cellflux_fid_pixels(images, real):
    """Match CellFlux's 8-bit preprocessing before Inception features.

    Upstream CellFlux dequantizes source uint8 images, maps them to ``[-1, 1]``,
    then floors both endpoints back onto the 8-bit grid before FID.  This
    adapter keeps evaluation images noise-free, so adding half a bin before
    flooring recovers their source uint8 value. Generated continuous values
    use CellFlux's original floor operation without that offset.
    """
    pixels = images.add(1).mul(127.5)
    if real:
        pixels = pixels.add(0.5)
    return pixels.floor().clamp(0, 255).to(torch.float32).div(255)


def cellflux_fid_rgb(images):
    """Apply CellFlux's dataset-specific channel projection before FID.

    BBBC021 is already RGB. JUMP/CPG0000 uses the first three channels. RxRx1
    uses the released CellFlux six-stain-to-RGB mixing matrix.
    """
    if images.ndim != 4 or images.shape[1] not in {3, 5, 6}:
        raise ValueError(f"Expected BCHW CellFlux 3-, 5-, or 6-channel images, got {tuple(images.shape)}")
    if images.shape[1] == 6:
        weights = images.new_tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ]
        )
        return torch.einsum("bchw,cn->bnhw", images, weights).clamp(-1, 1)
    return images[:, :3]


class CellFluxFIDEvaluator:
    """Matched CellFlux FID using the training process and bridge sampler."""

    def __init__(
        self,
        diffusion,
        dataloader,
        output_dir,
        nfe=20,
        sampler="dbim",
        eta=0.0,
        order=2,
        cfg_scale=0.2,
        seed=42,
        use_fp16=True,
        rho=7.0,
        ecsi_reconstruction_steps=2,
        retain_pngs=False,
        moa_eval_repo=None,
        moa_checkpoint=None,
        moa_config=None,
        moa_batch_size=32,
        moa_num_workers=10,
        jump_retrieval_script=None,
        jump_retrieval_metadata=None,
        jump_retrieval_image_root=None,
        jump_retrieval_weights=None,
        jump_retrieval_weights_sha256=None,
        jump_retrieval_real_cache=None,
        jump_retrieval_expected_key_sha256=None,
        allencell_retrieval_script=None,
        allencell_retrieval_metadata=None,
        allencell_retrieval_image_root=None,
        allencell_retrieval_weights=None,
        allencell_retrieval_weights_sha256=None,
        allencell_retrieval_real_cache=None,
        allencell_retrieval_expected_key_sha256=None,
        allen_moa_manifest=None,
        allen_moa_heads=None,
    ):
        if sampler not in {"dbim", "dbim_high_order", *ECSI_SAMPLERS}:
            raise ValueError(f"Unsupported formal CellFlux FID sampler: {sampler}")
        if nfe < 2:
            raise ValueError(f"DBIM NFE must be at least 2, got {nfe}")
        self.diffusion = diffusion
        self.dataloader = dataloader
        self.output_dir = Path(output_dir)
        self.nfe = nfe
        self.sampler = sampler
        self.eta = eta
        self.order = order
        self.cfg_scale = cfg_scale
        self.seed = seed
        self.use_fp16 = use_fp16
        self.rho = rho
        self.ecsi_reconstruction_steps = ecsi_reconstruction_steps
        self.retain_pngs = bool(retain_pngs)
        self.moa_eval_repo = Path(moa_eval_repo) if moa_eval_repo else None
        self.moa_checkpoint = Path(moa_checkpoint) if moa_checkpoint else None
        self.moa_config = Path(moa_config) if moa_config else None
        self.moa_batch_size = int(moa_batch_size)
        self.moa_num_workers = int(moa_num_workers)
        self.allen_moa_manifest = allen_moa_manifest
        self.allen_moa_heads = allen_moa_heads
        if allen_moa_manifest:
            if not retain_pngs or moa_eval_repo:
                raise ValueError('Allen frozen MoA needs retained PNGs and no BBBC evaluator')
            from .allen_moa import validate_inputs
            validate_inputs(allen_moa_manifest, allen_moa_heads)
        self.jump_retrieval_script = Path(jump_retrieval_script) if jump_retrieval_script else None
        self.jump_retrieval_metadata = Path(jump_retrieval_metadata) if jump_retrieval_metadata else None
        self.jump_retrieval_image_root = Path(jump_retrieval_image_root) if jump_retrieval_image_root else None
        self.jump_retrieval_weights = Path(jump_retrieval_weights) if jump_retrieval_weights else None
        self.jump_retrieval_weights_sha256 = jump_retrieval_weights_sha256
        self.jump_retrieval_real_cache = Path(jump_retrieval_real_cache) if jump_retrieval_real_cache else None
        self.jump_retrieval_expected_key_sha256 = jump_retrieval_expected_key_sha256
        self.allencell_retrieval_script = (
            Path(allencell_retrieval_script) if allencell_retrieval_script else None
        )
        self.allencell_retrieval_metadata = (
            Path(allencell_retrieval_metadata) if allencell_retrieval_metadata else None
        )
        self.allencell_retrieval_image_root = (
            Path(allencell_retrieval_image_root) if allencell_retrieval_image_root else None
        )
        self.allencell_retrieval_weights = (
            Path(allencell_retrieval_weights) if allencell_retrieval_weights else None
        )
        self.allencell_retrieval_weights_sha256 = allencell_retrieval_weights_sha256
        self.allencell_retrieval_real_cache = (
            Path(allencell_retrieval_real_cache) if allencell_retrieval_real_cache else None
        )
        self.allencell_retrieval_expected_key_sha256 = (
            allencell_retrieval_expected_key_sha256
        )
        if self.moa_eval_repo is not None:
            if not self.retain_pngs:
                raise ValueError("Inline MoA requires retained FID PNGs")
            if self.moa_checkpoint is None or self.moa_config is None:
                raise ValueError("Inline MoA requires checkpoint and config paths")
            if self.moa_batch_size <= 0 or self.moa_num_workers < 0:
                raise ValueError("Invalid inline MoA batch/worker configuration")
        if self.jump_retrieval_script is not None:
            if not self.retain_pngs:
                raise ValueError("Inline JUMP retrieval requires retained FID PNGs")
            required = (
                self.jump_retrieval_script,
                self.jump_retrieval_metadata,
                self.jump_retrieval_image_root,
                self.jump_retrieval_weights,
                self.jump_retrieval_real_cache,
            )
            if any(path is None for path in required):
                raise ValueError("Inline JUMP retrieval paths are incomplete")
            if not self.jump_retrieval_weights_sha256 or not self.jump_retrieval_expected_key_sha256:
                raise ValueError("Inline JUMP retrieval hashes are incomplete")
        if self.allencell_retrieval_script is not None:
            if not self.retain_pngs:
                raise ValueError("Inline Allen retrieval requires retained FID PNGs")
            required = (
                self.allencell_retrieval_script,
                self.allencell_retrieval_metadata,
                self.allencell_retrieval_image_root,
                self.allencell_retrieval_weights,
                self.allencell_retrieval_real_cache,
            )
            if any(path is None for path in required):
                raise ValueError("Inline Allen retrieval paths are incomplete")
            if not self.allencell_retrieval_weights_sha256 or not self.allencell_retrieval_expected_key_sha256:
                raise ValueError("Inline Allen retrieval hashes are incomplete")

    def retained_image_root(self, completed_epoch):
        return self.output_dir.parent / "fid_samples" / f"epoch-{completed_epoch}"

    def moa_result_path(self, completed_epoch):
        return self.output_dir.parent / "moa" / f"moa_e{completed_epoch:04d}.json"

    def jump_retrieval_result_path(self, completed_epoch):
        return self.output_dir.parent / "retrieval" / f"retrieval_e{completed_epoch:04d}.json"

    def allencell_retrieval_result_path(self, completed_epoch):
        return self.output_dir.parent / "retrieval" / f"allencell_retrieval_e{completed_epoch:04d}.json"

    def _read_complete_allencell_retrieval(self, completed_epoch):
        if self.allencell_retrieval_script is None:
            return None
        path = self.allencell_retrieval_result_path(completed_epoch)
        if not path.is_file():
            return None
        try:
            result = json.loads(path.read_text())
            retrieval = result["generated_test_to_real_train"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return None
        if result.get("generated_sample_count") != len(self.dataloader.dataset):
            return None
        if not all(
            math.isfinite(float(retrieval.get(key, math.nan)))
            for key in ("recall_at_1", "recall_at_2", "mrr", "mean_rank", "median_rank")
        ):
            return None
        return result

    def run_allencell_retrieval_if_due(self, completed_epoch):
        if self.allencell_retrieval_script is None:
            return None
        result = None
        if dist.get_rank() == 0:
            result = self._read_complete_allencell_retrieval(completed_epoch)
            if result is None:
                output = self.allencell_retrieval_result_path(completed_epoch)
                output.parent.mkdir(parents=True, exist_ok=True)
                generated_cache = output.with_suffix(".features.npz")
                subprocess.run(
                    [
                        sys.executable,
                        str(self.allencell_retrieval_script),
                        "--generated-root", str(self.retained_image_root(completed_epoch)),
                        "--metadata-path", str(self.allencell_retrieval_metadata),
                        "--image-root", str(self.allencell_retrieval_image_root),
                        "--encoder-weights-path", str(self.allencell_retrieval_weights),
                        "--encoder-weights-sha256", self.allencell_retrieval_weights_sha256,
                        "--expected-generated-key-sha256", self.allencell_retrieval_expected_key_sha256,
                        "--expected-generated-count", str(len(self.dataloader.dataset)),
                        "--real-feature-cache", str(self.allencell_retrieval_real_cache),
                        "--generated-feature-cache", str(generated_cache),
                        "--output", str(output),
                        "--batch-size", "64",
                        "--num-workers", "16",
                        "--device", "cuda",
                    ],
                    check=True,
                )
                result = self._read_complete_allencell_retrieval(completed_epoch)
                if result is None:
                    raise RuntimeError(f"Inline Allen retrieval failed validation: {output}")
                retrieval = result["generated_test_to_real_train"]
                logger.log(
                    f"Allen retrieval e{completed_epoch}: R@1={retrieval['recall_at_1']:.6f}, "
                    f"R@2={retrieval['recall_at_2']:.6f}, MRR={retrieval['mrr']:.6f}"
                )
        dist.barrier()
        return result

    def _read_complete_jump_retrieval(self, completed_epoch):
        if self.jump_retrieval_script is None:
            return None
        path = self.jump_retrieval_result_path(completed_epoch)
        if not path.is_file():
            return None
        try:
            result = json.loads(path.read_text())
            retrieval = result["generated_cpr_at_5"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return None
        if result.get("generated_sample_count") != len(self.dataloader.dataset):
            return None
        if not all(
            math.isfinite(float(retrieval.get(key, math.nan)))
            for key in ("cpr_at_k", "strict_cpr_at_1", "mean_target_rank")
        ):
            return None
        return result

    def run_jump_retrieval_if_due(self, completed_epoch):
        if self.jump_retrieval_script is None:
            return None
        result = None
        if dist.get_rank() == 0:
            result = self._read_complete_jump_retrieval(completed_epoch)
            if result is None:
                output = self.jump_retrieval_result_path(completed_epoch)
                output.parent.mkdir(parents=True, exist_ok=True)
                generated_cache = output.with_suffix(".features.npz")
                subprocess.run(
                    [
                        sys.executable,
                        str(self.jump_retrieval_script),
                        "--generated-root", str(self.retained_image_root(completed_epoch)),
                        "--metadata-path", str(self.jump_retrieval_metadata),
                        "--image-root", str(self.jump_retrieval_image_root),
                        "--encoder-weights-path", str(self.jump_retrieval_weights),
                        "--encoder-weights-sha256", self.jump_retrieval_weights_sha256,
                        "--expected-generated-key-sha256", self.jump_retrieval_expected_key_sha256,
                        "--expected-generated-count", str(len(self.dataloader.dataset)),
                        "--real-feature-cache", str(self.jump_retrieval_real_cache),
                        "--generated-feature-cache", str(generated_cache),
                        "--output", str(output),
                        "--batch-size", "64",
                        "--num-workers", "16",
                        "--device", "cuda",
                        "--skip-real-ceiling",
                    ],
                    check=True,
                )
                result = self._read_complete_jump_retrieval(completed_epoch)
                if result is None:
                    raise RuntimeError(f"Inline JUMP retrieval failed validation: {output}")
                retrieval = result["generated_cpr_at_5"]
                logger.log(
                    f"JUMP retrieval e{completed_epoch}: CPR@5={retrieval['cpr_at_k']:.6f}, "
                    f"top1={retrieval['strict_cpr_at_1']:.6f}"
                )
        dist.barrier()
        return result

    def _read_complete_moa(self, completed_epoch):
        if self.moa_eval_repo is None and not self.allen_moa_manifest:
            return None
        path = self.moa_result_path(completed_epoch)
        if not path.is_file():
            return None
        try:
            result = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if result.get("total") != len(self.dataloader.dataset):
            return None
        if not all(
            math.isfinite(float(result.get(key, math.nan)))
            for key in ("accuracy", "macro_f1", "weighted_f1")
        ):
            return None
        return result

    def _run_moa(self, completed_epoch, image_root):
        output_path = self.moa_result_path(completed_epoch)
        existing = self._read_complete_moa(completed_epoch)
        if existing is not None:
            return existing
        if self.allen_moa_manifest:
            from .allen_moa import score_images
            return score_images(image_root, output_path, self.allen_moa_manifest,
                                self.allen_moa_heads, torch.device('cuda', torch.cuda.current_device()))
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite incomplete inline MoA result: {output_path}")
        for path in (
            self.moa_eval_repo / "moa" / "train_moa.py",
            self.moa_checkpoint,
            self.moa_config,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(".json.incomplete")
        if temporary_path.exists():
            raise FileExistsError(f"Refusing to overwrite inline MoA staging result: {temporary_path}")
        environment = os.environ.copy()
        prior_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            item for item in (str(self.moa_eval_repo), prior_pythonpath) if item
        )
        subprocess.run(
            [
                sys.executable,
                str(self.moa_eval_repo / "moa" / "train_moa.py"),
                "--config_path",
                str(self.moa_config),
                "--mode=eval",
                "--selection=official",
                f"--max_samples={len(self.dataloader.dataset)}",
                "--sample_seed=0",
                f"--img_root_path={image_root}",
                f"--ckpt_path={self.moa_checkpoint}",
                f"--batch_size={self.moa_batch_size}",
                f"--num_workers={self.moa_num_workers}",
                f"--output_json={temporary_path}",
            ],
            cwd=self.moa_eval_repo,
            env=environment,
            check=True,
        )
        try:
            result = json.loads(temporary_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Inline MoA staging result is invalid: {temporary_path}") from error
        if result.get("total") != len(self.dataloader.dataset) or not all(
            math.isfinite(float(result.get(key, math.nan)))
            for key in ("accuracy", "macro_f1", "weighted_f1")
        ):
            raise RuntimeError(f"Inline MoA staging result failed validation: {temporary_path}")
        temporary_path.replace(output_path)
        result = self._read_complete_moa(completed_epoch)
        if result is None:
            raise RuntimeError(f"Inline MoA result failed validation: {output_path}")
        return result

    def _ecsi_variant_metadata(self):
        if self.sampler not in ECSI_SAMPLERS or self.sampler == "ecsi":
            return {}
        options = ECSI_SAMPLER_OPTIONS[self.sampler]
        metadata = {
            "eta_schedule": options["eta_schedule"],
            "second_order_x0": options["second_order_x0"],
        }
        if options["eta_schedule"] == "cosine":
            metadata["final_eta_forced_zero"] = True
        if "eta_schedule_power" in options:
            metadata["eta_schedule_power"] = options["eta_schedule_power"]
        if options["second_order_x0"]:
            metadata["second_order_step_guard"] = ECSI_SECOND_ORDER_STEP_GUARD
            metadata["second_order_x0_clamp"] = [-1.0, 1.0]
        if options.get("second_order_lower_order_tail", False):
            metadata["second_order_lower_order_tail"] = True
        if "second_order_scale" in options:
            metadata["second_order_scale"] = options["second_order_scale"]
        if "second_order_scale_schedule" in options:
            metadata["second_order_scale_schedule"] = options[
                "second_order_scale_schedule"
            ]
        if "second_order_blend_weight" in options:
            metadata["second_order_blend_weight"] = options[
                "second_order_blend_weight"
            ]
        if "second_order_tail_scale" in options:
            metadata["second_order_tail_scale"] = options[
                "second_order_tail_scale"
            ]
        if "second_order_progress_window" in options:
            metadata["second_order_progress_window"] = options[
                "second_order_progress_window"
            ]
        if "time_grid" in options:
            metadata["time_grid"] = options["time_grid"]
        for key in (
            "sde_integrator",
            "solver_order",
            "solver_type",
            "solver_lower_order_tail",
            "terminal_lower_order",
            "reconstruction_tail_eta",
        ):
            if key in options:
                metadata[key] = options[key]
        return metadata

    def result_path(self, completed_epoch):
        return self.output_dir / f"fid_e{completed_epoch:04d}.json"

    def is_complete(self, completed_epoch):
        path = self.result_path(completed_epoch)
        if not path.is_file():
            return False
        try:
            result = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        expected = {
            "completed_epoch": int(completed_epoch),
            "num_samples": len(self.dataloader.dataset),
            "sampler": self.sampler,
            "nfe": self.nfe,
            "eta": self.eta if self.sampler == "dbim" or self.sampler in ECSI_SAMPLERS else None,
            "order": self.order if self.sampler == "dbim_high_order" else None,
            "rho": self.rho if self.sampler in ECSI_SAMPLERS else None,
            "ecsi_reconstruction_steps": (
                self.ecsi_reconstruction_steps if self.sampler in ECSI_SAMPLERS else None
            ),
            "train_t_min": self.diffusion.t_min if self.sampler in ECSI_SAMPLERS else None,
            "train_t_max": self.diffusion.t_max if self.sampler in ECSI_SAMPLERS else None,
            "sample_t_min": self.diffusion.ecsi_sample_t_min if self.sampler in ECSI_SAMPLERS else None,
            "sample_t_max": self.diffusion.ecsi_sample_t_max if self.sampler in ECSI_SAMPLERS else None,
            "cfg_scale": self.cfg_scale,
            "seed": self.seed,
            "pixel_protocol": CELLFLUX_FID_PIXEL_PROTOCOL,
        }
        expected.update(self._ecsi_variant_metadata())
        complete = (
            all(
                (
                    result.get(key, 2)
                    if key == "ecsi_reconstruction_steps" and self.sampler in ECSI_SAMPLERS
                    else result.get(key)
                )
                == value
                for key, value in expected.items()
            )
            and "fid" in result
            and "pair_sha256" in result
        )
        if not complete or not self.retain_pngs:
            return complete
        image_root = self.retained_image_root(completed_epoch)
        retained_complete = (
            result.get("retained_pngs") is True
            and result.get("retained_image_root") == str(image_root)
            and result.get("retained_png_count") == len(self.dataloader.dataset)
            and len(str(result.get("retained_png_manifest_sha256", ""))) == 64
            and image_root.is_dir()
            and sum(1 for _ in image_root.rglob("*.png")) == len(self.dataloader.dataset)
        )
        if not retained_complete:
            return False
        moa_result = self._read_complete_moa(completed_epoch)
        if self.moa_eval_repo is None and not self.allen_moa_manifest:
            return True
        return (
            moa_result is not None
            and result.get("moa_accuracy") == moa_result["accuracy"]
            and result.get("moa_macro_f1") == moa_result["macro_f1"]
            and result.get("moa_weighted_f1") == moa_result["weighted_f1"]
        )

    @torch.no_grad()
    def __call__(self, model, completed_epoch):
        device = dist_util.dev()
        metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        local_target_keys = []
        local_pairs = []
        local_png_records = []
        was_training = model.training
        model.eval()

        retained_root = self.retained_image_root(completed_epoch)
        staging_root = retained_root.with_name(f".{retained_root.name}.incomplete")
        if self.retain_pngs:
            if dist.get_rank() == 0:
                if retained_root.exists() or staging_root.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite retained FID images: {retained_root} or {staging_root}"
                    )
                staging_root.mkdir(parents=True)
            dist.barrier()

        for x0_cpu, xT_cpu, metadata in self.dataloader:
            x0 = x0_cpu.to(device, non_blocking=True)
            xT = xT_cpu.to(device, non_blocking=True)
            condition = metadata["condition"].to(device, non_blocking=True)
            seeds = metadata["index"].numpy() + self.seed
            model_kwargs = {"xT": xT, "condition": condition}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16):
                generated, _, realized_nfe, _, _, _ = karras_sample(
                    self.diffusion,
                    model,
                    xT,
                    x0,
                    steps=self.nfe if self.sampler in ECSI_SAMPLERS else self.nfe - 1,
                    model_kwargs=model_kwargs,
                    device=device,
                    clip_denoised=True,
                    sampler=self.sampler,
                    eta=self.eta,
                    order=self.order,
                    seed=seeds,
                    cfg_scale=self.cfg_scale,
                    progress=False,
                    rho=self.rho,
                    ecsi_reconstruction_steps=self.ecsi_reconstruction_steps,
                )
            if realized_nfe != self.nfe:
                raise RuntimeError(f"Expected {self.sampler} NFE={self.nfe}, got {realized_nfe}")
            real = cellflux_fid_pixels(cellflux_fid_rgb(x0), real=True)
            generated_rgb = cellflux_fid_rgb(generated)
            fake_uint8 = generated_rgb.add(1).mul(127.5).floor().clamp(0, 255).to(torch.uint8)
            fake = fake_uint8.to(torch.float32).div(255)
            metric.update(real, real=True)
            metric.update(fake, real=False)
            batch_target_keys = [str(key) for key in metadata["target_key"]]
            local_target_keys.extend(batch_target_keys)
            local_pairs.extend(
                f"{target}\t{control}"
                for target, control in zip(metadata["target_key"], metadata["control_key"])
            )
            if self.retain_pngs:
                local_png_records.extend(
                    save_cellflux_fid_png_batch(
                        fake_uint8.cpu(),
                        [str(value) for value in metadata["molecule"]],
                        batch_target_keys,
                        staging_root,
                    )
                )
        fid = float(metric.compute().detach().cpu())
        gathered_keys = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_keys, local_target_keys)
        all_target_keys = [key for rank_keys in gathered_keys for key in rank_keys]
        expected_count = len(self.dataloader.dataset)
        if len(all_target_keys) != expected_count or len(set(all_target_keys)) != expected_count:
            raise RuntimeError(
                "Matched FID target set is not exact: "
                f"rows={len(all_target_keys)}, unique={len(set(all_target_keys))}, expected={expected_count}"
            )
        key_hash = hashlib.sha256("\n".join(sorted(all_target_keys)).encode()).hexdigest()
        gathered_pairs = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_pairs, local_pairs)
        all_pairs = [pair for rank_pairs in gathered_pairs for pair in rank_pairs]
        if len(all_pairs) != expected_count or len(set(all_pairs)) != expected_count:
            raise RuntimeError(
                "Matched FID pair set is not exact: "
                f"rows={len(all_pairs)}, unique={len(set(all_pairs))}, expected={expected_count}"
            )
        pair_hash = hashlib.sha256("\n".join(sorted(all_pairs)).encode()).hexdigest()

        png_manifest_sha256 = None
        if self.retain_pngs:
            gathered_records = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_records, local_png_records)
            all_records = [record for rank_records in gathered_records for record in rank_records]
            if dist.get_rank() == 0:
                png_count = sum(1 for _ in staging_root.rglob("*.png"))
                if len(all_records) != expected_count or png_count != expected_count:
                    raise RuntimeError(
                        f"Retained FID tree is incomplete: records={len(all_records)}, "
                        f"pngs={png_count}, expected={expected_count}"
                    )
                record_lines = [
                    "\t".join((item["relative_path"], item["target_key"], item["png_sha256"]))
                    for item in all_records
                ]
                png_manifest_sha256 = hashlib.sha256(
                    "\n".join(sorted(record_lines)).encode()
                ).hexdigest()
                staging_root.replace(retained_root)

        moa_result = None
        if dist.get_rank() == 0 and (self.moa_eval_repo is not None or self.allen_moa_manifest):
            torch.cuda.empty_cache()
            moa_result = self._run_moa(completed_epoch, retained_root)

        if dist.get_rank() == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            result = {
                "completed_epoch": int(completed_epoch),
                "fid": fid,
                "num_samples": expected_count,
                "target_key_sha256": key_hash,
                "pair_sha256": pair_hash,
                "sampler": self.sampler,
                "nfe": self.nfe,
                "eta": self.eta if self.sampler == "dbim" or self.sampler in ECSI_SAMPLERS else None,
                "order": self.order if self.sampler == "dbim_high_order" else None,
                "rho": self.rho if self.sampler in ECSI_SAMPLERS else None,
                "ecsi_reconstruction_steps": (
                    self.ecsi_reconstruction_steps if self.sampler in ECSI_SAMPLERS else None
                ),
                "train_t_min": self.diffusion.t_min if self.sampler in ECSI_SAMPLERS else None,
                "train_t_max": self.diffusion.t_max if self.sampler in ECSI_SAMPLERS else None,
                "sample_t_min": self.diffusion.ecsi_sample_t_min if self.sampler in ECSI_SAMPLERS else None,
                "sample_t_max": self.diffusion.ecsi_sample_t_max if self.sampler in ECSI_SAMPLERS else None,
                "cfg_scale": self.cfg_scale,
                "seed": self.seed,
                "pixel_protocol": CELLFLUX_FID_PIXEL_PROTOCOL,
            }
            result.update(self._ecsi_variant_metadata())
            if self.retain_pngs:
                result.update(
                    retained_pngs=True,
                    retained_image_root=str(retained_root),
                    retained_png_count=expected_count,
                    retained_png_manifest_sha256=png_manifest_sha256,
                )
            if moa_result is not None:
                result.update(
                    moa_accuracy=moa_result["accuracy"],
                    moa_macro_f1=moa_result["macro_f1"],
                    moa_weighted_f1=moa_result["weighted_f1"],
                    moa_result_path=str(self.moa_result_path(completed_epoch)),
                )
            output_path = self.result_path(completed_epoch)
            temporary_path = output_path.with_suffix(".json.tmp")
            with temporary_path.open("w") as handle:
                json.dump(result, handle, indent=2, sort_keys=True)
            temporary_path.replace(output_path)
            logger.logkv(f"fid_e{completed_epoch}", fid)
            logger.log(
                f"CellFlux matched FID e{completed_epoch}: {fid:.6f} "
                f"({expected_count} unique targets, {self.sampler}, NFE={self.nfe}, CFG={self.cfg_scale})"
            )
        dist.barrier()
        if was_training:
            model.train()
        return fid
