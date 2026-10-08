"""Reusable feature-space auxiliary losses for bridge training.

The default encoder deliberately matches the TorchMetrics/Torch-Fidelity
InceptionV3 used by :mod:`ddbm.cellflux_fid`.  Its forward below follows
``torch_fidelity.feature_extractor_inceptionv3`` (Apache-2.0) while accepting
floating point pixels so gradients can reach the denoiser output.
"""

from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.interpolate_compat_tensorflow import (
    interpolate_bilinear_2d_like_tensorflow1x,
)


AUXILIARY_PREPROCESSING = "st_clamp_unit_centered_dither_float255_v1"
FID_INCEPTION_ENCODER = "torch_fidelity_inception_v3_2048"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_condition_moa_labels(metadata_path, condition_names):
    """Resolve one stable MoA label for every cached molecular condition."""
    expected = tuple(str(name) for name in condition_names)
    expected_set = set(expected)
    labels_by_condition = {}
    with open(metadata_path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"CPD_NAME", "ANNOT"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("MoA metadata must contain CPD_NAME and ANNOT columns")
        for row in reader:
            condition = str(row["CPD_NAME"])
            if condition not in expected_set:
                continue
            label = str(row["ANNOT"]).strip()
            if not label:
                raise ValueError(f"condition {condition!r} has an empty MoA label")
            previous = labels_by_condition.setdefault(condition, label)
            if previous != label:
                raise ValueError(
                    f"condition {condition!r} maps to multiple MoA labels: "
                    f"{previous!r} and {label!r}"
                )
    missing = sorted(expected_set.difference(labels_by_condition))
    if missing:
        raise ValueError(f"missing MoA labels for conditions: {missing}")
    moa_names = sorted(set(labels_by_condition.values()))
    if len(moa_names) < 2:
        raise ValueError("supervised InfoNCE requires at least two MoA classes")
    moa_id_by_name = {name: index for index, name in enumerate(moa_names)}
    prototype_moa_ids = torch.tensor(
        [moa_id_by_name[labels_by_condition[name]] for name in expected],
        dtype=torch.long,
    )
    return {
        "moa_names": moa_names,
        "prototype_moa_ids": prototype_moa_ids,
    }


def auxiliary_encoder_input(
    images,
    *,
    straight_through=True,
    add_dither=True,
    generator=None,
):
    """Map bridge images in ``[-1, 1]`` to differentiable 8-bit-scale pixels.

    Generated images use a straight-through clamp both before and after the
    centered one-bin dither.  Real-image cache construction uses the same
    function with ``straight_through=False`` and a seeded generator.
    """
    if not torch.is_tensor(images) or images.ndim != 4:
        raise ValueError("auxiliary images must be an NCHW tensor")
    images = images.to(torch.float32)
    bounded = images.clamp(-1.0, 1.0)
    if straight_through:
        bounded = images + (bounded - images).detach()
    unit = bounded.add(1.0).mul(0.5)
    if add_dither:
        dither = torch.rand(
            unit.shape,
            dtype=unit.dtype,
            device=unit.device,
            generator=generator,
        ).sub(0.5).div(255.0)
        unit = unit + dither
    clipped = unit.clamp(0.0, 1.0)
    if straight_through:
        clipped = unit + (clipped - unit).detach()
    return clipped.mul(255.0)


def cfg_distillation_loss(
    conditional,
    unconditional,
    loss_weights,
    *,
    student_cfg_scale,
    teacher_cfg_scale,
    clip_denoised=True,
    teacher_conditional=None,
    teacher_unconditional=None,
):
    """Distill a stronger CellFlux CFG prediction into a weaker one.

    By default the teacher is the model's current classifier-free guided
    prediction.  Callers may instead provide a fixed pair of conditional and
    unconditional teacher predictions.  Teacher branches are always detached,
    while the lower-CFG student keeps gradients through both of its branches.
    Clipping uses a straight-through estimator and mirrors sampling.
    """
    if conditional.shape != unconditional.shape:
        raise ValueError("conditional and unconditional predictions must have matching shapes")
    if conditional.ndim < 2:
        raise ValueError("CFG distillation expects a batched prediction tensor")
    if student_cfg_scale < 0 or teacher_cfg_scale <= student_cfg_scale:
        raise ValueError("CFG distillation requires 0 <= student scale < teacher scale")
    loss_weights = torch.as_tensor(
        loss_weights,
        device=conditional.device,
        dtype=conditional.dtype,
    ).reshape(-1)
    if loss_weights.shape[0] != conditional.shape[0]:
        raise ValueError("CFG distillation requires one loss weight per sample")
    if not torch.isfinite(loss_weights).all() or (loss_weights < 0).any():
        raise ValueError("CFG distillation loss weights must be finite and nonnegative")

    if (teacher_conditional is None) != (teacher_unconditional is None):
        raise ValueError("fixed CFG teacher requires both conditional and unconditional predictions")
    if teacher_conditional is None:
        conditional_teacher = conditional.detach()
        unconditional_teacher = unconditional.detach()
    else:
        if teacher_conditional.shape != conditional.shape:
            raise ValueError("teacher and student conditional predictions must have matching shapes")
        if teacher_unconditional.shape != unconditional.shape:
            raise ValueError("teacher and student unconditional predictions must have matching shapes")
        conditional_teacher = teacher_conditional.detach()
        unconditional_teacher = teacher_unconditional.detach()
    guidance_delta = conditional_teacher - unconditional_teacher
    teacher = conditional_teacher + float(teacher_cfg_scale) * guidance_delta
    unclipped_teacher = teacher
    student = conditional + float(student_cfg_scale) * (conditional - unconditional)
    if clip_denoised:
        teacher = teacher.clamp(-1.0, 1.0)
        student = student + (student.clamp(-1.0, 1.0) - student).detach()

    reduce_dims = tuple(range(1, conditional.ndim))
    squared_error = (student - teacher).square().mean(dim=reduce_dims)
    delta_rms = guidance_delta.square().mean(dim=reduce_dims).sqrt()
    target_shift_rms = (teacher - conditional_teacher).square().mean(dim=reduce_dims).sqrt()
    student_target_gap_rms = (teacher - student.detach()).square().mean(dim=reduce_dims).sqrt()
    teacher_cond_gap_rms = (
        conditional_teacher - conditional.detach()
    ).square().mean(dim=reduce_dims).sqrt()
    teacher_uncond_gap_rms = (
        unconditional_teacher - unconditional.detach()
    ).square().mean(dim=reduce_dims).sqrt()
    clip_fraction = (unclipped_teacher != teacher).to(conditional.dtype).mean(dim=reduce_dims)
    return {
        "total": loss_weights * squared_error,
        "mse": squared_error,
        "delta_rms": delta_rms,
        "target_shift_rms": target_shift_rms,
        "student_target_gap_rms": student_target_gap_rms,
        "teacher_cond_gap_rms": teacher_cond_gap_rms,
        "teacher_uncond_gap_rms": teacher_uncond_gap_rms,
        "clip_fraction": clip_fraction,
    }


class FIDInceptionV3Encoder(FeatureExtractorInceptionV3):
    """Frozen, differentiable 2048-d Torch-Fidelity InceptionV3 encoder."""

    num_features = 2048

    def __init__(self, weights_path, expected_sha256=None):
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"Inception weights do not exist: {weights_path}")
        actual_sha256 = sha256_file(weights_path)
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ValueError(
                f"Inception weights SHA-256 {actual_sha256} != expected {expected_sha256}"
            )
        super().__init__(
            FID_INCEPTION_ENCODER,
            ["2048"],
            feature_extractor_weights_path=str(weights_path),
        )
        self.weights_sha256 = actual_sha256
        self.eval()
        self.requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    def forward(self, pixels):
        if not torch.is_tensor(pixels) or pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError("FID Inception expects NCHW three-channel pixels")
        if not pixels.is_floating_point():
            raise TypeError("differentiable FID Inception expects floating point pixels")
        device_type = pixels.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            x = pixels.to(torch.float32)
            x = interpolate_bilinear_2d_like_tensorflow1x(
                x,
                size=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE),
                align_corners=False,
            )
            x = (x - 128.0) / 128.0

            x = self.Conv2d_1a_3x3(x)
            x = self.Conv2d_2a_3x3(x)
            x = self.Conv2d_2b_3x3(x)
            x = self.MaxPool_1(x)
            x = self.Conv2d_3b_1x1(x)
            x = self.Conv2d_4a_3x3(x)
            x = self.MaxPool_2(x)
            x = self.Mixed_5b(x)
            x = self.Mixed_5c(x)
            x = self.Mixed_5d(x)
            x = self.Mixed_6a(x)
            x = self.Mixed_6b(x)
            x = self.Mixed_6c(x)
            x = self.Mixed_6d(x)
            x = self.Mixed_6e(x)
            x = self.Mixed_7a(x)
            x = self.Mixed_7b(x)
            x = self.Mixed_7c(x)
            x = self.AvgPool(x)
            return torch.flatten(x, 1)


def load_conditional_statistics(
    path,
    *,
    expected_encoder=FID_INCEPTION_ENCODER,
    expected_weights_sha256=None,
):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"conditional-statistics cache does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "condition_ids",
        "condition_names",
        "counts",
        "mean",
        "variance",
        "encoder",
        "encoder_weights_sha256",
        "preprocessing",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"conditional-statistics cache is missing fields: {missing}")
    if payload["encoder"] != expected_encoder:
        raise ValueError(f"cache encoder {payload['encoder']!r} != {expected_encoder!r}")
    if expected_weights_sha256 and payload["encoder_weights_sha256"] != expected_weights_sha256:
        raise ValueError("conditional-statistics encoder weights do not match the requested weights")
    if payload["preprocessing"] != AUXILIARY_PREPROCESSING:
        raise ValueError(
            f"cache preprocessing {payload['preprocessing']!r} != {AUXILIARY_PREPROCESSING!r}"
        )
    condition_ids = torch.as_tensor(payload["condition_ids"], dtype=torch.long)
    counts = torch.as_tensor(payload["counts"], dtype=torch.long)
    mean = torch.as_tensor(payload["mean"], dtype=torch.float32)
    variance = torch.as_tensor(payload["variance"], dtype=torch.float32)
    expected_ids = torch.arange(condition_ids.numel(), dtype=torch.long)
    if not torch.equal(condition_ids, expected_ids):
        raise ValueError("condition IDs must be contiguous and zero-based")
    if mean.ndim != 2 or variance.shape != mean.shape:
        raise ValueError("cached mean and variance must have the same [conditions, features] shape")
    if counts.shape != condition_ids.shape or len(payload["condition_names"]) != condition_ids.numel():
        raise ValueError("cached condition metadata has inconsistent lengths")
    if mean.shape[1] != 2048:
        raise ValueError(f"expected 2048 cached features, got {mean.shape[1]}")
    if (counts <= 0).any() or not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
        raise ValueError("conditional-statistics cache contains invalid values")
    if (variance < 0).any():
        raise ValueError("conditional-statistics cache contains negative variances")
    payload = dict(payload)
    payload.update(condition_ids=condition_ids, counts=counts, mean=mean, variance=variance)
    return payload


class ConditionalMomentLoss(nn.Module):
    """Match conditional feature means and diagonal variances without loops."""

    def __init__(
        self,
        real_mean,
        real_variance,
        *,
        beta=0.5,
        min_group_size=4,
        eps=1e-6,
        kernel="diagonal_moments",
        ema_enabled=False,
        ema_decay=0.9,
    ):
        super().__init__()
        if kernel != "diagonal_moments":
            raise NotImplementedError("kernelized conditional matching is reserved for a future MMD loss")
        if beta < 0 or min_group_size < 2 or eps <= 0:
            raise ValueError("invalid conditional-moment hyperparameters")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1)")
        real_mean = torch.as_tensor(real_mean, dtype=torch.float32)
        real_variance = torch.as_tensor(real_variance, dtype=torch.float32)
        if real_mean.ndim != 2 or real_variance.shape != real_mean.shape:
            raise ValueError("real statistics must be matching [conditions, features] tensors")
        self.register_buffer("real_mean", real_mean)
        self.register_buffer("real_variance", real_variance)
        self.register_buffer("ema_mean", torch.zeros_like(real_mean), persistent=False)
        self.register_buffer("ema_variance", torch.zeros_like(real_variance), persistent=False)
        self.register_buffer(
            "ema_initialized",
            torch.zeros(real_mean.shape[0], dtype=torch.bool),
            persistent=False,
        )
        self.beta = float(beta)
        self.min_group_size = int(min_group_size)
        self.eps = float(eps)
        self.kernel = kernel
        self.ema_enabled = bool(ema_enabled)
        self.ema_decay = float(ema_decay)

    def forward(self, features, condition_ids, sample_weights):
        if features.ndim != 2:
            raise ValueError("features must have shape [samples, dimensions]")
        condition_ids = condition_ids.to(device=features.device, dtype=torch.long).reshape(-1)
        sample_weights = sample_weights.to(device=features.device, dtype=features.dtype).reshape(-1)
        if features.shape[0] != condition_ids.numel() or condition_ids.shape != sample_weights.shape:
            raise ValueError("feature, condition-ID, and sample-weight counts must match")
        if condition_ids.numel() == 0:
            raise ValueError("conditional moment loss received an empty batch")
        if condition_ids.min() < 0 or condition_ids.max() >= self.real_mean.shape[0]:
            raise ValueError("batch condition ID is absent from the real-statistics cache")
        if (sample_weights < 0).any() or not torch.isfinite(sample_weights).all():
            raise ValueError("sample weights must be finite and nonnegative")

        group_ids, inverse = torch.unique(condition_ids, sorted=True, return_inverse=True)
        num_groups = group_ids.numel()
        dimensions = features.shape[1]
        counts = torch.zeros(num_groups, dtype=features.dtype, device=features.device)
        weight_sums = torch.zeros_like(counts)
        weighted_sum = torch.zeros(num_groups, dimensions, dtype=features.dtype, device=features.device)
        weighted_square_sum = torch.zeros_like(weighted_sum)
        counts.index_add_(0, inverse, torch.ones_like(sample_weights))
        weight_sums.index_add_(0, inverse, sample_weights)
        weighted_sum.index_add_(0, inverse, features * sample_weights[:, None])
        weighted_square_sum.index_add_(0, inverse, features.square() * sample_weights[:, None])

        denominator = weight_sums.clamp_min(self.eps)[:, None]
        batch_mean = weighted_sum / denominator
        batch_variance = (weighted_square_sum / denominator - batch_mean.square()).clamp_min(0.0)
        valid = (counts >= self.min_group_size) & (weight_sums > self.eps)
        valid_float = valid.to(features.dtype)
        valid_denominator = valid_float.sum().clamp_min(1.0)

        selected_mean = batch_mean
        selected_variance = batch_variance
        if self.ema_enabled:
            prior_valid = self.ema_initialized[group_ids]
            prior_mean = self.ema_mean[group_ids].detach()
            prior_variance = self.ema_variance[group_ids].detach()
            mixed_mean = prior_mean * self.ema_decay + batch_mean * (1.0 - self.ema_decay)
            mixed_variance = prior_variance * self.ema_decay + batch_variance * (1.0 - self.ema_decay)
            selected_mean = torch.where(prior_valid[:, None], mixed_mean, batch_mean)
            selected_variance = torch.where(prior_valid[:, None], mixed_variance, batch_variance)
            with torch.no_grad():
                # A defensive min-group rejection must also prevent an
                # undersized group from contaminating an initialized history.
                valid_columns = valid[:, None]
                self.ema_mean[group_ids] = torch.where(
                    valid_columns,
                    selected_mean.detach(),
                    self.ema_mean[group_ids],
                )
                self.ema_variance[group_ids] = torch.where(
                    valid_columns,
                    selected_variance.detach(),
                    self.ema_variance[group_ids],
                )
                self.ema_initialized[group_ids] |= valid

        real_mean = self.real_mean[group_ids]
        real_variance = self.real_variance[group_ids]
        mean_per_group = (selected_mean - real_mean).square().mean(dim=1)
        variance_per_group = (selected_variance - real_variance).square().mean(dim=1)
        mean_loss = (mean_per_group * valid_float).sum() / valid_denominator
        variance_loss = (variance_per_group * valid_float).sum() / valid_denominator
        valid_columns = valid_float[:, None]
        variance_ratio = (selected_variance * valid_columns).sum() / (
            (real_variance * valid_columns).sum().clamp_min(self.eps)
        )
        total = mean_loss + self.beta * variance_loss
        return {
            "total": total,
            "mean": mean_loss,
            "variance": variance_loss,
            "num_conditions": valid_float.sum(),
            "variance_ratio": variance_ratio,
        }

    def runtime_state_dict(self):
        if not self.ema_enabled:
            return {}
        return {
            "ema_mean": self.ema_mean.detach().cpu(),
            "ema_variance": self.ema_variance.detach().cpu(),
            "ema_initialized": self.ema_initialized.detach().cpu(),
        }

    def load_runtime_state_dict(self, state):
        if not self.ema_enabled:
            if state:
                raise ValueError("received EMA moment state while EMA is disabled")
            return
        required = {"ema_mean", "ema_variance", "ema_initialized"}
        if set(state) != required:
            raise ValueError("conditional-moment EMA state has an invalid schema")
        for name in required:
            target = getattr(self, name)
            value = torch.as_tensor(state[name], device=target.device, dtype=target.dtype)
            if value.shape != target.shape:
                raise ValueError(f"conditional-moment EMA state shape mismatch for {name}")
            target.copy_(value)


class SupervisedPrototypeInfoNCELoss(nn.Module):
    """SupCon loss from Khosla et al. with fixed real-condition prototypes.

    Generated features are anchors. Real molecular feature means are the
    contrast set, and every prototype with the anchor's MoA label is positive.
    The log-probability reduction follows the authors' public ``SupConLoss``.
    """

    def __init__(
        self,
        real_prototypes,
        prototype_moa_ids,
        *,
        temperature=0.1,
        base_temperature=0.1,
        eps=1e-12,
    ):
        super().__init__()
        real_prototypes = torch.as_tensor(real_prototypes, dtype=torch.float32)
        prototype_moa_ids = torch.as_tensor(prototype_moa_ids, dtype=torch.long).reshape(-1)
        if real_prototypes.ndim != 2 or real_prototypes.shape[0] != prototype_moa_ids.numel():
            raise ValueError("InfoNCE prototypes and MoA IDs must have matching first dimensions")
        if temperature <= 0 or base_temperature <= 0 or eps <= 0:
            raise ValueError("InfoNCE temperatures and epsilon must be positive")
        if torch.unique(prototype_moa_ids).numel() < 2:
            raise ValueError("InfoNCE requires prototypes from at least two MoA classes")
        if not torch.isfinite(real_prototypes).all():
            raise ValueError("InfoNCE prototypes must be finite")
        prototype_norms = torch.linalg.vector_norm(real_prototypes, dim=1)
        if (prototype_norms <= eps).any():
            raise ValueError("InfoNCE prototypes must have nonzero norm")
        self.register_buffer("prototypes", F.normalize(real_prototypes, dim=1))
        self.register_buffer("prototype_moa_ids", prototype_moa_ids)
        self.temperature = float(temperature)
        self.base_temperature = float(base_temperature)
        self.eps = float(eps)

    def forward(self, features, condition_ids, sample_weights):
        if features.ndim != 2 or features.shape[1] != self.prototypes.shape[1]:
            raise ValueError("InfoNCE features must match the prototype feature dimension")
        condition_ids = condition_ids.to(device=features.device, dtype=torch.long).reshape(-1)
        sample_weights = sample_weights.to(device=features.device, dtype=features.dtype).reshape(-1)
        if features.shape[0] != condition_ids.numel() or condition_ids.shape != sample_weights.shape:
            raise ValueError("InfoNCE feature, condition-ID, and sample-weight counts must match")
        if condition_ids.numel() == 0:
            raise ValueError("InfoNCE received an empty batch")
        if condition_ids.min() < 0 or condition_ids.max() >= self.prototype_moa_ids.numel():
            raise ValueError("InfoNCE condition ID is absent from the prototype table")
        if (sample_weights < 0).any() or not torch.isfinite(sample_weights).all():
            raise ValueError("InfoNCE sample weights must be finite and nonnegative")

        normalized_features = F.normalize(features.to(torch.float32), dim=1)
        similarities = torch.matmul(normalized_features, self.prototypes.T)
        logits = similarities / self.temperature
        logits_max = logits.max(dim=1, keepdim=True).values
        logits = logits - logits_max.detach()

        anchor_moa_ids = self.prototype_moa_ids[condition_ids]
        positive_mask = anchor_moa_ids[:, None].eq(self.prototype_moa_ids[None, :])
        positive_count = positive_mask.sum(dim=1)
        if (positive_count == 0).any():
            raise RuntimeError("InfoNCE anchor has no positive real prototype")

        log_prob = logits - torch.log(torch.exp(logits).sum(dim=1, keepdim=True).clamp_min(self.eps))
        mean_log_prob_positive = (
            positive_mask.to(log_prob.dtype) * log_prob
        ).sum(dim=1) / positive_count.to(log_prob.dtype)
        per_anchor = -(self.temperature / self.base_temperature) * mean_log_prob_positive
        denominator = sample_weights.sum().clamp_min(self.eps)
        total = (per_anchor * sample_weights).sum() / denominator

        negative_mask = ~positive_mask
        positive_similarity = (
            similarities * positive_mask.to(similarities.dtype)
        ).sum() / positive_mask.sum().clamp_min(1)
        negative_similarity = (
            similarities * negative_mask.to(similarities.dtype)
        ).sum() / negative_mask.sum().clamp_min(1)
        nearest_moa = self.prototype_moa_ids[similarities.argmax(dim=1)]
        prototype_accuracy = nearest_moa.eq(anchor_moa_ids).to(similarities.dtype).mean()
        return {
            "total": total,
            "positive_similarity": positive_similarity,
            "negative_similarity": negative_similarity,
            "similarity_margin": positive_similarity - negative_similarity,
            "prototype_accuracy": prototype_accuracy,
            "positive_count": positive_count.to(similarities.dtype).mean(),
        }

class AuxiliaryLossManager(nn.Module):
    """Shared frozen encoder plus independently registered feature losses."""

    def __init__(self, encoder, losses):
        super().__init__()
        self.encoder = encoder.eval().requires_grad_(False)
        self.losses = nn.ModuleDict(losses)
        if not self.losses:
            raise ValueError("at least one auxiliary loss must be registered")

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, denoised, context, sample_weights, loss_weights):
        pixels = auxiliary_encoder_input(denoised, straight_through=True, add_dither=True)
        features = self.encoder(pixels)
        if not torch.isfinite(features).all():
            raise FloatingPointError("auxiliary encoder produced non-finite features")
        outputs = {}
        total = features.new_zeros(())
        for name, module in self.losses.items():
            weight = float(loss_weights.get(name, 0.0))
            if weight <= 0.0:
                continue
            if name in {"moment", "infonce"}:
                result = module(features, context["condition_id"], sample_weights)
            else:
                result = module(features, context, sample_weights)
            total = total + weight * result["total"]
            outputs.update({f"aux_{name}_{key}": value for key, value in result.items()})
        outputs["total"] = total
        return outputs

    def runtime_state_dict(self):
        state = {}
        for name, module in self.losses.items():
            if hasattr(module, "runtime_state_dict"):
                module_state = module.runtime_state_dict()
                if module_state:
                    state[name] = module_state
        return state

    def load_runtime_state_dict(self, state):
        unknown = set(state).difference(self.losses)
        if unknown:
            raise ValueError(f"auxiliary runtime state has unknown losses: {sorted(unknown)}")
        for name, module in self.losses.items():
            if hasattr(module, "load_runtime_state_dict"):
                module.load_runtime_state_dict(state.get(name, {}))


@dataclass(frozen=True)
class AuxiliaryLossSchedule:
    max_weight: float
    start_fraction: float = 0.6
    end_fraction: float = 1.0

    def __post_init__(self):
        if self.max_weight < 0:
            raise ValueError("auxiliary max weight must be nonnegative")
        if not 0.0 <= self.start_fraction < 1.0:
            raise ValueError("auxiliary ramp start must lie in [0, 1)")
        if not self.start_fraction < self.end_fraction <= 1.0:
            raise ValueError("auxiliary ramp end must lie in (start, 1]")

    def weight(self, successful_step, total_steps):
        if total_steps <= 0 or successful_step < 0:
            raise ValueError("auxiliary schedule requires nonnegative step and positive total")
        progress = min((successful_step + 1) / total_steps, 1.0)
        if progress <= self.start_fraction:
            return 0.0
        if progress >= self.end_fraction:
            return self.max_weight
        ramp = (progress - self.start_fraction) / (
            self.end_fraction - self.start_fraction
        )
        return self.max_weight * ramp
