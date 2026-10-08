"""Precompute real BBBC021 treated-image moments in the auxiliary encoder."""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from datasets import DistributedEvalSampler
from datasets.cellflux_bbbc021 import BBBC021BridgeDataset
from ddbm import dist_util
from ddbm.auxiliary_losses import (
    AUXILIARY_PREPROCESSING,
    FID_INCEPTION_ENCODER,
    FIDInceptionV3Encoder,
    auxiliary_encoder_input,
    load_conditional_statistics,
    sha256_file,
)


class TreatedImageView(Dataset):
    """Avoid loading an unused control endpoint while caching real moments."""

    def __init__(self, bridge_dataset):
        self.bridge_dataset = bridge_dataset
        self.condition_names = bridge_dataset.condition_names
        self.condition_to_indices = bridge_dataset.condition_to_indices

    def __len__(self):
        return len(self.bridge_dataset)

    def __getitem__(self, index):
        target = self.bridge_dataset.treated.iloc[index]
        image = self.bridge_dataset._load_image(str(target["SAMPLE_KEY"]))
        return image, int(self.bridge_dataset.condition_ids[index]), int(index)


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--embedding_path", required=True)
    parser.add_argument("--excluded_compounds", default="")
    parser.add_argument("--encoder_weights_path", required=True)
    parser.add_argument("--encoder_weights_sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dither_seed", type=int, default=20260810)
    return parser


def _validate_existing(args, dataset):
    payload = load_conditional_statistics(
        args.output,
        expected_encoder=FID_INCEPTION_ENCODER,
        expected_weights_sha256=args.encoder_weights_sha256,
    )
    expected_counts = torch.tensor(
        [len(dataset.condition_to_indices[index]) for index in range(len(dataset.condition_names))],
        dtype=torch.long,
    )
    checks = {
        "condition_names": list(payload["condition_names"]) == dataset.condition_names,
        "counts": torch.equal(payload["counts"], expected_counts),
        "dataset_size": int(payload.get("dataset_size", -1)) == len(dataset),
        "metadata_sha256": payload.get("metadata_sha256") == sha256_file(args.metadata_path),
        "dither_seed": int(payload.get("dither_seed", -1)) == args.dither_seed,
        "world_size": int(payload.get("world_size", -1)) == dist.get_world_size(),
    }
    failed = sorted(key for key, value in checks.items() if not value)
    if failed:
        raise ValueError(f"existing conditional-statistics cache failed checks: {failed}")


@torch.no_grad()
def main(args):
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    dist_util.setup_dist()
    device = dist_util.dev()
    excluded = [item.strip() for item in args.excluded_compounds.split(",") if item.strip()]
    bridge_dataset = BBBC021BridgeDataset(
        image_root=args.data_dir,
        metadata_path=args.metadata_path,
        embedding_path=args.embedding_path,
        split="train",
        image_size=96,
        train=False,
        pair_seed=42,
        excluded_compounds=excluded,
    )
    dataset = TreatedImageView(bridge_dataset)
    output = Path(args.output)
    if output.is_file():
        _validate_existing(args, dataset)
        if dist.get_rank() == 0:
            print(f"COND_STATS_CACHE_VALID path={output} rows={len(dataset)}")
        return

    sampler = DistributedEvalSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    encoder = FIDInceptionV3Encoder(
        args.encoder_weights_path,
        expected_sha256=args.encoder_weights_sha256,
    ).to(device)
    num_conditions = len(dataset.condition_names)
    dimensions = encoder.num_features
    counts = torch.zeros(num_conditions, dtype=torch.float64, device=device)
    sums = torch.zeros(num_conditions, dimensions, dtype=torch.float64, device=device)
    square_sums = torch.zeros_like(sums)
    generator = torch.Generator(device=device).manual_seed(args.dither_seed + dist.get_rank())

    for images_cpu, condition_ids_cpu, _ in loader:
        images = images_cpu.to(device, non_blocking=True)
        condition_ids = condition_ids_cpu.to(device, non_blocking=True, dtype=torch.long)
        pixels = auxiliary_encoder_input(
            images,
            straight_through=False,
            add_dither=True,
            generator=generator,
        )
        features = encoder(pixels).to(torch.float64)
        if not torch.isfinite(features).all():
            raise FloatingPointError("non-finite real auxiliary features")
        counts.index_add_(0, condition_ids, torch.ones_like(condition_ids, dtype=torch.float64))
        sums.index_add_(0, condition_ids, features)
        square_sums.index_add_(0, condition_ids, features.square())

    for tensor in (counts, sums, square_sums):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if (counts <= 0).any() or int(counts.sum().item()) != len(dataset):
        raise RuntimeError("distributed conditional-statistics count audit failed")
    means = sums / counts[:, None]
    variances = (square_sums / counts[:, None] - means.square()).clamp_min(0.0)
    if not torch.isfinite(means).all() or not torch.isfinite(variances).all():
        raise FloatingPointError("non-finite conditional statistics")

    if dist.get_rank() == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "condition_ids": torch.arange(num_conditions, dtype=torch.long),
            "condition_names": list(dataset.condition_names),
            "counts": counts.to(torch.long).cpu(),
            "mean": means.to(torch.float32).cpu(),
            "variance": variances.to(torch.float32).cpu(),
            "encoder": FID_INCEPTION_ENCODER,
            "encoder_weights_sha256": args.encoder_weights_sha256,
            "preprocessing": AUXILIARY_PREPROCESSING,
            "variance_estimator": "population_biased_e_x2_minus_mean2",
            "dataset_size": len(dataset),
            "metadata_sha256": sha256_file(args.metadata_path),
            "embedding_sha256": sha256_file(args.embedding_path),
            "excluded_compounds": excluded,
            "dither_seed": args.dither_seed,
            "world_size": dist.get_world_size(),
        }
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, output)
        print(
            f"COND_STATS_SUCCESS path={output} rows={len(dataset)} conditions={num_conditions} "
            f"features={dimensions}"
        )
    dist.barrier()
    _validate_existing(args, dataset)


if __name__ == "__main__":
    main(create_argparser().parse_args())
