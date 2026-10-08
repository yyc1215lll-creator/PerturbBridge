#!/usr/bin/env python3
"""Build the immutable real-feature targets for JUMP auxiliary training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_features import (
    cpg0000_image_path,
    extract_inception_features,
    plate_control_statistics,
)
from ddbm.auxiliary_losses import JUMP_RETRIEVAL_PROTOCOL as JUMP_AUXILIARY_PROTOCOL, sha256_file


TREATED_STATES = {"1", "trt", "treated"}
CONTROL_STATES = {"0", "ctrl", "control"}
REQUIRED_COLUMNS = {
    "SAMPLE_KEY",
    "BROAD_SAMPLE",
    "PLATE",
    "WELL",
    "PERT_TYPE",
    "STATE",
    "SPLIT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--inception-weights", type=Path, required=True)
    parser.add_argument("--inception-weights-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eps", type=float, default=1e-6)
    return parser.parse_args()


def normalize_train_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"JUMP metadata is missing: {sorted(missing)}")
    result = frame.copy()
    for column in REQUIRED_COLUMNS:
        result[column] = result[column].astype(str)
    result["SPLIT"] = result["SPLIT"].str.strip().str.lower()
    result["STATE"] = result["STATE"].str.strip().str.lower()
    result = result.loc[result["SPLIT"].eq("train")].reset_index(drop=True)
    unknown = set(result["STATE"]).difference(TREATED_STATES | CONTROL_STATES)
    if unknown:
        raise ValueError(f"unknown JUMP train STATE values: {sorted(unknown)}")
    if result["SAMPLE_KEY"].duplicated().any() or result.empty:
        raise ValueError("JUMP train SAMPLE_KEY values must be unique and non-empty")
    return result


def main() -> None:
    args = parse_args()
    if args.eps <= 0:
        raise ValueError("eps must be positive")
    actual_weights_sha = sha256_file(args.inception_weights)
    if actual_weights_sha != args.inception_weights_sha256:
        raise ValueError("Inception weights SHA-256 mismatch")
    metadata_sha = sha256_file(args.metadata)

    if args.output.is_file():
        payload = torch.load(args.output, map_location="cpu", weights_only=False)
        expected = {
            "protocol": JUMP_AUXILIARY_PROTOCOL,
            "metadata_sha256": metadata_sha,
            "inception_weights_sha256": actual_weights_sha,
        }
        mismatches = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise ValueError(f"existing JUMP auxiliary cache identity mismatch: {mismatches}")
        print("JUMP_AUXILIARY_CACHE_REUSED", args.output)
        return

    metadata = normalize_train_metadata(args.metadata)
    controls = metadata.loc[metadata["STATE"].isin(CONTROL_STATES)].reset_index(drop=True)
    treated = metadata.loc[metadata["STATE"].isin(TREATED_STATES)].reset_index(drop=True)
    if controls.empty or treated.empty:
        raise ValueError("JUMP train split must contain controls and treatments")
    condition_table = (
        treated[["BROAD_SAMPLE", "PERT_TYPE"]]
        .drop_duplicates()
        .sort_values("BROAD_SAMPLE")
        .reset_index(drop=True)
    )
    if condition_table["BROAD_SAMPLE"].duplicated().any():
        raise ValueError("a JUMP condition occurs in multiple perturbation types")
    if len(condition_table) != 100:
        raise ValueError(f"expected 100 JUMP train conditions, got {len(condition_table)}")

    control_paths = [cpg0000_image_path(args.image_root, row) for _, row in controls.iterrows()]
    treated_paths = [cpg0000_image_path(args.image_root, row) for _, row in treated.iterrows()]
    control_features = extract_inception_features(
        control_paths,
        kind="real",
        encoder_weights_path=args.inception_weights,
        encoder_weights_sha256=actual_weights_sha,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
    )
    treated_features = extract_inception_features(
        treated_paths,
        kind="real",
        encoder_weights_path=args.inception_weights,
        encoder_weights_sha256=actual_weights_sha,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
    )

    control_stats = plate_control_statistics(controls, control_features, eps=args.eps)
    plate_names = sorted(control_stats)
    treated_plates = sorted(set(treated["PLATE"]))
    if treated_plates != plate_names:
        raise ValueError(
            f"train treated/control plate mismatch: {treated_plates} != {plate_names}"
        )
    plate_center = np.stack([control_stats[name][0] for name in plate_names]).astype(np.float32)
    plate_scale = np.stack([control_stats[name][1] for name in plate_names]).astype(np.float32)
    plate_to_id = {name: index for index, name in enumerate(plate_names)}
    plate_ids = treated["PLATE"].map(plate_to_id).to_numpy(dtype=np.int64)
    robust = (treated_features - plate_center[plate_ids]) / plate_scale[plate_ids]
    robust /= np.maximum(np.linalg.norm(robust, axis=1, keepdims=True), args.eps)
    if not np.isfinite(robust).all():
        raise ValueError("JUMP train robust features are non-finite")

    condition_names = condition_table["BROAD_SAMPLE"].astype(str).tolist()
    type_names = sorted(condition_table["PERT_TYPE"].astype(str).unique())
    type_to_id = {name: index for index, name in enumerate(type_names)}
    condition_type_ids = condition_table["PERT_TYPE"].map(type_to_id).to_numpy(np.int64)
    condition_mean = []
    condition_variance = []
    condition_counts = []
    treated_names = treated["BROAD_SAMPLE"].to_numpy(str)
    for name in condition_names:
        values = robust[treated_names == name]
        if len(values) < 4:
            raise ValueError(f"JUMP condition {name!r} has fewer than four train cells")
        condition_counts.append(len(values))
        condition_mean.append(values.mean(axis=0))
        condition_variance.append(values.var(axis=0))

    payload = {
        "protocol": JUMP_AUXILIARY_PROTOCOL,
        "metadata_sha256": metadata_sha,
        "inception_weights_sha256": actual_weights_sha,
        "preprocessing": "first3_floor_uint8_inception_then_plate_control_well_median_mad_l2_v1",
        "condition_names": condition_names,
        "condition_type_names": type_names,
        "condition_type_ids": torch.from_numpy(condition_type_ids),
        "condition_counts": torch.tensor(condition_counts, dtype=torch.long),
        "condition_mean": torch.from_numpy(np.asarray(condition_mean, dtype=np.float32)),
        "condition_variance": torch.from_numpy(
            np.asarray(condition_variance, dtype=np.float32)
        ),
        "plate_names": plate_names,
        "plate_center": torch.from_numpy(plate_center),
        "plate_scale": torch.from_numpy(plate_scale),
        "control_well_counts": [int(control_stats[name][2]) for name in plate_names],
        "train_control_cells": len(controls),
        "train_treated_cells": len(treated),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    print(
        "JUMP_AUXILIARY_CACHE_SUCCESS",
        {
            "path": str(args.output),
            "conditions": len(condition_names),
            "types": dict(zip(type_names, np.bincount(condition_type_ids).tolist())),
            "plates": len(plate_names),
            "control_cells": len(controls),
            "treated_cells": len(treated),
        },
    )


if __name__ == "__main__":
    main()
