"""TRAIN image features and plate control normalization."""
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

def cpg0000_image_path(image_root: Path | str, row: pd.Series) -> Path:
    key = str(row["SAMPLE_KEY"])
    parts = key.split("_")
    if len(parts) < 4:
        raise ValueError(f"invalid CPG0000 SAMPLE_KEY: {key!r}")
    plate = parts[0]
    well = f"{parts[1]}_{parts[2]}"
    if plate != str(row["PLATE"]) or parts[1] != str(row["WELL"]):
        raise ValueError(
            f"SAMPLE_KEY metadata mismatch for {key}: "
            f"parsed plate/well={plate}/{parts[1]}, metadata={row['PLATE']}/{row['WELL']}"
        )
    filename = "_".join(parts[1:]) + ".npy"
    return Path(image_root) / plate / well / filename

def _load_rgb(path: Path, kind: str) -> np.ndarray:
    if kind == "generated":
        from PIL import Image

        with Image.open(path) as image:
            pixels = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    elif kind == "real":
        pixels = np.load(path, allow_pickle=False)
        if pixels.ndim != 3 or pixels.shape[-1] < 3:
            raise ValueError(f"real JUMP image must be HWC with >=3 channels: {path}")
        pixels = np.floor(np.clip(pixels[..., :3], 0, 255)).astype(np.uint8)
    else:
        raise ValueError(f"unsupported image kind: {kind}")
    if pixels.shape != (96, 96, 3):
        raise ValueError(f"expected a 96x96 RGB image at {path}, got {pixels.shape}")
    return np.ascontiguousarray(pixels).copy()

def extract_inception_features(
    paths: Sequence[Path],
    *,
    kind: str,
    encoder_weights_path: Path | str,
    encoder_weights_sha256: str,
    batch_size: int,
    num_workers: int,
    device_name: str,
) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, Dataset

    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")

    class ImageDataset(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, index):
            pixels = _load_rgb(paths[index], kind)
            return torch.from_numpy(pixels).permute(2, 0, 1)

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA feature extraction requested but CUDA is unavailable")

    from ddbm.auxiliary_losses import FIDInceptionV3Encoder

    encoder = FIDInceptionV3Encoder(
        encoder_weights_path,
        expected_sha256=encoder_weights_sha256,
    ).to(device)
    loader = DataLoader(
        ImageDataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    chunks = []
    with torch.inference_mode():
        for pixels in loader:
            encoded = encoder(pixels.to(device=device, dtype=torch.float32, non_blocking=True))
            chunks.append(encoded.cpu().numpy().astype(np.float32, copy=False))
    features = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 2048), np.float32)
    if features.shape != (len(paths), 2048) or not np.isfinite(features).all():
        raise ValueError(f"invalid Inception feature matrix: {features.shape}")
    return features

def aggregate_profiles(
    frame: pd.DataFrame,
    features: np.ndarray,
    group_columns: Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    frame = frame.reset_index(drop=True)
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != len(frame):
        raise ValueError("feature rows do not match metadata rows")
    records = []
    profile_features = []
    grouped = frame.groupby(list(group_columns), sort=True, dropna=False).indices
    for key, indices in grouped.items():
        if not isinstance(key, tuple):
            key = (key,)
        indices = np.asarray(indices, dtype=np.int64)
        record = dict(zip(group_columns, key))
        record["cell_count"] = int(len(indices))
        records.append(record)
        profile_features.append(features[indices].mean(axis=0))
    if not records:
        raise ValueError(f"no profiles for grouping {group_columns}")
    return pd.DataFrame(records), np.stack(profile_features)

def plate_control_statistics(
    control_frame: pd.DataFrame,
    control_features: np.ndarray,
    *,
    eps: float,
) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    wells, well_features = aggregate_profiles(
        control_frame,
        control_features,
        ("PLATE", "WELL"),
    )
    statistics = {}
    for plate, indices in wells.groupby("PLATE", sort=True).indices.items():
        values = well_features[np.asarray(indices, dtype=np.int64)]
        if len(values) < 2:
            raise ValueError(f"plate {plate} needs at least two control wells")
        center = np.median(values, axis=0)
        mad = np.median(np.abs(values - center), axis=0)
        scale = np.maximum(1.4826 * mad, eps)
        statistics[str(plate)] = (center, scale, int(len(values)))
    return statistics
