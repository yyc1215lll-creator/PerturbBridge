from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _state_mask(values, treated):
    """Accept the numeric and string BBBC021 state encodings used by CellFlux."""
    numeric = pd.to_numeric(values, errors="coerce")
    normalized = values.astype(str).str.lower()
    names = {"1", "trt", "treated"} if treated else {"0", "ctrl", "control"}
    return numeric.eq(1 if treated else 0) | normalized.isin(names)


def resolve_bbbc021_image(image_root, sample_key):
    """Resolve a CellFlux SAMPLE_KEY to its preprocessed ``.npy`` image."""
    image_root = Path(image_root)
    sample_key = str(sample_key)

    direct = image_root / sample_key
    for candidate in (direct, direct.with_suffix(".npy")):
        if candidate.is_file():
            return candidate

    # This mirrors CellFlux's two supported preprocessed SAMPLE_KEY layouts.
    key_parts = sample_key.split("-")
    if len(key_parts) > 1:
        fields = key_parts[1].split("_")
        if len(fields) < 4:
            raise ValueError(f"Unsupported dashed BBBC021 SAMPLE_KEY: {sample_key}")
        return image_root / "_".join(fields[:2]) / fields[2] / ("_".join(fields[3:]) + ".npy")

    fields = key_parts[0].split("_")
    if len(fields) < 3:
        raise ValueError(f"Unsupported BBBC021 SAMPLE_KEY: {sample_key}")
    return image_root / fields[0] / fields[1] / ("_".join(fields[2:]) + ".npy")


class BBBC021BridgeDataset(Dataset):
    """CellFlux BBBC021 endpoint pairs in DDBM orientation.

    Each treated image is the clean endpoint ``x0``.  Its source endpoint
    ``xT`` is sampled from controls in the same experimental batch.  Exact
    before/after cells do not exist in CellFlux, so this same-batch empirical
    coupling is the bridge joint distribution used for training.
    """

    def __init__(
        self,
        image_root,
        metadata_path,
        embedding_path,
        split,
        image_size=96,
        train=False,
        pair_seed=42,
        compounds=None,
        excluded_compounds=None,
        max_targets=None,
    ):
        super().__init__()
        self.image_root = Path(image_root)
        self.image_size = image_size
        self.train = train
        self.pair_seed = pair_seed

        metadata = pd.read_csv(metadata_path, index_col=0)
        required = {"SAMPLE_KEY", "SPLIT", "STATE", "BATCH", "CPD_NAME"}
        missing = sorted(required.difference(metadata.columns))
        if missing:
            raise ValueError(f"BBBC021 metadata is missing columns: {missing}")

        metadata = metadata.loc[metadata["SPLIT"].astype(str) == split].copy()
        controls = metadata.loc[_state_mask(metadata["STATE"], treated=False)]
        treated_rows = metadata.loc[_state_mask(metadata["STATE"], treated=True)]
        if compounds:
            # A compound subset applies to targets only. Controls generally
            # carry a vehicle name (for example DMSO) and must remain available
            # for same-batch endpoint pairing.
            treated_rows = treated_rows.loc[treated_rows["CPD_NAME"].isin(compounds)]
        if excluded_compounds:
            # Match CellFlux's OOD holdout: exclusions apply to treated
            # targets only. Vehicle controls must remain available for the
            # same-batch bridge endpoint coupling.
            treated_rows = treated_rows.loc[
                ~treated_rows["CPD_NAME"].isin(excluded_compounds)
            ]
        if max_targets is not None:
            max_targets = int(max_targets)
            if max_targets <= 0:
                raise ValueError(f"max_targets must be positive, got {max_targets}")
            treated_rows = treated_rows.iloc[:max_targets]

        self.controls = controls.reset_index(drop=True)
        self.treated = treated_rows.reset_index(drop=True)
        if self.controls.empty or self.treated.empty:
            raise ValueError(
                f"BBBC021 split={split!r} needs controls and treated rows; "
                f"found {len(self.controls)} controls and {len(self.treated)} treated"
            )

        self.controls_by_batch = {
            batch: np.asarray(indexes, dtype=np.int64)
            for batch, indexes in self.controls.groupby("BATCH", sort=False).indices.items()
        }
        missing_control_batches = sorted(set(self.treated["BATCH"]) - set(self.controls_by_batch))
        if missing_control_batches:
            raise ValueError(f"No same-batch controls for BBBC021 batches: {missing_control_batches}")

        embeddings = pd.read_csv(embedding_path, index_col=0)
        embeddings.index = embeddings.index.astype(str)
        molecules = self.treated["CPD_NAME"].astype(str)
        missing_embeddings = sorted(set(molecules) - set(embeddings.index))
        if missing_embeddings:
            raise ValueError(f"Missing molecular fingerprints for: {missing_embeddings}")
        self.embeddings = embeddings.astype(np.float32)
        self.condition_dim = self.embeddings.shape[1]
        self.condition_names = sorted(molecules.unique().tolist())
        self.condition_id_by_name = {
            name: condition_id for condition_id, name in enumerate(self.condition_names)
        }
        self.condition_ids = molecules.map(self.condition_id_by_name).to_numpy(dtype=np.int64)
        self.condition_to_indices = {
            condition_id: np.flatnonzero(self.condition_ids == condition_id).astype(np.int64)
            for condition_id in range(len(self.condition_names))
        }

    def __len__(self):
        return len(self.treated)

    def _control_index(self, index, batch):
        candidates = self.controls_by_batch[batch]
        if self.train:
            return int(candidates[np.random.randint(len(candidates))])
        # Evaluation pairing is stable across workers, ranks, and reruns.
        rng = np.random.default_rng(self.pair_seed + int(index))
        return int(rng.choice(candidates))

    def _load_image(self, sample_key):
        path = resolve_bbbc021_image(self.image_root, sample_key)
        if not path.is_file():
            raise FileNotFoundError(f"BBBC021 image does not exist: {path}")
        image = torch.from_numpy(np.load(path)).to(torch.float32)
        if image.ndim != 3:
            raise ValueError(f"Expected HWC BBBC021 image, got shape {tuple(image.shape)} at {path}")
        image = image.permute(2, 0, 1)
        if image.shape != (3, self.image_size, self.image_size):
            raise ValueError(
                f"Expected BBBC021 image shape {(3, self.image_size, self.image_size)}, "
                f"got {tuple(image.shape)} at {path}"
            )
        if self.train:
            image = image + torch.rand_like(image)
        image = image / 255.0
        if self.train and torch.rand(()) < 0.3:
            image = image.flip(-1)
        if self.train and torch.rand(()) < 0.3:
            image = image.flip(-2)
        return image.mul(2).sub(1)

    def __getitem__(self, index):
        target = self.treated.iloc[index]
        control_index = self._control_index(index, target["BATCH"])
        control = self.controls.iloc[control_index]

        target_key = str(target["SAMPLE_KEY"])
        control_key = str(control["SAMPLE_KEY"])
        molecule = str(target["CPD_NAME"])
        x0 = self._load_image(target_key)
        xT = self._load_image(control_key)
        condition = torch.from_numpy(self.embeddings.loc[molecule].to_numpy(dtype=np.float32, copy=True))

        metadata = {
            "index": int(index),
            "condition": condition,
            "condition_id": int(self.condition_ids[index]),
            "molecule": molecule,
            "target_key": target_key,
            "control_key": control_key,
            "target_batch": str(target["BATCH"]),
            "control_batch": str(control["BATCH"]),
        }
        return x0, xT, metadata
