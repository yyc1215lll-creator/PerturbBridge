from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _state_mask(values, treated):
    """Accept the numeric and string state encodings used by CellFlux."""
    numeric = pd.to_numeric(values, errors="coerce")
    normalized = values.astype(str).str.lower()
    names = {"1", "trt", "treated"} if treated else {"0", "ctrl", "control"}
    return numeric.eq(1 if treated else 0) | normalized.isin(names)


def resolve_allencell_image(image_root, sample_key):
    """Resolve ``<plate>__<cell>`` to Allen Cell's preprocessed NPY."""
    image_root = Path(image_root)
    sample_key = str(sample_key)
    direct = image_root / sample_key
    for candidate in (direct, direct.with_suffix(".npy")):
        if candidate.is_file():
            return candidate

    fields = sample_key.split("__")
    if len(fields) != 2 or not all(fields):
        raise ValueError(f"Unsupported Allen Cell SAMPLE_KEY: {sample_key}")
    return image_root / fields[0] / f"{fields[1]}.npy"


class AllenPerturbBridgeDataset(Dataset):
    """Allen Cell drug/structure endpoint pairs in DDBM orientation.

    Treated images are the clean endpoint ``x0``.  Controls from the same
    plate are the source endpoint ``xT``.  Allen Cell plates each image one
    tagged structure, so same-plate pairing also preserves the structure of
    the source endpoint.  The condition is the released 5-drug plus
    7-structure additive one-hot vector.
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
        self.image_size = int(image_size)
        self.train = bool(train)
        self.pair_seed = int(pair_seed)

        metadata = pd.read_csv(metadata_path, index_col=0)
        required = {
            "SAMPLE_KEY",
            "SPLIT",
            "STATE",
            "BATCH",
            "CPD_NAME",
            "STRUCTURE",
        }
        missing = sorted(required.difference(metadata.columns))
        if missing:
            raise ValueError(f"Allen Cell metadata is missing columns: {missing}")

        metadata = metadata.loc[metadata["SPLIT"].astype(str) == split].copy()
        controls = metadata.loc[_state_mask(metadata["STATE"], treated=False)]
        treated_rows = metadata.loc[_state_mask(metadata["STATE"], treated=True)]
        if compounds:
            treated_rows = treated_rows.loc[treated_rows["CPD_NAME"].isin(compounds)]
        if excluded_compounds:
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
                f"Allen Cell split={split!r} needs controls and treated rows; "
                f"found {len(self.controls)} controls and {len(self.treated)} treated"
            )

        self.controls_by_batch = {
            batch: np.asarray(indexes, dtype=np.int64)
            for batch, indexes in self.controls.groupby("BATCH", sort=False).indices.items()
        }
        missing_control_batches = sorted(
            set(self.treated["BATCH"]) - set(self.controls_by_batch)
        )
        if missing_control_batches:
            raise ValueError(
                f"No same-plate controls for Allen Cell batches: {missing_control_batches}"
            )

        structures_per_batch = metadata.groupby("BATCH")["STRUCTURE"].nunique()
        mixed_batches = structures_per_batch[structures_per_batch != 1]
        if not mixed_batches.empty:
            raise ValueError(
                "Allen Cell batches must image exactly one structure: "
                f"{mixed_batches.to_dict()}"
            )

        embeddings = pd.read_csv(embedding_path, index_col=0)
        embeddings.index = embeddings.index.astype(str)
        molecules = self.treated["CPD_NAME"].astype(str)
        missing_embeddings = sorted(set(molecules) - set(embeddings.index))
        if missing_embeddings:
            raise ValueError(f"Missing Allen Cell conditions: {missing_embeddings}")
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
        self.structure_names = sorted(self.treated["STRUCTURE"].astype(str).unique().tolist())
        self.structure_id_by_name = {
            name: structure_id for structure_id, name in enumerate(self.structure_names)
        }
        self.plate_names = sorted(self.treated["BATCH"].astype(str).unique().tolist())
        self.plate_id_by_name = {
            name: plate_id for plate_id, name in enumerate(self.plate_names)
        }

    def __len__(self):
        return len(self.treated)

    def _control_index(self, index, batch):
        candidates = self.controls_by_batch[batch]
        if self.train:
            return int(candidates[np.random.randint(len(candidates))])
        rng = np.random.default_rng(self.pair_seed + int(index))
        return int(rng.choice(candidates))

    def _load_image(self, sample_key):
        path = resolve_allencell_image(self.image_root, sample_key)
        if not path.is_file():
            raise FileNotFoundError(f"Allen Cell image does not exist: {path}")
        image = torch.from_numpy(np.load(path)).to(torch.float32)
        if image.ndim != 3:
            raise ValueError(
                f"Expected HWC Allen Cell image, got shape {tuple(image.shape)} at {path}"
            )
        image = image.permute(2, 0, 1)
        expected_shape = (3, self.image_size, self.image_size)
        if image.shape != expected_shape:
            raise ValueError(
                f"Expected Allen Cell image shape {expected_shape}, "
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
        condition = torch.from_numpy(
            self.embeddings.loc[molecule].to_numpy(dtype=np.float32, copy=True)
        )
        metadata = {
            "index": int(index),
            "condition": condition,
            "condition_id": int(self.condition_ids[index]),
            "molecule": molecule,
            "target_key": target_key,
            "control_key": control_key,
            "target_batch": str(target["BATCH"]),
            "control_batch": str(control["BATCH"]),
            "structure": str(target["STRUCTURE"]),
            "structure_id": self.structure_id_by_name[str(target["STRUCTURE"])],
            "plate_id": self.plate_id_by_name[str(target["BATCH"])],
        }
        return x0, xT, metadata
