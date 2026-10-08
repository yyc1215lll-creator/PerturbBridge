from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def resolve_cpg0000_image(image_root, sample_key):
    """Resolve a CellFlux JUMP/CPG0000 ``SAMPLE_KEY`` to its 5-channel NPY."""
    image_root = Path(image_root)
    sample_key = str(sample_key)

    direct = image_root / sample_key
    for candidate in (direct, direct.with_suffix(".npy")):
        if candidate.is_file():
            return candidate

    fields = sample_key.split("_")
    if len(fields) < 4:
        raise ValueError(f"Unsupported CPG0000 SAMPLE_KEY: {sample_key}")
    return image_root / fields[0] / f"{fields[1]}_{fields[2]}" / ("_".join(fields[1:]) + ".npy")


class CPG0000BridgeDataset(Dataset):
    """JUMP/CPG0000 CellFlux endpoint pairs in DDBM orientation.

    A treated cell is the clean endpoint ``x0``. Its source endpoint ``xT``
    is a control cell from the same plate, matching the empirical coupling in
    the released CellFlux JUMP loader. Evaluation targets keep metadata order,
    and their control pairing is deterministic across epochs and ranks.
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
        required = {"SAMPLE_KEY", "SPLIT", "STATE", "PLATE", "BROAD_SAMPLE"}
        missing = sorted(required.difference(metadata.columns))
        if missing:
            raise ValueError(f"CPG0000 metadata is missing columns: {missing}")

        metadata = metadata.loc[metadata["SPLIT"].astype(str) == str(split)].copy()
        states = metadata["STATE"].astype(str).str.lower()
        controls = metadata.loc[states.isin({"control", "ctrl", "0"})]
        treated_rows = metadata.loc[states.isin({"trt", "treated", "1"})]
        if compounds:
            treated_rows = treated_rows.loc[treated_rows["BROAD_SAMPLE"].isin(compounds)]
        if excluded_compounds:
            treated_rows = treated_rows.loc[
                ~treated_rows["BROAD_SAMPLE"].isin(excluded_compounds)
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
                f"CPG0000 split={split!r} needs controls and treated rows; "
                f"found {len(self.controls)} controls and {len(self.treated)} treated"
            )

        self.controls_by_plate = {
            plate: np.asarray(indices, dtype=np.int64)
            for plate, indices in self.controls.groupby("PLATE", sort=False).indices.items()
        }
        missing_control_plates = sorted(set(self.treated["PLATE"]) - set(self.controls_by_plate))
        if missing_control_plates:
            raise ValueError(f"No same-plate controls for CPG0000 plates: {missing_control_plates}")

        embeddings = pd.read_csv(embedding_path, index_col=0)
        embeddings.index = embeddings.index.astype(str)
        molecules = self.treated["BROAD_SAMPLE"].astype(str)
        missing_embeddings = sorted(set(molecules) - set(embeddings.index))
        if missing_embeddings:
            raise ValueError(f"Missing CPG0000 condition embeddings for: {missing_embeddings}")
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
        plates = self.treated["PLATE"].astype(str)
        self.plate_names = sorted(plates.unique().tolist())
        self.plate_id_by_name = {
            name: plate_id for plate_id, name in enumerate(self.plate_names)
        }
        self.plate_ids = plates.map(self.plate_id_by_name).to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.treated)

    def _control_index(self, index, plate):
        candidates = self.controls_by_plate[plate]
        if self.train:
            return int(candidates[np.random.randint(len(candidates))])
        rng = np.random.default_rng(self.pair_seed + int(index))
        return int(rng.choice(candidates))

    def _load_image(self, sample_key):
        path = resolve_cpg0000_image(self.image_root, sample_key)
        if not path.is_file():
            raise FileNotFoundError(f"CPG0000 image does not exist: {path}")
        image = torch.from_numpy(np.load(path)).to(torch.float32)
        if image.ndim != 3:
            raise ValueError(f"Expected HWC CPG0000 image, got shape {tuple(image.shape)} at {path}")
        image = image.permute(2, 0, 1)
        expected_shape = (5, self.image_size, self.image_size)
        if image.shape != expected_shape:
            raise ValueError(
                f"Expected CPG0000 image shape {expected_shape}, got {tuple(image.shape)} at {path}"
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
        control_index = self._control_index(index, target["PLATE"])
        control = self.controls.iloc[control_index]

        target_key = str(target["SAMPLE_KEY"])
        control_key = str(control["SAMPLE_KEY"])
        molecule = str(target["BROAD_SAMPLE"])
        x0 = self._load_image(target_key)
        xT = self._load_image(control_key)
        condition = torch.from_numpy(
            self.embeddings.loc[molecule].to_numpy(dtype=np.float32, copy=True)
        )
        metadata = {
            "index": int(index),
            "condition": condition,
            "condition_id": int(self.condition_ids[index]),
            "plate_id": int(self.plate_ids[index]),
            "molecule": molecule,
            "target_key": target_key,
            "control_key": control_key,
            "target_batch": str(target["PLATE"]),
            "control_batch": str(control["PLATE"]),
        }
        return x0, xT, metadata
