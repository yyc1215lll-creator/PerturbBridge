import torch
import numpy as np
import math
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2.0 - 1.0
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x


class UniformDequant(object):
    def __call__(self, x):
        return x + torch.rand_like(x) / 256


class RASampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU).
    Heavily based on 'torch.utils.data.DistributedSampler'.
    This is borrowed from the DeiT Repo:
    https://github.com/facebookresearch/deit/blob/main/samplers.py
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, repetitions=3):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * float(repetitions) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle
        self.seed = seed
        self.repetitions = repetitions

    def __iter__(self):
        if self.shuffle:
            # Deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(self.repetitions)]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class InfiniteBatchSampler(Sampler):
    def __init__(self, dataset_len, batch_size, seed=0, filling=False, shuffle=True, drop_last=False):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.iters_per_ep = dataset_len // batch_size if drop_last else (dataset_len + batch_size - 1) // batch_size
        self.max_p = self.iters_per_ep * batch_size
        self.filling = filling
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed
        self.indices = self.gener_indices()

    def gener_indices(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(self.dataset_len, generator=g).numpy()
        else:
            indices = torch.arange(self.dataset_len).numpy()

        tails = self.batch_size - (self.dataset_len % self.batch_size)
        if tails != self.batch_size and self.filling:
            tails = indices[:tails]
            np.random.shuffle(indices)
            indices = np.concatenate((indices, tails))

        # built-in list/tuple is faster than np.ndarray (when collating the data via a for-loop)
        # noinspection PyTypeChecker
        return tuple(indices.tolist())

    def __iter__(self):
        p, q = 0, 0
        while p < self.max_p:
            q = p + self.batch_size
            yield self.indices[p:q]
            p = q
        self.epoch += 1
        if self.shuffle:
            self.indices = self.gener_indices()

    def __len__(self):
        return self.iters_per_ep

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.indices = self.gener_indices()


class DistributedEvalSampler(Sampler):
    """Shard evaluation indices without padding or duplicating samples."""

    def __init__(self, dataset, num_replicas=None, rank=None):
        self.dataset = dataset
        self.num_replicas = dist.get_world_size() if num_replicas is None else num_replicas
        self.rank = dist.get_rank() if rank is None else rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        return 0 if remaining <= 0 else (remaining + self.num_replicas - 1) // self.num_replicas


class DistInfiniteBatchSampler(InfiniteBatchSampler):
    def __init__(
        self, world_size, rank, dataset_len, glb_batch_size, seed=0, repeated_aug=0, filling=False, shuffle=True
    ):
        # from torchvision.models import ResNet50_Weights
        # RA sampler: https://github.com/pytorch/vision/blob/5521e9d01ee7033b9ee9d421c1ef6fb211ed3782/references/classification/sampler.py

        assert glb_batch_size % world_size == 0
        self.world_size, self.rank = world_size, rank
        self.dataset_len = dataset_len
        self.glb_batch_size = glb_batch_size
        self.batch_size = glb_batch_size // world_size

        self.iters_per_ep = (dataset_len + glb_batch_size - 1) // glb_batch_size
        self.filling = filling
        self.shuffle = shuffle
        self.repeated_aug = repeated_aug
        self.epoch = 0
        self.seed = seed
        self.indices = self.gener_indices()

    def gener_indices(self):
        global_max_p = (
            self.iters_per_ep * self.glb_batch_size
        )  # global_max_p % world_size must be 0 cuz glb_batch_size % world_size == 0
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            global_indices = torch.randperm(self.dataset_len, generator=g)
            if self.repeated_aug > 1:
                global_indices = global_indices[
                    : (self.dataset_len + self.repeated_aug - 1) // self.repeated_aug
                ].repeat_interleave(self.repeated_aug, dim=0)[:global_max_p]
        else:
            global_indices = torch.arange(self.dataset_len)
        filling = global_max_p - global_indices.shape[0]
        if filling > 0 and self.filling:
            global_indices = torch.cat((global_indices, global_indices[:filling]))
        global_indices = tuple(global_indices.numpy().tolist())

        seps = torch.linspace(0, len(global_indices), self.world_size + 1, dtype=torch.int)
        local_indices = global_indices[seps[self.rank] : seps[self.rank + 1]]
        self.max_p = len(local_indices)
        return local_indices


class ConditionalGroupedBatchSampler(Sampler):
    """Build deterministic distributed batches with ``k`` rows per condition.

    Every rank receives whole condition blocks, so local/microbatch moment
    estimates never depend on differentiable cross-rank gathers.
    """

    def __init__(
        self,
        dataset,
        glb_batch_size,
        condition_group_size,
        seed=0,
        rank=None,
        world_size=None,
    ):
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.world_size = dist.get_world_size() if world_size is None else int(world_size)
        self.rank = dist.get_rank() if rank is None else int(rank)
        self.glb_batch_size = int(glb_batch_size)
        self.condition_group_size = int(condition_group_size)
        if self.glb_batch_size % self.world_size != 0:
            raise ValueError("global batch must be divisible by world size")
        self.batch_size = self.glb_batch_size // self.world_size
        if self.condition_group_size < 2:
            raise ValueError("condition group size must be at least two")
        if self.batch_size % self.condition_group_size != 0:
            raise ValueError("rank-local batch must contain whole condition groups")
        if not hasattr(dataset, "condition_to_indices"):
            raise ValueError("dataset does not expose condition_to_indices")
        self.condition_to_indices = {
            int(key): torch.as_tensor(value, dtype=torch.long)
            for key, value in dataset.condition_to_indices.items()
        }
        self.num_condition_groups = self.glb_batch_size // self.condition_group_size
        if self.num_condition_groups > len(self.condition_to_indices):
            raise ValueError("global batch requests more distinct conditions than the dataset provides")
        too_small = {
            key: len(value)
            for key, value in self.condition_to_indices.items()
            if len(value) < self.condition_group_size
        }
        if too_small:
            raise ValueError(f"conditions smaller than k={self.condition_group_size}: {too_small}")
        self.iters_per_ep = (self.dataset_len + self.glb_batch_size - 1) // self.glb_batch_size
        self.seed = int(seed)
        self.epoch = 0
        self.indices = self._generate_indices()

    def _generate_indices(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        condition_ids = torch.tensor(sorted(self.condition_to_indices), dtype=torch.long)
        local_batches = []
        for _ in range(self.iters_per_ep):
            selected = condition_ids[
                torch.randperm(condition_ids.numel(), generator=generator)[: self.num_condition_groups]
            ]
            blocks = []
            for condition_id in selected.tolist():
                candidates = self.condition_to_indices[condition_id]
                chosen = candidates[
                    torch.randperm(candidates.numel(), generator=generator)[: self.condition_group_size]
                ]
                blocks.append(chosen)
            global_batch = torch.cat(blocks).reshape(self.world_size, self.batch_size)
            local_batches.append(global_batch[self.rank])
        return tuple(torch.cat(local_batches).tolist())

    def __iter__(self):
        for start in range(0, len(self.indices), self.batch_size):
            yield self.indices[start : start + self.batch_size]
        self.epoch += 1
        self.indices = self._generate_indices()

    def __len__(self):
        return self.iters_per_ep

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.indices = self._generate_indices()


def load_data(
    data_dir,
    dataset,
    batch_size,
    image_size,
    deterministic=False,
    include_test=False,
    seed=42,
    num_workers=2,
    cellflux_metadata_path=None,
    cellflux_embedding_path=None,
    cellflux_compounds=None,
    cellflux_excluded_compounds=None,
    cellflux_max_eval_samples=0,
    cellflux_condition_group_size=0,
    eval_batch_size=None,
):
    # Compute batch size for this worker.
    root = data_dir

    if dataset == "edges2handbags":

        from .aligned_dataset import EdgesDataset

        trainset = EdgesDataset(dataroot=root, train=True, img_size=image_size, random_crop=True, random_flip=True)

        valset = EdgesDataset(dataroot=root, train=True, img_size=image_size, random_crop=False, random_flip=False)
        if include_test:
            testset = EdgesDataset(
                dataroot=root, train=False, img_size=image_size, random_crop=False, random_flip=False
            )

    elif dataset == "diode":

        from .aligned_dataset import DIODE

        trainset = DIODE(
            dataroot=root, train=True, img_size=image_size, random_crop=True, random_flip=True, disable_cache=True
        )

        valset = DIODE(
            dataroot=root, train=True, img_size=image_size, random_crop=False, random_flip=False, disable_cache=True
        )

        if include_test:
            testset = DIODE(dataroot=root, train=False, img_size=image_size, random_crop=False, random_flip=False)

    elif "imagenet_inpaint" in dataset:
        corrupt_type = dataset.split("_")[-1]
        assert corrupt_type in ["center", "freeform2030"]
        from .imagenet_inpaint import ImageNetInpaintingDataset, InpaintingVal10kSubset

        trainset = ImageNetInpaintingDataset(root, image_size, corrupt_type, train=True)
        valset = ImageNetInpaintingDataset(root, image_size, corrupt_type, train=False)

        if include_test:
            testset = InpaintingVal10kSubset(root, image_size, corrupt_type)

    elif dataset in {"cellflux_bbbc021", "cellflux_cpg0000"}:
        if not cellflux_metadata_path or not cellflux_embedding_path:
            raise ValueError(
                f"{dataset} requires --cellflux_metadata_path and "
                "--cellflux_embedding_path"
            )
        if dataset == "cellflux_bbbc021":
            from .cellflux_bbbc021 import BBBC021BridgeDataset as BridgeDataset
        else:
            from .cellflux_cpg0000 import CPG0000BridgeDataset as BridgeDataset

        if isinstance(cellflux_compounds, str):
            cellflux_compounds = [item.strip() for item in cellflux_compounds.split(",") if item.strip()]
        if isinstance(cellflux_excluded_compounds, str):
            cellflux_excluded_compounds = [
                item.strip()
                for item in cellflux_excluded_compounds.split(",")
                if item.strip()
            ]
        max_eval_targets = (
            int(cellflux_max_eval_samples)
            if int(cellflux_max_eval_samples) > 0
            else None
        )

        dataset_kwargs = dict(
            image_root=root,
            metadata_path=cellflux_metadata_path,
            embedding_path=cellflux_embedding_path,
            image_size=image_size,
            pair_seed=seed,
            compounds=cellflux_compounds,
            excluded_compounds=cellflux_excluded_compounds,
        )
        trainset = BridgeDataset(split="train", train=True, **dataset_kwargs)
        valset = BridgeDataset(
            split="test", train=False, max_targets=max_eval_targets, **dataset_kwargs
        )
        if include_test:
            testset = BridgeDataset(
                split="test", train=False, max_targets=max_eval_targets, **dataset_kwargs
            )

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if int(cellflux_condition_group_size) > 0:
        if dataset not in {"cellflux_bbbc021", "cellflux_cpg0000"}:
            raise ValueError("condition-group sampling is only implemented for CellFlux datasets")
        train_batch_sampler = ConditionalGroupedBatchSampler(
            dataset=trainset,
            glb_batch_size=batch_size * dist.get_world_size(),
            condition_group_size=int(cellflux_condition_group_size),
            seed=seed,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
        )
    else:
        train_batch_sampler = DistInfiniteBatchSampler(
            dataset_len=len(trainset),
            glb_batch_size=batch_size * dist.get_world_size(),
            seed=seed,
            shuffle=not deterministic,
            filling=True,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
        )

    loader = DataLoader(
        dataset=trainset,
        num_workers=num_workers,
        pin_memory=True,
        batch_sampler=train_batch_sampler,
    )

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler = DistributedEvalSampler(valset, num_replicas=num_tasks, rank=global_rank)
    eval_batch_size = batch_size if eval_batch_size is None else eval_batch_size
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=eval_batch_size, sampler=sampler, num_workers=num_workers, drop_last=False
    )

    if include_test:

        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        sampler = DistributedEvalSampler(testset, num_replicas=num_tasks, rank=global_rank)
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=eval_batch_size,
            sampler=sampler,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
        )

        return loader, val_loader, test_loader
    else:
        return loader, val_loader
