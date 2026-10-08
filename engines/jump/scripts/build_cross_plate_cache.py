"""Frozen Inception targets from training cells ONLY; torchrun-sharded extraction."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets.cellflux_cpg0000 import resolve_cpg0000_image
from ddbm.auxiliary_losses import (FIDInceptionV3Encoder, auxiliary_encoder_input,
    sha256_file, load_jump_retrieval_statistics, JumpCrossPlateSupConLoss)


class TrainingImages(Dataset):
    def __init__(self, rows, root, indices):
        self.keys = rows.SAMPLE_KEY.astype(str).tolist()
        self.root, self.indices = root, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = self.indices[i]
        x = torch.from_numpy(np.load(resolve_cpg0000_image(self.root, self.keys[index]))).float()
        assert x.shape == (96, 96, 5) and torch.isfinite(x).all()
        x = x.permute(2, 0, 1)[:3].div(255).mul(2).sub(1)
        # Per-index RNG makes targets independent of world size and batching.
        g = torch.Generator().manual_seed(42 + index)
        return index, auxiliary_encoder_input(x[None], straight_through=False, generator=g)[0]


def metadata_audit(base, metadata):
    frame = pd.read_csv(metadata, index_col=0)
    rows = frame.loc[frame.SPLIT.eq('train') & frame.STATE.eq('trt')].reset_index(drop=True)
    test = frame.loc[frame.SPLIT.eq('test')]
    assert len(rows) == 35960 and rows.SAMPLE_KEY.is_unique
    assert not set(rows.SAMPLE_KEY) & set(test.SAMPLE_KEY)
    assert sorted(rows.BROAD_SAMPLE.astype(str).unique()) == base['condition_names']
    assert sorted(rows.PLATE.astype(str).unique()) == base['plate_names']
    assert sha256_file(metadata) == base['metadata_sha256']
    c = rows.BROAD_SAMPLE.astype(str).map({x: i for i, x in enumerate(base['condition_names'])}).to_numpy()
    p = rows.PLATE.astype(str).map({x: i for i, x in enumerate(base['plate_names'])}).to_numpy()
    counts = np.bincount(c, minlength=len(base['condition_names']))
    assert np.array_equal(counts, np.asarray(base['condition_counts']))
    pairs = np.unique(np.stack((c, p), 1), axis=0)
    nplates = np.bincount(pairs[:, 0], minlength=len(counts))
    audit = dict(train_cells=len(rows), train_profiles=len(pairs), conditions=len(counts),
                 conditions_with_cross_plate_positive=int((nplates > 1).sum()),
                 eligible_cells=int((nplates[c] > 1).sum()), plates_per_condition=nplates.tolist(),
                 training_only=True, test_features_used=False)
    assert audit['conditions_with_cross_plate_positive'] > 0
    return rows, c, p, audit


def aggregate(base, features, conditions, plates):
    x = torch.as_tensor(features, dtype=torch.float64)
    c, p = torch.as_tensor(conditions), torch.as_tensor(plates)
    center, scale = base['plate_center'].double(), base['plate_scale'].double()
    normalized = F.normalize((x - center[p]) / scale[p], dim=1)
    pairs = torch.unique(torch.stack((c, p), 1), dim=0, sorted=True)
    profiles, counts = [], []
    for ci, pi in pairs:
        mask = (c == ci) & (p == pi)
        profiles.append(F.normalize((x[mask].mean(0) - center[pi]) / scale[pi], dim=0))
        counts.append(int(mask.sum()))
    means, variances = [], []
    for ci in range(len(base['condition_names'])):
        z = normalized[c == ci]
        means.append(z.mean(0)); variances.append(z.var(0, unbiased=False))
    result = dict(base)
    result.update(condition_mean=torch.stack(means).float(),
                  condition_variance=torch.stack(variances).float(),
                  cross_plate_protocol='train_condition_plate_raw_mean_robust_l2_v1',
                  cross_plate_prototypes=torch.stack(profiles).float(),
                  cross_plate_condition_ids=pairs[:, 0], cross_plate_plate_ids=pairs[:, 1],
                  cross_plate_counts=torch.tensor(counts),
                  preprocessing='first3rgb; auxiliary_encoder_input centered dither; per-row seed42+index',
                  prototype_order='mean raw features within condition/plate, train control median/MAD, L2',
                  moment_order='per-cell train-control median/MAD, L2, conditional mean/population variance')
    JumpCrossPlateSupConLoss(result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-cache', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    base = load_jump_retrieval_statistics(args.base_cache)
    assert sha256_file(args.weights) == base['inception_weights_sha256']
    rows, c, plates, audit = metadata_audit(base, args.metadata)
    if args.prepare_only:
        print(json.dumps(audit, indent=2)); return
    rank, world, local = [int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')]
    torch.set_num_threads(1); torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl')
    assert not args.output.exists(), 'Refuse to overwrite immutable targets'
    shard_dir = args.output.with_suffix('.shards'); shard_dir.mkdir(parents=True, exist_ok=True)
    model = FIDInceptionV3Encoder(args.weights, expected_sha256=base['inception_weights_sha256']).cuda().eval()
    loader = DataLoader(TrainingImages(rows, args.images, range(rank, len(rows), world)),
                        batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    ids, values = [], []
    with torch.inference_mode():
        for index, pixels in loader:
            x = model(pixels.cuda(non_blocking=True)).float()
            assert x.shape == (len(index), 2048) and torch.isfinite(x).all()
            ids.append(index.numpy()); values.append(x.cpu().numpy())
    np.savez(shard_dir / f'rank{rank}.npz', indices=np.concatenate(ids), features=np.concatenate(values))
    dist.barrier(device_ids=[local])
    if rank == 0:
        shards = [np.load(shard_dir / f'rank{r}.npz') for r in range(world)]
        indices = np.concatenate([s['indices'] for s in shards])
        x = np.concatenate([s['features'] for s in shards])[np.argsort(indices)]
        assert np.array_equal(np.sort(indices), np.arange(len(rows)))
        result = aggregate(base, x, c, plates)
        result.update(source_cache_sha256=sha256_file(args.base_cache), train_audit=audit,
                      builder_sha256=sha256_file(__file__), training_keys=rows.SAMPLE_KEY.tolist())
        torch.save(result, args.output)
        audit.update(output=str(args.output), sha256=sha256_file(args.output), world=world)
        args.output.with_suffix('.json').write_text(json.dumps(audit, indent=2) + '\n')
        print('CACHE_SUCCESS', json.dumps(audit), flush=True)
    dist.barrier(device_ids=[local]); dist.destroy_process_group()


if __name__ == '__main__':
    main()
