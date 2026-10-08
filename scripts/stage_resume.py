"""Stage a complete checkpoint produced by the previous phase of this run."""
import argparse
import copy
from pathlib import Path
import shutil


def main(a):
    import torch
    a.destination.mkdir(parents=True, exist_ok=True)
    if list(a.destination.glob('model_*.pt')):
        # The worker validates/loads the existing phase rather than overwriting it.
        return
    matches = []
    for path in a.source.glob('train_state_rank000_*.pt'):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if int(state['completed_epoch']) == a.epoch:
            matches.append((int(state['step']), state))
    if not matches:
        raise FileNotFoundError(f'No complete E{a.epoch} state in {a.source}')
    step, state = max(matches, key=lambda pair: pair[0])
    model = a.source / f'model_{step:06d}.pt'
    files = [model, a.source / f'opt_{step:06d}.pt', a.source / f'ema_0.999_{step:06d}.pt']
    original_ranks = sorted(a.source.glob(f'train_state_rank*_{step:06d}.pt'))
    assert original_ranks and all(p.is_file() for p in files), 'Incomplete checkpoint'
    for path in files[1:]:
        shutil.copy2(path, a.destination / path.name)
    for rank in range(a.world):
        name = f'train_state_rank{rank:03d}_{step:06d}.pt'
        source = a.source / name
        if source.is_file():
            shutil.copy2(source, a.destination / name)
        else:
            # New ranks cannot inherit independent streams that did not exist.
            # Preserve all optimizer/clock/auxiliary state; fork only their RNG.
            fork = copy.deepcopy(state)
            fork['_fork_seed'] = a.seed + 100003 * rank + 10000019 * a.epoch
            fork['cuda_rng'] = None
            torch.save(fork, a.destination / name)
    # Publish the model last, matching the engine's checkpoint convention.
    shutil.copy2(model, a.destination / model.name)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--world', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    main(p.parse_args())
