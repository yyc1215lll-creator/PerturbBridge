"""Stage/seed adapter. Scientific model, loss, optimizer and sampler stay in engines/."""
import argparse
import importlib.util
import json
import random
from pathlib import Path
import sys
import itertools


class ShortLoader:
    """Bound a smoke epoch without altering the original dataset or transforms."""
    def __init__(self, loader, batches):
        self.loader, self.batches = loader, batches
        self.dataset, self.batch_sampler = loader.dataset, loader.batch_sampler

    def __len__(self):
        return self.batches

    def __iter__(self):
        epoch = self.batch_sampler.epoch
        yield from itertools.islice(self.loader, self.batches)
        self.batch_sampler.set_epoch(epoch + 1)


def main(a):
    import numpy as np
    import torch
    import torch.distributed as dist
    root = Path(__file__).resolve().parents[1]
    engine = root / 'engines' / a.dataset
    sys.path.insert(0, str(engine))
    spec = importlib.util.spec_from_file_location('recipe_engine', engine / 'train.py')
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    from ddbm import dist_util
    from ddbm.train_util import TrainLoop
    dist_util.setup_dist()
    assert dist.get_world_size() == a.expected_world
    torch.set_num_threads(2)

    def seed_all(seed):
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    seed_all(a.seed)
    # Identical initialization on each rank before EMA construction.
    if hasattr(train, 'seed_training_process'):
        train.seed_training_process = lambda seed, rank: seed_all(a.seed)
    cfg = json.loads(a.config.read_text())
    args = train.create_argparser().parse_args([f'--{k}={v}' for k, v in cfg.items()])
    original_load = train.load_data

    def load_data(**kw):
        loaders = original_load(**kw)
        sampler = loaders[0].batch_sampler
        sampler.seed = a.seed
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(0)
        if a.smoke_batches:
            loaders = (ShortLoader(loaders[0], a.smoke_batches), *loaders[1:])
        return loaders

    train.load_data = load_data
    original_init = TrainLoop.__init__
    current_loop = []

    def equal(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left.cpu(), right.cpu())
        elif isinstance(left, np.ndarray):
            assert np.array_equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for k in left:
                equal(left[k], right[k])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for x, y in zip(left, right):
                equal(x, y)
        else:
            assert left == right

    def initialize(self, *v, **kw):
        original_init(self, *v, **kw)
        assert len(self.data.dataset) == a.train_count, 'Wrong TRAIN subset'
        assert len(self.test_data.dataset) == a.test_count, 'Wrong TEST subset'
        assert self.global_batch == int(cfg['global_batch_size'])
        assert self.auxiliary_total_steps == int(cfg['epochs']) * len(self.data)
        assert self.completed_epoch <= a.stop_epoch
        self._stage_start_epoch = self.completed_epoch
        current_loop.append(self)
        self._smoke_start_step = self.step
        self._smoke_gradients = {}
        if not self.resume_step:
            seed_all(a.seed + 100003 * dist.get_rank())
        else:
            state = torch.load(self._training_state_path(), map_location='cpu', weights_only=False)
            if a.smoke_batches:
                equal(self.scaler.state_dict(), state['scaler'])
                equal(self.opt.state_dict(), torch.load(Path(self.resume_checkpoint).with_name(
                    f'opt_{self.resume_step:06d}.pt'), map_location='cpu', weights_only=False))
                equal(self.model.state_dict(), torch.load(self.resume_checkpoint, map_location='cpu', weights_only=False))
                ema = torch.load(Path(self.resume_checkpoint).with_name(
                    f'ema_0.999_{self.resume_step:06d}.pt'), map_location='cpu', weights_only=False)
                for (name, _), value in zip(self.model.named_parameters(), self.ema_params[0]):
                    equal(value, ema[name])
                assert self.step == state['step'] and self.attempted_steps == state['attempted_steps']
                equal(self.diffusion.auxiliary_runtime_state_dict(), state.get('auxiliary_loss_state', {}))
                assert self.data.batch_sampler.epoch == state['sampler_epoch']
                equal(torch.get_rng_state(), state['torch_rng'])
                equal(np.random.get_state(), state['numpy_rng'])
                equal(random.getstate(), state['python_rng'])
                if '_fork_seed' not in state:
                    equal(torch.cuda.get_rng_state(), state['cuda_rng'])
            if '_fork_seed' in state:
                seed_all(state['_fork_seed'])
        if a.smoke_batches:
            self._smoke_ema_before = [p.detach().cpu().clone() for p in self.ema_params[0]]
            manager = self.diffusion.auxiliary_loss_manager
            if manager is not None:
                for name, loss_module in manager.losses.items():
                    self._smoke_gradients[name] = 0.0
                    def gradient_hook(module, inputs, result, name=name):
                        value = result['total'] if isinstance(result, dict) else result
                        if inputs[0].requires_grad and value.requires_grad:
                            grad = torch.autograd.grad(value.sum(), inputs[0], retain_graph=True, allow_unused=True)[0]
                            if grad is not None:
                                assert torch.isfinite(grad).all(), name
                                self._smoke_gradients[name] += float(grad.detach().abs().sum())
                    loss_module.register_forward_hook(gradient_hook)

    TrainLoop.__init__ = initialize
    original_eval = TrainLoop._evaluate_fid_if_due

    class StageComplete(Exception):
        pass

    def evaluate(self, epoch):
        # A parent phase already evaluated its endpoint using the parent recipe.
        if epoch != self._stage_start_epoch or epoch == a.stop_epoch:
            original_eval(self, epoch)
        if epoch >= a.stop_epoch:
            raise StageComplete

    TrainLoop._evaluate_fid_if_due = evaluate
    try:
        train.main(args)
    except StageComplete:
        pass
    if a.smoke_batches:
        loop = current_loop[0]
        assert loop.step > loop._smoke_start_step, 'No successful optimizer update'
        assert any(not torch.equal(x, y.detach().cpu()) for x, y in zip(loop._smoke_ema_before, loop.ema_params[0]))
        gradients = {}
        for name, value in loop._smoke_gradients.items():
            total = torch.tensor(value, dtype=torch.float64, device=dist_util.dev())
            dist.all_reduce(total)
            assert torch.isfinite(total) and total > 0, f'Inactive/nonfinite auxiliary gradient: {name}'
            gradients[name] = float(total)
        out = Path(args.workdir_root) / args.exp
        assert (out / f'train_state_rank{dist.get_rank():03d}_{loop.step:06d}.pt').is_file()
        (out / f'SMOKE_rank{dist.get_rank()}.json').write_text(json.dumps(dict(
            status='PASS', dataset=a.dataset, epoch=loop.completed_epoch, step=loop.step,
            updates=loop.step-loop._smoke_start_step, auxiliary_gradients=gradients,
            strict_resume=bool(loop.resume_step), world=dist.get_world_size(),
            ema_updated=True, reduced_batch_function_test=True), indent=2) + '\n')
    if dist.get_rank() == 0:
        print(f'PHASE_COMPLETE dataset={a.dataset} epoch={a.stop_epoch}', flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=['bbbc', 'jump', 'allen'])
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--stop-epoch', type=int, required=True)
    p.add_argument('--expected-world', type=int, required=True)
    p.add_argument('--train-count', type=int, required=True)
    p.add_argument('--test-count', type=int, required=True)
    p.add_argument('--smoke-batches', type=int, default=0)
    main(p.parse_args())
