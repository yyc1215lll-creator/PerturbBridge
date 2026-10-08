"""Portable controller for the three selected training recipes (no scheduler)."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['bbbc', 'jump', 'allen'], required=True)
    p.add_argument('--phase', type=int, help='One-based phase; omitted runs all phases in order')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data-root', type=Path, default=ROOT / 'data')
    p.add_argument('--output-root', type=Path, default=ROOT / 'outputs')
    p.add_argument('--weights', type=Path, default=ROOT / 'external/inception_v3.pth')
    p.add_argument('--eta', type=float, help='Override the recipe sampling eta')
    p.add_argument('--rho', type=float, help='Override the recipe sampling rho')
    p.add_argument('--cfg-scale', type=float, help='Extra CFG strength: 0.2 means 1.2 cond - 0.2 uncond')
    p.add_argument('--sampling-seed', type=int, default=42)
    p.add_argument('--prepare', action='store_true', help='Build TRAIN-only auxiliary caches, then exit')
    p.add_argument('--dry-run', action='store_true', help='Print commands/configuration without writes or GPU use')
    p.add_argument('--smoke', action='store_true', help='Reduced-batch functional test, not a reported experiment')
    p.add_argument('--smoke-world', type=int, default=2)
    p.add_argument('--smoke-batches', type=int, default=4)
    return p


def resolve_config(args, phase):
    cfg = read(ROOT / phase['config'])
    data = args.data_root.resolve() / args.dataset
    out = args.output_root.resolve()
    cfg.update(data_dir=str(data / 'images'), cellflux_metadata_path=str(data / 'metadata.csv'),
               cellflux_embedding_path=str(data / 'embeddings.csv'),
               auxiliary_encoder_weights_path=str(args.weights.resolve()),
               fid_seed=args.sampling_seed, fid_moa_eval=False,
               fid_moa_repo='', fid_moa_checkpoint='', fid_moa_config='')
    for key in ('moment_stats_path', 'jump_retrieval_stats_path'):
        if cfg.get(key):
            cfg[key] = str(out / 'cache' / Path(cfg[key]).name)
    for arg, key in [('eta', 'fid_eta'), ('rho', 'fid_rho'), ('cfg_scale', 'fid_cfg_scale')]:
        if getattr(args, arg) is not None:
            cfg[key] = getattr(args, arg)
    if 'training_seed' in cfg:
        cfg['training_seed'] = args.seed
    # Keep the original horizon in epochs: stage endpoints MUST NOT shorten the ramp.
    assert int(cfg['epochs']) == phase['auxiliary_horizon_epochs']
    checkpoints = {int(x) for x in str(cfg['epoch_checkpoint_epochs']).split(',') if x}
    checkpoints.add(phase['end_epoch'])
    cfg['epoch_checkpoint_epochs'] = ','.join(map(str, sorted(checkpoints)))
    return cfg


def command(cmd, args, engine):
    print(shlex.join(map(str, cmd)), flush=True)
    if not args.dry_run:
        env = {k: v for k, v in os.environ.items() if not k.startswith('CELLFLUX_FID_MOA_')}
        env.update(PYTHONPATH=str(engine), WANDB_MODE='disabled')
        subprocess.run(list(map(str, cmd)), cwd=ROOT, env=env, check=True)


def distributed(world, script, *options):
    return [sys.executable, '-m', 'torch.distributed.run', '--standalone',
            f'--nproc_per_node={world}', str(script), *options]


def prepare(args, plan, engine):
    cfg = resolve_config(args, plan['phases'][0])
    cache = args.output_root.resolve() / 'cache'
    if not args.dry_run:
        cache.mkdir(parents=True, exist_ok=True)
    if args.dataset == 'jump':
        common = ['--metadata', cfg['cellflux_metadata_path']]
        command([sys.executable, engine / 'scripts/build_base_cache.py', *common,
                 '--image-root', cfg['data_dir'], '--inception-weights', cfg['auxiliary_encoder_weights_path'],
                 '--inception-weights-sha256', cfg['auxiliary_encoder_weights_sha256'],
                 '--output', cache / 'jump_base.pt'], args, engine)
        command(distributed(8, engine / 'scripts/build_cross_plate_cache.py', *common,
                 '--base-cache', cache / 'jump_base.pt', '--images', cfg['data_dir'],
                 '--weights', cfg['auxiliary_encoder_weights_path'],
                 '--output', cache / 'jump_cross_plate.pt'), args, engine)
        return
    for world in ([4, 3] if args.dataset == 'bbbc' else [4]):
        name = f'bbbc_ws{world}.pt' if args.dataset == 'bbbc' else 'allen.pt'
        options = ['--data_dir', cfg['data_dir'], '--metadata_path', cfg['cellflux_metadata_path'],
                   '--embedding_path', cfg['cellflux_embedding_path'],
                   '--encoder_weights_path', cfg['auxiliary_encoder_weights_path'],
                   '--encoder_weights_sha256', cfg['auxiliary_encoder_weights_sha256'],
                   '--output', cache / name, '--dither_seed', '20260810']
        if args.dataset == 'allen':
            options += ['--dataset', 'cellflux_allencell']
        else:
            options += ['--excluded_compounds', cfg['cellflux_excluded_compounds']]
        command(distributed(world, engine / 'scripts/compute_cond_stats.py', *options), args, engine)


def main(args):
    plan = read(ROOT / 'configs/plans.json')[args.dataset]
    engine = ROOT / 'engines' / args.dataset
    if args.smoke:
        if args.prepare or args.smoke_world < 1 or args.smoke_batches < 1:
            raise ValueError('Smoke training uses existing TRAIN caches and positive world/batch counts')
        for index, phase in enumerate(plan['phases'], 1):
            phase['end_epoch'] = index
            phase['world_size'] = 1 if args.dataset == 'allen' and index == 1 else args.smoke_world
    if args.prepare:
        prepare(args, plan, engine)
        return
    if args.phase is not None and not 1 <= args.phase <= len(plan['phases']):
        raise ValueError('phase out of range')
    parent = None
    base = args.output_root.resolve() / ('smoke' if args.smoke else '') / args.dataset / f'seed{args.seed}'
    for index, phase in enumerate(plan['phases'], 1):
        phase_root = base / f'phase{index}'
        training = phase_root / 'training'
        if args.phase is None or args.phase == index:
            cfg = resolve_config(args, phase)
            cfg.update(workdir_root=str(phase_root), exp='training')
            if args.smoke:
                group = int(cfg['cellflux_condition_group_size']) or 8
                cfg.update(global_batch_size=group * phase['world_size'], microbatch=group,
                           cellflux_condition_group_size=group, num_workers=0,
                           cellflux_max_eval_samples=8, fid_batch_size=4,
                           fid_eval_epochs=str(index), epoch_checkpoint_epochs=str(index),
                           auxiliary_gradient_log_interval=0, log_interval=1)
                # Exercise the active auxiliary branch without waiting 90 epochs.
                for key in list(cfg):
                    if key.endswith('ramp_start_fraction'):
                        cfg[key] = 0.0
                    elif key.endswith('ramp_end_fraction'):
                        cfg[key] = 0.001
            config = phase_root / 'config.json'
            if not args.dry_run:
                phase_root.mkdir(parents=True, exist_ok=True)
                config.write_text(json.dumps(cfg, indent=2) + '\n')
            else:
                print(json.dumps({'phase': index, 'end_epoch': phase['end_epoch'],
                                  'world_size': phase['world_size'], 'config': cfg}, indent=2))
            # Carry model + optimizer + scaler + EMA + per-rank RNG to the next output.
            if parent is not None:
                command([sys.executable, ROOT / 'scripts/stage_resume.py', '--source', parent,
                         '--destination', training, '--epoch', plan['phases'][index-2]['end_epoch'],
                         '--world', phase['world_size'], '--seed', args.seed], args, engine)
            command(distributed(phase['world_size'], ROOT / 'scripts/train_worker.py',
                    '--dataset', args.dataset, '--config', config, '--seed', args.seed,
                    '--stop-epoch', phase['end_epoch'], '--expected-world', phase['world_size'],
                    '--train-count', plan['train_count'], '--test-count', 8 if args.smoke else plan['test_count'],
                    '--smoke-batches', args.smoke_batches if args.smoke else 0), args, engine)
        parent = training


if __name__ == '__main__':
    main(parser().parse_args())
