"""CPU/stdlib-only release checks. This is not an end-to-end training test."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def parser_without_framework(engine):
    # Execute only the pure argparse/default functions, not framework imports.
    namespace = {'argparse': argparse, 'os': SimpleNamespace(environ={}),
                 'FID_INCEPTION_ENCODER': 'torch_fidelity_inception_v3_2048'}
    wanted = {'model_and_diffusion_defaults', 'sample_defaults', 'str2bool',
              'add_dict_to_argparser', 'create_argparser'}
    definitions = []
    for path in (engine / 'ddbm/script_util.py', engine / 'train.py'):
        definitions += [node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name in wanted]
    module = ast.Module(body=definitions, type_ignores=[])
    exec(compile(module, '<configuration-parser>', 'exec'), namespace)
    return namespace['create_argparser']()


def main():
    ignored = {'.git', '__pycache__', '.venv', 'venv'}
    runtime_roots = {'data', 'external', 'outputs', 'test_outputs', 'wandb'}
    files = sorted(p for p in ROOT.rglob('*') if p.is_file()
                   and not ignored.intersection(p.relative_to(ROOT).parts)
                   and p.relative_to(ROOT).parts[0] not in runtime_roots)
    allowed = {'.py', '.json', '.md', '.txt'}
    for p in files:
        assert not p.is_symlink(), p
        assert p.suffix in allowed or p.name.startswith('LICENSE') or p.name == '.gitignore', p
        assert p.stat().st_size < 1_000_000, p
        text = p.read_text()
        assert not re.search(r'/(?:home|Users)/[A-Za-z0-9_.-]+', text), p
        if p.suffix == '.py':
            ast.parse(text, filename=str(p.relative_to(ROOT)))
        if p.suffix == '.json':
            json.loads(text)
    spec = importlib.util.spec_from_file_location('release_controller', ROOT / 'train.py')
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    plans = json.loads((ROOT / 'configs/plans.json').read_text())
    assert set(plans) == {'bbbc', 'jump', 'allen'}
    for dataset, plan in plans.items():
        parser = parser_without_framework(ROOT / 'engines' / dataset)
        previous = 0
        for phase in plan['phases']:
            config = json.loads((ROOT / phase['config']).read_text())
            values = parser.parse_args([f'--{k}={v}' for k, v in config.items()])
            assert previous < phase['end_epoch'] <= values.epochs
            assert values.epochs == phase['auxiliary_horizon_epochs']
            assert values.global_batch_size % phase['world_size'] == 0
            assert values.noise_schedule == 'ecsi_linear'
            assert values.condition_dim == {'bbbc': 1024, 'jump': 1224, 'allen': 12}[dataset]
            assert values.fid_seed == 42 and values.fid_cfg_scale == 0.2
            assert float(values.ema_rate) == 0.999
            assert values.resume_checkpoint == ''
            if values.cellflux_condition_group_size:
                assert values.global_batch_size // phase['world_size'] % values.cellflux_condition_group_size == 0
            if dataset == 'bbbc':
                assert values.epochs == 150 and values.moment_ramp_start_fraction == 0.6
                assert values.fid_sampler == 'ecsi_2m_reverse_02_tail_001'
            if dataset == 'allen':
                assert values.epochs == 240 and values.moment_ramp_start_fraction == 0.4
                assert values.optimizer_name == 'adamw' and values.global_batch_size == 64
            if dataset == 'jump':
                assert values.epochs == 140 and not values.jump_zoe_lite_loss_enabled
                assert values.microbatch == -1 and values.cellflux_max_eval_samples == 9077
            previous = phase['end_epoch']
        # Entry point and cache commands must be usable without datasets/framework.
        for options in ([], ['--prepare']):
            run = subprocess.run([sys.executable, '-B', ROOT / 'train.py', '--dataset', dataset,
                                  '--dry-run', *options], capture_output=True, text=True)
            assert run.returncode == 0, run.stderr
            assert 'torch.distributed.run' in run.stdout
    jump = json.loads((ROOT / 'configs/jump_e140.json').read_text())
    assert float(jump['jump_auxiliary_loss_budget_ratio']) == 0.065
    assert abs(float(jump['jump_moment_budget_fraction']) * 0.065 - 0.0025) < 1e-12
    assert float(jump['jump_retrieval_temperature']) == 0.12
    allen = json.loads((ROOT / 'configs/allen_e240.json').read_text())
    assert allen['lambda_infonce_max'] == 0.03 and allen['lambda_moment_max'] == 0.15
    print(f'PASS: {len(files)} text source/config/documentation files; three staged recipes; six CLI dry runs.')
    print('No data, checkpoint loading, GPU computation, or end-to-end training performed.')


if __name__ == '__main__':
    main()
