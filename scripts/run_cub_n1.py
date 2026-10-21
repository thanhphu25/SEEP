"""Kaggle launcher: fresh repo config or optional trusted baseline configuration."""
import argparse
import gc
import importlib
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from srm_config import add_srm_args, validate_srm
from n1_runtime import (trusted_load, write_json, run_identity, latest_checkpoint,
                        git_provenance, environment_audit, OPERATIONAL, jsonable)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}


def preflight_data(root):
    folder = Path(root) / 'CUB_200_2011'
    mappings, counts = {}, {}
    for split in ('train', 'test'):
        path = folder / split
        if not path.is_dir():
            raise ValueError(f'Missing {path}. For raw CUB use scripts/prepare_cub.py with '
                             'an output root under /kaggle/working; never move input images.')
        classes = sorted(p.name for p in path.iterdir() if p.is_dir())
        if len(classes) != 200:
            raise ValueError(f'{path}: expected 200 classes, found {len(classes)}')
        mappings[split] = {name: i for i, name in enumerate(classes)}
        counts[split] = {}
        for name in classes:
            count = sum(1 for p in (path / name).rglob('*')
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            if not count:
                raise ValueError(f'Empty image class: {split}/{name}')
            counts[split][name] = count
    if mappings['train'] != mappings['test']:
        raise ValueError('Train/test class_to_idx mismatch')
    return dict(root=str(Path(root).resolve()), class_to_idx=mappings['train'], counts=counts,
                evaluation_split='official test (not tuning validation)', read_only_supported=True)


def baseline_configuration(path, supplied_seed):
    checkpoint = trusted_load(path)
    if 'args' not in checkpoint:
        raise ValueError('Baseline checkpoint lacks args. Supply the exact saved Namespace/notebook '
                         'configuration and seed log; research defaults will not be guessed.')
    saved = checkpoint['args']
    baseline = dict(saved) if isinstance(saved, dict) else vars(saved).copy()
    meta = {key: checkpoint.get(key) for key in
            ('lr_pre_scaling', 'resolved_seed', 'git', 'git_commit', 'source_commit', 'environment')}
    del checkpoint, saved
    gc.collect()  # release pretrained/model/optimizer tensors before any GPU imports
    tasks = baseline.get('num_tasks')
    if (baseline.get('dataset') != 'Split-CUB200' or tasks not in (10, 20)
            or baseline.get('subparser_name') != f'cub200_{tasks}task'
            or baseline.get('lgsp') != 'YES' or baseline.get('lgsp_type') != 'LGSP'
            or baseline.get('distributed', False) or baseline.get('world_size', 1) != 1
            or baseline.get('task_inc') is not False
            or baseline.get('srm_mode', 'off') != 'off'):
        raise ValueError('Expected a single-GPU CUB 10/20 LGSP baseline checkpoint, task_inc=False, '
                         'matching subparser_name, srm_mode=off')
    parser = argparse.ArgumentParser()
    importlib.import_module(f'configs.cub200_{tasks}task').get_args_parser(parser)
    defaults = vars(parser.parse_args([]))
    needed = set(defaults) - {key for key in defaults if key.startswith('srm_')} - OPERATIONAL
    missing = sorted(needed - baseline.keys())
    if missing:
        raise ValueError(f'Baseline args missing research fields {missing}; exact notebook patches/config required')
    if supplied_seed <= 0:
        raise ValueError('--seed must be >0 and be the actual seed from the baseline log')
    old_seed = baseline.get('seed')
    if old_seed is None or old_seed < 0:
        raise ValueError('Missing/invalid baseline seed')
    if old_seed > 0 and old_seed != supplied_seed:
        raise ValueError('--seed differs from positive checkpoint seed')
    resolved = meta['resolved_seed'] or baseline.get('resolved_seed')
    if resolved and resolved != supplied_seed:
        raise ValueError('--seed differs from checkpoint resolved_seed')
    if old_seed == 0:
        print('Baseline args.seed=0 made initialize_experiment choose a random seed. '
              '--seed must come from its Seed: log; this value is USER-CONFIRMED, '
              'the launcher has not verified the log.')
    scale = baseline['batch_size'] * (1 if baseline['unscale_lr'] else baseline.get('world_size', 1)) / 256.0
    if scale <= 0 or not math.isfinite(baseline['lr']) or baseline['lr'] <= 0:
        raise ValueError('Invalid baseline LR/batch size')
    pre = meta['lr_pre_scaling']
    if pre is None:
        pre = baseline.get('lr_pre_scaling', baseline['lr'] / scale)
    if not math.isclose(pre * scale, baseline['lr'], rel_tol=1e-7, abs_tol=1e-12):
        raise ValueError('lr_pre_scaling metadata disagrees with saved effective args.lr')
    meta['lr'] = dict(pre_scaling=pre, effective=baseline['lr'], scale=scale,
                      lr_local=baseline['lr_local'], lr_Frequency_mask=baseline['lr_Frequency_mask'])
    return baseline, meta


def prepare(options):
    fresh = options.config is not None
    if fresh:
        if options.seed <= 0:
            raise ValueError('--seed must be >0 for a reproducible fresh run')
        config_parser = argparse.ArgumentParser()
        importlib.import_module('configs.' + options.config).get_args_parser(config_parser)
        baseline = vars(config_parser.parse_args([]))
        baseline['subparser_name'] = options.config
        scale = baseline['batch_size'] / 256.0  # fresh single-GPU repo defaults
        meta = dict(lr=dict(pre_scaling=baseline['lr'], effective=baseline['lr'] * scale,
                            scale=scale, lr_local=baseline['lr_local'],
                            lr_Frequency_mask=baseline['lr_Frequency_mask']),
                    resolved_seed=None)
    else:
        baseline, meta = baseline_configuration(options.baseline_checkpoint, options.seed)
    values = baseline.copy()
    updates = dict(data_path=str(Path(options.data_path).resolve()),
                   output_dir=str(Path(options.output_dir).resolve()),
                   trial=f'n1_cub{baseline["num_tasks"]}_{options.srm_mode}_lambda{options.srm_lambda}_seed{options.seed}',
                   device='cuda:0', eval=False, distributed=False, world_size=1, rank=0, gpu=0,
                   num_tasks_to_run=options.num_tasks_to_run,
                   seed=options.seed, lr=meta['lr']['pre_scaling'],
                   lr_pre_scaling=meta['lr']['pre_scaling'])
    updates.update({k: v for k, v in vars(options).items() if k.startswith('srm_')})
    values.update(updates)
    values['n1_launcher'] = True
    args = argparse.Namespace(**values)
    validate_srm(args)
    if options.num_tasks_to_run < 0 or options.num_tasks_to_run > args.num_tasks:
        raise ValueError('--num-tasks-to-run must be 0..num_tasks (0 means all)')
    if any(k in os.environ for k in ('RANK', 'WORLD_SIZE', 'SLURM_PROCID')):
        raise ValueError('Launcher requires a single process; distributed environment variables are set')
    diff = dict(operational={}, N1={}, normalization={}, research={})
    for key, value in updates.items():
        if key not in baseline or jsonable(value) != jsonable(baseline[key]):
            category = ('N1' if key.startswith('srm_') else
                        'normalization' if key in ('seed', 'lr', 'lr_pre_scaling') else
                        'operational' if key in OPERATIONAL else 'research')
            diff[category][key] = dict(baseline=baseline.get(key), effective=value)
    if diff['research']:
        raise ValueError(f'Unexpected research overrides: {diff["research"]}')
    output = Path(args.output_dir)
    latest = latest_checkpoint(output)
    if latest:
        old = trusted_load(latest)
        if old.get('run_identity') != run_identity(args) or 'srm_state' not in old:
            raise ValueError('Output contains an incompatible/legacy checkpoint; choose a separate OUT')
        del old
        gc.collect()
    data = preflight_data(args.data_path)
    current_git = git_provenance()
    old_git = meta.get('git')
    old_commit = (old_git.get('commit') if isinstance(old_git, dict) else old_git) or meta.get('git_commit') or meta.get('source_commit')
    provenance = dict(baseline_commit=old_commit, current=current_git,
                      status='fresh_run_repo_defaults' if fresh else 'unknown' if not old_commit else
                      'same_commit' if old_commit == current_git['commit'] else 'different_commit',
                      baseline_environment=meta.get('environment'),
                      note='Fresh experiment from repo defaults; no historical baseline comparison asserted' if fresh
                      else 'Audited base is not proof of the historical notebook source or patches')
    report = dict(args=vars(args), run_identity=run_identity(args), learning_rates=meta['lr'],
                  config_source='repo_defaults' if fresh else 'baseline_checkpoint_args',
                  seed=dict(value=options.seed, confirmation='user-selected fresh experiment seed' if fresh
                            else 'user-confirmed baseline log seed; log not inspected',
                            checkpoint_seed=None if fresh else baseline['seed'], checkpoint_resolved_seed=meta['resolved_seed']),
                  baseline_checkpoint=None if fresh else str(Path(options.baseline_checkpoint).resolve()),
                  provenance=provenance, environment=environment_audit(), dataset=data,
                  initialization='fresh pretrained initialization; no baseline checkpoint' if fresh
                  else 'same pretrained initialization; baseline weights are NOT loaded',
                  resume='latest complete task boundary only; interrupted task restarts')
    write_json(output / 'effective_config.json', report)
    write_json(output / 'config_diff.json', diff)
    print(f'Preflight OK: {args.subparser_name}, mode={args.srm_mode}, seed={args.seed}')
    print(f'LR pre={args.lr}, effective={meta["lr"]["effective"]}; local/frequency LRs unchanged')
    print(f'Config source: {report["config_source"]}; provenance: {provenance["status"]}. '
          f'Review {output / "effective_config.json"}')
    return args


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config', choices=('cub200_10task', 'cub200_20task'),
                        help='Fresh experiment from repository defaults; no old checkpoint needed')
    source.add_argument('--baseline-checkpoint', help='Optional: reuse args from your trusted baseline .pth')
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-tasks-to-run', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    add_srm_args(parser)
    return parser


def cli(argv=None):
    parser = get_parser()
    options = parser.parse_args(argv)
    try:
        args = prepare(options)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if not options.dry_run:
        from main import main
        main(args)


if __name__ == '__main__':
    cli()
