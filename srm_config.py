"""Shared N1 CLI, intentionally independent of torch/training imports."""
import math

MODES = ('off', 'l2', 'semantic_diag', 'semantic_full', 'freeze_spectral')
DEFAULTS = dict(srm_mode='off', srm_lambda=1.0, srm_proj_dim=16,
                srm_samples_per_class=8, srm_probe_batch_size=1,
                srm_memory_seed=3407, srm_normalize='trace')


def add_srm_args(parser):
    parser.add_argument('--srm-mode', choices=MODES, default='off')
    parser.add_argument('--srm-lambda', type=float, default=1.0, help='Untuned pilot strength')
    parser.add_argument('--srm-proj-dim', type=int, default=16)
    parser.add_argument('--srm-samples-per-class', type=int, default=8)
    parser.add_argument('--srm-probe-batch-size', type=int, default=1)
    parser.add_argument('--srm-memory-seed', type=int, default=3407)
    parser.add_argument('--srm-normalize', choices=('trace', 'none'), default='trace')


def srm_config(args):
    return {key: getattr(args, key, default) for key, default in DEFAULTS.items()}


def validate_srm(args):
    cfg = srm_config(args)
    if cfg['srm_mode'] not in MODES:
        raise ValueError('Unknown srm_mode')
    if not math.isfinite(cfg['srm_lambda']) or cfg['srm_lambda'] < 0:
        raise ValueError('srm_lambda must be finite and nonnegative')
    for key in ('srm_proj_dim', 'srm_samples_per_class', 'srm_probe_batch_size'):
        if cfg[key] <= 0:
            raise ValueError(key + ' must be positive')
    if cfg['srm_normalize'] not in ('trace', 'none'):
        raise ValueError('Unsupported srm_normalize')
    if cfg['srm_mode'] != 'off':
        if (args.dataset != 'Split-CUB200' or args.num_tasks not in (10, 20)
                or args.lgsp != 'YES' or args.lgsp_type != 'LGSP'
                or getattr(args, 'distributed', False) or getattr(args, 'world_size', 1) != 1
                or args.task_inc or not args.pretrained or not args.freeze):
            raise ValueError('N1 MVP requires single-GPU pretrained frozen-reference CUB 10/20, '
                             'LGSP YES/LGSP and task_inc=False')
    return cfg
