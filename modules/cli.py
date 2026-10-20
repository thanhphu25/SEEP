"""Shared additions to every legacy config parser."""
import argparse


def boolean(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 'yes', '1', 'on'):
        return True
    if value.lower() in ('false', 'no', '0', 'off'):
        return False
    raise argparse.ArgumentTypeError('Expected True/False, yes/no, or 1/0')


def add_research_args(parser):
    # Preserve accepted legacy spelling/defaults while fixing bool('False').
    for action in parser._actions:
        if action.type is bool or action.dest == 'pretrained':
            action.type = boolean
    parser.add_argument('--method', choices=['seep', 'image_gate', 'ccsp'], default='seep')
    parser.add_argument('--eval_router', choices=['legacy_batch', 'per_image'], default='legacy_batch')
    parser.add_argument('--eval_scope', choices=['legacy_head', 'seen'], default='legacy_head')
    parser.add_argument('--reference_topk', type=int, default=4)
    parser.add_argument('--reference_temperature', type=float, default=0.1)
    parser.add_argument('--gate_width', type=int, default=64)
    parser.add_argument('--gate_lr', type=float, default=0.0005)
    parser.add_argument('--gate_weight_decay', type=float, default=0.)
    parser.add_argument('--gate_init', type=float, default=0.9)
    parser.add_argument('--gate_mode', choices=['learned', 'identity', 'zero'], default='learned')
    parser.add_argument('--dev_fraction', type=float, default=0.)
    parser.add_argument('--split_seed', type=int, default=89359)
    parser.add_argument('--eval_split', choices=['dev', 'test'], default='test')
    parser.add_argument('--max_train_batches', type=int, default=0)
    parser.add_argument('--max_eval_batches', type=int, default=0)
    parser.add_argument('--pretrained_checkpoint', default='')
    parser.add_argument('--resume', default='', help='Explicit trusted checkpoint; continue next epoch/task')
    parser.add_argument('--eval_checkpoint', default='', help='Evaluate one trusted checkpoint (implies --eval)')
    parser.add_argument('--checkpoint_task', type=int, default=-1,
                        help='Zero-based last learned task for legacy checkpoint with no taskN filename')
    parser.add_argument('--parse_only', action='store_true', help='Validate CLI without loading data/weights')
    return parser


def validate(args):
    if not 0 <= args.dev_fraction < 1:
        raise ValueError('dev_fraction must be in [0, 1)')
    if args.eval_split == 'dev' and args.dev_fraction == 0:
        raise ValueError('eval_split=dev requires dev_fraction>0')
    if args.reference_topk < 1 or args.reference_temperature <= 0 or args.gate_width < 1:
        raise ValueError('Invalid reference/gate dimensions or temperature')
    if not 0 < args.gate_init < 1:
        raise ValueError('gate_init must be strictly between 0 and 1')
    if min(args.max_train_batches, args.max_eval_batches, args.num_tasks_to_run) < 0:
        raise ValueError('Batch/task limits must be nonnegative')
    if args.epochs < 1 or args.warm_up >= args.epochs:
        raise ValueError('epochs must exceed token warm_up so token caches are trained')
    if args.task_inc and (args.eval_router == 'per_image' or args.eval_scope == 'seen' or args.method != 'seep'):
        raise ValueError('Scientific class-incremental protocol requires task_inc=False')
    if args.method != 'seep' and (args.lgsp != 'YES' or args.lgsp_type not in ('LGSP', 'LSP')):
        raise ValueError('Gate requires an enabled local prompt branch')
    if not args.reinit_optimizer and args.num_tasks > 1:
        raise ValueError('Dynamic generators require reinit_optimizer=True')
    if args.resume and args.eval_checkpoint:
        raise ValueError('Choose resume or eval_checkpoint')
    return args
