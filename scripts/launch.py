import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    'smoke': ['--epochs', '1', '--warm_up', '0', '--warmup-epochs', '0',
              '--num_tasks_to_run', '2', '--max_train_batches', '2', '--max_eval_batches', '2'],
    'pilot': ['--epochs', '10', '--warm_up', '2', '--warmup-epochs', '1', '--num_tasks_to_run', '3'],
    'full': ['--epochs', '30', '--warm_up', '5', '--warmup-epochs', '5', '--num_tasks_to_run', '0'],
}


def base_command(python, data_path, seed=89359, dev_fraction=.1, eval_split='dev'):
    return [str(python), '-u', 'main.py', 'cub200_10task',
        '--model', 'vit_base_patch16_224', '--batch-size', '32',
        '--data-path', str(data_path), '--device', 'cuda:0', '--seed', str(seed),
        '--num_tasks', '10', '--sched', 'cosine', '--lr', '0.03',
        '--lr_local', '0.0005', '--lr_Frequency_mask', '0.03',
        '--lgsp', 'YES', '--lgsp_type', 'LGSP', '--length', '20', '--top_k', '1', '--size', '10',
        '--D1', '56', '--D2', '96', '--use_linear', 'True', '--relation_type', 'attention',
        '--balancing', '0.01', '--e_prompt_layer_idx', *map(str, range(12)),
        '--self_attn_idx', *map(str, range(6)), '--temperature', '1.0', '--print_freq', '100',
        '--eval_router', 'per_image', '--eval_scope', 'seen', '--task_inc', 'False',
        '--reference_topk', '4', '--reference_temperature', '0.1', '--gate_width', '64',
        '--gate_lr', '0.0005', '--gate_weight_decay', '0.0', '--gate_init', '0.9',
        '--gate_mode', 'learned', '--dev_fraction', str(dev_fraction), '--eval_split', eval_split]


def run_command(base, method, stage, output_root='/kaggle/working/runs',
                resume='', eval_checkpoint='', pretrained_checkpoint=''):
    if method not in ('seep', 'image_gate', 'ccsp') or stage not in PROFILES:
        raise ValueError((method, stage))
    seed = base[base.index('--seed') + 1]
    fraction = base[base.index('--dev_fraction') + 1]
    split = base[base.index('--eval_split') + 1]
    output = Path(output_root) / stage / method / ('seed' + seed) / f'{split}_holdout{fraction}'
    command = list(base) + ['--method', method, '--output_dir', str(output),
                           '--trial', f'{stage}_{method}_s{seed}'] + PROFILES[stage]
    for flag, value in (('--resume', resume), ('--eval_checkpoint', eval_checkpoint),
                        ('--pretrained_checkpoint', pretrained_checkpoint)):
        if value:
            command += [flag, str(value)]
    return command


def launch(command, project=ROOT):
    output = Path(command[command.index('--output_dir') + 1])
    output.mkdir(parents=True, exist_ok=True)
    print(shlex.join(command), flush=True)
    # Record the actual venv, not the notebook kernel environment.
    frozen = subprocess.check_output([command[0], '-m', 'pip', 'freeze'], text=True)
    (output / 'pip-freeze.txt').write_text(frozen, encoding='utf-8')
    (output / 'launch-command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
    subprocess.run(command, cwd=project, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=PROFILES, required=True)
    parser.add_argument('--method', choices=['seep', 'image_gate', 'ccsp', 'all'], default='seep')
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--seed', type=int, default=89359)
    parser.add_argument('--dev-fraction', type=float, default=.1)
    parser.add_argument('--eval-split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--output-root', default='runs')
    parser.add_argument('--resume', default='')
    parser.add_argument('--eval-checkpoint', default='')
    parser.add_argument('--pretrained-checkpoint', default='')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.method == 'all' and (args.resume or args.eval_checkpoint):
        parser.error('Resume/evaluate one method at a time')
    base = base_command(sys.executable, args.data_path, args.seed, args.dev_fraction, args.eval_split)
    for method in ('seep', 'image_gate', 'ccsp') if args.method == 'all' else [args.method]:
        command = run_command(base, method, args.stage, args.output_root, args.resume,
                              args.eval_checkpoint, args.pretrained_checkpoint)
        if args.dry_run:
            print(shlex.join(command))
        else:
            launch(command)


if __name__ == '__main__':
    main()
