"""Task-boundary checkpoint, reproducibility metadata and metrics helpers."""
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from srm_config import srm_config

BASE_COMMIT = '80081081ab42769d102f83d447705da0639e2fb1'
OPERATIONAL = {'data_path', 'output_dir', 'trial', 'device', 'eval', 'distributed',
               'world_size', 'rank', 'gpu', 'dist_url', 'num_tasks_to_run'}
DERIVED = {'lr_pre_scaling', 'lr_effective', 'resolved_seed', 'nb_classes'}


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def trusted_load(path):
    # Only user-owned/trusted baseline and this run's own checkpoints.
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # older torch without weights_only
        return torch.load(path, map_location='cpu')


def environment_audit():
    versions = {}
    for package in ('torch', 'torchvision', 'timm', 'numpy', 'Pillow', 'loguru'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    versions.update(python=platform.python_version(), platform=platform.platform(),
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    cuda_available=torch.cuda.is_available())
    if torch.cuda.is_available():
        versions['gpu'] = torch.cuda.get_device_name(0)
    return versions


def git_provenance():
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=root, text=True).strip())
        return dict(commit=commit, tracked_dirty=dirty, audited_base=BASE_COMMIT)
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, tracked_dirty=None, audited_base=BASE_COMMIT)


def run_identity(args):
    values = vars(args).copy()
    values['lr'] = values.get('lr_pre_scaling', values['lr'])
    values['seed'] = values.get('resolved_seed', values['seed'])
    values.update(srm_config(args))
    return jsonable({k: v for k, v in values.items()
                     if k not in OPERATIONAL | DERIVED and not k.startswith('n1_')})


def optimizer_audit(model, optimizer, task):
    names = {id(p): n for n, p in model.named_parameters()}
    return dict(task=task, groups=[dict(index=i, lr=g['lr'], weight_decay=g.get('weight_decay'),
                                      names=[names[id(p)] for p in g['params']],
                                      numel=sum(p.numel() for p in g['params']))
                                     for i, g in enumerate(optimizer.param_groups)],
                trainable=[n for n, p in model.named_parameters() if p.requires_grad])


def latest_checkpoint(output_dir):
    files = list((Path(output_dir) / 'checkpoint').glob('task*_checkpoint.pth'))
    if not files:
        return None
    def task(path):
        try:
            return int(path.name.split('_')[0][4:])
        except ValueError:
            raise ValueError(f'Unrecognized task checkpoint: {path}')
    return max(files, key=task)


def validate_boundary(checkpoint, args, class_mask):
    required = {'srm_state', 'run_identity', 'completed_task', 'acc_matrix', 'rng_state',
                'per_task', 'resolved_seed', 'class_mask', 'requires_grad', 'optimizer_audit',
                'class_to_idx', 'environment', 'git', 'lr_pre_scaling'}
    if not required <= checkpoint.keys():
        raise ValueError('Boundary checkpoint lacks N1 memory/config/reproducibility metadata; '
                         'use a fresh output directory. Baseline weights cannot initialize N1.')
    if checkpoint['run_identity'] != run_identity(args):
        raise ValueError('Output checkpoint belongs to a different research/N1 config or seed')
    if jsonable(checkpoint['class_mask']) != jsonable(class_mask):
        raise ValueError('Checkpoint class mapping/masks mismatch')
    task = checkpoint['completed_task']
    if task != checkpoint['srm_state']['completed_task'] or not 0 <= task < args.num_tasks:
        raise ValueError('Incomplete/inconsistent task boundary')
    if len(checkpoint['per_task']) != task + 1 or [r['task'] for r in checkpoint['per_task']] != list(range(task + 1)):
        raise ValueError('Incomplete/duplicate boundary metrics')
    matrix = np.asarray(checkpoint['acc_matrix'])
    if matrix.shape != (args.num_tasks, args.num_tasks) or not np.isfinite(matrix).all():
        raise ValueError('Invalid accuracy matrix')
    expected_classes = set(c for mask in class_mask[:task + 1] for c in mask)
    if getattr(args, 'srm_mode', 'off') not in ('off', 'freeze_spectral'):
        if set(checkpoint['srm_state']['class_ids'].tolist()) != expected_classes:
            raise ValueError('Checkpoint memory is missing completed classes')
    if checkpoint['resolved_seed'] != args.resolved_seed:
        raise ValueError('Resolved seed mismatch')


def task_metrics(matrix, task, srm_loss, collector):
    current = matrix[:task + 1, task]
    return dict(task=task, acc_avg=float(current.mean()),
                acc_old=float(current[:-1].mean()) if task else None,
                acc_new=float(current[-1]),
                forgetting=float((matrix[:task, :task + 1].max(axis=1) - current[:-1]).mean()) if task else None,
                bwt=float((current[:-1] - np.diag(matrix)[:task]).mean()) if task else None,
                srm_loss=float(srm_loss), memory_bytes=collector['memory_bytes'],
                collector_seconds=collector['seconds'])


def write_metrics(output_dir, matrix, per_task, total_tasks):
    completed = len(per_task)
    final = per_task[-1] if completed else {}
    # A[i,t], zero-based. Unevaluated cells are null, never invented zero accuracies.
    visible = [[float(matrix[i, t]) if i <= t < completed else None
                for t in range(total_tasks)] for i in range(total_tasks)]
    result = dict(convention='A[i,t] = accuracy percent on task i after training task t',
                  task_indexing='zero-based', A=visible, per_task=per_task,
                  summary=dict(avg_incremental_accuracy=float(np.mean([r['acc_avg'] for r in per_task])) if completed else None,
                               final_accuracy=final.get('acc_avg'), final_forgetting=final.get('forgetting'),
                               completed_tasks=completed, total_tasks=total_tasks, is_complete=completed == total_tasks))
    write_json(Path(output_dir) / 'metrics.json', result)
    # Regenerate from committed boundaries so resume cannot append duplicates.
    (Path(output_dir) / 'acc.txt').write_text(''.join(f"{r['acc_avg']}\n" for r in per_task), encoding='utf-8')
    (Path(output_dir) / 'forgetting.txt').write_text(''.join(f"{r['forgetting']}\n" for r in per_task if r['forgetting'] is not None), encoding='utf-8')
    return result
