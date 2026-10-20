"""Trusted local checkpoints, explicit resume and RNG restoration."""
import copy
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path
import numpy as np
import torch


BASE_COMMIT = '80081081ab42769d102f83d447705da0639e2fb1'


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def source_metadata():
    root = Path(__file__).resolve().parents[1]
    metadata_file = root / 'source_metadata.json'
    metadata = json.loads(metadata_file.read_text(encoding='utf-8')) if metadata_file.exists() else {}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        commit = metadata.get('source_commit', BASE_COMMIT)
    sources = list(root.glob('*.py'))
    for directory in ('configs', 'modules', 'scripts', 'continual_datasets'):
        sources.extend((root / directory).rglob('*.py'))
    digest = hashlib.sha256()
    for path in sorted(sources):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return {'base_commit': BASE_COMMIT, 'source_commit': commit,
            'runtime_source_sha256': digest.hexdigest(), 'package_metadata': metadata}


def canonical_config(args):
    ignored = {'output_dir', 'trial', 'resume', 'eval_checkpoint', 'checkpoint_task', 'eval', 'parse_only', 'device',
               'num_workers', 'pin_mem', 'print_freq', 'data_path', 'pretrained_checkpoint',
               'requested_config', 'resolved_config', 'split_ids', 'source_metadata', 'requested_lr',
               'rank', 'gpu', 'world_size', 'distributed', 'dist_url', 'num_tasks_to_run'}
    return {k: v for k, v in vars(args).items() if k not in ignored}


def validate_checkpoint(checkpoint, args, evaluation=False):
    if 'protocol_config' not in checkpoint:
        if not evaluation or args.method != 'seep':
            raise ValueError('Legacy checkpoint supports seep evaluation only; no causal bank/resume metadata')
        return
    stored, current = checkpoint['protocol_config'], canonical_config(args)
    ignored = {'eval_router', 'eval_scope', 'gate_mode', 'max_eval_batches'} if evaluation else set()
    # Evaluation may select another official partition deliberately; architecture,
    # trained split and method still must match. Use the same dev_fraction/split seed.
    if evaluation:
        ignored |= {'eval_split', 'epochs', 'warm_up', 'warmup_epochs', 'max_train_batches'}
    differences = [k for k in set(stored) | set(current)
                   if k not in ignored and json.dumps(stored.get(k), sort_keys=True) != json.dumps(current.get(k), sort_keys=True)]
    if differences:
        raise ValueError('Checkpoint/config mismatch: ' + ', '.join(sorted(differences)))
    if checkpoint['split_ids'] != args.split_ids:
        raise ValueError('Checkpoint train/dev sample IDs differ')
    current_source = getattr(args, 'source_metadata', {}).get('runtime_source_sha256')
    if not evaluation and current_source and checkpoint.get('runtime_source_sha256') != current_source:
        raise ValueError('Checkpoint/source implementation hash mismatch')


def load_trusted(path):
    # Includes Python/NumPy RNG objects; load only checkpoints you trust.
    return torch.load(path, map_location='cpu', weights_only=False)


def evaluation_task(checkpoint, path, args):
    if 'task_id' in checkpoint:
        task = checkpoint['task_id']
        if not checkpoint['task_complete'] and checkpoint['epoch'] < checkpoint['protocol_config']['warm_up']:
            raise ValueError('Checkpoint is still in token warm-up; trained token cache is unavailable')
    elif args.checkpoint_task >= 0:
        task = args.checkpoint_task
    else:
        match = re.fullmatch(r'task(\d+)_checkpoint\.pth', Path(path).name)
        if not match:
            raise ValueError('Legacy checkpoint has no task metadata; supply --checkpoint_task (zero-based)')
        task = int(match.group(1)) - 1
    if not 0 <= task < args.num_tasks:
        raise ValueError('Checkpoint task is outside configured task split')
    return task


def evaluation_history(checkpoint, args):
    history = np.full((args.num_tasks, args.num_tasks), np.nan)
    protocol = checkpoint.get('protocol_config', {})
    fields = ('method', 'eval_router', 'eval_scope', 'eval_split', 'gate_mode', 'max_eval_batches')
    if 'accuracy_matrix' in checkpoint and all(protocol.get(k) == getattr(args, k) for k in fields):
        history = checkpoint['accuracy_matrix'].copy()
    return history


def initialize_local_weights(reference, adaptation, path):
    if not path:
        return
    path = Path(path)
    if path.suffix == '.npz':
        reference.load_pretrained(str(path))
        adaptation.load_pretrained(str(path))
        return
    checkpoint = load_trusted(path)
    state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    expected = reference.state_dict()
    # The pretraining head can have a different class count; do not replace CIL head.
    state = {k: v for k, v in state.items() if not k.startswith(('head.', 'fc.'))}
    required = {k for k in expected if not k.startswith('head.')}
    if set(state) != required or any(state[k].shape != expected[k].shape for k in required):
        raise ValueError('Local weights must contain exactly the matching unprompted backbone (head optional)')
    for model in (reference, adaptation):
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise ValueError(result.unexpected_keys)


def atomic_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def checkpoint_state(model, reference, optimizer, scheduler, task, epoch, complete, matrix, args):
    return {'model': model.state_dict(), 'reference_model': reference.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict() if scheduler is not None else None,
            'task_id': task, 'epoch': epoch, 'task_complete': complete, 'rng': rng_state(),
            'accuracy_matrix': matrix, 'class_order': args.class_order, 'split_ids': args.split_ids,
            'protocol_config': canonical_config(args), 'requested_config': args.requested_config,
            'resolved_config': copy.deepcopy(vars(args)), 'seed': args.seed,
            **args.source_metadata}
