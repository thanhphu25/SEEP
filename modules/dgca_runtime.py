"""Opt-in task-boundary integration; the original SEEP engine remains the off path."""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

from modules.dgca import DGCA, capture_rng, restore_rng, isolated_rng, model_hashes, masked_seen, state_hash
from modules.dgca_data import current_loader, collect_features, verify_pairs, resolve_indices
from tools.summarize_dgca import export_metrics

BASE_COMMIT = '80081081ab42769d102f83d447705da0639e2fb1'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    root = Path(__file__).resolve().parents[1]
    return digest_json({str(p.relative_to(root)).replace('\\', '/'): file_sha256(p)
                        for d in (root, root/'modules', root/'tools', root/'configs', root/'tests')
                        for p in sorted(d.glob('*.py'))})


def original_hash(model):
    return state_hash({k: v for k, v in model.state_dict().items() if not k.startswith('head.')})


def split_manifest(data_loader, class_mask):
    # Read metadata only, never load old/future training images to fit DGCA.
    tasks = []
    for task in data_loader:
        entry = {}
        for split in ('train', 'val'):
            base, ids = resolve_indices(task[split].dataset)
            samples = getattr(base, 'samples', None)
            entry[split] = [{'index': i, 'label': int(base.targets[i]),
                             'path': Path(samples[i][0]).parent.name+'/'+Path(samples[i][0]).name
                             if samples is not None else str(i)} for i in ids]
        tasks.append(entry)
    value = {'class_mask': class_mask, 'tasks': tasks, 'batch_size': 32,
             'stats_sampler': 'sequential', 'stats_drop_last': False}
    return {**value, 'sha256': digest_json(value)}


def prepare_dynamic_model(model, state):
    generators = [int(k.split('.')[1]) for k in state if k.startswith('prompt_generators.')]
    if generators:
        model.ensure_prompt_generator(max(generators))
    stored = state.get('stored_unified_prompts')
    if stored is not None:
        device = next(model.parameters()).device
        if 'stored_unified_prompts' not in model._buffers:
            if hasattr(model, 'stored_unified_prompts'):
                delattr(model, 'stored_unified_prompts')
            model.register_buffer('stored_unified_prompts', torch.empty_like(stored, device=device))
        elif model.stored_unified_prompts.shape != stored.shape:
            model.stored_unified_prompts = torch.empty_like(stored, device=device)


def task_optimizer(model, task, args, optimizer, scheduler):
    # Same groups, order and initialization as the original engine.
    if args.lgsp == 'YES' and args.lgsp_type in ['LGSP', 'LSP']:
        model.ensure_prompt_generator(task)
    excluded, groups = set(), []
    steering = getattr(model, 'cls_steering', None)
    if steering is not None:
        steering.activate(task)
        params = [steering.banks[task]]
        groups.append({'params': params, 'lr': args.cls_steering_lr, 'name': 'cls_steering'})
        excluded.update(id(p) for p in params)
    if args.lgsp == 'YES':
        selectors = []
        if args.lgsp_type in ['LGSP', 'LSP']:
            selectors.append((lambda n: 'prompt_generators' in n, getattr(args, 'lr_local', 2e-4), 'local'))
        if args.lgsp_type in ['LGSP', 'GSP']:
            selectors.append((lambda n: n == 'weights', getattr(args, 'lr_Frequency_mask', .03), 'frequency'))
        if args.lgsp_type == 'LGSP':
            selectors.append((lambda n: n in ('alpha', 'beta'), args.lr, 'local_frequency_mixing'))
        for selector, lr, name in selectors:
            params = [p for n, p in model.named_parameters() if selector(n) and p.requires_grad]
            if params:
                groups.append({'params': params, 'lr': lr, 'name': name})
                excluded.update(id(p) for p in params)
    network = [{'params': [p for p in model.parameters() if p.requires_grad and id(p) not in excluded], 'name': 'seep'}, *groups]
    if task == 0 or args.reinit_optimizer:
        optimizer = create_optimizer(args, network)
        if args.sched != 'constant':
            scheduler, _ = create_scheduler(args, optimizer)
    if steering is not None:
        from modules.cls_steering import optimizer_audit
        logger.info('CLS optimizer task {}: {}', task, optimizer_audit(model, optimizer))
    return optimizer, scheduler


@torch.no_grad()
def evaluate_heads(model, original_model, data_loader, device, task, class_mask, dgca, matrices):
    model.eval()
    original_model.eval()
    seen = [c for scope in class_mask[:task+1] for c in scope]
    for test_task in range(task+1):
        correct = {name: 0 for name in matrices}
        count = 0
        for images, labels in data_loader[test_task]['val']:
            images, labels = images.to(device), labels.to(device)
            cls = original_model(images)['pre_logits']
            # task_id is deliberately unavailable to inference routing.
            output = model(images, task_id=-1, learned_id=task, cls_features=cls, train=False)
            logits = {'raw_exact': output['logits'], 'raw_seen': masked_seen(output['logits'], seen),
                      'dgca': masked_seen(dgca.alignment_head(output['classifier_features']), seen)}
            for name, values in logits.items():
                correct[name] += int((values.argmax(1) == labels).sum())
            count += len(labels)
        if count == 0:
            raise ValueError('Empty evaluation task')
        for name in matrices:
            matrices[name][test_task, task] = 100.*correct[name]/count
    logger.info('Task {}: {}', task+1, {k: float(v[:task+1, task].mean()) for k, v in matrices.items()})


def dgca_from_args(args):
    return DGCA(mode=args.dgca, ridge=args.dgca_ridge, shrinkage=args.dgca_shrinkage,
                samples_per_class=args.dgca_samples_per_class, epochs=args.dgca_epochs,
                lr=args.dgca_lr, seed=args.seed)


def validate_args(args):
    if args.distributed or args.task_inc or args.dataset != 'Split-CUB200' or args.batch_size != 32:
        raise ValueError('DGCA protocol requires single device, Split-CUB200, task_inc=False, batch_size=32')
    if not args.reinit_optimizer:
        raise ValueError('DGCA requires baseline reinit_optimizer=True for boundary resume')
    if getattr(args, 'cls_steering', 'off') == 'task' and args.dgca != 'align_only':
        raise ValueError('CLS screening never enables transport')


def experiment_config(args):
    return {k: v for k, v in vars(args).items() if k not in
            ('dgca_resume', 'eval', 'output_dir', 'data_path', 'num_workers', 'pin_mem', 'print_freq', 'num_tasks_to_run')}


def write_provenance(output, args, split, original_model):
    root = Path(__file__).resolve().parents[1]
    def git(*arguments):
        return subprocess.check_output(['git', *arguments], cwd=root).decode('utf-8').strip()
    embedded_patch = root/('cls_steering_implementation.patch' if getattr(args, 'cls_steering', 'off') == 'task' else 'dgca_implementation.patch')
    diff = embedded_patch.read_bytes() if embedded_patch.exists() else subprocess.check_output(['git', 'diff', BASE_COMMIT, '--'], cwd=root)
    (output/'implementation.patch').write_bytes(diff)
    files = {str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
             for directory in (root, root/'modules', root/'tools', root/'configs')
             for p in directory.glob('*.py')}
    cfg = getattr(original_model, 'pretrained_cfg', {})
    cache = Path(torch.hub.get_dir())/'checkpoints'
    weights = {p.name: file_sha256(p) for p in cache.glob('*') if p.is_file()}
    write_json(output/'source_manifest.json', {
        'base_commit': BASE_COMMIT, 'implementation_commit': git('rev-parse', 'HEAD'),
        'branch': git('branch', '--show-current'), 'working_tree_status': git('status', '--short'),
        'diff_sha256': hashlib.sha256(diff).hexdigest(), 'source_files_sha256': files,
        'pretrained_source': cfg, 'cached_weights_sha256': weights,
        'original_representation_sha256': original_hash(original_model),
        'delivery_manifest': json.loads((root/'cls_steering_delivery.json').read_text()) if (root/'cls_steering_delivery.json').exists() else (json.loads((root/'dgca_delivery.json').read_text()) if (root/'dgca_delivery.json').exists() else None)})
    write_json(output/'run_config.json', {**vars(args), 'effective_lr': args.lr,
                                        'historical_environment': {'torch': '2.11.0+cu128', 'torchvision': '0.26.0+cu128', 'timm': '0.6.11'}})
    write_json(output/'split_manifest.json', split)
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode()
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'
    (output/'environment.txt').write_text(f'{platform.platform()}\n{sys.version}\nGPU: {gpu}\n{freeze}', encoding='utf-8')


def save_boundary(path, model, optimizer, scheduler, dgca, matrices, reports, split, args, original_sha256):
    state = {'format': 'SEEP_CLS_STEERING_v1' if getattr(model, 'cls_steering', None) is not None else 'SEEP_DGCA_v1', 'model': model.state_dict(), 'dgca': dgca.state_dict(),
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict() if scheduler is not None else None,
             'requires_grad': {n: p.requires_grad for n, p in model.named_parameters()},
             'matrices': matrices, 'reports': reports, 'split_manifest': split,
             'experiment_config': experiment_config(args), 'rng': capture_rng(),
             'original_representation_sha256': original_sha256}
    if getattr(model, 'cls_steering', None) is not None:
        state['implementation_fingerprint'] = implementation_fingerprint()
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_boundary(path, model, dgca, split, args, original_model):
    # Explicit opt-in --dgca-resume: only load checkpoints created/trusted by user.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    expected = 'SEEP_CLS_STEERING_v1' if getattr(model, 'cls_steering', None) is not None else 'SEEP_DGCA_v1'
    if checkpoint.get('format') != expected:
        raise ValueError('Not a DGCA boundary checkpoint')
    if expected == 'SEEP_CLS_STEERING_v1' and checkpoint.get('implementation_fingerprint') != implementation_fingerprint():
        raise ValueError('CLS implementation changed; refusing a different-source resume')
    if checkpoint['split_manifest'] != split or checkpoint['experiment_config'] != experiment_config(args):
        raise ValueError('Checkpoint split/config mismatch; refusing accidental resume')
    if checkpoint['original_representation_sha256'] != original_hash(original_model):
        raise ValueError('Original pretrained representation differs from checkpoint')
    with isolated_rng():
        prepare_dynamic_model(model, checkpoint['model'])
        model.load_state_dict(checkpoint['model'], strict=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(checkpoint['requires_grad'][name])
        dgca.load_state_dict(checkpoint['dgca'], model.head)
    return checkpoint


def publish_metrics(output, matrices, reports, args=None):
    if getattr(args, 'cls_steering', 'off') == 'task':
        from tools.summarize_cls_steering import export_cls_metrics
        summary = export_cls_metrics(output, matrices, reports, args)
    else:
        summary = export_metrics(output, matrices)
    for task, report in enumerate(reports):
        write_json(output/f'dgca_task_{task+1:02d}.json', report)
    return summary


def _train_and_evaluate_dgca(model, model_without_ddp, original_model, criterion, data_loader,
                            optimizer, lr_scheduler, device, class_mask, args):
    from datasets import build_transform
    from engine import train_one_epoch
    validate_args(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output/'checkpoint'/'dgca_latest.pth'
    path.parent.mkdir(exist_ok=True)
    if not args.dgca_resume and (path.exists() or (output/'run_config.json').exists()):
        raise FileExistsError('Output already contains a DGCA run; choose a new directory or --dgca-resume')
    if args.dgca_resume and not path.exists():
        raise FileNotFoundError('No committed DGCA checkpoint to resume')
    logger.add(str(output/'run.log'), format='{time} | {level} | {message}')
    split = split_manifest(data_loader, class_mask)
    original_sha256 = original_hash(original_model)
    dgca = dgca_from_args(args)
    matrices = {k: np.full((args.num_tasks, args.num_tasks), np.nan) for k in ('dgca', 'raw_seen', 'raw_exact')}
    reports, first = [], 0
    if args.dgca_resume:
        checkpoint = load_boundary(path, model_without_ddp, dgca, split, args, original_model)
        with isolated_rng():
            optimizer, lr_scheduler = task_optimizer(model_without_ddp, dgca.completed_boundary, args, optimizer, lr_scheduler)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(checkpoint['scheduler'])
        matrices, reports = checkpoint['matrices'], checkpoint['reports']
        first = dgca.completed_boundary+1
        restore_rng(checkpoint['rng'])
        publish_metrics(output, matrices, reports, args)
    else:
        with isolated_rng():
            write_provenance(output, args, split, original_model)
    for task in range(first, args.num_tasks):
        if args.num_tasks_to_run > 0 and task >= args.num_tasks_to_run:
            break
        extraction_start = time.perf_counter()
        with isolated_rng():
            loader = current_loader(data_loader[task]['train'].dataset, build_transform(False, args),
                                    seed=args.seed+task, workers=args.num_workers, pin_memory=args.pin_mem)
        # MUST precede ensure_prompt_generator and optimizer construction.
        before = collect_features(model_without_ddp, original_model, loader, device, task-1) if task else None
        before_seconds = time.perf_counter()-extraction_start
        steering = getattr(model_without_ddp, 'cls_steering', None)
        old_banks = steering.hashes(task) if steering is not None else None
        optimizer, lr_scheduler = task_optimizer(model_without_ddp, task, args, optimizer, lr_scheduler)
        if steering is not None:
            from modules.cls_steering import optimizer_audit
            optimizer_report = optimizer_audit(model_without_ddp, optimizer)
        training_start = time.perf_counter()
        for epoch in range(args.epochs):
            train_one_epoch(model, original_model, criterion, data_loader[task]['train'], optimizer,
                            device, epoch, max_norm=args.clip_grad, set_training_mode=True,
                            task_id=task, class_mask=class_mask, args=args)
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)
        if steering is not None and steering.hashes(task) != old_banks:
            raise RuntimeError('Training updated a previous steering bank')
        training_seconds = time.perf_counter()-training_start
        hashes = model_hashes(model_without_ddp)
        extraction_start = time.perf_counter()
        after = collect_features(model_without_ddp, original_model, loader, device, task)
        if set(after['labels'].tolist()) != set(class_mask[task]):
            raise ValueError('Current feature labels differ from current task mask')
        if before is not None:
            verify_pairs(before, after)
        after_seconds = time.perf_counter()-extraction_start
        report = dgca.boundary(task, model_without_ddp.head, after['features'], after['labels'],
                               before['features'] if before is not None else None)
        report.update(feature_before_seconds=before_seconds, feature_after_seconds=after_seconds,
                      seep_training_seconds=training_seconds,
                      pairing_images_sha256=after['images_sha256'], sample_count=len(after['ids']),
                      sample_ids_sha256=digest_json(after['ids'].tolist()), hashes_before=hashes,
                      hashes_after=model_hashes(model_without_ddp))
        if report['hashes_before'] != report['hashes_after']:
            raise RuntimeError('DGCA changed SEEP state')
        if steering is not None:
            from modules.cls_diagnostics import attention_diagnostic
            diagnostics = steering.diagnostics(task)
            if not diagnostics['all_finite'] or not diagnostics['backward_calls']:
                raise RuntimeError('CLS steering did not receive finite training gradients')
            report['cls_steering'] = {
                'old_banks_before': old_banks, 'old_banks_after': steering.hashes(task),
                'all_banks_after': steering.hashes(), 'learning': diagnostics,
                'optimizer_at_boundary': optimizer_report,
                'attention': attention_diagnostic(model_without_ddp, original_model,
                    data_loader[task]['train'].dataset, build_transform(False, args), task, device, args.seed+task)}
        logger.info('DGCA boundary {}: stats={:.2f} MiB, factors={:.2f} MiB, module={:.3f}s, peak RSS={} bytes',
                    task+1, report['statistics_bytes']/2**20, report['factor_bytes']/2**20,
                    report['total_seconds'], report['peak_process_rss_bytes'])
        del before, after, loader
        evaluate_heads(model, original_model, data_loader, device, task, class_mask, dgca, matrices)
        reports.append(report)
        # Commit once, then derive/overwrite exports. Crash recovery cannot append
        # duplicate metrics or apply transport to an already completed boundary.
        save_boundary(path, model_without_ddp, optimizer, lr_scheduler, dgca, matrices, reports, split, args, original_sha256)
        publish_metrics(output, matrices, reports, args)
    return dgca


def train_and_evaluate_dgca(*arguments, **kwargs):
    args = kwargs.get('args', arguments[-1] if arguments else None)
    if getattr(args, 'cls_steering', 'off') != 'task':
        return _train_and_evaluate_dgca(*arguments, **kwargs)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # Do not overwrite evidence from an existing run without explicit resume.
    if not args.dgca_resume and (output/'run_config.json').exists():
        raise FileExistsError('Use a fresh output or --dgca-resume for this CLS run')
    write_json(output/'run_status.json', {'status': 'running', 'full_protocol_complete': False})
    try:
        result = _train_and_evaluate_dgca(*arguments, **kwargs)
        write_json(output/'run_status.json', {'status': 'success', 'session_seconds': time.perf_counter()-started})
        matrices = {k: np.loadtxt(output/f'task_accuracy_matrix_{k}.csv', delimiter=',', ndmin=2)
                    for k in ('dgca', 'raw_seen', 'raw_exact')}
        reports = [json.loads(p.read_text()) for p in sorted(output.glob('dgca_task_*.json'))]
        publish_metrics(output, matrices, reports, args)
        return result
    except BaseException as error:
        write_json(output/'run_status.json', {'status': 'failed', 'error': repr(error),
                                            'session_seconds': time.perf_counter()-started,
                                            'full_protocol_complete': False})
        summary_path = output/'summary.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            summary.update(full_protocol_complete=False, screening_pass=False, practical_priority=False)
            write_json(summary_path, summary)
        raise


def evaluate_checkpoint(model, original_model, data_loader, device, class_mask, args):
    validate_args(args)
    split = split_manifest(data_loader, class_mask)
    dgca = dgca_from_args(args)
    checkpoint = load_boundary(Path(args.output_dir)/'checkpoint'/'dgca_latest.pth', model, dgca, split, args, original_model)
    matrices = {k: v.copy() for k, v in checkpoint['matrices'].items()}
    with isolated_rng():
        evaluate_heads(model, original_model, data_loader, device, dgca.completed_boundary, class_mask, dgca, matrices)
    return {k: v[:dgca.completed_boundary+1, dgca.completed_boundary].tolist() for k, v in matrices.items()}
