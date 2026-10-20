# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------

import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch

import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

import utils
from loguru import logger
import time
from modules.routing import adaptation_forward, mask_seen
from modules.checkpoint import (rng_state, restore_rng, checkpoint_state, atomic_save,
                                load_trusted, validate_checkpoint)

def _prepare_dynamic_buffers_for_checkpoint(model: torch.nn.Module, model_state: dict):
    """
    Register runtime-created buffers before strict checkpoint loading.
    """
    generator_ids = {int(k.split('.')[1]) for k in model_state if k.startswith('prompt_generators.')}
    if generator_ids:
        state = rng_state()
        try:
            model.ensure_prompt_generator(max(generator_ids))
        finally:
            restore_rng(state)
    stored_prompts = model_state.get('stored_unified_prompts')
    if stored_prompts is not None and 'stored_unified_prompts' not in model._buffers:
        if hasattr(model, 'stored_unified_prompts'):
            delattr(model, 'stored_unified_prompts')
        model.register_buffer('stored_unified_prompts', torch.zeros_like(stored_prompts, device=next(model.parameters()).device))


def _update_diagnostics(meters, model, reference, targets, current_classes, batch_size):
    """Labels are used only for reporting; never passed into bank.describe or gate."""
    for name, value in getattr(model, '_gate_diagnostics', {}).items():
        meters.meters[name].update(value, n=batch_size)
    if not hasattr(model, 'reference_bank'):
        return
    _, _, candidates = model.reference_bank.describe(reference['pre_logits'],
        model.args.reference_topk, model.args.reference_temperature)
    if candidates.shape[0] != len(targets):
        raise ValueError('Candidate diagnostics currently require image inputs')
    correct1 = candidates[:, 0].eq(targets)
    correctk = candidates.eq(targets[:, None]).any(-1)
    new = torch.zeros_like(targets, dtype=torch.bool)
    for cls in current_classes:
        new |= targets == cls
    for prefix, selector in (('', torch.ones_like(new)), ('old_', ~new), ('new_', new)):
        count = int(selector.sum())
        if count:
            meters.meters[prefix + 'candidate_top1'].update(correct1[selector].float().mean().item() * 100, n=count)
            meters.meters[prefix + 'candidate_topk'].update(correctk[selector].float().mean().item() * 100, n=count)

def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,
                    feature_prefix_et=None):

    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    optimizer.zero_grad()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    logger.info(f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]')
                
    for step, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq)):
        if args.max_train_batches and step >= args.max_train_batches:
            break
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Detect multi-segment inputs and flatten for the ViT backbone
        is_multiseg = (input.ndim == 5)
        if is_multiseg:
            B, S, C, H, W = input.shape
            flat_input = input.reshape(B * S, C, H, W)
        else:
            flat_input = input
            B = input.shape[0]

        with torch.no_grad():
            if original_model is not None:
                output = original_model(flat_input)
                cls_features = output['pre_logits']
            else:
                cls_features = None

        reference = output
        output = adaptation_forward(model, flat_input, reference, task_id, args,
                                    train=set_training_mode, epoch=epoch)
        logits = output['logits']

        # TSN consensus: average logits over segments before loss
        if is_multiseg:
            logits = logits.reshape(B, S, -1).mean(dim=1)

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
        loss = criterion(logits, target) 
        loss = loss - args.balancing* output['sim_loss']
        unscaled_loss = loss.item()
            
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(unscaled_loss):
            logger.info("Loss is {}, stopping training".format(unscaled_loss))
            sys.exit(1)

        loss = loss / gradient_accumulation_steps
        loss.backward() 
        
        effective_steps = min(len(data_loader), args.max_train_batches or len(data_loader))
        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == effective_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        metric_logger.update(Loss=unscaled_loss)
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=B)
        metric_logger.meters['Acc@5'].update(acc5.item(), n=B)
        _update_diagnostics(metric_logger, model, reference, target, class_mask[task_id], B)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, cur_id=-1, class_mask=None, args=None,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    logger.info('Test: [Task {}]'.format(task_id + 1))

    model.eval()
    original_model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq)):
            if args.max_eval_batches and step >= args.max_eval_batches:
                break
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # -----------------------------------------------------------
            # 1. Detect multi-segment inputs and flatten for the ViT
            # -----------------------------------------------------------
            is_multiseg = (input.ndim == 5)
            if is_multiseg:
                B, S, C, H, W = input.shape
                flat_input = input.reshape(B * S, C, H, W)
            else:
                flat_input = input
                B = input.shape[0]
            # -----------------------------------------------------------

            if original_model is not None:
                # Pass flat_input instead of input
                output = original_model(flat_input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
                
            # Pass flat_input instead of input
            reference = output
            output = adaptation_forward(model, flat_input, reference, cur_id, args)
            logits = output['logits']
            model._gate_diagnostics = output.get('diagnostics', {})

            # -----------------------------------------------------------
            # 2. Average the logits over the temporal segments (TSN consensus)
            # -----------------------------------------------------------
            if is_multiseg:
                # Reshape from (B*S, nb_classes) -> (B, S, nb_classes) and mean across S
                logits = logits.reshape(B, S, -1).mean(dim=1)
            # -----------------------------------------------------------

            if args.eval_scope == 'seen':
                logits = mask_seen(logits, class_mask, cur_id)
            elif args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            # Loss and accuracy are now calculated on the video-level logits
            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            
            # -----------------------------------------------------------
            # 3. Update meters using B (number of videos) instead of frames
            # -----------------------------------------------------------
            metric_logger.meters['Acc@1'].update(acc1.item(), n=B)
            metric_logger.meters['Acc@5'].update(acc5.item(), n=B)
            _update_diagnostics(metric_logger, model, reference, target, class_mask[cur_id], B)
            # -----------------------------------------------------------

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    log_stats = {f'train_task': str(cur_id), f'test_task': str(task_id), f'acc': str(metric_logger.meters['Acc@1'])}

    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, 'test_stats.txt'), 'a') as f:
            f.write(json.dumps(log_stats) + '\n')

    return {**{k: meter.global_avg for k, meter in metric_logger.meters.items()},
            'n': metric_logger.meters['Acc@1'].count,
            'metric_counts': {k: meter.count for k, meter in metric_logger.meters.items()}}


@torch.no_grad()
def evaluate_till_now(model, original_model, data_loader, device, task_id=-1,
                      class_mask=None, acc_matrix=None, args=None):
    from modules.metrics import continual_metrics
    reports, sizes = [], []
    for evaluated_task in range(task_id + 1):
        report = evaluate(model, original_model, data_loader[evaluated_task]['val'], device,
                          task_id=evaluated_task, cur_id=task_id, class_mask=class_mask, args=args)
        reports.append(report)
        sizes.append(report['n'])
        acc_matrix[task_id, evaluated_task] = report['Acc@1']
    summary = continual_metrics(acc_matrix, task_id, sizes)
    keys = set().union(*(r.keys() for r in reports)) - {'metric_counts', 'n'}
    for key in keys:
        numerator = sum(r.get(key, 0.) * r['metric_counts'].get(key, 0) for r in reports)
        count = sum(r['metric_counts'].get(key, 0) for r in reports)
        if count:
            summary[key] = numerator / count
    summary.update(task_id=task_id, method=args.method, eval_router=args.eval_router,
                   eval_scope=args.eval_scope, eval_split=args.eval_split,
                   debug_limited=bool(args.max_train_batches or args.max_eval_batches))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / 'metrics.jsonl', 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(summary, allow_nan=False) + '\n')
    np.save(Path(args.output_dir) / 'accuracy_matrix.npy', acc_matrix)
    np.savetxt(Path(args.output_dir) / 'accuracy_matrix.csv', acc_matrix, delimiter=',')
    logger.info(f'All-seen evaluation: {summary}')
    return summary


def task_optimizer(model, args):
    groups, excluded = [], set()
    def group(name, parameters, lr, weight_decay=None):
        parameters = [p for p in parameters if p.requires_grad]
        if not parameters:
            return
        entry = {'params': parameters, 'lr': lr, 'name': name}
        if weight_decay is not None:
            entry['weight_decay'] = weight_decay
        groups.append(entry)
        excluded.update(id(p) for p in parameters)
    if hasattr(model, 'spatial_gate'):
        group('gate', model.spatial_gate.parameters(), args.gate_lr, args.gate_weight_decay)
    if args.lgsp == 'YES':
        if args.lgsp_type in ('LGSP', 'LSP'):
            group('local', model.prompt_generators.parameters(), args.lr_local)
        if args.lgsp_type in ('LGSP', 'GSP'):
            group('frequency_weights', [model.weights], args.lr_Frequency_mask)
        if args.lgsp_type == 'LGSP':
            group('fusion', [model.alpha, model.beta], args.lr)
    remaining = [p for p in model.parameters() if p.requires_grad and id(p) not in excluded]
    groups.insert(0, {'params': remaining, 'lr': args.lr, 'name': 'general'})
    optimizer = create_optimizer(args, groups)
    scheduler = create_scheduler(args, optimizer)[0] if args.sched != 'constant' else None
    return optimizer, scheduler


def train_and_evaluate(model, model_without_ddp, original_model, criterion, data_loader,
                       optimizer, lr_scheduler, device, class_mask=None, args=None):
    from modules.reference_bank import build_prototypes
    acc_matrix = np.full((args.num_tasks, args.num_tasks), np.nan)
    resume = load_trusted(args.resume) if args.resume else None
    first_task, first_epoch = 0, 0
    if resume:
        validate_checkpoint(resume, args)
        if resume['class_order'] != args.class_order:
            raise ValueError('Resume class order differs')
        _prepare_dynamic_buffers_for_checkpoint(model_without_ddp, resume['model'])
        model_without_ddp.load_state_dict(resume['model'], strict=True)
        original_model.load_state_dict(resume['reference_model'], strict=True)
        acc_matrix = resume['accuracy_matrix']
        first_task = resume['task_id'] + int(resume['task_complete'])
        first_epoch = 0 if resume['task_complete'] else resume['epoch'] + 1
        restore_rng(resume['rng'])
    tasks_to_run = args.num_tasks_to_run or args.num_tasks
    if first_task >= tasks_to_run:
        logger.info('Checkpoint already completed requested task range')
        return
    for task_id in range(first_task, min(tasks_to_run, args.num_tasks)):
        start = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        if args.lgsp == 'YES' and args.lgsp_type in ('LGSP', 'LSP'):
            model_without_ddp.ensure_prompt_generator(task_id)
        prototype_seconds = 0.
        if args.method == 'ccsp':
            bank = model_without_ddp.reference_bank
            existing = set(bank.class_ids[bank.valid].tolist())
            current = set(class_mask[task_id])
            expected_old = {c for scope in class_mask[:task_id] for c in scope}
            resuming_current = bool(resume and task_id == first_task and not resume['task_complete'])
            expected = expected_old | (current if resuming_current else set())
            if existing != expected:
                raise ValueError('Bank lifecycle mismatch: old/future/missing prototypes')
            if not resuming_current:
                before = time.perf_counter()
                random_state = rng_state()
                try:
                    build_prototypes(bank, original_model, data_loader[task_id]['prototype'],
                                     class_mask[task_id], device)
                finally:
                    restore_rng(random_state)
                prototype_seconds = time.perf_counter() - before
        optimizer, lr_scheduler = task_optimizer(model_without_ddp, args)
        if resume and task_id == first_task and not resume['task_complete']:
            optimizer.load_state_dict(resume['optimizer'])
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(resume['lr_scheduler'])
            restore_rng(resume['rng'])
        logger.info('LR groups: ' + str([
            {'name': g['name'], 'current_lr': g['lr'], 'base_lr': g.get('initial_lr', g['lr']),
             'weight_decay': g['weight_decay']} for g in optimizer.param_groups]))
        train_stats = {}
        for epoch in range(first_epoch if task_id == first_task else 0, args.epochs):
            train_stats = train_one_epoch(model, original_model, criterion, data_loader[task_id]['train'],
                optimizer, device, epoch, max_norm=args.clip_grad, task_id=task_id,
                class_mask=class_mask, args=args)
            if lr_scheduler:
                lr_scheduler.step(epoch)
            state = checkpoint_state(model_without_ddp, original_model, optimizer, lr_scheduler,
                                     task_id, epoch, False, acc_matrix, args)
            atomic_save(state, Path(args.output_dir) / 'checkpoint/last.pth')
        test_stats = evaluate_till_now(model, original_model, data_loader, device, task_id,
                                      class_mask, acc_matrix, args)
        state = checkpoint_state(model_without_ddp, original_model, optimizer, lr_scheduler,
                                 task_id, args.epochs - 1, True, acc_matrix, args)
        atomic_save(state, Path(args.output_dir) / f'checkpoint/task{task_id + 1}_checkpoint.pth')
        atomic_save(state, Path(args.output_dir) / 'checkpoint/last.pth')
        bank = getattr(model_without_ddp, 'reference_bank', None)
        prompt_bytes = sum(t.numel() * t.element_size() for name, t in model_without_ddp.named_buffers()
                           if name in ('stored_unified_prompts', 'rainbow_prompt.stored_rainbow_prompts'))
        stats = {'task_id': task_id, 'method': args.method, 'seed': args.seed,
            'eval_router': args.eval_router, 'eval_scope': args.eval_scope, 'eval_split': args.eval_split,
            'train': train_stats, 'evaluation': test_stats,
            'total_params': sum(p.numel() for p in model_without_ddp.parameters()),
            'trainable_params': sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad),
            'gate_params': sum(p.numel() for p in model_without_ddp.spatial_gate.parameters()) if hasattr(model_without_ddp, 'spatial_gate') else 0,
            'gate_parameter_bytes': sum(p.numel() * p.element_size() for p in model_without_ddp.spatial_gate.parameters()) if hasattr(model_without_ddp, 'spatial_gate') else 0,
            'bank_allocated_bytes': bank.memory_bytes if bank is not None else 0,
            'bank_valid_bytes': int(bank.valid.sum()) * (bank.vectors.shape[-1] * bank.vectors.element_size() + 16) if bank is not None else 0,
            'baseline_prompt_bytes': prompt_bytes, 'prototype_pass_seconds': prototype_seconds,
            'task_seconds': time.perf_counter() - start,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
            'lr_requested': args.requested_lr, 'lr_resolved_general': args.lr,
            'lr_groups': [{k: g[k] for k in ('name', 'lr', 'weight_decay')} for g in optimizer.param_groups]}
        with open(Path(args.output_dir) / 'task_stats.jsonl', 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(stats, allow_nan=False) + '\n')
        logger.info(f'Task resources: {stats}')
        resume = None
