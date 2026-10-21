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

def _prepare_dynamic_buffers_for_checkpoint(model: torch.nn.Module, model_state: dict):
    """
    Register runtime-created buffers before strict checkpoint loading.
    """
    generator_ids = [int(k.split('.')[1]) for k in model_state if k.startswith('prompt_generators.')]
    if generator_ids:
        from spectral_response_memory import capture_rng, restore_rng
        rng = capture_rng()
        try:
            model.ensure_prompt_generator(max(generator_ids))
        finally:
            restore_rng(rng)
    stored_prompts = model_state.get('stored_unified_prompts')
    if stored_prompts is not None and 'stored_unified_prompts' not in model._buffers:
        if hasattr(model, 'stored_unified_prompts'):
            delattr(model, 'stored_unified_prompts')
        model.register_buffer('stored_unified_prompts', torch.zeros_like(stored_prompts, device=next(model.parameters()).device))

def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,
                    feature_prefix_et=None, srm=None):

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

        output = model(flat_input, task_id=task_id, learned_id=task_id, cls_features=cls_features,
                       train=set_training_mode, epoch_info=epoch)
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
        classification_loss = criterion(logits, target)
        loss = classification_loss
        loss = loss - args.balancing* output['sim_loss']
        if srm is not None and srm.mode not in ('off', 'freeze_spectral'):
            raw_srm = srm.penalty(model.module if hasattr(model, 'module') else model)
            weighted_srm = args.srm_lambda * raw_srm
            loss = loss + weighted_srm
            metric_logger.update(srm_loss=raw_srm.item(), srm_weighted_loss=weighted_srm.item())
            if epoch % 5 == 0 and step == 0:
                from spectral_response_memory import gradient_diagnostics
                logger.info(f'N1 diagnostics: {gradient_diagnostics(model, classification_loss, weighted_srm, srm.anchors[-1] if srm.anchors is not None else None)}')
        elif srm is not None and srm.mode == 'freeze_spectral' and epoch % 5 == 0 and step == 0:
            from spectral_response_memory import gradient_diagnostics
            logger.info(f'N1 diagnostics: {gradient_diagnostics(model, classification_loss, classification_loss.new_zeros(()))}')
        unscaled_loss = loss.item()
            
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(unscaled_loss):
            logger.info("Loss is {}, stopping training".format(unscaled_loss))
            sys.exit(1)

        loss = loss / gradient_accumulation_steps
        loss.backward() 
        
        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(data_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        metric_logger.update(Loss=unscaled_loss)
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=B)
        metric_logger.meters['Acc@5'].update(acc5.item(), n=B)

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
        for input, target in metric_logger.log_every(data_loader, args.print_freq):
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
            output = model(flat_input, task_id=task_id, learned_id=cur_id, cls_features=cls_features)
            logits = output['logits']

            # -----------------------------------------------------------
            # 2. Average the logits over the temporal segments (TSN consensus)
            # -----------------------------------------------------------
            if is_multiseg:
                # Reshape from (B*S, nb_classes) -> (B, S, nb_classes) and mean across S
                logits = logits.reshape(B, S, -1).mean(dim=1)
            # -----------------------------------------------------------

            if args.task_inc and class_mask is not None:
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
            # -----------------------------------------------------------

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    log_stats = {f'train_task': str(cur_id), f'test_task': str(task_id), f'acc': str(metric_logger.meters['Acc@1'])}

    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, 'test_stats.txt'), 'a') as f:
            f.write(json.dumps(log_stats) + '\n')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id+1):
                
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                            device=device, task_id=i, cur_id=task_id, class_mask=class_mask, args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    # Append Acc@1 to /output_dir/acc.txt
    with open(os.path.join(args.output_dir, 'acc.txt'), 'a') as f:
        f.write(f"{avg_stat[0]}\n")

    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
        # Append Forgetting to /output_dir/forgetting.txt
        with open(os.path.join(args.output_dir, 'forgetting.txt'), 'a') as f:
            f.write(f"{forgetting}\n")
    
    logger.info(result_str)

    return test_stats

def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, args = None,):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    feature_prefix_e = {}
    feature_prefix_et = None
    managed = getattr(args, 'srm_mode', 'off') != 'off' or getattr(args, 'n1_launcher', False)
    srm, per_task, start_task = None, [], 0
    if managed:
        from spectral_response_memory import SpectralResponseMemory, capture_rng, restore_rng
        from n1_runtime import (latest_checkpoint, trusted_load, validate_boundary, run_identity,
                                optimizer_audit, write_json, write_metrics, task_metrics,
                                git_provenance, environment_audit, jsonable)
        from datasets import build_transform, _unwrap_to_base_dataset
        srm = SpectralResponseMemory(args, original_model.num_features, device)
        class_to_idx = getattr(_unwrap_to_base_dataset(data_loader[0]['train'].dataset), 'class_to_idx', None)
        environment = environment_audit()
        provenance = git_provenance()
        latest = latest_checkpoint(args.output_dir)
        if latest is not None:
            checkpoint = trusted_load(latest)
            validate_boundary(checkpoint, args, class_mask)
            resume_audit = dict(previous_environment=checkpoint['environment'], current_environment=environment,
                                previous_git=checkpoint['git'], current_git=provenance)
            write_json(Path(args.output_dir) / 'resume_audit.json', resume_audit)
            if checkpoint['environment'] != environment or checkpoint['git'] != provenance:
                logger.warning('Resume source/environment differs; see resume_audit.json. '
                               'Exact reproduction across environments is unconfirmed.')
            if jsonable(checkpoint['class_to_idx']) != jsonable(class_to_idx):
                raise ValueError('Checkpoint ImageFolder class mapping mismatch')
            _prepare_dynamic_buffers_for_checkpoint(model_without_ddp, checkpoint['model'])
            model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
            for name, param in model_without_ddp.named_parameters():
                param.requires_grad_(checkpoint['requires_grad'][name])
            srm.load_state_dict(checkpoint['srm_state'])
            names = dict(model_without_ddp.named_parameters())
            groups = [{'params': [names[n] for n in group['names']], 'lr': group['lr']}
                      for group in checkpoint['optimizer_audit']['groups']]
            optimizer = create_optimizer(args, groups)
            if args.sched != 'constant':
                lr_scheduler, _ = create_scheduler(args, optimizer)
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # timm scheduler construction writes initial/warmup LRs into groups.
            # Restore the saved optimizer *after* constructing the scheduler.
            optimizer.load_state_dict(checkpoint['optimizer'])
            acc_matrix = np.asarray(checkpoint['acc_matrix']).copy()
            per_task = checkpoint['per_task']
            start_task = checkpoint['completed_task'] + 1
            restore_rng(checkpoint['rng_state'])
            del checkpoint
            write_metrics(args.output_dir, acc_matrix, per_task, args.num_tasks)
            logger.info(f'Restored complete task boundary {start_task}; no re-evaluation or collection')

        
    for task_id in range(start_task, args.num_tasks):
        if args.num_tasks_to_run > 0 and task_id >= args.num_tasks_to_run:
            print(f"Stopping after task {task_id}, as requested by --num_tasks_to_run")
            break
        
        # Ensure pixel prompt generator exists for this task (before optimizer creation)
        if args.lgsp == 'YES' and args.lgsp_type in ['LGSP', 'LSP']:
            if hasattr(model_without_ddp, 'ensure_prompt_generator'):
                model_without_ddp.ensure_prompt_generator(task_id)
        
        if srm is not None:
            srm.begin_task(model_without_ddp, task_id)

        lgsp_params_set = set() # Store ids to exclude from general params
        lgsp_groups = []

        # LGSP Params separation and Optimizer creation
        if args.lgsp == 'YES':
            if args.lgsp_type in ['LGSP', 'LSP']:
                prompt_branch_params = [p for n, p in model.named_parameters() if 'prompt_generators' in n and p.requires_grad]
                if prompt_branch_params:
                    lgsp_groups.append({'params': prompt_branch_params, 'lr': getattr(args, 'lr_local', 2e-4)})
                    for p in prompt_branch_params: lgsp_params_set.add(id(p))
                    
            if args.lgsp_type in ['LGSP', 'GSP']:
                freq_params = [p for n, p in model.named_parameters() if n == 'weights' and p.requires_grad]
                if freq_params:
                    lgsp_groups.append({'params': freq_params, 'lr': getattr(args, 'lr_Frequency_mask', 0.03)})
                    for p in freq_params: lgsp_params_set.add(id(p))
            
            if args.lgsp_type == 'LGSP':
                adapt_params = [p for n, p in model.named_parameters() if n in ('alpha', 'beta') and p.requires_grad]
                if adapt_params:
                    lgsp_groups.append({'params': adapt_params, 'lr': args.lr}) 
                    for p in adapt_params: lgsp_params_set.add(id(p))

        remaining_params = [p for n, p in model.named_parameters() if id(p) not in lgsp_params_set and p.requires_grad]
        
        network_params = [{'params': remaining_params}] # Uses default lr/wd from args inside create_optimizer
        network_params.extend(lgsp_groups)
        
        # Create new optimizer for each task (or task 0) to ensure LGSP params are handled
        if task_id == 0 or (task_id > 0 and args.reinit_optimizer):
            optimizer = create_optimizer(args, network_params)
            if args.sched != 'constant':
                print(f"Re-creating scheduler for task {task_id}")
                lr_scheduler, _ = create_scheduler(args, optimizer)

        # Resume capability: Check if checkpoint exists
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
        if not managed and os.path.exists(checkpoint_path):
            print(f"Checkpoint found at {checkpoint_path}. Resuming...")
            from n1_runtime import trusted_load
            checkpoint = trusted_load(checkpoint_path)
            _prepare_dynamic_buffers_for_checkpoint(model_without_ddp, checkpoint['model'])
            model_without_ddp.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint and lr_scheduler is not None:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            
            # Evaluate to populate stats and matrices
            print(f"Evaluating resumed model for task {task_id}...")
            test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, 
                                        device=device, task_id=task_id, class_mask=class_mask, 
                                        acc_matrix=acc_matrix, args=args)
            continue # Skip training loop
                
        if managed:
            group_audit = optimizer_audit(model_without_ddp, optimizer, task_id)
            logger.info(f'Optimizer groups: {group_audit}')
            write_json(Path(args.output_dir) / f'optimizer_task{task_id + 1}.json', group_audit)
        epoch_srm_losses = []
        for epoch in range(args.epochs):           
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion,
                                          data_loader=data_loader[task_id]['train'], optimizer=optimizer,
                                          device=device, epoch=epoch, max_norm=args.clip_grad,
                                          set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,
                                          feature_prefix_et=feature_prefix_et, srm=srm)
            epoch_srm_losses.append(train_stats.get('srm_loss', 0.0))
            if lr_scheduler:
                lr_scheduler.step(epoch)
        
        
        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
        if managed:
            collector = srm.collect(model_without_ddp, original_model, data_loader[task_id]['train'].dataset,
                                    class_mask[task_id], build_transform(False, args), task_id)
            per_task.append(task_metrics(acc_matrix, task_id, np.mean(epoch_srm_losses), collector))
            write_json(Path(args.output_dir) / f'collector_task{task_id + 1}.json', collector)
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if managed:
                pre_scaling_args = vars(args).copy()
                pre_scaling_args['lr'] = args.lr_pre_scaling
                pre_scaling_args.pop('lr_effective', None)
                state_dict.update(srm_state=srm.state_dict(), completed_task=task_id,
                                  acc_matrix=acc_matrix, per_task=per_task, rng_state=capture_rng(),
                                  run_identity=run_identity(args), resolved_seed=args.resolved_seed,
                                  lr_pre_scaling=args.lr_pre_scaling, lr_effective=args.lr,
                                  args_pre_scaling=pre_scaling_args, effective_args=vars(args).copy(), class_mask=class_mask,
                                  class_to_idx=class_to_idx, git=provenance, environment=environment,
                                  optimizer_audit=optimizer_audit(model_without_ddp, optimizer, task_id),
                                  requires_grad={n: p.requires_grad for n, p in model_without_ddp.named_parameters()})
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            if managed:
                utils.save_on_master(state_dict, checkpoint_path + '.tmp')
                os.replace(checkpoint_path + '.tmp', checkpoint_path)
                write_metrics(args.output_dir, acc_matrix, per_task, args.num_tasks)
            else:
                utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
