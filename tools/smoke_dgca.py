"""Two current-task batches on the real pretrained SEEP path; no accuracy claim.

Run in a separate process before the full experiment. Does not save training
checkpoints or alter full-run RNG. Task 1 images are used only by task 1 probes.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import Subset
from timm.models import create_model

import models
from configs.cub200_10task import get_args_parser
from datasets import build_continual_dataloader, build_transform
from main import set_seed
from modules.dgca import DGCA, model_hashes, peak_process_rss_bytes
from modules.dgca_data import current_loader, collect_features, verify_pairs, resolve_indices
from modules.dgca_runtime import prepare_dynamic_model, task_optimizer


def smoke(args):
    assert torch.cuda.is_available(), 'Smoke requires the Kaggle GPU'
    args.distributed = False
    set_seed(3407)
    loaders, masks = build_continual_dataloader(args)
    device = torch.device(args.device)
    original = create_model(args.model, pretrained=True, num_classes=200).to(device).eval().requires_grad_(False)
    options = {name: getattr(args, name) for name in
               ('embedding_key', 'prompt_pool', 'prompt_key', 'top_k', 'batchwise_prompt', 'prompt_key_init',
                'head_type', 'use_prompt_mask', 'use_g_prompt', 'g_prompt_length', 'g_prompt_layer_idx',
                'use_prefix_tune_for_g_prompt', 'use_e_prompt', 'e_prompt_layer_idx',
                'use_prefix_tune_for_e_prompt', 'same_key_value', 'D1', 'D2', 'relation_type',
                'use_linear', 'warm_up', 'KI_iter', 'self_attn_idx')}
    options.update(prompt_length=args.length, prompt_init=args.prompt_key_init,
                   pool_size=args.size, n_tasks=args.num_tasks, args=args)
    model = create_model(args.model, pretrained=True, num_classes=200, **options).to(device)
    if args.cls_steering == 'task':
        from modules.cls_pretrained import verify_pretrained_models
        verify_pretrained_models(original, model)
    for name, parameter in model.named_parameters():
        if name.startswith(tuple(args.freeze)):
            parameter.requires_grad = False
    args.lr *= args.batch_size/256
    dgca = DGCA(mode=args.dgca if args.dgca != 'off' else 'transport', epochs=1, samples_per_class=4)
    reports = []
    for task in range(1 if args.cls_steering == 'task' else 2):
        current = loaders[task]['train'].dataset
        base, ids = resolve_indices(current)
        selected = list(range(min(64, len(current))))
        counts = {label: sum(int(base.targets[ids[i]]) == label for i in selected) for label in masks[task]}
        selected = [i for i in selected if counts[int(base.targets[ids[i]])] >= 2]
        subset = Subset(current, selected)
        probe = current_loader(subset, build_transform(False, args), 3407+task)
        before = collect_features(model, original, probe, device, task-1) if task else None
        optimizer, _ = task_optimizer(model, task, args, None, None)
        model.train()
        for images, labels, _ in probe:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                cls = original(images)['pre_logits']
            output = model(images, task_id=task, learned_id=task, cls_features=cls, train=True, epoch_info=args.warm_up)
            seen_current = torch.tensor(masks[task], device=device)
            local_targets = (labels[:, None] == seen_current[None, :]).long().argmax(1)
            loss = nn.functional.cross_entropy(output['logits'][:, seen_current], local_targets)-args.balancing*output['sim_loss']
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
        after = collect_features(model, original, probe, device, task)
        if before is not None:
            verify_pairs(before, after)
        hashes = model_hashes(model)
        report = dgca.boundary(task, model.head, after['features'], after['labels'], before['features'] if task else None)
        assert hashes == model_hashes(model)
        report['feature_shape'] = list(after['features'].shape)
        reports.append(report)
    # Strict checkpoint reload, including both dynamically created generators.
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory)/'smoke.pth'
        torch.save({'model': model.state_dict(), 'dgca': dgca.state_dict()}, checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        prepare_dynamic_model(model, state['model'])
        model.load_state_dict(state['model'], strict=True)
        clone = DGCA(epochs=1, samples_per_class=4)
        clone.load_state_dict(state['dgca'], model.head)
        torch.testing.assert_close(clone.alignment_head(after['features'].to(device)),
                                   dgca.alignment_head(after['features'].to(device)), rtol=0, atol=0)
    print(json.dumps({'kind': 'smoke_only_no_scientific_accuracy', 'tasks': reports,
                      'cls_steering': model.cls_steering.diagnostics(0) if model.cls_steering is not None else None,
                      'peak_process_rss_bytes': peak_process_rss_bytes(),
                      'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    get_args_parser(parser)
    smoke(parser.parse_args())
