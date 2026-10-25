"""Fixed current-train eval-view probe. Never writes prompt caches or uses test data."""
import hashlib

import torch
from torch.utils.data import Subset

from modules.dgca import isolated_rng, model_hashes
from modules.dgca_data import current_loader


@torch.no_grad()
def attention_diagnostic(model, original_model, current_train, transform, task, device, seed):
    bank = model.cls_steering
    hashes = model_hashes(model)
    modes = [(m, m.training) for root in (model, original_model) for m in root.modules()]
    routing = [(obj, n, hasattr(obj, n), getattr(obj, n, None))
               for obj in (model, getattr(model, 'rainbow_prompt', model))
               for n in ('task_id', 'matched_task_id')]
    attentions = [(i, model.blocks[i].attn) for i in bank.layers]
    attributes = [(attn, n, hasattr(attn, n), getattr(attn, n, None))
                  for _, attn in attentions for n in ('_cls_capture', 'key_')]
    disabled = getattr(bank, '_diagnostic_disable', False)
    captures, routes, ids = {}, {}, []
    try:
        with isolated_rng(seed):
            # Local permutation, independent of training loader and global RNG.
            selected = torch.randperm(len(current_train), generator=torch.Generator().manual_seed(seed))[:64].tolist()
            for enabled in (True, False):
                probe = current_loader(Subset(current_train, selected), transform, seed, workers=0)
                bank._diagnostic_disable = not enabled
                for _, attn in attentions:
                    attn._cls_capture = []
                routes[enabled] = []
                model.eval()
                original_model.eval()
                for images, _, stable_ids in probe:
                    cls = original_model(images.to(device))['pre_logits']
                    model(images.to(device), task_id=-1, learned_id=task, cls_features=cls, train=False)
                    routes[enabled].append(int(model.matched_task_id))
                    if enabled:
                        ids.extend(stable_ids.tolist())
                captures[enabled] = {i: list(attn._cls_capture) for i, attn in attentions}
    finally:
        bank._diagnostic_disable = disabled
        for module, mode in modes:
            module.training = mode
        for obj, name, existed, value in routing + attributes:
            if existed:
                setattr(obj, name, value)
            elif hasattr(obj, name):
                delattr(obj, name)
    if hashes != model_hashes(model) or routes[True] != routes[False]:
        raise RuntimeError('Diagnostic mutated model/cache or changed routing')
    layers = {}
    for layer in bank.layers:
        rows = []
        for enabled in (True, False):
            for record in captures[enabled][layer]:
                prefix = record['prefix_length']
                weights = record['cls_attention']
                # Keys: [prefix K, real CLS K, image patch K...].
                image = weights[:, :, prefix+1:]
                conditional = image / image.sum(-1, keepdim=True).clamp_min(1e-12)
                rows.append((enabled, weights[:, :, :prefix].sum(-1), image.sum(-1),
                             -(conditional*conditional.clamp_min(1e-12).log()).sum(-1), conditional))
        on = [r for r in rows if r[0]]
        off = [r for r in rows if not r[0]]
        summary = {}
        for name, records in (('steering', on), ('deltas_disabled_same_state', off)):
            summary[name] = {'cls_to_prefix_mass': float(torch.cat([r[1] for r in records]).mean()),
                             'cls_to_image_patches_mass': float(torch.cat([r[2] for r in records]).mean()),
                             'image_conditional_entropy_nats': float(torch.cat([r[3] for r in records]).mean())}
        summary['image_conditional_l1_change'] = float(torch.cat([
            (a[4]-b[4]).abs().sum(-1) for a, b in zip(on, off)]).mean())
        layers[str(layer)] = summary
    return {'task': task, 'source': 'current_train_eval_view_only', 'sample_count': len(ids),
            'sample_ids': ids, 'sample_ids_sha256': hashlib.sha256(str(ids).encode()).hexdigest(),
            'routed_tasks_per_batch': routes[True], 'layers': layers, 'hashes_unchanged': True,
            'comparison': 'disable all learned CLS deltas at fixed post-training state',
            'limitation': 'Original router chooses one task per batch; no foreground semantics inferred.'}
