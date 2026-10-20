"""Explicit subgroup routes; no true labels or evaluation-task IDs enter routing."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def predict_routes(raw_cls, task_keys, learned_id):
    if not 0 <= learned_id < len(task_keys):
        raise ValueError('Invalid learned task range')
    return (F.normalize(raw_cls.detach(), dim=-1)
            @ F.normalize(task_keys[:learned_id + 1].detach(), dim=-1).T).argmax(-1)


def adaptation_forward(model, images, reference, learned_id, args, train=False, epoch=None):
    cls = reference['pre_logits'].detach()
    raw_patches = reference['x'][:, model.num_prefix_tokens:].detach()
    kwargs = dict(learned_id=learned_id, cls_features=cls, raw_patches=raw_patches,
                  train=train, epoch_info=epoch)
    if train:
        return model(images, task_id=learned_id, **kwargs)
    if args.eval_router == 'legacy_batch':
        return model(images, **kwargs)
    routes = predict_routes(cls, model.rainbow_prompt.base_key, learned_id)
    logits = None
    diagnostics = {}
    for route in routes.unique(sorted=True).tolist():
        indices = (routes == route).nonzero(as_tuple=True)[0]
        output = model(images[indices], learned_id=learned_id, cls_features=cls[indices],
                       raw_patches=raw_patches[indices], forced_task_id=route)
        if logits is None:
            logits = output['logits'].new_empty((len(images), output['logits'].shape[-1]))
        logits[indices] = output['logits']
        for name, value in output.get('diagnostics', {}).items():
            diagnostics[name] = diagnostics.get(name, 0.) + value * len(indices) / len(images)
    return {'logits': logits, 'routes': routes, 'diagnostics': diagnostics}


def mask_seen(logits, class_mask, learned_id):
    seen = sorted({int(c) for task in class_mask[:learned_id + 1] for c in task})
    future = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
    future[seen] = False
    return logits.masked_fill(future.unsqueeze(0), float('-inf'))
