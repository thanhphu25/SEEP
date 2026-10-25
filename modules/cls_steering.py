"""Task-isolated CLS Q/K/V additive prompts, inspired by CASP section 4.1.

Implemented from the supplied equations; no CASP source code is copied.
https://arxiv.org/abs/2601.16773 (Huang, Lin and Lu, 2026).
"""
import math

import torch
from torch import nn

from modules.dgca import state_hash


class CLSSteering(nn.Module):
    def __init__(self, num_tasks, layers, dim, depth):
        super().__init__()
        self.layers = tuple(int(i) for i in layers)
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError('Steering layers must be nonempty and unique')
        if min(self.layers) < 0 or max(self.layers) >= depth or num_tasks < 1:
            raise ValueError('Steering task/layer outside model bounds')
        # zeros consume no random numbers. One leaf per task; views are never
        # registered with another module, so Adam cannot update inactive slices.
        self.banks = nn.ParameterList([
            nn.Parameter(torch.zeros(len(self.layers), 3, dim), requires_grad=False)
            for _ in range(num_tasks)])
        self.register_buffer('active_task', torch.tensor(-1, dtype=torch.long))
        self._gradient_hook = None
        self._gradient_samples = []

    def activate(self, task):
        if not 0 <= task < len(self.banks):
            raise ValueError('Invalid active steering task')
        if self._gradient_hook is not None:
            self._gradient_hook.remove()
        for i, bank in enumerate(self.banks):
            bank.requires_grad_(i == task)
            bank.grad = None
        self.active_task.fill_(task)
        self._gradient_samples = []
        self._gradient_hook = self.banks[task].register_hook(self._record_gradient)

    def _record_gradient(self, gradient):
        self._gradient_samples.append(gradient.detach().float().norm(dim=-1).cpu())

    def delta(self, task, layer):
        if getattr(self, '_diagnostic_disable', False) or layer not in self.layers or task < 0:
            return None
        if task >= len(self.banks):
            raise ValueError('Router selected an unavailable steering bank')
        return self.banks[task][self.layers.index(layer)]

    def hashes(self, stop=None):
        return {str(t): state_hash({'delta': p}) for t, p in enumerate(self.banks)
                if stop is None or t < stop}

    def diagnostics(self, task):
        norms = self.banks[task].detach().float().norm(dim=-1).cpu()
        gradients = torch.stack(self._gradient_samples) if self._gradient_samples else None
        result = {}
        for i, layer in enumerate(self.layers):
            result[str(layer)] = {
                name: {'delta_norm': float(norms[i, j]),
                       'gradient_norm_mean': float(gradients[:, i, j].mean()) if gradients is not None else 0.,
                       'gradient_norm_max': float(gradients[:, i, j].max()) if gradients is not None else 0.}
                for j, name in enumerate(('dq', 'dk', 'dv'))}
        return {'task': task, 'layers': result, 'backward_calls': len(self._gradient_samples),
                'module_learned': bool(norms.sum() > 0 and gradients is not None and gradients.sum() > 0),
                'gradient_norm_stage': 'before baseline gradient clipping',
                'all_finite': all(math.isfinite(v) for d in result.values() for e in d.values() for v in e.values())}


def optimizer_audit(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    seen, groups = set(), []
    for index, group in enumerate(optimizer.param_groups):
        members = []
        for p in group['params']:
            if id(p) in seen or not p.requires_grad:
                raise RuntimeError('Duplicate or frozen tensor in task optimizer')
            seen.add(id(p))
            members.append(names[id(p)])
        groups.append({'name': group.get('name', f'baseline_{index}'), 'lr': group['lr'],
                       'lr_before_scheduler': group.get('initial_lr', group['lr']),
                       'count': sum(p.numel() for p in group['params']), 'parameters': members})
    for n, p in model.named_parameters():
        if n.startswith(('blocks.', 'patch_embed.', 'cls_token', 'pos_embed', 'norm.')) and p.requires_grad:
            raise RuntimeError('Frozen backbone was unfrozen: '+n)
    return {'groups': groups, 'requires_grad': {n: p.requires_grad for n, p in model.named_parameters()}}
