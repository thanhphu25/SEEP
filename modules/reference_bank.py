"""Append-only class means of normalized frozen raw CLS (never a classifier)."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ReferenceBank(nn.Module):
    def __init__(self, capacity, dim):
        super().__init__()
        self.register_buffer('vectors', torch.zeros(capacity, dim))
        self.register_buffer('class_ids', torch.full((capacity,), -1, dtype=torch.long))
        self.register_buffer('counts', torch.zeros(capacity, dtype=torch.long))

    @property
    def valid(self):
        return self.counts > 0

    @property
    def memory_bytes(self):
        # Report allocated storage, including empty capacity and mapping/counts.
        return sum(t.numel() * t.element_size() for t in self.buffers())

    @torch.no_grad()
    def append(self, features, labels, new_classes):
        allowed = set(map(int, new_classes))
        existing = set(self.class_ids[self.valid].tolist())
        if allowed & existing:
            raise ValueError('Cannot update old prototypes')
        if set(labels.tolist()) != allowed:
            raise ValueError('Prototype pass must contain exactly new-class training samples')
        if len(allowed) > int((~self.valid).sum()):
            raise ValueError('Bank capacity exceeded')
        features = F.normalize(features.detach().to(self.vectors), dim=-1)
        for cls in sorted(allowed):
            slot = int((~self.valid).nonzero()[0])
            selected = features[labels.to(features.device) == cls]
            self.vectors[slot].copy_(F.normalize(selected.mean(0), dim=0))
            self.class_ids[slot] = cls
            self.counts[slot] = len(selected)

    @torch.no_grad()
    def describe(self, raw_cls, topk=4, temperature=0.1):
        if topk < 1 or temperature <= 0 or not self.valid.any():
            raise ValueError('Need a nonempty bank, topk>=1, temperature>0')
        vectors = self.vectors[self.valid].to(raw_cls)
        ids = self.class_ids[self.valid]
        scores = F.normalize(raw_cls.detach(), dim=-1) @ vectors.T
        k = min(topk, vectors.shape[0])
        values, indices = scores.topk(k, dim=-1)
        first = vectors[indices[:, 0]]
        competitor = torch.zeros_like(first)
        gap = torch.zeros_like(values[:, 0])
        entropy = torch.zeros_like(gap)
        if k > 1:
            competitor = (values[:, 1:].div(temperature).softmax(-1).unsqueeze(-1)
                          * vectors[indices[:, 1:]]).sum(1)
            gap = values[:, 0] - values[:, 1]
            probs = values.div(temperature).softmax(-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1) / math.log(k)
        context = torch.cat([first, competitor, first - competitor, first * competitor], -1)
        scalars = torch.stack([gap, entropy, values[:, 0], torch.full_like(gap, k / topk)], -1)
        return context, scalars, ids[indices]


@torch.no_grad()
def build_prototypes(bank, encoder, loader, new_classes, device):
    """One deterministic pass over the exact task training subset; no debug truncation."""
    encoder.eval()
    features, labels = [], []
    for images, target in loader:
        if images.ndim != 4:
            raise ValueError('CCSP prototype pass currently supports image datasets only')
        features.append(encoder(images.to(device))['pre_logits'].detach())
        labels.append(target.to(device))
    if not features:
        raise ValueError('Empty prototype training subset')
    bank.append(torch.cat(features), torch.cat(labels), new_classes)
