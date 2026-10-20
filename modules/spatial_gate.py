import math
import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def image_descriptor(raw_cls, raw_patches):
    q = F.normalize(raw_cls.detach(), dim=-1)
    patches = raw_patches.detach()
    mean = F.normalize(patches.mean(1), dim=-1)
    maximum = F.normalize(patches.amax(1), dim=-1)
    std = F.normalize(patches.std(1, unbiased=False), dim=-1)
    cosine = (F.normalize(patches, dim=-1) * q[:, None]).sum(-1)
    context = torch.cat([q, mean, maximum, std], -1)
    scalars = torch.stack([cosine.mean(1), cosine.std(1, unbiased=False),
                           cosine.amax(1), (q * mean).sum(-1)], -1)
    return context, scalars


class SpatialGate(nn.Module):
    def __init__(self, dim, width=64, init=0.9):
        super().__init__()
        self.patch = nn.Sequential(nn.Linear(dim, width), nn.LayerNorm(width), nn.GELU())
        self.context = nn.Sequential(nn.Linear(4 * dim, width), nn.GELU())
        self.spatial = nn.Sequential(nn.Linear(2 * width + 4, width), nn.GELU(), nn.Linear(width, 1))
        self.image = nn.Sequential(nn.Linear(dim + width + 4, width), nn.GELU(), nn.Linear(width, 1))
        # Identical policy for B3/A0: exact initial branch sigmoid(init), with learning
        # into hidden projections beginning after the first output-head update.
        for head in (self.spatial[-1], self.image[-1]):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, math.log(init / (1 - init)))

    def forward(self, raw_cls, raw_patches, context, scalars, grid, image_size):
        if raw_patches.shape[1] != grid[0] * grid[1]:
            raise ValueError('Raw patch count does not match backbone grid')
        cls, patches = raw_cls.detach(), raw_patches.detach()
        projected = self.context(context.detach())
        conditioning = torch.cat([projected, scalars.detach()], -1)
        patch_features = self.patch(patches)
        joined = torch.cat([patch_features, conditioning[:, None].expand(-1, patches.shape[1], -1)], -1)
        spatial = self.spatial(joined).sigmoid().transpose(1, 2).reshape(-1, 1, *grid)
        amplitude = self.image(torch.cat([cls, conditioning], -1)).sigmoid().reshape(-1, 1, 1, 1)
        mask = amplitude * F.interpolate(spatial, size=image_size, mode='bilinear', align_corners=False)
        return mask, amplitude, spatial


def tensor_stats(name, tensor):
    t = tensor.detach()
    return {name + '_mean': t.mean().item(), name + '_std': t.std(unbiased=False).item(),
            name + '_low': (t < .05).float().mean().item(),
            name + '_high': (t > .95).float().mean().item()}


def attach_research_modules(model, args):
    if args.method == 'seep':
        return
    # fork_rng restores CPU initialization stream; module built on CPU then moved.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 7919)
        model.spatial_gate = SpatialGate(model.embed_dim, args.gate_width, args.gate_init)
    if args.method == 'ccsp':
        from modules.reference_bank import ReferenceBank
        model.reference_bank = ReferenceBank(args.nb_classes, model.embed_dim)
