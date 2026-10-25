"""Fail closed if either routing or prompted ViT silently kept random weights."""
import hashlib
from pathlib import Path

import numpy as np
import torch

EXPECTED = 'ac7478131e1b77e5c2f0225849730bd19ee89cfffe600c2dfd2272b34a50a43d'


def verify_pretrained_models(*models):
    path = Path(torch.hub.get_dir())/'checkpoints/ViT-B_16.npz'
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != EXPECTED:
        raise ValueError('CLS screen requires the audited pretrained NPZ SHA256')
    with np.load(path) as weights:
        prefix = 'opt/target/' if 'opt/target/embedding/kernel' in weights else ''
        embedding = torch.from_numpy(weights[prefix+'embedding/kernel'].transpose(3, 2, 0, 1))
        for model in models:
            torch.testing.assert_close(model.patch_embed.proj.weight.detach().cpu(), embedding, rtol=0, atol=0)
            for layer in (0, 9, 10, 11):
                key = prefix+f'Transformer/encoderblock_{layer}/MultiHeadDotProductAttention_1/'
                qkv = torch.cat([torch.from_numpy(weights[key+n+'/kernel']).flatten(1).T
                                 for n in ('query', 'key', 'value')])
                torch.testing.assert_close(model.blocks[layer].attn.qkv.weight.detach().cpu(), qkv, rtol=0, atol=0)
    return {'sha256': EXPECTED, 'model_count': len(models), 'tensor_load_verified': True,
            'tensors': ['patch_embed.proj.weight', 'blocks.{0,9,10,11}.attn.qkv.weight']}
