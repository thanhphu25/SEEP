import argparse
import sys
import os
import uuid
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(1)


@pytest.fixture
def tmp_path(request, tmp_path_factory):
    if os.name != 'nt':
        return tmp_path_factory.mktemp(request.node.name)
    # Restricted Windows workspaces reject pytest's mode=0700 directory ACL.
    # Default inherited ACLs work; retain these artifacts for local inspection.
    path = Path(request.config.rootpath) / '.n1-test-runs' / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def cub_args(tasks=10, **overrides):
    from importlib import import_module
    parser = argparse.ArgumentParser()
    import_module(f'configs.cub200_{tasks}task').get_args_parser(parser)
    args = parser.parse_args([])
    args.subparser_name = f'cub200_{tasks}task'
    args.distributed, args.rank, args.gpu = False, 0, 0
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def tiny_model(args, prompted=True, model_class=None):
    from vision_transformer import VisionTransformer
    cls = model_class or VisionTransformer
    kwargs = dict(img_size=8, patch_size=4, num_classes=50, embed_dim=12,
                  depth=1, num_heads=3, args=args if prompted else None)
    if prompted:
        kwargs.update(use_e_prompt=True, use_prefix_tune_for_e_prompt=True,
                      e_prompt_layer_idx=[0], prompt_length=4, prompt_pool=True,
                      prompt_key=True, pool_size=10, top_k=1, batchwise_prompt=True,
                      n_tasks=10, D1=6, D2=6, use_linear=True, relation_type='attention',
                      warm_up=0, KI_iter=1, self_attn_idx=[0])
    model = cls(**kwargs)
    for name, param in model.named_parameters():
        if not prompted or name.startswith(tuple(args.freeze)):
            param.requires_grad_(False)
    return model
