from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from modules.arguments import parse_args
from modules.spatial_gate import attach_research_modules
from vision_transformer import VisionTransformer


torch.set_num_threads(1)


@pytest.fixture
def arguments():
    args = parse_args(['cub200_10task', '--epochs', '1', '--warm_up', '0', '--warmup-epochs', '0',
                       '--pretrained', 'False', '--num_workers', '0', '--task_inc', 'False'])
    args.nb_classes = 6
    args.num_tasks = 3
    args.num_tasks_to_run = 2
    args.input_size = 16
    args.gate_width = 4
    args.seed = 89359
    args.eval_router = 'per_image'
    args.eval_scope = 'seen'
    args.batch_size = 2
    args.print_freq = 100
    args.requested_lr = .03
    args.lr = .03 * 2 / 256
    args.class_order = [[4, 1], [5, 0], [2, 3]]
    args.split_ids = {'train': ['a', 'b'], 'dev': []}
    args.source_metadata = {'base_commit': 'toy', 'source_commit': 'toy'}
    args.requested_config = {}
    args.distributed = False
    return args


def make_model(args, cls=VisionTransformer):
    model = cls(img_size=16, patch_size=8, embed_dim=12, depth=2, num_heads=3,
                num_classes=6, args=args, use_e_prompt=True, e_prompt_layer_idx=[0, 1],
                use_prefix_tune_for_e_prompt=True, prompt_length=4, pool_size=3, top_k=1,
                n_tasks=3, use_linear=True, D1=4, D2=4, relation_type='attention',
                self_attn_idx=[0], KI_iter=1, warm_up=0, head_type='token',
                g_prompt_layer_idx=[], prompt_pool=True)
    for name, parameter in model.named_parameters():
        if name.startswith(('blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed')):
            parameter.requires_grad_(False)
    attach_research_modules(model, args)
    return model


def make_reference():
    model = VisionTransformer(img_size=16, patch_size=8, embed_dim=12, depth=2,
                              num_heads=3, num_classes=6, head_type='token')
    model.eval()
    model.requires_grad_(False)
    return model
