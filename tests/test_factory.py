import pytest
import torch
from timm.models import create_model
from modules.spatial_gate import attach_research_modules
from modules.routing import adaptation_forward
import models  # Register the actual entry points used by main.py.


@pytest.mark.parametrize('method', ['seep', 'image_gate', 'ccsp'])
def test_real_timm_factory_one_reference_forward_224(arguments, method):
    arguments.method = method
    reference = create_model('vit_tiny_patch16_224', pretrained=False, num_classes=6)
    reference.requires_grad_(False).eval()
    model = create_model('vit_tiny_patch16_224', pretrained=False, num_classes=6,
        prompt_length=4, pool_size=3, top_k=1, use_e_prompt=True,
        e_prompt_layer_idx=list(range(12)), use_prefix_tune_for_e_prompt=True,
        g_prompt_layer_idx=[], n_tasks=3, D1=4, D2=4, use_linear=True,
        relation_type='attention', KI_iter=1, self_attn_idx=[0], warm_up=0,
        prompt_pool=True, args=arguments)
    attach_research_modules(model, arguments)
    calls = []
    hook = reference.register_forward_hook(lambda *_: calls.append(1))
    images = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        raw = reference(images)
        assert raw['x'].shape == (2, 197, 192)
        if method == 'ccsp':
            model.reference_bank.append(raw['pre_logits'], torch.tensor([4, 1]), [4, 1])
        adaptation_forward(model, images, raw, 0, arguments, train=True, epoch=0)
        model.eval()
        logits = adaptation_forward(model, images, raw, 0, arguments)['logits']
    hook.remove()
    assert len(calls) == 1
    assert logits.shape == (2, 6) and torch.isfinite(logits).all()
