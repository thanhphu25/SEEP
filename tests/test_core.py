import copy
import inspect
import subprocess
import types
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from modules.reference_bank import ReferenceBank, build_prototypes
from modules.routing import adaptation_forward, mask_seen, predict_routes
from modules.spatial_gate import SpatialGate, image_descriptor
from modules.splits import stratified_indices, prototype_loader
from modules.metrics import continual_metrics
from modules.checkpoint import rng_state, restore_rng, load_trusted
from engine import _prepare_dynamic_buffers_for_checkpoint, task_optimizer
from conftest import make_model, make_reference


def test_bank_append_mapping_edges_and_roundtrip():
    bank = ReferenceBank(5, 3)
    features = torch.tensor([[2., 0, 0], [1., 1, 0], [0., 0, 4.]])
    bank.append(features, torch.tensor([19, 19, 2]), [19, 2])
    assert bank.class_ids[bank.valid].tolist() == [2, 19]
    assert bank.counts[bank.valid].tolist() == [1, 2]
    expected = F.normalize(F.normalize(features[:2], dim=-1).mean(0), dim=0)
    torch.testing.assert_close(bank.vectors[1], expected)
    clone = ReferenceBank(5, 3).double()
    clone.load_state_dict(bank.state_dict(), strict=True)
    context, scalars, ids = clone.describe(torch.ones(1, 3, dtype=torch.double), 9, .1)
    assert ids.shape == (1, 2) and context.dtype == torch.double
    assert scalars[0, 3] == 2 / 9
    assert bank.memory_bytes == 5 * (3 * 4 + 8 + 8)
    with pytest.raises(ValueError, match='old'):
        bank.append(features[:1], torch.tensor([19]), [19])
    with pytest.raises(ValueError, match='exactly'):
        bank.append(features[:1], torch.tensor([17]), [3])
    single = ReferenceBank(1, 3)
    single.append(features[:1], torch.tensor([99]), [99])
    context, scalars, ids = single.describe(torch.zeros(1, 3), 4)
    assert torch.isfinite(context).all() and torch.isfinite(scalars).all()
    assert (context[:, 3:6] == 0).all()
    assert (scalars[:, :2] == 0).all() and ids.item() == 99


def test_descriptor_exact_and_no_label_interface():
    bank = ReferenceBank(3, 3)
    bank.append(torch.eye(3), torch.tensor([8, 1, 20]), [8, 1, 20])
    raw = torch.tensor([[1., 0, 0]], requires_grad=True)
    context, scalars, ids = bank.describe(raw, 2, .1)
    assert ids[0, 0].item() == 8
    assert scalars[0, 0].item() == 1.  # raw cosine gap, not temperature-scaled
    assert not context.requires_grad and not scalars.requires_grad
    assert set(inspect.signature(bank.describe).parameters) == {'raw_cls', 'topk', 'temperature'}
    assert set(inspect.signature(SpatialGate.forward).parameters) == {
        'self', 'raw_cls', 'raw_patches', 'context', 'scalars', 'grid', 'image_size'}
    logits = mask_seen(torch.zeros(1, 22), [[8, 1], [20]], 1)
    assert torch.isfinite(logits).nonzero()[:, 1].tolist() == [1, 8, 20]


def test_image_descriptor_zero_and_parameter_parity(arguments):
    context, scalars = image_descriptor(torch.zeros(1, 12), torch.zeros(1, 1, 12))
    assert torch.isfinite(context).all() and torch.isfinite(scalars).all()
    arguments.method = 'image_gate'
    torch.manual_seed(3)
    before = torch.get_rng_state()
    model = make_model(arguments)
    state = copy.deepcopy(model.state_dict())
    arguments.method = 'ccsp'
    torch.set_rng_state(before)
    control = make_model(arguments)
    for name, value in state.items():
        torch.testing.assert_close(value, control.state_dict()[name], rtol=0, atol=0)
    gate = control.spatial_gate
    mask, a, spatial = gate(torch.randn(2, 12), torch.randn(2, 6, 12),
                           torch.randn(2, 48), torch.randn(2, 4), (2, 3), (16, 24))
    torch.testing.assert_close(a, torch.full_like(a, .9))
    torch.testing.assert_close(spatial, torch.full_like(spatial, .9))
    torch.testing.assert_close(mask, torch.full_like(mask, .81))


@pytest.mark.parametrize('method', ['seep', 'image_gate', 'ccsp'])
def test_grouped_routes_match_single_reordered_and_token_local_tasks(arguments, method):
    arguments.method = method
    model = make_model(arguments).eval()
    model.ensure_prompt_generator(1)
    model.register_buffer('test_dummy', torch.tensor(0))
    # Real caches differ strongly for each task and every prompted layer.
    del model.stored_unified_prompts
    model.register_buffer('stored_unified_prompts', torch.randn(3, 3, 16, 16))
    model.rainbow_prompt.stored_rainbow_prompts.copy_(torch.randn(3, 2, 4, 12))
    model.rainbow_prompt.base_key.data[:2] = torch.eye(12)[:2]
    raw = torch.eye(12)[torch.tensor([1, 0, 1, 0])]
    patches = torch.randn(4, 4, 12)
    images = torch.randn(4, 3, 16, 16)
    reference = {'pre_logits': raw, 'x': torch.cat([raw[:, None], patches], 1)}
    if method == 'ccsp':
        model.reference_bank.append(torch.randn(4, 12), torch.tensor([4, 1, 5, 0]), [4, 1, 5, 0])
    observed = []
    def record(module, inputs):
        observed.append((model.matched_task_id, model.rainbow_prompt.task_id))
    hook = model.blocks[0].register_forward_pre_hook(record)
    with torch.no_grad():
        grouped = adaptation_forward(model, images, reference, 1, arguments)
        assert grouped['routes'].tolist() == [1, 0, 1, 0]
        assert observed == [(0, 0), (1, 1)]
        single = torch.cat([adaptation_forward(model, images[i:i+1],
            {k: v[i:i+1] for k, v in reference.items()}, 1, arguments)['logits'] for i in range(4)])
        order = torch.tensor([3, 0, 2, 1])
        reordered = adaptation_forward(model, images[order], {k: v[order] for k, v in reference.items()},
                                       1, arguments)['logits']
    hook.remove()
    torch.testing.assert_close(grouped['logits'], single, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(grouped['logits'][order], reordered, atol=2e-6, rtol=1e-5)
    seen = mask_seen(grouped['logits'], arguments.class_order, 1)
    assert torch.isfinite(seen).all(0).nonzero().flatten().tolist() == [0, 1, 4, 5]


@pytest.mark.parametrize('method', ['image_gate', 'ccsp'])
def test_gradient_cache_identity_and_zero(arguments, method):
    arguments.method = method
    model = make_model(arguments)
    reference_model = make_reference()
    images = torch.rand(2, 3, 16, 16)
    with torch.no_grad():
        reference = reference_model(images)
    reference['pre_logits'].requires_grad_(True)
    reference['x'].requires_grad_(True)
    model.ensure_prompt_generator(1)
    if method == 'ccsp':
        model.reference_bank.append(torch.randn(4, 12), torch.tensor([4, 1, 5, 0]), [4, 1, 5, 0])
    old = copy.deepcopy(model.prompt_generators[0].state_dict())
    state = rng_state()
    output = adaptation_forward(model, images, reference, 1, arguments, train=True, epoch=0)
    ungated_cache = model.stored_unified_prompts.clone()
    loss = output['logits'].square().mean() - .01 * output['sim_loss']
    loss.backward()
    for component in (model.spatial_gate, model.prompt_generators[1]):
        grads = [p.grad for p in component.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0
    assert all(p.grad is None for p in model.prompt_generators[0].parameters())
    assert all(p.grad is None for p in reference_model.parameters())
    assert reference['pre_logits'].grad is None and reference['x'].grad is None
    arguments.gate_mode = 'identity'
    restore_rng(state)
    identity = adaptation_forward(model, images, reference, 1, arguments, train=True, epoch=0)['logits']
    torch.testing.assert_close(ungated_cache, model.stored_unified_prompts, rtol=0, atol=0)
    arguments.method = 'seep'
    restore_rng(state)
    baseline = adaptation_forward(model, images, reference, 1, arguments, train=True, epoch=0)['logits']
    torch.testing.assert_close(identity, baseline, rtol=0, atol=0)
    arguments.gate_mode = 'zero'
    restore_rng(state)
    zero = adaptation_forward(model, images, reference, 1, arguments, train=True, epoch=0)['logits']
    assert not torch.equal(zero, baseline)
    torch.testing.assert_close(ungated_cache, model.stored_unified_prompts, rtol=0, atol=0)
    for key, value in old.items():
        torch.testing.assert_close(value, model.prompt_generators[0].state_dict()[key], rtol=0, atol=0)


def test_actual_legacy_source_train_and_eval_logits(arguments, monkeypatch):
    # Compare to the audited source, not a hand-written imitation of baseline.
    source = (Path(__file__).parent / 'fixtures/audited_vision_transformer.py.txt').read_text(encoding='utf-8')
    baseline_module = types.ModuleType('audited_baseline')
    monkeypatch.setitem(sys.modules, baseline_module.__name__, baseline_module)
    import timm.models.registry as registry
    with monkeypatch.context() as temporary:
        temporary.setattr(registry, 'register_model', lambda function: function)
        exec(compile(source, 'audited_baseline.py', 'exec'), baseline_module.__dict__)
    arguments.method = 'seep'
    arguments.eval_router = 'legacy_batch'
    torch.manual_seed(17)
    baseline = make_model(arguments, baseline_module.VisionTransformer)
    torch.manual_seed(17)
    current = make_model(arguments)
    images, raw = torch.rand(2, 3, 16, 16), torch.rand(2, 12)
    for task in range(2):
        baseline.ensure_prompt_generator(task)
        current.ensure_prompt_generator(task)
        current.prompt_generators[task].load_state_dict(baseline.prompt_generators[task].state_dict())
        state = rng_state()
        expected = baseline(images, task_id=task, learned_id=task, cls_features=raw, train=True, epoch_info=0)
        restore_rng(state)
        actual = current(images, task_id=task, learned_id=task, cls_features=raw, train=True, epoch_info=0)
        torch.testing.assert_close(actual['logits'], expected['logits'], rtol=0, atol=0)
    baseline.eval()
    current.eval()
    torch.testing.assert_close(current(images, learned_id=1, cls_features=raw)['logits'],
                               baseline(images, learned_id=1, cls_features=raw)['logits'], rtol=0, atol=0)


@pytest.mark.parametrize('method', ['seep', 'image_gate', 'ccsp'])
def test_dynamic_strict_load_and_lr_groups(arguments, method):
    arguments.method = method
    model = make_model(arguments)
    model.ensure_prompt_generator(1)
    images = torch.rand(2, 3, 16, 16)
    reference = make_reference()(images)
    if method == 'ccsp':
        model.reference_bank.append(reference['pre_logits'], torch.tensor([4, 1]), [4, 1])
    adaptation_forward(model, images, reference, 1, arguments, train=True, epoch=0)
    state = copy.deepcopy(model.state_dict())
    fresh = make_model(arguments)
    rng = torch.get_rng_state()
    _prepare_dynamic_buffers_for_checkpoint(fresh, state)
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
    fresh.load_state_dict(state, strict=True)
    for key, value in state.items():
        torch.testing.assert_close(value, fresh.state_dict()[key], rtol=0, atol=0)
    optimizer, scheduler = task_optimizer(fresh, arguments)
    groups = {g['name']: g for g in optimizer.param_groups}
    assert groups['general']['lr'] == arguments.lr
    assert groups['local']['lr'] == .0005
    assert groups['frequency_weights']['lr'] == .03
    if method != 'seep':
        assert groups['gate']['lr'] == .0005 and groups['gate']['weight_decay'] == 0
    scale = fresh.frequency_residual_scale
    assert any(p is scale for p in groups['general']['params'])


def test_metrics_known_matrix_and_incomplete_history():
    matrix = np.array([[90., np.nan, np.nan], [80., 70., np.nan], [60., 65., 85.]])
    result = continual_metrics(matrix, 2, [2, 1, 1])
    assert result['all_seen_accuracy'] == 67.5
    assert result['task_mean_accuracy'] == 70.
    assert result['forgetting'] == 17.5
    assert result['old_accuracy'] == pytest.approx(61.6666667)
    assert result['old_errors'] == pytest.approx(1.15)
    assert result['new_errors'] == pytest.approx(.15)
    assert continual_metrics(matrix, 0, [2])['forgetting'] is None
    matrix[0, 0] = np.nan
    result = continual_metrics(matrix, 2, [2, 1, 1])
    assert result['AvgIncAcc'] is None and result['forgetting'] is None


def test_gate_hidden_projections_receive_gradient_after_output_step():
    gate = SpatialGate(12, 4)
    raw = torch.randn(2, 12, requires_grad=True)
    patches = torch.randn(2, 4, 12, requires_grad=True)
    context, scalars = image_descriptor(raw, patches)
    optimizer = torch.optim.SGD(gate.parameters(), lr=.1)
    for _ in range(2):
        optimizer.zero_grad()
        mask, _, _ = gate(raw, patches, context, scalars, (2, 2), (16, 16))
        (mask * torch.arange(16).float()[None, None, None, :]).mean().backward()
        optimizer.step()
    assert gate.patch[0].weight.grad.abs().sum() > 0
    assert gate.context[0].weight.grad.abs().sum() > 0
    assert raw.grad is None and patches.grad is None
