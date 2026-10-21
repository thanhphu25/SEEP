import ast
import copy
import subprocess

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, Subset

from conftest import cub_args, tiny_model
from spectral_response_memory import (SpectralResponseMemory, coordinates, global_response,
                                     frequency_transform, jacobian_rows, normalize_gram,
                                     capture_rng, restore_rng, ProbeView)


def baseline_class():
    import vision_transformer as vit
    source = subprocess.check_output(['git', 'show', '8008108:vision_transformer.py'], text=True)
    original = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'VisionTransformer')
    namespace = vars(vit).copy()
    exec(compile(ast.Module(body=[original], type_ignores=[]), '<audited baseline>', 'exec'), namespace)
    return namespace['VisionTransformer']


@pytest.mark.parametrize('seed', [11, 37, 91])
@pytest.mark.parametrize('num_r', [5, 100])
def test_off_fft_logits_rng_and_source_equation(seed, num_r):
    torch.manual_seed(seed)
    args = cub_args(num_r=num_r, temperature=1.7, input_size=8)
    model = tiny_model(args)
    original = tiny_model(args, model_class=baseline_class())
    original.load_state_dict(model.state_dict())
    # Nondegenerate random coordinates, including unconstrained negative alpha/beta.
    with torch.no_grad():
        model.weights.normal_(0, .5)
        model.alpha.fill_(-.3 + .01 * seed)
        model.beta.fill_(.6)
        model.frequency_residual_scale.fill_(.4)
    original.load_state_dict(model.state_dict())
    x = torch.randn(2, 3, 8, 8)
    state = capture_rng()
    # Off initialization and boundary bookkeeping must consume no RNG or forwards.
    off = SpectralResponseMemory(args, 12, 'cpu')
    off.collect(model, None, None, None, None, 0)
    assert torch.equal(state['torch'], torch.get_rng_state())
    torch.testing.assert_close(model.get_Frequency_mask(x), original.get_Frequency_mask(x), rtol=0, atol=0)
    local = torch.randn_like(x) * .2
    expected = original.alpha * (x + local) + original.beta * original.get_Frequency_mask(x)
    actual = global_response(model, x) + model.alpha * local
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    actual.square().mean().backward()
    for param in (model.weights, model.alpha, model.beta, model.frequency_residual_scale):
        assert torch.isfinite(param.grad).all() and param.grad.norm() > 0
    features = torch.randn(2, 12)
    state = capture_rng()
    out = model(x, task_id=0, learned_id=0, cls_features=features, train=True, epoch_info=0)
    after = capture_rng()
    restore_rng(state)
    baseline = original(x, task_id=0, learned_id=0, cls_features=features, train=True, epoch_info=0)
    assert torch.equal(after['torch'], torch.get_rng_state())
    torch.testing.assert_close(out['logits'], baseline['logits'], rtol=0, atol=0)
    torch.testing.assert_close(model.stored_unified_prompts, original.stored_unified_prompts)
    for m in (model, original):
        m.eval()
    out = model(x, task_id=0, learned_id=0, cls_features=features)['logits']
    baseline = original(x, task_id=0, learned_id=0, cls_features=features)['logits']
    torch.testing.assert_close(out, baseline, rtol=0, atol=0)


def test_jacobian_finite_difference_and_per_sample_gram():
    torch.manual_seed(3)
    args = cub_args(num_r=3, temperature=.7)
    model = tiny_model(args)
    model.weights.data.normal_(0, .3)
    q = coordinates(model).detach().requires_grad_(True)
    samples = torch.randn(2, 3, 8, 8)
    # Nonlinear synthetic feature map, so normalization and cross-sample terms matter.
    projection = torch.randn(4, 3 * 8 * 8)
    def probe(q, x):
        z = global_response(model, x.unsqueeze(0), q).flatten()
        features = torch.tanh(projection @ z)
        return torch.nn.functional.normalize(features, dim=0)
    J = jacobian_rows(lambda qq: probe(qq, samples[0]), q)
    finite = []
    step = 1e-3
    for i in range(q.numel()):
        offset = torch.zeros_like(q)
        offset[i] = step
        finite.append((probe(q + offset, samples[0]) - probe(q - offset, samples[0])) / (2 * step))
    torch.testing.assert_close(J, torch.stack(finite, dim=1), rtol=.05, atol=.003)
    other = jacobian_rows(lambda qq: probe(qq, samples[1]), q)
    mean_gram = (J.T @ J + other.T @ other) / 2
    wrong = ((J + other) / 2).T @ ((J + other) / 2)
    assert not torch.allclose(mean_gram, wrong, atol=1e-6)
    gram = normalize_gram(mean_gram, 'trace')
    torch.testing.assert_close(gram, gram.T)
    assert torch.linalg.eigvalsh(gram).min() > -1e-6
    assert gram.trace().item() == pytest.approx(1)
    assert normalize_gram(torch.eye(3) * 1e-12, 'trace').count_nonzero() == 0
    with pytest.raises(ValueError):
        normalize_gram(-torch.eye(3), 'trace')
    with pytest.raises(ValueError):
        normalize_gram(torch.full((3, 3), float('nan')), 'trace')


def test_penalty_analytic_gradient_diag_full_and_roundtrip():
    torch.manual_seed(7)
    args = cub_args(srm_mode='semantic_full', num_r=3, srm_proj_dim=3)
    model = tiny_model(args)
    memory = SpectralResponseMemory(args, 12, 'cpu')
    assert memory.penalty(model).item() == 0
    q = coordinates(model).detach()
    memory.anchors = torch.stack([q.clone(), q.clone()])
    raw = torch.randn(2, 5, 5)
    memory.matrices = raw.transpose(-1, -2) @ raw
    assert memory.penalty(model).item() == 0
    model.alpha.data.add_(.3)
    current = coordinates(model)
    delta = current.detach() - q
    expected_qgrad = (2 * memory.matrices @ delta).mean(0)
    actual_grad = torch.autograd.grad(memory.penalty(model), model.alpha)[0]
    assert actual_grad.item() == pytest.approx(expected_qgrad[-2].item(), rel=1e-5)
    memory.mode = 'semantic_diag'
    expected = (delta.square() * memory.matrices.diagonal(dim1=-2, dim2=-1)).sum(-1).mean()
    torch.testing.assert_close(memory.penalty(model), expected)
    memory.mode = 'semantic_full'
    memory.class_ids, memory.counts, memory.completed_task = [0, 1], [2, 2], 0
    restored = SpectralResponseMemory(args, 12, 'cpu')
    restored.load_state_dict(memory.state_dict())
    torch.testing.assert_close(restored.R, memory.R)
    torch.testing.assert_close(restored.penalty(model), memory.penalty(model))
    broken = memory.state_dict()
    broken.pop('R')
    with pytest.raises(ValueError):
        restored.load_state_dict(broken)


class Images(Dataset):
    def __init__(self):
        self.targets = [0, 0, 1, 1, 2, 2]
        self.transform = lambda x: x + torch.rand_like(x)
        self.images = torch.linspace(.01, .9, 6 * 3 * 8 * 8).reshape(6, 3, 8, 8)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        x = self.images[index]
        return self.transform(x) if self.transform else x, self.targets[index]


@pytest.mark.parametrize('mode', ['l2', 'semantic_full', 'semantic_diag'])
def test_collector_isolation_subset_and_batch_aggregation(mode):
    torch.manual_seed(17)
    args = cub_args(srm_mode=mode, srm_proj_dim=2, num_r=3, srm_samples_per_class=2)
    model, reference = tiny_model(args), tiny_model(args, prompted=False)
    model.ensure_prompt_generator(0)
    model.register_buffer('stored_unified_prompts', torch.randn(10, 3, 8, 8)) if 'stored_unified_prompts' in model._buffers else None
    # Buffer construction follows the actual dynamic registration contract.
    if model.stored_unified_prompts is None:
        delattr(model, 'stored_unified_prompts')
        model.register_buffer('stored_unified_prompts', torch.randn(10, 3, 8, 8))
    model.train()
    model.prompt_generators[0].eval()
    reference.train()
    before = copy.deepcopy(model.state_dict())
    ref_before = copy.deepcopy(reference.state_dict())
    modes = [m.training for m in list(model.modules()) + list(reference.modules())]
    flags = [p.requires_grad for p in list(model.parameters()) + list(reference.parameters())]
    rng = capture_rng()
    data = Images()
    nested = Subset(Subset(data, [2, 3, 4, 5]), [0, 1])
    transform = data.transform
    memory = SpectralResponseMemory(args, 12, 'cpu')
    stats = memory.collect(model, reference, nested, [1], lambda x: x, 0)
    assert memory.class_ids == [1] and memory.counts == [2]
    assert set(memory.metadata[0]['selected_train_ids']) == {2, 3}
    assert data.transform is transform
    assert torch.equal(rng['torch'], torch.get_rng_state())
    after_rng = capture_rng()
    assert rng['python'] == after_rng['python']
    assert np.array_equal(rng['numpy'][1], after_rng['numpy'][1])
    assert modes == [m.training for m in list(model.modules()) + list(reference.modules())]
    assert flags == [p.requires_grad for p in list(model.parameters()) + list(reference.parameters())]
    for k, value in before.items():
        torch.testing.assert_close(value, model.state_dict()[k], rtol=0, atol=0)
    for k, value in ref_before.items():
        torch.testing.assert_close(value, reference.state_dict()[k], rtol=0, atol=0)
    assert all(p.grad is None for p in reference.parameters())
    assert stats['memory_bytes'] == memory.memory_bytes
    args.srm_probe_batch_size = 2
    batched = SpectralResponseMemory(args, 12, 'cpu')
    batched.collect(model, reference, nested, [1], lambda x: x, 0)
    torch.testing.assert_close(memory.matrices, batched.matrices, rtol=0, atol=0)
    if mode != 'l2':
        assert memory.matrices[0].trace().item() > 0
    with pytest.raises(ValueError):
        memory.collect(model, reference, nested, [1], lambda x: x, 0)


def test_freeze_spectral_core_only():
    args = cub_args(srm_mode='freeze_spectral')
    model = tiny_model(args)
    memory = SpectralResponseMemory(args, 12, 'cpu')
    memory.begin_task(model, 0)
    assert model.weights.requires_grad and model.frequency_residual_scale.requires_grad
    memory.begin_task(model, 1)
    assert not model.weights.requires_grad and not model.frequency_residual_scale.requires_grad
    assert model.alpha.requires_grad and model.beta.requires_grad


def test_full_diag_same_gram_and_l2_same_anchors():
    torch.manual_seed(71)
    args = cub_args(srm_mode='semantic_full', srm_proj_dim=2, num_r=3, srm_samples_per_class=2)
    model, reference = tiny_model(args), tiny_model(args, prompted=False)
    data = Subset(Images(), [0, 1])
    memories = []
    for mode in ('semantic_full', 'semantic_diag', 'l2'):
        args.srm_mode = mode
        memory = SpectralResponseMemory(args, 12, 'cpu')
        memory.collect(model, reference, data, [0], lambda x: x, 0)
        memories.append(memory)
    full, diag, l2 = memories
    torch.testing.assert_close(full.R, diag.R, rtol=0, atol=0)
    torch.testing.assert_close(full.matrices, diag.matrices, rtol=0, atol=0)
    torch.testing.assert_close(full.anchors, l2.anchors, rtol=0, atol=0)
    assert full.metadata[0]['selected_train_ids'] == l2.metadata[0]['selected_train_ids']
    torch.testing.assert_close(l2.matrices[0], torch.eye(5) / 5)
