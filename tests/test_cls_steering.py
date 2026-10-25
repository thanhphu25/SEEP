import argparse
import copy
import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from attention import PreT_Attention, apply_cls_delta
from modules.cls_steering import CLSSteering, optimizer_audit
from modules.cls_diagnostics import attention_diagnostic
from modules.dgca import model_hashes, capture_rng, state_hash
from modules.dgca_data import current_loader, collect_features
from modules.dgca_runtime import task_optimizer, load_boundary
from tools.summarize_cls_steering import extended_metrics, reference_matrices, export_cls_metrics
from test_dgca import image_folder


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def options(mode='task', prefix=True):
    args = SimpleNamespace(lgsp='YES' if prefix else 'NO', lgsp_type='LGSP', num_tasks=3,
                           num_r=4, prompt_hid_dim=3, Dropout_Prompt=.1, temperature=1.,
                           cls_steering=mode, cls_steering_layers=[0, 1], cls_steering_lr=.0005)
    return dict(img_size=16, patch_size=8, embed_dim=24, depth=2, num_heads=3,
                num_classes=7, fc_norm=True, prompt_length=4, prompt_pool=prefix,
                pool_size=3, top_k=1, use_e_prompt=prefix, e_prompt_layer_idx=[0, 1],
                use_prefix_tune_for_e_prompt=prefix, n_tasks=3, D1=8, D2=12,
                use_linear=True, relation_type='attention', KI_iter=1, self_attn_idx=[0, 1],
                warm_up=0, args=args)


def make_model(mode='task', prefix=True):
    from vision_transformer import VisionTransformer
    return VisionTransformer(**options(mode, prefix))


def freeze(model):
    for name, parameter in model.named_parameters():
        if name.startswith(('blocks.', 'patch_embed.', 'cls_token', 'norm.', 'pos_embed')):
            parameter.requires_grad_(False)


def config():
    from configs.cub200_10task import get_args_parser
    parser = argparse.ArgumentParser()
    get_args_parser(parser)
    args = parser.parse_args(['--cls-steering', 'task', '--cls-steering-layers', '0', '1', '--dgca', 'align_only'])
    args.lr *= 32/256
    return args


class Original(nn.Module):
    def forward(self, images):
        return {'pre_logits': images.mean((2, 3)).repeat(1, 8)}


def legacy_constructor():
    """Actual pinned pre-edit implementation, not another invocation of new code."""
    root = Path(__file__).resolve().parents[1]
    base = '80081081ab42769d102f83d447705da0639e2fb1'
    source = lambda name: subprocess.check_output(['git', 'show', f'{base}:{name}'], cwd=root).decode()
    old_attention = types.ModuleType('attention')
    exec(compile(source('attention.py'), '<pinned attention>', 'exec'), old_attention.__dict__)
    old_vit = types.ModuleType('legacy_vision_transformer')
    old_vit.__file__ = str(root/'vision_transformer.py')
    current = sys.modules['attention']
    try:
        sys.modules['attention'] = old_attention
        exec(compile(source('vision_transformer.py').replace('@register_model\n', ''), '<pinned ViT>', 'exec'), old_vit.__dict__)
    finally:
        sys.modules['attention'] = current
    return old_vit.VisionTransformer


@pytest.mark.parametrize('prefix', [False, True])
def test_off_matches_pinned_base_forward_and_training_rng(prefix):
    legacy = legacy_constructor()
    def trajectory(constructor):
        torch.manual_seed(77)
        model = constructor(**options('off', prefix))
        freeze(model)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
        trace = []
        for _ in range(2):
            x = torch.randn(3, 3, 16, 16)
            output = model(x, task_id=0, learned_id=0, cls_features=torch.ones(3, 24), train=True, epoch_info=1)
            loss = nn.functional.cross_entropy(output['logits'], torch.tensor([0, 1, 2]))
            if prefix:
                loss = loss-.1*output['sim_loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trace.append(loss.item())
        return trace, model_hashes(model), capture_rng()['torch']
    from vision_transformer import VisionTransformer
    before, after = trajectory(legacy), trajectory(VisionTransformer)
    assert before[:2] == after[:2]
    assert torch.equal(before[2], after[2])


@pytest.mark.parametrize('prefix', [False, True])
def test_zero_delta_matches_logits_features_and_consumes_no_init_rng(prefix):
    torch.manual_seed(99)
    off = make_model('off', prefix)
    rng_off = capture_rng()['torch']
    torch.manual_seed(99)
    on = make_model('task', prefix)
    assert torch.equal(rng_off, capture_rng()['torch'])
    for name, value in off.state_dict().items():
        assert torch.equal(value, on.state_dict()[name]), name
    if prefix:
        with torch.no_grad():
            off.rainbow_prompt.stored_rainbow_prompts.normal_()
            on.rainbow_prompt.stored_rainbow_prompts.copy_(off.rainbow_prompt.stored_rainbow_prompts)
    x = torch.randn(3, 3, 16, 16)
    off.eval()
    on.eval()
    cls = torch.ones(3, 24) if prefix else None
    off_output = off(x, task_id=0, learned_id=0, cls_features=cls)
    on_output = on(x, task_id=0, learned_id=0, cls_features=cls)
    for key in ('logits', 'classifier_features'):
        torch.testing.assert_close(off_output[key], on_output[key], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('prefix_length', [0, 2])
def test_nonzero_delta_real_cls_only_prefix_unchanged_manual_attention(prefix_length):
    torch.manual_seed(8)
    attn = PreT_Attention(24, num_heads=3, qkv_bias=True)
    x = torch.randn(2, 5, 24, requires_grad=True)
    delta = torch.randn(3, 24, requires_grad=True)*.1
    original = copy.deepcopy(attn.state_dict())
    qkv = attn.qkv(x).reshape(2, 5, 3, 3, 8).permute(2, 0, 3, 1, 4)
    changed = apply_cls_delta(*qkv.unbind(0), delta)
    for before, after, d in zip(qkv, changed, delta):
        assert torch.equal(before[:, :, 1:], after[:, :, 1:])
        torch.testing.assert_close(after[:, :, 0]-before[:, :, 0], d.reshape(1, 3, 8).expand(2, -1, -1))
    prompt = [torch.randn(2, prefix_length, 24) for _ in range(2)] if prefix_length else None
    q, k, v = changed
    if prompt is not None:
        pk, pv = [p.reshape(2, -1, 3, 8).permute(0, 2, 1, 3) for p in prompt]
        k, v = torch.cat((pk, k), 2), torch.cat((pv, v), 2)
        assert torch.equal(k[:, :, :prefix_length], pk)
        assert torch.equal(v[:, :, :prefix_length], pv)
    expected = attn.proj((((q @ k.transpose(-2, -1))*attn.scale).softmax(-1) @ v).transpose(1, 2).reshape(2, 5, 24))
    actual = attn(x, prompt, steering=delta)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == x.shape and torch.isfinite(actual).all()
    actual.sum().backward()
    assert x.grad is not None
    assert all(torch.equal(value, attn.state_dict()[name]) for name, value in original.items())


def test_bank_counts_optimizer_backbone_and_gradient_isolation():
    bank = CLSSteering(10, [9, 10, 11], 768, 12)
    assert bank.banks[0].numel() == 6912
    assert sum(p.numel() for p in bank.parameters()) == 69120
    model = make_model()
    freeze(model)
    args = config()
    optimizer, scheduler = task_optimizer(model, 0, args, None, None)
    assert model.cls_steering.layers == (0, 1)
    audit = optimizer_audit(model, optimizer)
    group = next(g for g in audit['groups'] if g['name'] == 'cls_steering')
    assert group['lr_before_scheduler'] == .0005 and group['count'] == 144
    assert sum(g['count'] for g in audit['groups']) == sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone = state_hash(model.blocks.state_dict())
    x = torch.randn(3, 3, 16, 16)
    for task in (0, 1):
        old = model.cls_steering.hashes(task)
        if task:
            optimizer, scheduler = task_optimizer(model, task, args, optimizer, scheduler)
        future = model.cls_steering.hashes()[str(task+1)]
        out = model(x, task_id=task, learned_id=task, cls_features=torch.ones(3, 24), train=True, epoch_info=1)
        loss = nn.functional.cross_entropy(out['logits'], torch.tensor([0, 1, 2]))-.1*out['sim_loss']
        optimizer.zero_grad()
        loss.backward()
        assert model.cls_steering.banks[task].grad.abs().sum() > 0
        assert all(p.grad is None for i, p in enumerate(model.cls_steering.banks) if i != task)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if n.startswith('rainbow_prompt.'))
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.prompt_generators[task].parameters())
        optimizer.step()
        assert model.cls_steering.hashes(task) == old
        assert model.cls_steering.hashes()[str(task+1)] == future
        assert model.cls_steering.diagnostics(task)['backward_calls'] == 1
    assert backbone == state_hash(model.blocks.state_dict())


def test_router_selects_bank_with_no_oracle_and_original_unchanged():
    model = make_model().eval()
    original = make_model('off', prefix=False).eval()
    x = torch.randn(2, 3, 16, 16)
    cls = model.rainbow_prompt.base_key[1].detach().expand(2, -1)
    with torch.no_grad():
        model.cls_steering.banks[0].fill_(.2)
        model.cls_steering.banks[1].fill_(.4)
    used = []
    handle = model.blocks[0].attn.register_forward_pre_hook(lambda module, args, kwargs: used.append(kwargs['steering'].clone()), with_kwargs=True)
    before = state_hash(original.state_dict())
    original_features = original(x)['classifier_features'].detach().clone()
    first = model(x, task_id=-1, learned_id=1, cls_features=cls)
    second = model(x, task_id=999, learned_id=1, cls_features=cls)
    handle.remove()
    assert model.matched_task_id == 1 and all(torch.equal(d, model.cls_steering.banks[1][0]) for d in used)
    assert torch.equal(first['logits'], second['logits'])
    assert before == state_hash(original.state_dict())
    assert original.cls_steering is None
    assert torch.equal(original_features, original(x)['classifier_features'])


def test_feature_collection_and_attention_restore_state_modes_rng(tmp_path):
    base = image_folder(tmp_path)
    model = make_model()
    freeze(model)
    model.cls_steering.activate(0)
    model(torch.randn(3, 3, 16, 16), task_id=0, learned_id=0,
          cls_features=torch.ones(3, 24), train=True, epoch_info=1)
    model.train()
    hashes, rng = model_hashes(model), capture_rng()
    modes = [m.training for m in model.modules()]
    route = model.matched_task_id
    trainable = [p.requires_grad for p in model.parameters()]
    probe = current_loader(base, transforms.ToTensor(), 43)
    features = collect_features(model, Original(), probe, 'cpu', 0)
    diagnostic = attention_diagnostic(model, Original(), base, transforms.ToTensor(), 0, 'cpu', 43)
    assert features['features'].shape == (12, 24)
    assert diagnostic['sample_count'] == 12 and diagnostic['hashes_unchanged']
    assert set(diagnostic['layers']) == {'0', '1'}
    assert hashes == model_hashes(model)
    assert model.matched_task_id == route and int(model.cls_steering.active_task) == 0
    assert modes == [m.training for m in model.modules()]
    assert trainable == [p.requires_grad for p in model.parameters()]
    assert torch.equal(capture_rng()['torch'], rng['torch'])
    assert capture_rng()['python'] == rng['python']
    np.testing.assert_equal(capture_rng()['numpy'], rng['numpy'])


def test_cls_runner_resume_matches_uninterrupted_and_does_not_repeat(tmp_path, monkeypatch):
    from modules.dgca_runtime import train_and_evaluate_dgca
    import modules.dgca_runtime as runtime
    import engine
    base = image_folder(tmp_path/'images')
    base.transform = transforms.ToTensor()
    data = [{'train': DataLoader(Subset(base, ids), batch_size=32, shuffle=True),
             'val': DataLoader(Subset(base, ids), batch_size=32)}
            for ids in (list(range(4)), list(range(4, 12)))]
    args = config()
    args.dgca_epochs, args.dgca_samples_per_class = 1, 4
    args.num_tasks, args.epochs, args.nb_classes, args.input_size = 2, 1, 7, 16
    args.seed, args.num_workers, args.distributed, args.device = 123, 0, False, 'cpu'
    monkeypatch.setattr(runtime, 'write_provenance', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    calls, active_before = [], []
    train = engine.train_one_epoch
    collect = runtime.collect_features
    def record(*a, **k):
        calls.append(k['task_id'])
        return train(*a, **k)
    def record_collection(model, original, loader, device, learned):
        active_before.append((learned, int(model.cls_steering.active_task)))
        return collect(model, original, loader, device, learned)
    monkeypatch.setattr(engine, 'train_one_epoch', record)
    monkeypatch.setattr(runtime, 'collect_features', record_collection)
    def launch(directory, stop=0, resume=False):
        import random
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        model = make_model()
        freeze(model)
        local = copy.deepcopy(args)
        local.output_dir = str(directory)
        local.num_tasks_to_run, local.dgca_resume = stop, resume
        initial = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=local.lr)
        dgca = train_and_evaluate_dgca(model, model, Original(), nn.CrossEntropyLoss(), data,
                                     initial, None, 'cpu', [[0], [1, 2]], local)
        return model, dgca, capture_rng()
    full, full_dgca, full_rng = launch(tmp_path/'full')
    calls.clear()
    launch(tmp_path/'resumed', stop=1)
    resumed, resumed_dgca, resumed_rng = launch(tmp_path/'resumed', resume=True)
    assert calls == [0, 1]
    assert all(learned == active for learned, active in active_before)
    assert model_hashes(full) == model_hashes(resumed)
    assert [p.requires_grad for p in full.parameters()] == [p.requires_grad for p in resumed.parameters()]
    assert int(resumed.cls_steering.active_task) == 1
    assert torch.equal(full_rng['torch'], resumed_rng['torch'])
    assert full_rng['python'] == resumed_rng['python']
    np.testing.assert_equal(full_rng['numpy'], resumed_rng['numpy'])
    torch.testing.assert_close(full_dgca.alignment_head.weight, resumed_dgca.alignment_head.weight, rtol=0, atol=0)
    for label in full_dgca.stats:
        for name in ('mean', 'covariance'):
            assert torch.equal(full_dgca.stats[label][name], resumed_dgca.stats[label][name])
    full.eval()
    resumed.eval()
    x = torch.ones(2, 3, 16, 16)
    assert torch.equal(full(x, task_id=-1, learned_id=1, cls_features=Original()(x)['pre_logits'])['logits'],
                       resumed(x, task_id=-1, learned_id=1, cls_features=Original()(x)['pre_logits'])['logits'])
    for name in ('dgca', 'raw_seen', 'raw_exact'):
        np.testing.assert_equal(np.loadtxt(tmp_path/'full'/f'task_accuracy_matrix_{name}.csv', delimiter=','),
                                np.loadtxt(tmp_path/'resumed'/f'task_accuracy_matrix_{name}.csv', delimiter=','))
    saved = (tmp_path/'resumed'/'acc.txt').read_bytes()
    calls.clear()
    launch(tmp_path/'resumed', resume=True)
    assert not calls and saved == (tmp_path/'resumed'/'acc.txt').read_bytes()
    summary = json.loads((tmp_path/'resumed'/'summary.json').read_text())
    assert not summary['full_protocol_complete'] and not summary['screening_pass']

    checkpoint = torch.load(tmp_path/'resumed/checkpoint/dgca_latest.pth', weights_only=False)
    split = runtime.split_manifest(data, [[0], [1, 2]])
    broken = tmp_path/'wrong.pth'
    checkpoint['format'] = 'SEEP_DGCA_v1'
    torch.save(checkpoint, broken)
    with pytest.raises(ValueError, match='Not a DGCA'):
        load_boundary(broken, resumed, resumed_dgca, split, args, Original())
    checkpoint['format'] = 'SEEP_CLS_STEERING_v1'
    checkpoint['implementation_fingerprint'] = 'wrong-source'
    torch.save(checkpoint, broken)
    with pytest.raises(ValueError, match='implementation changed'):
        load_boundary(broken, resumed, resumed_dgca, split, args, Original())


def test_reference_and_screening_requires_full_successful_protocol(tmp_path):
    refs = reference_matrices()
    assert extended_metrics(refs['A'])['aia'] == pytest.approx(87.67822862125138)
    assert extended_metrics(refs['B'])['final_old_task_mean'] == pytest.approx(86.4391818026145)
    args = config()
    args.seed = 3407
    args.cls_steering_layers = [9, 10, 11]
    matrices = {'raw_seen': refs['A']+1, 'raw_exact': refs['A']+1, 'dgca': refs['B']+1}
    summary = export_cls_metrics(tmp_path, matrices, [], args)
    assert summary['comparisons_pp']['C-A']['aia'] == pytest.approx(1.)
    assert summary['comparisons_pp']['D-B']['aia'] == pytest.approx(1.)
    assert not summary['full_protocol_complete'] and not summary['screening_pass']
    assert summary['historical_reference_separate']['aia'] == 87.5441

    # Synthetic metric fixture: test gating, not an experiment result.
    matrices['raw_seen'][:, -1] += .2
    reports = [{'mode': 'align_only', 'transport': None, 'hashes_before': {}, 'hashes_after': {},
                'cls_steering': {'old_banks_before': {}, 'old_banks_after': {},
                                 'learning': {'all_finite': True}, 'attention': {'hashes_unchanged': True}}}
               for _ in range(10)]
    (tmp_path/'run_status.json').write_text(json.dumps({'status': 'success'}))
    summary = export_cls_metrics(tmp_path, matrices, reports, args)
    assert summary['full_protocol_complete'] and summary['screening_pass']
    (tmp_path/'run_status.json').write_text(json.dumps({'status': 'failed'}))
    assert not export_cls_metrics(tmp_path, matrices, reports, args)['full_protocol_complete']


def test_failed_run_revokes_completion_flags(tmp_path, monkeypatch):
    import modules.dgca_runtime as runtime
    args = config()
    args.output_dir = str(tmp_path)
    def fail(*a, **k):
        (tmp_path/'summary.json').write_text(json.dumps({'full_protocol_complete': True, 'screening_pass': True}))
        raise RuntimeError('synthetic failure fixture')
    monkeypatch.setattr(runtime, '_train_and_evaluate_dgca', fail)
    with pytest.raises(RuntimeError, match='synthetic failure'):
        runtime.train_and_evaluate_dgca(args=args)
    assert json.loads((tmp_path/'run_status.json').read_text())['status'] == 'failed'
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert not summary['full_protocol_complete'] and not summary['screening_pass']


def test_full_width_depth_selected_blocks_synthetic_single_batch():
    from vision_transformer import VisionTransformer
    kwargs = options('task', prefix=False)
    kwargs.update(embed_dim=768, depth=12, num_heads=12, num_classes=200, n_tasks=10)
    kwargs['args'].num_tasks = 10
    kwargs['args'].cls_steering_layers = [9, 10, 11]
    model = VisionTransformer(**kwargs)
    freeze(model)
    model.cls_steering.activate(0)
    assert sum(p.numel() for p in model.cls_steering.parameters()) == 69120
    assert sum(p.numel() for p in model.cls_steering.parameters() if p.requires_grad) == 6912
    selected, handles = [], []
    for i, block in enumerate(model.blocks):
        def hook(module, args, kwargs, layer=i):
            if kwargs.get('steering') is not None:
                selected.append(layer)
        handles.append(block.attn.register_forward_pre_hook(hook, with_kwargs=True))
    out = model(torch.randn(1, 3, 16, 16), task_id=0, learned_id=0, train=True, epoch_info=1)
    assert out['classifier_features'].shape == (1, 768) and torch.isfinite(out['logits']).all()
    nn.functional.cross_entropy(out['logits'], torch.tensor([0])).backward()
    assert selected == [9, 10, 11]
    assert (model.cls_steering.banks[0].grad.norm(dim=-1) > 0).all()
    for handle in handles:
        handle.remove()
