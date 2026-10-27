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
from torch.utils.data import DataLoader, TensorDataset

from modules.frequency_state import (CHANNELS, PROTOCOL, FrequencyBank, checkpoint_mapping,
    dynamic_strict_load, inventory, load_checkpoint, matrix_metrics, paired_evaluate,
    state_from, tensor_hash)
from tools.frequency_run import Collector, write_metrics
from tools.frequency_setup import export, shim_timm_text

ROOT = Path(__file__).resolve().parents[1]
BASE = '80081081ab42769d102f83d447705da0639e2fb1'


@pytest.fixture
def tmp_path():
    # Python 3.9 on a restricted Windows token cannot reopen mode=0700 dirs
    # created by pytest's default tmp_path. Inherit the workspace ACL instead.
    import uuid
    path = ROOT / '.test-runs' / str(uuid.uuid4())
    path.mkdir(parents=True)
    return path


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def options():
    args = SimpleNamespace(lgsp='YES', lgsp_type='LGSP', num_tasks=3, input_size=16,
        num_r=100, prompt_hid_dim=3, Dropout_Prompt=.1, temperature=1.)
    return dict(img_size=16, patch_size=8, embed_dim=24, depth=2, num_heads=3,
        num_classes=7, fc_norm=True, prompt_length=4, prompt_pool=True,
        pool_size=3, top_k=1, use_e_prompt=True, e_prompt_layer_idx=[0, 1],
        use_prefix_tune_for_e_prompt=True, n_tasks=3, D1=8, D2=12, use_linear=True,
        relation_type='attention', KI_iter=1, self_attn_idx=[0, 1], warm_up=0, args=args)


def make_model():
    from vision_transformer import VisionTransformer
    return VisionTransformer(**options())


def prepare_model(model, tasks=3):
    model.ensure_prompt_generator(tasks - 1)
    model.register_buffer('stored_unified_prompts', torch.randn(3, 3, 16, 16)) if not hasattr(model, 'stored_unified_prompts') else None
    if 'stored_unified_prompts' not in model._buffers:
        delattr(model, 'stored_unified_prompts')
        model.register_buffer('stored_unified_prompts', torch.randn(3, 3, 16, 16))
    with torch.no_grad():
        model.rainbow_prompt.stored_rainbow_prompts.normal_()
    return model.eval()


def legacy_constructor():
    source = subprocess.check_output(['git', 'show', f'{BASE}:vision_transformer.py'], cwd=ROOT).decode()
    legacy = types.ModuleType('legacy_frequency_vit')
    legacy.__file__ = str(ROOT / 'vision_transformer.py')
    exec(compile(source.replace('@register_model\n', ''), '<pinned base ViT>', 'exec'), legacy.__dict__)
    return legacy.VisionTransformer


def test_off_matches_actual_base_logits_head_features_and_training_rng():
    from vision_transformer import VisionTransformer
    old = legacy_constructor()
    def trace(constructor):
        torch.manual_seed(77)
        model = constructor(**options())
        init_rng = torch.get_rng_state().clone()
        for name, param in model.named_parameters():
            if name.startswith(('blocks.', 'patch_embed.', 'cls_token', 'norm.', 'pos_embed')):
                param.requires_grad_(False)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
        losses = []
        for _ in range(2):
            model.train()
            x = torch.randn(3, 3, 16, 16)
            out = model(x, task_id=0, learned_id=0, cls_features=torch.ones(3, 24), train=True, epoch_info=1)
            loss = nn.functional.cross_entropy(out['logits'], torch.tensor([0, 1, 2])) - .1 * out['sim_loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        captured = []
        handle = model.head.register_forward_pre_hook(lambda m, inputs: captured.append(inputs[0].detach().clone()))
        out = model(x, learned_id=0, cls_features=torch.ones(3, 24))
        handle.remove()
        return losses, tensor_hash(model.state_dict()), torch.get_rng_state(), out['logits'], captured[0], init_rng
    a, b = trace(old), trace(VisionTransformer)
    assert a[:2] == b[:2]
    for i in (2, 3, 4, 5):
        assert torch.equal(a[i], b[i])


@pytest.mark.parametrize('route', [0, 1, 2])
def test_equal_bank_any_original_router_route_logits_features_and_hook(route):
    model = prepare_model(make_model())
    bank = FrequencyBank()
    for t in range(3):
        bank.add(t, model, dict(boundary=t + 1))
    cls = model.rainbow_prompt.base_key[route].detach().reshape(1, -1).repeat(2, 1)
    x = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        live = model(x, cls_features=cls, learned_id=2)
        assert model.matched_task_id == route
        hooks = []
        handle = model.head.register_forward_pre_hook(lambda m, inputs: hooks.append(inputs[0].clone()))
        snap = model(x, cls_features=cls, learned_id=2, frequency_bank=bank)
        handle.remove()
    for key in ('logits', 'classifier_features'):
        assert torch.equal(live[key], snap[key])
    assert torch.equal(hooks[0], snap['classifier_features'])
    assert not torch.equal(snap['pre_logits'], snap['classifier_features'])  # fc_norm=True


def test_only_selected_slot_changes_frequency_mixing_path_and_keeps_caches():
    model = prepare_model(make_model())
    bank = FrequencyBank()
    for t in range(3):
        bank.add(t, model, {})
    selected = 1
    cls = model.rainbow_prompt.base_key[selected].detach()[None].repeat(2, 1)
    x = torch.randn(2, 3, 16, 16)
    before = tensor_hash(model.state_dict())
    with torch.no_grad():
        live = model(x, cls_features=cls, learned_id=2)
        bank._slots[0]['alpha'].add_(100)
        off = model(x, cls_features=cls, learned_id=2, frequency_bank=bank)
        assert torch.equal(live['logits'], off['logits'])
        bank._slots[selected]['weights'].neg_()
        bank._slots[selected]['frequency_residual_scale'].fill_(.8)
        bank._slots[selected]['alpha'].fill_(.8)
        bank._slots[selected]['beta'].fill_(.2)
        captured = []
        handle = model.patch_embed.register_forward_pre_hook(lambda m, inputs: captured.append(inputs[0].clone()))
        changed = model(x, cls_features=cls, learned_id=2, frequency_bank=bank)
        handle.remove()
        state = bank.select(selected, 2, x.device)
        expected = state['alpha'] * (x + model.stored_unified_prompts[selected]) + state['beta'] * model.get_Frequency_mask(x, state)
        assert torch.equal(captured[0], expected)
        assert not torch.allclose(live['logits'], changed['logits'])
    assert before == tensor_hash(model.state_dict())


def test_bank_clones_validates_and_rejects_missing_future_duplicate():
    model = make_model()
    bank = FrequencyBank()
    rng = torch.get_rng_state().clone()
    bank.add(0, model, dict(checkpoint='boundary1'))
    assert torch.equal(rng, torch.get_rng_state())
    selected = bank.select(0, 0, 'cpu')
    selected['weights'].zero_()
    assert not torch.equal(selected['weights'], bank.select(0, 0, 'cpu')['weights'])
    assert all(not v.requires_grad and v.device.type == 'cpu' for v in bank.select(0, 0, 'cpu').values())
    assert sum(v.numel() for v in selected.values()) == 103
    for slot, horizon in [(1, 0), (1, 1), (-1, 1), (0, -1)]:
        with pytest.raises(ValueError):
            bank.select(slot, horizon, 'cpu')
    with pytest.raises(ValueError):
        bank.add(0, model, {})
    bad = state_from(model)
    bad['alpha'] = bad['alpha'].double()
    with pytest.raises(ValueError):
        bank.add(1, bad, {})


class Original(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('fixed', torch.tensor(1.))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return {'pre_logits': x.mean((2, 3)).repeat(1, 8)}


def test_paired_eval_hashes_routes_one_original_call_same_batch_and_restored_rng():
    model = prepare_model(make_model())
    original = Original()
    model.train()
    original.train()
    bank = FrequencyBank()
    for i in range(3):
        bank.add(i, model, {})
    loaders = [dict(val=DataLoader(TensorDataset(torch.randn(35, 3, 16, 16), torch.arange(35) % 7), batch_size=32)) for _ in range(3)]
    matrices = {k: np.full((10, 10), np.nan) for k in CHANNELS}
    model.matched_task_id, model.rainbow_prompt.task_id = 7, 6
    rng, before = torch.get_rng_state().clone(), tensor_hash(model.state_dict())
    routes, diag, check = paired_evaluate(model, original, loaders, 'cpu', 2, [[0, 1], [2, 3], [4, 5, 6]], bank, matrices, lambda _: None)
    assert check['passed'] and before == tensor_hash(model.state_dict())
    assert original.calls == 6
    assert torch.equal(rng, torch.get_rng_state())
    assert model.training and original.training
    assert model.matched_task_id == 7 and model.rainbow_prompt.task_id == 6
    assert sum(r['count'] for r in routes) == 105
    assert all(d['max_abs_logit_delta'] == 0 and d['max_abs_classifier_feature_delta'] == 0 for d in diag)
    np.testing.assert_equal(matrices['live_raw_exact'], matrices['snapshot_raw_exact'])


def test_dynamic_strict_checkpoint_load_shapes_generators_and_ddp(tmp_path):
    model = prepare_model(make_model(), 2)
    state = copy.deepcopy(model.state_dict())
    loaded = make_model()
    rng = torch.get_rng_state().clone()
    dynamic_strict_load(loaded, state, 2)
    assert torch.equal(rng, torch.get_rng_state())
    assert tensor_hash(loaded.state_dict()) == tensor_hash(state)
    assert len(loaded.prompt_generators) == 2
    with pytest.raises(ValueError, match='generators'):
        dynamic_strict_load(make_model(), state, 1)
    broken = dict(state)
    del broken['head.bias']
    with pytest.raises(RuntimeError):
        dynamic_strict_load(make_model(), broken, 2)
    path = tmp_path / 'ddp.pth'
    torch.save(dict(model={'module.' + k: v for k, v in state.items()}, args=options()['args'], epoch=29), path)
    _, restored, _, stripped = load_checkpoint(path, trusted=True)
    assert stripped and tensor_hash(restored) == tensor_hash(state)
    mixed = dict(state)
    mixed['module.head.bias'] = mixed.pop('head.bias')
    torch.save(dict(model=mixed, args={}, epoch=29), path)
    with pytest.raises(ValueError, match='Mixed DDP'):
        load_checkpoint(path, trusted=True)


def synthetic_checkpoints(directory, n=10, dgca=False):
    directory.mkdir(exist_ok=True)
    args = dict(PROTOCOL, temperature=1., freeze=['blocks'], lr=.00375)
    for b in range(1, n + 1):
        state = dict(weights=torch.arange(100, dtype=torch.float32) + b, alpha=torch.tensor(.5),
            beta=torch.tensor(.5), frequency_residual_scale=torch.tensor(.1), radii=torch.arange(100, dtype=torch.float32),
            **{'blocks.0.weight': torch.ones(2)})
        for t in range(b):
            state[f'prompt_generators.{t}.0.weight'] = torch.ones(1)
        payload = dict(model=state, args=args, epoch=29, run_id='synthetic-only')
        if dgca:
            payload.pop('args'); payload.pop('epoch')
            payload.update(experiment_config=dict(args, dgca='align_only'),
                           dgca=dict(config=dict(mode='align_only'), completed_boundary=b - 1))
        torch.save(payload, directory / f'task{b}_checkpoint.pth')


@pytest.mark.parametrize('dgca', [False, True])
def test_inventory_extracts_own_boundary_sha_provenance_and_epoch29(tmp_path, dgca):
    inputs, output = tmp_path / 'input', tmp_path / 'output'
    synthetic_checkpoints(inputs, dgca=dgca)
    report, bank = inventory(inputs, output, trusted=True)
    assert report['full_trajectory']
    assert len(report['boundaries']) == 10
    for task in range(10):
        assert bank.select(task, 9, 'cpu')['weights'][0] == task + 1
        assert bank.provenance[task]['boundary'] == task + 1
    saved = json.loads((output / 'frequency_bank.json').read_text())
    assert saved['tensor_bytes'] == 4120 and saved['tensor_values'] == 1030


@pytest.mark.parametrize('mutation', ['seed', 'temperature', 'backbone', 'radii', 'cls', 'ccsr', 'n1', 'epoch', 'run_id', 'transport'])
def test_inventory_rejects_mixed_runs_enhancements_frozen_changes(tmp_path, mutation):
    inputs, output = tmp_path / 'input', tmp_path / 'output'
    synthetic_checkpoints(inputs)
    path = inputs / 'task2_checkpoint.pth'
    payload, _, _, _ = load_checkpoint(path, trusted=True)
    if mutation in ('seed', 'temperature'):
        payload['args'][mutation] += 1
    elif mutation == 'backbone':
        payload['model']['blocks.0.weight'].add_(1)
    elif mutation == 'radii':
        payload['model']['radii'].add_(1)
    elif mutation in ('cls', 'ccsr', 'n1'):
        payload['model'][{'cls': 'cls_steering.banks.0', 'ccsr': 'ccsr.weight', 'n1': 'n1.weight'}[mutation]] = torch.ones(1)
    elif mutation == 'epoch':
        payload['epoch'] = 28
    elif mutation == 'transport':
        payload['args']['dgca'] = 'transport'
    else:
        payload['run_id'] = 'another-run'
    torch.save(payload, path)
    with pytest.raises(ValueError):
        inventory(inputs, output, trusted=True)
    assert json.loads((output / 'checkpoint_inventory.json').read_text())['status'] == 'failed'


def test_mapping_ambiguity_missing_and_final_only_do_not_train(tmp_path):
    synthetic_checkpoints(tmp_path, n=1)
    with pytest.raises(ValueError, match='Missing checkpoint boundaries'):
        inventory(tmp_path, tmp_path / 'reports', trusted=True)
    report = json.loads((tmp_path / 'reports/checkpoint_inventory.json').read_text())
    assert report['missing_boundaries'] == list(range(2, 11))
    other = tmp_path / 'other'
    synthetic_checkpoints(other, n=1)
    with pytest.raises(ValueError, match='Duplicate boundaries'):
        checkpoint_mapping(tmp_path)
    assert checkpoint_mapping(tmp_path, {'1': str(other / 'task1_checkpoint.pth')})[1].parent == other


def test_metrics_rows_columns_task_average_peaks_partial_and_completion(tmp_path):
    toy = np.array([[90., 80., 85.], [np.nan, 70., 65.], [np.nan, np.nan, 60.]])
    m = matrix_metrics(toy)
    assert m['aia'] == pytest.approx((90 + 75 + 70) / 3)
    assert m['final_accuracy'] == 70
    assert m['final_forgetting'] == 5
    assert m['old_task_final_mean'] == 75
    assert m['old_task_peak_mean'] == 80
    assert m['bwt'] == -5
    assert m['diagonal_initial_mean'] == pytest.approx(220 / 3)
    partial = toy.copy(); partial[:, -1] = np.nan
    assert matrix_metrics(partial)['aia'] is None
    with pytest.raises(ValueError):
        matrix_metrics(toy.T)
    rising = toy.copy(); rising[1, 2] = 95
    assert matrix_metrics(rising)['final_forgetting'] == 2.5
    matrices = {k: np.full((10, 10), np.nan) for k in CHANNELS}
    for a in matrices.values():
        for t in range(10): a[:t + 1, t] = np.arange(t + 1) + 70
    assert not write_metrics(tmp_path, matrices)['full_protocol_complete']
    assert write_metrics(tmp_path, matrices, True, True)['full_protocol_complete']
    (tmp_path / 'checkpoint').mkdir()
    torch.save(dict(secret='not exported'), tmp_path / 'checkpoint/task1_checkpoint.pth')
    import zipfile
    with zipfile.ZipFile(export(tmp_path)) as archive:
        assert not any(n.endswith('.pth') for n in archive.namelist())


def test_collector_deterministic_training_trace_no_optimizer_membership_duplicate_guard(tmp_path):
    def trajectory(collect):
        torch.manual_seed(123)
        model = make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        bank = FrequencyBank()
        matrices = {k: np.full((10, 10), np.nan) for k in CHANNELS}
        observer = Collector(tmp_path, matrices, bank)
        losses = []
        for task in range(2):
            for epoch in range(2):
                model.train()
                x = torch.randn(3, 3, 16, 16)
                out = model(x, task_id=task, learned_id=task, cls_features=torch.ones(3, 24), train=True, epoch_info=epoch)
                loss = out['logits'].square().mean() - .01 * out['sim_loss']
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(float(loss))
            if collect:
                observer.capture(model, task)
                with pytest.raises(ValueError): observer.capture(model, task)  # resume/retry never overwrites a slot
        if collect:
            optimizer_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
            for state in bank._slots.values():
                for tensor in state.values(): assert not tensor.requires_grad and id(tensor) not in optimizer_ids
        return losses, tensor_hash(model.state_dict()), torch.get_rng_state()
    a, b = trajectory(False), trajectory(True)
    assert a[:2] == b[:2] and torch.equal(a[2], b[2])


def test_timm_shim_idempotent():
    text = 'from dataclasses import dataclass, replace\nconv_cfg: MaxxVitConvCfg = MaxxVitConvCfg()\ntransformer_cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg()\n'
    once = shim_timm_text(text)
    assert once == shim_timm_text(once) and once.count(', field') == 1


def test_actual_runner_reeval_orchestrates_ten_boundaries_without_optimizer(tmp_path, monkeypatch):
    """Synthetic integration of actual runner, paired loop, strict load and exports.

    Only protocol-size/data/NPZ/GPU setup is substituted. This is NOT CUB evidence.
    """
    import tools.frequency_run as runtime
    import datasets
    import timm.optim
    import main  # Bind baseline engine imports before installing the optimizer tripwire.
    args = runtime.default_args()
    args.input_size, args.nb_classes, args.device = 16, 7, 'cpu'
    opts = options()
    opts.update(n_tasks=10, pool_size=10)
    opts['args'] = args
    from vision_transformer import VisionTransformer
    source = VisionTransformer(**opts)
    original = Original()
    inputs = tmp_path / 'inputs'; inputs.mkdir()
    bank = FrequencyBank()
    entries = []
    for b in range(1, 11):
        source.ensure_prompt_generator(b - 1)
        if 'stored_unified_prompts' not in source._buffers:
            delattr(source, 'stored_unified_prompts')
            source.register_buffer('stored_unified_prompts', torch.randn(10, 3, 16, 16))
        with torch.no_grad(): source.alpha.add_(.001)
        path = inputs / f'task{b}_checkpoint.pth'
        torch.save(dict(model=source.state_dict(), args=args, epoch=29), path)
        entries.append(dict(boundary=b, path=str(path), sha256=runtime.file_sha(path), args=vars(args).copy()))
        bank.add(b - 1, source, dict(boundary=b))
    def factory(a, path):
        return VisionTransformer(**{**opts, 'args': a}), original
    loader = DataLoader(TensorDataset(torch.randn(2, 3, 16, 16), torch.tensor([0, 1])), batch_size=32)
    loaders = [dict(val=loader) for _ in range(10)]
    monkeypatch.setattr(runtime, 'preflight', lambda config: (dict(boundaries=entries), bank))
    monkeypatch.setattr(runtime, 'validate_config', lambda args: None)
    monkeypatch.setattr(runtime, 'pretrained', lambda p, output: p)
    monkeypatch.setattr(runtime, 'build_models', factory)
    monkeypatch.setattr(runtime, 'verify_original', lambda *a: {'synthetic_test': True})
    monkeypatch.setattr(runtime, 'require_gpu', lambda args: None)
    monkeypatch.setattr(runtime, 'split_manifest', lambda *a: {'sha256': json.loads((ROOT/'references/align_only_seed3407/provenance.json').read_text())['split_sha256']})
    monkeypatch.setattr(datasets, 'build_continual_dataloader', lambda args: (loaders, [[0, 1]] * 10))
    def forbidden(*a, **kw): raise AssertionError('reeval constructed an optimizer')
    monkeypatch.setattr(timm.optim, 'create_optimizer', forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, 'step', forbidden)
    output = tmp_path / 'output'; output.mkdir()
    (output / 'tests_result.json').write_text(json.dumps(dict(returncode=0)))
    config = dict(output=str(output), run_mode='reeval', trusted_checkpoints=True,
                  data_path='synthetic', npz_path='synthetic', device='cpu', num_workers=0)
    runtime.run(config)
    status = json.loads((output / 'run_status.json').read_text())
    summary = json.loads((output / 'summary.json').read_text())
    assert status['status'] == 'complete'
    assert all(m['observed_cells'] == 55 for m in summary['metrics'].values())
    assert original.calls == 55
    assert json.loads((output/'run_config.json').read_text())['actual_args']['lr'] == args.lr
    # Duplicate invocation preserves the previously committed result exactly.
    before = (output / 'summary.json').read_bytes()
    with pytest.raises(ValueError, match='fresh output'):
        runtime.run(config)
    assert (output / 'summary.json').read_bytes() == before


def test_runner_missing_checkpoints_failure_archives_false_completion(tmp_path):
    from tools.frequency_run import run
    config = dict(output=str(tmp_path / 'output'), run_mode='reeval', checkpoint_root=str(tmp_path / 'missing'),
        trusted_checkpoints=True, data_path='unused', npz_path='unused', device='cpu')
    with pytest.raises(ValueError, match='Missing checkpoint boundaries'):
        run(config)
    output = tmp_path / 'output'
    assert not json.loads((output / 'summary.json').read_text())['full_protocol_complete']
    assert json.loads((output / 'run_status.json').read_text())['status'] == 'failed'
    assert (tmp_path / 'SEEP_FREQUENCY_STATE_CUB200_metrics.zip').is_file()


@pytest.mark.parametrize('save_checkpoints', [True, False])
def test_actual_runner_full_training_without_checkpoint_input(tmp_path, monkeypatch, save_checkpoints):
    """Train all ten boundaries without source checkpoints; save outputs by default."""
    import engine
    import datasets
    import modules.frequency_state as frequency_state
    import tools.frequency_run as runtime
    from vision_transformer import VisionTransformer
    args = runtime.default_args()
    args.input_size, args.nb_classes, args.epochs, args.warm_up = 16, 7, 2, 0
    opts = {**options(), 'n_tasks': 10, 'pool_size': 10}
    loader = DataLoader(TensorDataset(torch.randn(2, 3, 16, 16), torch.tensor([0, 1])), batch_size=32)
    loaders, masks = [dict(train=loader, val=loader) for _ in range(10)], [[0, 1]] * 10
    monkeypatch.setattr(runtime, 'default_args', lambda: args)
    monkeypatch.setattr(runtime, 'validate_config', lambda args: None)
    monkeypatch.setattr(runtime, 'pretrained', lambda p, output: p)
    def build_models(a, path):
        model = VisionTransformer(**{**opts, 'args': a})
        for name, param in model.named_parameters():
            if name.startswith(tuple(a.freeze)):
                param.requires_grad_(False)
        return model, Original()
    monkeypatch.setattr(runtime, 'build_models', build_models)
    monkeypatch.setattr(runtime, 'verify_original', lambda *a: {'synthetic_test': True})
    monkeypatch.setattr(runtime, 'require_gpu', lambda args: None)
    monkeypatch.setattr(runtime, 'split_manifest', lambda *a: {'sha256': json.loads(
        (ROOT / 'references/align_only_seed3407/provenance.json').read_text())['split_sha256']})
    monkeypatch.setattr(datasets, 'build_continual_dataloader', lambda args: (loaders, masks))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    def forbidden(*a, **kw): raise AssertionError('Fresh training touched a full model checkpoint')
    monkeypatch.setattr(runtime, 'load_checkpoint', forbidden)
    output = tmp_path / 'output'; output.mkdir()
    if save_checkpoints:
        # Permit tiny synthetic protocol dimensions; validate the real emitted files.
        monkeypatch.setattr(frequency_state, 'validate_config', lambda args: None)
        def output_inventory(root, target, **kwargs):
            assert Path(root) == output / 'checkpoint' and Path(target) == output
            return inventory(root, target, **kwargs)
        monkeypatch.setattr(runtime, 'inventory', output_inventory)
    else:
        monkeypatch.setattr(runtime, 'inventory', forbidden)
        monkeypatch.setattr(torch, 'load', forbidden)
        monkeypatch.setattr(engine.utils, 'save_on_master', forbidden)
    (output / 'tests_result.json').write_text(json.dumps(dict(returncode=0)))
    # No checkpoint input fields; omission of save_checkpoints must save outputs.
    run_config = dict(output=str(output), run_mode='train_collect', data_path='synthetic',
                      npz_path='synthetic', device='cpu', num_workers=0)
    if not save_checkpoints:
        run_config['save_checkpoints'] = False
    runtime.run(run_config)
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['full_protocol_complete']
    assert all(m['observed_cells'] == 55 for m in summary['metrics'].values())
    bank = json.loads((output / 'frequency_bank.json').read_text())
    assert len(bank['slots']) == 10 and bank['tensor_values'] == 1030
    manifest = json.loads((output / 'training_manifest.json').read_text())
    assert manifest['completed_boundaries'] == list(range(1, 11))
    assert manifest['save_checkpoints'] is save_checkpoints and not manifest['source_checkpoints_required']
    config = json.loads((output / 'run_config.json').read_text())
    assert config['optimizer_created'] and config['lr_rescaled']
    assert config['save_checkpoints'] is save_checkpoints
    assert config['actual_args']['lr'] == .00375
    if save_checkpoints:
        assert len(list((output / 'checkpoint').glob('task*_checkpoint.pth'))) == 10
        for boundary in range(1, 11):
            saved, state, _, _ = load_checkpoint(output / f'checkpoint/task{boundary}_checkpoint.pth', trusted=True)
            assert saved['epoch'] == 1 and 'optimizer' in saved and 'lr_scheduler' in saved
            assert bank['slots'][boundary - 1]['sha256'] == tensor_hash(state_from(state))
    else:
        assert not (output / 'checkpoint').exists()
        assert not any(output.rglob('*.pth'))
    # Retrying cannot overwrite a completed bank or metrics.
    before = (output / 'summary.json').read_bytes()
    with pytest.raises(ValueError, match='fresh output'):
        runtime.run(dict(output=str(output), run_mode='train_collect'))
    assert before == (output / 'summary.json').read_bytes()


def test_actual_engine_observer_keeps_training_losses_states_optimizer_and_rng(tmp_path, monkeypatch):
    import engine
    from tools.frequency_run import default_args
    from main import set_seed
    from timm.optim import create_optimizer
    from timm.scheduler import create_scheduler
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    def trajectory(enabled, save_checkpoints=True):
        set_seed(812)
        args = default_args()
        args.input_size, args.num_tasks, args.num_tasks_to_run = 16, 3, 2
        args.nb_classes = 7
        args.epochs, args.warm_up, args.device = 2, 0, 'cpu'
        args.lr *= 32 / 256
        output = tmp_path / (('on' if enabled else 'off') + ('_saved' if save_checkpoints else '_unsaved')); output.mkdir()
        args.output_dir = str(output)
        from vision_transformer import VisionTransformer
        model = VisionTransformer(**{**options(), 'args': args})
        for name, param in model.named_parameters():
            if name.startswith(tuple(args.freeze)): param.requires_grad_(False)
        original = Original()
        masks = [[0, 1, 2], [3, 4], [5, 6]]
        loaders = []
        for mask in masks:
            data = TensorDataset(torch.randn(2, 3, 16, 16), torch.tensor(mask[:2]))
            loaders.append(dict(train=DataLoader(data, batch_size=2), val=DataLoader(data, batch_size=32)))
        optimizer = create_optimizer(args, model)
        scheduler = create_scheduler(args, optimizer)[0]
        bank = FrequencyBank()
        collector = Collector(output, {k: np.full((10, 10), np.nan) for k in CHANNELS}, bank) if enabled else None
        optimizers = []
        def track_optimizer(*a, **kw):
            result = create_optimizer(*a, **kw)
            optimizers.append(result)
            return result
        with monkeypatch.context() as patch:
            patch.setattr(engine, 'create_optimizer', track_optimizer)
            engine.train_and_evaluate(model, model, original, nn.CrossEntropyLoss(), loaders, optimizer,
                scheduler, torch.device('cpu'), masks, args, boundary_observer=collector,
                save_checkpoints=save_checkpoints)
        state, optimizer_state = model.state_dict(), optimizers[-1].state_dict()
        if save_checkpoints:
            saved, restored, _, _ = load_checkpoint(output / 'checkpoint/task2_checkpoint.pth', True)
            assert tensor_hash(state) == tensor_hash(restored)
            optimizer_state = saved['optimizer']
        else:
            assert not (output / 'checkpoint').exists()
        records = [json.loads(line) for file in output.glob('*_stats.txt') for line in file.read_text().splitlines()]
        losses = [r['train_Loss'] for r in records if 'train_Loss' in r]
        assert len(losses) == 2
        return losses, tensor_hash(state), optimizer_state, torch.get_rng_state()
    off = trajectory(False)
    for on in (trajectory(True), trajectory(True, save_checkpoints=False)):
        assert off[:2] == on[:2] and torch.equal(off[3], on[3])
        for key in off[2]['state']:
            for field, value in off[2]['state'][key].items():
                other = on[2]['state'][key][field]
                assert torch.equal(value, other) if torch.is_tensor(value) else value == other
