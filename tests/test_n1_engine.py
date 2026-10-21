import copy
import json
import random
import subprocess

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler
from timm.optim import create_optimizer

from conftest import cub_args, tiny_model
from datasets import build_transform
from engine import train_and_evaluate, _prepare_dynamic_buffers_for_checkpoint
from n1_runtime import trusted_load, task_metrics, write_metrics
from spectral_response_memory import SpectralResponseMemory, capture_rng


class SyntheticCUB(Dataset):
    def __init__(self, transform):
        self.targets = [c for c in range(50) for _ in range(2)]
        self.class_to_idx = {f'class{c}': c for c in range(50)}
        self.transform = transform
        gen = np.random.RandomState(8)
        self.images = [Image.fromarray(gen.randint(0, 256, (8, 8, 3), dtype=np.uint8)) for _ in self.targets]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.targets[i]


def experiment(out, stop=2, mode='semantic_full', sched='cosine', reinit=True):
    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    out.mkdir(parents=True, exist_ok=True)
    args = cub_args(srm_mode=mode, input_size=8, num_r=4, srm_proj_dim=2,
                    srm_samples_per_class=1, seed=23, resolved_seed=23,
                    nb_classes=50, device='cpu', output_dir=str(out), batch_size=2,
                    num_tasks_to_run=stop, epochs=2, warmup_epochs=0, sched=sched,
                    num_workers=0, print_freq=100, n1_launcher=True, reinit_optimizer=reinit)
    args.lr_pre_scaling = args.lr
    args.lr *= args.batch_size / 256
    args.lr_effective = args.lr
    reference = tiny_model(args, prompted=False)
    model = tiny_model(args)
    # Exercise unsaturated spectral gradients in the synthetic test only.
    model.weights.data.copy_(torch.linspace(-.3, .3, args.num_r))
    model.frequency_residual_scale.data.fill_(.4)
    train, val = SyntheticCUB(build_transform(True, args)), SyntheticCUB(build_transform(False, args))
    loaders, masks = [], []
    for task in range(args.num_tasks):
        mask = list(range(task * 5, task * 5 + 5))
        ids = [i for i, c in enumerate(train.targets) if c in mask]
        subset = Subset(train, ids)
        loaders.append(dict(train=DataLoader(subset, sampler=RandomSampler(subset), batch_size=2),
                            val=DataLoader(Subset(val, ids), batch_size=2)))
        masks.append(mask)
    return args, model, reference, loaders, masks


def run(out, stop=2, **kwargs):
    args, model, reference, loaders, masks = experiment(out, stop, **kwargs)
    train_and_evaluate(model, model, reference, torch.nn.CrossEntropyLoss(), loaders,
                       create_optimizer(args, model), None, torch.device('cpu'), masks, args)
    return args, model, reference, loaders, masks


@pytest.mark.parametrize('sched,reinit', [('cosine', True), ('constant', False), ('cosine', False)])
def test_two_task_boundary_resume_matches_uninterrupted(tmp_path, sched, reinit):
    whole = tmp_path / 'whole'
    resumed = tmp_path / 'resumed'
    run(whole, sched=sched, reinit=reinit)
    run(resumed, stop=1, sched=sched, reinit=reinit)
    first = trusted_load(resumed / 'checkpoint' / 'task1_checkpoint.pth')
    assert first['args_pre_scaling']['lr'] == .03
    assert first['effective_args']['lr'] == pytest.approx(.03 * 2 / 256)
    assert first['srm_state']['class_ids'].tolist() == list(range(5))
    assert first['srm_state']['anchors'].shape == (5, 6)
    run(resumed, stop=2, sched=sched, reinit=reinit)
    a = trusted_load(whole / 'checkpoint' / 'task2_checkpoint.pth')
    b = trusted_load(resumed / 'checkpoint' / 'task2_checkpoint.pth')
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    for key in ('R', 'anchors', 'matrices', 'class_ids', 'counts'):
        torch.testing.assert_close(a['srm_state'][key], b['srm_state'][key], rtol=0, atol=0)
    for key in ('anchors', 'matrices'):
        torch.testing.assert_close(first['srm_state'][key], b['srm_state'][key][:5], rtol=0, atol=0)
    assert a['requires_grad'] == b['requires_grad']
    assert torch.equal(a['rng_state']['torch'], b['rng_state']['torch'])
    assert a['rng_state']['python'] == b['rng_state']['python']
    assert np.array_equal(a['rng_state']['numpy'][1], b['rng_state']['numpy'][1])
    np.testing.assert_array_equal(a['acc_matrix'], b['acc_matrix'])
    assert b['per_task'][1]['srm_loss'] > 0
    metrics = json.loads((resumed / 'metrics.json').read_text())
    assert len(metrics['per_task']) == 2 and metrics['summary']['completed_tasks'] == 2
    assert metrics['summary']['is_complete'] is False
    assert metrics['per_task'][0]['forgetting'] is None
    assert len((resumed / 'acc.txt').read_text().splitlines()) == 2
    # Re-running a completed smoke boundary does not collect, evaluate or append.
    run(resumed, stop=2, sched=sched, reinit=reinit)
    assert json.loads((resumed / 'metrics.json').read_text()) == metrics


@pytest.mark.parametrize('mode', ['off', 'l2', 'semantic_diag', 'freeze_spectral'])
def test_modes_smoke(tmp_path, mode):
    _, model, _, _, _ = run(tmp_path / mode, mode=mode)
    checkpoint = trusted_load(tmp_path / mode / 'checkpoint' / 'task2_checkpoint.pth')
    assert checkpoint['srm_state']['completed_task'] == 1
    if mode in ('off', 'freeze_spectral'):
        assert checkpoint['srm_state']['R'] is None
        assert checkpoint['srm_state']['anchors'] is None
    if mode == 'freeze_spectral':
        assert not model.weights.requires_grad and not model.frequency_residual_scale.requires_grad
        assert model.alpha.requires_grad and model.beta.requires_grad


def test_strict_dynamic_load_and_eval_never_collects(tmp_path, monkeypatch):
    args, model, ref, loaders, masks = run(tmp_path / 'eval', stop=1)
    cp = trusted_load(tmp_path / 'eval' / 'checkpoint' / 'task1_checkpoint.pth')
    fresh = tiny_model(args)
    state = capture_rng()
    _prepare_dynamic_buffers_for_checkpoint(fresh, cp['model'])
    assert torch.equal(state['torch'], torch.get_rng_state())
    fresh.load_state_dict(cp['model'], strict=True)
    torch.testing.assert_close(model.stored_unified_prompts, fresh.stored_unified_prompts)
    import main as main_module
    args.eval = True
    monkeypatch.setattr(main_module, 'build_continual_dataloader', lambda args: (loaders, masks))
    monkeypatch.setattr(main_module, 'create_model', lambda *a, **kw: tiny_model(args, prompted='args' in kw))
    def fail(*a, **kw):
        pytest.fail('eval attempted memory collection or training')
    monkeypatch.setattr(SpectralResponseMemory, 'collect', fail)
    monkeypatch.setattr(main_module, 'train_and_evaluate', fail)
    main_module.main(args)  # evaluates saved task 1 then stops at absent task 2


def test_missing_memory_fails_resume(tmp_path):
    out = tmp_path / 'broken'
    run(out, stop=1)
    path = out / 'checkpoint' / 'task1_checkpoint.pth'
    checkpoint = trusted_load(path)
    del checkpoint['srm_state']
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='lacks N1 memory'):
        run(out, stop=2)


def test_metrics_conventions(tmp_path):
    matrix = np.array([[80., 70., 60.], [0., 90., 85.], [0., 0., 75.]])
    rows = [task_metrics(matrix, i, 0, dict(seconds=0, memory_bytes=0)) for i in range(3)]
    report = write_metrics(tmp_path, matrix, rows, 3)
    assert rows[1]['acc_old'] == 70 and rows[1]['acc_new'] == 90
    assert rows[2]['forgetting'] == 12.5 and rows[2]['bwt'] == -12.5
    assert report['summary']['final_accuracy'] == pytest.approx(220 / 3)
    assert report['summary']['avg_incremental_accuracy'] == pytest.approx((80 + 80 + 220 / 3) / 3)
    assert report['A'][1][0] is None and report['A'][0][1] == 70


def test_off_training_rng_trajectory_matches_baseline_engine(tmp_path, monkeypatch):
    args, model, reference, loaders, masks = run(tmp_path / 'off_current', mode='off')
    current_state = copy.deepcopy(model.state_dict())
    rng = capture_rng()
    import engine
    namespace = vars(engine).copy()
    source = subprocess.check_output(['git', 'show', '8008108:engine.py'], text=True)
    exec(compile(source, '<audited engine>', 'exec'), namespace)
    args, baseline, reference, loaders, masks = experiment(tmp_path / 'off_baseline', mode='off')
    # Source's unconditional CUDA sync is not available on this CPU test host.
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a, **kw: None)
    namespace['train_and_evaluate'](baseline, baseline, reference, torch.nn.CrossEntropyLoss(), loaders,
                                    create_optimizer(args, baseline), None, torch.device('cpu'), masks, args)
    for key, value in current_state.items():
        torch.testing.assert_close(value, baseline.state_dict()[key], rtol=0, atol=0)
    assert torch.equal(rng['torch'], torch.get_rng_state())
    assert rng['python'] == random.getstate()
    assert np.array_equal(rng['numpy'][1], np.random.get_state()[1])
