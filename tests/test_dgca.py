import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from modules.dgca import (DGCA, fit_residual_ridge, covariance_factor, masked_seen,
                          model_hashes, capture_rng, isolated_rng, sample_gaussian)
from modules.dgca_data import current_loader, collect_features, verify_pairs
from modules.dgca_runtime import prepare_dynamic_model, task_optimizer
from tools.summarize_dgca import matrix_metrics, export_metrics


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_vit():
    from vision_transformer import VisionTransformer
    args = SimpleNamespace(lgsp='YES', lgsp_type='LGSP', num_tasks=3, num_r=4,
                           prompt_hid_dim=3, Dropout_Prompt=.1, temperature=1.)
    return VisionTransformer(img_size=16, patch_size=8, embed_dim=24, depth=2, num_heads=3,
                             num_classes=7, fc_norm=True, prompt_length=4, prompt_pool=True,
                             pool_size=3, top_k=1, use_e_prompt=True, e_prompt_layer_idx=[0, 1],
                             n_tasks=3, D1=8, D2=12, use_linear=True, relation_type='attention',
                             KI_iter=1, self_attn_idx=[0, 1], warm_up=0, args=args)


def image_folder(tmp_path):
    for label in ('a', 'b', 'c'):
        directory = tmp_path/label
        directory.mkdir(parents=True)
        for i in range(4):
            Image.new('RGB', (16, 16), (30*i, 60, ord(label))).save(directory/f'{i}.png')
    return ImageFolder(str(tmp_path), transform=transforms.RandomHorizontalFlip(1.),
                       target_transform=None)


def test_row_transport_affine_identity_rank_deficient():
    torch.manual_seed(17)
    x = torch.randn(200, 5, dtype=torch.float64)
    a_true = torch.eye(5, dtype=torch.float64)+torch.randn(5, 5, dtype=torch.float64)*.1
    b_true = torch.randn(5, dtype=torch.float64)
    y = x @ a_true+b_true
    a, b, _ = fit_residual_ridge(x, y, ridge=0)
    torch.testing.assert_close(a, a_true)
    torch.testing.assert_close(b, b_true)
    cloud = torch.randn(50000, 5, dtype=torch.float64)*torch.arange(1, 6)+2
    moved = cloud @ a+b
    mu = cloud.mean(0) @ a+b
    cov = a.T @ torch.cov(cloud.T) @ a
    torch.testing.assert_close(moved.mean(0), mu)
    torch.testing.assert_close(torch.cov(moved.T), cov)
    identity, zero, _ = fit_residual_ridge(x, x)
    torch.testing.assert_close(identity, torch.eye(5, dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(zero, torch.zeros(5, dtype=torch.float64), rtol=0, atol=0)
    x_small = torch.randn(3, 20)
    a, b, _ = fit_residual_ridge(x_small, x_small+.4)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    with pytest.raises(FloatingPointError):
        fit_residual_ridge(x*float('nan'), y)


def test_covariance_transport_boundary_and_sampling(tmp_path):
    torch.manual_seed(8)
    head = nn.Linear(5, 12)
    x = torch.randn(24, 5)
    labels = torch.tensor([2]*12+[9]*12)
    dgca = DGCA(epochs=2, samples_per_class=8)
    dgca.boundary(0, head, x, labels)
    torch.testing.assert_close(dgca.stats[2]['covariance'], torch.cov(x[:12].T))
    old = copy.deepcopy(dgca.stats)
    before = torch.randn(30, 5)
    after = before @ (torch.eye(5)+.2)+.3
    a, b, _ = fit_residual_ridge(before, after)
    report = dgca.boundary(1, head, after, torch.tensor([4]*15+[7]*15), before)
    torch.testing.assert_close(dgca.stats[2]['mean'].double(), old[2]['mean'].double() @ a+b)
    torch.testing.assert_close(dgca.stats[2]['covariance'].double(), a.T @ old[2]['covariance'].double() @ a)
    torch.testing.assert_close(dgca.stats[4]['mean'], after[:15].mean(0))
    for label, stat in dgca.stats.items():
        assert stat['covariance'].device.type == 'cpu'
        assert stat['covariance'].dtype == torch.float32
        assert torch.linalg.eigvalsh(stat['covariance']).min() > -1e-5
        factor, diagnostic = covariance_factor(stat['covariance'], stat['count'])
        torch.testing.assert_close((factor @ factor.T).float(), stat['covariance'], atol=1e-5, rtol=1e-5)
    assert report['transport']['holdout_count'] == 6
    with pytest.raises(ValueError, match='already committed'):
        dgca.boundary(1, head, after, labels)
    logits = torch.randn(20, 12)
    logits[:, 0] = 1000
    for classifier in (logits, dgca.alignment_head(torch.randn(20, 5))):
        assert set(masked_seen(classifier, [2, 4, 7, 9]).argmax(1).tolist()) <= {2, 4, 7, 9}
    # Directly verify non-contiguous remapping trains seen rows only.
    assert torch.equal(head.weight[0], dgca.alignment_head.weight[0])
    assert not torch.equal(head.weight[2], dgca.alignment_head.weight[2])


def test_feature_view_pairs_rng_and_real_classifier_hook(tmp_path):
    base = image_folder(tmp_path)
    training_transform = base.transform
    base.target_transform = lambda label: label+2
    dataset = Subset(Subset(base, [0, 1, 4, 5, 8, 9]), [0, 1, 2, 3])
    loader = current_loader(dataset, transforms.ToTensor(), 123, workers=0)
    model = tiny_vit()
    model.ensure_prompt_generator(0)
    model.train()
    model(torch.rand(4, 3, 16, 16), task_id=0, learned_id=0, cls_features=torch.rand(4, 24), train=True, epoch_info=0)
    class Original(nn.Module):
        def forward(self, images):
            return {'pre_logits': images.mean((2, 3)).repeat(1, 8)}
    original = Original()
    captured = []
    hook = model.head.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().cpu().clone()))
    before_rng = capture_rng()
    before_hashes = model_hashes(model)
    first = collect_features(model, original, loader, 'cpu', 0)
    second = collect_features(model, original, loader, 'cpu', 0)
    hook.remove()
    verify_pairs(first, second)
    torch.testing.assert_close(first['features'], captured[0], rtol=0, atol=0)
    assert first['ids'].tolist() == [0, 1, 4, 5]
    assert first['labels'].tolist() == [2, 2, 3, 3]
    assert base.transform is training_transform and model.training
    assert model_hashes(model) == before_hashes
    torch.testing.assert_close(capture_rng()['torch'], before_rng['torch'], rtol=0, atol=0)
    assert capture_rng()['python'] == before_rng['python']
    np.testing.assert_equal(capture_rng()['numpy'], before_rng['numpy'])
    state = capture_rng()
    with isolated_rng():
        torch.rand(10)
    torch.testing.assert_close(capture_rng()['torch'], state['torch'], rtol=0, atol=0)
    broken = dict(second, images_sha256='different')
    with pytest.raises(ValueError, match='images differ'):
        verify_pairs(first, broken)


def test_alignment_rng_and_strict_dynamic_resume(tmp_path):
    torch.manual_seed(71)
    model = tiny_vit()
    model.ensure_prompt_generator(1)
    model(torch.rand(4, 3, 16, 16), task_id=1, learned_id=1, cls_features=torch.rand(4, 24), train=True, epoch_info=0)
    hashes = model_hashes(model)
    dgca = DGCA(epochs=2, samples_per_class=8)
    x = torch.randn(20, 24)
    rng = capture_rng()
    dgca.boundary(0, model.head, x, torch.tensor([1]*10+[5]*10))
    assert hashes == model_hashes(model)
    torch.testing.assert_close(capture_rng()['torch'], rng['torch'], rtol=0, atol=0)
    assert capture_rng()['python'] == rng['python']
    np.testing.assert_equal(capture_rng()['numpy'], rng['numpy'])
    checkpoint_path = tmp_path/'own.pth'
    torch.save({'model': model.state_dict(), 'dgca': dgca.state_dict()}, checkpoint_path)
    state = torch.load(checkpoint_path, weights_only=False)
    restored = tiny_vit()
    with isolated_rng():
        prepare_dynamic_model(restored, state['model'])
        restored.load_state_dict(state['model'], strict=True)
    assert len(restored.prompt_generators) == 2
    assert hashes == model_hashes(restored)
    restored_dgca = DGCA(epochs=2, samples_per_class=8)
    restored_dgca.load_state_dict(state['dgca'], restored.head)
    torch.testing.assert_close(dgca.alignment_head(x), restored_dgca.alignment_head(x), rtol=0, atol=0)
    y = torch.randn(16, 24)
    labels = torch.tensor([2]*8+[6]*8)
    dgca.boundary(1, model.head, y+.1, labels, y)
    restored_dgca.boundary(1, restored.head, y+.1, labels, y)
    for label in dgca.stats:
        torch.testing.assert_close(dgca.stats[label]['covariance'], restored_dgca.stats[label]['covariance'], rtol=0, atol=0)
    torch.testing.assert_close(dgca.alignment_head(x), restored_dgca.alignment_head(x), rtol=0, atol=0)


def test_off_on_raw_training_trace_identical():
    def trajectory(enabled):
        import random
        random.seed(14)
        np.random.seed(14)
        torch.manual_seed(14)
        model = tiny_vit()
        dgca = DGCA(epochs=1, samples_per_class=4)
        trace = []
        images = torch.rand(8, 3, 16, 16)
        cls = torch.rand(8, 24)
        for task in range(2):
            before = None
            if enabled and task:
                with isolated_rng():
                    model.eval()
                    before = model(images, learned_id=task-1, cls_features=cls)['classifier_features'].detach()
            model.ensure_prompt_generator(task)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
            model.train()
            for step in range(2):
                # Include Python/NumPy/torch training randomness and dynamic head/prompt init.
                mixed = images+torch.rand_like(images)*.01+random.random()*.001+np.random.rand()*.001
                output = model(mixed, task_id=task, learned_id=task, cls_features=cls, train=True, epoch_info=0)
                target = torch.tensor([task*2, task*2+1]*4)
                loss = nn.functional.cross_entropy(output['logits'], target)-.01*output['sim_loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                trace.append(loss.item())
            if enabled:
                with isolated_rng():
                    model.eval()
                    after = model(images, learned_id=task, cls_features=cls)['classifier_features'].detach()
                    dgca.boundary(task, model.head, after, target, before)
        return trace, model_hashes(model), capture_rng()
    off, off_hashes, off_rng = trajectory(False)
    on, on_hashes, on_rng = trajectory(True)
    assert off == on and off_hashes == on_hashes
    torch.testing.assert_close(off_rng['torch'], on_rng['torch'], rtol=0, atol=0)


def test_metrics_orientation_precision_and_idempotence(tmp_path):
    matrix = np.array([[90., 80., 85.], [np.nan, 75., 70.], [np.nan, np.nan, 95.]])
    metrics = matrix_metrics(matrix)
    assert metrics['final_forgetting'] == 5.
    assert metrics['final_accuracy'] == pytest.approx(250/3)
    matrices = {name: matrix for name in ('dgca', 'raw_seen', 'raw_exact')}
    export_metrics(tmp_path, matrices)
    first = (tmp_path/'acc.txt').read_bytes()
    export_metrics(tmp_path, matrices)
    assert (tmp_path/'acc.txt').read_bytes() == first
    assert len(first.splitlines()) == 3
    loaded = np.loadtxt(tmp_path/'task_accuracy_matrix_dgca.csv', delimiter=',')
    assert np.isnan(loaded[2, 0]) and loaded[0, 2] == 85.
    partial = np.full((3, 3), np.nan)
    partial[0, 0] = 94.
    assert matrix_metrics(partial)['completed_tasks'] == 1


def test_engine_resume_same_trajectory_no_duplicate_metrics(tmp_path, monkeypatch):
    import argparse
    from configs.cub200_10task import get_args_parser
    from modules.dgca_runtime import train_and_evaluate_dgca
    import modules.dgca_runtime as runtime
    import engine
    base = image_folder(tmp_path/'images')
    base.transform = transforms.ToTensor()
    data = []
    for ids in (list(range(4)), list(range(4, 12))):
        subset = Subset(base, ids)
        data.append({'train': DataLoader(subset, batch_size=32, shuffle=True),
                     'val': DataLoader(subset, batch_size=32)})
    parser = argparse.ArgumentParser()
    get_args_parser(parser)
    args = parser.parse_args([])
    args.dgca = 'transport'
    args.dgca_epochs = 1
    args.dgca_samples_per_class = 4
    args.num_tasks, args.epochs, args.nb_classes, args.input_size = 2, 1, 7, 16
    args.seed, args.num_workers, args.distributed, args.device = 123, 0, False, 'cpu'
    args.lr *= 32/256
    # Provenance IO is independent of training/resume and tested by notebook.
    monkeypatch.setattr(runtime, 'write_provenance', lambda *arguments: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    calls = []
    original_train = engine.train_one_epoch
    def record_train(*arguments, **kwargs):
        calls.append(kwargs['task_id'])
        return original_train(*arguments, **kwargs)
    monkeypatch.setattr(engine, 'train_one_epoch', record_train)
    class Original(nn.Module):
        def forward(self, images):
            return {'pre_logits': images.mean((2, 3)).repeat(1, 8)}
    def launch(directory, stop=0, resume=False):
        import random
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        model = tiny_vit()
        for name, param in model.named_parameters():
            if name.startswith(('blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed')):
                param.requires_grad = False
        local = copy.deepcopy(args)
        local.output_dir = str(directory)
        local.num_tasks_to_run, local.dgca_resume = stop, resume
        optimizer = torch.optim.Adam(model.parameters(), lr=local.lr)
        dgca = train_and_evaluate_dgca(model, model, Original(), nn.CrossEntropyLoss(), data,
                                     optimizer, None, 'cpu', [[0], [1, 2]], local)
        return model, dgca
    full_model, full = launch(tmp_path/'full')
    calls.clear()
    launch(tmp_path/'resumed', stop=1)
    resumed_model, resumed = launch(tmp_path/'resumed', resume=True)
    assert calls == [0, 1]
    assert model_hashes(full_model) == model_hashes(resumed_model)
    torch.testing.assert_close(full.alignment_head.weight, resumed.alignment_head.weight, rtol=0, atol=0)
    for label in full.stats:
        torch.testing.assert_close(full.stats[label]['covariance'], resumed.stats[label]['covariance'], rtol=0, atol=0)
    first = (tmp_path/'resumed'/'acc.txt').read_bytes()
    calls.clear()
    launch(tmp_path/'resumed', resume=True)
    assert calls == []
    assert (tmp_path/'resumed'/'acc.txt').read_bytes() == first
    for name in ('dgca', 'raw_seen', 'raw_exact'):
        np.testing.assert_equal(np.loadtxt(tmp_path/'full'/f'task_accuracy_matrix_{name}.csv', delimiter=','),
                                np.loadtxt(tmp_path/'resumed'/f'task_accuracy_matrix_{name}.csv', delimiter=','))


def test_low_rank_sampler_retains_full_covariance_without_repeated_shrinkage():
    generator = torch.Generator().manual_seed(456)
    factor = torch.randn(5, 2, generator=generator)
    covariance = factor @ factor.T
    stat = {'mean': torch.arange(5.), 'covariance': covariance, 'count': 3}
    untouched = covariance.clone()
    samples = sample_gaussian(stat, factor, 100000, .1, generator)
    variance = float(torch.trace(covariance)/5)
    expected = .9*covariance+(.1*variance+max(1e-8, 1e-6*variance))*torch.eye(5)
    torch.testing.assert_close(samples.mean(0), stat['mean'], rtol=0, atol=.025)
    torch.testing.assert_close(torch.cov(samples.T), expected, rtol=.035, atol=.025)
    torch.testing.assert_close(covariance, untouched, rtol=0, atol=0)
