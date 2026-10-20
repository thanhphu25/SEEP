import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import transforms
from torch.utils.data import Subset
from modules.arguments import parse_args, CONFIGS
from modules.cub import ReadOnlyCUB
from modules.splits import stratified_indices, prototype_loader
from modules.reference_bank import ReferenceBank, build_prototypes
from scripts.launch import base_command, run_command


@pytest.mark.parametrize('stage', ['smoke', 'pilot', 'full'])
@pytest.mark.parametrize('method', ['seep', 'image_gate', 'ccsp'])
def test_launcher_parser(stage, method):
    command = run_command(base_command('python', '/readonly/cub'), method, stage)
    args = parse_args(command[3:])
    assert args.method == method and args.num_tasks == 10
    assert args.task_inc is False and args.dev_fraction == .1 and args.eval_split == 'dev'
    assert args.num_tasks_to_run == {'smoke': 2, 'pilot': 3, 'full': 0}[stage]
    if stage == 'smoke':
        assert args.warm_up == args.warmup_epochs == 0
        assert args.max_train_batches == args.max_eval_batches == 2


@pytest.mark.parametrize('config', CONFIGS)
def test_legacy_bool_compatibility(config):
    args = parse_args([config, '--task_inc', 'False', '--train_mask', 'True',
                       '--pretrained', 'False', '--use_linear', 'False'])
    assert args.task_inc is False and args.train_mask is True
    assert args.pretrained is False and args.use_linear is False
    default = parse_args([config])
    assert default.method == 'seep' and default.eval_router == 'legacy_batch'
    assert default.eval_scope == 'legacy_head' and default.dev_fraction == 0
    for token, expected in [('True', True), ('False', False), ('1', True), ('0', False)]:
        assert parse_args([config, '--task_inc', token]).task_inc == expected


def test_invalid_protocol():
    with pytest.raises(ValueError, match='dev_fraction'):
        parse_args(['cub200_10task', '--eval_split', 'dev'])
    with pytest.raises(ValueError, match='task_inc'):
        parse_args(['cub200_10task', '--method', 'ccsp', '--task_inc', 'True'])
    with pytest.raises(ValueError, match='caches'):
        parse_args(['cub200_10task', '--epochs', '1'])


def make_raw_cub(root):
    paths, split = [], []
    index = 0
    for name in ('001.Bird_A', '002.Bird_B'):
        for sample in range(12):
            index += 1
            relative = f'{name}/{sample:03d}.jpg'
            image = root / 'images' / relative
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (16, 16), (sample * 10, 20, 30)).save(image)
            paths.append(f'{index} {relative}')
            split.append(f'{index} {int(sample < 10)}')
    (root / 'images.txt').write_text('\n'.join(paths))
    (root / 'train_test_split.txt').write_text('\n'.join(split))


def test_readonly_raw_presplit_and_holdout_bank(tmp_path, arguments):
    make_raw_cub(tmp_path)
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    train = ReadOnlyCUB(tmp_path, True, transforms.ToTensor())
    test = ReadOnlyCUB(tmp_path, False, transforms.ToTensor())
    assert len(train) == 20 and len(test) == 4
    assert train.class_to_idx == {'001.Bird_A': 0, '002.Bird_B': 1}
    tr, dev = stratified_indices(train.targets, .1, 89359)
    assert len(tr) == 18 and len(dev) == 2 and not set(tr) & set(dev)
    assert (tr, dev) == stratified_indices(train.targets, .1, 89359)
    bank = ReferenceBank(2, 3)
    class Encoder(torch.nn.Module):
        def forward(self, x):
            return {'pre_logits': x.mean((2, 3))}
    arguments.num_workers = 0
    # Apply an intentionally invalid stochastic train transform to prove prepass replacement.
    train.transform = lambda x: (_ for _ in ()).throw(AssertionError('train augmentation used'))
    loader = prototype_loader(Subset(train, tr), transforms.ToTensor(), arguments, 0)
    rng = torch.get_rng_state()
    build_prototypes(bank, Encoder(), loader, [0, 1], torch.device('cpu'))
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
    assert bank.counts.tolist() == [9, 9]
    assert not set(train.sample_ids[i] for i in tr) & set(test.sample_ids)
    after = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert before == after
    prepared = tmp_path / 'prepared'
    for partition, dataset in [('train', train), ('test', test)]:
        for (path, label), relative in zip(dataset.samples, dataset.sample_ids):
            target = prepared / partition / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(path).read_bytes())
    alternative = ReadOnlyCUB(prepared, True)
    assert alternative.class_to_idx == train.class_to_idx
    assert alternative.sample_ids == train.sample_ids


def test_actual_continual_loaders_no_dev_or_future_prototypes(tmp_path, arguments):
    from datasets import build_continual_dataloader
    make_raw_cub(tmp_path)
    arguments.data_path = str(tmp_path)
    arguments.dataset = 'Split-CUB200'
    arguments.num_tasks = 2
    arguments.dev_fraction = .1
    arguments.eval_split = 'dev'
    arguments.method = 'ccsp'
    arguments.shuffle = False
    loaders, masks = build_continual_dataloader(arguments)
    assert masks == [[0], [1]]
    assert [len(entry['train'].dataset) for entry in loaders] == [9, 9]
    assert [len(entry['val'].dataset) for entry in loaders] == [1, 1]
    for task, entry in enumerate(loaders):
        labels = torch.cat([label for _, label in entry['prototype']])
        assert set(labels.tolist()) == {task}
    assert len(arguments.split_ids['train']) == 18
    assert len(arguments.split_ids['dev']) == 2
