import argparse
import copy
import json
import os
import builtins
from pathlib import Path

import pytest
import torch
from PIL import Image

from conftest import cub_args
from scripts.run_cub_n1 import get_parser, prepare, baseline_configuration, preflight_data, cli
from scripts.prepare_cub import prepare as prepare_raw
from n1_runtime import run_identity
from srm_config import validate_srm


@pytest.fixture
def data_root(tmp_path):
    for split in ('train', 'test'):
        for c in range(200):
            folder = tmp_path / 'data' / 'CUB_200_2011' / split / f'{c + 1:03d}.class'
            folder.mkdir(parents=True)
            Image.new('RGB', (2, 2), color=(c, 1, 2)).save(folder / 'image.jpg')
    return tmp_path / 'data'


def save_baseline(path, tasks=10, **overrides):
    args = cub_args(tasks, seed=23, **overrides)
    args.lr *= args.batch_size / 256
    torch.save(dict(args=args, model={'unused': torch.ones(2)}), path)
    return args


def options(checkpoint, data_root, output, *extra):
    return get_parser().parse_args(['--baseline-checkpoint', str(checkpoint), '--seed', '23',
                                   '--data-path', str(data_root), '--output-dir', str(output),
                                   '--srm-mode', 'semantic_full', *extra])


@pytest.mark.parametrize('tasks', [10, 20])
@pytest.mark.parametrize('unscale', [True, False])
def test_launcher_lr_types_seed_and_dry_rerun(tmp_path, data_root, tasks, unscale, capsys):
    path = tmp_path / 'baseline.pth'
    baseline = save_baseline(path, tasks, unscale_lr=unscale)
    opt = options(path, data_root, tmp_path / 'out')
    args = prepare(opt)
    assert args.lr == pytest.approx(.03)
    assert args.lr * args.batch_size / 256 == pytest.approx(baseline.lr)
    assert args.lr_local == baseline.lr_local and args.lr_Frequency_mask == baseline.lr_Frequency_mask
    for key in ('freeze', 'self_attn_idx', 'e_prompt_layer_idx', 'opt_betas', 'pin_mem', 'use_g_prompt', 'task_inc'):
        assert getattr(args, key) == getattr(baseline, key)
        assert type(getattr(args, key)) is type(getattr(baseline, key))
    assert args.num_tasks == tasks and args.num_tasks_to_run == 0
    assert prepare(opt).lr == args.lr  # existing dry-run config is not a resume checkpoint
    config = json.loads((tmp_path / 'out' / 'effective_config.json').read_text())
    assert config['provenance']['status'] == 'unknown'
    assert not json.loads((tmp_path / 'out' / 'config_diff.json').read_text())['research']
    assert not (tmp_path / 'out' / 'checkpoint').exists()


def test_seed_zero_and_lr_metadata_guards(tmp_path, capsys):
    path = tmp_path / 'baseline.pth'
    save_baseline(path)
    with pytest.raises(ValueError, match='must be >0'):
        baseline_configuration(path, 0)
    with pytest.raises(ValueError, match='differs'):
        baseline_configuration(path, 42)
    args = save_baseline(path)
    args.seed = 0
    torch.save(dict(args=args), path)
    baseline_configuration(path, 23)
    assert 'USER-CONFIRMED' in capsys.readouterr().out
    torch.save(dict(args=args, lr_pre_scaling=.03, resolved_seed=23), path)
    _, meta = baseline_configuration(path, 23)
    assert meta['lr']['pre_scaling'] == .03
    torch.save(dict(args=args, lr_pre_scaling=.5), path)
    with pytest.raises(ValueError, match='metadata disagrees'):
        baseline_configuration(path, 23)
    torch.save(dict(model={}), path)
    with pytest.raises(ValueError, match='lacks args'):
        baseline_configuration(path, 23)


def test_output_mismatch_and_smoke_mapping(tmp_path, data_root):
    path = tmp_path / 'baseline.pth'
    save_baseline(path, num_tasks_to_run=2)
    out = tmp_path / 'out'
    args = prepare(options(path, data_root, out))
    assert args.num_tasks_to_run == 0
    smoke = prepare(options(path, data_root, out, '--num-tasks-to-run', '2', '--srm-normalize', 'none'))
    assert smoke.num_tasks_to_run == 2 and smoke.num_tasks == 10 and smoke.srm_normalize == 'none'
    (out / 'checkpoint').mkdir()
    torch.save(dict(run_identity=run_identity(args), srm_state={}), out / 'checkpoint' / 'task1_checkpoint.pth')
    with pytest.raises(ValueError, match='incompatible'):
        prepare(options(path, data_root, out, '--srm-lambda', '2'))
    with pytest.raises(ValueError, match='incompatible'):
        prepare(options(path, data_root, out, '--srm-normalize', 'none'))
    prepare(options(path, data_root, out, '--num-tasks-to-run', '2'))


def test_cub_guard_works_without_archive_or_input_writes(tmp_path, data_root, monkeypatch):
    from continual_datasets import continual_datasets as module
    def fail(*args, **kwargs):
        pytest.fail('Prepared CUB attempted download, extraction or split')
    monkeypatch.setattr(module, 'download_url', fail)
    monkeypatch.setattr(module.CUB200, 'split', fail)
    # Explicitly reject any input tree writes. On Windows chmod alone is not a reliable ACL test.
    original_open = open
    def protected_open(file, mode='r', *args, **kwargs):
        if any(c in mode for c in 'wax+') and Path(file).resolve().is_relative_to(data_root.resolve()):
            pytest.fail('Write attempted inside read-only dataset root')
        return original_open(file, mode, *args, **kwargs)
    monkeypatch.setattr('builtins.open', protected_open)
    assert len(module.CUB200(str(data_root), train=True, download=True).data) == 200
    assert len(module.CUB200(str(data_root), train=False, download=True).data) == 200
    assert len(preflight_data(data_root)['class_to_idx']) == 200
    (data_root / 'CUB_200_2011' / 'test' / '001.class').rename(data_root / 'CUB_200_2011' / 'test' / 'wrong')
    with pytest.raises(ValueError, match='class_to_idx mismatch'):
        preflight_data(data_root)


def test_official_prepare_copy_and_symlink(tmp_path):
    raw = tmp_path / 'raw'
    (raw / 'images' / '001.bird').mkdir(parents=True)
    (raw / 'images.txt').write_text('2 001.bird/b.jpg\n1 001.bird/a.jpg\n')
    (raw / 'train_test_split.txt').write_text('1 1\n2 0\n')
    for name in ('a', 'b'):
        Image.new('RGB', (2, 2)).save(raw / 'images' / '001.bird' / f'{name}.jpg')
    out = tmp_path / 'prepared'
    prepare_raw(raw, out)
    prepare_raw(raw, out)
    assert (out / 'CUB_200_2011' / 'train' / '001.bird' / 'a.jpg').exists()
    assert (out / 'CUB_200_2011' / 'test' / '001.bird' / 'b.jpg').exists()
    assert len(list((raw / 'images').rglob('*.jpg'))) == 2
    with pytest.raises(ValueError, match='separate'):
        prepare_raw(raw, raw / 'nested')


@pytest.mark.parametrize('tasks', [10, 20])
def test_defaults_and_scope(tasks):
    args = cub_args(tasks)
    assert validate_srm(args)['srm_mode'] == 'off'
    args.srm_mode = 'semantic_full'
    validate_srm(args)
    args.lgsp_type = 'GSP'
    with pytest.raises(ValueError):
        validate_srm(args)


def test_dry_run_no_training_imports_or_weights(tmp_path, data_root, monkeypatch):
    path = tmp_path / 'baseline.pth'
    save_baseline(path)
    old_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name in ('main', 'models', 'vision_transformer', 'engine', 'timm'):
            pytest.fail(f'dry-run imported training module {name}')
        return old_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    cli(['--baseline-checkpoint', str(path), '--seed', '23', '--data-path', str(data_root),
         '--output-dir', str(tmp_path / 'dry'), '--srm-mode', 'semantic_full', '--dry-run'])


@pytest.mark.parametrize('tasks', [10, 20])
def test_fresh_run_uses_repo_defaults_without_loading_any_checkpoint(tmp_path, data_root, monkeypatch, tasks):
    import scripts.run_cub_n1 as launcher
    def fail(*a, **kw):
        pytest.fail('Fresh run tried to read an old checkpoint')
    monkeypatch.setattr(launcher, 'trusted_load', fail)
    opt = get_parser().parse_args(['--config', f'cub200_{tasks}task', '--seed', '3407',
                                  '--data-path', str(data_root), '--output-dir', str(tmp_path / 'fresh'),
                                  '--srm-mode', 'semantic_full', '--dry-run'])
    args = prepare(opt)
    defaults = cub_args(tasks)
    assert args.num_tasks == tasks and args.seed == 3407
    for key in ('batch_size', 'num_r', 'model', 'epochs', 'pretrained', 'lr', 'lr_local',
                'lr_Frequency_mask', 'num_workers', 'pin_mem', 'freeze', 'opt_betas'):
        assert getattr(args, key) == getattr(defaults, key)
        assert type(getattr(args, key)) is type(getattr(defaults, key))
    report = json.loads((tmp_path / 'fresh' / 'effective_config.json').read_text())
    assert report['baseline_checkpoint'] is None and report['config_source'] == 'repo_defaults'
    assert report['learning_rates']['pre_scaling'] == .03
    assert report['learning_rates']['effective'] == pytest.approx(.00375)
    assert report['provenance']['status'] == 'fresh_run_repo_defaults'
    assert 'log' not in report['seed']['confirmation']
    assert args.num_tasks_to_run == 0
    assert prepare(opt).lr == .03  # rerun preflight does not scale again
    opt.seed = 0
    with pytest.raises(ValueError, match='reproducible fresh run'):
        prepare(opt)


def test_config_sources_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        get_parser().parse_args(['--config', 'cub200_10task', '--baseline-checkpoint', 'old.pth',
                                 '--seed', '7', '--data-path', 'data', '--output-dir', 'out'])
