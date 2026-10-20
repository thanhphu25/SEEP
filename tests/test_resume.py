import copy
import json
import random
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
import engine
from modules.checkpoint import load_trusted, canonical_config, validate_checkpoint, evaluation_task, evaluation_history
from conftest import make_model, make_reference


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def toy_loaders():
    generator = torch.Generator().manual_seed(71)
    loaders = []
    for scope in ([4, 1], [5, 0], [2, 3]):
        dataset = TensorDataset(torch.rand(4, 3, 16, 16, generator=generator), torch.tensor(scope * 2))
        loaders.append({'train': DataLoader(dataset, batch_size=2, shuffle=True),
                        'val': DataLoader(dataset, batch_size=2),
                        'prototype': DataLoader(dataset, batch_size=2,
                                                generator=torch.Generator().manual_seed(19))})
    return loaders


def run(args, output, resume=''):
    args = copy.deepcopy(args)
    args.output_dir, args.resume = str(output), str(resume)
    args.requested_config = {'seed': args.seed}
    seed_all(args.seed)
    reference, model = make_reference(), make_model(args)
    engine.train_and_evaluate(model, model, reference, torch.nn.CrossEntropyLoss(), toy_loaders(),
                             None, None, torch.device('cpu'), args.class_order, args)
    return load_trusted(output / 'checkpoint/last.pth')


def assert_equal_terminal(left, right):
    for key in left['model']:
        torch.testing.assert_close(left['model'][key], right['model'][key], rtol=0, atol=0)
    np.testing.assert_equal(left['accuracy_matrix'], right['accuracy_matrix'])
    torch.testing.assert_close(left['rng']['torch'], right['rng']['torch'], rtol=0, atol=0)
    assert [g['lr'] for g in left['optimizer']['param_groups']] == [g['lr'] for g in right['optimizer']['param_groups']]


@pytest.mark.parametrize('method', ['seep', 'image_gate', 'ccsp'])
def test_completed_task_resume_exact(arguments, tmp_path, method):
    arguments.method = method
    complete = run(arguments, tmp_path / 'complete')
    checkpoint = tmp_path / 'complete/checkpoint/task1_checkpoint.pth'
    resumed = run(arguments, tmp_path / 'resumed', checkpoint)
    assert_equal_terminal(complete, resumed)
    if method == 'ccsp':
        bank = resumed['model']
        valid_ids = bank['reference_bank.class_ids'][bank['reference_bank.counts'] > 0].tolist()
        assert set(valid_ids) == {4, 1, 5, 0}  # task 2's future classes never enter bank
        assert bank['reference_bank.counts'].tolist() == [2, 2, 2, 2, 0, 0]
    stats = [json.loads(line) for line in (tmp_path / 'resumed/task_stats.jsonl').read_text().splitlines()]
    assert stats[-1]['evaluation']['eval_scope'] == 'seen'
    assert stats[-1]['evaluation']['forgetting'] is not None


def test_epoch_resume_restores_rng_optimizer_and_bank(arguments, tmp_path, monkeypatch):
    arguments.method = 'ccsp'
    arguments.epochs = 2
    arguments.warmup_epochs = 1
    complete = run(arguments, tmp_path / 'complete')
    original_save = engine.atomic_save
    class Interrupted(Exception):
        pass
    def save_then_interrupt(state, path):
        original_save(state, path)
        if not state['task_complete'] and state['task_id'] == 0 and state['epoch'] == 0:
            raise Interrupted()
    monkeypatch.setattr(engine, 'atomic_save', save_then_interrupt)
    with pytest.raises(Interrupted):
        run(arguments, tmp_path / 'interrupted')
    monkeypatch.setattr(engine, 'atomic_save', original_save)
    resumed = run(arguments, tmp_path / 'resumed', tmp_path / 'interrupted/checkpoint/last.pth')
    assert_equal_terminal(complete, resumed)


def test_mismatched_resume_fails(arguments):
    checkpoint = {'protocol_config': canonical_config(arguments), 'split_ids': arguments.split_ids}
    arguments.method = 'ccsp'
    with pytest.raises(ValueError, match='method'):
        validate_checkpoint(checkpoint, arguments)
    with pytest.raises(ValueError, match='Legacy'):
        validate_checkpoint({'model': {}}, arguments, evaluation=True)
    arguments.method = 'seep'
    validate_checkpoint({'model': {}}, arguments, evaluation=True)


def test_legacy_task_metadata_and_warmup_guard(arguments):
    assert evaluation_task({}, 'task3_checkpoint.pth', arguments) == 2
    with pytest.raises(ValueError, match='checkpoint_task'):
        evaluation_task({}, 'unknown.pth', arguments)
    arguments.checkpoint_task = 1
    assert evaluation_task({}, 'unknown.pth', arguments) == 1
    with pytest.raises(ValueError, match='warm-up'):
        evaluation_task({'task_id': 0, 'epoch': 0, 'task_complete': False,
                         'protocol_config': {'warm_up': 2}}, 'last.pth', arguments)


@pytest.mark.parametrize('field,value', [('eval_split', 'dev'), ('gate_mode', 'zero'),
                                        ('eval_router', 'legacy_batch'), ('max_eval_batches', 1)])
def test_eval_override_does_not_mix_incompatible_history(arguments, field, value):
    checkpoint = {'protocol_config': canonical_config(arguments),
                  'accuracy_matrix': np.eye(arguments.num_tasks)}
    np.testing.assert_equal(evaluation_history(checkpoint, arguments), checkpoint['accuracy_matrix'])
    setattr(arguments, field, value)
    assert np.isnan(evaluation_history(checkpoint, arguments)).all()
