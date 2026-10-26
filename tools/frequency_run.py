"""Paired raw SEEP frequency-state evaluation; reeval never builds an optimizer."""
import argparse
import csv
import gc
import json
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from modules.frequency_state import (CHANNELS, FrequencyBank, dynamic_strict_load, file_sha,
    inventory, isolated_rng, json_write, load_checkpoint, matrix_metrics, paired_evaluate,
    tensor_hash, validate_config)
from tools.frequency_setup import export, pretrained, NPZ_SHA, DGCA_SHA

BASE = '80081081ab42769d102f83d447705da0639e2fb1'


def default_args():
    from configs.cub200_10task import get_args_parser
    parser = argparse.ArgumentParser()
    get_args_parser(parser)
    args = parser.parse_args([])
    args.seed, args.distributed, args.nb_classes = 3407, False, 200
    return args


def require_gpu(args):
    if not str(args.device).startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('Full CUB protocol requires Kaggle GPU; CPU unit tests are not an experiment')


def environment_freeze():
    result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True)
    if result.returncode == 0:
        return result.stdout
    import importlib.metadata
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata['Name'])
    return ('# pip unavailable; installed distributions from importlib.metadata\n' + '\n'.join(packages) + '\n').encode()


def build_models(args, npz):
    from timm.models import create_model
    import models  # register SEEP ViT, not timm's stock ViT
    common = dict(pretrained=False, num_classes=args.nb_classes, drop_rate=args.drop,
                  drop_path_rate=args.drop_path, drop_block_rate=None)
    original = create_model(args.model, **common)
    options = dict(prompt_length=args.length, embedding_key=args.embedding_key, prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool, prompt_key=args.prompt_key, pool_size=args.size, top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt, prompt_key_init=args.prompt_key_init, head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask, use_g_prompt=args.use_g_prompt, g_prompt_length=args.g_prompt_length,
        g_prompt_layer_idx=args.g_prompt_layer_idx, use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
        use_e_prompt=args.use_e_prompt, e_prompt_layer_idx=args.e_prompt_layer_idx,
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt, same_key_value=args.same_key_value,
        n_tasks=args.num_tasks, D1=args.D1, D2=args.D2, relation_type=args.relation_type,
        use_linear=args.use_linear, warm_up=args.warm_up, KI_iter=args.KI_iter, self_attn_idx=args.self_attn_idx, args=args)
    model = create_model(args.model, **common, **options)
    # Same custom NPZ loader as pretrained=True in pinned SEEP/timm, with verified local file.
    original.load_pretrained(str(npz))
    model.load_pretrained(str(npz))
    with np.load(npz) as weights:
        prefix = 'opt/target/' if 'opt/target/embedding/kernel' in weights else ''
        kernel = torch.from_numpy(weights[prefix + 'embedding/kernel'].transpose(3, 2, 0, 1))
        for loaded in (original, model):
            torch.testing.assert_close(loaded.patch_embed.proj.weight.detach().cpu(), kernel, rtol=0, atol=0)
    original.requires_grad_(False)
    for name, param in model.named_parameters():
        if name.startswith(tuple(args.freeze)):
            param.requires_grad_(False)
    return model, original


def split_manifest(loaders, masks, args):
    from torch.utils.data import Subset
    tasks = []
    for task in loaders:
        entry = {}
        for split in ('train', 'val'):
            base = task[split].dataset
            indices = list(range(len(base)))
            while isinstance(base, Subset):
                indices = [int(base.indices[i]) for i in indices]
                base = base.dataset
            samples = base.samples
            entry[split] = [dict(index=i, label=int(base.targets[i]),
                                path=Path(samples[i][0]).parent.name + '/' + Path(samples[i][0]).name) for i in indices]
        tasks.append(entry)
    value = dict(class_mask=masks, tasks=tasks, batch_size=32, stats_sampler='sequential', stats_drop_last=False)
    import hashlib
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if masks != [list(range(t * 20, (t + 1) * 20)) for t in range(10)]:
        raise ValueError('Unexpected class order')
    if sum(len(t['train']) for t in tasks) != 5994 or sum(len(t['val']) for t in tasks) != 5794:
        raise ValueError('Official CUB counts differ')
    return dict(**value, sha256=digest, train_transform=str(loaders[0]['train'].dataset.dataset.transform),
                val_transform=str(loaders[0]['val'].dataset.dataset.transform))


def verify_original(model, original, args):
    if any(p.requires_grad for p in original.parameters()):
        raise ValueError('Original routing ViT is not frozen')
    state, raw = model.state_dict(), original.state_dict()
    shared = [k for k in raw if k in state and k.startswith(tuple(args.freeze))]
    if not shared or any(not torch.equal(state[k].cpu(), raw[k].cpu()) for k in shared):
        raise ValueError('Checkpoint frozen backbone disagrees with verified routing NPZ. Investigate pretrained/source.')
    return dict(frozen=True, compared_tensors=len(shared), frozen_tensors_sha256=tensor_hash({k: raw[k] for k in shared}),
                routing_representation_sha256=tensor_hash({k: v for k, v in raw.items() if not k.startswith('head.')}),
                pretrained_sha256=NPZ_SHA)


def dgca_representation_hash(original):
    """Exact hash serialization used by the supplied ZIP (not a learned DGCA module)."""
    import hashlib
    digest = hashlib.sha256()
    for key, value in sorted(original.state_dict().items()):
        if key.startswith('head.'):
            continue
        value = value.detach().cpu().contiguous()
        inner = hashlib.sha256(str((value.dtype, tuple(value.shape))).encode())
        inner.update(value.numpy().tobytes())
        digest.update(key.encode())
        digest.update(inner.hexdigest().encode())
    return digest.hexdigest()


def write_metrics(output, matrices, process_success=False, checks_pass=False):
    output = Path(output)
    metrics = {}
    for name, matrix in matrices.items():
        np.savetxt(output / f'task_accuracy_matrix_{name}.csv', matrix, delimiter=',', fmt='%.17g')
        # Recompute from saved bytes so exported metrics never depend on in-memory rounding.
        matrix = np.loadtxt(output / f'task_accuracy_matrix_{name}.csv', delimiter=',')
        metrics[name] = matrix_metrics(matrix)
        with (output / f'per_task_{name}.csv').open('w', newline='', encoding='utf8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['task', 'initial', 'peak_observed', 'final'])
            for i in range(10):
                row = matrix[i, i:]
                writer.writerow([i + 1, '' if not np.isfinite(matrix[i, i]) else matrix[i, i],
                    '' if not np.isfinite(row).any() else np.nanmax(row),
                    '' if not np.isfinite(matrix[i, -1]) else matrix[i, -1]])
    full = all(m['full_matrix'] and m['observed_cells'] == 55 for m in metrics.values())
    complete = bool(full and process_success and checks_pass)
    deltas = {}
    for channel in ('raw_exact', 'raw_seen'):
        live, snap = metrics['live_' + channel], metrics['snapshot_' + channel]
        deltas[channel] = {k: None if live[k] is None else snap[k] - live[k]
            for k in ('aia', 'final_accuracy', 'final_forgetting', 'old_task_final_mean', 'old_task_peak_mean', 'diagonal_initial_mean', 'bwt')}
    primary = deltas['raw_exact']
    signal = (primary['aia'] >= .3 and primary['final_accuracy'] > 0 and primary['old_task_final_mean'] > 0) if complete else None
    summary = dict(full_protocol_complete=complete, process_success=process_success, checks_pass=checks_pass,
        metrics=metrics, snapshot_minus_live_percentage_points=deltas, screening_signal=signal,
        forgetting_goal=None if signal is None else bool(signal and primary['final_forgetting'] < 0),
        primary='paired raw_exact on the same checkpoint trajectory',
        interpretation='One seed; screening rule is not statistical significance. Inspect peaks with forgetting. '
        'No test-based state tuning or novelty/zero-forgetting claim.',
        historical_reference_only=dict(original=[87.5441, 86.4980, 2.9515],
            align_only_raw=[87.67822862125138, 86.4383690615546, 3.3992381159689904]))
    json_write(output / 'summary.json', summary)
    with (output / 'comparison.csv').open('w', newline='', encoding='utf8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['channel', 'metric', 'live', 'snapshot', 'snapshot_minus_live_pp'])
        for channel, delta in deltas.items():
            for key, value in delta.items():
                writer.writerow([channel, key, metrics['live_' + channel][key], metrics['snapshot_' + channel][key], value])
    return summary


def compare_source(entry, payload, matrices, output):
    reference = payload.get('matrices', {}).get('raw_exact')
    records = json.loads((output / 'source_accuracy_comparison.json').read_text()) if (output / 'source_accuracy_comparison.json').exists() else []
    b = entry['boundary'] - 1
    comparison = dict(boundary=b + 1, source='saved checkpoint raw_exact matrix', available=reference is not None,
        limitations='No substitution of historical values. Differences can arise from source/transforms/pretrained/batching; '
                     'inspect manifests. Missing metadata remains unknown.')
    if reference is not None:
        ref = np.asarray(reference)[:b + 1, b]
        comparison.update(saved=ref.tolist(), reeval=matrices['live_raw_exact'][:b + 1, b].tolist(),
                          delta_pp=(matrices['live_raw_exact'][:b + 1, b] - ref).tolist())
    records.append(comparison)
    json_write(output / 'source_accuracy_comparison.json', records)


def preflight(config):
    output = Path(config['output'])
    if config['run_mode'] == 'reeval':
        report, bank = inventory(config['checkpoint_root'], output, config.get('checkpoint_map'), config['trusted_checkpoints'])
        for e in report['boundaries']:
            print(f"boundary={e['boundary']:2d} {e['status']:10s} epoch={e.get('epoch')} size={e.get('size_bytes',0)/2**20:.1f}MiB {e['path']}")
        return report, bank
    if config['run_mode'] != 'train_collect':
        raise ValueError('run_mode must be reeval or train_collect')
    # Fresh run only; never resume silently from final checkpoint or overwrite old metrics/bank.
    if (output / 'checkpoint').exists() or (output / 'frequency_bank.json').exists():
        raise ValueError('train_collect requires a fresh output. Interrupted collected checkpoints remain usable for explicit reeval; '
                         'training resume is intentionally unsupported, so boundary metrics/snapshots cannot duplicate.')
    json_write(output / 'checkpoint_inventory.json', dict(status='new_train_collect', missing_boundaries=list(range(1, 11)),
        full_trajectory=False, note='New full 10x30 run. No final-checkpoint resume.'))
    return None, FrequencyBank()


class Collector:
    def __init__(self, output, matrices, bank):
        self.output, self.matrices, self.bank = output, matrices, bank
        self.routes, self.diagnostics, self.checks = [], [], []

    def capture(self, model, task):
        with isolated_rng():
            self.bank.add(task, model, dict(boundary=task + 1, source='after final optimizer/scheduler, before evaluation'))
            self.bank.save(self.output / 'frequency_bank.json')

    def evaluate(self, model, original, loaders, device, task, masks):
        # Includes data-loader base-seed consumption in isolated_rng so collector changes no training RNG.
        routes, diagnostics, check = paired_evaluate(model, original, loaders, device, task, masks, self.bank, self.matrices)
        self.routes.extend(routes)
        self.diagnostics.extend(diagnostics)
        self.checks.append(check)
        json_write(self.output / 'route_counts.json', self.routes)
        json_write(self.output / 'paired_diagnostics.json', self.diagnostics)
        json_write(self.output / 'mutation_checks.json', self.checks)
        write_metrics(self.output, self.matrices)


def run(config, check_only=False):
    from datasets import build_continual_dataloader
    from main import initialize_experiment
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    if not check_only and any(output.glob('task_accuracy_matrix_*.csv')):
        raise ValueError('Use a fresh output; existing metrics will not be overwritten')
    start = time.perf_counter()
    matrices = {name: np.full((10, 10), np.nan, dtype=np.float64) for name in CHANNELS}
    json_write(output / 'run_status.json', dict(status='running', phase='preflight', full_protocol_complete=False))
    try:
        report, bank = preflight(config)
        args = SimpleNamespace(**report['boundaries'][0]['args']) if report else default_args()
        saved = vars(args).copy()
        overrides = dict(data_path=str(config['data_path']).rstrip('/\\') + '/', output_dir=str(output),
                         device=config.get('device', 'cuda:0'), num_workers=config.get('num_workers', 4),
                         pin_mem=config.get('pin_mem', getattr(args, 'pin_mem', True)),
                         print_freq=config.get('print_freq', getattr(args, 'print_freq', 100)),
                         distributed=False, world_size=1, eval=config['run_mode'] == 'reeval', num_tasks_to_run=0)
        for key, value in overrides.items():
            setattr(args, key, value)
        args.nb_classes = getattr(args, 'nb_classes', 200)
        validate_config(vars(args))
        if config['run_mode'] == 'train_collect':
            # Scale the fresh CLI LR ONCE, exactly as main.py. Saved reeval args never scale.
            args.lr *= args.batch_size / 256.
        initialize_experiment(args.seed)
        # Preserve pinned main.py execution policy; log actual behavior.
        torch.backends.cudnn.benchmark = True
        npz = pretrained(config['npz_path'], output)
        loaders, masks = build_continual_dataloader(args)
        split = split_manifest(loaders, masks, args)
        json_write(output / 'split_manifest.json', split)
        reference = json.loads((ROOT / 'references/align_only_seed3407/provenance.json').read_text())
        if reference.get('split_sha256') and split['sha256'] != reference['split_sha256']:
            raise ValueError('Prepared task split differs from verified run')
        model, original = build_models(args, npz)
        json_write(output / 'run_config.json', dict(mode=config['run_mode'], saved_args=saved, actual_args=vars(args),
            path_overrides=overrides, optimizer_created=config['run_mode'] == 'train_collect', lr_rescaled=report is None,
            class_order=list(range(200)), cudnn_benchmark=torch.backends.cudnn.benchmark))
        import torchvision, timm
        json_write(output / 'environment.json', dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
            torchvision=torchvision.__version__, timm=timm.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None))
        (output / 'pip_freeze.txt').write_bytes(environment_freeze())
        if report:
            entry = report['boundaries'][0]
            if file_sha(entry['path']) != entry['sha256']:
                raise ValueError('Source checkpoint changed after inventory')
            payload, state, _, _ = load_checkpoint(entry['path'], config['trusted_checkpoints'])
            del payload
            dynamic_strict_load(model, state, 1)
            del state
        verification = verify_original(model, original, args)
        json_write(output / 'routing_pretrained_verification.json', verification)
        tests = json.loads((output / 'tests_result.json').read_text())
        if tests.get('returncode') != 0:
            raise ValueError('Required implementation tests have not passed')
        require_gpu(args)
        json_write(output / 'preflight_results.json', dict(passed=True, checks=['protocol', 'official_split', 'NPZ_SHA256',
            'strict_dynamic_first_checkpoint' if report else 'fresh_run', 'frozen_original', 'tests', 'GPU'],
            elapsed_seconds=time.perf_counter() - start))
        if check_only:
            json_write(output / 'run_status.json', dict(status='preflight_passed', full_protocol_complete=False))
            return
        device = torch.device(args.device)
        model.to(device)
        original.to(device)
        observer = Collector(output, matrices, bank)
        boundary_times = []
        if report:
            # Single checkpoint payload at a time, never ten optimizer states in RAM.
            for entry in report['boundaries']:
                tick = time.perf_counter()
                if file_sha(entry['path']) != entry['sha256']:
                    raise ValueError('Checkpoint changed after preflight')
                payload, state, _, _ = load_checkpoint(entry['path'], config['trusted_checkpoints'])
                source_matrix = payload.get('matrices', {}).get('raw_exact')
                source_original = payload.get('original_representation_sha256')
                del payload
                if source_original is not None and source_original != dgca_representation_hash(original):
                    raise ValueError('Saved routing ViT hash does not match verified NPZ representation')
                dynamic_strict_load(model, state, entry['boundary'])
                del state
                gc.collect()
                verify_original(model, original, args)
                observer.evaluate(model, original, loaders, device, entry['boundary'] - 1, masks)
                compare_source(entry, dict(matrices={} if source_matrix is None else dict(raw_exact=source_matrix)), matrices, output)
                boundary_times.append(dict(boundary=entry['boundary'], seconds=time.perf_counter() - tick))
                elapsed = time.perf_counter() - start
                done = entry['boundary'] * (entry['boundary'] + 1) / 2
                print(f"boundary={entry['boundary']}/10 complete elapsed={elapsed:.1f}s ETA_run={elapsed*(55-done)/done:.1f}s", flush=True)
                json_write(output / 'timing.json', dict(boundaries=boundary_times, elapsed_seconds=elapsed))
        else:
            from engine import train_and_evaluate
            from timm.optim import create_optimizer
            from timm.scheduler import create_scheduler
            optimizer = create_optimizer(args, model)
            scheduler = create_scheduler(args, optimizer)[0] if args.sched != 'constant' else None
            train_and_evaluate(model, model, original, torch.nn.CrossEntropyLoss().to(device), loaders,
                optimizer, scheduler, device, class_mask=masks, args=args, boundary_observer=observer)
            # Reinspect actual emitted checkpoints and verify snapshots came from their own boundary.
            report, extracted = inventory(output / 'checkpoint', output, trusted=True)
            for task in range(10):
                if tensor_hash(bank.select(task, 9, 'cpu')) != tensor_hash(extracted.select(task, 9, 'cpu')):
                    raise ValueError('Collected snapshot differs from emitted checkpoint boundary')
            extracted.save(output / 'frequency_bank.json')
        checks_pass = len(observer.checks) == 10 and all(c['passed'] for c in observer.checks)
        summary = write_metrics(output, matrices, process_success=True, checks_pass=checks_pass)
        if not summary['full_protocol_complete']:
            raise ValueError('Incomplete protocol after evaluation')
        json_write(output / 'timing.json', dict(boundaries=boundary_times, elapsed_seconds=time.perf_counter() - start))
        json_write(output / 'run_status.json', dict(status='complete', full_protocol_complete=True, process_returncode=0))
    except BaseException as exc:
        # Preserve partial matrices; never claim full completion on exception/interrupt.
        write_metrics(output, matrices, process_success=False, checks_pass=False)
        json_write(output / 'run_status.json', dict(status='failed', error=str(exc), full_protocol_complete=False, process_returncode=1))
        json_write(output / 'timing.json', dict(elapsed_seconds=time.perf_counter() - start))
        traceback.print_exc()
        raise
    finally:
        try:
            export(output)
        except BaseException as exc:
            write_metrics(output, matrices, process_success=False, checks_pass=False)
            json_write(output / 'run_status.json', dict(status='failed', error='Export failed: ' + str(exc), full_protocol_complete=False))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--inventory-only', action='store_true')
    cli = parser.parse_args()
    config = json.loads(cli.config.read_text(encoding='utf8'))
    if cli.inventory_only:
        preflight(config)
    else:
        run(config, cli.preflight_only)
