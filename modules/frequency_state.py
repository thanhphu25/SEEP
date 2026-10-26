"""Task frequency recall only. No learned parameters, head fitting, or DGCA.

Bank slots are zero based; checkpoint boundaries and inventory entries are one based.
The bank deliberately stays outside nn.Module and optimizer state.
"""
import contextlib
import copy
import hashlib
import inspect
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

FIELDS = {'weights': (100,), 'frequency_residual_scale': (), 'alpha': (), 'beta': ()}
CHANNELS = ('live_raw_exact', 'snapshot_raw_exact', 'live_raw_seen', 'snapshot_raw_seen')
PROTOCOL = dict(seed=3407, model='vit_base_patch16_224', num_tasks=10, batch_size=32,
                shuffle=False, lgsp='YES', lgsp_type='LGSP', D1=56, D2=96, epochs=30,
                dataset='Split-CUB200', task_inc=False, num_r=100, input_size=224,
                opt='adam', sched='cosine')
PATH_OVERRIDES = {'data_path', 'output_dir', 'device', 'num_workers', 'pin_mem', 'eval',
                  'trial', 'print_freq', 'dist_url', 'distributed', 'world_size', 'gpu',
                  'subparser_name', 'num_tasks_to_run', 'historical_environment', 'effective_lr'}


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(path)


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update((name + str(value.dtype) + str(tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_from(source):
    state = source.state_dict() if hasattr(source, 'state_dict') else source
    result = {}
    for name, shape in FIELDS.items():
        value = state[name]
        if value.shape != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f'Invalid frequency tensor {name}: shape={value.shape}, dtype={value.dtype}')
        result[name] = value.detach().cpu().clone()
    return result


class FrequencyBank:
    def __init__(self):
        self._slots = {}
        self.provenance = {}

    def add(self, task, source, provenance):
        if type(task) is not int or not 0 <= task < 10 or task in self._slots:
            raise ValueError(f'Invalid/duplicate zero-based bank slot: {task}')
        self._slots[task] = state_from(source)
        self.provenance[task] = copy.deepcopy(provenance)

    def select(self, task, horizon, device):
        if type(task) is not int or not 0 <= task <= horizon < 10 or task not in self._slots:
            raise ValueError(f'Missing/future frequency slot {task}, horizon={horizon}')
        # Clones make the public accessor read-only even on CPU.
        return {k: v.to(device=device).clone() for k, v in self._slots[task].items()}

    def save(self, path):
        records = []
        for task, state in sorted(self._slots.items()):
            previous = self._slots.get(task - 1)
            records.append(dict(task_zero_based=task, boundary=task + 1,
                tensors={k: v.tolist() for k, v in state.items()}, sha256=tensor_hash(state),
                norms={k: float(v.norm()) for k, v in state.items()},
                delta_from_previous=None if previous is None else
                    {k: float((v - previous[k]).norm()) for k, v in state.items()},
                provenance=self.provenance[task]))
        json_write(path, dict(slots=records, tensor_values=103 * len(records),
                              tensor_bytes=412 * len(records), dtype='float32', device='cpu'))


@contextlib.contextmanager
def isolated_rng():
    python_state, numpy_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def load_checkpoint(path, trusted=False):
    kwargs = dict(map_location='cpu')
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = not trusted
    elif not trusted:
        raise ValueError('This PyTorch cannot safely inspect pickle: set trusted_checkpoints for your own files')
    try:
        payload = torch.load(path, **kwargs)
    except Exception as exc:
        raise ValueError('Cannot load checkpoint. Namespace pickle requires trusted_checkpoints=true '
                         'ONLY for your own trusted checkpoints. ' + str(path)) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get('model'), dict) or not any(k in payload for k in ('args', 'experiment_config')):
        raise ValueError('Expected SEEP/DGCA dictionary containing model and args: ' + str(path))
    raw = payload.get('args', payload.get('experiment_config'))
    args = copy.deepcopy(raw if isinstance(raw, dict) else vars(raw))
    state = payload['model']
    if payload.get('format') not in (None, 'SEEP_DGCA_v1'):
        raise ValueError('Unsupported/enhanced checkpoint format: ' + str(payload['format']))
    prefixes = [key.startswith('module.') for key in state]
    stripped = bool(prefixes) and all(prefixes)
    if any(prefixes) and not stripped:
        raise ValueError('Mixed DDP and non-DDP state keys')
    if stripped:
        state = {key[7:]: value for key, value in state.items()}
    forbidden = re.compile(r'(^|[._])(cls_steering|ccsr|n1)([._]|$)', re.I)
    if any(forbidden.search(key) for key in state):
        raise ValueError('Representation-enhanced checkpoint (CLS Steering/CCSR/N1) is not eligible')
    for key, value in args.items():
        if forbidden.search(key) and value not in (None, False, 0, '', 'off', 'none', []):
            raise ValueError('Representation enhancement flag enabled: ' + key)
    if args.get('dgca', 'off') not in ('off', 'none', None, 'align_only'):
        raise ValueError('Only original SEEP or DGCA align_only source checkpoints are supported')
    if 'dgca' in payload and payload['dgca'].get('config', {}).get('mode') != 'align_only':
        raise ValueError('DGCA payload must be explicitly align_only')
    if not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Non-tensor model state')
    return payload, state, args, stripped


def generators_in(state):
    ids = sorted({int(m.group(1)) for key in state
                  if (m := re.match(r'prompt_generators\.(\d+)\.', key))})
    if ids != list(range(len(ids))):
        raise ValueError('Non-contiguous CNN generators')
    return len(ids)


def dynamic_strict_load(model, state, boundary):
    count = generators_in(state)
    if count != boundary:
        raise ValueError(f'Boundary {boundary} requires {boundary} CNN generators, found {count}')
    if len(model.prompt_generators) > count:
        raise ValueError('Use a fresh model when loading an earlier boundary')
    with isolated_rng():
        model.ensure_prompt_generator(count - 1)
    prompt = state.get('stored_unified_prompts')
    if prompt is None or prompt.shape != (model.args.num_tasks, 3, model.args.input_size, model.args.input_size):
        raise ValueError('Missing/invalid runtime stored_unified_prompts shape')
    device = next(model.parameters()).device
    if 'stored_unified_prompts' not in model._buffers:
        delattr(model, 'stored_unified_prompts')
        model.register_buffer('stored_unified_prompts', torch.empty_like(prompt, device=device))
    model.load_state_dict(state, strict=True)


def config_signature(args):
    return {k: v for k, v in args.items() if k not in PATH_OVERRIDES
            and not k.startswith(('dgca', 'frequency_'))}


def validate_config(args):
    missing = [k for k in PROTOCOL if k not in args]
    if missing:
        raise ValueError('Cannot verify required saved args: ' + ', '.join(missing))
    bad = {k: (args[k], v) for k, v in PROTOCOL.items() if args[k] != v}
    if bad:
        raise ValueError('Saved protocol mismatch (actual, expected): ' + str(bad))
    if args.get('class_order', list(range(200))) != list(range(200)):
        raise ValueError('Class order differs from unshuffled 0..199 protocol')
    for key, expected in dict(lr_local=.0005, lr_Frequency_mask=.03, reinit_optimizer=True,
                              train_mask=True, pretrained=True).items():
        if key in args and args[key] != expected:
            raise ValueError(f'Saved protocol mismatch {key}={args[key]} expected {expected}')


def checkpoint_mapping(root, explicit=None):
    if explicit:
        data = json.loads(Path(explicit).read_text(encoding='utf8')) if isinstance(explicit, (str, Path)) else explicit
        mapping = {int(k): Path(v) for k, v in data.items()}
        if any(not 1 <= k <= 10 for k in mapping) or len(set(map(str, mapping.values()))) != len(mapping):
            raise ValueError('Mapping must use unique paths and one-based boundaries 1..10')
        return mapping
    candidates = {}
    for path in sorted(Path(root).rglob('*')):
        if path.is_file() and path.suffix.lower() in ('.pth', '.pt', '.ckpt'):
            m = re.search(r'task[_-]?(\d+)[_-]checkpoint', path.stem, re.I)
            if m and 1 <= int(m[1]) <= 10:
                candidates.setdefault(int(m[1]), []).append(path)
    duplicates = {k: list(map(str, v)) for k, v in candidates.items() if len(v) > 1}
    if duplicates:
        raise ValueError('Duplicate boundaries; provide explicit checkpoint_map JSON. ' + str(duplicates))
    return {k: v[0] for k, v in candidates.items()}


def inventory(root, output, explicit=None, trusted=False):
    output = Path(output)
    report = dict(status='inspecting', boundaries=[], unknown=['source commit/environment unless saved',
        'same-run identity unless saved; tensor/config consistency is checked'], full_trajectory=False)
    bank, reference, fixed, frozen, run_id = FrequencyBank(), None, None, None, None
    try:
        mapping = checkpoint_mapping(root, explicit)
        missing = [b for b in range(1, 11) if b not in mapping or not mapping[b].is_file()]
        report['missing_boundaries'] = missing
        for boundary in range(1, 11):
            path = mapping.get(boundary)
            entry = dict(boundary=boundary, path=str(path) if path else None, status='missing')
            report['boundaries'].append(entry)
            if path is None or not path.is_file():
                continue
            entry.update(size_bytes=path.stat().st_size, sha256=file_sha(path), status='checking')
            payload, state, args, stripped = load_checkpoint(path, trusted)
            # Drop the optimizer and any learned DGCA head immediately, retaining only model tensors.
            epoch = payload.get('epoch')
            dgca = payload.get('dgca', {})
            saved_boundary = payload.get('boundary', dgca.get('completed_boundary') if isinstance(dgca, dict) else None)
            identity = payload.get('run_id', args.get('run_id'))
            entry.update(format='dgca_align_only' if 'dgca' in payload else 'seep', epoch=epoch,
                         args=args, ddp_prefix_stripped=stripped, saved_boundary_zero_based=saved_boundary,
                         run_id=identity, source_commit=payload.get('source_commit', 'unknown'),
                         environment=payload.get('environment', 'unknown'))
            del payload
            validate_config(args)
            if 'lr' in args and abs(args['lr'] - .00375) > 1e-12:
                raise ValueError('Saved LR differs from .00375 after scaling; inspect original training config')
            entry['learning_rate_verification'] = 'verified saved scaled LR' if 'lr' in args else 'unknown'
            if epoch is not None and epoch != args['epochs'] - 1:
                raise ValueError(f'Boundary {boundary}: expected final epoch 29 (zero based), got {epoch}')
            if epoch is None:
                if entry['format'] != 'dgca_align_only' or saved_boundary != boundary - 1:
                    raise ValueError('Final epoch cannot be verified from checkpoint')
                entry['epoch_verification'] = 'unknown; DGCA completed_boundary marker only'
            if saved_boundary is not None and saved_boundary != boundary - 1:
                raise ValueError('Saved boundary disagrees with mapping')
            if generators_in(state) != boundary:
                raise ValueError('CNN generator count disagrees with mapped boundary')
            signature = config_signature(args)
            if reference is None:
                reference, run_id = signature, identity
            elif reference != signature or (run_id is not None and identity != run_id):
                raise ValueError('Mixed-run/config mismatch across checkpoints')
            entry['same_run_limitation'] = 'Tensor/config consistency cannot prove common training lineage without run identity'
            this_fixed = dict(radii=tensor_hash({'radii': state['radii']}), temperature=args.get('temperature', 1.0),
                              ring_beta=4.0, num_r=args['num_r'], input_size=args['input_size'])
            if state['radii'].shape != (100,) or state['radii'].dtype != torch.float32 or not torch.isfinite(state['radii']).all():
                raise ValueError('Invalid fixed radii tensor')
            prefixes = tuple(args.get('freeze', []))
            frozen_tensors = {k: v for k, v in state.items() if k.startswith(prefixes)}
            if not frozen_tensors:
                raise ValueError('No declared frozen backbone tensors to verify')
            this_frozen = tensor_hash(frozen_tensors)
            if fixed is not None and (fixed != this_fixed or frozen != this_frozen):
                raise ValueError('Fixed frequency config or declared frozen backbone changed')
            fixed, frozen = this_fixed, this_frozen
            entry.update(status='validated', frozen_backbone_sha256=frozen, fixed_frequency=this_fixed)
            bank.add(boundary - 1, state, dict(checkpoint=str(path), checkpoint_sha256=entry['sha256'], boundary=boundary))
            del state
        bank.save(output / 'frequency_bank.json')
        if missing:
            raise ValueError(f'Missing checkpoint boundaries: {missing}. Add the same-run Input or select train_collect explicitly. '
                             'A final checkpoint alone cannot recover old states; no automatic training.')
        report.update(status='validated', full_trajectory=True, same_run_identity='verified' if run_id is not None else 'unknown',
                      fixed_frequency=fixed, frozen_backbone_sha256=frozen)
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        json_write(output / 'checkpoint_inventory.json', report)
    return report, bank


def matrix_metrics(matrix):
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError('Expected square A[eval task, boundary]')
    n = len(a)
    if any(np.isfinite(a[i, :i]).any() for i in range(n)):
        raise ValueError('Future cells must be NaN')
    completed = [t for t in range(n) if np.isfinite(a[:t + 1, t]).all()]
    observed = int(np.isfinite(a).sum())
    full = completed == list(range(n)) and observed == n * (n + 1) // 2
    result = dict(full_matrix=full, observed_cells=observed, completed_boundaries=[t + 1 for t in completed],
                  aia=None, final_accuracy=None, final_forgetting=None, old_task_final_mean=None,
                  old_task_peak_mean=None, diagonal_initial_mean=None, bwt=None)
    if not full:
        return result
    peaks = np.array([a[i, i:].max() for i in range(n)])
    result.update(aia=float(np.mean([a[:t + 1, t].mean() for t in range(n)])),
                  final_accuracy=float(a[:, -1].mean()), final_forgetting=float((peaks[:-1] - a[:-1, -1]).mean()),
                  old_task_final_mean=float(a[:-1, -1].mean()), old_task_peak_mean=float(peaks[:-1].mean()),
                  diagonal_initial_mean=float(np.diag(a).mean()), bwt=float((a[:-1, -1] - np.diag(a)[:-1]).mean()))
    return result


@torch.no_grad()
def paired_evaluate(model, original, loaders, device, boundary, masks, bank, matrices, progress=print):
    before = dict(model=tensor_hash(model.state_dict()), original=tensor_hash(original.state_dict()))
    modes = [(m, m.training) for root in (model, original) for m in root.modules()]
    attrs = [(obj, key, hasattr(obj, key), getattr(obj, key, None))
             for obj in (model, getattr(model, 'rainbow_prompt', model))
             for key in ('task_id', 'matched_task_id', 'e_prefix_feature')]
    routes, diagnostics = [], []
    start, processed, total = time.perf_counter(), 0, sum(len(loaders[i]['val'].dataset) for i in range(boundary + 1))
    try:
        with isolated_rng():
            model.eval()
            original.eval()
            seen = torch.zeros(model.head.out_features, dtype=torch.bool, device=device)
            seen[sum((list(masks[i]) for i in range(boundary + 1)), [])] = True
            for task in range(boundary + 1):
                loader = loaders[task]['val']
                if loader.batch_size != 32 or loader.drop_last or not isinstance(loader.sampler, torch.utils.data.SequentialSampler):
                    raise ValueError('Evaluation requires per-task sequential batches of 32, drop_last=False')
                correct = {name: 0 for name in CHANNELS}
                count, route_counts, max_logit, max_feature = 0, {}, 0., 0.
                for batch, (images, target) in enumerate(loader):
                    images, target = images.to(device), target.to(device)
                    cls = original(images)['pre_logits']
                    live = model(images, task_id=-1, learned_id=boundary, cls_features=cls, train=False)
                    route = model.matched_task_id
                    snap = model(images, task_id=-1, learned_id=boundary, cls_features=cls, train=False, frequency_bank=bank)
                    if route != model.matched_task_id or not 0 <= route <= boundary:
                        raise AssertionError('Paired route disagreement or future route')
                    for branch, out in (('live', live), ('snapshot', snap)):
                        logits = out['logits']
                        if not torch.isfinite(logits).all() or not torch.isfinite(out['classifier_features']).all():
                            raise ValueError('Non-finite paired output')
                        correct[branch + '_raw_exact'] += int((logits.argmax(1) == target).sum())
                        correct[branch + '_raw_seen'] += int((logits.masked_fill(~seen[None], -torch.inf).argmax(1) == target).sum())
                    max_logit = max(max_logit, float((live['logits'] - snap['logits']).abs().max()))
                    max_feature = max(max_feature, float((live['classifier_features'] - snap['classifier_features']).abs().max()))
                    route_counts[route] = route_counts.get(route, dict(batches=0, images=0))
                    route_counts[route]['batches'] += 1
                    route_counts[route]['images'] += len(target)
                    count += len(target)
                    processed += len(target)
                    if batch % 10 == 0 or count == len(loader.dataset):
                        elapsed = time.perf_counter() - start
                        progress(f'boundary={boundary+1}/10 eval_task={task+1} processed={count}/{len(loader.dataset)} '
                                 f'elapsed={elapsed:.1f}s ETA_boundary={elapsed*(total-processed)/max(processed,1):.1f}s')
                if count != len(loader.dataset) or not count:
                    raise ValueError('Incomplete/empty validation task')
                for name in CHANNELS:
                    if np.isfinite(matrices[name][task, boundary]):
                        raise ValueError('Refusing to overwrite previously committed metric')
                    matrices[name][task, boundary] = 100. * correct[name] / count
                routes.append(dict(boundary=boundary + 1, eval_task=task + 1, count=count, routes=route_counts))
                diagnostics.append(dict(boundary=boundary + 1, eval_task=task + 1,
                                        max_abs_logit_delta=max_logit, max_abs_classifier_feature_delta=max_feature))
    finally:
        for module, mode in modes:
            module.training = mode
        for obj, key, exists, value in attrs:
            if exists:
                setattr(obj, key, value)
            elif hasattr(obj, key):
                delattr(obj, key)
    after = dict(model=tensor_hash(model.state_dict()), original=tensor_hash(original.state_dict()))
    check = dict(boundary=boundary + 1, before=before, after=after, passed=before == after,
                 restored_modes=True, restored_routing=True)
    if not check['passed']:
        raise AssertionError('Paired evaluation mutated checkpoint/router/cache/head tensors')
    return routes, diagnostics, check
