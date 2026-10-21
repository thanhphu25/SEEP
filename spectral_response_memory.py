"""N1: classwise response Gram memory in differentiable LGSP coordinates.

No image replay, no inference hooks, no external implementation copied.
"""
import copy
import math
import random
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
from loguru import logger

from srm_config import srm_config

VERSION = 1
EPS = 1e-8


def frequency_transform(x, radii, p):
    """Source FFT arithmetic, including soft rings and abs(complex IFFT)."""
    fft_im = torch.fft.fftn(x, dim=(-2, -1))
    fft_im_center = torch.fft.fftshift(fft_im, dim=(-2, -1))
    h, w = x.shape[-2:]
    y, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    distances = torch.sqrt((y - h // 2) ** 2 + (xx - w // 2) ** 2).to(x.device)
    rings = []
    radii = radii.to(x.device)
    for i, radius in enumerate(radii):
        inner = torch.tensor(0.0, device=x.device) if i == 0 else radii[i - 1] + 1e-6
        rings.append((torch.sigmoid(-4.0 * (distances - radius))
                      - torch.sigmoid(-4.0 * (distances - inner))).float())
    ring_masks = torch.stack(rings, dim=0).to(x.device)
    final_mask = (p[:, None, None] * ring_masks).sum(dim=0)
    fft_selected = fft_im_center * final_mask[None, None, :, :]
    fft_residual = fft_im_center + fft_selected
    inverse = torch.fft.ifftshift(fft_residual, dim=(-2, -1))
    return torch.abs(torch.fft.ifftn(inverse, dim=(-2, -1)))


def coordinates(model):
    p = torch.softmax(model.weights * getattr(model.args, 'temperature', 1.0), dim=0)
    return torch.cat((p, (model.alpha + model.beta).reshape(1),
                      (model.beta * model.frequency_residual_scale).reshape(1)))


def global_response(model, x, q=None):
    q = coordinates(model) if q is None else q
    return q[-2] * x + q[-1] * (frequency_transform(x, model.radii, q[:-2]) - x)


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


@contextmanager
def isolated_probe(model, reference):
    """Preserve heterogeneous module modes, parameter flags, RNG and routing attrs."""
    rng = capture_rng()
    modules = list(model.modules()) + list(reference.modules())
    modes = [(m, m.training) for m in modules]
    flags = [(p, p.requires_grad) for p in reference.parameters()]
    attrs = [(m, key, getattr(m, key)) for m in modules
             for key in ('matched_task_id', 'task_id') if hasattr(m, key)]
    try:
        reference.eval()
        for p, _ in flags:
            p.requires_grad_(False)
        with torch.enable_grad(), torch.autocast(device_type=next(model.parameters()).device.type,
                                                enabled=False):
            yield
    finally:
        for m, mode in modes:
            m.training = mode
        for p, flag in flags:
            p.requires_grad_(flag)
        for m, key, value in attrs:
            setattr(m, key, value)
        restore_rng(rng)


class ProbeView(Dataset):
    """Independent deterministic transform on *training* ImageFolder samples."""
    def __init__(self, dataset, transform):
        ids = list(range(len(dataset)))
        while isinstance(dataset, Subset):
            ids = [int(dataset.indices[i]) for i in ids]
            dataset = dataset.dataset
        self.ids = ids
        self.base = copy.copy(dataset)
        self.base.transform = transform
        self.targets = [int(dataset.targets[i]) for i in ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        image, label = self.base[self.ids[i]]
        return image, label, self.ids[i]


def jacobian_rows(fn, q):
    """Reverse-mode r VJPs, releasing graph on the last row; no higher derivatives."""
    h = fn(q).reshape(-1)
    rows = [torch.autograd.grad(h[j], q, retain_graph=j + 1 < h.numel(),
                                create_graph=False)[0] for j in range(h.numel())]
    return torch.stack(rows).detach()


def normalize_gram(gram, normalization):
    gram = (gram + gram.T) * 0.5
    if not torch.isfinite(gram).all():
        raise ValueError('Non-finite response Gram')
    eigmin = torch.linalg.eigvalsh(gram.double()).min().item()
    if eigmin < -1e-6 * max(1.0, gram.norm().item()):
        raise ValueError(f'Response Gram not PSD within tolerance: {eigmin}')
    trace = gram.trace().item()
    if trace <= EPS:
        logger.warning(f'N1 near-zero Gram trace={trace}; storing zero matrix')
        return torch.zeros_like(gram)
    return gram / max(trace, EPS) if normalization == 'trace' else gram


class SpectralResponseMemory:
    def __init__(self, args, feature_dim, device):
        self.config = srm_config(args)
        self.mode = self.config['srm_mode']
        self.device = torch.device(device)
        self.completed_task = -1
        self.R = None
        if self.mode.startswith('semantic_'):
            gen = torch.Generator(device='cpu').manual_seed(self.config['srm_memory_seed'])
            r = self.config['srm_proj_dim']
            self.R = (torch.randint(0, 2, (r, feature_dim), generator=gen).float() * 2 - 1) / math.sqrt(r)
        self.class_ids = []
        self.anchors = None
        self.matrices = None
        self.counts = []
        self.metadata = []

    def penalty(self, model):
        q = coordinates(model)
        if self.anchors is None:
            return q.sum() * 0.0
        delta = q.unsqueeze(0) - self.anchors
        if self.mode == 'semantic_diag':
            return (delta.square() * self.matrices.diagonal(dim1=-2, dim2=-1)).sum(-1).mean()
        return torch.einsum('ci,cij,cj->c', delta, self.matrices, delta).mean()

    def begin_task(self, model, task_id):
        if self.mode == 'freeze_spectral' and task_id > 0:
            model.weights.requires_grad_(False)
            model.frequency_residual_scale.requires_grad_(False)
            # An existing optimizer can retain an old gradient; do not let it step.
            model.weights.grad = None
            model.frequency_residual_scale.grad = None

    @property
    def memory_bytes(self):
        tensors = (self.R, self.anchors, self.matrices)
        # Serialized IDs/counts are int64 tensors as well.
        return sum(t.numel() * t.element_size() for t in tensors if t is not None) + 16 * len(self.class_ids)

    def collect(self, model, reference, train_dataset, class_ids, transform, task_id):
        if task_id != self.completed_task + 1:
            raise ValueError('Task boundary out of order / duplicate collection')
        if self.mode in ('off', 'freeze_spectral'):
            self.completed_task = task_id
            return dict(seconds=0.0, memory_bytes=self.memory_bytes, peak_cuda_bytes=0)
        if set(class_ids) & set(self.class_ids):
            raise ValueError('Refusing to overwrite old-class memory')
        if any(p.dtype != torch.float32 for p in reference.parameters()):
            raise ValueError('N1 reference must be float32')
        started = time.perf_counter()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        new_anchors, new_grams, new_counts = [], [], []
        records = []
        with isolated_probe(model, reference):
            view = ProbeView(train_dataset, transform)
            if set(view.targets) != set(map(int, class_ids)):
                raise ValueError('Collector dataset/classes must be exactly the current train task')
            anchor = coordinates(model).detach().float()
            R = self.R.to(self.device) if self.R is not None else None
            for c in map(int, class_ids):
                candidates = [i for i, label in enumerate(view.targets) if label == c]
                gen = torch.Generator().manual_seed(self.config['srm_memory_seed'] + 1000003 * c)
                order = torch.randperm(len(candidates), generator=gen).tolist()
                selected = [candidates[i] for i in order[:self.config['srm_samples_per_class']]]
                if not selected:
                    raise ValueError(f'No training samples for class {c}')
                gram = torch.zeros(anchor.numel(), anchor.numel(), device=self.device)
                jac_norms, selected_ids = [], []
                raw_trace = None
                if self.mode == 'l2':
                    gram = torch.eye(anchor.numel(), device=self.device) / anchor.numel()
                    selected_ids = [view.ids[i] for i in selected]
                else:
                    loader_gen = torch.Generator().manual_seed(self.config['srm_memory_seed'] + c)
                    loader = DataLoader(Subset(view, selected), shuffle=False, num_workers=0,
                                        batch_size=self.config['srm_probe_batch_size'], generator=loader_gen)
                    for images, labels, ids in loader:
                        # Never differentiate a batch mean: each J_x contributes separately.
                        for image, label, sample_id in zip(images, labels, ids):
                            if int(label) != c:
                                raise ValueError('Probe class mismatch')
                            x = image.unsqueeze(0).to(self.device, dtype=torch.float32)
                            leaf = anchor.clone().detach().requires_grad_(True)
                            def probe(q):
                                features = reference(global_response(model, x, q))['pre_logits']
                                return F.normalize(features, dim=-1) @ R.T
                            J = jacobian_rows(probe, leaf)
                            gram += J.T @ J
                            jac_norms.append(J.norm().item())
                            selected_ids.append(int(sample_id))
                    gram = gram / len(selected)
                    raw_trace = gram.trace().item()
                    gram = normalize_gram(gram, self.config['srm_normalize'])
                new_anchors.append(anchor.clone())
                new_grams.append(gram.detach())
                new_counts.append(len(selected))
                records.append(dict(task=task_id, class_id=c, selected_train_ids=selected_ids,
                                    jacobian_norm_mean=float(np.mean(jac_norms)) if jac_norms else None,
                                    jacobian_norm_min=min(jac_norms) if jac_norms else None,
                                    jacobian_norm_max=max(jac_norms) if jac_norms else None,
                                    raw_gram_trace=raw_trace,
                                    gram_trace=gram.trace().item()))
        # Commit memory only after the complete collector succeeds.
        anchors, grams = torch.stack(new_anchors), torch.stack(new_grams)
        self.anchors = anchors if self.anchors is None else torch.cat((self.anchors, anchors))
        self.matrices = grams if self.matrices is None else torch.cat((self.matrices, grams))
        self.class_ids.extend(map(int, class_ids))
        self.counts.extend(new_counts)
        self.metadata.extend(records)
        self.completed_task = task_id
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        stats = dict(seconds=time.perf_counter() - started, memory_bytes=self.memory_bytes,
                     peak_cuda_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else 0,
                     scalar_vjps=sum(new_counts) * self.config['srm_proj_dim'] if R is not None else 0,
                     classes=records)
        logger.info(f'N1 collector: {stats}')
        return stats

    def state_dict(self):
        return dict(version=VERSION, config=self.config, completed_task=self.completed_task,
                    R=self.R, anchors=self.anchors.cpu() if self.anchors is not None else None,
                    matrices=self.matrices.cpu() if self.matrices is not None else None,
                    class_ids=torch.tensor(self.class_ids, dtype=torch.int64),
                    counts=torch.tensor(self.counts, dtype=torch.int64), metadata=self.metadata)

    def load_state_dict(self, state):
        required = {'version', 'config', 'completed_task', 'R', 'anchors', 'matrices', 'class_ids', 'counts', 'metadata'}
        if not required <= state.keys() or state['version'] != VERSION or state['config'] != self.config:
            raise ValueError('Missing/incompatible N1 memory version or configuration')
        R = state['R']
        if (R is None) != (self.R is None) or (R is not None and not torch.equal(R, self.R)):
            raise ValueError('N1 projection mismatch')
        self.completed_task = state['completed_task']
        self.anchors = state['anchors'].to(self.device) if state['anchors'] is not None else None
        self.matrices = state['matrices'].to(self.device) if state['matrices'] is not None else None
        self.class_ids = state['class_ids'].tolist()
        self.counts = state['counts'].tolist()
        self.metadata = state['metadata']
        n = len(self.class_ids)
        if len(set(self.class_ids)) != n or len(self.counts) != n or any(c <= 0 for c in self.counts):
            raise ValueError('Invalid N1 class/count state')
        if self.mode not in ('off', 'freeze_spectral') and self.completed_task >= 0 and not n:
            raise ValueError('Completed N1 checkpoint has no memory')
        if n:
            if self.anchors is None or self.matrices is None or self.anchors.shape[0] != n:
                raise ValueError('Invalid N1 memory shapes')
            d = self.anchors.shape[1]
            if self.matrices.shape != (n, d, d) or not torch.isfinite(self.anchors).all():
                raise ValueError('Invalid N1 anchors/matrices')
            for gram in self.matrices:
                normalize_gram(gram, 'none')


def gradient_diagnostics(model, classification_loss, weighted_penalty, previous_q=None):
    """Low frequency autograd.grad calls do not accumulate or modify .grad."""
    params = [p for p in (model.weights, model.alpha, model.beta, model.frequency_residual_scale)
              if p.requires_grad]
    def norm(loss):
        if not loss.requires_grad:
            return 0.0
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        return math.sqrt(sum(g.detach().square().sum().item() for g in grads if g is not None))
    q = coordinates(model).detach()
    p = q[:-2]
    stats = dict(entropy=-(p * p.clamp_min(1e-30).log()).sum().item(),
                 a=q[-2].item(), b=q[-1].item(), s=model.frequency_residual_scale.item(),
                 alpha=model.alpha.item(), beta=model.beta.item(),
                 classification_grad_norm=norm(classification_loss),
                 weighted_regularizer_grad_norm=norm(weighted_penalty))
    if previous_q is not None:
        delta = q - previous_q.to(q)
        stats.update(delta_p=delta[:-2].norm().item(), delta_a=delta[-2].item(), delta_b=delta[-1].item())
    return stats
