"""DGCA: current-data residual transport and a separate Gaussian-aligned head.

Row vectors are used throughout. No SEEP model is registered inside this object.
Statistics are raw, full CPU float32 covariance matrices; sampling shrinkage is
applied only to temporary factors. See README_DGCA.md for method attribution.
"""
import copy
import hashlib
import math
import random
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


@contextmanager
def isolated_rng(seed=None):
    state = capture_rng()
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        yield
    finally:
        restore_rng(state)


def tensor_hash(tensor):
    tensor = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(str((tensor.dtype, tuple(tensor.shape))).encode())
    h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(tensor_hash(value).encode())
    return h.hexdigest()


def model_hashes(model):
    state = model.state_dict()
    return {'representation': state_hash({k: v for k, v in state.items() if not k.startswith('head.')}),
            'raw_head': state_hash(model.head.state_dict()),
            'stored_prompts': state_hash({k: v for k, v in state.items() if 'stored_' in k})}


def finite(x, name):
    if not torch.isfinite(x).all():
        raise FloatingPointError('Non-finite ' + name)
    return x


def peak_process_rss_bytes():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == 'darwin' else value*1024)
    except ImportError:
        try:
            import psutil
            return int(psutil.Process().memory_info().peak_wset)
        except (ImportError, AttributeError):
            return None


def fit_residual_ridge(x, y, ridge=1.0):
    x = finite(x.detach().to(device='cpu', dtype=torch.float64), 'X')
    y = finite(y.detach().to(device='cpu', dtype=torch.float64), 'Y')
    if x.shape != y.shape or x.ndim != 2 or len(x) < 2 or ridge < 0:
        raise ValueError('Expected paired [N,d] features, N>=2 and nonnegative ridge')
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    gram = xc.T @ xc
    identity = torch.eye(x.shape[1], dtype=torch.float64)
    lam = ridge * torch.trace(gram) / x.shape[1] + 1e-8
    system = gram + lam * identity
    a = identity + torch.linalg.solve(system, xc.T @ (yc - xc))
    b = ym - xm @ a
    finite(a, 'transport A')
    finite(b, 'transport b')
    return a, b, {'ridge_lambda': float(lam), 'condition_number': float(torch.linalg.cond(system)),
                  'relative_residual_frobenius': float(torch.linalg.norm(a-identity) / math.sqrt(x.shape[1])),
                  'translation_norm': float(torch.linalg.norm(b)),
                  'current_drift_rmse': float((y-x).square().mean().sqrt()),
                  'current_fit_rmse': float((x @ a + b-y).square().mean().sqrt())}


def transport_diagnostics(x, y, labels, ridge, seed):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(x), generator=generator)
    cut = min(len(x)-1, max(2, int(.8*len(x))))
    fit, holdout = order[:cut], order[cut:]
    a_hold, b_hold, _ = fit_residual_ridge(x[fit], y[fit], ridge)
    xd, yd = x[holdout].double(), y[holdout].double()
    a, b, diagnostics = fit_residual_ridge(x, y, ridge)
    diagnostics.update({'holdout_count': len(holdout), 'fit_count': len(fit),
                        'holdout_identity_rmse': float((xd-yd).square().mean().sqrt()),
                        'holdout_transport_rmse': float((xd @ a_hold+b_hold-yd).square().mean().sqrt()),
                        'per_class_current_drift_rmse': {
                            str(int(c)): float((y[labels == c]-x[labels == c]).square().mean().sqrt())
                            for c in labels.unique()}})
    return a, b, diagnostics


def covariance_factor(covariance, count):
    """One CPU eigh per old class/boundary; restore known empirical rank.

    At most count-1 positive eigenvalues can be real signal. Remaining float32
    eigenvalues are roundoff. Their removal and all negative clamps are logged.
    """
    cov = finite(covariance.double().cpu(), 'covariance')
    cov = (cov + cov.T) * .5
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    rank = min(len(eigenvalues), count-1)
    positive = eigenvalues[-rank:].clamp_min(0)
    factor = eigenvectors[:, -rank:] * positive.sqrt()
    corrected = factor @ factor.T
    diagnostics = {'min_eigenvalue': float(eigenvalues.min()), 'rank_bound': rank,
                   'psd_correction_frobenius': float(torch.linalg.norm(corrected-cov)),
                   'removed_roundoff_positive_trace': float(eigenvalues[:-rank].clamp_min(0).sum())}
    return factor, diagnostics


def masked_seen(logits, labels):
    labels = torch.as_tensor(labels, dtype=torch.long, device=logits.device)
    result = torch.full_like(logits, float('-inf'))
    result[:, labels] = logits[:, labels]
    return result


def sample_gaussian(stat, factor, n, shrinkage, generator):
    """CPU low-rank plus isotropic sampler for the specified full covariance."""
    d = stat['mean'].numel()
    average_variance = float(torch.trace(stat['covariance']) / d)
    eps = max(1e-8, 1e-6*average_variance)
    sample = torch.randn(n, factor.shape[1], generator=generator) @ factor.T
    sample *= math.sqrt(1-shrinkage)
    sample += torch.randn(n, d, generator=generator) * math.sqrt(shrinkage*average_variance+eps)
    return sample + stat['mean']


class DGCA:
    def __init__(self, mode='transport', ridge=1., shrinkage=.1, samples_per_class=128,
                 epochs=5, lr=.01, seed=3407):
        if mode not in ('transport', 'align_only'):
            raise ValueError(mode)
        if not 0 <= shrinkage <= 1 or samples_per_class < 1 or epochs < 1 or lr <= 0 or ridge < 0:
            raise ValueError('Invalid DGCA configuration')
        self.config = dict(mode=mode, ridge=ridge, shrinkage=shrinkage,
                           samples_per_class=samples_per_class, epochs=epochs, lr=lr, seed=seed)
        self.stats = {}
        self.alignment_head = None
        self.completed_boundary = -1

    def state_dict(self):
        return {'config': self.config.copy(), 'stats': self.stats,
                'completed_boundary': self.completed_boundary,
                'alignment_head': None if self.alignment_head is None else
                {k: v.detach().cpu() for k, v in self.alignment_head.state_dict().items()}}

    def load_state_dict(self, state, raw_head):
        if state['config'] != self.config:
            raise ValueError('DGCA config differs from checkpoint')
        self.stats = state['stats']
        self.completed_boundary = state['completed_boundary']
        if state['alignment_head'] is not None:
            self.alignment_head = copy.deepcopy(raw_head)
            self.alignment_head.load_state_dict(state['alignment_head'], strict=True)
            self.alignment_head.eval().requires_grad_(False)

    def boundary(self, task, raw_head, after, labels, before=None):
        if task <= self.completed_boundary:
            raise ValueError('Boundary already committed; refusing repeated transport')
        if task != self.completed_boundary+1:
            raise ValueError('Boundaries must be consecutive')
        if not isinstance(raw_head, nn.Linear):
            raise TypeError('DGCA requires a linear raw classifier')
        finite(after, 'current features')
        if after.ndim != 2 or len(after) != len(labels) or after.shape[1] != raw_head.in_features:
            raise ValueError('Invalid classifier features / labels')
        if set(int(c) for c in labels.unique()) & set(self.stats):
            raise ValueError('Current classes overlap old statistics')
        start = time.perf_counter()
        with isolated_rng(self.config['seed'] + 100003*task):
            report = {'task': task, 'mode': self.config['mode'], 'transport': None}
            a = b = None
            if task and self.config['mode'] == 'transport':
                if before is None or before.shape != after.shape or len(after) < 4:
                    raise ValueError('Transport needs paired before/after current features')
                a, b, report['transport'] = transport_diagnostics(
                    before, after, labels, self.config['ridge'], self.config['seed']+task)
            factors, corrections, new_stats = {}, {}, {}
            # Work transactionally: a failed boundary does not partially transport.
            for label, stat in sorted(self.stats.items()):
                factor, corrections[str(label)] = covariance_factor(stat['covariance'], stat['count'])
                mu = stat['mean'].double()
                if a is not None:
                    factor = a.T @ factor
                    mu = mu @ a + b
                covariance = factor @ factor.T
                new_stats[label] = {'mean': finite(mu.float(), 'transported mean'),
                                    'covariance': finite(((covariance+covariance.T)*.5).float(), 'transported covariance'),
                                    'count': stat['count']}
                factors[label] = factor.float()
            for label_tensor in labels.unique(sorted=True):
                label = int(label_tensor)
                features = after[labels == label].double().cpu()
                if len(features) < 2:
                    raise ValueError('Unbiased class covariance requires >=2 examples')
                mu = features.mean(0)
                factor = (features-mu).T / math.sqrt(len(features)-1)
                covariance = factor @ factor.T
                new_stats[label] = {'mean': mu.float(), 'covariance': covariance.float(), 'count': len(features)}
                factors[label] = factor.float()
            report['factor_and_transport_seconds'] = time.perf_counter()-start
            report['covariance_corrections'] = corrections
            report['class_counts'] = {str(k): v['count'] for k, v in new_stats.items()}
            report['statistics_bytes'] = sum(s['mean'].numel()*4+s['covariance'].numel()*4 for s in new_stats.values())
            report['factor_bytes'] = sum(f.numel()*f.element_size() for f in factors.values())
            head, losses, alignment_seconds = self._align(raw_head, new_stats, factors, task)
            self.stats = new_stats
            self.alignment_head = head
            self.completed_boundary = task
            report.update(alignment_loss=losses, alignment_seconds=alignment_seconds,
                          total_seconds=time.perf_counter()-start,
                          peak_process_rss_bytes=peak_process_rss_bytes())
            return report

    def _align(self, raw_head, stats, factors, task):
        start = time.perf_counter()
        head = copy.deepcopy(raw_head).requires_grad_(True)
        head.train()
        device = head.weight.device
        labels = sorted(stats)
        seen = torch.tensor(labels, device=device)
        generator = torch.Generator(device='cpu').manual_seed(self.config['seed']+7919*(task+1))
        optimizer = torch.optim.SGD(head.parameters(), lr=self.config['lr'], momentum=.9, weight_decay=0.)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'], eta_min=0.)
        losses = []
        n = self.config['samples_per_class']
        # All full covariances remain on CPU. Low rank + isotropic noise is an
        # exact sampler for (1-s)*Sigma + s*trace(Sigma)/d*I + eps*I.
        for _ in range(self.config['epochs']):
            samples, targets = [], []
            for local_label, label in enumerate(labels):
                stat, factor = stats[label], factors[label]
                samples.append(sample_gaussian(stat, factor, n, self.config['shrinkage'], generator))
                targets.append(torch.full((n,), local_label, dtype=torch.long))
            samples, targets = torch.cat(samples), torch.cat(targets)
            order = torch.randperm(len(samples), generator=generator)
            total = 0.
            for indices in order.split(256):
                optimizer.zero_grad(set_to_none=True)
                logits = head(samples[indices].to(device))[:, seen]
                loss = F.cross_entropy(logits, targets[indices].to(device))
                finite(loss, 'alignment loss')
                loss.backward()
                optimizer.step()
                total += loss.item()*len(indices)
            losses.append(total/len(samples))
            scheduler.step()
        return head.eval().requires_grad_(False), losses, time.perf_counter()-start
