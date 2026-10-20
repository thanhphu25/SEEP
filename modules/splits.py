import copy
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader


def stratified_indices(targets, fraction, seed):
    rng = np.random.RandomState(seed)
    targets = np.asarray(targets)
    train, dev = [], []
    for cls in sorted(set(targets.tolist())):
        indices = np.flatnonzero(targets == cls)
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * fraction))) if fraction else 0
        if count >= len(indices):
            raise ValueError('Each class needs at least one training sample after holdout')
        dev.extend(indices[:count].tolist())
        train.extend(indices[count:].tolist())
    return sorted(train), sorted(dev)


def deterministic_subset(dataset, transform):
    if isinstance(dataset, Subset):
        return Subset(deterministic_subset(dataset.dataset, transform), list(dataset.indices))
    cloned = copy.copy(dataset)
    cloned.transform = transform
    return cloned


def prototype_loader(dataset, transform, args, task):
    generator = torch.Generator().manual_seed(args.split_seed + 104729 + task)
    return DataLoader(deterministic_subset(dataset, transform), batch_size=args.batch_size,
                      shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem,
                      generator=generator)
