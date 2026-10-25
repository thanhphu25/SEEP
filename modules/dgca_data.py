"""Independent current-train view with stable underlying ImageFolder indices."""
import copy
import hashlib

import torch
from torch.utils.data import Dataset, Subset, DataLoader, SequentialSampler

from modules.dgca import isolated_rng, finite, model_hashes


def resolve_indices(dataset):
    indices = list(range(len(dataset)))
    while isinstance(dataset, Subset):
        indices = [int(dataset.indices[i]) for i in indices]
        dataset = dataset.dataset
    return dataset, indices


class CurrentTrainView(Dataset):
    def __init__(self, current_train, transform):
        base, self.ids = resolve_indices(current_train)
        # ImageFolder metadata and loader are read-only; a private dataset object
        # owns its transforms. Never set transform on the shared training base.
        self.dataset = copy.copy(base)
        self.dataset.transform = copy.deepcopy(transform)
        self.dataset.target_transform = copy.deepcopy(getattr(base, 'target_transform', None))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        stable_id = self.ids[index]
        image, target = self.dataset[stable_id]
        return image, target, stable_id


def current_loader(current_train, transform, seed, workers=0, pin_memory=False):
    view = CurrentTrainView(current_train, transform)
    return DataLoader(view, sampler=SequentialSampler(view), batch_size=32, drop_last=False,
                      num_workers=workers, pin_memory=pin_memory,
                      generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def collect_features(model, original_model, loader, device, learned_id, max_batches=None):
    hashes = model_hashes(model)
    modes = [(module, module.training) for root in (model, original_model) if root is not None
             for module in root.modules()]
    # Routing updates plain Python attributes even in eval. Restore them too.
    routing = [(obj, name, hasattr(obj, name), getattr(obj, name, None))
               for obj in (model, getattr(model, 'rainbow_prompt', model))
               for name in ('task_id', 'matched_task_id')]
    features, labels, ids = [], [], []
    image_hash = hashlib.sha256()
    try:
        with isolated_rng():
            model.eval()
            if original_model is not None:
                original_model.eval()
            for batch, (images, target, stable_id) in enumerate(loader):
                if max_batches is not None and batch >= max_batches:
                    break
                image_hash.update(images.contiguous().numpy().tobytes())
                images = images.to(device)
                cls = original_model(images)['pre_logits'] if original_model is not None else None
                output = model(images, task_id=-1, learned_id=learned_id, cls_features=cls, train=False)
                features.append(finite(output['classifier_features'], 'classifier features').detach().cpu())
                labels.append(target.cpu())
                ids.append(stable_id.cpu())
    finally:
        for module, mode in modes:
            module.training = mode
        for obj, name, existed, value in routing:
            if existed:
                setattr(obj, name, value)
            elif hasattr(obj, name):
                delattr(obj, name)
    if model_hashes(model) != hashes:
        raise RuntimeError('Feature collection mutated representation/head/stored prompts')
    if not features:
        raise ValueError('Empty current task')
    return {'features': torch.cat(features), 'labels': torch.cat(labels), 'ids': torch.cat(ids),
            'images_sha256': image_hash.hexdigest()}


def verify_pairs(before, after):
    for key in ('ids', 'labels'):
        if not torch.equal(before[key], after[key]):
            raise ValueError('Before/after pairing differs: '+key)
    if before['images_sha256'] != after['images_sha256']:
        raise ValueError('Before/after transformed images differ')
