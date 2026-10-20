"""Read-only CUB official metadata or already split ImageFolder trees."""
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class ReadOnlyCUB(Dataset):
    def __init__(self, root, train=True, transform=None):
        root = Path(root)
        if (root / 'CUB_200_2011').is_dir():
            root = root / 'CUB_200_2011'
        self.transform = transform
        self.target_transform = None
        self.samples, self.sample_ids = [], []
        if (root / 'images.txt').is_file() and (root / 'train_test_split.txt').is_file():
            paths = dict(line.strip().split(maxsplit=1) for line in (root / 'images.txt').read_text().splitlines())
            split = dict(line.split() for line in (root / 'train_test_split.txt').read_text().splitlines())
            self.classes = sorted({p.split('/')[0] for p in paths.values()})
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            for key in sorted(paths, key=int):
                if int(split[key]) == int(train):
                    relative = paths[key]
                    # Some prepared inputs retain metadata after moving images.
                    path = root / 'images' / relative
                    if not path.is_file():
                        path = root / ('train' if train else 'test') / relative
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    self.samples.append((str(path), self.class_to_idx[relative.split('/')[0]]))
                    self.sample_ids.append(relative)
        elif (root / 'train').is_dir() and (root / 'test').is_dir():
            self.classes = sorted(p.name for p in (root / 'train').iterdir() if p.is_dir())
            test_classes = sorted(p.name for p in (root / 'test').iterdir() if p.is_dir())
            if self.classes != test_classes:
                raise ValueError('Train/test class directories must share ImageFolder mapping')
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            partition = root / ('train' if train else 'test')
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.ppm', '.pgm', '.tif', '.tiff', '.webp'}
            for cls in self.classes:
                for path in sorted((partition / cls).rglob('*')):
                    if path.suffix.lower() in extensions:
                        self.samples.append((str(path), self.class_to_idx[cls]))
                        self.sample_ids.append(path.relative_to(partition).as_posix())
        else:
            raise FileNotFoundError('CUB requires images.txt + train_test_split.txt + images/, or train/ and test/: ' + str(root))
        # Match ImageFolder's class/filename ordering for equivalent layouts.
        ordered = sorted(zip(self.sample_ids, self.samples))
        self.sample_ids = [sample_id for sample_id, _ in ordered]
        self.samples = [sample for _, sample in ordered]
        self.targets = [y for _, y in self.samples]
        self.imgs = self.samples
        if not self.samples:
            raise ValueError('Empty CUB partition')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
