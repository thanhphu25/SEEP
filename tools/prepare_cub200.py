"""Validate all CUB-200-2011 official records and build SEEP ImageFolder trees."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def read_records(path):
    result = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        key, value = line.strip().split(maxsplit=1)
        key = int(key)
        if key in result:
            raise ValueError(f'Duplicate id {key}: {path}')
        result[key] = value
    return result


def validate_official(root):
    root = Path(root)
    images = read_records(root/'images.txt')
    labels = {i: int(v) for i, v in read_records(root/'image_class_labels.txt').items()}
    split = {i: int(v) for i, v in read_records(root/'train_test_split.txt').items()}
    expected = set(range(1, 11789))
    if set(images) != expected or set(labels) != expected or set(split) != expected:
        raise ValueError('Official metadata must contain matching ids 1..11788')
    if set(labels.values()) != set(range(1, 201)) or set(split.values()) != {0, 1}:
        raise ValueError('Expected 200 classes and binary official split')
    if sum(split.values()) != 5994:
        raise ValueError('Expected official counts: train=5994/test=5794')
    if len(set(images.values())) != 11788:
        raise ValueError('Duplicate image paths')
    for i, relative in images.items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or len(path.parts) != 2:
            raise ValueError('Invalid image path: '+relative)
        if int(path.parts[0].split('.')[0]) != labels[i]:
            raise ValueError(f'Label/path mismatch for image {i}')
        if not (root/'images'/path).is_file():
            raise FileNotFoundError(root/'images'/path)
    if len({Path(p).parts[0] for p in images.values()}) != 200:
        raise ValueError('Expected exactly 200 class directories')
    metadata = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                for name in ('images.txt', 'image_class_labels.txt', 'train_test_split.txt')}
    return images, labels, split, metadata


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    images, labels, split, metadata = validate_official(root)
    manifest = {'source': str(root), 'image_count': len(images), 'train': 5994, 'test': 5794,
                'classes': 200, 'official_metadata_sha256': metadata}
    output.mkdir(parents=True, exist_ok=True)
    target_root = output/'CUB_200_2011'
    target_root.mkdir(exist_ok=True)
    for i, relative in images.items():
        target = target_root/('train' if split[i] else 'test')/relative
        source = root/'images'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.resolve() != source.resolve():
                raise FileExistsError(f'Existing file is not the official source: {target}')
        else:
            target.symlink_to(source)
    expected = {str(Path('train' if split[i] else 'test')/path) for i, path in images.items()}
    actual = {str(path.relative_to(target_root)) for scope in ('train', 'test')
              for path in (target_root/scope).rglob('*') if path.is_file()}
    if actual != expected:
        raise ValueError('Prepared split includes missing/extra files')
    for name in metadata:
        shutil.copyfile(root/name, target_root/name)
    (output/'official_split_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
