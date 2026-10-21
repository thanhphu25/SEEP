"""Prepare official CUB splits by copy/symlink; never move/delete source files."""
import argparse
import shutil
from pathlib import Path, PurePosixPath


def prepare(source, output, method='copy'):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not (source / 'images.txt').is_file():
        source = source / 'CUB_200_2011'
    source = source.resolve()
    destination = output / 'CUB_200_2011'
    if destination == source or source in destination.parents:
        raise ValueError('Output must be separate from the raw source')
    images = {}
    for line in (source / 'images.txt').read_text().splitlines():
        key, relative = line.split(maxsplit=1)
        part = PurePosixPath(relative)
        if part.is_absolute() or '..' in part.parts or len(part.parts) != 2:
            raise ValueError(f'Invalid image path {relative}')
        if key in images:
            raise ValueError('Duplicate image id')
        images[key] = relative
    splits = dict(line.split() for line in (source / 'train_test_split.txt').read_text().splitlines())
    if images.keys() != splits.keys() or set(splits.values()) != {'0', '1'}:
        raise ValueError('Official split metadata mismatch')
    for key, relative in images.items():
        src = source / 'images' / relative
        if not src.is_file():
            raise ValueError(f'Missing {src}')
        dst = destination / ('train' if splits[key] == '1' else 'test') / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            if (method == 'symlink' and dst.resolve() == src.resolve()) or (
                    method == 'copy' and dst.is_file() and dst.read_bytes() == src.read_bytes()):
                continue
            raise ValueError(f'Refusing to overwrite {dst}')
        if method == 'symlink':
            dst.symlink_to(src)
        else:
            shutil.copy2(src, dst)
    print(f'Prepared official train/test splits at {destination}; source retained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--method', choices=('copy', 'symlink'), default='copy')
    opts = parser.parse_args()
    prepare(opts.source, opts.output_root, opts.method)
