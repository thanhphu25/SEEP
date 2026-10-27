"""Kaggle setup utilities. Reuses only the verified DGCA CUB adapter/preparation.

No torch/torchvision installation and no representation enhancement modules.
"""
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NPZ_SHA = 'ac7478131e1b77e5c2f0225849730bd19ee89cfffe600c2dfd2272b34a50a43d'
NPZ_URL = 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz'
DGCA_SHA = 'ecec522a42518819d2504890b6c7b6fede6938c5aca7d5d47322c2cd6cc9996e'


def shim_timm_text(text):
    text = text.replace('from dataclasses import dataclass, replace\n',
                        'from dataclasses import dataclass, replace, field\n')
    for member, kind in (('conv_cfg', 'MaxxVitConvCfg'), ('transformer_cfg', 'MaxxVitTransformerCfg')):
        text = text.replace(f'{member}: {kind} = {kind}()\n',
                            f'{member}: {kind} = field(default_factory={kind})\n')
    return text


def dependencies(output):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', 'timm==0.6.11'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'loguru', 'pytest', 'pyyaml', 'kagglehub'])
    path = Path(importlib.metadata.distribution('timm').locate_file('timm/models/maxxvit.py'))
    old = path.read_text(encoding='utf8')
    new = shim_timm_text(old)
    assert shim_timm_text(new) == new
    if new != old:
        path.write_text(new, encoding='utf8')
    report = dict(file=str(path), before_sha256=hashlib.sha256(old.encode()).hexdigest(),
                  after_sha256=hashlib.sha256(new.encode()).hexdigest(), idempotent=True)
    (Path(output) / 'timm_compatibility.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    subprocess.check_call([sys.executable, '-c', "import timm,torch,torchvision; print(timm.__version__, torch.__version__, torchvision.__version__)"])


def find_official(base):
    return sorted({p.parent for p in Path(base).rglob('images.txt') if
        all((p.parent / name).is_file() for name in ('image_class_labels.txt', 'train_test_split.txt'))
        and (p.parent / 'images').is_dir()})


def prepare_data(source, ready, handle='trungkiennguyn/lgsp-dataset'):
    from tools.prepare_cub200 import prepare
    search_root = Path(source) if source is not None else Path('/kaggle/input')
    candidates = find_official(search_root)
    if not candidates:
        if source is None:
            import kagglehub
            try:
                downloaded = kagglehub.dataset_download(handle)
                candidates = find_official(downloaded)
            except Exception as exc:
                raise RuntimeError('Add Input ' + handle + ' (also for background Save Version). '
                                   'Official images.txt, image_class_labels.txt and train_test_split.txt are required.') from exc
        if not candidates:
            raise FileNotFoundError(f'No official CUB root found inside {search_root}. '
                'Expected images/ plus images.txt, image_class_labels.txt and train_test_split.txt '
                'in the same folder. Leave DATA_ROOT=None to search all Kaggle Inputs automatically.')
    if len(candidates) != 1:
        raise ValueError(f'Found {len(candidates)} official CUB roots. Set DATA_ROOT to one of: {candidates}')
    source = candidates[0]
    print('CUB root:', source, flush=True)
    manifest = prepare(source, ready)
    reference = json.loads((ROOT / 'references/align_only_seed3407/official_split_manifest.json').read_text(encoding='utf8'))
    if manifest['official_metadata_sha256'] != reference['official_metadata_sha256']:
        raise ValueError('Official metadata SHA256 differs from verified ZIP run')
    return manifest


def pretrained(path, output):
    from modules.frequency_state import file_sha, json_write
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.download')
        urllib.request.urlretrieve(NPZ_URL, temporary)
        if file_sha(temporary) != NPZ_SHA:
            raise ValueError('Downloaded NPZ SHA256 mismatch')
        temporary.replace(path)
    actual = file_sha(path)
    if actual != NPZ_SHA:
        raise ValueError('Pretrained NPZ SHA256 mismatch')
    json_write(Path(output) / 'pretrained_manifest.json', dict(path=str(path), url=NPZ_URL, sha256=actual))
    return path


def export(output, destination=None):
    output = Path(output)
    destination = Path(destination or output.parent / 'SEEP_FREQUENCY_STATE_CUB200_metrics.zip')
    # Explicit allowlist by suffix; no pickle/checkpoints, image data, or full model states.
    allowed = {'.json', '.csv', '.txt', '.log', '.patch', '.md', '.ipynb', '.xml'}
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix.lower() in allowed and 'checkpoint' not in path.relative_to(output).parts:
                archive.write(path, str(path.relative_to(output)).replace('\\', '/'))
    return destination
