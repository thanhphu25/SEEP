"""Resolve the full protocol, official split and actual pretrained weights before GPU training."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torchvision
import timm
from timm.models import create_model
from timm.models.hub import download_cached_file

import models
from configs.cub200_10task import get_args_parser
from datasets import build_continual_dataloader
from main import set_seed
from modules.dgca_runtime import split_manifest, BASE_COMMIT, write_json


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def preflight(args, report_dir):
    assert torch.cuda.is_available(), 'Enable GPU in Kaggle settings'
    assert timm.__version__ == '0.6.11'
    assert args.num_tasks == 10 and args.epochs == 30 and args.num_tasks_to_run == 0
    assert args.task_inc is False and args.pretrained is True and args.batch_size == 32
    assert args.seed == 3407 and args.D1 == 56 and args.D2 == 96
    assert args.lgsp == 'YES' and args.lgsp_type == 'LGSP'
    assert args.dgca in ('transport', 'align_only') and args.lr == .03
    if args.cls_steering == 'task':
        assert args.dgca == 'align_only'
        assert args.cls_steering_layers == [9, 10, 11] and args.cls_steering_lr == .0005
        assert args.dgca_shrinkage == .1 and args.dgca_samples_per_class == 128
        assert args.dgca_epochs == 5 and args.dgca_lr == .01
        assert args.shuffle is False and args.train_mask is True
        assert args.lr_local == .0005 and args.lr_Frequency_mask == .03
        assert args.model == 'vit_base_patch16_224' and args.sched == 'cosine' and args.opt == 'adam'
    args.distributed = False
    set_seed(args.seed)
    loaders, masks = build_continual_dataloader(args)
    assert args.nb_classes == 200
    split = split_manifest(loaders, masks)
    assert sum(len(task['train']) for task in split['tasks']) == 5994
    assert sum(len(task['val']) for task in split['tasks']) == 5794
    # Use the repository-registered model and its resolved cfg, exactly as main.
    model = create_model(args.model, pretrained=True, num_classes=200,
                         drop_rate=args.drop, drop_path_rate=args.drop_path, drop_block_rate=None)
    cfg = model.pretrained_cfg
    url = cfg.get('url', '')
    assert url == 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz', cfg
    path = Path(download_cached_file(url, check_hash=True, progress=True))
    assert path.name == Path(urlparse(url).path).name and path.stat().st_size > 100_000_000
    weights_hash = file_sha256(path)
    expected = os.environ.get('SEEP_PRETRAINED_SHA256')
    if expected and weights_hash != expected:
        raise ValueError('Pretrained SHA256 differs from provided baseline checksum')
    # Verify tensors were actually loaded; timm can warn and return random init
    # when a pretrained source/custom loader is missing.
    with np.load(path) as weights:
        prefix = 'opt/target/' if 'opt/target/embedding/kernel' in weights else ''
        kernel = torch.from_numpy(weights[prefix+'embedding/kernel'].transpose(3, 2, 0, 1))
        torch.testing.assert_close(model.patch_embed.proj.weight.detach().cpu(), kernel, rtol=0, atol=0)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir/'resolved_config.json', {**vars(args), 'effective_lr': args.lr*32/256})
    write_json(report_dir/'split_manifest.json', split)
    root = Path(__file__).resolve().parents[1]
    write_json(report_dir/'pretrained_manifest.json', {'source': url, 'file': path.name,
               'sha256': weights_hash, 'tensor_load_verified': True, 'expected_sha256_provided': bool(expected),
               'base_commit': BASE_COMMIT,
               'checkout_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
               'delivery': json.loads((root/('cls_steering_delivery.json' if args.cls_steering == 'task' else 'dgca_delivery.json')).read_text())})
    versions = {'python': sys.version, 'torch': torch.__version__, 'torchvision': torchvision.__version__,
                'timm': timm.__version__, 'gpu': torch.cuda.get_device_name(),
                'historical_torch': '2.11.0+cu128', 'historical_torchvision': '0.26.0+cu128',
                'historical_stack_match': torch.__version__ == '2.11.0+cu128' and torchvision.__version__ == '0.26.0+cu128'}
    write_json(report_dir/'versions.json', versions)
    (report_dir/'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode(), encoding='utf-8')
    print(json.dumps({'config': vars(args), 'effective_lr': args.lr*32/256, 'versions': versions,
                      'weights_sha256': weights_hash, 'split_sha256': split['sha256'], 'class_mask': masks}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--report-dir', type=Path, default=Path('/kaggle/working/dgca_preflight'))
    get_args_parser(parser)
    args = parser.parse_args()
    directory = args.report_dir
    del args.report_dir
    preflight(args, directory)
