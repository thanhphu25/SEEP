"""Build the standalone notebook from committed sources, with no old outputs."""
import base64
import hashlib
import json
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_dgca_delivery import SOURCES as DGCA_SOURCES, BASE

SOURCES = sorted(set(DGCA_SOURCES + [
    '.gitattributes', '.gitignore', 'attention.py', 'README_CLS_STEERING.md',
    'modules/cls_steering.py', 'modules/cls_diagnostics.py', 'modules/cls_pretrained.py',
    'tools/summarize_cls_steering.py', 'tools/build_cls_delivery.py',
    'tools/verify_cls_delivery.py', 'tests/test_cls_steering.py']
    + [str(p.relative_to(ROOT)).replace('\\', '/') for p in (ROOT/'references/align_only_seed3407').glob('*')]))


def cell(kind, text):
    source = textwrap.dedent(text).strip()+'\n'
    result = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        result.update(outputs=[], execution_count=None)
    return result


def build():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    source_hashes = {}
    for name in SOURCES:
        blob = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
        if blob.replace(b'\r\n', b'\n') != (ROOT/name).read_bytes().replace(b'\r\n', b'\n'):
            raise ValueError('Commit source before building: '+name)
        source_hashes[name] = hashlib.sha256(blob).hexdigest()
    reference = json.loads((ROOT/'references/align_only_seed3407/provenance.json').read_text())
    audited = (ROOT/'references/align_only_seed3407/dgca_implementation.patch').read_bytes()
    assert hashlib.sha256(audited).hexdigest() == reference['dgca_patch_sha256']
    # The mathematical alignment and feature collection modules must be intact.
    for name, sha in {'modules/dgca.py': '0580058d687cba11745ce69f72560126aa90a7585bcd043b20d53c9f79100acd',
                      'modules/dgca_data.py': '3ef5c119c4e2193d4d96eaddd116927ec926bed8dfdba10031ac3739dc64095e'}.items():
        assert source_hashes[name] == sha, name
    patch = subprocess.check_output(['git', 'diff', '--binary', BASE, commit, '--', *SOURCES], cwd=ROOT)
    manifest = {'base_commit': BASE, 'implementation_source_commit': commit,
                'branch': 'feat/cls-steering-cil', 'patch_sha256': hashlib.sha256(patch).hexdigest(),
                'audited_dgca_patch_sha256': reference['dgca_patch_sha256'],
                'source_files_sha256': source_hashes, 'reference': reference,
                'delivery_mode': 'embedded_combined_patch_on_pinned_base',
                'local_full_cub_run': False, 'local_gpu_available': False}
    cells = [cell('markdown', '''
        # SEEP: task-specific CLS steering — CUB200, seed 3407

        Bật **một GPU** và **Internet**, dùng **Add Input → trungkiennguyn/lgsp-dataset**
        trước **Save Version**, rồi Run All. Notebook có thể tự gắn dataset qua
        kagglehub trong session tương tác. Không cần upload source hay push branch.

        Một run mới: 10 task × 30 epoch SEEP + 5 epoch align-only mỗi task.
        CLS Q/K/V tại block 9/10/11; không transport hay module/loss khác.
        Raw_seen (C) và align-only (D) dùng cùng feature; A/B lấy từ ZIP đã audit.
        Mốc baseline lịch sử giữ riêng. Ước lượng cũ T4 khoảng 3 giờ chỉ để dự trù;
        notebook đo thời gian thực. Các tests CPU không phải kết quả full CUB.

        Ý tưởng additive CLS Q/K/V: [CASP §4.1](https://arxiv.org/html/2601.16773v1),
        Huang, Lin, Lu (2026); adaptation task-specific cho SEEP CIL, chưa xác lập novelty.
        Routing gốc chọn một task cho mỗi batch. Không dùng test để chọn LR/layer.
    '''), cell('code', '''
        import os, sys, json, subprocess, hashlib, base64, shutil, zipfile, time
        from pathlib import Path
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use one device even on Kaggle T4 x2.
        import torch, torchvision

        REPO = Path('/kaggle/working/SEEP_CLS_STEERING')
        READY = Path('/kaggle/working/cub200_ready')
        OUTPUT = Path('/kaggle/working/seep_cls_steering_align_only_seed3407')
        PREFLIGHT = Path('/kaggle/working/cls_steering_preflight')
        DATASET_HANDLE = 'trungkiennguyn/lgsp-dataset'
        DATA_ROOT = None  # Chỉ cần điền nếu Input chứa nhiều official CUB roots.
        RESUME = False
        RESUME_FROM = None  # Optional: thư mục output cũ, gồm checkpoint/dgca_latest.pth.
        assert torch.cuda.is_available(), 'Bật GPU trong Settings của notebook'
        assert torch.cuda.device_count() == 1, 'Chọn một GPU cho protocol này'
        assert not OUTPUT.exists() or RESUME, 'Output đã tồn tại; chỉ dùng RESUME cho chính run này.'
        if RESUME_FROM is not None:
            assert RESUME and not OUTPUT.exists()
            shutil.copytree(Path(RESUME_FROM), OUTPUT)
        print('Python:', sys.version, 'Torch:', torch.__version__, 'Torchvision:', torchvision.__version__)
        print('GPU:', torch.cuda.get_device_name(0))
        print('Historical stack match:', torch.__version__ == '2.11.0+cu128' and torchvision.__version__ == '0.26.0+cu128')
        print('Khác environment không được gọi là exact reproduction.')
        PREFLIGHT.mkdir(parents=True, exist_ok=True)
    '''), cell('code', '''
        # Giữ nguyên torch/torchvision Kaggle; không cài requirements.txt.
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', 'timm==0.6.11'])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'loguru', 'pytest', 'pyyaml', 'kagglehub'])
        import importlib.metadata
        maxxvit = Path(importlib.metadata.distribution('timm').locate_file('timm/models/maxxvit.py'))
        original = maxxvit.read_text()
        fixed = original.replace('from dataclasses import dataclass, replace\\n',
                                 'from dataclasses import dataclass, replace, field\\n')
        fixed = fixed.replace('conv_cfg: MaxxVitConvCfg = MaxxVitConvCfg()\\n',
                              'conv_cfg: MaxxVitConvCfg = field(default_factory=MaxxVitConvCfg)\\n')
        fixed = fixed.replace('transformer_cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg()\\n',
                              'transformer_cfg: MaxxVitTransformerCfg = field(default_factory=MaxxVitTransformerCfg)\\n')
        maxxvit.write_text(fixed)
        TIMM_COMPAT = {'file': str(maxxvit), 'before_sha256': hashlib.sha256(original.encode()).hexdigest(),
                       'after_sha256': hashlib.sha256(fixed.encode()).hexdigest(), 'idempotent': True}
        subprocess.check_call([sys.executable, '-c', "import timm; assert timm.__version__ == '0.6.11'; print('timm OK')"])
    '''), cell('code', f'''
        BASE_COMMIT = {BASE!r}
        PATCH_B64 = {base64.b64encode(patch).decode()!r}
        DELIVERY = {manifest!r}
        PATCH_SHA256 = DELIVERY['patch_sha256']
        patch = base64.b64decode(PATCH_B64)
        assert hashlib.sha256(patch).hexdigest() == PATCH_SHA256
        if not REPO.exists():
            subprocess.check_call(['git', 'clone', '--config', 'core.autocrlf=false', 'https://github.com/thanhphu25/SEEP.git', str(REPO)])
            subprocess.check_call(['git', 'checkout', '--detach', BASE_COMMIT], cwd=REPO)
            subprocess.check_call(['git', 'switch', '-c', 'feat/cls-steering-cil'], cwd=REPO)
            patch_file = REPO/'cls_steering_implementation.patch'
            patch_file.write_bytes(patch)
            subprocess.check_call(['git', 'apply', '--check', str(patch_file)], cwd=REPO)
            subprocess.check_call(['git', 'apply', str(patch_file)], cwd=REPO)
        else:
            assert RESUME, 'Repo đã tồn tại; dùng session mới hoặc RESUME cho chính run này'
            assert (REPO/'cls_steering_implementation.patch').read_bytes() == patch
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO).decode().strip() == BASE_COMMIT
        for name, expected in DELIVERY['source_files_sha256'].items():
            assert hashlib.sha256((REPO/name).read_bytes()).hexdigest() == expected, name
        (REPO/'cls_steering_delivery.json').write_text(json.dumps(DELIVERY, indent=2))
        (REPO/'timm_compatibility.json').write_text(json.dumps(TIMM_COMPAT, indent=2))
        print('Verified source:', DELIVERY['implementation_source_commit'], 'patch:', PATCH_SHA256)
    '''), cell('code', '''
        def find_cub(base):
            base = Path(base)
            return sorted({p.parent for p in base.rglob('images.txt')
                           if all((p.parent/n).is_file() for n in ('image_class_labels.txt', 'train_test_split.txt'))
                           and (p.parent/'images').is_dir()}) if base.exists() else []
        if DATA_ROOT is None:
            candidates = find_cub('/kaggle/input')
            if not candidates:
                import kagglehub
                try:
                    attached = kagglehub.dataset_download(DATASET_HANDLE)
                except Exception as exc:
                    raise RuntimeError('Dùng Add Input để gắn '+DATASET_HANDLE+' trước Save Version, rồi chạy lại.') from exc
                candidates = find_cub(attached) or find_cub('/kaggle/input')
            print('CUB candidates:', [str(p) for p in candidates])
            assert len(candidates) == 1, 'Cần đúng một official CUB root; đặt DATA_ROOT trong cell cấu hình.'
            DATA_ROOT = candidates[0]
        subprocess.check_call([sys.executable, 'tools/prepare_cub200.py', '--source', str(DATA_ROOT),
                               '--output', str(READY)], cwd=REPO)
        official = json.loads((READY/'official_split_manifest.json').read_text())
        reference_official = json.loads((REPO/'references/align_only_seed3407/official_split_manifest.json').read_text())
        assert official['official_metadata_sha256'] == reference_official['official_metadata_sha256']
        assert (official['image_count'], official['train'], official['test'], official['classes']) == (11788, 5994, 5794, 200)
        print(json.dumps(official, indent=2))
    '''), cell('code', '''
        ARGS = ['--seed', '3407', '--batch-size', '32', '--epochs', '30', '--num_tasks', '10',
                '--num_tasks_to_run', '0', '--lgsp', 'YES', '--lgsp_type', 'LGSP',
                '--data-path', str(READY)+'/', '--output_dir', str(OUTPUT), '--dgca', 'align_only',
                '--dgca-shrinkage', '0.1', '--dgca-samples-per-class', '128', '--dgca-epochs', '5', '--dgca-lr', '0.01',
                '--cls-steering', 'task', '--cls-steering-layers', '9', '10', '11', '--cls-steering-lr', '0.0005']
        RUN_ENV = os.environ.copy()
        RUN_ENV['SEEP_PRETRAINED_SHA256'] = DELIVERY['reference']['pretrained_sha256']
        def run_logged(command, log_path):
            start = time.perf_counter()
            with Path(log_path).open('w') as log:
                process = subprocess.Popen(command, cwd=REPO, env=RUN_ENV, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout:
                    print(line, end='')
                    log.write(line)
                    log.flush()
                returncode = process.wait()
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, command)
            return {'returncode': returncode, 'seconds': time.perf_counter()-start, 'command': command}
        TEST_LOG = PREFLIGHT/'tests.log'
        TEST_RESULT = run_logged([sys.executable, '-m', 'pytest', 'tests/test_dgca.py', 'tests/test_cls_steering.py',
                                  '-q', '--junitxml='+str(PREFLIGHT/'tests.xml')], TEST_LOG)
        (PREFLIGHT/'test_result.json').write_text(json.dumps(TEST_RESULT, indent=2))
    '''), cell('code', '''
        PREFLIGHT_RESULT = run_logged([sys.executable, 'tools/preflight_dgca.py', *ARGS,
                                       '--report-dir', str(PREFLIGHT)], PREFLIGHT/'preflight.log')
        resolved = json.loads((PREFLIGHT/'resolved_config.json').read_text())
        assert resolved['dgca'] == 'align_only' and resolved['cls_steering'] == 'task'
        assert resolved['effective_lr'] == .00375 and resolved['cls_steering_lr'] == .0005
        assert resolved['cls_steering_layers'] == [9, 10, 11]
        assert resolved['num_tasks_to_run'] == 0 and resolved['task_inc'] is False
        assert json.loads((PREFLIGHT/'split_manifest.json').read_text())['sha256'] == DELIVERY['reference']['split_sha256']
        weights = json.loads((PREFLIGHT/'pretrained_manifest.json').read_text())
        assert weights['sha256'] == DELIVERY['reference']['pretrained_sha256'] and weights['tensor_load_verified']
        # Separate process: exactly two current-task batches, no HP/accuracy selection.
        SMOKE_RESULT = run_logged([sys.executable, 'tools/smoke_dgca.py', *ARGS], PREFLIGHT/'smoke.log')
        (PREFLIGHT/'checks.json').write_text(json.dumps({'tests': TEST_RESULT, 'preflight': PREFLIGHT_RESULT,
                                                      'gpu_smoke': SMOKE_RESULT}, indent=2))
    '''), cell('markdown', '''
        Full run duy nhất. Khi tiếp tục checkpoint, dùng `RESUME=True` trong cấu hình;
        ở session mới, `RESUME_FROM` trỏ đến thư mục output của chính run này đã gắn vào Input.
        Checkpoint transport/align-only cũ bị từ chối. Không resume giữa epoch.
    '''), cell('code', '''
        if RESUME:
            assert (OUTPUT/'checkpoint/dgca_latest.pth').is_file(), 'Chưa có task-boundary checkpoint của run này'
        else:
            assert not OUTPUT.exists(), 'Không ghi chồng run'
        STREAM_LOG = PREFLIGHT/'full_stream.log'
        RUN_RESULT = run_logged([sys.executable, '-u', 'main.py', 'cub200_10task', *ARGS,
                                 *(['--dgca-resume'] if RESUME else [])], STREAM_LOG)
        (PREFLIGHT/'full_run_result.json').write_text(json.dumps(RUN_RESULT, indent=2))
    '''), cell('code', '''
        # Chỉ gói metrics sau một full run thành công. Không yêu cầu raw C bằng A.
        run_logged([sys.executable, 'tools/summarize_cls_steering.py', str(OUTPUT)], PREFLIGHT/'summary.log')
        summary = json.loads((OUTPUT/'summary.json').read_text())
        assert summary['full_protocol_complete'] and summary['completed_tasks'] == 10
        assert json.loads((OUTPUT/'run_status.json').read_text())['status'] == 'success'
        source = json.loads((OUTPUT/'source_manifest.json').read_text())
        assert source['diff_sha256'] == PATCH_SHA256
        assert source['cached_weights_sha256']['ViT-B_16.npz'] == DELIVERY['reference']['pretrained_sha256']
        assert json.loads((OUTPUT/'split_manifest.json').read_text())['sha256'] == DELIVERY['reference']['split_sha256']
        for name, expected in DELIVERY['source_files_sha256'].items():
            assert hashlib.sha256((REPO/name).read_bytes()).hexdigest() == expected, name
        reports = [json.loads(p.read_text()) for p in sorted(OUTPUT.glob('dgca_task_*.json'))]
        assert len(reports) == 10
        assert all(r['mode'] == 'align_only' and r['transport'] is None and r['hashes_before'] == r['hashes_after'] for r in reports)
        print(json.dumps({'comparisons_pp': summary['comparisons_pp'], 'screening_pass': summary['screening_pass'],
                          'interpretation': summary['interpretation']}, indent=2))
        BUNDLE = Path('/kaggle/working/SEEP_CLS_STEERING_CUB200_metrics.zip')
        with zipfile.ZipFile(BUNDLE, 'w', zipfile.ZIP_DEFLATED) as archive:
            for p in sorted(OUTPUT.iterdir()):
                if p.is_file() and p.suffix in {'.json', '.csv', '.txt', '.log', '.patch'}:
                    archive.write(p, 'metrics/'+p.name)
            for p in sorted(PREFLIGHT.iterdir()):
                if p.is_file():
                    archive.write(p, 'preflight/'+p.name)
            for name in ('cls_steering_delivery.json', 'cls_steering_implementation.patch', 'timm_compatibility.json'):
                archive.write(REPO/name, 'source/'+name)
            for p in sorted((REPO/'references/align_only_seed3407').iterdir()):
                archive.write(p, 'reference/'+p.name)
            archive.write(READY/'official_split_manifest.json', 'data/official_split_manifest.json')
            notebook = Path('/kaggle/working/__notebook__.ipynb')
            if notebook.is_file():
                archive.write(notebook, 'notebook/executed.ipynb')
        from IPython.display import display, FileLink
        display(FileLink(str(BUNDLE)))
        print('Checkpoint lớn, tải riêng:')
        display(FileLink(str(OUTPUT/'checkpoint/dgca_latest.pth')))
    ''')]
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                'language_info': {'name': 'python'}, 'kaggle': {'isGpuEnabled': True, 'isInternetEnabled': True}},
                'nbformat': 4, 'nbformat_minor': 5}
    for i, c in enumerate(notebook['cells']):
        c['id'] = f'cls-steering-{i:02d}'
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), f'<cell {i}>', 'exec')
    out = ROOT/'artifacts/cls_steering'
    out.mkdir(parents=True, exist_ok=True)
    path = ROOT/'notebooks/SEEP_CLS_Steering_CUB200_Kaggle.ipynb'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
    (out/'cls_steering_implementation.patch').write_bytes(patch)
    (ROOT/'cls_steering_delivery.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    with zipfile.ZipFile(out/'SEEP_CLS_STEERING_Kaggle_delivery.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(path, path.name)
        z.writestr('cls_steering_implementation.patch', patch)
        z.write(ROOT/'cls_steering_delivery.json', 'cls_steering_delivery.json')
        z.write(ROOT/'README_CLS_STEERING.md', 'README_CLS_STEERING.md')
        for name in ('delivery_validation.json', 'reconstructed_tests.log', 'reconstructed_tests.xml'):
            if (out/name).is_file():
                z.write(out/name, 'local_validation/'+name)
    print(json.dumps({'source_commit': commit, 'patch_sha256': manifest['patch_sha256'], 'notebook': str(path)}, indent=2))


if __name__ == '__main__':
    build()
