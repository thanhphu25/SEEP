"""Build a small latest-branch Kaggle notebook and an optional offline source patch."""
import difflib
import hashlib
import json
import subprocess
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = '80081081ab42769d102f83d447705da0639e2fb1'
SOURCES = ['.gitattributes', '.gitignore', 'vision_transformer.py', 'engine.py', 'datasets.py',
    'continual_datasets/continual_datasets.py', 'modules/frequency_state.py',
    'tools/frequency_setup.py', 'tools/frequency_run.py', 'tools/prepare_cub200.py',
    'tools/frequency_notebook.py',
    'tools/build_frequency_delivery.py', 'tools/verify_frequency_delivery.py',
    'tests/test_frequency_state.py', 'README_FREQUENCY_STATE.md',
    'references/align_only_seed3407/provenance.json',
    'references/align_only_seed3407/official_split_manifest.json',
    'references/align_only_seed3407/source_manifest.json']


def cell(kind, source, **metadata):
    value = dict(cell_type=kind, metadata=metadata,
                 source=textwrap.dedent(source).strip().splitlines(keepends=True))
    if kind == 'code':
        value.update(execution_count=None, outputs=[])
    return value


def build():
    patch = ''
    hashes = {}
    for name in SOURCES:
        new = (ROOT / name).read_bytes().replace(b'\r\n', b'\n')
        hashes[name] = hashlib.sha256(new).hexdigest()
        base = subprocess.run(['git', 'show', f'{BASE}:{name}'], cwd=ROOT, capture_output=True)
        old = base.stdout.replace(b'\r\n', b'\n') if base.returncode == 0 else b''
        if old == new:
            continue
        patch += f'diff --git a/{name} b/{name}\n'
        if base.returncode:
            patch += 'new file mode 100644\n'
        lines = difflib.unified_diff(old.decode().splitlines(keepends=True), new.decode().splitlines(keepends=True),
                                    fromfile=f'a/{name}' if base.returncode == 0 else '/dev/null', tofile=f'b/{name}')
        for line in lines:
            patch += line if line.endswith('\n') else line + '\n\\ No newline at end of file\n'
    patch = patch.encode()
    sha = hashlib.sha256(patch).hexdigest()
    manifest = dict(base_commit=BASE, branch='feat/task-frequency-state', patch_sha256=sha,
                    source_files_sha256=hashes, serialization='LF', source_mode='latest branch HEAD at runtime; offline patch for local verification',
                    runtime_repo_url='https://github.com/thanhphu25/SEEP.git', runtime_branch='feat/task-frequency-state',
                    reused_dgca_patch_sha256='ecec522a42518819d2504890b6c7b6fede6938c5aca7d5d47322c2cd6cc9996e')
    cells = [cell('markdown', """
        # SEEP CUB200 — Run All

        1. Bật **GPU + Internet** trong Kaggle Settings.
        2. **Add Input** bộ `trungkiennguyn/lgsp-dataset` (CUB200).
        3. Chọn **Run All**. Không cần sửa đường dẫn hay gắn checkpoint đầu vào.

        Notebook tự clone commit mới nhất của nhánh `feat/task-frequency-state`,
        tìm thư mục CUB kể cả khi lồng nhiều cấp, train **10 task × 30 epoch**
        và lưu checkpoint từng task trong output. Mỗi lần Run All tạo thư mục
        mới, giữ lại các kết quả cũ. GPU full training chưa được chạy sẵn.
    """), cell('code', """
        # 1. Lấy code mới nhất đúng nhánh; không checkout commit cũ hay apply patch.
        import subprocess, sys, tempfile
        from pathlib import Path
        SOURCE_URL = 'https://github.com/thanhphu25/SEEP.git'
        SOURCE_BRANCH = 'feat/task-frequency-state'
        DATA_ROOT = None  # Tự tìm CUB trong Kaggle Inputs, kể cả CUB_200_2011/CUB_200_2011.
        SAVE_CHECKPOINTS = True
        WORK = Path('/kaggle/working')
        REPO = Path(tempfile.mkdtemp(prefix='SEEP_Frequency_State_', dir=WORK))
        subprocess.check_call(['git', 'clone', '--single-branch', '--branch', SOURCE_BRANCH, SOURCE_URL, str(REPO)])
        sys.path.insert(0, str(REPO))
        for name in ('tools.frequency_notebook', 'tools.frequency_setup'):
            sys.modules.pop(name, None)
        from tools.frequency_notebook import KaggleRun
        run = KaggleRun(REPO, data_root=DATA_ROOT, save_checkpoints=SAVE_CHECKPOINTS, work=WORK)
    """), cell('code', """
        # 2. Cài phụ thuộc, tự tìm dữ liệu và kiểm thử. Có thể chạy lại cell này nếu setup lỗi.
        run.setup()
    """), cell('code', """
        # 3. Train đầy đủ 10 × 30 epoch; lưu checkpoint và đánh giá sau từng task.
        run.train()
    """), cell('code', """
        # 4. Kết quả và link tải metrics. Checkpoint nằm riêng trong output/checkpoint/.
        from IPython.display import display, FileLink
        display(FileLink(str(run.results())))
    """)]
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python'), kaggle=dict(isGpuEnabled=True, isInternetEnabled=True)), nbformat=4, nbformat_minor=5)
    import ast
    for i, c in enumerate(cells):
        c['id'] = f'frequency-{i:02d}'
        if c['cell_type'] == 'code':
            ast.parse(''.join(c['source']))
    (ROOT / 'frequency_state_implementation.patch').write_bytes(patch)
    (ROOT / 'frequency_state_delivery.json').write_bytes((json.dumps(manifest, indent=2) + '\n').encode('utf8'))
    (ROOT / 'SEEP_Frequency_State_CUB200_Kaggle.ipynb').write_bytes((json.dumps(notebook, ensure_ascii=False, indent=1) + '\n').encode('utf8'))
    with zipfile.ZipFile(ROOT / 'SEEP_FREQUENCY_STATE_CUB200_delivery.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ['SEEP_Frequency_State_CUB200_Kaggle.ipynb', 'frequency_state_implementation.patch',
                     'frequency_state_delivery.json', 'README_FREQUENCY_STATE.md']:
            z.write(ROOT / name, name)
    print(json.dumps(dict(patch_sha256=sha, files=len(SOURCES), cells=len(cells))))


if __name__ == '__main__':
    build()
