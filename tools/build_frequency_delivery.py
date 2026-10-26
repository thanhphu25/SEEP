"""Build a standalone Kaggle notebook from LF source files against the pinned base."""
import base64
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
                    source_files_sha256=hashes, serialization='LF', source_mode='embedded patch; no remote feature branch required',
                    reused_dgca_patch_sha256='ecec522a42518819d2504890b6c7b6fede6938c5aca7d5d47322c2cd6cc9996e')
    payload = base64.b64encode(patch).decode()
    cells = [cell('markdown', '''
        # SEEP frequency-state recall — CUB200, seed 3407

        Enable **GPU + Internet**. Add Input `trungkiennguyn/lgsp-dataset` and
        **your original SEEP task1..task10 checkpoints from the same run**.
        Attach Inputs before background Save Version. The supplied metrics ZIP
        contains no checkpoints. Reeval runs first; missing inputs never trigger training.
        No CLS Steering / N1 / CCSR / Gaussian alignment / transport is used.
        Local validation is synthetic; this notebook has no precomputed CUB result.
    '''), cell('code', '''
        import sys, os, json, hashlib, base64, subprocess, shutil, time, traceback
        from pathlib import Path
        RUN_MODE = 'reeval'  # Alternative: 'train_collect' starts a NEW full 10 x 30 run.
        CHECKPOINT_ROOT = '/kaggle/input'
        CHECKPOINT_MAP = None  # JSON path or dict {"1": "/kaggle/input/my-run/task1_checkpoint.pth", ... "10": ...}
        TRUSTED_CHECKPOINTS = False  # True ONLY for your own trusted Namespace/pickle checkpoints.
        DATA_ROOT = None  # Set official CUB root here if discovery finds duplicates.
        REPO = Path('/kaggle/working/SEEP_Frequency_State')
        READY = Path('/kaggle/working/cub200_ready_frequency_state')
        OUTPUT = Path('/kaggle/working/seep_frequency_state_seed3407')
        NPZ_PATH = Path('/kaggle/working/pretrained/ViT-B_16.npz')
        assert RUN_MODE in ('reeval', 'train_collect')
        assert not OUTPUT.exists(), 'Use a fresh OUTPUT directory to preserve previous results.'
        OUTPUT.mkdir(parents=True)
        (OUTPUT/'run_status.json').write_text(json.dumps({'status':'configured','full_protocol_complete':False}))
        NOTEBOOK_STARTED = time.perf_counter()
        def record_cell_failure(result):
            error = result.error_before_exec or result.error_in_exec
            if error is None: return
            (OUTPUT/'run_status.json').write_text(json.dumps({'status':'failed','error':str(error),'full_protocol_complete':False}))
            (OUTPUT/'timing.json').write_text(json.dumps({'notebook_elapsed_seconds':time.perf_counter()-NOTEBOOK_STARTED}))
            with (OUTPUT/'run.log').open('a',encoding='utf8') as stream:
                traceback.print_exception(type(error),error,error.__traceback__,file=stream)
            summary_path = OUTPUT/'summary.json'
            if summary_path.exists():
                summary=json.loads(summary_path.read_text()); summary['full_protocol_complete']=False
                summary['process_success']=False; summary_path.write_text(json.dumps(summary,indent=2))
            # Works even when clone/dependency setup fails before project imports.
            import zipfile
            archive=OUTPUT.parent/'SEEP_FREQUENCY_STATE_CUB200_metrics.zip'
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
                for path in sorted(OUTPUT.rglob('*')):
                    if path.is_file() and path.suffix.lower() in {'.json','.csv','.txt','.log','.patch','.md','.ipynb','.xml'}:
                        bundle.write(path,str(path.relative_to(OUTPUT)))
            print('Failed/partial archive:',archive)
        if 'get_ipython' in globals():
            get_ipython().events.register('post_run_cell',record_cell_failure)
        files = sorted(p for p in Path(CHECKPOINT_ROOT).rglob('*')
                       if p.is_file() and p.suffix.lower() in {'.pth','.pt','.ckpt'})
        print('Checkpoint candidate files (not yet loaded/validated):', len(files))
        for p in files[:40]: print(f'{p.stat().st_size/2**20:.1f} MiB  {p}')
    '''), cell('code', f'''
        BASE_COMMIT = {BASE!r}
        PATCH_SHA256 = {sha!r}
        PATCH_B64 = {payload!r}
        DELIVERY = {manifest!r}
        assert not REPO.exists(), 'Use a new REPO directory; existing work is preserved.'
        subprocess.check_call(['git','clone','https://github.com/thanhphu25/SEEP.git',str(REPO)])
        subprocess.check_call(['git','config','core.autocrlf','false'], cwd=REPO)
        subprocess.check_call(['git','checkout','--detach',BASE_COMMIT], cwd=REPO)
        patch = base64.b64decode(PATCH_B64)
        assert hashlib.sha256(patch).hexdigest() == PATCH_SHA256
        patch_path = REPO/'implementation.patch'
        patch_path.write_bytes(patch)
        subprocess.check_call(['git','apply','--check',str(patch_path)], cwd=REPO)
        subprocess.check_call(['git','apply',str(patch_path)], cwd=REPO)
        for name, expected in DELIVERY['source_files_sha256'].items():
            assert hashlib.sha256((REPO/name).read_bytes()).hexdigest() == expected, name
        (OUTPUT/'source_manifest.json').write_text(json.dumps(DELIVERY,indent=2))
        shutil.copyfile(patch_path, OUTPUT/'implementation.patch')
        shutil.copyfile(REPO/'README_FREQUENCY_STATE.md',OUTPUT/'README.md')
        sys.path.insert(0,str(REPO))
        from tools.frequency_setup import export
        def checked_command(command, phase):
            tick = time.perf_counter()
            with (OUTPUT/'run.log').open('a',encoding='utf8') as log:
                log.write('\\nPHASE '+phase+'\\n'); log.flush()
                process = subprocess.Popen(command,cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                    text=True,encoding='utf8',errors='replace',env={{**os.environ,'PYTHONUNBUFFERED':'1','PYTHONIOENCODING':'utf8'}})
                try:
                    for line in process.stdout: print(line,end=''); log.write(line); log.flush()
                    result = process.wait()
                except BaseException:
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()
                    raise
            (OUTPUT/(phase+'_result.json')).write_text(json.dumps({{'returncode':result,'seconds':time.perf_counter()-tick}}))
            if result != 0:
                (OUTPUT/'run_status.json').write_text(json.dumps({{'status':'failed','phase':phase,'returncode':result,'full_protocol_complete':False}}))
                summary_path = OUTPUT/'summary.json'
                if summary_path.exists():
                    summary=json.loads(summary_path.read_text()); summary['full_protocol_complete']=False
                    summary_path.write_text(json.dumps(summary,indent=2))
                print('Partial/failure archive:',export(OUTPUT))
                raise subprocess.CalledProcessError(result,command)
        print('Pinned source and patch verified:',PATCH_SHA256)
        checked_command([sys.executable,'tools/build_frequency_delivery.py'],'notebook_export')
        shutil.copyfile(REPO/'SEEP_Frequency_State_CUB200_Kaggle.ipynb',OUTPUT/'SEEP_Frequency_State_CUB200_Kaggle.ipynb')
    ''', tags=['embedded-source'], jupyter={'source_hidden': True}, collapsed=True), cell('code', '''
        # Keep installed torch/torchvision; timm shim is idempotent on Python 3.13.
        checked_command([sys.executable,'-c',
            "from tools.frequency_setup import dependencies; dependencies("+repr(str(OUTPUT))+")"], 'dependencies')
        import torch, torchvision
        print('Actual stack:',sys.version,torch.__version__,torchvision.__version__)
        assert torch.cuda.is_available(), 'Enable GPU in Kaggle settings.'
        print('GPU:',torch.cuda.get_device_name())
    '''), cell('code', '''
        # Tests use synthetic CPU tensors; no scientific accuracy claim.
        checked_command([sys.executable,'-m','pytest','-q','tests/test_frequency_state.py',
                         '-p','no:cacheprovider','--junitxml='+str(OUTPUT/'tests.xml')], 'tests')
    '''), cell('code', '''
        CONFIG = dict(run_mode=RUN_MODE,checkpoint_root=CHECKPOINT_ROOT,checkpoint_map=CHECKPOINT_MAP,
            trusted_checkpoints=TRUSTED_CHECKPOINTS,data_path=str(READY),output=str(OUTPUT),
            npz_path=str(NPZ_PATH),device='cuda:0',num_workers=4)
        CONFIG_PATH = OUTPUT/'notebook_config.json'
        CONFIG_PATH.write_text(json.dumps(CONFIG,indent=2))
        # Inventory and extraction happen BEFORE expensive evaluation. No images or optimizer step.
        checked_command([sys.executable,'tools/frequency_run.py','--config',str(CONFIG_PATH),'--inventory-only'], 'inventory')
        report=json.loads((OUTPUT/'checkpoint_inventory.json').read_text())
        for row in report.get('boundaries',[]):
            print(row['boundary'],row['status'],'epoch=',row.get('epoch'),'path=',row['path'])
        print('Missing:',report.get('missing_boundaries',[]))
    '''), cell('code', '''
        # Official metadata + split hashes, never a random train/test split.
        checked_command([sys.executable,'-c',
            "from tools.frequency_setup import prepare_data; import json; print(json.dumps(prepare_data("
            +repr(None if DATA_ROOT is None else str(DATA_ROOT))+","+repr(str(READY))+"),indent=2))"], 'data')
        shutil.copyfile(READY/'official_split_manifest.json',OUTPUT/'official_split_manifest.json')
    '''), cell('code', '''
        # Full preflight verifies saved args, frozen routing NPZ, strict dynamic load, transforms and GPU.
        # The runner performs that preflight before evaluating any boundary. No optimizer in reeval.
        # Progress includes boundary/task/images/elapsed/ETA; all stdout is retained in run.log.
        checked_command([sys.executable,'tools/frequency_run.py','--config',str(CONFIG_PATH)], 'evaluation')
    '''), cell('code', '''
        # Recompute from exported CSV matrices and require successful process exit.
        import numpy as np
        from modules.frequency_state import CHANNELS
        from tools.frequency_run import write_metrics
        status=json.loads((OUTPUT/'run_status.json').read_text())
        result=json.loads((OUTPUT/'evaluation_result.json').read_text())
        matrices={k:np.loadtxt(OUTPUT/f'task_accuracy_matrix_{k}.csv',delimiter=',') for k in CHANNELS}
        checks=json.loads((OUTPUT/'mutation_checks.json').read_text())
        summary=write_metrics(OUTPUT,matrices,process_success=result['returncode']==0,
            checks_pass=len(checks)==10 and all(c['passed'] for c in checks))
        assert status['status']=='complete' and summary['full_protocol_complete']
        archive=export(OUTPUT)
        print(json.dumps({'primary_delta_pp':summary['snapshot_minus_live_percentage_points']['raw_exact'],
                          'screening_signal':summary['screening_signal'],'full_protocol_complete':True},indent=2))
        from IPython.display import display, FileLink
        print('Metrics ZIP (no model checkpoints):',archive)
        display(FileLink(str(archive)))
    ''')]
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
