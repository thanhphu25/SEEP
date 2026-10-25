"""Build a self-contained Kaggle notebook and patch ZIP from a real source commit.

The patch excludes generated delivery files to avoid a recursive payload. Run
after committing implementation sources; never embeds or publishes user changes.
"""
import base64
import hashlib
import json
import subprocess
import textwrap
import zipfile
from pathlib import Path

BASE = '80081081ab42769d102f83d447705da0639e2fb1'
SOURCES = ['configs/cub200_10task.py', 'continual_datasets/continual_datasets.py',
           'datasets.py', 'engine.py', 'main.py', 'vision_transformer.py',
           'modules/dgca.py', 'modules/dgca_data.py', 'modules/dgca_runtime.py',
           'tools/summarize_dgca.py', 'tools/prepare_cub200.py', 'tools/preflight_dgca.py',
           'tools/smoke_dgca.py', 'tools/build_dgca_delivery.py', 'tests/test_dgca.py', 'README_DGCA.md']


def cell(kind, source):
    source = textwrap.dedent(source).strip()+'\n'
    value = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        value.update(execution_count=None, outputs=[])
    return value


def build():
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip()
    for name in SOURCES:
        committed = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root)
        if committed.replace(b'\r\n', b'\n') != (root/name).read_bytes().replace(b'\r\n', b'\n'):
            raise RuntimeError('Commit implementation before delivery: '+name)
    patch = subprocess.check_output(['git', 'diff', '--binary', BASE, commit, '--', *SOURCES], cwd=root)
    sha = hashlib.sha256(patch).hexdigest()
    manifest = {'base_commit': BASE, 'implementation_source_commit': commit,
                'branch': 'feat/dgca-classifier-alignment', 'patch_sha256': sha,
                'delivery_mode': 'embedded_patch_on_pinned_base',
                'source_files': SOURCES}
    payload = base64.b64encode(patch).decode()
    cells = [cell('markdown', '''
        # SEEP + DGCA — full Split-CUB200, seed 3407

        Enable **GPU** and **Internet**, attach official CUB-200-2011, then Run All.
        This notebook embeds the exact implementation patch and pins the original
        base. It does not require a pushed branch or a second notebook. It keeps
        the installed torch/torchvision stack and records differences from the
        historical baseline. A separate 2-batch smoke is followed by the full
        10-task × 30-epoch experiment; smoke has no scientific accuracy claim.
    '''), cell('code', '''
        import sys, subprocess, json, hashlib, base64, shutil, os, zipfile
        from pathlib import Path
        import torch, torchvision
        assert torch.cuda.is_available(), "Enable GPU in notebook settings"
        print(sys.version, torch.__version__, torchvision.__version__, torch.cuda.get_device_name())
        # Install timm without letting pip replace torch/torchvision.
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "timm==0.6.11"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "loguru", "pytest", "pyyaml"])
        # Python >=3.11 rejects mutable dataclass defaults in an unrelated
        # MaxxVit model imported by timm. Keep the pinned version and apply the
        # equivalent default_factory declarations; the SEEP ViT code is untouched.
        import importlib.metadata
        maxxvit = Path(importlib.metadata.distribution("timm").locate_file("timm/models/maxxvit.py"))
        original_text = maxxvit.read_text()
        fixed_text = original_text.replace("from dataclasses import dataclass, replace",
                                           "from dataclasses import dataclass, replace, field")
        fixed_text = fixed_text.replace("conv_cfg: MaxxVitConvCfg = MaxxVitConvCfg()\\n",
                                        "conv_cfg: MaxxVitConvCfg = field(default_factory=MaxxVitConvCfg)\\n")
        fixed_text = fixed_text.replace("transformer_cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg()\\n",
                                        "transformer_cfg: MaxxVitTransformerCfg = field(default_factory=MaxxVitTransformerCfg)\\n")
        maxxvit.write_text(fixed_text)
        TIMM_COMPAT = {"file": str(maxxvit), "before_sha256": hashlib.sha256(original_text.encode()).hexdigest(),
                       "after_sha256": hashlib.sha256(fixed_text.encode()).hexdigest()}
        print("timm Python compatibility:", TIMM_COMPAT)
        subprocess.check_call([sys.executable, "-c", "import timm; assert timm.__version__ == '0.6.11'; print(timm.__version__)"])
    '''), cell('code', f'''
        BASE_COMMIT = {BASE!r}
        SOURCE_COMMIT = {commit!r}
        PATCH_SHA256 = {sha!r}
        PATCH_B64 = {payload!r}
        DELIVERY = {manifest!r}
        REPO = Path("/kaggle/working/SEEP_DGCA")
        assert not REPO.exists(), "Use a fresh session/directory to avoid overwriting work"
        subprocess.check_call(["git", "clone", "https://github.com/thanhphu25/SEEP.git", str(REPO)])
        assert (REPO/".git").is_dir(), "Clone did not create .git"
        subprocess.check_call(["git", "checkout", "--detach", BASE_COMMIT], cwd=REPO)
        subprocess.check_call(["git", "switch", "-c", "feat/dgca-classifier-alignment"], cwd=REPO)
        patch = base64.b64decode(PATCH_B64)
        assert hashlib.sha256(patch).hexdigest() == PATCH_SHA256
        patch_path = REPO/"dgca_implementation.patch"
        patch_path.write_bytes(patch)
        subprocess.check_call(["git", "apply", "--check", str(patch_path)], cwd=REPO)
        subprocess.check_call(["git", "apply", str(patch_path)], cwd=REPO)
        (REPO/"dgca_delivery.json").write_text(json.dumps(DELIVERY, indent=2))
        (REPO/"timm_compatibility.json").write_text(json.dumps(TIMM_COMPAT, indent=2))
        print(json.dumps(DELIVERY, indent=2))
    '''), cell('code', '''
        # Set this to an official CUB_200_2011 directory if duplicate datasets exist.
        DATA_ROOT = None
        candidates = sorted({p.parent for p in Path("/kaggle/input").rglob("images.txt")
            if all((p.parent/name).is_file() for name in
                   ("image_class_labels.txt", "train_test_split.txt")) and (p.parent/"images").is_dir()})
        print("Official CUB candidates:", [str(p) for p in candidates])
        if DATA_ROOT is None:
            assert len(candidates) == 1, "Set DATA_ROOT explicitly when zero/multiple roots are found"
            DATA_ROOT = candidates[0]
        DATA_ROOT = Path(DATA_ROOT)
        READY = Path("/kaggle/working/cub200_ready")
        subprocess.check_call([sys.executable, "tools/prepare_cub200.py", "--source", str(DATA_ROOT),
                               "--output", str(READY)], cwd=REPO)
        print((READY/"official_split_manifest.json").read_text())
    '''), cell('code', '''
        OUTPUT = Path("/kaggle/working/dgca_cub200_10task_seed3407")
        assert not OUTPUT.exists(), "Use a fresh output directory for the first full run"
        FULL_ARGS = ["--seed", "3407", "--batch-size", "32", "--epochs", "30", "--num_tasks", "10",
            "--num_tasks_to_run", "0", "--lgsp", "YES", "--lgsp_type", "LGSP",
            "--data-path", str(READY)+"/", "--output_dir", str(OUTPUT), "--dgca", "transport",
            "--dgca-ridge", "1.0", "--dgca-shrinkage", "0.1", "--dgca-samples-per-class", "128",
            "--dgca-epochs", "5", "--dgca-lr", "0.01"]
        PREFLIGHT = Path("/kaggle/working/dgca_preflight")
        subprocess.check_call([sys.executable, "tools/preflight_dgca.py", *FULL_ARGS,
                               "--report-dir", str(PREFLIGHT)], cwd=REPO)
        resolved = json.loads((PREFLIGHT/"resolved_config.json").read_text())
        assert resolved["effective_lr"] == .00375
        assert resolved["num_tasks"] == 10 and resolved["epochs"] == 30
        assert resolved["num_tasks_to_run"] == 0 and resolved["task_inc"] is False and resolved["pretrained"] is True
        print(json.dumps(resolved, indent=2))
        print((PREFLIGHT/"versions.json").read_text())
        print((PREFLIGHT/"pretrained_manifest.json").read_text())
    '''), cell('code', '''
        subprocess.check_call([sys.executable, "-m", "pytest", "tests/test_dgca.py", "-q"], cwd=REPO)
        smoke_log = Path("/kaggle/working/dgca_smoke.log")
        with smoke_log.open("w") as stream:
            smoke = subprocess.Popen([sys.executable, "-u", "tools/smoke_dgca.py", "--seed", "3407",
                                      "--data-path", str(READY)+"/"], cwd=REPO,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in smoke.stdout:
                print(line, end="")
                stream.write(line)
            assert smoke.wait() == 0, "Smoke failed; inspect dgca_smoke.log"
    '''), cell('code', '''
        command = [sys.executable, "-u", "main.py", "cub200_10task", *FULL_ARGS]
        print("Running:", " ".join(command))
        # Stream stdout+stderr and preserve the actual process return code.
        stream_log = Path("/kaggle/working/dgca_full_stream.log")
        with stream_log.open("w") as stream:
            process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="")
                stream.write(line)
                stream.flush()
            returncode = process.wait()
        assert returncode == 0, f"Training failed with exit {returncode}; inspect {stream_log}"
        shutil.copy2(stream_log, OUTPUT/"run.log")
    '''), cell('code', '''
        subprocess.check_call([sys.executable, "tools/summarize_dgca.py", str(OUTPUT)], cwd=REPO)
        summary = json.loads((OUTPUT/"summary.json").read_text())
        assert summary["full_protocol_complete"], "Incomplete full run"
        source = json.loads((OUTPUT/"source_manifest.json").read_text())
        pretrained = json.loads((PREFLIGHT/"pretrained_manifest.json").read_text())
        assert source["cached_weights_sha256"][pretrained["file"]] == pretrained["sha256"]
        assert json.loads((OUTPUT/"split_manifest.json").read_text())["sha256"] == json.loads((PREFLIGHT/"split_manifest.json").read_text())["sha256"]
        assert source["delivery_manifest"]["patch_sha256"] == PATCH_SHA256
        bundle = Path("/kaggle/working/SEEP_DGCA_CUB200_metrics.zip")
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
            for pattern in ("*.json", "*.csv", "*.txt", "*.log", "*.patch"):
                for path in OUTPUT.glob(pattern):
                    archive.write(path, "metrics/"+path.name)
            for path in PREFLIGHT.iterdir():
                archive.write(path, "preflight/"+path.name)
            archive.write(REPO/"dgca_implementation.patch", "source/implementation.patch")
            archive.write(REPO/"dgca_delivery.json", "source/dgca_delivery.json")
            archive.write(REPO/"timm_compatibility.json", "source/timm_compatibility.json")
            archive.write(stream_log, "logs/full_stream.log")
            archive.write(smoke_log, "logs/smoke.log")
            archive.write(READY/"official_split_manifest.json", "data/official_split_manifest.json")
        from IPython.display import display, FileLink
        print(json.dumps(summary, indent=2))
        display(FileLink(str(bundle)))
        display(FileLink(str(OUTPUT/"checkpoint"/"dgca_latest.pth")))
    '''), cell('markdown', '''
        The metrics ZIP excludes model checkpoints. Download the checkpoint separately
        if needed. Primary deltas compare DGCA with raw_seen on the same trajectory.
        Historical metrics are a separate reference. Do not report smoke accuracy,
        claim a successful full run after a failed cell, sweep test-set hyperparameters,
        or infer novelty from one seed. See README_DGCA.md for sources and limitations.
    ''')]
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
        'accelerator': 'GPU', 'kaggle': {'isInternetEnabled': True, 'accelerator': 'gpu'}}, 'cells': cells}
    for i, value in enumerate(cells):
        value['id'] = f'dgca-{i:02d}'
        if value['cell_type'] == 'code':
            compile(''.join(value['source']), f'cell_{i}', 'exec')
    (root/'notebooks').mkdir(exist_ok=True)
    (root/'artifacts').mkdir(exist_ok=True)
    notebook_path = root/'notebooks'/'SEEP_DGCA_CUB200_Kaggle.ipynb'
    notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
    (root/'dgca_delivery.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    (root/'artifacts'/'SEEP_DGCA_implementation.patch').write_bytes(patch)
    with zipfile.ZipFile(root/'artifacts'/'SEEP_DGCA_Kaggle_delivery.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(notebook_path, 'SEEP_DGCA_CUB200_Kaggle.ipynb')
        archive.writestr('implementation.patch', patch)
        archive.write(root/'dgca_delivery.json', 'dgca_delivery.json')
        archive.write(root/'README_DGCA.md', 'README_DGCA.md')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    build()
