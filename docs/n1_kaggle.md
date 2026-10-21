# Fresh N1 training on Kaggle

Start a new Kaggle notebook, enable **GPU** and **Internet**, and add CUB as an
input. No old SEEP checkpoint, seed log or notebook is needed. N1 starts from
the repository's pretrained ImageNet initialization and CUB defaults, with a
new experiment seed. Run the five cells in order; only DATA_ROOT must be edited.
The same cells are in [n1_kaggle.ipynb](../notebooks/n1_kaggle.ipynb).

## Cell 1 - clone the N1 branch

```python
import os, sys, subprocess
from pathlib import Path

REPO = Path("/kaggle/working/SEEP_N1")
BRANCH = "feat/n1-spectral-response-memory"
if not REPO.exists():
    subprocess.run(["git", "clone", "--single-branch", "--branch", BRANCH,
                    "https://github.com/thanhphu25/SEEP.git", str(REPO)], check=True)
os.chdir(REPO)
subprocess.run(["git", "checkout", BRANCH], check=True)
subprocess.run(["git", "pull", "--ff-only", "origin", BRANCH], check=True)
subprocess.run(["git", "rev-parse", "HEAD"], check=True)
```

## Cell 2 - install an isolated training environment

This explicitly installs Python 3.10, Torch 2.0.1/CUDA 11.8, torchvision 0.15.2
and timm 0.6.11 in a separate environment. The notebook kernel's Torch is not
changed. The Torch/torchvision combination follows the
[official installation instructions](https://pytorch.org/get-started/previous-versions/).
[uv environments](https://docs.astral.sh/uv/pip/environments/) and
[Python management](https://docs.astral.sh/uv/guides/install-python/) provide
the isolated interpreter. First setup downloads several GB. No requirements.txt
installation or video/NLP dependencies are needed for CUB.

```python
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
ENV = Path("/kaggle/working/seep_n1_env")
PY = str(ENV / "bin/python")
if not Path(PY).exists():
    subprocess.run([sys.executable, "-m", "uv", "venv", "--python", "3.10",
                    "--seed", str(ENV)], check=True)
subprocess.run([PY, "-m", "pip", "install", "torch==2.0.1", "torchvision==0.15.2",
                "--index-url", "https://download.pytorch.org/whl/cu118"], check=True)
subprocess.run([PY, "-m", "pip", "install", "numpy==1.26.4", "Pillow==10.3.0",
                "timm==0.6.11", "loguru==0.7.2", "huggingface-hub==0.26.2"], check=True)
subprocess.run([PY, "-c", "import torch, torchvision, timm, main; "
                "print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'timm', timm.__version__); "
                "assert torch.cuda.is_available(), 'Enable GPU in Kaggle notebook settings'; "
                "print('GPU:', torch.cuda.get_device_name(0))"], check=True)
```

If you already have a compatible environment, point PY to its Python instead.

## Cell 3 - data and new experiment settings

Edit DATA_ROOT. Prepared input may have CUB_200_2011/train and test; raw input
needs images/, images.txt and train_test_split.txt. Raw official splits are
symlinked into /kaggle/working, retaining source images. No random resplit occurs.
SEED is a new user-selected positive seed; it need not match an old log.

```python
DATA_ROOT = Path("/kaggle/input/YOUR_CUB_DATASET")  # EDIT THIS PATH
CONFIG = "cub200_10task"  # or "cub200_20task"
SEED = 3407
MODE, LAMBDA = "semantic_full", 1.0

if DATA_ROOT.name == "CUB_200_2011":
    DATA_ROOT = DATA_ROOT.parent
cub = DATA_ROOT / "CUB_200_2011"
if not all((cub / split).is_dir() for split in ("train", "test")):
    raw = cub if (cub / "images.txt").is_file() else DATA_ROOT
    assert (raw / "images.txt").is_file(), f"Set DATA_ROOT to your CUB dataset: {DATA_ROOT}"
    prepared = Path("/kaggle/working/cub_n1_prepared")
    subprocess.run([PY, "scripts/prepare_cub.py", "--source", str(raw),
                    "--output-root", str(prepared), "--method", "symlink"], check=True)
    DATA_ROOT = prepared
OUT = Path(f"/kaggle/working/{CONFIG}_n1_{MODE}_lambda{LAMBDA}_seed{SEED}")
CMD = [PY, "scripts/run_cub_n1.py", "--config", CONFIG, "--seed", str(SEED),
       "--data-path", str(DATA_ROOT), "--output-dir", str(OUT), "--srm-mode", MODE,
       "--srm-lambda", str(LAMBDA), "--srm-proj-dim", "16", "--srm-samples-per-class", "8",
       "--srm-probe-batch-size", "1", "--srm-memory-seed", "3407", "--srm-normalize", "trace"]
print("Output:", OUT)
```

## Cell 4 - dry-run

Checks the dataset/config and writes effective_config.json/config_diff.json,
without reading an old checkpoint or loading/downloading a model.

```python
import json
subprocess.run(CMD + ["--dry-run"], check=True)
report = json.loads((OUT / "effective_config.json").read_text())
print("Config:", report["config_source"], report["args"]["subparser_name"])
print("Learning rates:", report["learning_rates"])
print("N1:", {k: v for k, v in report["args"].items() if k.startswith("srm_")})
```

## Cell 5 - full train

Downloads the repository pretrained weights if absent from cache, then trains
the complete selected CUB task sequence with N1.

```python
subprocess.run(CMD, check=True)
```

Repo defaults are batch=32, epochs=30, ViT-B/16 pretrained, num_r=100, frozen
original encoder, and the original local/token/classifier training protocol.
LR=.03 is pre-scaling; main scales it once to .00375. lr_local=.0005 and
lr_Frequency_mask=.03 are unchanged. lambda=1 is an untuned pilot. This fresh
run is not certified to reproduce or improve a historical baseline.

## Optional smoke and controls

```python
def set_arg(command, flag, value):
    command = list(command)
    command[command.index(flag) + 1] = str(value)
    return command
SMOKE = set_arg(CMD, "--output-dir", str(OUT) + "_smoke")
subprocess.run(SMOKE + ["--num-tasks-to-run", "2"], check=True)
```

Smoke retains num_tasks=10/20 and all model/batch settings. For l2, semantic_diag
or freeze_spectral, change MODE in cell 3 and rerun cells 3-5. Choose a separate
OUT for lambda/projection/norm changes. Official test images (called val in the
engine) are not a tuning validation set.

A new OUT starts fresh. Rerunning CMD in a compatible OUT resumes **that N1
run**, at its latest completed task boundary. An interrupted task is rerun.
Missing memory/config or incompatible config fails; metrics do not duplicate
completed rows. Preserve the whole OUT as Kaggle output, including checkpoint/,
metrics.json and config/collector/optimizer audit files.

The optional --baseline-checkpoint interface remains available only for users
deliberately importing historical baseline configuration. It is mutually
exclusive with --config; none of the fresh-run cells uses it.

## Verification limits

Fresh CUB10/20 preflight, LR, types, seed guards, no-checkpoint reads and repeat
dry-run are unit tested. Existing math/parity/collector/two-task engine/resume
and optional baseline-launcher tests remain in the suite. Notebook cells are
validated as Python. Local verification uses CPU; the new isolated Linux/CUDA
environment installation and full Kaggle/CUB training have not been run here.
