# Run N1 on Kaggle

Use the dependency setup and pretrained cache from your working SEEP baseline
notebook. **Do not automatically install requirements.txt or upgrade/downgrade
torch/timm.** N1 adds no training dependency. CUB no longer imports the optional
video decoder. Dry-run imports torch for trusted checkpoint metadata but does
not import training/model modules, allocate a model, or download pretrained files.

Missing historical artifacts: your trusted baseline checkpoint containing args,
the actual positive `Seed:` value from its log (especially when saved args.seed=0),
the CUB input path and any notebook source/dependency patches. None were available
for local validation. The audited source matches the requested commit, but this
does not confirm it matches your historical notebook run. Review provenance,
environment and the full effective config before training.

## Four notebook cells

Cell 1 — clone the new branch (a fresh Kaggle working directory):

```python
%cd /kaggle/working
!git clone --single-branch --branch feat/n1-spectral-response-memory https://github.com/thanhphu25/SEEP.git SEEP_N1
%cd /kaggle/working/SEEP_N1
!git rev-parse HEAD
```

Cell 2 — set your paths and the actual baseline seed. Only edit these values;
all research hyperparameters come from checkpoint args. This reads metadata
only; baseline model/optimizer tensors are released and never initialize N1.

```python
import sys, subprocess
from scripts.run_cub_n1 import baseline_configuration
BASELINE = "/kaggle/input/YOUR_BASELINE/checkpoint/task10_checkpoint.pth"
DATA = "/kaggle/input/YOUR_CUB_ROOT"  # contains CUB_200_2011/train and test
SEED = None  # replace with the actual positive integer from baseline's Seed: log
assert isinstance(SEED, int) and SEED > 0, "Copy the actual baseline log seed"
MODE, LAMBDA = "semantic_full", 1.0
baseline_args, _ = baseline_configuration(BASELINE, SEED)
OUT = f"/kaggle/working/cub200_{baseline_args['num_tasks']}task_{MODE}_lambda{LAMBDA}_seed{SEED}"
CMD = [sys.executable, "scripts/run_cub_n1.py", "--baseline-checkpoint", BASELINE,
       "--seed", str(SEED), "--data-path", DATA, "--output-dir", OUT,
       "--srm-mode", MODE, "--srm-lambda", str(LAMBDA), "--srm-proj-dim", "16",
       "--srm-samples-per-class", "8", "--srm-probe-batch-size", "1",
       "--srm-memory-seed", "3407", "--srm-normalize", "trace"]
```

Cell 3 — dry-run and review config/diff. Repeating this cell before checkpoints
exist is supported, including changing an untuned config before the first train.

```python
from IPython.display import display, JSON
subprocess.run(CMD + ["--dry-run"], check=True)
display(JSON(filename=OUT + "/effective_config.json"))
display(JSON(filename=OUT + "/config_diff.json"))
```

Cell 4 — full train from pretrained initialization; no full baseline rerun:

```python
subprocess.run(CMD, check=True)
```

The same four cells are supplied in [n1_kaggle.ipynb](../notebooks/n1_kaggle.ipynb).
Use the final task10 **or** task20 baseline checkpoint as a configuration source;
subparser_name must agree with num_tasks. The launcher preserves Python bool,
tuple/list and legacy flags by building a Namespace directly. Saved args.lr is
effective: the launcher divides by batch_size/256 (or batch_size*world_size/256)
before main scales it once. lr_local/lr_Frequency_mask are left unchanged.
When available lr_pre_scaling metadata is used and verified against this formula.
Actual optimizer group LRs/names are written at runtime in optimizer_taskN.json
and in checkpoints. num_workers and pin_mem remain the baseline values.

Checkpoint seed=0 cannot identify the actual seed because baseline initialization
chose it randomly. The launcher records your supplied seed as user-confirmed;
it does not pretend to have inspected a log. Conflicting positive seeds fail.
Source commit without historical metadata is `unknown`, not certified matched.

## Optional two-task smoke and controls

Run the two-task smoke into a separate output; num_tasks remains 10/20, only the
task-stop changes. Neither batch size, backbone nor num_r is reduced:

```python
def set_arg(command, flag, value):
    command = list(command)
    command[command.index(flag) + 1] = str(value)
    return command
SMOKE = set_arg(CMD, "--output-dir", OUT + "_smoke")
subprocess.run(SMOKE + ["--num-tasks-to-run", "2", "--dry-run"], check=True)
subprocess.run(SMOKE + ["--num-tasks-to-run", "2"], check=True)
```

Optional ablation preflights, each with its own output. Execute a training line
only when you intend that full run; lambda=1 has not been selected by tuning:

```python
for mode in ["l2", "semantic_diag", "freeze_spectral"]:
    ablation = set_arg(CMD, "--srm-mode", mode)
    ablation = set_arg(ablation, "--output-dir", OUT.replace(MODE, mode))
    subprocess.run(ablation + ["--dry-run"], check=True)
    # subprocess.run(ablation, check=True)  # optional full ablation
```

For a lambda/projection/sample/norm change use a new OUT. Existing checkpoints
with a different research/N1 config fail rather than silently resume. Operational
paths and task-stop may change. To resume in a new Kaggle session, copy the full
previous output tree (especially checkpoint/taskN_checkpoint.pth) into the chosen
writable OUT and rerun the same CMD. Resume uses the latest completed boundary;
an interrupted next task is rerun. Metrics are rebuilt without duplicate rows.
Keep effective_config.json, config_diff.json, metrics.json, collector/optimizer
audits, logs and task checkpoints as notebook output. Evaluation-only loading
uses model state and never trains or collects N1 memory.

## Data preflight and raw CUB

Prepared input must have ROOT/CUB_200_2011/train and test with 200 nonempty class
folders and identical sorted class_to_idx. Read-only Kaggle input works without
the original download ZIP. Folder/mapping checks do not certify that images
are the official split; use your established baseline tree or the official
metadata preparation below. No random resplit is performed. `val` in the engine
means the official **test** split; it is not a hyperparameter tuning set.

If input only contains raw images/images.txt/train_test_split.txt, create splits
outside /kaggle/input before cell 2 (source may be the CUB folder or its parent):

```python
subprocess.run([sys.executable, "scripts/prepare_cub.py",
                "--source", "/kaggle/input/YOUR_RAW_CUB/CUB_200_2011",
                "--output-root", "/kaggle/working/cub_prepared",
                "--method", "symlink"], check=True)  # or copy
DATA = "/kaggle/working/cub_prepared"
```

The script uses official IDs even if metadata lines are out of order, retains
source images, and refuses to overwrite different destination content.

## Local verification and limits

Local tests use torch 2.0.1+cpu / torchvision 0.15.2+cpu / timm 0.6.11 on Windows.
Run `python -m pytest tests -q -p no:cacheprovider` in the existing environment.
Synthetic tests intentionally use tiny models/images for verification only;
the launcher never applies those reductions to production CUB.

Verified: source FFT/logits and off RNG/training trajectory parity; source
equation/gradients; reverse Jacobian versus finite differences; per-image Gram,
PSD, normalization and diagonal control; collector train-only IDs and state/RNG
isolation; all modes; two-task real-engine synthetic training; strict dynamic
checkpoint loads; continuous versus boundary resume with scheduler/reinitialized
optimizer and cosine/constant with a persistent optimizer; eval without train/collection;
launcher LR/type/seed/provenance/output/dry-run/data checks for CUB10/20.

No CUDA GPU, CUB dataset, user checkpoint or baseline log was available locally.
Full ViT-B/16 CUB training, Kaggle integration, numerical parity on CUDA and
collector CUDA peak/time are **not run**. CPU tiny-run results are correctness
checks, not research accuracy results. Report environment/GPU/data-order/source
differences and do not claim exact reproduction or improvement without matching
baseline evidence and training measurements.
