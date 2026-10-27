# SEEP task frequency-state recall — CUB200 / seed 3407

This implementation tests one mechanism: recall the frequency state belonging to
the task selected by the unchanged SEEP batch router. It does not introduce a new
loss, classifier, representation module, Gaussian alignment, or transport.

Import `SEEP_Frequency_State_CUB200_Kaggle.ipynb` into Kaggle. Enable GPU and
Internet. Add Input `trungkiennguyn/lgsp-dataset` containing official CUB images
and metadata. **No SEEP checkpoint dataset or upload is needed.**
For background Save Version, attach the image dataset with
Add Input first; do not rely on an interactive kagglehub attachment persisting.
Leave `DATA_ROOT = None` and choose **Run All**. The four code cells get the
latest code, set up data and dependencies, train, and show results.
The notebook clones the current HEAD of `feat/task-frequency-state` automatically
and prints the exact branch/commit used. It creates fresh checkout/output
directories on each Run All, preserving earlier runs.

CUB discovery searches recursively, including the LGSP layout
`LGSP/CUB_200_2011/CUB_200_2011`. Even an explicit `DATA_ROOT` may point to an
outer folder; the setup finds the nested folder with `images/` and all three
official metadata files. It can attach the image dataset with kagglehub when no
Input is attached. Missing or ambiguous official metadata produces a clear error;
the official train/test split and its hashes are still checked.

## Full training from images (default)

The notebook always uses `train_collect`. Run all cells to start a
**new full 10 × 30 epoch run** using all 5,994 official training images across
200 classes. The 5,794 official test images remain evaluation-only.
It uses the pinned original training loop and optimizer grouping, with fresh LR
.03 scaled once to .00375 (local .0005; frequency .03 before scheduler). The
observer clones snapshots after the last scheduler update, before original
evaluation. Both branches are evaluated online after every task, producing the
complete matrices without reloading model checkpoints. Its paired evaluation
preserves RNG, modes and routing attributes.

`SAVE_CHECKPOINTS = True` is the default. Training starts directly from images
without source checkpoints. After each task, the full model, optimizer and
scheduler state are saved to `OUTPUT/checkpoint/task<N>_checkpoint.pth`.
These files are part of the Kaggle notebook output; no checkpoint Input dataset
needs to be created or attached. The frequency bank (4,120 bytes of tensor values
after 10 tasks, plus JSON metadata), metrics and logs are also saved.
The verified ImageNet ViT NPZ is downloaded for pretrained initialization.

In runner JSON, `save_checkpoints` defaults to true when omitted. Set
`SAVE_CHECKPOINTS = False` only if you want to skip saving full model files.
Training resume is deliberately unsupported: an existing checkpoint directory
or bank is rejected. Use a new output for another run. With checkpoint saving
off, an interrupted training run must be restarted; partial metrics remain
available. Historical 2–3 hour training time is only a reference.

## Optional checkpoint re-evaluation

The script retains explicit `reeval` support for existing runs. Create a runner
JSON with `run_mode: "reeval"`, `checkpoint_root`, `trusted_checkpoints`,
`data_path`, `output` and `npz_path`, then invoke
`python tools/frequency_run.py --config <config.json>`. Complete the same setup
and tests first (`tests_result.json` with returncode 0 is required in output).
This mode performs no training, optimizer creation or head fitting.

Supply boundaries 1..10 from the same original SEEP run. The discovery pattern
is `task1_checkpoint.pth` ... `task10_checkpoint.pth`. For different filenames or
duplicate boundaries, provide `checkpoint_map` as a JSON path or dictionary of
**one-based** boundaries to absolute paths (all 10 entries). The bank itself
uses **zero-based** task indices. Files are read on CPU one at a time in place.

Set `trusted_checkpoints: true` only for your own trusted Namespace/pickle
files. Unidentified files, CLS Steering, N1, CCSR and DGCA transport checkpoints
are rejected. DGCA `align_only` checkpoints may supply `experiment_config`
instead of `args`; only their raw SEEP model is used. A lone `dgca_latest.pth`
is not a trajectory. Unknown source lineage/commit/environment is reported.

## What the inputs permit

| Available artifacts | Valid result |
|---|---|
| Boundaries 1..10, consistent same-run provenance | Four matrices, AIA, final accuracy, forgetting |
| Final checkpoint + complete historical bank | Final metrics only; this notebook requires the trajectory for full reeval |
| Final checkpoint without historical bank | Old frequency states cannot be recovered |

Missing checkpoints stop reeval with their boundary numbers. There is no
automatic switch to training, substitution of the latest state, or use of
historical baseline numbers as measurements.

## Protocol and implementation

Runtime source: the latest `feat/task-frequency-state` commit from
<https://github.com/thanhphu25/SEEP>. There is no embedded source patch or checkout
of an old commit in the notebook. `source_manifest.json` records the actual
remote, branch and commit. The separate delivery patch against original base
`80081081ab42769d102f83d447705da0639e2fb1` is retained for offline verification.
Only the CUB
ImageFolder adapter, lazy video imports and `tools/prepare_cub200.py` were reused
from the supplied align-only ZIP. Original DGCA patch SHA256:
`ecec522a42518819d2504890b6c7b6fede6938c5aca7d5d47322c2cd6cc9996e`.
See `references/align_only_seed3407/provenance.json`. No DGCA runtime is included.
Original source attribution and copyright comments are retained.

Official CUB train/test counts are 5,994/5,794; 200 classes, 10 tasks × 20,
unshuffled class order, seed 3407, batch 32, sequential per-task validation with
drop_last=False. Original Resize/CenterCrop/ToTensor evaluation preprocessing is
unchanged. The router selects a task for a whole batch, so validation task loaders
must not be combined. Main accuracy uses all classifier logits (`raw_exact`);
`raw_seen` separately masks seen classes, never the true task. Test labels are
used only to count correct predictions.

Each slot contains float32 `weights[100]`, `frequency_residual_scale`, `alpha`,
and `beta`: 103 values/task, 4,120 bytes total tensor values for 10 tasks, plus
metadata. Radii, temperature, fixed configuration and declared frozen backbone
tensors must agree across checkpoints. Dynamic CNN generators and runtime
spatial buffers are reconstructed before strict loading. No `strict=False` is
used. Snapshot arguments are functional; no live Parameters are overwritten.
The original FFT/ifft/abs, rings, temperature and mixing expression are retained.
Actual head features are exposed after `fc_norm`, with no new learned projection.

Keep Kaggle torch/torchvision. Install timm 0.6.11 without dependencies and apply
the idempotent MaxxVit dataclass compatibility shim if needed. Actual versions
and GPU are recorded. The original ViT-B/16 NPZ is verified against
`ac7478131e1b77e5c2f0225849730bd19ee89cfffe600c2dfd2272b34a50a43d`.

## Outputs and interpretation

Each run creates `/kaggle/working/seep_frequency_state_seed3407_<run-id>/`
and prints its exact location.
`SEEP_FREQUENCY_STATE_CUB200_metrics.zip` contains four full-precision CSV
matrices, summaries, per-task initial/peak/final values, comparison deltas,
training/bank provenance, route counts, paired output diagnostics, mutation
checks, source/split/pretrained manifests, environment, configuration, timing,
logs, tests, patch, README and an output-free notebook. Model checkpoints are
saved separately in `OUTPUT/checkpoint/`
by default. They are retained in the Kaggle notebook output and excluded from
the metrics ZIP.

Convention: `A[i,t]` is task i accuracy after boundary t; future cells are NaN.
AIA is the average of boundary task means. Final forgetting averages
`max(A[i,i:10]) - A[i,9]` over old tasks i<9, using each branch's own matrix.
Old-task final/peak means, diagonal mean and BWT are reported. Deltas are
percentage points. Completion requires 10 boundaries, 55 cells per matrix,
passed tests/preflight/mutation checks and successful process exit. Failed or
interrupted runs remain failed/partial; CPU synthetic tests are not CUB results.

The primary comparison is snapshot minus live from the same model
trajectory. Screening requires AIA >= +.3 pp, final > 0 and old-task final > 0;
the additional forgetting goal requires F < 0 and interpretation alongside peaks.
One seed is not statistical significance. Do not tune slots on test accuracy.
This is prompt-state management/recall; novelty and zero forgetting are not
established. DualPrompt is related prior work, not evidence for this SEEP change:
<https://research.google/pubs/dualprompt-complementary-prompting-for-rehearsal-free-continual-learning/>.

Local checks:

```powershell
python -m pytest -q tests/test_frequency_state.py -p no:cacheprovider
python tools/build_frequency_delivery.py
python tools/verify_frequency_delivery.py
```

The delivery validation JSON records the actual local stack and test outcomes.
No local full CUB/GPU experiment is claimed without its own completed artifacts.
