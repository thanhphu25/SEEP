# SEEP task frequency-state recall — CUB200 / seed 3407

This implementation tests one mechanism: recall the frequency state belonging to
the task selected by the unchanged SEEP batch router. It does not introduce a new
loss, classifier, representation module, Gaussian alignment, or transport.

Import `SEEP_Frequency_State_CUB200_Kaggle.ipynb` into Kaggle. Enable GPU and
Internet. Add Input `trungkiennguyn/lgsp-dataset` and your own trusted original
SEEP checkpoint dataset. For background Save Version, attach both datasets with
Add Input first; do not rely on an interactive kagglehub attachment persisting.
The notebook can discover official CUB metadata or attach the image dataset with
kagglehub. If the dataset lacks official metadata it stops and requests an
official CUB-200-2011 Input. It never creates a random split.

## Two explicit modes

`RUN_MODE = 'reeval'` is the default. It performs no training, optimizer creation,
or head fitting. Supply boundaries 1..10 from the same original SEEP run. The
default discovery pattern is `task1_checkpoint.pth` ... `task10_checkpoint.pth`.
For different filenames, multiple runs, or duplicate boundaries, set
`CHECKPOINT_MAP` to a JSON path or dictionary of **one-based** boundaries to
absolute Input paths, for example `{"1": "/kaggle/input/my-run/task1_checkpoint.pth"}`
(include all 10 entries). The bank itself uses **zero-based** task indices.
Checkpoint files are read in place on CPU one at a time; they are not copied to
working and optimizer payloads are released before model evaluation.

Namespace checkpoints require `TRUSTED_CHECKPOINTS = True` for your own trusted
files. This explicitly enables pickle loading on current PyTorch. Unidentified
files, CLS Steering, N1, CCSR and DGCA transport checkpoints are rejected.
DGCA `align_only` checkpoints may supply `experiment_config` instead of `args`;
only their raw SEEP model is used. A lone `dgca_latest.pth` is not a trajectory.
If metadata cannot prove common run lineage/commit/environment, the inventory
records that limitation even when tensor/config consistency checks pass.

`RUN_MODE = 'train_collect'` explicitly starts a **new full 10 × 30 epoch run**.
It uses the pinned original training loop and optimizer grouping, with fresh LR
.03 scaled once to .00375 (local .0005; frequency .03 before scheduler). The
observer clones snapshots after the last scheduler update, before original
evaluation. Its paired evaluation preserves RNG, modes and routing attributes.
Training resume is deliberately unsupported: an existing checkpoint directory
or bank is rejected. Use a new output for another run. This prevents replaying
metrics or replacing old snapshots; a final checkpoint is never treated as a
training resume. Interrupted source checkpoints remain available for explicit
reeval when a complete sequence is supplied. Historical 2–3 hour training time
is only a reference; measured progress/ETA is printed during execution.

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

Pinned base: `80081081ab42769d102f83d447705da0639e2fb1` from
<https://github.com/thanhphu25/SEEP>. The notebook embeds a SHA256-verified patch
and clones this base; it does not require a pushed feature branch. Only the CUB
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

Default output: `/kaggle/working/seep_frequency_state_seed3407`.
`SEEP_FREQUENCY_STATE_CUB200_metrics.zip` contains four full-precision CSV
matrices, summaries, per-task initial/peak/final values, comparison deltas,
checkpoint/bank provenance, route counts, paired output diagnostics, mutation
checks, source/split/pretrained manifests, environment, configuration, timing,
logs, tests, patch, README and an output-free notebook. Model checkpoints are
excluded. In train_collect they remain separately in the working checkpoint
directory; retain them separately if needed.

Convention: `A[i,t]` is task i accuracy after boundary t; future cells are NaN.
AIA is the average of boundary task means. Final forgetting averages
`max(A[i,i:10]) - A[i,9]` over old tasks i<9, using each branch's own matrix.
Old-task final/peak means, diagonal mean and BWT are reported. Deltas are
percentage points. Completion requires 10 boundaries, 55 cells per matrix,
passed tests/preflight/mutation checks and successful process exit. Failed or
interrupted runs remain failed/partial; CPU synthetic tests are not CUB results.

The primary comparison is snapshot minus live from the same checkpoint
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
