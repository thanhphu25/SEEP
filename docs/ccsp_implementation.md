# CCSP implementation and operation

The repository implements B0 (`seep`), B3 (`image_gate`) and A0 (`ccsp`). This is
an untrained research implementation, not evidence of accuracy improvement or
novelty. The user runs all real-data smoke/training. Parent commit and audit are
in [baseline_audit.md](baseline_audit.md).

## Components and equations

`modules/routing.py` predicts one route per image by cosine between frozen raw CLS
and `rainbow_prompt.base_key[:learned_id+1]`. Images with the same route are grouped;
the explicit `forced_task_id` interface sets **both** local-cache and token-cache
task on every subgroup forward. Logits are scattered back to input order. Raw
reference ViT runs once on the whole batch, and each subgroup reuses its CLS and
patch tokens. The true evaluation task/label never enters adaptation routing.
Training still uses the current training task; candidates query every seen class.

`modules/reference_bank.py` stores a normalized mean of normalized raw training CLS
per class, its class ID and count. New prototypes are appended before training
their task, using exactly its training subset with deterministic evaluation
transforms. Old classes cannot be updated. No image/patch exemplars persist.
The prepass is not truncated by debug batch limits and its cost is reported
separately. This is offline multi-epoch CIL with an extra pass, not single-pass CL.

For `q=normalize(raw_CLS)`, scores are q dot every valid prototype. The top-k uses
`k=min(reference_topk,number_seen)`. Competitor is a softmax-weighted sum of ranks
2..k at `reference_temperature`; descriptor is `[mu1, mu_minus, mu1-mu_minus,
mu1*mu_minus]`. Scalars are raw cosine top1-top2 gap, top-k softmax entropy divided
by log(k), top1 cosine, and k/reference_topk. One-class policy: competitor/gap/
entropy zero. Inputs/candidates/prototypes are stop-gradient. Labels are used
separately for candidate quality reporting, never for candidate construction.

`modules/spatial_gate.py` uses identical dimensions, parameters, initialization
and optimizer schedule for B3/A0. Patch projection is d→width + LayerNorm + GELU;
context projection is 4d→width + GELU. A shared patch MLP receives projected patch,
projected context and four scalars, then width→1 sigmoid. Image MLP receives raw
CLS, projected context and scalars, then width→1 sigmoid. Grid is inferred from
image dimensions and backbone patch size, and checked against token count.

For B3 only, context is `[normalize(CLS), normalize(mean_patches),
normalize(max_patches), normalize(std_patches)]`, with population std (`unbiased=False`).
Scalars are mean/std/max of cosine(CLS, each patch), and cosine(CLS, mean patch).
`F.normalize` epsilon handles zero vectors. No bank, candidate IDs, class scores
or class competition scalars enter B3.

Both output heads start with zero output weights and bias logit(0.9): a=S=0.9,
M≈0.81 exactly initially. Output heads get gradients immediately; hidden
projections begin learning after output weights move. Gate initialization uses
an isolated CPU RNG stream, preserving core initialization and later generator
RNG streams across paired variants. Prototype loader has its own generator, and
prepass RNG is preserved/restored as well.

`M = a * bilinear_upsample(S)` is broadcast across RGB and multiplies the **already
mixed** local residual before baseline fusion:

```python
x_prime = alpha * (x + M * local_prompt) + beta * global_output
```

Cached local prompts remain ungated. Test computes M from the current raw image
and applies it to the predicted-task cache. `seep` always uses M=1 by default,
regardless of the default `gate_mode=learned`. Explicit `identity` and `zero`
diagnostics set M=1 or M=0 without rewriting caches. They are evaluation controls,
not ways to select good official-test outcomes. The reference path is no_grad;
the prompted frozen backbone must retain input autograd into gate/current generator.

At d=768,width=64, both gates have **308,226 parameters**, **1,232,904 bytes** at
FP32, plus gradients/optimizer state/activations. A0's fixed-capacity bank for 200
classes allocates **617,600 bytes** (200×[768×4 + 8-byte ID + 8-byte count]); valid
class bytes are logged separately. B3 has no prototype memory/prepass. B3/A0
share reference forward count per training/eval batch. Actual parameter totals,
baseline stored-prompt buffer bytes, prototype time and peak CUDA allocation are
reported per task. No FLOPs reduction is claimed: small M does not skip generators
or the prompted ViT. Shared gate can forget, despite frozen old generators.

## Data and protocol

Main comparison: `--eval_router per_image --eval_scope seen --task_inc False`.
All seen classifier IDs are kept, with a union of class masks (including shuffled,
noncontiguous IDs), and only future classes are masked. Reference top-k never
filters classifier logits. Default flags retain legacy batch routing/head scope.

Development holds out 10% per class from official train, rounded to nearest count
with at least one dev and one train sample, split seed 89359. Relative sample IDs
are recorded. Train and prototypes exclude all held-out samples. All-seen dev
samples remain available across tasks solely for evaluation; this is **validation
memory**, not training replay. Class shuffle remains the baseline default False.

Pilot: first 3 of the unchanged 10-task split (20 CUB classes/task), 10 epochs,
token warm_up=2, scheduler warmup-epochs=1. Full development uses all 10 tasks,
30 epochs, warm-ups 5/5, still dev_fraction=.1. Final official test requires a
locked configuration and consistent policy across methods, optionally retraining
all three from scratch on full official train (`dev_fraction=0`, `eval_split=test`).
No official-test tuning or early stopping is implemented.

General LR retains `requested_lr * global_batch_size / 256`: batch32, one GPU,
requested .03 → .00375. Local and gate base LR are .0005 directly; frequency
`weights` LR .03; `frequency_residual_scale` stays general LR. Scheduler warm-up
can temporarily change current LR; group base and current values are logged.
Token warm_up and scheduler warmup-epochs are distinct.

## Kaggle and exact commands

Upload [SEEP_CCSP_Kaggle.ipynb](../notebooks/SEEP_CCSP_Kaggle.ipynb) as a Kaggle
notebook. Upload `SEEP_CCSP_source.zip` as a Kaggle Dataset, add it and CUB as Inputs,
then edit CODE_ZIP/DATA_PATH. ZIP has `main.py` directly at root and extracts into
`/kaggle/working/SEEP_CCSP`. It excludes `.git`, datasets, weights, checkpoints,
venvs and run outputs. Archive source hashes are verified by the notebook.

The notebook creates Python3.9 via uv in a private venv, installs original pinned
torch2.0.1/torchvision0.15.2/timm0.6.11 there, verifies GPU, runs toy tests and checks
all nine launcher commands parse. It never installs old torch into the notebook
kernel. It records pip freeze from the actual venv for each run.

Shell equivalent (run from repo root with the matching Python3.9 environment):

```bash
export PYTHON=/kaggle/working/venv/bin/python
DATA=/kaggle/input/your-cub200-dataset/CUB_200_2011
bash scripts/run_smoke.sh --data-path "$DATA" --method all --output-root /kaggle/working/runs
# Run each pilot explicitly after smoke passes:
bash scripts/run_pilot.sh --data-path "$DATA" --method seep --output-root /kaggle/working/runs
bash scripts/run_pilot.sh --data-path "$DATA" --method image_gate --output-root /kaggle/working/runs
bash scripts/run_pilot.sh --data-path "$DATA" --method ccsp --output-root /kaggle/working/runs
# Only after deciding from development results:
bash scripts/run_full.sh --data-path "$DATA" --method seep --output-root /kaggle/working/runs
# Final full-partition setting, only after configuration is locked:
bash scripts/run_full.sh --data-path "$DATA" --method seep --dev-fraction 0 --eval-split test --output-root /kaggle/working/final
```

Use the same final command for image_gate/ccsp under separate method directories.
Each method/stage/seed/eval-split/holdout gets a separate output path. Existing
run manifests/checkpoints fail without explicit resume. `--dry-run` prints the
complete command. Smoke uses 2 tasks, 1 epoch, 2 train/eval batches and warm-ups 0;
the A0 prototype pass still uses the full current training subset.

For local pretrained weights add `--pretrained-checkpoint /kaggle/input/.../weights.npz`
to the launcher, or underscore spelling `--pretrained_checkpoint` to main.py.
`.npz` uses the original ViT loader. `.pth` must contain exactly the matching
unprompted backbone, with an optional head (head excluded to retain CIL head).
Both frozen reference and adaptation receive the same backbone initialization.
Default pretrained weights/preprocessing are unchanged. Modern full checkpoint
resume/evaluation restores both models without downloading pretrained weights again.

Resume and evaluate (same method, split, architecture and schedule as training):

```bash
CKPT=/kaggle/working/runs/pilot/ccsp/seed89359/dev_holdout0.1/checkpoint/last.pth
bash scripts/run_pilot.sh --data-path "$DATA" --method ccsp --output-root /kaggle/working/runs --resume "$CKPT"
bash scripts/run_pilot.sh --data-path "$DATA" --method ccsp --output-root /kaggle/working/evaluation --eval-checkpoint "$CKPT"
```

To evaluate identity/zero using the complete raw main command, add
`--eval_checkpoint PATH --gate_mode identity` (or zero) and use a separate
`--output_dir`. Evaluating official test with a dev-trained checkpoint keeps its
training `--dev_fraction 0.1` and changes only `--eval_split test`; this is not
the final full-partition retraining setting. Evaluation can override router,
scope and gate diagnostic mode, but comparisons must label the changed protocol.

## Checkpoints, metrics and limits

Checkpoints include full adaptation and frozen reference state, dynamic local
generators, ungated pixel/token buffers, gate/bank/IDs/counts, optimizer/scheduler,
task/epoch/completion, RNG states, actual seed, class order, split IDs, source
commit/hashes and requested/resolved config. `last.pth` is atomic at each completed
epoch and after task evaluation; taskN_checkpoint.pth retains end-of-task states.
Resume checks config, split and implementation hash, restores RNG, and appends no
future/duplicate prototypes. It supports **epoch boundaries**, not mid-minibatch
recovery. Only trusted pickle checkpoints should be loaded. Legacy checkpoints
are evaluated with `method=seep`, strict dynamic-state reconstruction; their absent
bank/causal metadata are not fabricated for resume into a new method.
For a single legacy `--eval_checkpoint`, the last learned task is read from a
`taskN_checkpoint.pth` filename; otherwise supply `--checkpoint_task` (zero-based).
Modern checkpoints still within token warm-up are rejected for evaluation.

`metrics.jsonl` reports all-seen sample-weighted accuracy, task-mean accuracy,
AvgIncAcc (mean of sample-weighted all-seen accuracies after each task), FinalAcc
(current trained endpoint), old/new accuracy/errors, forgetting, candidate
top1/top-k (A0) with old/new breakdown, and gate/norm diagnostics. Train candidate
self-inclusion can be optimistic. Every row includes method/router/scope/split.
`task_stats.jsonl` includes train diagnostics, exact resources and LR groups.
`accuracy_matrix.csv/.npy` uses **R[t,j]** (transpose of original engine's matrix).
Evaluation of task j always competes against all classes seen at t, not just j.
T=1 forgetting is N/A; a single legacy final checkpoint has no AvgIncAcc/forgetting
unless history exists. Debug-limited runs are explicitly flagged.
Changing evaluation split/router/scope/gate mode/batch limit discards incompatible
checkpoint history; a single checkpoint then reports AvgIncAcc/forgetting as N/A.
Evaluating the complete checkpoint sequence rebuilds a comparable history.

The current runner supports one process/GPU. Gated/reference-bank paths target
image datasets (CUB acceptance target); no CCSP video protocol is claimed.
Full Linux dependency installation, GPU smoke, real CUB runs, latency/VRAM,
candidate quality and any accuracy/forgetting gains remain to be measured by
the user. CPU tests cannot establish them.

## Rebuild and verify deliverables

```bash
python scripts/build_notebook.py
python scripts/package_source.py
python scripts/verify_package.py
python -m pytest tests -q
```

See [validation.md](validation.md) for the checks actually run in this workspace.
