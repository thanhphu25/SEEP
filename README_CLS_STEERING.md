# Task-specific CLS steering screen for SEEP

Open `notebooks/SEEP_CLS_Steering_CUB200_Kaggle.ipynb` in Kaggle. Enable a single GPU
and Internet; **Add Input → `trungkiennguyn/lgsp-dataset` before Save Version**.
Run All. The notebook can also discover/attach the dataset through kagglehub in
an interactive session. No separate source upload or pushed branch is required.
No GPU/CUB experiment has been run locally: local validation uses CPU synthetic
inputs. GPU smoke and the full scientific run are separate notebook stages.

The notebook clones `thanhphu25/SEEP`, checks out base
`80081081ab42769d102f83d447705da0639e2fb1`, verifies and applies its embedded
combined patch once. The source manifest identifies the implementation commit,
every source checksum and the patch checksum. The audited original DGCA patch
was extracted from the supplied ZIP, checked with `git apply --check` and applied
once during preparation; SHA256:
`ecec522a42518819d2504890b6c7b6fede6938c5aca7d5d47322c2cd6cc9996e`.
`modules/dgca.py` and `modules/dgca_data.py` retain that implementation exactly.

At blocks **9, 10, 11** (zero based), a task bank adds three zero-initialized
768-dimensional vectors to the real CLS Q/K/V rows before prefix concatenation.
Only the current bank trains: **6,912/task, 69,120 allocated for ten tasks**.
Training chooses the current task; inference and Gaussian feature extraction
choose the original router's `matched_task_id`, including `task_id=-1`.
The original routing model is unchanged. Routing still selects one task per
batch, so representation depends on batch composition.

`--cls-steering off` is the default. The screening command is:

```bash
python -u main.py cub200_10task \
  --seed 3407 --batch-size 32 --epochs 30 --num_tasks 10 --num_tasks_to_run 0 \
  --lgsp YES --lgsp_type LGSP --data-path /kaggle/working/cub200_ready/ \
  --output_dir /kaggle/working/seep_cls_steering_align_only_seed3407 \
  --dgca align_only --dgca-shrinkage 0.1 --dgca-samples-per-class 128 \
  --dgca-epochs 5 --dgca-lr 0.01 \
  --cls-steering task --cls-steering-layers 9 10 11 --cls-steering-lr 0.0005
```

SEEP retains Adam, cosine scheduling (including its existing warmup), CE train
mask, similarity loss, local/frequency prompts and original prompt layers. Before
the scheduler, effective main LR is .00375, local .0005, frequency .03 and
steering .0005. Each boundary records group names, base/current LRs, counts and
trainability. Five separate alignment epochs train a fresh clone of the raw
head using unchanged Gaussian alignment: shrinkage .1, 128 samples/class/epoch,
SGD .01/momentum .9. Transport is disabled. Statistics use current training
images only and start fresh for this representation.

Diagnostics record old-bank hashes, delta/gradient norms, representation/raw
head/stored-cache invariance, and fixed up-to-64 current-train eval-view images.
Attention reports separate CLS→prefix and CLS→image-patch mass, conditional image
entropy and L1 attention change when deltas are disabled at the same model state.
They make no foreground or bird-part claims. Diagnostics preserve RNG, routing,
trainability and model modes; they do not choose hyperparameters.

The official split remains 11,788 images: 5,994 train / 5,794 test, 200 classes,
ten ordered 20-class tasks. Reference metadata and split hashes are checked.
Pretrained NPZ must hash to
`ac7478131e1b77e5c2f0225849730bd19ee89cfffe600c2dfd2272b34a50a43d`;
preflight verifies loaded tensors and fails on random fallback. Kaggle's installed
torch/torchvision are preserved, timm .6.11 receives an idempotent dataclass shim,
and environment differences from torch 2.11.0+cu128 / torchvision .26.0+cu128 are
logged. A different stack is not an exact reproduction claim.

Resume only this run's `checkpoint/dgca_latest.pth`. Set `RESUME=True` in the
notebook and attach/copy its previous output directory to `OUTPUT` if starting a
new Kaggle session; the same pinned source is reconstructed automatically.
Checkpoints commit completed task boundaries atomically and include banks,
trainability, raw/aligned heads, statistics, matrices, optimizer/scheduler and RNG.
Format, split, configuration, pretrained router and implementation fingerprint
must match. Completed boundaries are not trained/aligned twice. An interrupted
task restarts from the previous boundary; there is no mid-epoch resume.
Old align-only/transport checkpoints are rejected.

The new run exports raw_seen (C), align_only (D) and raw_exact from the same
inference features. Audited ZIP matrices supply A (SEEP raw) and B (SEEP aligned).
Reports include C−A, D−B, D−A, D−C, every task, AIA, final accuracy, forgetting,
old-task final mean, initial/peak means and BWT; matrix convention is A[i,t].
Historical 87.5441/86.4980/2.9515 stays separate. Screening requires a successful
full protocol, C−A AIA ≥ .3 percentage points, higher final/old-task accuracy and
lower forgetting. D−B checks the benefit with matched calibration. Partial or
failed runs cannot pass. One seed is preliminary evidence; no novelty claim or
test-driven sweep is made.

The notebook streams logs and preserves subprocess exit codes. After ten tasks
it creates `SEEP_CLS_STEERING_CUB200_metrics.zip` with matrices, reports, logs,
configuration, split, environment, source/patch hashes and tests. Download the
large checkpoint separately. Runtime is measured; the prior T4 estimate of about
three hours is only a planning reference.

Local tests: `python -m pytest tests/test_dgca.py tests/test_cls_steering.py -q`.
They compare mode off against the actual pinned source, test zero/nonzero deltas,
real-row/prefix separation, gradients/frozen banks, routing, optimizer membership,
strict resume trajectory/RNG and diagnostic isolation. GPU smoke uses only two
current-task batches and makes no accuracy claim.

Attribution: [CASP §4.1, Huang, Lin and Lu (2026)](https://arxiv.org/html/2601.16773v1).
This is a task-specific SEEP CIL adaptation of its additive CLS Q/K/V idea, not a
reproduction of FSCIL CASP. No CASP source code was copied (author repository
declares MIT). No perturbation, domain prompt, manifold mixup, pooling, SupCon,
KD, N1 or CCSR is added. Prior-art and capacity controls are deferred until a
positive screen. Existing SEEP copyright notices remain.
