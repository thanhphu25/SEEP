# SEEP + DGCA (experimental)

DGCA adds current-data drift compensation and a separate Gaussian-aligned linear
classifier to SEEP. The default is `--dgca off`. Backbone, prompt training loss,
augmentation, class order, batch routing, optimizer groups and learning-rate
scaling follow base `80081081ab42769d102f83d447705da0639e2fb1`.

## Run on Kaggle

Open [SEEP_DGCA_CUB200_Kaggle.ipynb](notebooks/SEEP_DGCA_CUB200_Kaggle.ipynb),
enable GPU and Internet, and attach the official CUB-200-2011 dataset. Run every
cell. The notebook embeds the implementation patch, verifies its SHA256, clones
the pinned base, applies the patch, validates all 11,788 official image records,
creates `cub200_ready`, downloads/checks the baseline pretrained NPZ, runs tests
and a separate two-batch smoke process, then starts the full experiment.
It needs no other notebook, Kaggle token, pushed branch or placeholder commit.
`dgca_delivery.json` records the real source commit and patch hash. The delivery
ZIP includes the same notebook, patch and manifest for offline source transfer.

The notebook preserves installed torch/torchvision and installs only timm 0.6.11
and small dependencies. It reports differences from the historical environment
(torch 2.11.0+cu128, torchvision 0.26.0+cu128); a different environment is not a
claim of historical baseline reproduction. It never installs requirements.txt
over the Kaggle torch stack. Download failure or invalid weights stops the run.
The notebook changes two unrelated MaxxVit dataclass defaults to equivalent
`default_factory` declarations so timm 0.6.11 imports on Python >=3.11, and logs
the before/after file hashes. SEEP's ViT implementation is unchanged by that shim.

```bash
python -u main.py cub200_10task \
  --seed 3407 --batch-size 32 --epochs 30 --num_tasks 10 \
  --num_tasks_to_run 0 --lgsp YES --lgsp_type LGSP \
  --data-path /kaggle/working/cub200_ready/ \
  --output_dir /kaggle/working/dgca_cub200_10task_seed3407 \
  --dgca transport --dgca-ridge 1.0 --dgca-shrinkage 0.1 \
  --dgca-samples-per-class 128 --dgca-epochs 5 --dgca-lr 0.01
```

Effective SEEP LR is `0.03*32/256 = 0.00375`. D1=56, D2=96 and
`pretrained=True`, `task_inc=False` remain baseline defaults. Only single-device
Split-CUB200 with batch 32 is supported by this experimental runner.

If multiple official datasets are attached, set `DATA_ROOT` in the discovery
cell. The adapter creates symlinks into the attached read-only images, preserves
official train=5,994/test=5,794 membership, and uses the baseline ImageFolder
ordering. No random data split is performed. CUB's loader accepts prepared
train/test folders without downloading its legacy archive. Image datasets no
longer require importing the optional video decoder.

## Algorithm and isolation

At each boundary, deterministic eval-transform features are collected only from
the current training task, in sequential batches of 32 with drop_last=False.
Stable underlying indices, labels and transformed-image SHA256 must match before
and after training. Before-features precede dynamic prompt/optimizer mutation.
Extraction uses `eval()`, `train=False`, `task_id=-1`, learned_id=t-1/t and
the original model's pre_logits for the unchanged batch-wide SEEP routing.
The classifier feature is exactly `fc_norm(pre_logits)`, verified by a head hook.

For row vectors, float64 residual ridge solves
`A=I+solve(Xc.T@Xc+lambda*I, Xc.T@(Yc-Xc))`,
`b=mean(Y)-mean(X)@A`,
`lambda=ridge*trace(Xc.T@Xc)/d+1e-8`.
Old statistics become `mu@A+b`, `A.T@Sigma@A`; current classes receive fresh
unbiased statistics after transport. Diagnostics use a deterministic current-only
80/20 holdout, then the final map uses every current pair. These diagnostics do
not measure old-class transport quality. `align_only` skips transport; reserve
it for a full ablation after positive transport results.

All raw full covariances stay CPU float32. To avoid cubic covariance transport,
one CPU eigendecomposition per old class/boundary restores its known empirical
rank (at most count-1), with negative-clamp/roundoff corrections logged.
Transporting its low-rank factor gives the same full covariance formula.
Current factors come directly from centered examples. Sampling uses
`sqrt(.9)*F*z + sqrt(.1*trace(Sigma)/d+eps)*noise`, with
`eps=max(1e-8,1e-6*trace(Sigma)/d)`; this retains full covariance and applies
shrinkage once on a temporary copy. There is no per-batch/epoch decomposition.

Alignment clones the raw linear head, samples 128 examples per seen class per
epoch, shuffles with a private generator, and trains only that clone for 5 epochs
with SGD(lr=.01, momentum=.9, weight_decay=0), minibatch 256 and cosine to zero.
CE indexes actual seen labels with explicit remapping. No normalization, KD,
token loss, exemplar images or test-based head selection is added.

Python/NumPy/Torch CPU/CUDA RNG, loader generators, model modes and routing
attributes are isolated during extraction. The alignment head never enters a
SEEP optimizer. Representation, raw-head and stored-prompt hashes must stay
equal across extraction/alignment. Per-image feature caches are deleted before
the boundary checkpoint; only class statistics survive.

## Resume and metrics

Checkpoints are atomic **completed task boundaries** in
`checkpoint/dgca_latest.pth`. Only the latest large checkpoint is retained.
The raw model/head, dynamic generators/buffers, trainability, aligned head,
statistics, completed boundary, optimizer/scheduler, all RNG states, class masks,
indices, matrices and reports are saved. There is no mid-epoch resume: an
interrupted uncommitted task restarts from the last committed boundary.
Use the same command/config/output with `--dgca-resume`. Loading uses
`weights_only=False` only after this explicit trusted-checkpoint opt-in.
Config/split mismatches and repeated transport are rejected. Derived metric
files are overwritten from committed matrices, avoiding duplicate rows.
`--eval --dgca transport` evaluates the latest aligned checkpoint without
appending training metrics.

Every evaluation forwards each image batch once and records three full-precision
task-average matrices: `dgca`, `raw_seen` and `raw_exact` (the original unmasked
head). Unseen logits are excluded from dgca/raw_seen. All three share features,
input batches and inference routing; true task labels never select prompts.
Future matrix cells are NaN. Rows are evaluation tasks, columns are boundaries.
Forgetting at boundary t averages `max(A[i,i:t+1])-A[i,t]` over i<t.

```bash
python tools/summarize_dgca.py /kaggle/working/dgca_cub200_10task_seed3407
python -m pytest tests/test_dgca.py -q
python tools/smoke_dgca.py --data-path /kaggle/working/cub200_ready/ --seed 3407
```

Outputs include `run_config.json`, `environment.txt`, `source_manifest.json`,
`split_manifest.json`, all three matrices, `summary.json`, `comparison.csv`,
`acc.txt`, `forgetting.txt`, boundary diagnostic JSONs and `run.log`.
The notebook downloads a small metrics ZIP separately from the model checkpoint.
The paired `raw_seen` comparator is primary. Historical AIA/final/forgetting
87.5441/86.4980/2.9515 is reported separately, without assuming equivalence.
Screening requires AIA/final accuracy to rise and forgetting to fall. Practical
priority requires AIA delta>=.3, forgetting delta<=-.2 and final accuracy delta>0.
Partial runs cannot pass screening. No hyperparameter sweep is performed.

## Costs, limitations, attribution

Full 200-class 768x768 covariance storage uses 450 MiB plus means and transient
factors/synthetic samples. This is statistical memory, not memory-free or a
privacy guarantee. CPU factorization and two extra current-data feature passes
add compute; boundary timing/memory are logged. Batch-wide routing makes features
depend on batch composition. A current-class map can extrapolate poorly to old
classes; one Gaussian can miss multimodal fine-grained distributions. No accuracy
or runtime improvement is guaranteed. One seed is a screening experiment only.

This implementation is written from the supplied mathematical specification;
no external implementation code was copied. Its conceptual sources are
[SLCA (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_SLCA_Slow_Learner_with_Classifier_Alignment_for_Continual_Learning_on_ICCV_2023_paper.html),
[LDC (ECCV 2024)](https://arxiv.org/abs/2407.08536),
[SLDC (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/39698), and
[RanPAC (NeurIPS 2023)](https://papers.neurips.cc/paper_files/paper/2023/hash/2793dc35e14003dd367684d93d236847-Abstract-Conference.html).
DGCA is an internal experiment name, not a novelty claim or a reproduction of
all components/defaults of these methods. Existing SEEP copyright notices remain.

Local validation uses CPU synthetic/toy data. A real CUB/GPU smoke and the full
10-task scientific run must execute in Kaggle; their results are not fabricated.
