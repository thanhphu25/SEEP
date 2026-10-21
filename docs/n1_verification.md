# Verification record

2026-10-21, Windows CPU environment. Base source 80081081ab42769d102f83d447705da0639e2fb1.
No full training runs were performed on this machine.

```
.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --tb=short
35 passed (latest recorded suite includes cosine with a persistent optimizer)
```

Compilation/import checks passed for main/engine/model/memory/runtime, both CUB
configs, data loading, scripts and tests. git diff --check passed.

| Check | Result and practical limit |
| --- | --- |
| Off source FFT/logit/RNG parity | Exact CPU equality against class extracted from git base, seeds 11/37/91, K=5 and K=100, prompted train and eval. |
| Off training trajectory | Two-task tiny run matches original engine parameters and Python/NumPy/Torch RNG exactly. Only original unconditional CUDA sync is stubbed for the CPU test. |
| Source equation and gradients | z + alpha*local agrees with original LGSP; finite nonzero gradients to weights, alpha, beta, s with unsaturated synthetic coordinates. |
| Jacobian / Gram | Reverse VJPs versus finite differences, per-sample aggregation versus incorrect batch-mean Jacobian, symmetry/PSD, trace-zero and nonfinite rejection, analytic quadratic gradient. |
| Full / diag / L2 | Same collected full Gram and projection for full/diag; L2 shares anchors and sample IDs, uses I/d; empty-memory/anchor penalties zero. |
| Collector isolation | Exact model/reference state_dict equality, heterogeneous mode and requires_grad restoration, prompt buffers and global RNG unchanged; current train-only nested subset IDs; probe batch 1/2 gives equal matrices. |
| Two-task engine smoke | Actual train_one_epoch, evaluate, local and RainbowPrompt forward paths, boundary collection and save; task 2 raw penalty positive. All five modes checked. |
| Boundary resume | Continuous versus resumed model/memory/projection/IDs/RNG/accuracy matrix exactly equal, cosine with both rebuilt/persistent optimizer and constant with persistent optimizer. Repeated completed smoke does not duplicate metrics. |
| Strict loading / eval | Dynamic generators/template buffers reconstructed before strict load; missing memory fails training resume; eval cannot train/collect. |
| Launcher | CUB10/20, both LR scaling flags, LR metadata validation, bool/list/tuple preservation, positive-seed conflict and seed=0 disclosure, output config guard, repeated dry-run, no training/model imports in dry-run. |
| Data | Prepared 200-folder tree requires no ZIP/download/split and no file writes; mismatched class mapping rejected; official metadata copy preparation retains source and matches IDs even out of order. |
| Metrics | A[i,t], null unevaluated cells/first-task old/forgetting, old/new means, forgetting/BWT, incremental average versus final accuracy. |

The test fixture uses inherited Windows directory ACLs because pytest's default
private temporary directory is inaccessible in this restricted workspace. Test
artifacts are ignored under .n1-test-runs; no existing user directories were
deleted or reset. The original failed fixture setup and synthetic image-size
assertions were resolved before the passing run.

Not run: real CUB data, the user's baseline checkpoint/log/notebook patches,
full production ViT-B/16 initialization/training, CUDA parity and GPU collector
profiling, Kaggle notebook execution, raw-data symlink execution on Linux.
The runtime records actual collector time/CUDA peak/gradient statistics when
the user runs it. These synthetic checks are not evidence of N1 accuracy,
novelty, a tuned lambda or improved forgetting. Dependency versions were not changed.

## Fresh-run launcher follow-up (2026-10-22)

User requested a new independent run without an old baseline artifact. Added
--config cub200_10task/cub200_20task, mutually exclusive with the optional
--baseline-checkpoint path. Fresh runs use repo research defaults, accept a new
positive seed and retain pre-scaling LR=.03. No historical checkpoint is read.
Kaggle guide/notebook now contain five fresh-start cells, including explicit
installation into a separate Python 3.10/Torch 2.0.1 CUDA 11.8 environment.

Full suite: **38 passed in 16.66s**. Fresh-mode tests reject checkpoint reads,
verify defaults/types/LR/seed/reporting for CUB10/20 and repeated dry-run. The five
notebook cells parse as Python. The original parity/resume tests still pass.
The new Linux/CUDA environment installation and full Kaggle training remain
unexecuted locally; no local dependency installation or full training was done.
