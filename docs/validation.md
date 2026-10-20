# Validation record

Environment prepared specifically for CPU invariant checks: Python3.9.25,
torch2.0.1+cpu, torchvision0.15.2+cpu, timm0.6.11, NumPy1.26.4,
pytest8.3.5 on Windows. CUDA is unavailable. No CUB dataset or user checkpoint
was available. No real-data training, GPU smoke, full benchmark, remote push or
publication was performed.

Executed checks:

- 35 core/data/CLI tests passed: exact legacy train/eval logits against the audited
  source; grouped/single/reordered routing for all methods; local/token route
  agreement; identity and zero controls; ungated cache; reference/gate/generator
  gradient invariants; bank edge cases/mapping/counts/dtype/strict load; actual
  read-only CUB fixture loader, deterministic train/dev split and prototype subset;
  parser compatibility across all configs and all nine smoke/pilot/full commands;
  exact metrics on a known matrix.
- 5 checkpoint/resume tests passed: end-of-task resume for B0/B3/A0 and interrupted
  epoch resume for A0 matched uninterrupted toy runs **exactly**, including all
  model buffers/parameters, accuracy matrix, torch RNG and optimizer LR. Wrong
  method metadata/legacy-to-CCSP resume is rejected.
- Main CLI `--parse_only` entry point executed successfully in the isolated venv.
- Gate parameter count measured: 308,226 at d768,width64, equal for B3/A0.
  Bank allocation measured: 617,600 bytes at 200 classes,d768,FP32 + int64 IDs/counts.

Final complete suite: **52 passed, 0 skipped** (`8.14s` in this CPU environment).
This additionally covers the actual timm ViT-Tiny factory at 224px for all three
methods, one raw reference forward, hidden gate projection gradients after the
initial head update, legacy task metadata/warm-up guards, incompatible evaluation
history rejection, and notebook/archive invariants.

Notebook schema validated with nbformat5.10.4: **23 cells**, all unexecuted with
empty outputs. All **43 source/test Python files** compiled successfully.
`git diff --check` passed. The ZIP root/whitelist and SHA256 of every packaged
source file passed verification; `main.py` is at the archive root.

Command used for the complete suite:

```powershell
.venv\Scripts\python.exe -m pytest tests -q --disable-warnings --basetemp=D:\GITHUB_PROJECT\SEEP\.test-tmp-suite-final
```

Toy optimizer steps exercise implementation invariants only and are not training
experiments or evidence of a method gain.

Windows sandbox denied pytest's newly created temporary directories; tests were
rerun with approved access restricted to a named temporary directory inside the
repo. This is an environment issue, not a skipped model invariant.

Outstanding user checks: Kaggle Linux dependency installation and GPU smoke for
all three methods, then paired development pilots. Accuracy, candidate quality,
VRAM and runtime have not been measured on real data. Do not report historical
checkpoint accuracy or toy results as CCSP results.
