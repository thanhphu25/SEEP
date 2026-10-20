# Baseline audit

Base and current parent commit: `80081081ab42769d102f83d447705da0639e2fb1`.
Implementation is an uncommitted, reviewable working-tree diff on local branch
`ccsp-pilot`. The original untracked `prompt.md` is preserved.
`source_metadata.json` records parent commit, branch, archive hashes and the fact
that the implementation is uncommitted. Runtime source hashes also enter checkpoints.

No historical RainbowPrompt/LGSP config or checkpoint was supplied. Its equivalence
to this SEEP commit is not assumed. The supplied hyperparameters are ported to
the actual `cub200_10task` parser. Historical checkpoint accuracies are not tests
or results of this implementation.

| Source | Audited behavior | Implementation decision |
|---|---|---|
| `main.py:main` | Two ViTs, frozen original reference; full dataset linear classifier; general LR multiplied by batch/world size / 256 | Keep backbone/pretraining/head; add shared flags, local weights, explicit checkpoint entry points. Log requested/resolved LR. Reference is always frozen/eval. |
| `engine.py:train_one_epoch` | Original ViT receives raw image once; CE masked to current task minus `balancing * sim_loss` | Reuse the same reference output for CLS and normalized final patch tokens; retain loss. Prompted frozen backbone remains in autograd. |
| `engine.py:train_and_evaluate` | New local generator per task, previous generators frozen; optimizer/scheduler recreated per task; local LR separate, frequency `weights` separate, alpha/beta general LR | Preserve those parameter groups. `frequency_residual_scale` stays in general group. Add independent gate group and causal bank prepass. Replace implicit output-directory autoresume with explicit resume plus compatibility checks. |
| `engine.py:evaluate` | True evaluation task is used only for optional task-inc mask; original head otherwise unmasked; router averages cosine over entire minibatch | Default remains `legacy_batch` / `legacy_head`. Scientific protocol uses per-image predicted routes and all-seen head masking; task_inc must be False. |
| `vision_transformer.py:get_prompts` | Training: generators produce RGB images, per-pixel cosine along channels, generator-axis softmax, weighted sum; old outputs detached | Unchanged. Cache overwritten with `unified_prompt[0]` each training minibatch. Test retrieves this ungated cache. |
| `vision_transformer.py:forward_features` | LGSP is `alpha*(x+local) + beta*get_Frequency_mask(x)`; frequency output already includes x | Gate only local residual after mixing/cache and before fusion. Alpha/beta remain unconstrained. Frequency code unchanged. |
| `vision_transformer.py:forward_head` | Raw output exposes `pre_logits` and normalized `x`; linear head contains all classes from initialization | No prototype/cosine classifier; no reference-top-k head restriction. |
| `prompt.py:RainbowPrompt.forward` | Token warm_up selects Unique vs Rainbow; Rainbow stores per-task/per-layer token cache; test uses scalar `self.task_id` | Token evolution unchanged. Explicit forced subgroup routes set both local route and token `task_id` on every forward. |
| `datasets.py:build_transform` | RandomResizedCrop(scale .05..1), horizontal flip, ToTensor; eval resize 256/224, center crop, ToTensor | Retained without added normalization/clipping. Deterministic eval transform also used for prototypes. |
| `continual_datasets/continual_datasets.py:CUB200` | Old loader can download/extract and physically move images into train/test | Bypass with read-only `modules/cub.py`, preserving alphabetical ImageFolder class IDs and official split. Accept both raw metadata and pre-split trees. |
| `configs/cub200_10task.py` | 10 tasks, no class shuffle by default; many booleans use `type=bool` | Config file unchanged. Shared parser fixes bool actions (`False` now means False); compatibility tests cover all ten configs. |

The audited source is stored verbatim in
`tests/fixtures/audited_vision_transformer.py.txt`, with a SHA256 invariant, so the
baseline comparison runs in the source ZIP without `.git`. Train and legacy eval
logits on the tiny backbone match that source exactly with equal state and RNG.

Legacy default logits are covered; administrative behavior intentionally changes:
explicit resume, additional metadata, standard `R[t,j]` exports, correct bool parsing,
and read-only CUB loading. Existing legacy checkpoints support strict SEEP evaluation,
but cannot resume into CCSP without missing causal bank/split/RNG history.

Scientific comparisons must hold router, scope, split, class order, schedule and
seed fixed for B0/B3/A0. Routing corrections are common protocol changes and must
not be reported as CCSP gains. A historical local-on/off counterfactual does not
replace a newly trained B0 under the same development split and schedule.
