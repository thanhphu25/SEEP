# N1 source audit

Base: `80081081ab42769d102f83d447705da0639e2fb1` on local `main`, matching
the requested audit commit. No tracked modifications or AGENTS.md were present.
The pre-existing untracked `SEEP_CCSP_source.zip` is outside this change.
The user's Kaggle checkpoint/log/notebook have not been supplied: matching the
source/config/environment of that historical run is **unconfirmed**.

## Source map (before implementation)

* `main.py`: config subparsers, seed initialization (zero chooses a random seed),
  two pretrained model creations, prefix-based freezing, LR scaling and entrypoint.
* `configs/cub200_{10,20}task.py`: CUB defaults below.
* `vision_transformer.py`: constructor; dynamic local generators; get_prompts;
  get_Frequency_mask; forward_features/head. Original forward returns a dict;
  `pre_logits` precedes fc_norm/classifier. Evaluation matches tasks using a
  batch-wide sum of key similarities, then uses saved local prompt templates.
* `engine.py`: CE minus balancing times sim_loss; per-task optimizer rebuild;
  evaluation and task checkpoints. **evaluate_till_now already returns
  test_stats at this commit.** No missing-return bugfix is required. This return
  describes the last evaluated task; macro accuracy comes from acc_matrix.
* `datasets.py`: RandomSampler, class subsets, unchanged train/eval transforms.
  CUB `val` is the official test directory, not a tuning validation set.
* `continual_datasets/continual_datasets.py`: CUB ImageFolder wrapper; baseline
  checks for the download ZIP even if an extracted train/test tree exists.
* `models.py`: local timm model registrations; ViT-B/16 uses 768 features,
  depth 12, 12 heads and repository pretrained configuration.
* `prompt.py` (read only): RainbowPrompt base_knowledge, base_key, matchers,
  learned task state and stored_rainbow_prompts. Algorithm is left intact.

## Equations actually implemented

K defaults to 100. p=softmax(weights * temperature), **not** weights / temperature.
Rings are differences of sigmoid masks (slope 4, prior radius plus 1e-6);
the first ring is zero. FFT and inverse use shift/unshift and PyTorch's default
normalization. T is the absolute value of the **complex** inverse FFT of
F(x) + F(x)*sum(p_k M_k). s=frequency_residual_scale is learned.
GSP=x+s*(T-x); LGSP=alpha*(x+P_local(x))+beta*GSP.
alpha and beta are unconstrained scalar parameters. Therefore q=(p,a,b),
a=alpha+beta, b=beta*s, gives z=a*x+b*(T-x). Full input is z+alpha*P_local.
get_prompts(train=True) stores the first sample's unified prompt in a buffer;
the N1 collector must bypass that path entirely.

## Actual trainability and optimizer groups

With default freeze prefixes, blocks, patch_embed, cls_token, norm and pos_embed
are frozen; original_model is entirely frozen. The classifier, spectral weights,
s, alpha/beta, RainbowPrompt base keys/knowledge and its matcher/MLP modules
remain eligible for gradients, subject to RainbowPrompt's existing task logic.
Local generators are created at each task; prior generators are frozen.
Runtime audit records exact eligible parameter names/counts and group LRs.

Per-task groups, in source order:

1. remaining trainable parameters, including **frequency_residual_scale**:
   effective args.lr and args.weight_decay;
2. local prompt generators: args.lr_local;
3. weights alone: args.lr_Frequency_mask;
4. alpha and beta: effective args.lr.

main scales only args.lr by batch_size/256 if unscale_lr, otherwise by
batch_size*world_size/256. Local/frequency LRs are not scaled by this formula.
The launcher reverses the saved scaling once before calling main.

## CUB defaults

Both configs: Split-CUB200, batch=32, epochs=30, seed=0, ViT-B/16 pretrained,
input=224, Adam (eps=1e-8, betas=.9/.999), lr_pre=.03, effective_lr=.00375
on one GPU, cosine schedule, lr_local=.0005, lr_Frequency_mask=.03,
num_r=100, temperature=1, LGSP YES/LGSP, balancing=.01, D1=56, D2=96,
warm_up=5, KI_iter=10, train_mask=True, task_inc=False, reinit_optimizer=True,
num_workers=4, pin_mem=True, drop/drop_path=0. Training uses RandomResizedCrop
(scale .05..1, ratio .75..4/3), horizontal flip, ToTensor. Evaluation uses
bicubic Resize(256 for 224 input), CenterCrop, ToTensor; **no Normalize**.
10/20-task differences: num_tasks, pool size 10/20, trial and output path.

## Provenance

Immutable source: [SEEP tree](https://github.com/thanhphu25/SEEP/tree/80081081ab42769d102f83d447705da0639e2fb1).
Conceptual references only: [MAS](https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses),
[PGP](https://github.com/JingyangQiao/prompt-gradient-projection),
[RainbowPrompt](https://github.com/Kiseong0753/RainbowPrompt).
No external implementation is copied. N1 is a new experimental hypothesis;
the Gram is neither a claim of MAS implementation nor a Fisher matrix.

Local verification environment initially available: Windows, Python venv,
torch 2.0.1+cpu, torchvision 0.15.2+cpu, timm 0.6.11; CUDA unavailable.
This differs from Kaggle. Historical checkpoint, resolved seed log, GPU and
notebook patches remain to be checked by the user via the launcher.

Implementation also separates image-only data imports from optional video
decoder imports, accepts prepared CUB splits without a ZIP, guards raw input
against the legacy move-based split, prepares dynamic local generators on
strict checkpoint load, and guards CUDA synchronization on CPU test hosts.
These are import/I/O/checkpoint support changes, not changes to LGSP, transforms,
RainbowPrompt or evaluation math. See n1_method.md for tensor-byte accounting
and n1_kaggle.md for checks performed and hardware limitations.
