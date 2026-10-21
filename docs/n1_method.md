# N1 — Spectral Response Memory (experimental)

N1 is an unvalidated research hypothesis on SEEP's LGSP branch. No accuracy
improvement or novelty result is claimed. See [source audit](n1_source_audit.md)
and [Kaggle instructions](n1_kaggle.md).

## Source-aligned coordinates

For the exact already-transformed input x, the original soft ring masks M_k,
and shifted FFT/inverse FFT operators F/F^-1:

```
p = softmax(weights * temperature)
T(x,p) = abs(F^-1(F(x) + F(x) * sum_k p_k M_k))
GSP(x) = x + s * (T(x,p) - x),  s = frequency_residual_scale
LGSP(x) = alpha * (x + P_local(x)) + beta * GSP(x)
q = concat(p, a, b),  a = alpha + beta,  b = beta * s
z(x;q) = a*x + b*(T(x,p) - x)
LGSP(x) = z(x;q) + alpha*P_local(x)
```

q is a differentiable effective coordinate vector, not an independent Parameter.
The trainable original parameters and their baseline LR groups remain intact.
This replaces the old idealized `alpha_g*softmax(w/tau)` proposal: the source
multiplies temperature and has an identity contribution, a learned residual
scale, soft masks and complex magnitude. It is not valid to replace abs by real
or infer an energy-diagonal metric using an idealized Parseval argument.

## Probe, memory and loss

Reuse the already-created unprompted, pretrained, frozen original_model in eval:

```
h(x;q) = normalize(f0(z(x;q))['pre_logits'], dim=-1) @ R.T
R: [r,D], independent Rademacher signs / sqrt(r)
J_x = dh(x;q)/dq evaluated at a detached leaf copy of q_c
G_c = mean_x J_x.T @ J_x
Gbar_c = G_c / max(trace(G_c), 1e-8)          # normalize=trace
L_srm = mean_old_classes (q-q_c).T @ Gbar_c @ (q-q_c)
L = CE - balancing*sim_loss + srm_lambda*L_srm
```

R is generated once using a private CPU generator, and the actual matrix/seed
are checkpointed. h is not renormalized after projection. The collector uses
reverse mode with r scalar VJPs per image, no create_graph and no batch-mean
Jacobian. FFT/probes are float32 with autocast disabled; frozen reference
parameters do not prevent differentiation with respect to q. No no_grad,
inference_mode or detached features are used in the probe.

At each task boundary, each current class contributes up to 8 training images
chosen with a private deterministic RNG. An independent shallow view of the
current train subset uses the unchanged repository eval transform; the shared
training transform and official test dataset are never used or modified for
collection. Nested Subset indices are resolved to base-training sample IDs.
The collector never calls the prompted model's forward/get_prompts. It restores
Python/NumPy/Torch CPU/CUDA RNG, all module modes, reference requires_grad flags
and routing attributes. No optimizer step or reference parameter update occurs.
Current classes are added only after training; old memory is never refreshed.

Matrices are numerically symmetrized and checked for finite values and PSD
(negative eigenvalue tolerance 1e-6*max(1, Frobenius norm)). If trace <=1e-8,
a warning is logged and the matrix is zero, including for normalize=none.
No ridge/damping is added. normalize=none otherwise retains the unnormalized
mean Gram. Penalty gradients use current q without detaching it.

## Controls and costs

| Mode | Behavior |
| --- | --- |
| off | No projection, probe, penalty or extra RNG draws/forwards. |
| l2 | Same class anchors/subset selection; metric I/d. No semantic forward. |
| semantic_diag | Diagonal of the exact same normalized full Gram collected by semantic_full. |
| semantic_full | Full normalized Gram; main experimental hypothesis. |
| freeze_spectral | From task 2 freeze weights and s; alpha/beta remain baseline-trainable. |

All modes share one implementation. Diag currently stores the full Gram for
auditable equivalence, so it has the same memory footprint as full. L2 also
stores a dense I/d; it is a coordinate-distance control, not semantic importance.
freeze_spectral freezes the spectral core, not the entire global/identity input.
All enabled modes reject configurations outside single-GPU CUB10/20 LGSP MVP.

At K=100, d=102, C=200, float32 matrices alone occupy **8,323,200 bytes**.
Anchors add 81,600 bytes, R at r=16,D=768 adds 49,152 bytes and int64 class IDs/
counts add 3,200: **8,457,152 tensor bytes**. `memory_bytes` reports actual tensor
payload, excluding Python/JSON metadata, serialization overhead and model buffers.
Selected IDs/counts are metadata; images and per-image features are not retained.

The default full/diag collector performs at most 200*8*16 = **25,600 scalar VJPs**
over a complete CUB run. This cost is not free. Each collector writes wall time,
absolute peak CUDA allocated bytes (not reserved bytes), number of VJPs, selected
training IDs, raw/normalized Gram trace and min/mean/max Jacobian norm to
`collector_taskN.json`. On CPU, peak CUDA bytes is zero. Kaggle time/memory have
not yet been measured. Configured probe batches amortize loading only; each
image still gets an independent forward/Jacobian.

Training logs raw `srm_loss` and `srm_weighted_loss` separately. Every fifth epoch,
first batch, diagnostics report p entropy, a/b, s/alpha/beta, change from the last
anchor and CE/weighted-regularizer gradient norms over original spectral/gating
parameters. weights starts with std=10 in production, so saturated softmax may
give tiny gradients; a small penalty is not evidence of useful stability.

## Checkpoints, evaluation and interpretation

Collection happens after task training/evaluation and before atomically saving
the complete task checkpoint. Full acc_matrix, per-task metrics, original args,
pre/effective LR, parameter flags, optimizer group names/LRs, memory, R, completed
task, RNG, resolved seed, class masks/mapping, source commit and environment are
saved. Resume restores the latest complete boundary before creating the next
generator, with strict model load after constructing dynamic generators/buffers.
Missing/mismatched N1 memory/config fails. Resume does not re-evaluate or collect
completed tasks. A crash inside a task restarts that task from the last complete
boundary. Reference pretrained initialization is recreated using the same seed
and cache; it is not a per-task new encoder. Source/environment changes during
resume are explicitly reported in resume_audit.json.

The launcher enables these checkpoint/metrics facilities for off as well. Direct
legacy main with mode=off retains its old checkpoint/resume path. N1 adds no
inference component or unseen-class mask. Raw evaluation is the baseline
batch-level router, saved templates, task_inc=False and all classifier logits.
Batch size, backbone, pretrained weights, transforms, num_r, local/token prompts
and classifier training are not altered by the launcher.

metrics.json uses zero-based A[i,t], accuracy on task i after task t; unevaluated
cells are null. acc_avg is the baseline macro task mean. Average incremental
accuracy is the mean of completed acc_avg values, distinct from final accuracy.
First-task old accuracy/forgetting/BWT are null. Per-task srm_loss is the mean
of the epoch-average raw penalties for that task. Legacy acc.txt/forgetting.txt
are regenerated from committed boundaries to avoid resume duplicates.

This is a local quadratic surrogate of a global-only probe. It does not
guarantee zero forgetting or preserve the local branch, token prompts, routing
or classifier. lambda=1 is an untuned pilot. Never tune on the official CUB test
split. Any sweep needs a common train-derived validation split and equal tuning
budget. Analyze CUB10 and CUB20 separately and disclose environment/source/data
differences from the historical baseline.
