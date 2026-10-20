# Prompt gửi Codex: implement CCSP và đối chứng trực tiếp trên SEEP

Hãy thực hiện công việc trong tài liệu này đến khi có source chạy được, tests, hướng dẫn và notebook Kaggle. Đọc source trước, lập kế hoạch ngắn rồi implement; không dừng ở đề xuất kiến trúc hoặc pseudocode. Không tự chạy các lượt train dài, không tự push remote/publish. Nếu thiếu GPU/dataset, hoàn thành implementation, chạy checks thực hiện được và ghi rõ phần chưa chạy.

## 1. Mục tiêu và code gốc

- Base repo: https://github.com/kiennkt05/SEEP
- Commit đã audit: `80081081ab42769d102f83d447705da0639e2fb1`.
- Lệnh Kaggle lịch sử của người dùng chạy `kiennkt05/RainbowPrompt`, branch `LGSP`, config `cub200_RainbowPrompt`. Đây là tham khảo hyperparameters; không tự coi nhánh đó tương đương commit SEEP. Nếu checkpoint/config gốc mâu thuẫn, audit và báo khác biệt trước khi port.
- Tạo branch nghiên cứu local `ccsp-pilot`. Nếu workspace đã có source/chỉnh sửa, kiểm tra `AGENTS.md`, git status và diff trước; bảo toàn thay đổi của người dùng. Nếu dùng commit khác, ghi commit và khác biệt liên quan.
- Triển khai ba biến thể: `seep` (B0), `image_gate` (B3), `ccsp` (A0).
- Hướng nghiên cứu: predicted class competition điều tiết local spatial prompt bằng patch-wise mask và soft image-level bypass. Đây là giả thuyết chưa có kết quả/novelty đầy đủ.
- Setting: offline class-incremental learning, train theo task; inference không có nhãn/task ID thật. Main experiments không replay ảnh hoặc patch exemplars. Bộ nhớ bổ sung là prototypes theo lớp và tham số gate; phải tính bytes.
- Hướng chính vẫn là spatial regulation; việc sửa routing là điều kiện đánh giá chung, không được nhận thành gain/novelty của CCSP.

## 2. Trạng thái thực nghiệm để không hiểu nhầm

Baseline đã được người dùng chạy; chưa train CCSP. Diagnostics trên một checkpoint CUB-200, toàn bộ 5.794 test images:

| Inference | Local bật | Local tắt |
|---|---:|---:|
| Batch cùng task, router gốc | 86.4515% | 86.4688% |
| Batch trộn task, router gốc | 77.0107% | 77.4077% |
| Batch size 1 | 84.7774% | 84.8637% |

Ở batch size 1, local giúp 50 ảnh và gây hại 55 ảnh. Chỉ đổi batching đã ảnh hưởng cả token/local routing. Đây là counterfactual trên cùng checkpoint, không phải một mô hình train lại không local; không kết luận local prompting nói chung vô dụng. Các số trên là accuracy của checkpoint này, không phải AvgIncAcc hay kết quả của phương pháp mới. Không hardcode chúng thành expected test results.

## 3. Những chi tiết source phải bảo toàn

Audit `main.py`, `engine.py`, `vision_transformer.py`, `prompt.py`, `datasets.py`, `configs/cub200_10task.py`; viết mapping ngắn trong `docs/baseline_audit.md`.

1. Train: `get_prompts()` sinh prompt cho từng ảnh qua các local generators, cosine theo RGB channels ở mỗi pixel, softmax trên trục generators rồi trộn. Generators cũ frozen/detached theo baseline.
2. Test: lấy `stored_unified_prompts` của task được chọn; hiện buffer được ghi đè bằng `unified_prompt[0]` mỗi minibatch train. **MVP giữ nguyên cơ chế này.** Không âm thầm thay bằng prompt trung bình task hoặc chạy lại generators theo ảnh test.
3. Fusion thật ở LGSP:

   ```python
   global_output = model.get_Frequency_mask(x)  # đã chứa ảnh + frequency residual
   x_prime = alpha * (x + local_prompt) + beta * global_output
   ```

   Alpha/beta trainable và không được mặc định tổng bằng 1. Giữ frequency branch và `frequency_residual_scale` như source, kể cả quy tắc learning rate.
4. Frozen `original_model` đã chạy trên ảnh gốc; output có `pre_logits` (CLS) và `x` (normalized final token sequence). Tái sử dụng CLS và patch tokens từ cùng forward. Không thêm một ViT forward chỉ để lấy patches.
5. Classifier gốc là linear head cho toàn bộ dataset, được khởi tạo sẵn. Không tự thay thành cosine/prototype classifier hay đổi sang head mở rộng theo task.
6. Training loss gốc là CE có class masking theo baseline, trừ `balancing * sim_loss`. Giữ nguyên trong MVP, không thêm margin, sparsity, EWC, replay hoặc distillation.
7. Giữ preprocessing gốc của SEEP. Không tự thêm ImageNet/CLIP normalization, clipping ảnh hoặc đổi pretrained weights.
8. `--warm_up` là warm-up của token prompt; `--warmup-epochs` là warm-up của LR scheduler. Đây là hai tham số khác nhau.

## 4. Routing riêng cho từng ảnh và protocol chung

Implement `--eval_router {legacy_batch,per_image}` và `--eval_scope {legacy_head,seen}`.

- Default legacy phải giữ hành vi SEEP gốc khi không bật feature mới. Scientific pilot dùng `per_image` và `seen` cho cả B0/B3/A0.
- `per_image`: với mỗi ảnh, cosine raw frozen CLS với `rainbow_prompt.base_key` của các task đã học, argmax riêng từng ảnh. Không dùng task labels, true labels hoặc predicted class để đưa true task vào model.
- Có thể gom các ảnh chọn cùng task để forward chung; sau đó ghép logits theo thứ tự input ban đầu. Cả cached local prompt và cached token prompts ở mọi layer phải dùng đúng task được chọn cho ảnh đó.
- Không chỉ sửa `matched_task_id` thành một vector rồi để `rainbow_prompt.task_id` tiếp tục là scalar của cả batch. Thiết kế wrapper/forced-route interface rõ ràng, tránh state từ nhóm cuối bị dùng cho nhóm khác.
- Raw reference forward thực hiện một lần cho batch; chuyển các CLS/patch tokens tương ứng vào từng subgroup. Khi train, vẫn dùng current task theo baseline; candidates của gate vẫn truy vấn mọi seen class.
- `seen`: chỉ mask các lớp chưa học bằng tập hợp class IDs của những task đã thấy; không giới hạn logits theo task thật hoặc reference top-k. Không giả định IDs liên tục nếu class order shuffled.
- `legacy_head`: hành vi output-head gốc khi `task_inc=False`; không biến nó thành task-incremental evaluation. Scientific runs phải là `task_inc=False`.
- Batch size 1 là reference test của per-image routing, không phải yêu cầu phải inference từng ảnh suốt benchmark. Kiểm tra kết quả batched/grouped/reordered với router per-image tương đương trong tolerance số học.
- Ghi rõ protocol trong mọi bảng. Không so CCSP per-image trực tiếp với baseline legacy-batch rồi gọi chênh lệch đó là gain của CCSP.

## 5. Reference prototype bank và lifecycle causal

Implement module độc lập, ví dụ `modules/reference_bank.py`:

- Reference encoder cùng frozen unprompted ViT và weights/preprocessing với baseline. `eval()` và `no_grad()` cho reference path.
- Với mỗi class mới, lấy mean của normalized raw CLS trên **training subset thực sự dùng để train** của class đó, rồi normalize mean. Lưu vector, class ID và count.
- Tạo prototypes trước khi train task mới bằng một pass riêng với deterministic eval transform. Chỉ append class mới; prototypes cũ không cập nhật ở task sau.
- Không dùng validation/test/future-class samples; không lưu ảnh/patch exemplars; không online-update từ test. Đây là offline multi-epoch CIL, có thêm prototype pass, không phải single-pass online CL.
- Tách RNG của prototype loader/gate initialization khỏi RNG augmentation/sampling/core initialization để hạn chế làm lệch paired runs không cần thiết.
- Prototype bank phải vào checkpoint, `.to(device)`, save/load và resume được. Hỗ trợ class IDs không liên tục. Cất số đếm và valid-class mapping, không chỉ một tensor không có mapping.
- Không dùng reference prototypes trực tiếp làm classifier cho prompted features.

Candidate descriptor của CCSP:

```text
q = normalize(raw_CLS)
scores[c] = q dot prototype[c]             # mọi seen classes
k_eff = min(reference_topk, number_seen_classes)
candidates = topk(scores, k_eff)
mu1 = prototype của top-1
mu_minus = weighted sum của candidates hạng 2..k_eff
          weights = softmax(competitor_scores / reference_temperature)
context = concat(mu1, mu_minus, mu1 - mu_minus, mu1 * mu_minus)
scalars = [raw top1-top2 cosine gap, normalized top-k entropy,
           top1 cosine score, k_eff / reference_topk]
```

- Entropy dùng softmax(scores/top-k temperature), chia `log(k_eff)` nếu `k_eff>1`. Với một lớp: competitor zero, gap/entropy zero và policy ghi rõ; không NaN. Gap giữ raw cosine units.
- Toàn bộ raw features, scores, top-k indices và prototype vectors stop-gradient. Các projection/gate parameters vẫn nhận gradient qua adaptation path.
- Không chèn nhãn thật vào top-k khi train. Reference top-k không lọc output classifier.
- `reference_topk=4`, `reference_temperature=0.1` là điểm khởi đầu, không phải hyperparameters đã được chứng minh tốt.
- Log train/dev top-1 accuracy, top-k recall, old/new breakdown; train self-inclusion có thể lạc quan. Chưa thêm leave-one-out hoặc oracle vào main method.

## 6. Gate, vị trí can thiệp và đối chứng công bằng

Implement `modules/spatial_gate.py` hoặc tên tương đương:

- Raw patch tokens `[B,N,d]`, lấy grid từ backbone/ảnh; không hardcode 14x14/768 nếu có thể suy ra đúng. ViT-B/16, 224 thường là N=196, d=768.
- Patch projection `d -> 64`, LayerNorm, GELU.
- Context projection `4d -> 64`, GELU. Nối 4 scalars, broadcast tới patches.
- Shared patch MLP `input -> 64 -> 1`, sigmoid tạo S `[B,1,Hp,Wp]`.
- Image-level MLP nhận raw CLS/context/scalars, sigmoid tạo a `[B,1,1,1]`.
- `M = a * bilinear_upsample(S, image_size)`; broadcast chung cho RGB. Hai output sigmoid khởi tạo khoảng 0.9, nên mask tích khoảng 0.81. Cùng initialization policy cho B3/A0.
- Soft bypass không skip generator/ViT compute. Không claim tiết kiệm FLOPs chỉ vì mask nhỏ.
- Gate shared train qua từng task; không tuyên bố frozen old generators đảm bảo gate không quên. Chưa thêm task-specific gates hoặc distillation.

Hook **sau khi baseline tạo/trộn local prompt và trước fusion**:

```python
local_prompt = get_prompts_like_seep(x, ...)["prompts"]  # giữ baseline cache ungated
M = gate(raw_cls, raw_patches, context, scalars)
x_prime = alpha * (x + M * local_prompt) + beta * global_output
```

- Không ghi masked prompt hoặc mask vào `stored_unified_prompts`; cache vẫn là ungated baseline prompt.
- Train/test đều tính gate từ ảnh hiện tại. Test lấy ungated cached prompt theo predicted task rồi nhân mask của ảnh hiện tại.
- `seep`: không thêm gate, M=1. `image_gate`: cùng network/parameter count với CCSP nhưng mọi conditioning chỉ từ raw features của ảnh, không prototypes/candidate IDs/class scores/entropy cạnh tranh.
- Đề xuất B3 context 4d: `[normalized CLS, normalized patch mean, normalized patch max, normalized patch std]`; bốn scalars image-only lấy từ cosine CLS-vs-patches (mean/std/max) và cosine CLS-vs-mean-patch. Ghi công thức cụ thể, xử lý zero vectors an toàn. Có thể chọn công thức image-only tương đương nếu giải thích, không feed class context dưới tên image-only.
- B3/A0 dùng cùng dimensions, scalar head, optimizer steps, reference forward count và init policy. Báo exact parameters/overhead; prototype prepass của A0 phải được tính riêng.
- Thêm identity mode M=1 và zero-local mode M=0 để diagnostic. Identity mode phải tái tạo baseline tương ứng cùng checkpoint/router/scope.
- `gate_mode=identity/zero` là override diagnostic rõ ràng; khi method=seep và không override, M luôn bằng 1. Main B3/A0 train với `gate_mode=learned`, không dùng diagnostic override để chọn kết quả đẹp trên test.
- Chỉ reference forward `no_grad`. Backbone prompted frozen weights nhưng autograd phải đi xuyên backbone về gate/current local generator.
- Log mean/std/saturation của a/S/M, norm local prompt trước gate và actual norm `alpha * M * local_prompt` để phát hiện generator tăng amplitude bù mask.

## 7. Training, dữ liệu và so sánh

- Giữ nguyên optimizer/scheduler, train mask, local-generator freezing và token evolution của baseline. Gate có param group riêng `gate_lr=0.0005`, `gate_weight_decay=0.0` lúc bắt đầu. Gate LR này là giá trị group thực tế như `lr_local`, không áp batch scaling lần nữa.
- Source gốc scale general LR: `args.lr *= global_batch_size / 256`. Với một GPU, batch 32, `--lr 0.03` thành `0.00375`. Giữ policy cho tất cả variants; log requested/resolved LR từng group, không double-scale khi resume. Không tự sửa vì tên `unscale_lr` gây nhầm.
- CUB ở Kaggle Input là read-only. Source CUB constructor hiện có download/move ảnh; thêm loader chỉ đọc cho cả raw metadata layout và pre-split `train/test`. Giữ mapping giống ImageFolder và official split. Không tự tải lại nếu dữ liệu đã có.
- Development: stratified per-class holdout 10% từ official train partition, seed cố định; lưu sample IDs. Train và prototypes loại các mẫu holdout. Đánh giá all-seen development samples chỉ để evaluation; việc giữ old validation samples là validation memory, phải khai báo rõ, không đưa vào training/loss/bank.
- Không tune/early-stop bằng official test. Pilot dùng dev; chỉ evaluate official test khi cấu hình đã khóa. Main final có thể train lại full official train với lịch cố định, `dev_fraction=0`, giống nhau cho tất cả methods.
- Không dùng checkpoint cuối đã học đủ 200 lớp rồi train gate trên toàn bộ ảnh cũ và gọi đó là replay-free CIL. B3/A0 phải học theo chuỗi task từ đầu, bank append đúng thời điểm.
- Khi đổi train/dev split hoặc lịch epoch, baseline B0 cũng phải chạy cùng setting. Checkpoint lịch sử 84.78% không phải baseline trực tiếp cho những run khác split/lịch. Chỉ tái sử dụng B0 checkpoint khi toàn bộ protocol/config/seed thật sự tương thích và có metadata xác nhận.
- Train lần lượt một seed `89359` cho B0/B3/A0 trước; seed thứ hai nếu có tín hiệu, ba paired seeds cho main results. Giữ class order như baseline; seed không tự đồng nghĩa với shuffle class order.

## 8. CLI cần implement và cấu hình chạy

Đây là **CLI đích cần implement**, các flag mới chưa tồn tại trong SEEP gốc. Dùng chung parser/helper để không copy-paste rải rác tất cả config. Bảo toàn flags legacy.

| Flag mới | Giá trị / semantics |
|---|---|
| `--method` | `seep`, `image_gate`, `ccsp`; mặc định `seep` |
| `--eval_router` | `legacy_batch`, `per_image`; mặc định legacy |
| `--eval_scope` | `legacy_head`, `seen`; mặc định legacy |
| `--reference_topk` | mặc định 4, khác `--top_k` của token prompts |
| `--reference_temperature` | mặc định 0.1, khác `--temperature` của frequency mask |
| `--gate_width` | mặc định 64 |
| `--gate_lr` / `--gate_weight_decay` | 0.0005 / 0.0 |
| `--gate_init` | 0.9 cho từng nhánh sigmoid |
| `--gate_mode` | `learned`, `identity`, `zero`; bypass diagnostic không sửa cache |
| `--dev_fraction` | mặc định 0 để backward compatible; pilot dùng 0.1 |
| `--eval_split` | `dev` hoặc `test`; `dev` yêu cầu dev_fraction>0 |
| `--max_train_batches`, `--max_eval_batches` | 0 nghĩa không giới hạn; chỉ dành debug |
| `--pretrained_checkpoint` | optional local weights đúng backbone cho cả reference/adaptation initialization |

Thêm CLI resume/evaluate checkpoint rõ ràng nếu cần, nhưng không auto-resume nhầm run khác method/config. Dùng bool parser đúng, tránh `type=bool` cho flags mới. Nếu sửa parser bool legacy, có compatibility tests; không truyền `--task_inc False` qua parser `type=bool` rồi tưởng nó là False.

Ba mức chạy, chung `num_tasks=10` để giữ 20 classes/task:

| Mức | Task thực chạy | Epoch/task | Token warm_up | LR warmup-epochs | Dữ liệu |
|---|---:|---:|---:|---:|---|
| Smoke | 2 | 1 | 0 | 0 | giới hạn 2 train/eval batches |
| Pilot sàng lọc | 3 | 10 | 2 | 1 | đầy đủ train subset, dev 10% |
| Full sau khi có tín hiệu | 10 | 30 | 5 | 5 | config đã khóa; dev hoặc final test theo mục đích |

Smoke/pilot rút gọn không là kết quả paper hoặc bằng chứng loại bỏ phương pháp. Không đổi `num_tasks=3` để chạy pilot 3 task vì như vậy làm thay đổi class split; dùng `--num_tasks_to_run 3`.

## 9. Tests và acceptance criteria

Viết tests tập trung vào rủi ro thật, dùng toy data/model khi không có GPU:

1. Legacy mode giữ logits/behavior baseline với cùng state và RNG; identity gate khôi phục baseline trên cùng router/scope. B0 per-image không được vô tình dùng M≈0.81.
2. Per-image routing chọn task giống inference từng ảnh, không đổi khi trộn/sắp xếp lại batch; token/local cùng task; ghép outputs đúng thứ tự.
3. Candidate/gate interface không nhận y/task thật; true class không được chèn vào candidates. Classifier luôn chứa mọi seen class; masking future dùng mapping đúng.
4. Bank cập nhật một lần từ new-class train subset, không từ dev/test/old/future; save/load có IDs/counts. Edge cases 1 class, top-k lớn, batch 1, class IDs không liên tục, device/dtype.
5. Gradient reference bằng 0; gradient gate/current generator hữu hạn và không-zero trong learned mode với toy loss; old frozen parameters không đổi. Identity mode có thể không có gate gradient, không dùng nó để test học gate.
6. Strict load checkpoint có dynamic generators, stored prompt buffers, gate/bank. Legacy checkpoint evaluation vẫn được. Resume không mất bank, không tạo prototype từ future data hoặc reset LR/RNG sai.
7. Metrics bằng toy accuracy matrix/old-new examples có đáp án biết trước. `R[t,j]` lưu prediction trên mọi seen class, không mask theo task j. Code cũ lưu matrix theo chiều ngược, phải ghi rõ/chuyển đúng khi export.
8. Parser smoke/pilot/full của notebook đều parse thành công. Warm-up smoke không đọc cache token chưa được tạo mà vẫn coi accuracy là kết quả; kiểm tra train/eval path và đặt token warm_up=0 như spec.

Nếu không có CUDA/dataset, không giả vờ đã train hay xác minh gain. Tests toy/static pass không thay thế GPU smoke; ghi trạng thái từng check.

## 10. Logs, checkpoints và deliverables

Log cho từng task: all-seen accuracy, task-mean accuracy, AvgIncAcc, FinalAcc, forgetting, old/new accuracy/errors, candidate top-k recall, gate/norm stats, LR groups, trainable/total params, bank bytes, timing/peak VRAM. Không tính forgetting chỉ từ một final checkpoint; T=1 để N/A.

Save checkpoint đầy đủ: model/gate/bank/counts/mapping, classifier state, optimizer/scheduler, task/epoch, actual seed và RNG states, class order, split IDs, method, protocol, base/source commit, full requested/resolved config. Phân biệt prototype memory với baseline stored prompt memory và activation memory. Log params thật, không chỉ số đã loại matcher/MLP như một log phụ của source.

Đầu ra cần có:

- Source đã implement trên branch local, diff có thể review; không refactor unrelated code.
- `docs/baseline_audit.md`, `docs/ccsp_implementation.md`, hướng dẫn chính xác train/eval/resume.
- `scripts/run_smoke.sh`, `scripts/run_pilot.sh`, `scripts/run_full.sh`; mỗi method/seed/split có output_dir riêng, fail rõ khi config mismatch.
- Tests/invariant checks, results của checks đã chạy.
- `notebooks/SEEP_CCSP_Kaggle.ipynb` với các cell tương tự phần 11, không `%%capture` lúc debug, không auto chạy full suite dài.
- `SEEP_CCSP_source.zip`: chỉ source/config/tests/scripts/docs/notebook và metadata commit; không đóng gói .git, dataset, pretrained/checkpoints, venv, logs/output. Ghi root layout để giải nén được vào Kaggle.
- Báo cáo kết thúc ngắn: thay đổi gì, file nào, tests gì, giới hạn chưa kiểm chứng, lệnh kế tiếp. Không claim cải thiện/novelty khi chưa có thực nghiệm.

## 11. Cell Kaggle và cấu hình đã chỉnh để Codex đưa vào notebook

Giữ phong cách nhiều cell + gọi Python trong venv. Dùng `uv` để tạo Python 3.9 thay vì phụ thuộc `apt install python3.9` có sẵn trong image Kaggle. Đây là lựa chọn môi trường, không thay phương pháp. Không cài lại torch cũ vào kernel mặc định.

### Cell A — môi trường riêng, chạy lại được

```python
import subprocess
import sys
from pathlib import Path

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
VENV = Path('/kaggle/working/venv')
if not (VENV / 'bin/python').exists():
    subprocess.run([sys.executable, '-m', 'uv', 'venv', '--python', '3.9', str(VENV)], check=True)
PYTHON = str(VENV / 'bin/python')
```

### Cell B — đưa source đã sửa lên Kaggle

Sau khi Codex tạo source zip, upload nó thành Kaggle Dataset rồi Add Input. **Không clone SEEP gốc rồi chạy các flags mới khi code mới chưa có trong repo đó.** Ví dụ đường dẫn dưới đây là placeholder cần sửa đúng Input:

```python
import zipfile
CODE_ZIP = Path('/kaggle/input/your-ccsp-code/SEEP_CCSP_source.zip')
PROJECT = Path('/kaggle/working/SEEP_CCSP')
assert CODE_ZIP.is_file(), 'Sửa CODE_ZIP theo Input đã upload.'
PROJECT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(CODE_ZIP) as archive:
    archive.extractall(PROJECT)
# Source zip cần đặt main.py ngay ở root archive, không lồng thêm SEEP_CCSP/.
assert (PROJECT / 'main.py').is_file()
```

Hoặc clone repo/fork và checkout branch thực sự chứa implementation mới nếu người dùng đã tự push; không giả định branch mới đã có ở remote SEEP gốc.

### Cell C — cài dependencies và kiểm tra GPU

```python
subprocess.run([sys.executable, '-m', 'uv', 'pip', 'install', '--python', PYTHON,
                '-r', str(PROJECT / 'requirements.txt')], check=True)
subprocess.run([PYTHON, '-c',
               "import torch,timm; print(torch.__version__,timm.__version__); print(torch.cuda.is_available()); assert torch.cuda.is_available()"], check=True)
```

### Cell D — lệnh chung; giữ hyperparameters gốc của người dùng

Các flags mới ở đây là acceptance target để Codex implement. Config gốc đúng tên là `cub200_10task`, không phải `cub200_RainbowPrompt` của repo/branch cũ.

```python
DATA_PATH = '/kaggle/input/your-cub200-dataset/CUB_200_2011'  # sửa đúng Input
BASE = [
    PYTHON, '-u', 'main.py', 'cub200_10task',
    '--model', 'vit_base_patch16_224', '--batch-size', '32',
    '--data-path', DATA_PATH, '--device', 'cuda:0', '--seed', '89359',
    '--num_tasks', '10', '--sched', 'cosine', '--lr', '0.03',
    '--lr_local', '0.0005', '--lr_Frequency_mask', '0.03',
    '--lgsp', 'YES', '--lgsp_type', 'LGSP',
    '--length', '20', '--top_k', '1', '--size', '10',
    '--D1', '56', '--D2', '96', '--use_linear', 'True',
    '--relation_type', 'attention', '--balancing', '0.01',
    '--e_prompt_layer_idx', *map(str, range(12)),
    '--self_attn_idx', *map(str, range(6)),
    '--temperature', '1.0', '--print_freq', '100',
    '--eval_router', 'per_image', '--eval_scope', 'seen',
    '--reference_topk', '4', '--reference_temperature', '0.1',
    '--gate_width', '64', '--gate_lr', '0.0005',
    '--gate_weight_decay', '0.0', '--gate_init', '0.9',
    '--dev_fraction', '0.1', '--eval_split', 'dev',
]

def launch(method, stage):
    assert method in ('seep', 'image_gate', 'ccsp')
    profiles = {
        'smoke': ['--epochs', '1', '--warm_up', '0', '--warmup-epochs', '0',
                  '--num_tasks_to_run', '2', '--max_train_batches', '2', '--max_eval_batches', '2'],
        'pilot': ['--epochs', '10', '--warm_up', '2', '--warmup-epochs', '1',
                  '--num_tasks_to_run', '3'],
        'full': ['--epochs', '30', '--warm_up', '5', '--warmup-epochs', '5',
                 '--num_tasks_to_run', '0'],
    }
    if stage not in profiles:
        raise ValueError(stage)
    output = f'/kaggle/working/runs/{stage}/{method}/seed89359'
    command = BASE + ['--method', method, '--output_dir', output,
                      '--trial', f'{stage}_{method}_s89359'] + profiles[stage]
    print(' '.join(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)
```

Gate CCSP và các flag mới chưa có trong source gốc, nên không gọi launcher này trước khi Codex implement/parser tests pass.

### Cell E — thứ tự chạy thực tế

```python
# Kiểm tra đường dẫn, shape, gradient, bank và checkpoint. Không diễn giải accuracy smoke.
launch('seep', 'smoke')
launch('image_gate', 'smoke')
launch('ccsp', 'smoke')
```

```python
# Chỉ chạy sau khi smoke/tests thành công. Có thể chạy từng dòng trong một session.
launch('seep', 'pilot')
launch('image_gate', 'pilot')
launch('ccsp', 'pilot')
```

Chưa auto chạy `full`. Tín hiệu pilot được đánh giá trên dev, so cả architecture controls và candidate quality; không chỉ so train loss. Nếu promising, chuyển cả ba methods sang full development cùng lịch 30 epoch; replicate seed thứ hai trước khi mở rộng sweep. Main final test cần cấu hình khóa, train/dev policy thống nhất; không âm thầm lấy launch('full') ở đây là train-full-partition vì BASE vẫn giữ `dev_fraction=0.1`.

Lưu ý: nếu tiếp tục venv đã có, xác nhận đúng Python/dependencies trước khi tái dùng. Lưu source commit, full config và `pip freeze` cùng run. Không hứa thời gian train/GPU-hours trước khi profile một epoch thực tế.

## 12. Yêu cầu cuối cùng cho Codex

Thực hiện lần lượt audit -> implementation -> invariant tests -> Kaggle smoke tooling -> source package. Tự giải quyết các lựa chọn kỹ thuật thông thường trong scope. Chỉ hỏi khi thật sự thiếu input không thể suy ra hoặc gặp mâu thuẫn ảnh hưởng setting. Nếu không có GPU/data, vẫn hoàn thiện source, tests có thể chạy và notebook; nêu rõ GPU smoke/full training chưa chạy. Không dừng ở một bản kế hoạch khác.
