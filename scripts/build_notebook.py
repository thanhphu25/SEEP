"""Build the reviewable, unexecuted Kaggle notebook (stdlib only)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = []
    def markdown(source):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': source.splitlines(True)})
    def code(source):
        cells.append({'cell_type': 'code', 'metadata': {}, 'source': source.splitlines(True),
                      'execution_count': None, 'outputs': []})
    markdown('''# SEEP / Image Gate / CCSP — Kaggle

Bạn là người chạy train. Notebook này chưa được chạy trên GPU/CUB thật.
Source dựa trên SEEP commit `80081081ab42769d102f83d447705da0639e2fb1`, nhánh `ccsp-pilot` của `thanhphu25/SEEP`.
Không có kết quả CCSP hay khẳng định cải thiện/novelty.

1. Import notebook này lên Kaggle, rồi **Add Input** dataset CUB-200. Cell B clone source trực tiếp từ GitHub.
2. Bật **GPU** và Internet để tạo venv/tải dependencies và pretrained weights; có thể dùng weights local.
3. Chạy A → F (setup + tests), sau đó chủ động chạy một method ở G.
4. Smoke kiểm tra kỹ thuật; pilot dùng dev 10%; không dùng official test để chọn cấu hình.
5. Không bấm Run All để tự động chạy full: các cell train đều có công tắc mặc định `False`.

Protocol chung B0/B3/A0: `per_image`, `seen`, `task_inc=False`, 10 task definitions (20 classes/task), seed 89359.
Development holdout được lưu cho các lớp cũ chỉ để evaluation; không tham gia loss/prototypes.
''')
    markdown('## A. Python 3.9 trong venv riêng')
    code('''import subprocess
import sys
from pathlib import Path

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
VENV = Path('/kaggle/working/venv')
if not (VENV / 'bin/python').exists():
    subprocess.run([sys.executable, '-m', 'uv', 'venv', '--seed', '--python', '3.9', str(VENV)], check=True)
PYTHON = str(VENV / 'bin/python')
subprocess.run([PYTHON, '-c', "import sys; print(sys.version); assert sys.version_info[:2] == (3, 9)"], check=True)
# Ensure pip exists also when reusing a venv created without --seed.
subprocess.run([sys.executable, '-m', 'uv', 'pip', 'install', '--python', PYTHON, 'pip'], check=True)
''')
    markdown('''## B. Clone source từ GitHub
Repo: https://github.com/thanhphu25/SEEP/tree/ccsp-pilot

Cell clone đúng nhánh `ccsp-pilot` vào Kaggle Working và in commit thực tế.
Chạy lại cell sẽ dùng checkout hiện có sau khi xác nhận repo/nhánh; không tự pull giữa một run.
''')
    code('''REPO_URL = 'https://github.com/thanhphu25/SEEP.git'
BRANCH = 'ccsp-pilot'
PROJECT = Path('/kaggle/working/SEEP_CCSP')
if not PROJECT.exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--single-branch', '--branch', BRANCH,
                    REPO_URL, str(PROJECT)], check=True)
assert (PROJECT / '.git').is_dir(), f'{PROJECT} đã tồn tại nhưng không phải Git checkout; dùng PROJECT khác.'
origin = subprocess.check_output(['git', '-C', str(PROJECT), 'remote', 'get-url', 'origin'], text=True).strip()
branch = subprocess.check_output(['git', '-C', str(PROJECT), 'branch', '--show-current'], text=True).strip()
assert origin.removesuffix('.git') == REPO_URL.removesuffix('.git'), f'Sai repository: {origin}'
assert branch == BRANCH, f'Sai branch: {branch}'
for relative in ('main.py', 'modules/spatial_gate.py', 'scripts/launch.py', 'requirements-test.txt'):
    assert (PROJECT / relative).is_file(), f'Branch thiếu implementation: {relative}'
SOURCE_COMMIT = subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], text=True).strip()
print(f'Source: {origin} | branch: {branch} | commit: {SOURCE_COMMIT}')
''')
    markdown('## C. Dependencies và GPU\nTorch cũ chỉ được cài trong venv, không thay torch của kernel Kaggle. Khi tái dùng venv, cell vẫn xác nhận versions.')
    code('''subprocess.run([sys.executable, '-m', 'uv', 'pip', 'install', '--python', PYTHON,
                '-r', str(PROJECT / 'requirements.txt'),
                '-r', str(PROJECT / 'requirements-test.txt')], check=True)
subprocess.run([PYTHON, '-c',
    "import torch,timm; print('torch',torch.__version__,'timm',timm.__version__); "
    "print('CUDA:',torch.cuda.is_available()); assert torch.cuda.is_available(); "
    "print(torch.cuda.get_device_name(0)); assert torch.__version__.split('+')[0]=='2.0.1'; "
    "assert timm.__version__=='0.6.11'"], check=True)
''')
    markdown('## D. Invariant tests trên toy data\nCó vài bước optimizer trên mô hình ViT nhỏ để kiểm tra gradient/resume; không train CUB hoặc tải weights. Tests CPU không thay thế GPU smoke.')
    code('''subprocess.run([PYTHON, '-m', 'pytest', 'tests', '-q',
                '--basetemp=/kaggle/working/ccsp-test-tmp'], cwd=PROJECT, check=True)
''')
    markdown('''## E. Dataset và lệnh chung

Sửa `DATA_PATH`: nhận root `CUB_200_2011`, root cha chứa nó, hoặc cây `train/` + `test/`.
Raw layout cần `images.txt`, `train_test_split.txt`, `images/` (hoặc ảnh đã chuyển vào train/test).
Không ghi vào Kaggle Input. Preprocessing giữ nguyên SEEP, không thêm normalization/clipping.

`PRETRAINED_CHECKPOINT=''` dùng nguồn pretrained gốc của backbone; nếu offline, chỉ đến `.npz` gốc hoặc `.pth` chứa đúng unprompted backbone. Cả reference và adaptation cùng load weights này. Không dùng final SEEP checkpoint làm pretrained cho B3/A0.
''')
    code('''DATA_PATH = '/kaggle/input/your-cub200-dataset/CUB_200_2011'
PRETRAINED_CHECKPOINT = ''
SEED = 89359
DEV_FRACTION = 0.1
EVAL_SPLIT = 'dev'
OUTPUT_ROOT = '/kaggle/working/runs'
assert Path(DATA_PATH).is_dir(), 'Sửa DATA_PATH theo Input CUB.'
if PRETRAINED_CHECKPOINT:
    assert Path(PRETRAINED_CHECKPOINT).is_file()

sys.path.insert(0, str(PROJECT))
from scripts.launch import base_command, run_command, launch as run_process
BASE = base_command(PYTHON, DATA_PATH, SEED, DEV_FRACTION, EVAL_SPLIT)

def command_for(method, stage, resume='', eval_checkpoint=''):
    return run_command(BASE, method, stage, OUTPUT_ROOT, resume, eval_checkpoint, PRETRAINED_CHECKPOINT)

def launch(method, stage, resume='', eval_checkpoint=''):
    command = command_for(method, stage, resume, eval_checkpoint)
    run_process(command, PROJECT)

# General LR: 0.03 * (32/256) = 0.00375; local/gate LR = 0.0005 (không scale thêm).
# warm_up = token warm-up; warmup-epochs = scheduler warm-up.
print('BASE:', ' '.join(BASE))
''')
    markdown('## F. Kiểm tra parser cho cả 9 lệnh smoke/pilot/full\nChỉ parse; không load dataset/weights hoặc train.')
    code('''for stage in ('smoke', 'pilot', 'full'):
    for method in ('seep', 'image_gate', 'ccsp'):
        command = command_for(method, stage) + ['--parse_only']
        subprocess.run(command, cwd=PROJECT, check=True)
''')
    markdown('''## G. GPU smoke — bạn bật khi sẵn sàng

Hai task, một epoch/task, tối đa hai train/eval batches, token/LR warm-up = 0.
**CCSP vẫn chạy prototype pass đầy đủ trên training subset của từng task**, nên smoke không chỉ có hai batch tổng cộng.
Accuracy smoke không phải kết quả nghiên cứu. Bắt đầu với `seep`, rồi đổi `METHOD` lần lượt sang `image_gate`, `ccsp`.
Output có manifest: không auto resume. Khi chạy lại, dùng checkpoint explicit hoặc một OUTPUT_ROOT mới.
''')
    code('''RUN_SMOKE = False
METHOD = 'seep'  # 'seep', 'image_gate', 'ccsp'
if RUN_SMOKE:
    launch(METHOD, 'smoke')
''')
    markdown('''## H. Pilot — chỉ chạy sau khi cả ba smoke pass

Ba task đầu của cùng định nghĩa 10 task; 10 epochs/task; token warm-up 2, LR warm-up 1; đầy đủ train subset, dev 10%.
Train lần lượt B0 → B3 → A0 trên seed 89359. So sánh all-seen/dev, candidate quality, gate/norm stats và old/new errors.
Không kết luận từ train loss hay dùng checkpoint baseline lịch sử khác split/lịch.
''')
    code('''RUN_PILOT = False
METHOD = 'seep'
if RUN_PILOT:
    launch(METHOD, 'pilot')
# Sau run này, đổi METHOD sang 'image_gate', rồi 'ccsp' và chạy lại cell.
''')
    markdown('''## I. Xem logs và tài nguyên

Mỗi task xuất `metrics.jsonl`, `task_stats.jsonl`, `accuracy_matrix.csv/.npy` (`R[t,j]`), checkpoint đầy đủ,
`run_config.json`, source metadata, command và `pip-freeze.txt`.
Bank bytes tách khỏi cache prompt baseline và peak CUDA allocation. Timing gồm prototype pass (cũng được log riêng).
Mask nhỏ không skip compute. Gate shared có thể quên qua task; chưa có claim về FLOPs hay gain.
''')
    code('''import json
STAGE_TO_READ = 'pilot'
METHOD_TO_READ = 'seep'
command = command_for(METHOD_TO_READ, STAGE_TO_READ)
run_dir = Path(command[command.index('--output_dir') + 1])
for filename in ('metrics.jsonl', 'task_stats.jsonl'):
    path = run_dir / filename
    if path.exists():
        print(filename)
        for line in path.read_text().splitlines():
            print(json.loads(line))
    else:
        print('Chưa có:', path)
''')
    markdown('''## J. Resume / evaluate checkpoint explicit

Giữ nguyên BASE, method, stage và split khi resume. `last.pth` lưu sau mỗi epoch và sau evaluation cuối task.
Resume khôi phục RNG, optimizer/scheduler và bank, tiếp tục epoch/task kế tiếp; không append lại prototypes của task đang dở.
Muốn diagnostic identity/zero, chỉ đổi `--gate_mode` khi evaluate; dùng output riêng để dễ đọc.
Nếu chỉ có một checkpoint legacy, AvgIncAcc/forgetting là N/A khi thiếu lịch sử.
''')
    code('''RUN_RESUME = False
RUN_EVALUATE = False
METHOD = 'ccsp'
STAGE = 'pilot'
CHECKPOINT = '/kaggle/working/runs/pilot/ccsp/seed89359/dev_holdout0.1/checkpoint/last.pth'
if RUN_RESUME:
    launch(METHOD, STAGE, resume=CHECKPOINT)
if RUN_EVALUATE:
    launch(METHOD, STAGE, eval_checkpoint=CHECKPOINT)
''')
    markdown('''## K. Full development hoặc final test sau khi khóa cấu hình

Không tự chạy full suite. Nếu pilot có tín hiệu, chạy cả ba methods 10 tasks × 30 epochs, warm-up 5/5, cùng dev policy và paired seeds.
Full mặc định bên dưới vẫn có **dev_fraction=0.1**, chưa phải train full official partition.
Final test chỉ sau khi khóa cấu hình: đổi DEV_FRACTION=0, EVAL_SPLIT='test', tạo lại BASE và output mới, rồi train lại cả B0/B3/A0 từ đầu.
Không tune/early-stop bằng test. Seed thứ hai trước khi mở rộng; main results cần ba paired seeds.
''')
    code('''RUN_FULL = False
METHOD = 'seep'
if RUN_FULL:
    launch(METHOD, 'full')
''')
    notebook = {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.9'},
        'kaggle': {'accelerator': 'gpu', 'isInternetEnabled': True, 'language': 'python', 'sourceType': 'notebook'}},
        'nbformat': 4, 'nbformat_minor': 5}
    for index, cell in enumerate(cells):
        cell['id'] = f'ccsp-{index:02d}'
    path = ROOT / 'notebooks/SEEP_CCSP_Kaggle.ipynb'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
