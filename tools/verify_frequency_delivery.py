"""Verify the latest-branch notebook and reconstruct its optional offline patch."""
import ast
import hashlib
import json
import os
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def checked(command, cwd):
    result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(result.stdout.decode('utf8', errors='replace'))
    return result.stdout


def verify():
    manifest = json.loads((ROOT / 'frequency_state_delivery.json').read_text(encoding='utf8'))
    nb = json.loads((ROOT / 'SEEP_Frequency_State_CUB200_Kaggle.ipynb').read_text(encoding='utf8'))
    literals = {}
    for c in nb['cells']:
        assert c.get('outputs', []) == [] and c.get('execution_count') is None
        if c['cell_type'] != 'code':
            continue
        parsed = ast.parse(''.join(c['source']))
        for node in parsed.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in ('SOURCE_URL', 'SOURCE_BRANCH', 'DATA_ROOT', 'SAVE_CHECKPOINTS'):
                    literals[node.targets[0].id] = ast.literal_eval(node.value)
    assert literals['SOURCE_URL'] == manifest['runtime_repo_url']
    assert literals['SOURCE_BRANCH'] == manifest['runtime_branch']
    assert literals['DATA_ROOT'] is None
    assert literals['SAVE_CHECKPOINTS'] is True
    code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    assert "'--branch', SOURCE_BRANCH" in code and "'git', 'clone'" in code
    assert 'PATCH_B64' not in code and 'BASE_COMMIT' not in code and "'checkout'" not in code
    patch = (ROOT / 'frequency_state_implementation.patch').read_bytes()
    assert hashlib.sha256(patch).hexdigest() == manifest['patch_sha256']
    parent = ROOT / '.delivery-check'
    parent.mkdir(exist_ok=True)
    checkout = parent / str(uuid.uuid4())
    # Local clone only: no network or modification of the user's current branch.
    checked(['git', 'clone', '--shared', '--no-checkout', str(ROOT), str(checkout)], ROOT)
    checked(['git', 'config', 'core.autocrlf', 'false'], checkout)
    checked(['git', 'checkout', '--detach', manifest['base_commit']], checkout)
    path = checkout / 'implementation.patch'
    path.write_bytes(patch)
    checked(['git', 'apply', '--check', str(path)], checkout)
    checked(['git', 'apply', str(path)], checkout)
    for name, expected in manifest['source_files_sha256'].items():
        assert hashlib.sha256((checkout / name).read_bytes()).hexdigest() == expected, name
    checked([sys.executable, 'tools/build_frequency_delivery.py'], checkout)
    for name in ('frequency_state_implementation.patch', 'SEEP_Frequency_State_CUB200_Kaggle.ipynb'):
        assert (checkout / name).read_bytes() == (ROOT / name).read_bytes(), 'Non-reproducible rebuild: ' + name
    output = ROOT / 'validation/frequency_state'
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_frequency_state.py', '-p', 'no:cacheprovider',
               '--tb=short', '--junitxml=' + str(output / 'reconstructed_tests.xml')]
    result = subprocess.run(command, cwd=checkout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env={**os.environ, 'PYTHONIOENCODING': 'utf8'})
    (output / 'reconstructed_tests.log').write_bytes(result.stdout.replace(b'\r\n', b'\n'))
    suite = ET.parse(output / 'reconstructed_tests.xml').getroot()
    totals = {key: sum(int(s.attrib.get(key, 0)) for s in suite.iter('testsuite'))
              for key in ('tests', 'failures', 'errors', 'skipped')}
    import torch, torchvision, timm
    report = dict(validation='latest-branch notebook + clean offline patch reconstruction + synthetic CPU tests',
        base_commit=manifest['base_commit'], patch_sha256=manifest['patch_sha256'],
        all_source_hashes_match=True, notebook_code_cells_parse=True, notebook_has_no_outputs=True,
        notebook_runtime_branch=literals['SOURCE_BRANCH'], notebook_uses_latest_branch_head=True,
        notebook_auto_data_discovery=True, notebook_default_save_checkpoints=literals['SAVE_CHECKPOINTS'],
        notebook_rebuild_byte_exact=True,
        returncode=result.returncode, test_counts=totals, checkpoint_files_in_delivery=False,
        full_cub_run=False, actual_environment=dict(python=sys.version, torch=torch.__version__,
            torchvision=torchvision.__version__, timm=timm.__version__, cuda_available=torch.cuda.is_available()),
        not_run=['Real CUB 55 paired task-boundary evaluations', 'Full 10x30 GPU train_collect',
                 'User checkpoint provenance/strict loading (only synthetic checkpoints tested)'])
    (output / 'delivery_validation.json').write_bytes((json.dumps(report, indent=2) + '\n').encode('utf8'))
    with zipfile.ZipFile(ROOT / 'SEEP_FREQUENCY_STATE_CUB200_delivery.zip', 'a', zipfile.ZIP_DEFLATED) as archive:
        existing = set(archive.namelist())
        for name in ('delivery_validation.json', 'reconstructed_tests.xml', 'reconstructed_tests.log'):
            target = 'validation/' + name
            if target not in existing:
                archive.write(output / name, target)
        assert not any(n.lower().endswith(('.pth', '.pt', '.ckpt')) for n in archive.namelist())
    print(json.dumps(report, indent=2))
    if result.returncode:
        print(result.stdout.decode('utf8', errors='replace'))
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    verify()
