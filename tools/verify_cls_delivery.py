"""Reconstruct the embedded notebook patch on a clean pinned checkout and test it."""
import argparse
import ast
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def verify(notebook_path, scratch, report_dir):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    literals = {}
    count = 0
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        assert cell['execution_count'] is None and not cell['outputs']
        source = ''.join(cell['source'])
        tree = ast.parse(source, filename=f'<cell {i}>')
        compile(tree, f'<cell {i}>', 'exec')
        count += 1
        for statement in tree.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                name = getattr(statement.targets[0], 'id', None)
                if name in ('BASE_COMMIT', 'PATCH_B64', 'DELIVERY'):
                    literals[name] = ast.literal_eval(statement.value)
    patch = base64.b64decode(literals['PATCH_B64'])
    manifest = literals['DELIVERY']
    sha = hashlib.sha256(patch).hexdigest()
    assert sha == manifest['patch_sha256']
    scratch = scratch.resolve()
    if scratch.exists():
        raise FileExistsError('Use a new empty verification checkout: '+str(scratch))
    scratch.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(['git', 'clone', '--config', 'core.autocrlf=false', '--no-hardlinks', '--no-checkout', str(root), str(scratch)])
    subprocess.check_call(['git', 'checkout', '--detach', literals['BASE_COMMIT']], cwd=scratch)
    patch_file = scratch/'cls_steering_implementation.patch'
    patch_file.write_bytes(patch)
    subprocess.check_call(['git', 'apply', '--check', str(patch_file)], cwd=scratch)
    subprocess.check_call(['git', 'apply', str(patch_file)], cwd=scratch)
    for name, expected in manifest['source_files_sha256'].items():
        assert hashlib.sha256((scratch/name).read_bytes()).hexdigest() == expected, name
    (scratch/'cls_steering_delivery.json').write_text(json.dumps(manifest, indent=2))
    report_dir.mkdir(parents=True, exist_ok=True)
    log = report_dir/'reconstructed_tests.log'
    result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_dgca.py', 'tests/test_cls_steering.py',
                             '-q', '--tb=short', '-p', 'no:cacheprovider', '--basetemp', str(scratch/'.qa-tmp'),
                             '--junitxml', str(report_dir/'reconstructed_tests.xml')],
                            cwd=scratch, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log.write_text(result.stdout, encoding='utf-8')
    report = {'notebook_sha256': hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
              'patch_sha256': sha, 'base_commit': literals['BASE_COMMIT'],
              'implementation_source_commit': manifest['implementation_source_commit'],
              'compiled_code_cells': count, 'all_outputs_cleared': True,
              'clean_base_patch_apply_check': True, 'clean_base_patch_apply_count': 1,
              'all_source_hashes_match': True, 'reconstructed_tests_returncode': result.returncode,
              'python': sys.version, 'gpu_smoke_executed_locally': False,
              'full_cub_run_executed_locally': False}
    (report_dir/'delivery_validation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(result.stdout)
    print(json.dumps(report, indent=2))
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, result.args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--notebook', type=Path, default=Path(__file__).resolve().parents[1]/'notebooks/SEEP_CLS_Steering_CUB200_Kaggle.ipynb')
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, default=Path(__file__).resolve().parents[1]/'artifacts/cls_steering')
    args = parser.parse_args()
    verify(args.notebook.resolve(), args.scratch, args.report_dir.resolve())
