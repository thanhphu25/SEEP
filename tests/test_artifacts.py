import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_syntax_no_outputs_and_training_opt_in():
    path = ROOT / 'notebooks/SEEP_CCSP_Kaggle.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    assert notebook['nbformat'] == 4
    code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    for index, cell in enumerate(code_cells):
        source = ''.join(cell['source'])
        ast.parse(source, filename=f'notebook cell {index}')
        assert cell['execution_count'] is None and cell['outputs'] == []
        assert '%%capture' not in source
    source = '\n'.join(''.join(cell['source']) for cell in code_cells)
    for flag in ('RUN_SMOKE', 'RUN_PILOT', 'RUN_RESUME', 'RUN_EVALUATE', 'RUN_FULL'):
        assert f'{flag} = False' in source
    assert "'--seed', '--python', '3.9'" in source
    assert "'scripts/verify_package.py'" in source


def test_baseline_fixture_matches_audited_hash():
    fixture = ROOT / 'tests/fixtures/audited_vision_transformer.py.txt'
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == 'db391a9d25c5adbe3976eb5a9ba2137cb524095e10f5cb714d5f1e81786bf020'


def test_archive_whitelist_and_hashes_if_built():
    archive_path = ROOT / 'SEEP_CCSP_source.zip'
    if not archive_path.exists():
        pytest.skip('Source ZIP not present in extracted runtime; run packaging on development repo')
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert 'main.py' in names and 'notebooks/SEEP_CCSP_Kaggle.ipynb' in names
        assert not any(name.startswith(('.git/', '.venv/', '.python/', '.uv-cache/', 'runs/', 'logs/')) for name in names)
        assert not any(Path(name).suffix in ('.pth', '.pt', '.npz', '.jpg') for name in names)
        metadata = json.loads(archive.read('source_metadata.json'))
        for name, expected in metadata['files_sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
