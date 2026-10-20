"""Source-only archive; main.py at ZIP root. No datasets, weights, outputs or venv."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = '80081081ab42769d102f83d447705da0639e2fb1'


def source_files():
    root_files = list(ROOT.glob('*.py')) + list(ROOT.glob('*.md')) + list(ROOT.glob('requirements*.txt'))
    files = root_files + [ROOT / '.gitignore']
    for name in ('configs', 'continual_datasets', 'modules', 'scripts', 'tests', 'docs', 'notebooks'):
        files += [p for p in (ROOT / name).rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and p.suffix in ('.py', '.sh', '.txt', '.md', '.ipynb', '.json')]
    return sorted(set(files))


def main():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    files = source_files()
    metadata = {'base_commit': BASE_COMMIT, 'source_commit': commit, 'branch': branch,
                'contains_uncommitted_implementation': True, 'archive_root': 'main.py directly at archive root',
                'files_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (ROOT / 'source_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    archive_path = ROOT / 'SEEP_CCSP_source.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files + [ROOT / 'source_metadata.json']:
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f'{archive_path}: {len(files) + 1} files, {archive_path.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
