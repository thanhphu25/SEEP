import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    metadata = json.loads((ROOT / 'source_metadata.json').read_text(encoding='utf-8'))
    for name, expected in metadata['files_sha256'].items():
        path = ROOT / name
        path.resolve().relative_to(ROOT.resolve())
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Source hash mismatch: ' + name)
    print(f"Verified {len(metadata['files_sha256'])} source files; branch {metadata['branch']}")


if __name__ == '__main__':
    main()
