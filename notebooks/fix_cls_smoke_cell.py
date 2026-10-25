# Paste this entire cell into the existing Kaggle notebook, before its failed
# preflight/smoke cell. It fixes only the smoke reload and refreshes provenance.
from pathlib import Path
import base64, hashlib, json, subprocess

REPO = Path('/kaggle/working/SEEP_CLS_STEERING')
p = REPO/'tools/smoke_dgca.py'
old = b'clone = DGCA(epochs=1, samples_per_class=4)'
new = b'clone = DGCA(**dgca.config)'
source = p.read_bytes()
assert source.count(old) == 1 or source.count(new) == 1, 'Unexpected smoke source'
p.write_bytes(source.replace(old, new))

# The notebook verifies source/patch checksums again when exporting metrics.
DELIVERY = json.loads((REPO/'cls_steering_delivery.json').read_text())
DELIVERY.setdefault('smoke_reload_hotfix', {
    'previous_patch_sha256': DELIVERY['patch_sha256'],
    'reason': 'Reload smoke DGCA using the complete saved configuration',
})
DELIVERY['source_files_sha256']['tools/smoke_dgca.py'] = hashlib.sha256(p.read_bytes()).hexdigest()
names = list(DELIVERY['source_files_sha256'])
subprocess.check_call(['git', 'add', '-N', '--', *names], cwd=REPO)
BASE_COMMIT = DELIVERY['base_commit']
patch = subprocess.check_output(['git', 'diff', '--binary', BASE_COMMIT, '--', *names], cwd=REPO)
PATCH_SHA256 = hashlib.sha256(patch).hexdigest()
PATCH_B64 = base64.b64encode(patch).decode()
DELIVERY['patch_sha256'] = PATCH_SHA256
(REPO/'cls_steering_implementation.patch').write_bytes(patch)
(REPO/'cls_steering_delivery.json').write_text(json.dumps(DELIVERY, indent=2)+'\n', encoding='utf-8')
print('Smoke reload fixed; source and patch checksums refreshed:', PATCH_SHA256)
