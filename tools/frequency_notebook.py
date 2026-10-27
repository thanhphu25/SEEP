"""Small Kaggle notebook interface; all training runs in fresh Python processes."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.frequency_setup import export


class KaggleRun:
    def __init__(self, repo, data_root=None, save_checkpoints=True, work='/kaggle/working'):
        self.repo, work = Path(repo), Path(work)
        self.output = Path(tempfile.mkdtemp(prefix='seep_frequency_state_seed3407_', dir=work))
        self.config_path = self.output / 'notebook_config.json'
        self.config = dict(run_mode='train_collect', save_checkpoints=save_checkpoints,
            source=None if data_root is None else str(data_root),
            data_path=str(work / (self.output.name + '_data')), output=str(self.output),
            npz_path=str(work / 'pretrained/ViT-B_16.npz'), device='cuda:0', num_workers=4)
        self.config_path.write_text(json.dumps(self.config, indent=2), encoding='utf8')
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.repo, text=True).strip()
        branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=self.repo, text=True).strip()
        source = dict(source_mode='latest branch HEAD at notebook startup', branch=branch, commit=commit,
            remote=subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=self.repo, text=True).strip())
        (self.output / 'source_manifest.json').write_text(json.dumps(source, indent=2), encoding='utf8')
        for name in ('README_FREQUENCY_STATE.md', 'SEEP_Frequency_State_CUB200_Kaggle.ipynb'):
            shutil.copyfile(self.repo / name, self.output / name)
        print('Source:', branch, commit)
        print('Output:', self.output)

    def _command(self, command, phase):
        tick = time.perf_counter()
        with (self.output / 'run.log').open('a', encoding='utf8') as log:
            log.write('\nPHASE ' + phase + '\n')
            log.flush()
            process = subprocess.Popen(command, cwd=self.repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf8', errors='replace',
                env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf8'})
            try:
                for line in process.stdout:
                    print(line, end='')
                    log.write(line)
                    log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                self._failed(phase, 'Command interrupted')
                raise
        (self.output / (phase + '_result.json')).write_text(json.dumps(
            dict(returncode=code, seconds=time.perf_counter() - tick)), encoding='utf8')
        if code:
            self._failed(phase, f'Command exited with code {code}')
            raise subprocess.CalledProcessError(code, command)

    def _failed(self, phase, error):
        (self.output / 'run_status.json').write_text(json.dumps(
            dict(status='failed', phase=phase, error=error, full_protocol_complete=False)), encoding='utf8')
        summary = self.output / 'summary.json'
        if summary.exists():
            value = json.loads(summary.read_text(encoding='utf8'))
            value.update(full_protocol_complete=False, process_success=False)
            summary.write_text(json.dumps(value, indent=2), encoding='utf8')
        print('Partial output:', export(self.output))

    def setup(self):
        self._command([sys.executable, 'tools/frequency_notebook.py', '--setup',
                       '--config', str(self.config_path)], 'setup')

    def train(self):
        setup = self.output / 'setup_result.json'
        if not setup.exists() or json.loads(setup.read_text(encoding='utf8'))['returncode'] != 0:
            raise RuntimeError('Run the setup cell successfully before training.')
        self._command([sys.executable, 'tools/frequency_run.py', '--config', str(self.config_path)], 'evaluation')

    def results(self):
        status = json.loads((self.output / 'run_status.json').read_text(encoding='utf8'))
        result = json.loads((self.output / 'evaluation_result.json').read_text(encoding='utf8'))
        summary = json.loads((self.output / 'summary.json').read_text(encoding='utf8'))
        if status['status'] != 'complete' or result['returncode'] != 0 or not summary['full_protocol_complete']:
            raise RuntimeError('Training is incomplete; inspect run.log in ' + str(self.output))
        print(json.dumps(summary['snapshot_minus_live_percentage_points']['raw_exact'], indent=2))
        if self.config['save_checkpoints']:
            print('Checkpoints:', self.output / 'checkpoint')
        archive = export(self.output)
        print('Metrics ZIP:', archive)
        return archive


def setup(config):
    import torch
    from tools.frequency_setup import dependencies, prepare_data
    output = Path(config['output'])
    if not torch.cuda.is_available():
        raise RuntimeError('Enable a GPU accelerator in Kaggle Settings, then rerun this cell.')
    print('GPU:', torch.cuda.get_device_name(), flush=True)
    dependencies(output)
    manifest = prepare_data(config.get('source'), config['data_path'])
    shutil.copyfile(Path(config['data_path']) / 'official_split_manifest.json', output / 'official_split_manifest.json')
    print(f"CUB ready: {manifest['train']} train / {manifest['test']} test / {manifest['classes']} classes", flush=True)
    # Use the checked-out source and preserve the real test exit code.
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_frequency_state.py',
               '-p', 'no:cacheprovider', '--junitxml=' + str(output / 'tests.xml')]
    code = subprocess.call(command, cwd=ROOT, env={**os.environ, 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'})
    (output / 'tests_result.json').write_text(json.dumps(dict(returncode=code)), encoding='utf8')
    if code:
        raise subprocess.CalledProcessError(code, command)
    print('Setup passed. Run the training cell.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--setup', action='store_true', required=True)
    parser.add_argument('--config', type=Path, required=True)
    cli = parser.parse_args()
    setup(json.loads(cli.config.read_text(encoding='utf8')))
