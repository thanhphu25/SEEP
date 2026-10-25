"""A/B are audited SEEP references; C/D come from one new CLS steering run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    from tools.summarize_dgca import matrix_metrics, HISTORICAL
except ModuleNotFoundError:
    from summarize_dgca import matrix_metrics, HISTORICAL

REFERENCE = Path(__file__).resolve().parents[1]/'references/align_only_seed3407'
KEYS = ('aia', 'final_accuracy', 'final_forgetting', 'final_old_task_mean',
        'mean_initial_accuracy', 'mean_peak_accuracy', 'bwt')


def extended_metrics(matrix):
    value = matrix_metrics(matrix)
    t = value['completed_tasks']-1
    value.update(final_old_task_mean=float(matrix[:t, t].mean()) if t else None,
                 mean_initial_accuracy=float(np.diag(matrix)[:t+1].mean()),
                 mean_peak_accuracy=float(np.mean([np.max(matrix[i, i:t+1]) for i in range(t+1)])),
                 mean_old_initial_accuracy=float(np.diag(matrix)[:t].mean()) if t else None,
                 mean_old_peak_accuracy=float(np.mean([np.max(matrix[i, i:t+1]) for i in range(t)])) if t else None,
                 bwt=float(np.mean([matrix[i, t]-matrix[i, i] for i in range(t)])) if t else None,
                 per_task=[{'task': i, 'initial': float(matrix[i, i]),
                            'peak': float(np.max(matrix[i, i:t+1])), 'final': float(matrix[i, t]),
                            'bwt': float(matrix[i, t]-matrix[i, i])} for i in range(t+1)])
    return value


def reference_matrices():
    provenance = json.loads((REFERENCE/'provenance.json').read_text())
    for name, expected in provenance['files_sha256'].items():
        if hashlib.sha256((REFERENCE/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Reference checksum differs: '+name)
    return {key: np.loadtxt(REFERENCE/name, delimiter=',') for key, name in
            [('A', 'task_accuracy_matrix_raw_seen.csv'), ('B', 'task_accuracy_matrix_dgca.csv')]}


def export_cls_metrics(output, matrices, reports, args):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = {}
    clean = {}
    for name, matrix in matrices.items():
        matrix = np.array(matrix, dtype=np.float64, copy=True)
        matrix[np.tril_indices(len(matrix), -1)] = np.nan
        clean[name] = matrix
        np.savetxt(output/f'task_accuracy_matrix_{name}.csv', matrix, delimiter=',', fmt='%.17g')
        metrics[name] = extended_metrics(matrix)
    completed = metrics['raw_seen']['completed_tasks']
    references = reference_matrices()
    # Compare partial runs to the same number of completed reference boundaries.
    matched = {k: extended_metrics(v[:completed, :completed]) for k, v in references.items()}
    table = {**matched, 'C': metrics['raw_seen'], 'D': metrics['dgca']}
    comparisons = {}
    for name, left, right in [('C-A', 'C', 'A'), ('D-B', 'D', 'B'), ('D-A', 'D', 'A'), ('D-C', 'D', 'C')]:
        comparisons[name] = {k: table[left][k]-table[right][k]
                             if table[left][k] is not None and table[right][k] is not None else None for k in KEYS}
        comparisons[name]['per_task_final'] = [l['final']-r['final']
                                               for l, r in zip(table[left]['per_task'], table[right]['per_task'])]
    status = json.loads((output/'run_status.json').read_text()) if (output/'run_status.json').exists() else {}
    protocol = (getattr(args, 'cls_steering', None) == 'task' and args.dgca == 'align_only'
                and args.num_tasks == 10 and args.epochs == 30 and args.seed == 3407
                and args.batch_size == 32 and args.cls_steering_layers == [9, 10, 11]
                and args.cls_steering_lr == .0005 and args.dgca_shrinkage == .1
                and args.dgca_samples_per_class == 128 and args.dgca_epochs == 5 and args.dgca_lr == .01
                and args.lgsp == 'YES' and args.lgsp_type == 'LGSP' and args.D1 == 56 and args.D2 == 96
                and args.task_inc is False and args.shuffle is False and args.train_mask is True
                and args.pretrained is True and args.model == 'vit_base_patch16_224'
                and args.lr == .00375 and args.lr_local == .0005 and args.lr_Frequency_mask == .03
                and args.sched == 'cosine' and args.opt == 'adam')
    audited = len(reports) == completed and all(
        r['mode'] == 'align_only' and r['transport'] is None and r['hashes_before'] == r['hashes_after']
        and r['cls_steering']['old_banks_before'] == r['cls_steering']['old_banks_after']
        and r['cls_steering']['learning']['all_finite'] and r['cls_steering']['attention']['hashes_unchanged']
        for r in reports)
    complete = bool(protocol and audited and completed == 10 and status.get('status') == 'success')
    delta = comparisons['C-A']
    priority = bool(complete and delta['aia'] >= .3 and delta['final_accuracy'] > 0 and delta['final_old_task_mean'] > 0)
    summary = {'metrics': metrics, 'evaluation_table': table, 'comparisons_pp': comparisons,
               'historical_reference_separate': HISTORICAL,
               'reference_full_metrics': {k: extended_metrics(v) for k, v in references.items()},
               'full_protocol_complete': complete, 'practical_priority': priority,
               'screening_pass': bool(priority and delta['final_forgetting'] < 0),
               'calibration_benefit_remaining_D_minus_B': bool(complete and comparisons['D-B']['aia'] > 0
                                                              and comparisons['D-B']['final_accuracy'] > 0),
               'completed_tasks': completed, 'protocol_config_matches': bool(protocol),
               'interpretation': 'C-A tests the prompt with the raw head; D-B compares calibrated heads. '
                                 'D-C is head benefit on the new representation. D-A is the complete package. '
                                 'Lower forgetting alone does not establish retention if peaks also fall. '
                                 'One seed is a screen, not a novelty or statistical claim.'}
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with (output/'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['comparison', *KEYS])
        writer.writeheader()
        writer.writerows({'comparison': k, **{n: v[n] for n in KEYS}} for k, v in comparisons.items())
    with (output/'per_task_comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['task', 'A_final', 'B_final', 'C_final', 'D_final', 'C-A', 'D-B', 'D-A', 'D-C'])
        for i in range(completed):
            writer.writerow([i, *[table[k]['per_task'][i]['final'] for k in ('A', 'B', 'C', 'D')],
                             *[comparisons[k]['per_task_final'][i] for k in ('C-A', 'D-B', 'D-A', 'D-C')]])
    for name, values in [('acc.txt', metrics['dgca']['acc_per_task']),
                         ('forgetting.txt', metrics['dgca']['forgetting_per_task'][1:])]:
        (output/name).write_text(''.join(f'{v:.17g}\n' for v in values), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    config = SimpleNamespace(**json.loads((args.output/'run_config.json').read_text()))
    matrices = {k: np.loadtxt(args.output/f'task_accuracy_matrix_{k}.csv', delimiter=',', ndmin=2)
                for k in ('dgca', 'raw_seen', 'raw_exact')}
    reports = [json.loads(p.read_text()) for p in sorted(args.output.glob('dgca_task_*.json'))]
    print(json.dumps(export_cls_metrics(args.output, matrices, reports, config), indent=2))
