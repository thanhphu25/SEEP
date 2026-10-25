"""Full precision task-average metrics; matrix rows=eval task, columns=boundary."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

HISTORICAL = {'aia': 87.5441, 'final_accuracy': 86.4980, 'final_forgetting': 2.9515}


def matrix_metrics(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Expected square task accuracy matrix')
    completed = [t for t in range(len(matrix)) if np.isfinite(matrix[:t+1, t]).all()]
    if completed != list(range(len(completed))) or not completed:
        raise ValueError('Need consecutive completed columns')
    end = completed[-1]
    averages = [float(matrix[:t+1, t].mean()) for t in completed]
    forgetting = [0.] + [float(np.mean([
        np.max(matrix[i, i:t+1])-matrix[i, t] for i in range(t)])) for t in completed[1:]]
    return {'completed_tasks': len(completed), 'acc_per_task': averages,
            'forgetting_per_task': forgetting, 'aia': float(np.mean(averages)),
            'final_accuracy': averages[-1], 'final_forgetting': forgetting[-1]}


def export_metrics(output, matrices):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for name, matrix in matrices.items():
        matrix = np.array(matrix, dtype=np.float64, copy=True)
        matrix[np.tril_indices(len(matrix), -1)] = np.nan  # row i>column t: future task
        np.savetxt(output / f'task_accuracy_matrix_{name}.csv', matrix, delimiter=',', fmt='%.17g')
        metrics[name] = matrix_metrics(matrix)
    keys = ('aia', 'final_accuracy', 'final_forgetting')
    paired = {k: metrics['dgca'][k]-metrics['raw_seen'][k] for k in keys}
    historical = {k: metrics['dgca'][k]-HISTORICAL[k] for k in keys}
    raw_historical = {k: metrics['raw_exact'][k]-HISTORICAL[k] for k in keys}
    complete = metrics['dgca']['completed_tasks'] == len(matrices['dgca'])
    summary = {'metrics': metrics, 'dgca_minus_raw_seen': paired,
               'historical_reference': HISTORICAL, 'dgca_minus_historical': historical,
               'raw_exact_minus_historical': raw_historical, 'full_protocol_complete': complete,
               'screening_pass': complete and paired['aia'] > 0 and paired['final_accuracy'] > 0 and paired['final_forgetting'] < 0,
               'practical_priority': complete and paired['aia'] >= .3 and paired['final_accuracy'] > 0 and paired['final_forgetting'] <= -.2,
               'interpretation': 'Paired comparison is primary. Historical equivalence requires config, environment, split and weights comparison; one seed is screening only.'}
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with (output/'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['comparison', *keys])
        writer.writeheader()
        writer.writerows([{'comparison': 'dgca_minus_raw_seen', **paired},
                          {'comparison': 'dgca_minus_historical', **historical},
                          {'comparison': 'raw_exact_minus_historical', **raw_historical}])
    for file, values in [('acc.txt', metrics['dgca']['acc_per_task']),
                         ('forgetting.txt', metrics['dgca']['forgetting_per_task'][1:])]:
        (output/file).write_text(''.join(f'{value:.17g}\n' for value in values), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    matrices = {name: np.loadtxt(args.output/f'task_accuracy_matrix_{name}.csv', delimiter=',', ndmin=2)
                for name in ('dgca', 'raw_seen', 'raw_exact')}
    print(json.dumps(export_metrics(args.output, matrices), indent=2))
