import numpy as np


def continual_metrics(matrix, task, sizes):
    """R[t,j]: accuracy after training t, evaluated on task j against ALL seen classes."""
    row = np.asarray(matrix)[task, :task + 1]
    weights = np.asarray(sizes[:task + 1], dtype=float)
    mean = float(row.mean())
    all_seen = float(np.average(row, weights=weights))
    history = [float(np.average(matrix[t, :t + 1], weights=np.asarray(sizes[:t + 1])))
               for t in range(task + 1) if np.isfinite(matrix[t, :t + 1]).all()]
    complete_history = all(np.isfinite(matrix[t, :t + 1]).all() for t in range(task + 1))
    forgetting = None if task == 0 or not complete_history else float(np.mean([
        np.max(matrix[j:task + 1, j]) - matrix[task, j] for j in range(task)]))
    old = None if task == 0 else float(np.average(row[:-1], weights=weights[:-1]))
    return {'all_seen_accuracy': all_seen, 'task_mean_accuracy': mean,
            'AvgIncAcc': float(np.mean(history)) if complete_history else None, 'FinalAcc': all_seen,
            'forgetting': forgetting, 'old_accuracy': old, 'new_accuracy': float(row[-1]),
            'old_errors': None if old is None else float(np.sum(weights[:-1] * (1 - row[:-1] / 100))),
            'new_errors': float(weights[-1] * (1 - row[-1] / 100))}
