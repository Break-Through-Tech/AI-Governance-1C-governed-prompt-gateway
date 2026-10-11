"""Evaluate the Task 5 baseline models on the held-out test set (issue #7).

Issue #6 picked two winners on validation and saved them without touching test:
best_model.pkl (untuned) and best_model_tuned.pkl (tuned). This loads both,
evaluates them on test, and records the result as the baseline performance
floor. On top of issue #6's macro/weighted/per-label metrics it adds accuracy
and a confusion matrix. The test set is only measured here, never used to pick
or tune a model. `official_baseline` is left unset: the team records that in
docs/baseline_eval.md.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
# Let `python scripts/evaluate_baseline.py` find the sibling module.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_baseline_models import LABELS, evaluate

# The classes governance cares about; used to break selection ties.
MINORITY_LABELS = ('toxic', 'jailbreak')

# The two models issue #6 saved, each the winner of its own comparison.
CANDIDATE_PICKLES = {
    'untuned': 'best_model.pkl',
    'tuned': 'best_model_tuned.pkl',
}


def load_test_split(data_dir):
    data_dir = Path(data_dir)
    X = load_npz(data_dir / 'X_test.npz')
    y = np.load(data_dir / 'y_test.npy', allow_pickle=False)
    if X.shape[0] != y.shape[0]:
        raise ValueError('test: X and y row counts do not match')
    if set(np.unique(y)) - set(LABELS):
        raise ValueError(f'test: unexpected label outside {LABELS}')
    return X, y


def evaluate_on_test(model, X, y):
    """Issue #6's metrics plus accuracy and a confusion matrix (rows true,
    columns predicted, in LABELS order)."""
    pred = model.predict(X)
    metrics = evaluate(model, X, y)
    metrics['accuracy'] = float(accuracy_score(y, pred))
    metrics['confusion_matrix'] = {
        'labels': list(LABELS),
        'rows_true_cols_pred': confusion_matrix(y, pred, labels=LABELS).tolist(),
    }
    return metrics


def _minority_f1(test_metrics):
    return float(np.mean([test_metrics['per_label'][label]['f1']
                          for label in MINORITY_LABELS]))


def select_best(evaluations):
    """Highest test macro F1; ties broken by weighted F1, then mean
    toxic/jailbreak F1, then candidate key."""
    def metric_key(key):
        test = evaluations[key]['test']
        return (test['macro']['f1'], test['weighted']['f1'], _minority_f1(test))

    best_metric = max(metric_key(key) for key in evaluations)
    tied = [key for key in evaluations if metric_key(key) == best_metric]
    return min(tied)


def evaluate_baselines(data_dir, models_dir, output_dir):
    data_dir, models_dir, output_dir = Path(data_dir), Path(models_dir), Path(output_dir)
    X_test, y_test = load_test_split(data_dir)
    comparison = json.loads((models_dir / 'model_comparison.json').read_text())

    # Read each winner's identity/validation metrics from issue #6's report
    # rather than hard-coding SVM/LogReg, so a re-run stays correct.
    model_names = {
        'untuned': comparison['best_model'],
        'tuned': comparison['tuned']['best_model'],
    }
    validation = {
        'untuned': comparison['models'][model_names['untuned']]['validation'],
        'tuned': comparison['tuned']['models'][model_names['tuned']]['validation'],
    }

    evaluations = {}
    for key, pickle_name in CANDIDATE_PICKLES.items():
        with (models_dir / pickle_name).open('rb') as stream:
            model = pickle.load(stream)
        evaluations[key] = {
            'model_name': model_names[key],
            'pickle': pickle_name,
            'estimator': type(model).__name__,
            'validation': validation[key],
            'test': evaluate_on_test(model, X_test, y_test),
        }

    report = {
        'labels': list(LABELS),
        'seed': comparison.get('seed'),
        'selection_metric': 'test macro F1',
        'selection_rule': ('highest test macro F1; ties broken by weighted F1, '
                           'then mean toxic/jailbreak F1, then candidate key'),
        'success_criterion_f1': 0.80,
        'test_support': {label: int(np.sum(y_test == label)) for label in LABELS},
        'candidates': evaluations,
        'best_by_test_macro_f1': select_best(evaluations),
        'official_baseline': None,  # team records this in docs/baseline_eval.md
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'test_evaluation.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return report


def format_confusion_matrix(confusion, title=''):
    """Render a confusion-matrix dict as a labeled text table (rows true,
    columns predicted)."""
    labels = confusion['labels']
    matrix = confusion['rows_true_cols_pred']
    corner = 'true \\ pred'
    width = max([len(label) for label in labels] + [len(corner), 5])
    header = f'{corner:<{width}} | ' + ' '.join(f'{label:>{width}}' for label in labels)
    lines = [title] if title else []
    lines.append(header)
    lines.append('-' * len(header))
    for label, row in zip(labels, matrix):
        lines.append(f'{label:<{width}} | ' + ' '.join(f'{count:>{width}}' for count in row))
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/processed/tfidf')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'data/processed/models')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed/eval')
    args = parser.parse_args()
    report = evaluate_baselines(args.data_dir, args.models_dir, args.output_dir)

    print('Held-out test-set evaluation (issue #7)')
    print(f"Test support: {report['test_support']}\n")
    for key, candidate in report['candidates'].items():
        test = candidate['test']
        macro = test['macro']
        print(f"{key} — {candidate['model_name']} ({candidate['estimator']})")
        print(f"  accuracy={test['accuracy']:.3f}  "
              f"macro P={macro['precision']:.3f} R={macro['recall']:.3f} F1={macro['f1']:.3f}  "
              f"weighted F1={test['weighted']['f1']:.3f}")
        for label in report['labels']:
            per = test['per_label'][label]
            print(f"    {label:>9}: P={per['precision']:.3f} R={per['recall']:.3f} "
                  f"F1={per['f1']:.3f} (support {per['support']})")
        print(format_confusion_matrix(test['confusion_matrix'], title='  confusion matrix:'))
        print()

    best = report['best_by_test_macro_f1']
    best_f1 = report['candidates'][best]['test']['macro']['f1']
    print(f"Best by test macro F1: {best} "
          f"({report['candidates'][best]['model_name']}, F1={best_f1:.3f})")
    print(f"Project success criterion: {report['success_criterion_f1']:.0%} F1 — "
          f"{'met' if best_f1 >= report['success_criterion_f1'] else 'not met'} by the best candidate.")
    print('Official baseline is recorded in docs/baseline_eval.md after team review.')
    print(f"\nSaved test_evaluation.json to {args.output_dir}")


if __name__ == '__main__':
    main()
