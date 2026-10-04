"""Train and compare baseline risk classifiers on the TF-IDF features"""

from __future__ import annotations

import argparse
from itertools import product
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['jailbreak', 'safe', 'toxic']

# Grid-searched for the tuned comparison
TUNING_GRIDS = {
    'knn': {
        'n_neighbors': [3, 5, 7, 9, 15, 25],
        'metric': ['minkowski', 'cosine'],
        'weights': ['uniform', 'distance'],
    },
    'logistic_regression': {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    },
    'svm': {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    },
    'naive_bayes': {
        'alpha': [1.0, 0.1, 0.01, 0.001, 0.0001],
    },
}


def load_splits(data_dir):
    data_dir = Path(data_dir)
    splits = {
        name: {
            'X': load_npz(data_dir / f'X_{name}.npz'),
            'y': np.load(data_dir / f'y_{name}.npy', allow_pickle=False),
        }
        for name in ('train', 'val')
    }
    for name, split in splits.items():
        #checks that every row has a corresponding label -> error if not
        if split['X'].shape[0] != split['y'].shape[0]:
            raise ValueError(f'{name}: X and y row counts do not match')

        #checks that every label is one of the expected labels -> error if not
        if set(np.unique(split['y'])) - set(LABELS):
            raise ValueError(f'{name}: unexpected label outside {LABELS}')

    class_weights = json.loads((data_dir / 'class_weights.json').read_text())

    #checks that the class weights correspond to the expected labels -> error if not
    if set(class_weights) != set(LABELS):
        raise ValueError('class_weights.json does not match the expected labels')
    return splits, class_weights


def build_models(class_weights, seed=42):
    """Build the four untrained candidate models, at default (near-default)
    hyperparameters: KNN, Logistic Regression, SVM (LinearSVC), and Naive
    Bayes.

    Logistic Regression and SVM take `class_weight` directly, so a mistake on
    a rare class (toxic/jailbreak) costs more during training than a mistake
    on safe.

    KNN and Naive Bayes have no `class_weight` parameter in scikit-learn:
    Naive Bayes is instead fit with per-sample weights derived
    from class_weights;
    KNN has no equivalent mechanism and is trained unweighted"""
    return {
        'knn': KNeighborsClassifier(n_neighbors=5),
        'logistic_regression': LogisticRegression(
            class_weight=class_weights, max_iter=2000, random_state=seed),
        'svm': LinearSVC(class_weight=class_weights, max_iter=5000, random_state=seed),
        'naive_bayes': MultinomialNB(),
    }


def build_candidate(name, params, class_weights, seed=42):
    """Build one untrained model of the given name with specific
    hyperparameters (from TUNING_GRIDS). Class-imbalance handling stays the
    same as build_models() — only the grid-searched hyperparameters vary."""
    if name == 'knn':
        return KNeighborsClassifier(**params)
    if name == 'logistic_regression':
        return LogisticRegression(class_weight=class_weights, max_iter=2000,
                                  random_state=seed, **params)
    if name == 'svm':
        return LinearSVC(class_weight=class_weights, max_iter=5000,
                         random_state=seed, **params)
    if name == 'naive_bayes':
        return MultinomialNB(**params)
    raise ValueError(f'Unknown model name: {name}')


def fit_model(name, model, X, y, class_weights):
    if name == 'naive_bayes':
        sample_weight = np.array([class_weights[label] for label in y])
        model.fit(X, y, sample_weight=sample_weight)
    else:
        model.fit(X, y)
    return model


def evaluate(model, X, y):
    pred = model.predict(X)
    # Compute precision, recall, and F1 for the predictions
    # plain unweighted average -> no weighting
    macro = precision_recall_fscore_support(y, pred, average='macro', zero_division=0, labels=LABELS)

    # weighted average -> each class contributes proportionally to its support
    weighted = precision_recall_fscore_support(y, pred, average='weighted', zero_division=0, labels=LABELS)

    # per-label breakdown -> metrics for each class individually
    per_label = precision_recall_fscore_support(y, pred, average=None, zero_division=0, labels=LABELS)
    return {
        'macro': {'precision': float(macro[0]), 'recall': float(macro[1]), 'f1': float(macro[2])},
        'weighted': {'precision': float(weighted[0]), 'recall': float(weighted[1]), 'f1': float(weighted[2])},
        'per_label': {
            label: {'precision': float(per_label[0][i]), 'recall': float(per_label[1][i]),
                    'f1': float(per_label[2][i]), 'support': int(per_label[3][i])}
            for i, label in enumerate(LABELS)
        },
    }


def select_best(results):
    """Highest validation macro F1; ties broken by weighted F1, then alphabetically by name."""
    def metric_key(name):
        validation = results[name]['validation']
        return (validation['macro']['f1'], validation['weighted']['f1'])

    best_metric = max(metric_key(name) for name in results)
    tied = [name for name in results if metric_key(name) == best_metric]
    return min(tied)


def _param_combinations(grid):
    """Every combination of a {param: [values]} grid, as individual dicts,
    in a fixed, deterministic order."""
    keys = sorted(grid)
    for values in product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def tune_model(name, class_weights, X_train, y_train, X_val, y_val, seed=42):
    """Grid search TUNING_GRIDS[name] on train, scored on validation macro F1. 
    Ties broken by weighted F1, then by the parameter
    combination itself. 
    
    Returns the best fitted model, its
    params, its validation metrics, and every candidate's score."""
    candidates = []
    for params in _param_combinations(TUNING_GRIDS[name]):
        model = build_candidate(name, params, class_weights, seed)
        fit_model(name, model, X_train, y_train, class_weights)
        candidates.append({
            'params': params,
            'model': model,
            'validation': evaluate(model, X_val, y_val),
        })

    def metric_key(candidate):
        validation = candidate['validation']
        return (validation['macro']['f1'], validation['weighted']['f1'])

    best_metric = max(metric_key(c) for c in candidates)
    tied = [c for c in candidates if metric_key(c) == best_metric]
    best = min(tied, key=lambda c: sorted(c['params'].items()))

    grid_report = [
        {'params': c['params'], 'validation': c['validation']}
        for c in sorted(candidates, key=lambda c: sorted(c['params'].items()))
    ]
    return best['model'], best['params'], best['validation'], grid_report


def tune_all_models(class_weights, splits, seed=42):
    results, fitted, grids = {}, {}, {}
    for name in TUNING_GRIDS:
        model, params, validation, grid_report = tune_model(
            name, class_weights, splits['train']['X'], splits['train']['y'],
            splits['val']['X'], splits['val']['y'], seed)
        fitted[name] = model
        results[name] = {'best_params': params, 'validation': validation}
        grids[name] = grid_report
    return results, fitted, grids


def train_and_compare(data_dir, output_dir, seed=42):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    splits, class_weights = load_splits(data_dir)

    # Untuned: issue #6's original baseline, default/near-default hyperparameters.
    models = build_models(class_weights, seed)
    # Results dictionary to store evaluation metrics for each model
    # Fitted models dictionary to store the trained models
    results, fitted = {}, {}
    for name, model in models.items():
        fit_model(name, model, splits['train']['X'], splits['train']['y'], class_weights)
        fitted[name] = model
        results[name] = {'validation': evaluate(model, splits['val']['X'], splits['val']['y'])}

    best_name = select_best(results)
    best_model = fitted[best_name]

    # Tuned: grid search per model (TUNING_GRIDS), same train/validation
    # split and the same selection metric as the untuned comparison above.
    tuned_results, tuned_fitted, tuning_grid_results = tune_all_models(class_weights, splits, seed)
    best_tuned_name = select_best(tuned_results)
    best_tuned_model = tuned_fitted[best_tuned_name]

    report = {
        'seed': seed,
        'labels': LABELS,
        'class_weights': class_weights,
        'selection_metric': 'validation macro F1',
        'selection_rule': 'highest validation macro F1; ties broken by weighted F1, then name',
        'models': results,
        'best_model': best_name,
        'tuned': {
            'tuning_grids': TUNING_GRIDS,
            'selection_rule': ('highest validation macro F1; ties broken by weighted F1, '
                               'then the parameter combination'),
            'models': tuned_results,
            'best_model': best_tuned_name,
            'grid_results': tuning_grid_results,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'best_model.pkl').open('wb') as stream:
        pickle.dump(best_model, stream)
    with (output_dir / 'best_model_tuned.pkl').open('wb') as stream:
        pickle.dump(best_tuned_model, stream)
    (output_dir / 'model_comparison.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/processed/tfidf')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed/models')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    report = train_and_compare(args.data_dir, args.output_dir, args.seed)

    print('Untuned (default hyperparameters):')
    for name, result in report['models'].items():
        m = result['validation']['macro']
        print(f"  {name}: validation macro P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    best_name = report['best_model']
    best = report['models'][best_name]
    print(f"  Best (untuned): {best_name} (validation macro F1={best['validation']['macro']['f1']:.3f})")

    print('\nTuned (grid search per model):')
    for name, result in report['tuned']['models'].items():
        m = result['validation']['macro']
        print(f"  {name}: validation macro F1={m['f1']:.3f}  best_params={result['best_params']}")
    best_tuned_name = report['tuned']['best_model']
    best_tuned = report['tuned']['models'][best_tuned_name]
    print(f"  Best (tuned): {best_tuned_name} "
          f"(validation macro F1={best_tuned['validation']['macro']['f1']:.3f})")

    print(f'\nSaved model_comparison.json, best_model.pkl (untuned), and '
          f'best_model_tuned.pkl to {args.output_dir}')
    print('Test-set evaluation is out of scope here — see issue #7, which should '
          'decide which of the above (untuned or tuned) is the official baseline.')


if __name__ == '__main__':
    main()
