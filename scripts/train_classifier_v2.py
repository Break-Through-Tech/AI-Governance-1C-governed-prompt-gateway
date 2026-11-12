"""Train the v2 risk classifier (issue #18): Logistic Regression on TF-IDF
features concatenated with sentence embeddings, evaluated once on the
held-out test set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, load_npz, vstack
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
# Let `python scripts/train_classifier_v2.py` find the sibling module.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_baseline_models import LABELS, evaluate

MODEL_NAME = 'all-MiniLM-L6-v2'

C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

BASELINE_TEST_MACRO_F1 = 0.759
SUCCESS_CRITERION_F1 = 0.80


def load_text_splits(data_dir):
    data_dir = Path(data_dir)
    splits = {}
    for name in ('train', 'val', 'test'):
        df = pd.read_csv(data_dir / f'{name}.csv')
        if list(df.columns) != ['user_input', 'label']:
            raise ValueError(f'{name}.csv: expected exactly user_input,label columns')
        if set(df['label'].unique()) - set(LABELS):
            raise ValueError(f'{name}: unexpected label outside {LABELS}')
        splits[name] = df

    class_weights = json.loads((data_dir / 'class_weights.json').read_text())
    if set(class_weights) != set(LABELS):
        raise ValueError('class_weights.json does not match the expected labels')
    return splits, class_weights


def build_combined_features(text_splits, tfidf_dir, embedder):
    """TF-IDF vectors (already fit, task 5) concatenated with sentence
    embeddings, one block per split. Column order is TF-IDF first, then
    embedding dims -- classify_prompt.py must build single prompts the
    same way."""
    tfidf_dir = Path(tfidf_dir)
    combined = {}
    for name, df in text_splits.items():
        tfidf = load_npz(tfidf_dir / f'X_{name}.npz')
        embeddings = np.asarray(
            embedder.encode(df['user_input'].tolist(), batch_size=64,
                            normalize_embeddings=True, show_progress_bar=False),
            dtype='float32')
        if tfidf.shape[0] != embeddings.shape[0]:
            raise ValueError(f'{name}: TF-IDF and embedding row counts do not match')
        combined[name] = hstack([tfidf, csr_matrix(embeddings)], format='csr')
    return combined


def tune_C(X_fit, y_fit, class_weights, seed=42):
    """Grid-search C_GRID on a fresh internal split of the combined
    train+validation data, scored on that split's held-out portion.
    Returns the best C and every candidate's score (highest macro F1;
    ties broken by weighted F1, then the smallest C)."""
    X_tune_train, X_tune_val, y_tune_train, y_tune_val = train_test_split(
        X_fit, y_fit, test_size=0.15, random_state=seed, stratify=y_fit)

    candidates = []
    for C in C_GRID:
        model = LogisticRegression(C=C, class_weight=class_weights, max_iter=2000, random_state=seed)
        model.fit(X_tune_train, y_tune_train)
        candidates.append({'C': C, 'validation': evaluate(model, X_tune_val, y_tune_val)})

    def metric_key(candidate):
        validation = candidate['validation']
        return (validation['macro']['f1'], validation['weighted']['f1'])

    best_metric = max(metric_key(c) for c in candidates)
    tied = [c for c in candidates if metric_key(c) == best_metric]
    best = min(tied, key=lambda c: c['C'])
    return best['C'], candidates


def train_and_evaluate(data_dir, output_dir, seed=42, model_name=MODEL_NAME):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    text_splits, class_weights = load_text_splits(data_dir)
    y = {name: df['label'].values for name, df in text_splits.items()}

    embedder = SentenceTransformer(model_name)
    X = build_combined_features(text_splits, data_dir, embedder)

    # v2's training data: train + validation combined.
    X_fit = vstack([X['train'], X['val']], format='csr')
    y_fit = np.concatenate([y['train'], y['val']])

    best_C, tuning_candidates = tune_C(X_fit, y_fit, class_weights, seed)

    model = LogisticRegression(C=best_C, class_weight=class_weights, max_iter=2000, random_state=seed)
    model.fit(X_fit, y_fit)

    test_metrics = evaluate(model, X['test'], y['test'])
    macro_f1 = test_metrics['macro']['f1']

    report = {
        'seed': seed,
        'labels': LABELS,
        'model': 'logistic_regression',
        'features': f'TF-IDF (task 5) + sentence-transformers/{model_name} embeddings, concatenated',
        'hyperparameters': {'C': best_C},
        'tuning': {
            'grid': C_GRID,
            'selection_rule': ('highest macro F1 on an internal 85/15 split of train+validation; '
                               'ties broken by weighted F1, then smallest C'),
            'candidates': tuning_candidates,
        },
        'training_data': 'train + validation combined',
        'train_rows': int(X_fit.shape[0]),
        'test': test_metrics,
        'jailbreak_recall': test_metrics['per_label']['jailbreak']['recall'],
        'toxic_recall': test_metrics['per_label']['toxic']['recall'],
        'baseline_test_macro_f1': BASELINE_TEST_MACRO_F1,
        'success_criterion_f1': SUCCESS_CRITERION_F1,
        'beats_baseline': macro_f1 > BASELINE_TEST_MACRO_F1,
        'meets_success_criterion': macro_f1 >= SUCCESS_CRITERION_F1,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / 'classifier_v2.joblib')
    (output_dir / 'classifier_v2_evaluation.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/processed/tfidf')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed/classifier_v2')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    report = train_and_evaluate(args.data_dir, args.output_dir, args.seed)

    macro = report['test']['macro']
    print(f"v2 — logistic_regression (C={report['hyperparameters']['C']}, TF-IDF + embeddings), "
          f"trained on {report['train_rows']} rows")
    print(f"Test macro: P={macro['precision']:.3f} R={macro['recall']:.3f} F1={macro['f1']:.3f}")
    for label in LABELS:
        per = report['test']['per_label'][label]
        print(f"  {label:>9}: P={per['precision']:.3f} R={per['recall']:.3f} "
              f"F1={per['f1']:.3f} (support {per['support']})")
    print(f"\nBaseline (issue #7, untuned SVM) test macro F1: {BASELINE_TEST_MACRO_F1:.3f}")
    print(f"v2 test macro F1:                                {macro['f1']:.3f}  "
          f"({'beats' if report['beats_baseline'] else 'does not beat'} baseline)")
    print(f"Success criterion (80% F1): "
          f"{'met' if report['meets_success_criterion'] else 'not met'}")
    print(f"Jailbreak recall: {report['jailbreak_recall']:.3f}  "
          f"Toxic recall: {report['toxic_recall']:.3f}")
    print(f"\nSaved classifier_v2.joblib and classifier_v2_evaluation.json to {args.output_dir}")


if __name__ == '__main__':
    main()
