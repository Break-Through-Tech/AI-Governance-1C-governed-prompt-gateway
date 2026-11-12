"""Trains a risk classifier on sentence
embeddings ALONE (no TF-IDF), to test whether that closes the toxic-class
gap a TF-IDF-only classifier leaves open.

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_baseline_models import LABELS, evaluate
from scripts.train_classifier_v2 import C_GRID, load_text_splits, tune_C

MODEL_NAME = 'all-MiniLM-L6-v2'
BASELINE_TEST_MACRO_F1 = 0.759  # (untuned SVM)
ADOPTED_V2_TEST_MACRO_F1 = 0.780  # scripts/train_classifier_v2.py (TF-IDF + embeddings, adopted)
SUCCESS_CRITERION_F1 = 0.80


def embed_splits(splits, model):
    return {
        name: np.asarray(model.encode(df['user_input'].tolist(), batch_size=64,
                                       normalize_embeddings=True, show_progress_bar=False),
                          dtype='float32')
        for name, df in splits.items()
    }


def train_and_evaluate(data_dir, output_dir, seed=42, model_name=MODEL_NAME):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    splits, class_weights = load_text_splits(data_dir)

    embedder = SentenceTransformer(model_name)
    X = embed_splits(splits, embedder)
    y = {name: df['label'].values for name, df in splits.items()}

    X_fit = np.vstack([X['train'], X['val']])
    y_fit = np.concatenate([y['train'], y['val']])

    best_C, tuning_candidates = tune_C(X_fit, y_fit, class_weights, seed)

    clf = LogisticRegression(C=best_C, class_weight=class_weights, max_iter=2000, random_state=seed)
    clf.fit(X_fit, y_fit)

    test_metrics = evaluate(clf, X['test'], y['test'])
    macro_f1 = test_metrics['macro']['f1']

    report = {
        'seed': seed,
        'labels': LABELS,
        'model': 'logistic_regression',
        'features': f'sentence-transformers/{model_name} embeddings',
        'hyperparameters': {'C': best_C},
        'tuning': {
            'grid': C_GRID,
            'selection_rule': ('highest macro F1 on an internal 85/15 split of '
                               'train+validation; ties broken by weighted F1, then smallest C'),
            'candidates': tuning_candidates,
        },
        'training_data': 'train + validation combined',
        'train_rows': int(X_fit.shape[0]),
        'test': test_metrics,
        'jailbreak_recall': test_metrics['per_label']['jailbreak']['recall'],
        'toxic_recall': test_metrics['per_label']['toxic']['recall'],
        'baseline_test_macro_f1': BASELINE_TEST_MACRO_F1,
        'adopted_v2_test_macro_f1': ADOPTED_V2_TEST_MACRO_F1,
        'success_criterion_f1': SUCCESS_CRITERION_F1,
        'beats_baseline': macro_f1 > BASELINE_TEST_MACRO_F1,
        'beats_adopted_v2': macro_f1 > ADOPTED_V2_TEST_MACRO_F1,
        'meets_success_criterion': macro_f1 >= SUCCESS_CRITERION_F1,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_dir / 'classifier_embeddings.joblib')
    (output_dir / 'classifier_embeddings_evaluation.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/processed/tfidf')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed/classifier_embeddings')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    report = train_and_evaluate(args.data_dir, args.output_dir, args.seed)

    macro = report['test']['macro']
    print(f"embeddings — logistic_regression (C={report['hyperparameters']['C']}), "
          f"trained on {report['train_rows']} rows")
    print(f"Test macro: P={macro['precision']:.3f} R={macro['recall']:.3f} F1={macro['f1']:.3f}\n")
    for label in LABELS:
        per = report['test']['per_label'][label]
        print(f"  {label:>9}: P={per['precision']:.3f} R={per['recall']:.3f} "
              f"F1={per['f1']:.3f} (support {per['support']})")
    print(f"\nBaseline (issue #7, untuned SVM) test macro F1:   {BASELINE_TEST_MACRO_F1:.3f}")
    print(f"Adopted v2 (TF-IDF + embeddings) test macro F1:   {ADOPTED_V2_TEST_MACRO_F1:.3f}")
    print(f"This (embeddings-only) test macro F1:             {macro['f1']:.3f}")
    print(f"Success criterion (80% F1): {'met' if report['meets_success_criterion'] else 'not met'}")
    print(f"Jailbreak recall: {report['jailbreak_recall']:.3f}  Toxic recall: {report['toxic_recall']:.3f}")
    print(f"\nSaved classifier_embeddings.joblib and classifier_embeddings_evaluation.json to {args.output_dir}")


if __name__ == '__main__':
    main()
