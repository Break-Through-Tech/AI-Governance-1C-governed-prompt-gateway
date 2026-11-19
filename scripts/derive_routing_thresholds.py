"""Derive the issue #21 routing thresholds from training data only (no test rows).

Token thresholds: percentiles of o200k_base token counts on the training split.
Risk thresholds: percentiles of out-of-fold v2 risk scores (P(toxic) + P(jailbreak))
among prompts the out-of-fold model labels safe, i.e. the prompts governance would
pass. Out-of-fold scores are needed because v2 is trained on train + validation,
so its in-sample scores on those rows are near zero and say nothing about new prompts.

Folds group identical prompt text, so duplicates never sit on both sides of a fold.
The fold models reuse v2's features, class weights, and the C the trained v2 model
selected (read from its saved artifact).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import vstack
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from scripts.measure_token_baseline import ENCODING, count_tokens
from scripts.train_classifier_v2 import MODEL_NAME, build_combined_features, load_text_splits

ROOT = Path(__file__).resolve().parents[1]
TOKEN_PERCENTILES = (50, 75, 90, 95, 99)
RISK_PERCENTILES = (50, 75, 90, 95, 99)


def out_of_fold_risk(X, y, groups, *, C, class_weights, folds=5, seed=42):
    """Return out-of-fold (risk_score, predicted label) arrays for every row."""
    risk = np.full(len(y), np.nan)
    predicted = np.empty(len(y), dtype=object)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fit_idx, held_idx in splitter.split(X, y, groups):
        model = LogisticRegression(C=C, class_weight=class_weights, max_iter=2000, random_state=seed)
        model.fit(X[fit_idx], y[fit_idx])
        proba = model.predict_proba(X[held_idx])
        classes = list(model.classes_)
        risk[held_idx] = proba[:, classes.index('toxic')] + proba[:, classes.index('jailbreak')]
        predicted[held_idx] = np.asarray(classes)[proba.argmax(axis=1)]
    return risk, predicted


def derive(data_dir: Path, model_path: Path, output_path: Path, seed: int = 42) -> dict:
    splits, class_weights = load_text_splits(data_dir)
    train_tokens = np.array([count_tokens(t) for t in splits['train']['user_input']])

    fit = {name: splits[name] for name in ('train', 'val')}
    X = build_combined_features(fit, data_dir, SentenceTransformer(MODEL_NAME))
    X_fit = vstack([X['train'], X['val']], format='csr')
    texts = np.concatenate([fit['train']['user_input'].values, fit['val']['user_input'].values])
    y_fit = np.concatenate([fit['train']['label'].values, fit['val']['label'].values])
    _, groups = np.unique(texts, return_inverse=True)
    C = float(joblib.load(model_path).C)
    risk, predicted = out_of_fold_risk(X_fit, y_fit, groups, C=C, class_weights=class_weights, seed=seed)
    passed = risk[predicted == 'safe']

    report = {
        'purpose': 'Issue #21 routing thresholds, derived without test-set rows.',
        'token_counts': {
            'source': 'train split (data/processed/tfidf/train.csv)', 'encoding': ENCODING,
            'rows': int(len(train_tokens)),
            'percentiles': {str(p): float(np.percentile(train_tokens, p)) for p in TOKEN_PERCENTILES},
        },
        'risk_scores': {
            'source': 'out-of-fold v2 scores on train + validation, rows predicted safe',
            'folds': 5, 'grouping': 'identical prompt text', 'seed': seed, 'C': C,
            'rows': int(len(risk)), 'rows_predicted_safe': int(len(passed)),
            'percentiles': {str(p): float(np.percentile(passed, p)) for p in RISK_PERCENTILES},
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/processed/tfidf')
    parser.add_argument('--model', type=Path,
                        default=ROOT / 'data/processed/classifier_v2/classifier_v2.joblib')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/routing/routing_thresholds.json')
    args = parser.parse_args()
    print(json.dumps(derive(args.data_dir, args.model, args.output), indent=2))


if __name__ == '__main__':
    main()
