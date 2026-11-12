"""Check scripts/train_classifier_v2.py's data loading, feature building,
tuning, and the train/test boundary -- never fits on the test split.

These tests load the real sentence embedding model (small, CPU-fast, but
not instant), so they're skipped if the prepared TF-IDF data isn't there
to pair it with."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.train_classifier_v2 import (
    build_combined_features, load_text_splits, train_and_evaluate, tune_C,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data/processed/tfidf'


@unittest.skipUnless(DATA_DIR.is_dir(), 'data/processed/tfidf not built; run scripts/split_and_vectorize.py first')
class LoadTextSplitsTests(unittest.TestCase):
    def test_loads_train_val_test_and_matching_class_weights(self):
        splits, class_weights = load_text_splits(DATA_DIR)
        self.assertEqual(set(splits), {'train', 'val', 'test'})
        for name, df in splits.items():
            self.assertEqual(list(df.columns), ['user_input', 'label'])
        self.assertEqual(set(class_weights), {'jailbreak', 'safe', 'toxic'})


@unittest.skipUnless(DATA_DIR.is_dir(), 'data/processed/tfidf not built; run scripts/split_and_vectorize.py first')
class BuildCombinedFeaturesTests(unittest.TestCase):
    def test_combined_features_are_tfidf_columns_then_embedding_columns(self):
        from sentence_transformers import SentenceTransformer
        from scripts.train_classifier_v2 import MODEL_NAME
        splits, _ = load_text_splits(DATA_DIR)
        embedder = SentenceTransformer(MODEL_NAME)
        X = build_combined_features(splits, DATA_DIR, embedder)

        from scipy.sparse import load_npz
        tfidf_dims = load_npz(DATA_DIR / 'X_train.npz').shape[1]
        embedding_dims = embedder.get_sentence_embedding_dimension()
        for name, matrix in X.items():
            self.assertEqual(matrix.shape[0], len(splits[name]))
            self.assertEqual(matrix.shape[1], tfidf_dims + embedding_dims)


@unittest.skipUnless(DATA_DIR.is_dir(), 'data/processed/tfidf not built; run scripts/split_and_vectorize.py first')
class TuneCTests(unittest.TestCase):
    def test_tune_C_returns_a_value_from_the_grid_and_scores_every_candidate(self):
        from scripts.train_classifier_v2 import C_GRID
        rng = np.random.RandomState(0)
        # A tiny synthetic feature matrix is enough to check tune_C's
        # mechanics (grid coverage, tie-break rule) without the cost of
        # real embeddings for this test.
        X_fit = rng.normal(size=(60, 5))
        y_fit = np.array((['jailbreak'] * 20) + (['safe'] * 20) + (['toxic'] * 20))
        class_weights = {'jailbreak': 1.0, 'safe': 1.0, 'toxic': 1.0}

        best_C, candidates = tune_C(X_fit, y_fit, class_weights, seed=42)
        self.assertIn(best_C, C_GRID)
        self.assertEqual({c['C'] for c in candidates}, set(C_GRID))
        for candidate in candidates:
            self.assertIn('f1', candidate['validation']['macro'])


@unittest.skipUnless(DATA_DIR.is_dir(), 'data/processed/tfidf not built; run scripts/split_and_vectorize.py first')
class TrainAndEvaluateTests(unittest.TestCase):
    def test_report_shape_and_test_set_is_held_out(self):
        with tempfile.TemporaryDirectory() as output_dir:
            report = train_and_evaluate(DATA_DIR, output_dir, seed=42)

            self.assertEqual(report['model'], 'logistic_regression')
            self.assertIn('C', report['hyperparameters'])
            for key in ('macro', 'weighted', 'per_label'):
                self.assertIn(key, report['test'])
            self.assertIn('jailbreak_recall', report)
            self.assertIn('toxic_recall', report)
            self.assertIn('beats_baseline', report)
            self.assertIn('meets_success_criterion', report)

            # train_rows is train+validation only, never test.
            splits, _ = load_text_splits(DATA_DIR)
            expected_train_rows = len(splits['train']) + len(splits['val'])
            self.assertEqual(report['train_rows'], expected_train_rows)

            self.assertTrue((Path(output_dir) / 'classifier_v2.joblib').is_file())
            saved_report = json.loads((Path(output_dir) / 'classifier_v2_evaluation.json').read_text())
            self.assertEqual(saved_report['model'], 'logistic_regression')

    def test_is_deterministic_given_the_same_seed(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            report_a = train_and_evaluate(DATA_DIR, a, seed=42)
            report_b = train_and_evaluate(DATA_DIR, b, seed=42)
            self.assertEqual(report_a['test']['macro']['f1'], report_b['test']['macro']['f1'])
            self.assertEqual(report_a['hyperparameters']['C'], report_b['hyperparameters']['C'])


if __name__ == '__main__':
    unittest.main()
