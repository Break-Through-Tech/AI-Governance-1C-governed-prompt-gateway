"""Check the held-out test-set evaluation, selection, and saved artifacts."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import precision_recall_fscore_support

from scripts.evaluate_baseline import (
    LABELS, evaluate_baselines, format_confusion_matrix, select_best,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data/processed/tfidf'
MODELS_DIR = ROOT / 'data/processed/models'


@unittest.skipUnless(
    (DATA_DIR / 'X_test.npz').is_file() and (MODELS_DIR / 'model_comparison.json').is_file(),
    'run scripts/split_and_vectorize.py and scripts/train_baseline_models.py first')
class EvaluateBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(tempfile.mkdtemp())
        cls.report = evaluate_baselines(DATA_DIR, MODELS_DIR, cls.output_dir)
        cls.saved_report = json.loads((cls.output_dir / 'test_evaluation.json').read_text())
        cls.X_test = load_npz(DATA_DIR / 'X_test.npz')
        cls.y_test = np.load(DATA_DIR / 'y_test.npy', allow_pickle=False)

    def test_both_saved_candidates_are_evaluated(self):
        self.assertEqual(set(self.report['candidates']), {'untuned', 'tuned'})

    def test_each_candidate_has_accuracy_and_precision_recall_f1_on_test(self):
        for key, candidate in self.report['candidates'].items():
            test = candidate['test']
            with self.subTest(candidate=key):
                self.assertGreaterEqual(test['accuracy'], 0.0)
                self.assertLessEqual(test['accuracy'], 1.0)
            for average in ('macro', 'weighted'):
                for metric in ('precision', 'recall', 'f1'):
                    with self.subTest(candidate=key, average=average, metric=metric):
                        self.assertGreaterEqual(test[average][metric], 0.0)
                        self.assertLessEqual(test[average][metric], 1.0)
            for label in LABELS:
                with self.subTest(candidate=key, label=label):
                    self.assertIn(label, test['per_label'])

    def test_confusion_matrix_is_square_and_rows_sum_to_support(self):
        total = len(self.y_test)
        for key, candidate in self.report['candidates'].items():
            confusion = candidate['test']['confusion_matrix']
            matrix = np.array(confusion['rows_true_cols_pred'])
            with self.subTest(candidate=key):
                self.assertEqual(confusion['labels'], list(LABELS))
                self.assertEqual(matrix.shape, (len(LABELS), len(LABELS)))
                self.assertEqual(int(matrix.sum()), total)
                for i, label in enumerate(LABELS):
                    self.assertEqual(int(matrix[i].sum()),
                                     candidate['test']['per_label'][label]['support'])

    def test_accuracy_equals_confusion_matrix_diagonal_over_total(self):
        for key, candidate in self.report['candidates'].items():
            matrix = np.array(candidate['test']['confusion_matrix']['rows_true_cols_pred'])
            diagonal_accuracy = matrix.trace() / matrix.sum()
            with self.subTest(candidate=key):
                self.assertAlmostEqual(candidate['test']['accuracy'],
                                       float(diagonal_accuracy), places=10)

    def test_support_matches_the_test_labels(self):
        for label in LABELS:
            self.assertEqual(self.report['test_support'][label],
                             int(np.sum(self.y_test == label)))

    def test_best_candidate_has_the_highest_test_macro_f1(self):
        best = self.report['best_by_test_macro_f1']
        best_f1 = self.report['candidates'][best]['test']['macro']['f1']
        for candidate in self.report['candidates'].values():
            self.assertLessEqual(candidate['test']['macro']['f1'], best_f1)

    def test_official_baseline_is_left_unset_for_team_review(self):
        # Deferred to the team in the doc; the report must not pick one.
        self.assertIsNone(self.report['official_baseline'])

    def test_reported_macro_f1_matches_a_fresh_prediction(self):
        import pickle
        for key, candidate in self.report['candidates'].items():
            with (MODELS_DIR / candidate['pickle']).open('rb') as stream:
                model = pickle.load(stream)
            pred = model.predict(self.X_test)
            _, _, f1, _ = precision_recall_fscore_support(
                self.y_test, pred, average='macro', zero_division=0, labels=LABELS)
            with self.subTest(candidate=key):
                self.assertAlmostEqual(candidate['test']['macro']['f1'], float(f1), places=10)

    def test_saved_artifact_matches_the_returned_report(self):
        self.assertEqual(self.saved_report, self.report)

    def test_evaluation_is_deterministic(self):
        again = evaluate_baselines(DATA_DIR, MODELS_DIR, Path(tempfile.mkdtemp()))
        self.assertEqual(again, self.report)

    def test_test_metrics_differ_from_validation_metrics(self):
        # Catches accidentally re-reporting validation numbers as test.
        for key, candidate in self.report['candidates'].items():
            with self.subTest(candidate=key):
                self.assertNotEqual(candidate['test']['macro']['f1'],
                                    candidate['validation']['macro']['f1'])

    def test_select_best_breaks_ties_by_weighted_then_minority_f1(self):
        def candidate(macro, weighted, toxic_f1, jailbreak_f1):
            return {'test': {
                'macro': {'f1': macro},
                'weighted': {'f1': weighted},
                'per_label': {'toxic': {'f1': toxic_f1}, 'jailbreak': {'f1': jailbreak_f1},
                              'safe': {'f1': 1.0}},
            }}
        # Equal macro + weighted: the higher mean toxic/jailbreak F1 wins.
        evaluations = {
            'untuned': candidate(0.70, 0.90, 0.40, 0.40),
            'tuned': candidate(0.70, 0.90, 0.60, 0.60),
        }
        self.assertEqual(select_best(evaluations), 'tuned')
        # A strictly higher macro F1 wins outright, minority F1 notwithstanding.
        evaluations['untuned'] = candidate(0.80, 0.90, 0.10, 0.10)
        self.assertEqual(select_best(evaluations), 'untuned')

    def test_format_confusion_matrix_shows_every_label(self):
        confusion = {'labels': list(LABELS),
                     'rows_true_cols_pred': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}
        rendered = format_confusion_matrix(confusion)
        for label in LABELS:
            self.assertIn(label, rendered)


if __name__ == '__main__':
    unittest.main()
