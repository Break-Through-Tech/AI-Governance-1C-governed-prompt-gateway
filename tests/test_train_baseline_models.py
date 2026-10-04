"""Check model comparison, best-model selection, and the saved artifacts."""

import json
from pathlib import Path
import pickle
import shutil
import tempfile
import unittest

import numpy as np
from scipy.sparse import load_npz

from scripts.train_baseline_models import (
    LABELS, TUNING_GRIDS, _param_combinations, select_best, train_and_compare,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data/processed/tfidf'


@unittest.skipUnless(DATA_DIR.is_dir(), 'run scripts/split_and_vectorize.py first')
class BaselineModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Not a `with TemporaryDirectory()` block: that would delete
        # cls.output_dir the moment setUpClass returns, before any test
        # method (which reads files from it directly) gets to run.
        cls.output_dir = Path(tempfile.mkdtemp())
        cls.report = train_and_compare(DATA_DIR, cls.output_dir)
        with (cls.output_dir / 'best_model.pkl').open('rb') as stream:
            cls.best_model = pickle.load(stream)
        cls.saved_report = json.loads((cls.output_dir / 'model_comparison.json').read_text())

    def test_at_least_three_of_the_four_candidate_models_are_trained(self):
        candidates = {'knn', 'logistic_regression', 'svm', 'naive_bayes'}
        self.assertGreaterEqual(len(set(self.report['models']) & candidates), 3)

    def test_every_model_has_precision_recall_f1_on_validation(self):
        for name, result in self.report['models'].items():
            for average in ('macro', 'weighted'):
                metrics = result['validation'][average]
                for key in ('precision', 'recall', 'f1'):
                    with self.subTest(model=name, average=average, metric=key):
                        self.assertGreaterEqual(metrics[key], 0.0)
                        self.assertLessEqual(metrics[key], 1.0)
            for label in LABELS:
                with self.subTest(model=name, label=label):
                    self.assertIn(label, result['validation']['per_label'])

    def test_best_model_has_the_highest_validation_macro_f1(self):
        best_name = self.report['best_model']
        best_f1 = self.report['models'][best_name]['validation']['macro']['f1']
        for name, result in self.report['models'].items():
            self.assertLessEqual(result['validation']['macro']['f1'], best_f1)

    def test_train_and_compare_succeeds_without_any_test_split_files_present(self):
        # Proves (not just asserts) that this script never needs or reads
        # X_test.npz / y_test.npy: they are not even copied into this dir.
        with tempfile.TemporaryDirectory() as directory:
            no_test_dir = Path(directory)
            for filename in ('X_train.npz', 'y_train.npy', 'X_val.npz',
                             'y_val.npy', 'class_weights.json'):
                shutil.copy(DATA_DIR / filename, no_test_dir / filename)
            report = train_and_compare(no_test_dir, Path(tempfile.mkdtemp()))
            self.assertEqual(report['best_model'], self.report['best_model'])

    def test_selection_is_deterministic(self):
        report_again = train_and_compare(DATA_DIR, Path(tempfile.mkdtemp()))
        self.assertEqual(report_again['best_model'], self.report['best_model'])
        for name in self.report['models']:
            self.assertEqual(report_again['models'][name]['validation'],
                             self.report['models'][name]['validation'])

    def test_select_best_breaks_ties_by_weighted_f1_then_name(self):
        results = {
            'b': {'validation': {'macro': {'f1': 0.5}, 'weighted': {'f1': 0.6}}},
            'a': {'validation': {'macro': {'f1': 0.5}, 'weighted': {'f1': 0.6}}},
            'c': {'validation': {'macro': {'f1': 0.5}, 'weighted': {'f1': 0.7}}},
        }
        self.assertEqual(select_best(results), 'c')
        results['c']['validation']['weighted']['f1'] = 0.6
        self.assertEqual(select_best(results), 'a')

    def test_saved_artifacts_match_the_returned_report(self):
        self.assertEqual(self.saved_report, self.report)
        X_val = load_npz(DATA_DIR / 'X_val.npz')
        y_val = np.load(DATA_DIR / 'y_val.npy', allow_pickle=False)
        pred = self.best_model.predict(X_val)
        best_name = self.report['best_model']
        reported_f1 = self.report['models'][best_name]['validation']['macro']['f1']
        from sklearn.metrics import precision_recall_fscore_support
        _, _, f1, _ = precision_recall_fscore_support(
            y_val, pred, average='macro', zero_division=0, labels=LABELS)
        self.assertAlmostEqual(reported_f1, float(f1), places=10)

    def test_param_combinations_is_the_full_cartesian_product(self):
        grid = {'a': [1, 2], 'b': ['x', 'y', 'z']}
        combos = list(_param_combinations(grid))
        self.assertEqual(len(combos), 6)
        self.assertEqual(len(set(tuple(sorted(c.items())) for c in combos)), 6)
        for combo in combos:
            self.assertEqual(set(combo), {'a', 'b'})

    def test_every_model_has_a_tuning_grid_and_full_grid_results(self):
        tuned = self.report['tuned']
        self.assertEqual(set(tuned['models']), set(TUNING_GRIDS))
        for name, grid in TUNING_GRIDS.items():
            expected_combo_count = len(list(_param_combinations(grid)))
            with self.subTest(model=name):
                self.assertEqual(len(tuned['grid_results'][name]), expected_combo_count)
                self.assertIn('best_params', tuned['models'][name])
                self.assertIn(tuned['models'][name]['best_params'],
                              [g['params'] for g in tuned['grid_results'][name]])

    def test_tuned_grid_includes_the_untuned_default_so_it_never_scores_worse(self):
        for name in TUNING_GRIDS:
            untuned_f1 = self.report['models'][name]['validation']['macro']['f1']
            tuned_f1 = self.report['tuned']['models'][name]['validation']['macro']['f1']
            with self.subTest(model=name):
                self.assertGreaterEqual(tuned_f1, untuned_f1 - 1e-12)

    def test_tuned_best_model_is_saved_and_matches_the_report(self):
        with (self.output_dir / 'best_model_tuned.pkl').open('rb') as stream:
            tuned_model = pickle.load(stream)
        X_val = load_npz(DATA_DIR / 'X_val.npz')
        y_val = np.load(DATA_DIR / 'y_val.npy', allow_pickle=False)
        pred = tuned_model.predict(X_val)
        best_tuned_name = self.report['tuned']['best_model']
        reported_f1 = self.report['tuned']['models'][best_tuned_name]['validation']['macro']['f1']
        from sklearn.metrics import precision_recall_fscore_support
        _, _, f1, _ = precision_recall_fscore_support(
            y_val, pred, average='macro', zero_division=0, labels=LABELS)
        self.assertAlmostEqual(reported_f1, float(f1), places=10)

    def test_report_keeps_untuned_and_tuned_as_two_separate_winners(self):
        # The report must not collapse untuned and tuned into one answer —
        # each keeps its own winner, so neither silently overwrites the other.
        self.assertIn('best_model', self.report)           # untuned winner
        self.assertIn('best_model', self.report['tuned'])  # tuned winner


if __name__ == '__main__':
    unittest.main()
