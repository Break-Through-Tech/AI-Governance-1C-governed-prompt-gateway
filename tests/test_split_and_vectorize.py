"""Check split isolation, training-only TF-IDF, and the classifier handoff."""

from collections import Counter
import csv
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.split_and_vectorize import prepare_dataset, split_records, vectorize

ROOT = Path(__file__).resolve().parents[1]


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / 'data/combined_risk_dataset.csv').open(encoding='utf-8', newline='') as stream:
            cls.rows = list(csv.DictReader(stream))
        cls.splits = split_records(cls.rows)

    def test_every_eligible_row_is_used_once_and_test_is_unchanged(self):
        expected = {r['record_id'] for r in self.rows if r['training_eligible'] == 'true'}
        actual = [r['record_id'] for name in ('train', 'val') for r in self.splits[name]]
        self.assertEqual(set(actual), expected)
        self.assertEqual(len(actual), len(expected))
        original_test = [r for r in self.rows if r['source_dataset'] == 'toxic_chat'
                         and r['source_split'] == 'test']
        self.assertTrue(self.splits['test'] == original_test)
        self.assertEqual(len(original_test), 5083)

    def test_duplicate_groups_never_cross_splits(self):
        groups = {name: {r['duplicate_group'] for r in rows} for name, rows in self.splits.items()}
        for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            self.assertFalse(groups[left] & groups[right])
        reserved = {r['duplicate_group'] for r in self.rows
                    if r['dataset_role'] in ('behavior_benchmark', 'judge_evaluation')}
        self.assertFalse((groups['train'] | groups['val']) & reserved)

    def test_group_split_is_stratified_and_repeatable(self):
        self.assertTrue(split_records(self.rows) == self.splits)
        counts = {}
        for name in ('train', 'val'):
            group_labels = {r['duplicate_group']: r['label'] for r in self.splits[name]}
            counts[name] = Counter(group_labels.values())
        for label in ('Safe', 'Toxic', 'Jailbreak'):
            total = counts['train'][label] + counts['val'][label]
            fraction = sum(counts['val'].values()) / sum(
                sum(count.values()) for count in counts.values())
            self.assertLessEqual(abs(counts['val'][label] - fraction * total), 1)

    def test_bad_input_fails_before_splitting(self):
        for fraction in (0, 1, -0.2, float('nan')):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                split_records(self.rows, validation_size=fraction)
        with self.assertRaisesRegex(ValueError, 'Duplicate record IDs'):
            split_records(self.rows + [self.rows[0]])
        eligible = next(r for r in self.rows if r['training_eligible'] == 'true')
        conflict = {**eligible, 'record_id': 'conflict',
                    'label': 'Toxic' if eligible['label'] == 'Safe' else 'Safe'}
        with self.assertRaisesRegex(ValueError, 'conflicting labels'):
            split_records(self.rows + [conflict])
        overlap = {**eligible, 'record_id': 'overlap', 'source_split': 'test',
                   'dataset_role': 'held_out_test', 'training_eligible': 'false'}
        with self.assertRaisesRegex(ValueError, 'overlap'):
            split_records(self.rows + [overlap])

    def test_vocabulary_and_idf_use_training_text_only(self):
        splits = {
            'train': [{'prompt': 'apple banana'}, {'prompt': 'apple carrot'}],
            'val': [{'prompt': 'banana validationonly'}],
            'test': [{'prompt': 'carrot testonly'}],
        }
        vectorizer, matrices = vectorize(splits)
        expected = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
        expected.fit(['apple banana', 'apple carrot'])
        self.assertEqual(vectorizer.vocabulary_, expected.vocabulary_)
        np.testing.assert_allclose(vectorizer.idf_, expected.idf_)
        self.assertNotIn('validationonly', vectorizer.vocabulary_)
        self.assertNotIn('testonly', vectorizer.vocabulary_)
        self.assertEqual(matrices['test'].shape[1], matrices['train'].shape[1])

    def test_saved_artifacts_are_aligned_and_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = prepare_dataset(ROOT / 'data/combined_risk_dataset.csv', output)
            vectorizer = joblib.load(output / 'tfidf_vectorizer.joblib')
            for name in ('train', 'val', 'test'):
                with (output / f'{name}.csv').open(encoding='utf-8', newline='') as stream:
                    rows = list(csv.DictReader(stream))
                labels = np.load(output / f'y_{name}.npy', allow_pickle=False)
                matrix = sparse.load_npz(output / f'X_{name}.npz')
                self.assertTrue(rows == self.splits[name])
                np.testing.assert_array_equal(labels, [r['label'] for r in rows])
                expected = vectorizer.transform([r['prompt'] for r in rows])
                difference = matrix - expected
                if difference.nnz:
                    self.assertLess(np.max(np.abs(difference.data)), 1e-14)
                self.assertEqual(list(matrix.shape), report['splits'][name]['shape'])
            counts = Counter(r['label'] for r in self.splits['train'])
            weights = json.loads((output / 'class_weights.json').read_text())
            for label, count in counts.items():
                self.assertAlmostEqual(weights[label], sum(counts.values()) / (3 * count))
            self.assertEqual(json.loads((output / 'split_report.json').read_text()), report)


if __name__ == '__main__':
    unittest.main()
