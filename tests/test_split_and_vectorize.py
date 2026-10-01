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

from scripts.split_and_vectorize import (
    prepare_dataset, prompt_group, read_cleaned, split_records, vectorize,
)

ROOT = Path(__file__).resolve().parents[1]


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_rows = read_cleaned(ROOT / 'data/train_clean.csv')
        cls.test_rows = read_cleaned(ROOT / 'data/test_clean.csv')
        cls.splits = split_records(cls.train_rows, cls.test_rows)

    def test_cleaned_rows_are_preserved_and_test_is_unchanged(self):
        test_groups = {prompt_group(r) for r in self.test_rows}
        expected = Counter(tuple(r.items()) for r in self.train_rows
                           if prompt_group(r) not in test_groups)
        actual = Counter(tuple(r.items()) for name in ('train', 'val') for r in self.splits[name])
        self.assertTrue(actual == expected)
        self.assertEqual(sum(actual.values()), len(self.train_rows) - 50)
        self.assertTrue(self.splits['test'] == self.test_rows)
        self.assertEqual(len(self.test_rows), 4883)

    def test_duplicate_groups_never_cross_splits(self):
        groups = {name: {prompt_group(r) for r in rows} for name, rows in self.splits.items()}
        for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            self.assertFalse(groups[left] & groups[right])

    def test_group_split_is_stratified_and_repeatable(self):
        self.assertTrue(split_records(self.train_rows, self.test_rows) == self.splits)
        counts = {}
        for name in ('train', 'val'):
            group_labels = {prompt_group(r): r['label'] for r in self.splits[name]}
            counts[name] = Counter(group_labels.values())
        for label in ('safe', 'toxic', 'jailbreak'):
            total = counts['train'][label] + counts['val'][label]
            fraction = sum(counts['val'].values()) / sum(
                sum(count.values()) for count in counts.values())
            self.assertLessEqual(abs(counts['val'][label] - fraction * total), 1)

    def test_bad_input_fails_before_splitting(self):
        for fraction in (0, 1, -0.2, float('nan')):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                split_records(self.train_rows, self.test_rows, validation_size=fraction)
        row = self.splits['train'][0]
        conflict = {**row, 'label': 'toxic' if row['label'] == 'safe' else 'safe'}
        with self.assertRaisesRegex(ValueError, 'conflicting labels'):
            split_records(self.train_rows + [conflict], self.test_rows)
        for invalid in ({'prompt': 'wrong schema', 'label': 'safe'},
                        {'user_input': '   ', 'label': 'safe'},
                        {'user_input': 'a prompt', 'label': 'Unknown'}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                split_records(self.train_rows + [invalid], self.test_rows)

    def test_cleaned_reader_rejects_bad_csvs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train_clean.csv'
            for content in ('prompt,label\na,safe\n',
                            'user_input,label\na\n',
                            'user_input,label\na,safe,extra\n'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    read_cleaned(path)

    def test_overlap_is_removed_from_training_only(self):
        test = [{'user_input': 'FULL WIDTH', 'label': 'safe'}]
        overlap = {'user_input': '  ｆｕｌｌ  width ', 'label': 'safe'}
        splits = split_records(self.train_rows + [overlap], test)
        self.assertTrue(splits['test'] == test)
        self.assertEqual(sum(len(splits[n]) for n in ('train', 'val')), len(self.train_rows))
        self.assertTrue(all(prompt_group(r) != 'full width'
                            for name in ('train', 'val') for r in splits[name]))

    def test_vocabulary_and_idf_use_training_text_only(self):
        splits = {
            'train': [{'user_input': 'apple banana'}, {'user_input': 'apple carrot'}],
            'val': [{'user_input': 'banana validationonly'}],
            'test': [{'user_input': 'carrot testonly'}],
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
            report = prepare_dataset(ROOT / 'data/train_clean.csv',
                                     ROOT / 'data/test_clean.csv', output)
            self.assertEqual(report['excluded_training_overlap_rows'], 50)
            self.assertEqual(report['inputs']['train']['filename'], 'train_clean.csv')
            self.assertEqual(report['inputs']['test']['filename'], 'test_clean.csv')
            vectorizer = joblib.load(output / 'tfidf_vectorizer.joblib')
            for name in ('train', 'val', 'test'):
                with (output / f'{name}.csv').open(encoding='utf-8', newline='') as stream:
                    rows = list(csv.DictReader(stream))
                labels = np.load(output / f'y_{name}.npy', allow_pickle=False)
                matrix = sparse.load_npz(output / f'X_{name}.npz')
                self.assertTrue(rows == self.splits[name])
                np.testing.assert_array_equal(labels, [r['label'] for r in rows])
                expected = vectorizer.transform([r['user_input'] for r in rows])
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
