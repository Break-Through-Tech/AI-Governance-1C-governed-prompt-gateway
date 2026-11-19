"""Check scripts/simulate_workload.py: only safe rows are selected, sampling
is deterministic given a seed, and the saved workload is well-formed."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.simulate_workload import (
    CONTEXT_PER_QUERY, CONTEXT_POOL, build_workload, select_queries,
)

ROOT = Path(__file__).resolve().parents[1]
TEST_INPUT = ROOT / 'data/test_clean.csv'


@unittest.skipUnless(TEST_INPUT.is_file(), 'data/test_clean.csv not present')
class SelectQueriesTests(unittest.TestCase):
    def test_only_safe_rows_are_selected(self):
        import pandas as pd
        df = pd.read_csv(TEST_INPUT, encoding='utf-8-sig')
        safe_rows = set(df.index[df['label'] == 'safe'])

        queries = select_queries(TEST_INPUT, sample_size=50, seed=1)
        self.assertEqual(len(queries), 50)
        for row in queries:
            self.assertIn(row['test_row'], safe_rows)

    def test_sampling_is_deterministic_given_the_same_seed(self):
        a = select_queries(TEST_INPUT, sample_size=50, seed=7)
        b = select_queries(TEST_INPUT, sample_size=50, seed=7)
        self.assertEqual([r['test_row'] for r in a], [r['test_row'] for r in b])

    def test_different_seeds_can_select_different_rows(self):
        a = select_queries(TEST_INPUT, sample_size=50, seed=1)
        b = select_queries(TEST_INPUT, sample_size=50, seed=2)
        self.assertNotEqual([r['test_row'] for r in a], [r['test_row'] for r in b])


@unittest.skipUnless(TEST_INPUT.is_file(), 'data/test_clean.csv not present')
class BuildWorkloadTests(unittest.TestCase):
    def test_saved_workload_is_well_formed_and_reproducible(self):
        with tempfile.TemporaryDirectory() as output_dir:
            workload = build_workload(TEST_INPUT, Path(output_dir), sample_size=30, seed=42)

            self.assertEqual(len(workload['queries']), 30)
            design = workload['design']
            context_ids = set(design['context_pool'])
            self.assertEqual(context_ids, {snippet_id for snippet_id, _ in CONTEXT_POOL})

            for query in workload['queries']:
                self.assertEqual(len(query['retrieved_context_ids']), CONTEXT_PER_QUERY)
                self.assertEqual(len(set(query['retrieved_context_ids'])), CONTEXT_PER_QUERY)
                self.assertTrue(set(query['retrieved_context_ids']) <= context_ids)

            saved = Path(output_dir) / 'simulated_workload.json'
            self.assertTrue(saved.is_file())

            # Re-building with the same seed must select the exact same rows
            # and the exact same simulated retrieval per row.
            workload_again = build_workload(TEST_INPUT, Path(output_dir), sample_size=30, seed=42)
            self.assertEqual(workload['queries'], workload_again['queries'])


if __name__ == '__main__':
    unittest.main()
