"""Check scripts/generate_paraphrases.py: only safe rows are selected,
sampling is deterministic given a seed, and the Gemini client is called
once per row with the response text saved (the client itself is faked --
no real API call in tests)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.generate_paraphrases import generate, select_prompts

ROOT = Path(__file__).resolve().parents[1]
RESPONSES_PATH = ROOT / 'data/processed/semantic_cache/test_with_responses.csv'


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def generate_content(self, model, contents):
        return FakeResponse(f'PARAPHRASE OF: {contents}')


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


@unittest.skipUnless(RESPONSES_PATH.is_file(), 'test_with_responses.csv not present')
class SelectPromptsTests(unittest.TestCase):
    def test_only_safe_rows_are_selected(self):
        import pandas as pd
        df = pd.read_csv(RESPONSES_PATH, encoding='utf-8-sig')
        safe_rows = set(df.index[df['label'] == 'safe'])

        rows = select_prompts(RESPONSES_PATH, sample_size=10, seed=1)
        self.assertEqual(len(rows), 10)
        for row in rows:
            self.assertIn(row['test_row'], safe_rows)

    def test_sampling_is_deterministic_given_the_same_seed(self):
        a = select_prompts(RESPONSES_PATH, sample_size=10, seed=7)
        b = select_prompts(RESPONSES_PATH, sample_size=10, seed=7)
        self.assertEqual([r['test_row'] for r in a], [r['test_row'] for r in b])


@unittest.skipUnless(RESPONSES_PATH.is_file(), 'test_with_responses.csv not present')
class GenerateTests(unittest.TestCase):
    def test_saves_one_paraphrase_per_sampled_row_using_the_given_client(self):
        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / 'paraphrases.json'
            records = generate(
                responses_path=RESPONSES_PATH, output_path=output_path,
                sample_size=5, seed=42, delay=0, client=FakeClient(),
            )

            self.assertEqual(len(records), 5)
            for record in records:
                self.assertTrue(record['paraphrase'].startswith('PARAPHRASE OF:'))
                self.assertIn(record['original'], record['paraphrase'])
            self.assertTrue(output_path.is_file())


if __name__ == '__main__':
    unittest.main()
