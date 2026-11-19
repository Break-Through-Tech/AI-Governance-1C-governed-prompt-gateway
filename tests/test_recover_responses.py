"""Check scripts/recover_responses.py: the join finds the right response,
duplicates in the raw split keep the first occurrence, and a missing match
raises instead of silently producing a blank response."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from scripts.recover_responses import recover


def write_zip(zip_path, member_name, raw_df):
    csv_path = Path(tempfile.mkstemp(suffix='.csv')[1])
    raw_df.to_csv(csv_path, index=False)
    with zipfile.ZipFile(zip_path, 'w') as z:
        z.write(csv_path, arcname=member_name)
    csv_path.unlink()


class RecoverTests(unittest.TestCase):
    def test_joins_the_first_matching_response_by_exact_text(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            clean_path = directory / 'clean.csv'
            pd.DataFrame({'user_input': ['hello', 'bye'], 'label': ['safe', 'safe']}).to_csv(clean_path, index=False)

            zip_path = directory / 'raw.zip'
            write_zip(zip_path, 'split/test.csv', pd.DataFrame({
                'user_input': ['hello', 'hello', 'bye'],
                'model_output': ['hi there', 'a later duplicate response', 'see ya'],
            }))

            result = recover(clean_path, zip_path, raw_member='split/test.csv')
            self.assertEqual(list(result.columns), ['user_input', 'label', 'model_output'])
            self.assertEqual(result.set_index('user_input')['model_output']['hello'], 'hi there')
            self.assertEqual(result.set_index('user_input')['model_output']['bye'], 'see ya')

    def test_raises_if_a_cleaned_row_has_no_match_in_the_raw_split(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            clean_path = directory / 'clean.csv'
            pd.DataFrame({'user_input': ['hello', 'unmatched'], 'label': ['safe', 'safe']}).to_csv(clean_path, index=False)

            zip_path = directory / 'raw.zip'
            write_zip(zip_path, 'split/test.csv', pd.DataFrame({
                'user_input': ['hello'], 'model_output': ['hi there'],
            }))

            with self.assertRaises(ValueError):
                recover(clean_path, zip_path, raw_member='split/test.csv')

    def test_rejects_a_clean_csv_with_unexpected_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            clean_path = directory / 'clean.csv'
            pd.DataFrame({'user_input': ['hello'], 'label': ['safe'], 'extra': [1]}).to_csv(clean_path, index=False)
            zip_path = directory / 'raw.zip'
            write_zip(zip_path, 'split/test.csv', pd.DataFrame({'user_input': ['hello'], 'model_output': ['hi']}))

            with self.assertRaises(ValueError):
                recover(clean_path, zip_path, raw_member='split/test.csv')


if __name__ == '__main__':
    unittest.main()
