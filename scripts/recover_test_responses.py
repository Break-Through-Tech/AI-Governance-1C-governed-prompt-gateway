"""Recover model_output for the cleaned test set, for the semantic cache task"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEST_CLEAN_PATH = ROOT / 'data/test_clean.csv'
RAW_ZIP_PATH = ROOT / 'data/lmsys_toxicchat.zip'
OUTPUT_DIR = ROOT / 'data/processed/semantic_cache'


def recover(test_clean_path: Path, zip_path: Path) -> pd.DataFrame:
    clean = pd.read_csv(test_clean_path, encoding='utf-8-sig')
    if list(clean.columns) != ['user_input', 'label']:
        raise ValueError('Expected exactly user_input,label columns in test_clean.csv')

    with zipfile.ZipFile(zip_path) as z:
        with z.open('lmsys_toxicchat/test.csv') as f:
            raw = pd.read_csv(f)
    if 'user_input' not in raw.columns or 'model_output' not in raw.columns:
        raise ValueError('Expected user_input and model_output columns in the raw test split')

    # look up the first model_output for each user_input in the raw test split, and join it to the cleaned test set
    # keeps only the first occurrence of each user_input in the raw test split, in case there are duplicates
    # {user_input: model_output}
    first_response = raw.drop_duplicates(subset='user_input', keep='first').set_index('user_input')['model_output']

    missing = ~clean['user_input'].isin(first_response.index)
    if missing.any():
        raise ValueError(f'{missing.sum()} cleaned test rows have no match in the raw test split')

    result = clean.copy()
    result['model_output'] = result['user_input'].map(first_response)
    return result[['user_input', 'label', 'model_output']]


def main() -> None:
    result = recover(TEST_CLEAN_PATH, RAW_ZIP_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'test_with_responses.csv'
    result.to_csv(out_path, index=False)


if __name__ == '__main__':
    main()