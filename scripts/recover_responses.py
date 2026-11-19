"""Recover model_output for a cleaned split, by rejoining its prompts to
the matching split of the raw Toxic Chat zip. The cleaned CSVs
(data/test_clean.csv, data/train_clean.csv) only keep user_input/label --
earlier cleanup dropped model_output, since the classifier never needed it.
The semantic cache needs real response text to store and reuse."""

from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_ZIP_PATH = ROOT / 'data/lmsys_toxicchat.zip'
OUTPUT_DIR = ROOT / 'data/processed/semantic_cache'

SPLITS = {
    'test': {
        'clean_path': ROOT / 'data/test_clean.csv',
        'raw_member': 'lmsys_toxicchat/test.csv',
        'output_name': 'test_with_responses.csv',
    },
    'train': {
        'clean_path': ROOT / 'data/train_clean.csv',
        'raw_member': 'lmsys_toxicchat/train.csv',
        'output_name': 'train_with_responses.csv',
    },
}


def recover(clean_path: Path, zip_path: Path, raw_member: str = 'lmsys_toxicchat/test.csv') -> pd.DataFrame:
    clean = pd.read_csv(clean_path, encoding='utf-8-sig')
    if list(clean.columns) != ['user_input', 'label']:
        raise ValueError(f'Expected exactly user_input,label columns in {clean_path.name}')

    with zipfile.ZipFile(zip_path) as z:
        with z.open(raw_member) as f:
            raw = pd.read_csv(f)
    if 'user_input' not in raw.columns or 'model_output' not in raw.columns:
        raise ValueError(f'Expected user_input and model_output columns in {raw_member}')

    # look up the first model_output for each user_input in the raw split, and join it to the cleaned split
    # keeps only the first occurrence of each user_input in the raw split, in case there are duplicates
    first_response = raw.drop_duplicates(subset='user_input', keep='first').set_index('user_input')['model_output']

    missing = ~clean['user_input'].isin(first_response.index)
    if missing.any():
        raise ValueError(f'{missing.sum()} cleaned rows have no match in {raw_member}')

    result = clean.copy()
    result['model_output'] = result['user_input'].map(first_response)
    return result[['user_input', 'label', 'model_output']]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, config in SPLITS.items():
        result = recover(config['clean_path'], RAW_ZIP_PATH, config['raw_member'])
        out_path = OUTPUT_DIR / config['output_name']
        result.to_csv(out_path, index=False)
        print(f'{split}: recovered {len(result)} responses -> {out_path}')


if __name__ == '__main__':
    main()
