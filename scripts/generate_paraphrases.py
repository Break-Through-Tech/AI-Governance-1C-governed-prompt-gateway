"""Generate genuine AI-paraphrased versions of real safe test prompts, to
re-test the semantic cache's hit rate on paraphrase robustness rather than
near-identical duplicate text.

scripts/evaluate_semantic_cache.py's duplicate-ground-truth check only
measures hit rate on near-identical normalized text (its 115 duplicate
groups are near-exact repeats) -- genuine reworded paraphrases were never
tested. This calls the Gemini API once per sampled prompt to produce a real
paraphrase, saved for scripts/evaluate_semantic_cache_paraphrases.py to
re-test the cache against.

Requires GEMINI_API_KEY or GOOGLE_API_KEY to be set in the environment.
Never hardcode a key in this file or any committed file.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import errors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESPONSES_PATH = ROOT / 'data/processed/semantic_cache/test_with_responses.csv'
OUTPUT_PATH = ROOT / 'data/processed/semantic_cache/paraphrases.json'

MODEL_NAME = 'gemini-3.5-flash-lite'
SAMPLE_SIZE = 100
SEED = 42
# Free tier allows 15 requests/minute for this model; 5s keeps us under that
# with margin instead of hitting 429s.
REQUEST_DELAY_SECONDS = 5.0
MAX_RETRIES = 4
RATE_LIMIT_RETRY_SECONDS = 60.0

PROMPT_TEMPLATE = (
    "Reword the following customer message in different words, keeping its "
    "exact meaning and intent unchanged. Output only the reworded message, "
    "with no quotes, labels, or explanation.\n\nMessage: {text}"
)


def select_prompts(responses_path=RESPONSES_PATH, sample_size=SAMPLE_SIZE, seed=SEED):
    df = pd.read_csv(responses_path, encoding='utf-8-sig')
    safe = df[df['label'] == 'safe'].reset_index().rename(columns={'index': 'test_row'})
    if len(safe) < sample_size:
        raise ValueError(f'Only {len(safe)} safe rows available, need {sample_size}')
    sampled = safe.sample(n=sample_size, random_state=seed).sort_values('test_row')
    return sampled[['test_row', 'user_input']].to_dict('records')


def paraphrase_one(client, text, model_name=MODEL_NAME, max_retries=MAX_RETRIES):
    prompt = PROMPT_TEMPLATE.format(text=text)
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            return response.text.strip()
        except errors.ClientError as exc:
            if exc.code != 429 or attempt == max_retries - 1:
                raise
            print(f'  rate limited, waiting {RATE_LIMIT_RETRY_SECONDS:.0f}s before retrying...')
            time.sleep(RATE_LIMIT_RETRY_SECONDS)
    raise RuntimeError('unreachable')


def generate(responses_path=RESPONSES_PATH, output_path=OUTPUT_PATH,
             sample_size=SAMPLE_SIZE, seed=SEED, delay=REQUEST_DELAY_SECONDS,
             client=None):
    if client is None:
        if not os.environ.get('GEMINI_API_KEY') and not os.environ.get('GOOGLE_API_KEY'):
            raise RuntimeError('Set GEMINI_API_KEY or GOOGLE_API_KEY before running this script.')
        client = genai.Client()

    rows = select_prompts(responses_path, sample_size, seed)

    records = []
    for i, row in enumerate(rows):
        paraphrase = paraphrase_one(client, row['user_input'])
        records.append({
            'test_row': int(row['test_row']),
            'original': row['user_input'],
            'paraphrase': paraphrase,
        })
        print(f"[{i + 1}/{len(rows)}] test_row={row['test_row']} done")
        if i < len(rows) - 1:
            time.sleep(delay)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        'model_name': MODEL_NAME,
        'sample_size': sample_size,
        'seed': seed,
        'source': str(responses_path.relative_to(ROOT)) if responses_path.is_relative_to(ROOT) else str(responses_path),
        'source_label_filter': 'safe',
        'records': records,
    }, indent=2) + '\n', encoding='utf-8')
    return records


def main():
    records = generate()
    print(f"\nGenerated {len(records)} paraphrases. Saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
