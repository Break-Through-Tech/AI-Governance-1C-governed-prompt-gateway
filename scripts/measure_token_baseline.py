"""Measure cleaned test prompt input tokens and reference cost for issue #8."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
from importlib.metadata import version
import json
import io
from pathlib import Path
import statistics

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'gpt-4o-mini-2024-07-18'
ENCODING = 'o200k_base'
RATE = Decimal('0.15')
PRICING_URL = 'https://developers.openai.com/api/docs/models/gpt-4o-mini'
PRICING_CHECKED = '2026-10-07'
FIELDS = ['prompt_id', 'test_row', 'prompt_sha256', 'label', 'original_tokens',
          'baseline_input_cost_usd', 'reference_model', 'encoding']


def count_tokens(text: str) -> int:
    """Count literal prompt text, including strings resembling special tokens."""
    return len(tiktoken.get_encoding(ENCODING).encode_ordinary(text))


def measure(input_path: Path, output_dir: Path) -> dict:
    # Snapshot bytes once: checksum and parsed rows refer to the same input.
    raw = input_path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
    if reader.fieldnames != ['user_input', 'label']:
        raise ValueError('Expected exactly user_input,label columns')
    source = list(reader)
    if not source:
        raise ValueError('Test set is empty')
    records = []
    for index, row in enumerate(source):
        if (None in row or any(v is None for v in row.values())
                or not row['user_input'].strip()
                or row['label'] not in {'safe', 'toxic', 'jailbreak'}):
            raise ValueError(f'Invalid cleaned test row {index}')
        text = row['user_input']
        tokens = count_tokens(text)
        records.append(dict(zip(FIELDS, [f'test-{index:06d}', index,
            hashlib.sha256(text.encode('utf-8')).hexdigest(), row['label'], tokens,
            format(Decimal(tokens) * RATE / Decimal(1000000), '.12f'), MODEL, ENCODING])))
    counts = [r['original_tokens'] for r in records]
    total_cost = Decimal(sum(counts)) * RATE / Decimal(1000000)
    report = {
        'input_filename': input_path.name,
        'input_sha256': hashlib.sha256(raw).hexdigest(),
        'prompt_count': len(records), 'label_counts': dict(sorted(Counter(r['label'] for r in records).items())),
        'reference_model': MODEL, 'encoding': ENCODING,
        'tiktoken_version': version('tiktoken'),
        'pricing': {'input_usd_per_million_tokens': str(RATE), 'source': PRICING_URL,
                    'checked_date': PRICING_CHECKED, 'mode': 'standard uncached input'},
        'token_statistics': {'total': sum(counts), 'average': statistics.mean(counts),
                             'median': statistics.median(counts), 'min': min(counts), 'max': max(counts)},
        'total_baseline_input_cost_usd': format(total_cost, '.12f'),
        'average_baseline_input_cost_usd': format(total_cost / len(records), '.12f'),
        'scope': 'Cleaned user_input text only; excludes output, system instructions, chat framing, tools, caching, and compression compute.',
        'row_alignment': 'Zero-based test_row in source order; duplicates retained; join with input checksum + test_row and verify prompt_sha256.',
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'token_baseline.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    (output_dir / 'token_baseline_summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-input', type=Path, default=ROOT / 'data/test_clean.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/token_baseline')
    args = parser.parse_args()
    print(json.dumps(measure(args.test_input, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
