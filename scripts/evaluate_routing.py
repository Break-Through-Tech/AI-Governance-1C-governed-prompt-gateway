"""Run issue #21 routing over the cleaned test set and compare costs.

Each test prompt goes through the same steps as the gateway (PR #32):
v2 classify -> governance (logged) -> blocked prompts stop; passed prompts are
compressed (#19) and routed on the compressed token count. The routed input
cost is compared with always sending the same passed prompts to the largest tier.

Row alignment follows #8: the test file checksum must match the token baseline,
and each row's prompt_sha256 is verified. All rows, including duplicates, are kept.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack

from scripts.compress_prompts import compress_prompt
from scripts.governance import POLICY_VERSION, apply_governance
from scripts.route_model import (LARGEST, PRICING_CHECKED, PRICING_MODE, PRICING_SOURCE, TIERS,
                                 ModelRouter, usd)

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['prompt_id', 'test_row', 'prompt_sha256', 'source_label', 'predicted_label',
          'risk_score', 'action', 'original_tokens', 'routed_tokens', 'compression_status',
          'tier', 'model', 'reason', 'estimated_cost_usd', 'largest_tier_cost_usd']


def load_test_set(test_input: Path, baseline_dir: Path) -> tuple[list[dict], str]:
    """Return test rows joined to the #8 baseline, after checksum and hash checks."""
    raw = test_input.read_bytes()
    input_sha256 = hashlib.sha256(raw).hexdigest()
    summary = json.loads((baseline_dir / 'token_baseline_summary.json').read_text(encoding='utf-8'))
    if summary['input_sha256'] != input_sha256:
        raise ValueError('Test file checksum does not match the #8 token baseline')
    source = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline='')))
    with (baseline_dir / 'token_baseline.csv').open(encoding='utf-8', newline='') as f:
        baseline = list(csv.DictReader(f))
    if len(source) != len(baseline):
        raise ValueError('Test set and token baseline have different row counts')
    rows = []
    for index, (row, base) in enumerate(zip(source, baseline)):
        text = row['user_input']
        if (int(base['test_row']) != index
                or base['prompt_sha256'] != hashlib.sha256(text.encode('utf-8')).hexdigest()):
            raise ValueError(f'Token baseline row {index} does not match the test set')
        rows.append({'prompt_id': base['prompt_id'], 'test_row': index, 'text': text,
                     'prompt_sha256': base['prompt_sha256'], 'source_label': row['label'],
                     'original_tokens': int(base['original_tokens'])})
    return rows, input_sha256


def classify_batch(texts: list[str]) -> list[dict]:
    """Vectorized equivalent of scripts.classify_prompt.classify (same features and
    risk_score definition). Prompts are embedded one at a time (batch_size=1):
    batching pads inputs and shifts risk scores by ~1e-6, and routing compares
    those scores to thresholds. Spot-checked against classify() before use."""
    from scripts import classify_prompt as cp  # Heavy import: sentence-transformers.
    vectorizer, embedder, model = cp._load()
    embeddings = np.asarray(embedder.encode(texts, batch_size=1, normalize_embeddings=True,
                                            show_progress_bar=False), dtype='float32')
    features = hstack([vectorizer.transform(texts), csr_matrix(embeddings)], format='csr')
    proba = model.predict_proba(features)
    classes = list(model.classes_)
    risk_columns = [classes.index(label) for label in cp.RISK_LABELS]
    results = [{'label': str(classes[p.argmax()]), 'risk_score': float(p[risk_columns].sum()),
                'score_type': cp.SCORE_TYPE} for p in proba]
    for text, batched in zip(texts[:5], results[:5]):
        single = cp.classify(text)
        if single['label'] != batched['label'] or single['risk_score'] != batched['risk_score']:
            raise RuntimeError('Batched classification does not match classify()')
    return results


def evaluate(rows: list[dict], classifications: list[dict], log_path: Path,
             router: ModelRouter | None = None) -> list[dict]:
    """Govern, compress, and route each row. Blocked rows get no tier or cost."""
    router = router or ModelRouter()
    largest = router.tiers[LARGEST]
    records = []
    for row, classification in zip(rows, classifications, strict=True):
        decision = apply_governance(classification, prompt_id=row['prompt_id'], log_path=log_path)
        record = {k: row[k] for k in ('prompt_id', 'test_row', 'prompt_sha256',
                                      'source_label', 'original_tokens')}
        record.update(predicted_label=decision['label'], risk_score=decision['risk_score'],
                      action=decision['action'], routed_tokens='', compression_status='blocked',
                      tier='', model='', reason='blocked', estimated_cost_usd='',
                      largest_tier_cost_usd='')
        if decision['action'] == 'pass':
            compressed = compress_prompt(row['text'], 'pass')
            route = router.recommend(compressed.compressed_prompt, decision['risk_score'],
                                     compressed.compressed_tokens)
            record.update(routed_tokens=compressed.compressed_tokens,
                          compression_status=compressed.status, tier=route['tier'],
                          model=route['model'], reason=route['reason'],
                          estimated_cost_usd=route['estimated_cost_usd_exact'],
                          largest_tier_cost_usd=usd(
                              largest.input_cost(compressed.compressed_tokens)))
        records.append(record)
    return records


def _pct(part, whole) -> float:
    return round(float(100 * part / whole), 2) if whole else 0.0


def summarize(records: list[dict], input_sha256: str, router: ModelRouter | None = None) -> dict:
    router = router or ModelRouter()
    passed = [r for r in records if r['action'] == 'pass']
    tiers = Counter(r['tier'] for r in passed)
    routed = sum(Decimal(r['estimated_cost_usd']) for r in passed)
    largest = sum(Decimal(r['largest_tier_cost_usd']) for r in passed)
    passed_tokens = sum(r['routed_tokens'] for r in passed)
    by_source = defaultdict(Counter)
    for r in records:
        by_source[r['source_label']][r['tier'] or 'blocked'] += 1
    tier_tokens = Counter()
    tier_cost = defaultdict(Decimal)
    for r in passed:
        tier_tokens[r['tier']] += r['routed_tokens']
        tier_cost[r['tier']] += Decimal(r['estimated_cost_usd'])
    return {
        'input_filename': 'test_clean.csv', 'input_sha256': input_sha256,
        'policy_version': POLICY_VERSION, 'classifier': 'v2 (scripts/classify_prompt.py)',
        'prompts': len(records),
        'governance': {'passed': len(passed), 'blocked': len(records) - len(passed)},
        'thresholds': {'short_tokens': router.short_tokens, 'long_tokens': router.long_tokens,
                       'elevated_risk': router.elevated_risk, 'high_risk': router.high_risk},
        'tiers': {name: {'model': t.model, 'input_usd_per_million': str(t.input_usd_per_million),
                         'output_usd_per_million': str(t.output_usd_per_million)}
                  for name, t in router.tiers.items()},
        'pricing': {'source': PRICING_SOURCE, 'checked_date': PRICING_CHECKED, 'mode': PRICING_MODE},
        'routing_distribution': {
            name: {'prompts': tiers[name], 'percent_of_passed': _pct(tiers[name], len(passed)),
                   'routed_tokens': tier_tokens[name],
                   'estimated_cost_usd': usd(tier_cost[name])}
            for name in router.tiers},
        'routing_reasons': dict(sorted(Counter(r['reason'] for r in passed).items())),
        'source_label_by_outcome': {label: dict(sorted(c.items()))
                                    for label, c in sorted(by_source.items())},
        'cost_comparison': {
            'scope': 'passed prompts only, input tokens after compression',
            'routed_tokens': passed_tokens,
            'routed_cost_usd': usd(routed),
            'always_largest_cost_usd': usd(largest),
            'savings_usd': usd(largest - routed),
            'savings_percent': _pct(largest - routed, largest),
        },
        'row_alignment': 'Zero-based test_row; join with input_sha256 + test_row and verify prompt_sha256 (as #8).',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-input', type=Path, default=ROOT / 'data/test_clean.csv')
    parser.add_argument('--baseline-dir', type=Path, default=ROOT / 'data/token_baseline')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/routing')
    parser.add_argument('--log-path', type=Path,
                        default=ROOT / 'data/processed/routing/routing_decisions.jsonl')
    args = parser.parse_args()

    rows, input_sha256 = load_test_set(args.test_input, args.baseline_dir)
    classifications = classify_batch([r['text'] for r in rows])
    args.log_path.unlink(missing_ok=True)  # The log is this run's snapshot, not the live audit log.
    records = evaluate(rows, classifications, args.log_path)
    summary = summarize(records, input_sha256)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'routing_results.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    (args.output_dir / 'routing_summary.json').write_text(json.dumps(summary, indent=2) + '\n',
                                                          encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
