"""Conservative, governance-gated prompt compression and issue #8 evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import re
import statistics

try:
    from .measure_token_baseline import ENCODING, MODEL, ROOT, count_tokens
except ImportError:  # Direct script invocation.
    from measure_token_baseline import ENCODING, MODEL, ROOT, count_tokens

# Only remove a request wrapper before an explicit task verb. Keep the entire
# task body byte-for-byte; never rewrite numbers, negations, entities or code.
WRAPPER = re.compile(
    r'\A(?:please\s+|(?:could|can|would) you (?:please )?)'
    r'(?=(?:explain|describe|list|summarize|compare|write|create|give|show|'
    r'tell|help|translate|calculate|find|generate|provide|suggest|recommend)\b)',
    re.IGNORECASE,
)
# Formatting/literal tasks are deliberately left alone, including multiline
# text, quoted strings, code, and requests about exact wording.
PROTECTED = re.compile(
    r'[\r\n`"“”\[\]{}]|\b(?:verbatim|exact|exactly|literal|quote|quoted|'
    r'quotation|character|characters|whitespace|spacing|indentation|regex|'
    r'password|code|json|yaml|csv|poem|poetry|rhyme|acrostic|word|words|'
    r'prompt|sentence|prefix|suffix)\b',
    re.IGNORECASE,
)
THRESHOLD = 1.0


def digest(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def content_units(text: str) -> list[str]:
    """Ordered words and punctuation after at most one permitted wrapper."""
    return re.findall(r'\w+|[^\w\s]', WRAPPER.sub('', text, count=1), re.UNICODE)


def preservation_score(original: str, candidate: str) -> float:
    """Strict task-body lexical equivalence (1 or 0), a semantic proxy."""
    return float(content_units(original) == content_units(candidate))


@dataclass(frozen=True)
class CompressionResult:
    action: str
    status: str
    compressed_prompt: str | None
    original_tokens: int | None = None
    candidate_tokens: int | None = None
    compressed_tokens: int | None = None
    reduction_percent: float | None = None
    preservation_score: float | None = None


def compress_prompt(prompt: str, action: str) -> CompressionResult:
    """Call only after governance. A block yields no downstream prompt."""
    if action not in {'pass', 'block'}:
        raise ValueError('Governance action must be pass or block')
    if action == 'block':
        return CompressionResult(action, 'blocked', None)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Passed prompt must be nonempty text')
    before = count_tokens(prompt)
    candidate = prompt if PROTECTED.search(prompt) else WRAPPER.sub('', prompt, count=1)
    score = preservation_score(prompt, candidate)
    candidate_tokens = count_tokens(candidate)
    if score < THRESHOLD:
        status = 'rejected_preservation'
    elif candidate_tokens >= before:
        status = 'unchanged_protected' if PROTECTED.search(prompt) else 'unchanged_no_savings'
    else:
        status = 'compressed'
    output = candidate if status == 'compressed' else prompt
    after = count_tokens(output)
    return CompressionResult(action, status, output, before, candidate_tokens, after,
                             100 * (before - after) / before, score)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def load_baseline(test_input: Path, baseline_dir: Path) -> tuple[list[dict], list[dict], dict]:
    raw = test_input.read_bytes()
    metadata = json.loads((baseline_dir / 'token_baseline_summary.json').read_text())
    if metadata['input_sha256'] != hashlib.sha256(raw).hexdigest():
        raise ValueError('Test input checksum does not match issue #8 baseline')
    if (metadata['encoding'] != ENCODING or metadata['reference_model'] != MODEL
            or metadata['tiktoken_version'] != version('tiktoken')):
        raise ValueError('Tokenizer/model/version differs from issue #8 baseline')
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
    if reader.fieldnames != ['user_input', 'label']:
        raise ValueError('Expected user_input,label test columns')
    source = list(reader)
    baseline = read_csv(baseline_dir / 'token_baseline.csv')
    if not source or len(source) != len(baseline) or len(source) != metadata['prompt_count']:
        raise ValueError('Empty or mismatched test/baseline row counts')
    for index, (row, base) in enumerate(zip(source, baseline)):
        prompt = row.get('user_input')
        if (None in row or not isinstance(prompt, str) or not prompt.strip()
                or row.get('label') not in {'safe', 'toxic', 'jailbreak'}):
            raise ValueError(f'Invalid test row {index}')
        expected = {'prompt_id': f'test-{index:06d}', 'test_row': str(index),
                    'prompt_sha256': digest(prompt), 'label': row['label'],
                    'original_tokens': str(count_tokens(prompt)),
                    'encoding': ENCODING, 'reference_model': MODEL}
        if any(base.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Baseline alignment/count mismatch at row {index}')
    return source, baseline, metadata


def load_decisions(path: Path, baseline: list[dict], input_sha256: str) -> tuple[dict, str]:
    """Validate a complete decision snapshot; never infer a missing decision."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('input_sha256') != input_sha256:
        raise ValueError('Governance input checksum mismatch')
    source = payload.get('source')
    if not isinstance(source, str) or not source.strip():
        raise ValueError('Governance source is required')
    rows = payload.get('decisions')
    if not isinstance(rows, list) or len(rows) != len(baseline):
        raise ValueError('Require exactly one governance decision per baseline row')
    expected = {r['prompt_id']: r['prompt_sha256'] for r in baseline}
    decisions = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Invalid governance decision')
        prompt_id = row.get('prompt_id')
        if (not isinstance(prompt_id, str) or prompt_id not in expected
                or prompt_id in decisions
                or row.get('prompt_sha256') != expected[prompt_id]
                or row.get('action') not in {'pass', 'block'}):
            raise ValueError('Invalid, duplicate, unknown or misaligned governance decision')
        decisions[prompt_id] = row['action']
    return decisions, source


def evaluate(test_input: Path, baseline_dir: Path, output_dir: Path,
             governance_decisions: Path | None = None, *, demo_governance: bool = False) -> dict:
    if (governance_decisions is not None) == demo_governance:
        raise ValueError('Choose exactly one: governance decisions or explicit demo governance')
    source, baseline, metadata = load_baseline(test_input, baseline_dir)
    if demo_governance:
        decisions = {r['prompt_id']: 'pass' if r['label'] == 'safe' else 'block' for r in baseline}
        decision_source = 'SIMULATED: dataset safe label passes; other labels block; not issue #17'
    else:
        decisions, decision_source = load_decisions(governance_decisions, baseline, metadata['input_sha256'])
    records = []
    examples = []
    for row, base in zip(source, baseline):
        result = compress_prompt(row['user_input'], decisions[base['prompt_id']])
        record = {key: base[key] for key in ('prompt_id', 'test_row', 'prompt_sha256', 'label')}
        record.update(asdict(result))
        record.update(baseline_original_tokens=int(base['original_tokens']),
                      encoding=ENCODING, governance_source=decision_source,
                      simulated_governance=demo_governance)
        records.append(record)
        if result.status == 'compressed' and len(examples) < 5:
            examples.append({'prompt_id': base['prompt_id'], 'before_tokens': result.original_tokens,
                             'after_tokens': result.compressed_tokens})
    eligible = [r for r in records if r['action'] == 'pass']
    reductions = [r['reduction_percent'] for r in eligible]
    before = sum(r['original_tokens'] for r in eligible)
    after = sum(r['compressed_tokens'] for r in eligible)
    aggregate = 100 * (before - after) / before if before else None
    report = {
        'input_sha256': metadata['input_sha256'], 'encoding': ENCODING,
        'tiktoken_version': version('tiktoken'), 'governance_source': decision_source,
        'simulated_governance': demo_governance, 'total_prompts': len(records),
        'eligible_prompts': len(eligible), 'blocked_prompts': len(records) - len(eligible),
        'status_counts': dict(sorted(Counter(r['status'] for r in records).items())),
        'eligible_original_tokens': before, 'eligible_compressed_tokens': after,
        'tokens_saved': before - after,
        'mean_reduction_percent': statistics.mean(reductions) if reductions else None,
        'median_reduction_percent': statistics.median(reductions) if reductions else None,
        'aggregate_reduction_percent': aggregate,
        'target_percent': [30, 50],
        'target_assessment': ('no eligible prompts' if aggregate is None else
                              'below target' if aggregate < 30 else
                              'within target' if aggregate <= 50 else 'above target'),
        'preservation_metric': 'Exact ordered task-body lexical equivalence after one allowed request wrapper; binary semantic proxy',
        'preservation_threshold': THRESHOLD,
        'limitations': [
            'Lexical equivalence is not proof of semantic equivalence; request tone/force can change.',
            'Only leading polite request wrappers are removed; short or protected prompts often have no savings.',
            'No model responses or embedding similarity were evaluated; downstream quality needs independent evaluation.',
            'Blocked prompts are excluded from compression denominators and are never sent downstream.',
        ],
        'examples': examples,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'compression_results.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    (output_dir / 'compression_summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-input', type=Path, default=ROOT / 'data/test_clean.csv')
    parser.add_argument('--baseline-dir', type=Path, default=ROOT / 'data/token_baseline')
    parser.add_argument('--output-dir', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--governance-decisions', type=Path)
    group.add_argument('--demo-governance', action='store_true', help='Explicitly simulate decisions using dataset labels')
    args = parser.parse_args()
    print(json.dumps(evaluate(args.test_input, args.baseline_dir, args.output_dir,
                              args.governance_decisions, demo_governance=args.demo_governance), indent=2))


if __name__ == '__main__':
    main()
