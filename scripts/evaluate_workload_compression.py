"""Measure token reduction on the simulated customer-service workload built
by scripts/simulate_workload.py (issue #30).

Issue #19 measured compression on raw user messages only (~34 tokens
average) and reached 0.2% reduction. Here, each request also carries a
system prompt, retrieved FAQ/policy context, and chat history -- where
issue #19 predicted most of a real request's tokens actually live.

Compression per component:
- system prompt: swapped for its tightened version (both fixed text,
  authored once in simulate_workload.py -- not algorithmically derived).
- context: trimmed from `context_per_query` simulated-retrieved snippets
  down to `context_kept_after_compression` (keeps the first N, i.e. the
  top-N by simulated retrieval rank).
- chat history: trimmed from the full fixed template down to the last
  `history_turns_kept_after_compression` turns.
- user message: issue #19's compress_prompt(), gated by a REAL governance
  decision from scripts.classify_prompt (issue #18) + scripts.governance
  (issue #17) -- not a simulated decision. Issue #19's own doc flagged
  real-governance integration as pending until both existed; both now do.

Reports before/after tokens per request and per component, aggregate
reduction, estimated cost savings (issue #8's GPT-4o mini pricing), and an
explicit meaning-preservation check on the user message (the only
component whose text is rewritten rather than trimmed -- system prompt,
context, and history are trimmed/swapped, never reworded).

Results here are SIMULATED: the workload, retrieval, and history are
authored, not drawn from real production traffic. They are reported
honestly against the 30-50% target either way.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Let `python scripts/evaluate_workload_compression.py` find sibling modules.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classify_prompt import classify
from scripts.compress_prompts import compress_prompt
from scripts.governance import apply_governance
from scripts.measure_token_baseline import RATE, count_tokens
WORKLOAD_PATH = ROOT / 'data/processed/workload/simulated_workload.json'
OUTPUT_DIR = ROOT / 'data/processed/workload'
TARGET_PERCENT = [30, 50]


def _cost(tokens):
    return Decimal(tokens) * RATE / Decimal(1_000_000)


def render_history(turns):
    return '\n'.join(f'{role}: {text}' for role, text in turns)


def render_context(design, snippet_ids):
    return '\n'.join(design['context_pool'][snippet_id] for snippet_id in snippet_ids)


def evaluate_request(design, query, log_path):
    user_message = query['user_message']

    # Real classification + real governance decision (issue #18, #17) --
    # not the dataset's "safe" label, which was only used to select
    # candidates for this workload, never to decide pass/block here.
    classifier_output = classify(user_message)
    decision = apply_governance(classifier_output, prompt_id=f"workload-{query['test_row']}",
                                log_path=log_path)

    if decision['action'] == 'block':
        return {
            'test_row': query['test_row'], 'action': 'block',
            'classifier_label': classifier_output['label'],
            'risk_score': classifier_output['risk_score'],
        }

    compression = compress_prompt(user_message, action='pass')
    preserved = compression.preservation_score == 1.0

    before = {
        'system_prompt': count_tokens(design['system_prompt']),
        'context': count_tokens(render_context(design, query['retrieved_context_ids'])),
        'history': count_tokens(render_history(design['chat_history'])),
        'user_message': compression.original_tokens,
    }
    after = {
        'system_prompt': count_tokens(design['tightened_system_prompt']),
        'context': count_tokens(render_context(
            design, query['retrieved_context_ids'][:design['context_kept_after_compression']])),
        'history': count_tokens(render_history(
            design['chat_history'][-design['history_turns_kept_after_compression']:])),
        'user_message': compression.compressed_tokens,
    }

    return {
        'test_row': query['test_row'], 'action': 'pass',
        'classifier_label': classifier_output['label'],
        'risk_score': classifier_output['risk_score'],
        'before_by_source': before,
        'after_by_source': after,
        'before_total': sum(before.values()),
        'after_total': sum(after.values()),
        'user_message_status': compression.status,
        'user_message_preserved': preserved,
    }


def evaluate(workload_path=WORKLOAD_PATH, output_dir=OUTPUT_DIR):
    workload = json.loads(workload_path.read_text())
    design, queries = workload['design'], workload['queries']
    log_path = output_dir / 'workload_governance_decisions.jsonl'
    if log_path.exists():
        log_path.unlink()  # one fresh audit log per evaluation run

    records = [evaluate_request(design, query, log_path) for query in queries]
    blocked = [r for r in records if r['action'] == 'block']
    eligible = [r for r in records if r['action'] == 'pass']
    unpreserved = [r for r in eligible if not r['user_message_preserved']]

    by_source = {}
    for source in ('system_prompt', 'context', 'history', 'user_message'):
        before = sum(r['before_by_source'][source] for r in eligible)
        after = sum(r['after_by_source'][source] for r in eligible)
        by_source[source] = {
            'before_tokens': before, 'after_tokens': after,
            'reduction_percent': 100 * (before - after) / before if before else None,
        }

    before_total = sum(r['before_total'] for r in eligible)
    after_total = sum(r['after_total'] for r in eligible)
    aggregate = 100 * (before_total - after_total) / before_total if before_total else None
    per_request_reduction = [100 * (r['before_total'] - r['after_total']) / r['before_total']
                             for r in eligible if r['before_total']]

    report = {
        'simulated': True,
        'note': ('Workload, retrieval, and chat history are authored/simulated, not real '
                 'production traffic. Governance decisions are real (issue #17 + #18), not '
                 'the dataset label.'),
        'total_requests': len(records),
        'eligible_requests': len(eligible),
        'blocked_requests': len(blocked),
        'blocked_label_counts': dict(sorted(Counter(r['classifier_label'] for r in blocked).items())),
        'by_source': by_source,
        'eligible_before_total_tokens': before_total,
        'eligible_after_total_tokens': after_total,
        'tokens_saved': before_total - after_total,
        'aggregate_reduction_percent': aggregate,
        'mean_per_request_reduction_percent': statistics.mean(per_request_reduction) if per_request_reduction else None,
        'median_per_request_reduction_percent': statistics.median(per_request_reduction) if per_request_reduction else None,
        'target_percent': TARGET_PERCENT,
        'target_assessment': ('no eligible requests' if aggregate is None else
                              'below target' if aggregate < TARGET_PERCENT[0] else
                              'within target' if aggregate <= TARGET_PERCENT[1] else 'above target'),
        'estimated_cost_before_usd': format(_cost(before_total), '.6f'),
        'estimated_cost_after_usd': format(_cost(after_total), '.6f'),
        'estimated_cost_saved_usd': format(_cost(before_total) - _cost(after_total), '.6f'),
        'pricing_source': 'scripts/measure_token_baseline.py (issue #8): GPT-4o mini, $0.15/1M input tokens',
        'meaning_preservation': {
            'metric': ('User message: issue #19\'s exact task-body lexical equivalence check. '
                      'System prompt, context, and history are trimmed/swapped, never reworded, '
                      'so their meaning preservation is by construction, not separately scored.'),
            'eligible_requests_checked': len(eligible),
            'user_message_preservation_failures': len(unpreserved),
        },
        'blocked_by_real_governance_despite_safe_label': [
            r['test_row'] for r in blocked
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'workload_compression_results.json').write_text(
        json.dumps({'requests': records}, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'workload_compression_summary.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    report = evaluate()
    print(f"SIMULATED workload: {report['eligible_requests']}/{report['total_requests']} requests "
          f"eligible ({report['blocked_requests']} blocked by real governance despite a 'safe' dataset label)")
    print(f"\nBy source (eligible requests only):")
    for source, stats in report['by_source'].items():
        pct = stats['reduction_percent']
        print(f"  {source:>13}: {stats['before_tokens']:>7} -> {stats['after_tokens']:>7} tokens "
              f"({pct:.2f}% reduction)" if pct is not None else f"  {source:>13}: n/a")
    print(f"\nAggregate reduction: {report['aggregate_reduction_percent']:.2f}%  "
          f"({report['target_assessment']} of {report['target_percent']}% target)")
    print(f"Estimated cost: ${report['estimated_cost_before_usd']} -> "
          f"${report['estimated_cost_after_usd']} (saved ${report['estimated_cost_saved_usd']})")
    print(f"User-message meaning-preservation failures: "
          f"{report['meaning_preservation']['user_message_preservation_failures']}")
    print(f"\nSaved workload_compression_results.json and workload_compression_summary.json to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
