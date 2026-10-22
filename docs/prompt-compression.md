# Prompt compression / token optimization — issue #19

## Result and scope

A reproducible conservative compression baseline is implemented. It removes one
leading polite request wrapper before an explicit task verb, retains the task
body, and accepts the change only when it saves tokens and passes a preservation
check. It uses issue #8's `tiktoken==0.12.0`, `o200k_base`, and exact cleaned test
set. No paid API calls, model downloads, or additional dependencies are needed.

**The 30–50% target was not achieved.** The saved run uses **simulated governance**:
existing dataset `safe` labels pass; `toxic` and `jailbreak` labels block. These
are dataset annotations, not classifier predictions or real safety decisions.
Issue #17 has no implementation on the inspected `main` revision
`543bdca`. Real integration and a new run remain necessary when #17 is available.
Do not present these results as measured end-to-end gateway performance.

| Measure | Simulated-governance result |
|---|---:|
| Full test set | 4,883 |
| Eligible prompts | 4,549 |
| Blocked / excluded from compression | 334 |
| Prompts compressed | 187 |
| Unchanged: no supported saving | 3,142 |
| Unchanged: protected text | 1,220 |
| Eligible original tokens | 156,189 |
| Eligible compressed tokens | 155,876 |
| Tokens saved | 313 |
| Average per-prompt reduction | 0.5841% |
| Median per-prompt reduction | 0% |
| Aggregate token reduction | 0.2004% |

All eligible prompts, including unchanged prompts, contribute to the mean and
median. Aggregate reduction weights prompts by their original token count.
Blocked rows have blank compression measurements and no downstream prompt;
their issue #8 counts are retained separately for audit, never claimed as
compression savings. Duplicate observations remain separate rows.

## Approach and meaning preservation

Examples of removable prefixes: `Please `, `Can you `, `Could you please `.
They are removed only before a fixed task-verb list such as `explain`, `list`,
`write`, and `compare`. No sentence summarization, stop-word removal, global
whitespace normalization, or deletion of task constraints is performed.

Formatting-sensitive and literal tasks are skipped conservatively: multiline
input, double quotes, backticks, brackets/braces, or terms related to exact text,
code, structured formats, poetry, words, and prompt self-reference. These guards
are heuristics, not a complete detector of every formatting-sensitive task.
Unrecognized languages and request styles usually remain unchanged.

**Defined metric:** exact ordered task-body lexical equivalence, with threshold
**1.0**. Strip at most one permitted prefix from both strings, then compare the
ordered words and punctuation (case-sensitive). A match scores 1; any mismatch
scores 0. Changed numbers, negations, punctuation or word order fail this check.
A failed check or a candidate with no token saving falls back to the original.
The CSV retains the candidate token count and candidate preservation score even
when the original is returned.

This is a narrow **semantic proxy**, not an independent semantic evaluation or
proof of preserved meaning. The rule and metric share the allowed-prefix
assumption. Removing a question wrapper can change tone or request force; subtle
self-reference can escape the guards. No embeddings, downstream response-quality
comparison, or classifier-invariance claim is made. Broader rewriting should not
reuse this check alone: it needs an independently evaluated semantic metric and
constraint checks on development data before a held-out evaluation.

## Before/after examples from the held-out set

| ID | Before | After | Tokens |
|---|---|---|---:|
| test-000160 | can you write a script for the game snake in java? | write a script for the game snake in java? | 12 → 10 |
| test-000218 | Could you recommend movies? I don't like scary movies. The last scary movie I saw was The Ring (1996) | recommend movies? I don't like scary movies. The last scary movie I saw was The Ring (1996) | 24 → 22 |
| test-000292 | Please create a stat block of supergirl in the GURPS system | create a stat block of supergirl in the GURPS system | 14 → 13 |

The second example retains the negation and year. These illustrate the method;
they are not representative estimates of dataset-wide savings.

Failure/abstention examples covered by tests:

- `Can you swim?`: no supported task verb; unchanged.
- `Please write a poem about cats.`: poetry may depend on wording; unchanged.
- `Please list the first two words of this prompt.`: self-reference; unchanged.
- Quoted text, multiline content and literal special-token strings remain intact.
- A candidate deleting `not` or changing task order fails the lexical metric.
- Missing, invalid, duplicate, stale or misaligned governance decisions fail
  validation before output artifacts are written.

## Integration contract for #17 and #22

Runtime use:

```python
from scripts.compress_prompts import compress_prompt

# action comes from the governance layer, never from a dataset label in production.
result = compress_prompt(prompt, action=governance_result["action"])
if result.action == "pass":
    downstream_prompt = result.compressed_prompt
    # Forward downstream_prompt to the next gateway stage here.
# A blocked result has compressed_prompt=None. Never forward the original instead.
```

Only exact `pass` and `block` actions are accepted. Unknown/missing actions raise
an error. The blocked path returns before tokenization or candidate generation.
This module consumes a decision; it does not implement policy, authenticate a
caller, or replace #17's logging. The gateway must call it after governance.

For an offline evaluation, #17 should export this JSON shape:

```json
{
  "source": "governance-policy-version-and-run-id",
  "input_sha256": "<SHA-256 of the exact test_clean.csv bytes>",
  "decisions": [
    {
      "prompt_id": "test-000000",
      "prompt_sha256": "<SHA-256 of the UTF-8 prompt text>",
      "action": "pass"
    }
  ]
}
```

Include exactly one decision for every test observation, including blocked rows.
The example shows the shape of one entry, not a complete runnable decision file.
Order may differ; IDs and text hashes must match. The evaluator verifies the
input checksum, tokenizer version/encoding, every baseline ID, row, label, text
hash and original token count before producing artifacts. Dataset labels remain
provenance only when an explicit decision file is used.

## Reproduce

From the repository root after installing `requirements.txt`:

```bash
# Explicitly labeled demo; does not require or claim real governance.
python scripts/compress_prompts.py --demo-governance --output-dir data/compression_demo

# Real decisions, when #17 can supply them.
python scripts/compress_prompts.py \
  --governance-decisions path/to/governance_decisions.json \
  --output-dir data/compression_governed

python -m unittest discover -s tests -p test_compress_prompts.py -v
```

The CLI requires exactly one decision source and an explicit output directory.
An existing output directory's two result files are replaced. Alternative test
and baseline paths are available via `--test-input` and `--baseline-dir`.

- `data/compression_demo/compression_results.csv`: one row per test observation,
  containing alignment keys, original dataset label, action, status, candidate
  and final counts, reduction, preservation score, eligible compressed text,
  baseline counts, encoding, and explicit simulation/source fields.
- `data/compression_demo/compression_summary.json`: summary, denominator sizes,
  target comparison, metric definition, limitations, and example row IDs.
- Original text can be retrieved by `test_row` from the unchanged test CSV.

Per prompt: `100 * (original_tokens - compressed_tokens) / original_tokens`.
Aggregate: `100 * (sum(original_tokens) - sum(compressed_tokens)) / sum(original_tokens)`
over **the same eligible subset**. An all-blocked run reports null reduction
statistics, not a successful 100% reduction.

## Why the target was missed and what to do next

The approach only removes a few request-wrapper tokens. Most inputs have no
supported wrapper; many are already short, and protected tasks intentionally
remain unchanged. There were no preservation rejections in this run because the
accepted transformation preserves the defined task body by construction. This
is not evidence that a stronger compressor would preserve semantics.

The results establish that polite-wrapper removal alone is insufficient. A next
experiment should compare a stronger summarization or extractive approach on
training/validation data, using an independent semantic metric plus explicit
checks for negation, numbers, names, and required output format. Select its
thresholds there, then evaluate once on held-out data. Include compression
latency and compute/API cost: fewer input tokens need not mean lower total cost.
Do not tune rules to the displayed test examples or drop unchanged prompts to
inflate the reported average.

Issue #19 has an implemented compressor, aligned CSV, documented metric,
examples and an honest target comparison. Its real-governance evaluation remains
pending #17; this work should not automatically close that dependency or claim
that the integrated milestone is finished.
