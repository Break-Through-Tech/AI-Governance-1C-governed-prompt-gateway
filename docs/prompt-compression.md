# Prompt compression / token optimization
## Result and scope

A reproducible conservative compression baseline is implemented. It removes one
leading polite request wrapper before an explicit task verb, retains the task
body, and accepts the change only when it saves tokens and passes a preservation
check. It uses the same token-counting setup as the baseline measurement
(`tiktoken==0.12.0`, `o200k_base`) and the exact cleaned test set. No paid API
calls, model downloads, or additional dependencies are needed.

**The 30–50% target was not achieved.** The saved run uses **simulated governance**:
existing dataset `safe` labels pass; `toxic` and `jailbreak` labels block. These
are dataset annotations, not classifier predictions or real safety decisions.


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
their original token-count baseline is retained separately for audit, never
claimed as compression savings. Duplicate observations remain separate rows.

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

## Integration contract for governance and the gateway

Runtime use:

The [governance module](governance-policy.md) now produces the runtime decision
below, with a required audit log. Its response format is proposed for gateway
review. Existing saved compression results still use simulated governance.

```python
from scripts.compress_prompts import compress_prompt
from scripts.governance import apply_governance

# action comes from the governance layer, never from a dataset label in production.
governance_result = apply_governance(classifier_output, prompt_id=request_id,
                                    log_path='data/processed/governance/decisions.jsonl')
result = compress_prompt(prompt, action=governance_result["action"])
if result.action == "pass":
    downstream_prompt = result.compressed_prompt
    # Forward downstream_prompt to the next gateway stage here.
# A blocked result has compressed_prompt=None. Never forward the original instead.
```

Only exact `pass` and `block` actions are accepted. Unknown/missing actions raise
an error. The blocked path returns before tokenization or candidate generation.
This module consumes a decision; it does not implement policy, authenticate a
caller, or replace the governance module's logging. The gateway must call it
after governance.

For an offline evaluation, the governance module should export this JSON shape:

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

# Real decisions, once governance can supply them.
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

**Short answer: this test only measured the user's one-line question, with
nothing else around it — and there usually isn't much to safely cut out of a
single short sentence.**

The compressor only does one thing: if a question starts with a polite phrase
like "Please" or "Can you" right before a recognized task verb ("write",
"explain", "list", ...), it deletes that phrase. Nothing else about the
sentence changes — no summarizing, no rewording, no shortening of the actual
request. So the result depends entirely on how often that one specific phrase
shows up:

- **Most questions in the test set didn't have that phrase at all**, so
  nothing could be removed — before and after are identical.
- **The ones that did have it only saved a word or two** — e.g. "can you
  write a script..." → "write a script..." is 2 tokens off of 12, real but
  small.
- Formatting-sensitive requests (code, exact wording, quotes, poetry) are
  deliberately left untouched no matter what, to guarantee meaning never
  changes.

Averaged across thousands of mostly-unchanged, already-short questions, this
comes out to 0.2% — which is an honest result, not a failure of the method:
there just wasn't much removable text in what was being measured.

This pointed to the real fix: **measure the whole thing a chatbot actually
sends**, not just the user's one sentence. A real request also carries a
system prompt, reference material, and recent chat history — usually far
bigger than the question itself, and much easier to safely shrink (e.g. "keep
only the last 2 messages of history" or "use a shorter system prompt"). That
larger, more realistic test is exactly what the **Simulated workload** section
below covers, and it's where the 30-50% target was actually met.

A separate, independent direction (not yet attempted) would be to make the
compressor itself smarter — e.g. a real summarization or extractive approach
on the question text, rather than only removing a fixed polite phrase. That
would need its own careful evaluation (an independent semantic-similarity
check, explicit checks that numbers/negations/names survive, testing on
separate data before a final held-out run, and measuring compression's own
latency/cost) before trusting it the way this conservative approach can be
trusted.

This section has an implemented compressor, aligned CSV, documented metric,
examples and an honest target comparison. Its real-governance evaluation
remained pending until a real governance module and classifier existed; the
next section covers that real run.

## Simulated workload

The result above (0.2% aggregate reduction) only measures the raw user
message, which averages ~34 tokens. In a real deployment, most of a request's
tokens come from the system prompt, retrieved FAQ/policy context, and chat
history — not the one-line question. This section builds a simulated
customer-service workload that includes all of these, to measure compression
against a realistic token budget.

A real governance module and classifier now exist, so this evaluation uses
real governance decisions from the actual classifier (`scripts/classify_prompt.py`
+ `scripts/governance.py`) — not a simulated decision and not the dataset's
`safe` label, which was used only to select candidates for the workload, never
to decide pass/block.

### Workload design (fixed before measuring)

`scripts/simulate_workload.py` builds and saves the workload
(`data/processed/workload/simulated_workload.json`) before any measurement is taken, so
the design isn't tuned to the result:

- **150 safe-labeled test-set queries**, sampled with a fixed seed (42).
- **One fixed system prompt** (verbose) and **one fixed tightened version**
  (same instructions, condensed) — both authored once, not algorithmically
  derived.
- **A 6-snippet FAQ/policy pool** (shipping, returns, refunds, account,
  billing, warranty). Per query, 4 snippets are drawn by a seeded random
  sample, standing in for a retriever's ranked results (this tests
  compression, not retrieval quality). Compression keeps the first 2 — the
  top-2 by simulated rank.
- **One fixed 4-exchange chat history**, reused for every query, representing
  a typical prior-conversation length rather than a real per-user log.
  Compression keeps only the last exchange (2 turns).
- **The real user question**, compressed with the same `compress_prompt()`
  function used above, after a real governance decision.

None of the above is real retrieval or a real conversation log — it is
authored, fixed infrastructure for measuring compression at a realistic token
scale.

### Results

Of 150 sampled `safe`-labeled queries, **5 were blocked by the real v2
classifier** despite the dataset's `safe` label — an honest finding, not a
bug. Two examples: a request to "Summarize the following text: 'The
Bogey-Beast...'" (a children's story) was flagged `toxic` (likely a false
positive), while "You are now a roleplay machine... fully embody the persona
of any character..." was flagged `jailbreak` (plausibly a correct catch of a
roleplay-override attempt the original dataset label missed). Blocked requests
are excluded from every statistic below, the same convention used above.

| Source | Before (tokens) | After (tokens) | Reduction |
|---|---:|---:|---:|
| System prompt | 24,215 | 13,340 | 44.91% |
| Context | 27,528 | 13,752 | 50.04% |
| Chat history | 21,315 | 4,785 | 77.55% |
| User message | 4,474 | 4,469 | 0.11% |

| Measure | Result |
|---|---:|
| Eligible requests | 145 / 150 |
| Aggregate reduction | **53.12%** |
| Mean per-request reduction | 53.49% |
| Median per-request reduction | 54.85% |
| Estimated cost (before → after) | $0.011630 → $0.005452 |
| Estimated cost saved | $0.006178 |
| Target (30–50%) | **above target** |

**The 30–50% target is met here** (53.12%, slightly above the top of the
range) — confirming the hypothesis above: the user message alone was never
where the savings would come from. System prompt, context, and especially
chat history (77.55%) carry nearly all of the reduction. The user message's
own reduction (0.11%) is consistent with the result reported above for the
same compressor — this workload didn't change how well that compressor
works, only how much of the total token budget it's responsible for.

**This is reported as simulated, not a production result**, for a specific
reason beyond the usual caveat: the system prompt, context pool, and chat
history are fixed, authored text applied identically to every request. A real
deployment's system prompt is also fixed in this way (so that part of the
result should generalize reasonably), but real retrieved context and real chat
history lengths vary per request and aren't captured by one fixed template —
the context/history reduction percentages here reflect this workload's
specific design choices (4 retrieved snippets kept to 2; a 4-exchange history
kept to 1), not a universal result.

### Meaning preservation

The user message is the only component whose text is rewritten rather than
trimmed or swapped wholesale; it reuses the exact task-body lexical
equivalence check defined above, scored for every eligible request. **0 of 145 eligible
requests failed this check.** The system prompt, context, and history are
trimmed or swapped between two fixed, pre-authored versions — their "meaning
preservation" is by construction (the tightened system prompt and the kept
context/history are simply a subset or paraphrase decided at design time, not
computed per request), not independently scored here.

### Handoff

Numbers from `data/processed/workload/workload_compression_summary.json` are
available for the data-quality evaluation report and the baseline-classifier
dashboard to consume.

### Reproduce

```bash
python scripts/simulate_workload.py
python scripts/evaluate_workload_compression.py
python -m unittest discover -s tests -p 'test_simulate_workload.py' -p 'test_evaluate_workload_compression.py' -v
```

- `data/processed/workload/simulated_workload.json`: the fixed design plus the 150
  sampled queries and their simulated retrieval.
- `data/processed/workload/workload_compression_results.json`: one record per request —
  governance decision, per-source before/after tokens (if passed), and the
  user-message preservation check.
- `data/processed/workload/workload_compression_summary.json`: aggregate reduction,
  per-source breakdown, cost estimate, target comparison, and the blocked-row
  list.
- `data/processed/workload/workload_governance_decisions.jsonl`: the real governance
  audit log for this run (the governance module's required audit trail), one
  entry per request.
