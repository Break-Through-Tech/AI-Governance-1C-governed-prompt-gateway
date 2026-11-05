# Governance layer

Issue #17 implements the label policy between classification and cache lookup /
optimization in [the gateway architecture](Governed_Prompt_Gateway_Architecture.pdf).
It uses the classifier's lowercase labels and the existing compressor's exact
`pass` / `block` actions. The dataset builder's `allow` / `review` annotations
remain dataset metadata, not live decisions.

## Policy and classifier contract

| Classifier label | Action | Proposed status code |
|---|---|---|
| `safe` | `pass` | 200 |
| `toxic` | `block` | 403 |
| `jailbreak` | `block` | 403 |

`apply_governance(output, *, prompt_id, log_path)` requires a mapping containing
`label` and `risk_score`, plus a nonempty request ID and a log path. Labels must
match exactly. Scores must be finite real numbers; negative values and values
above one are valid. Strings, booleans, NaN, infinity, unknown labels (including
`harmful`), and missing fields raise `ValueError`. No pass decision or decision
log entry is produced for invalid input. The caller must stop processing on an
exception and route it to its error/review handling, never default to pass.

**There is no score threshold.** Only the label determines the action. The
optional `score_type` is a nonempty description, defaulting to `classifier_score`;
it is recorded without assigning a probability interpretation.

The current [official baseline](baseline_eval.md#the-baseline-decision) is the
untuned `LinearSVC` built in `scripts/train_baseline_models.py`. Its `predict`
returns labels; `decision_function` returns a score for each of the three
classes. It has no `predict_proba` and does not already return a `risk_score`
field. `classify_with_svm(prompt, model=model, vectorizer=vectorizer)` adapts that
interface: it takes the predicted class's score using `model.classes_` for column
order, and marks it `svm_decision_function`. This signed score is neither a
probability of harm nor a score comparable across models. A larger safe-class
score does not mean more danger. No calibration or retraining is performed.

## Response and logging

**Proposed for Manuel's review:** the flat response fields and HTTP-style codes
below. No implemented gateway HTTP handler or agreed blocked-response schema
was found; agreement is still outstanding. `200` means this policy check passed,
not that an LLM request succeeded. `403` is the proposed blocked HTTP response.

Every returned decision has the same fields, for example:

```json
{
  "timestamp": "2026-11-04T18:00:00+00:00",
  "prompt_id": "request-123",
  "policy_version": "label-policy-v1",
  "label": "toxic",
  "risk_score": 1.7,
  "score_type": "svm_decision_function",
  "action": "block",
  "status_code": 403,
  "reason": "blocked_label",
  "message": "Request blocked by the prompt safety policy."
}
```

The timestamp above is illustrative; each call uses the current UTC time. A pass
uses `reason: allowed_label` and `message: Request passed the prompt safety policy.`
Responses never include a downstream prompt. On block, return the refusal and
stop before cache lookup, optimization, or LLM execution.

The exact decision is appended as one JSON object per line (JSONL), including
timestamp, request ID, label, score, score type, action, and policy version.
Raw prompt text is not logged. The caller supplies an ID it can correlate with
the request. Repeated IDs still produce separate entries for separate calls.
Logging errors propagate, so an unlogged pass is never returned. Invalid inputs
are errors rather than valid decisions; error telemetry belongs to the caller.

This is a local, single-writer prototype log. It does not provide log rotation,
multi-process locking, crash-proof durability, or a tamper-resistant audit store.
The live JSONL audit is not the batch decision snapshot consumed by compression's
offline evaluator; that export requires the dataset checksum and per-row text
hashes documented in [the compression guide](prompt-compression.md).

## Run the example and tests

The policy and fixed stub example use only the Python standard library:

```bash
python3 scripts/governance_demo.py
cat data/processed/governance/demo_decisions.jsonl
python3 -m unittest discover -s tests -p test_governance.py -v
```

The example produces `pass`, `block`, `block` for fixed safe, toxic, and jailbreak
fixtures. These are synthetic classifier outputs, not a keyword classifier or
measured safety predictions. Running it again appends three more records. Use
`--log-path /tmp/governance-demo.jsonl` to choose another output file.

With the repository dependencies installed, run the SVM/compression integration
tests and the existing compression regression tests:

```bash
python -m unittest tests.test_governance_integration tests.test_compress_prompts -v
```

For the existing fitted baseline, load the matching trusted model and vectorizer
as in the classifier guide, then call:

```python
from scripts.governance import apply_governance, classify_with_svm
from scripts.compress_prompts import compress_prompt

classification = classify_with_svm(prompt, model=model, vectorizer=vectorizer)
decision = apply_governance(classification, prompt_id=request_id,
                            log_path='data/processed/governance/decisions.jsonl')
if decision['action'] == 'pass':
    # In the full gateway, cache lookup comes first; compress on a cache miss.
    compressed = compress_prompt(prompt, action=decision['action'])
else:
    blocked_response = decision  # Return it without forwarding the prompt.
```

This policy does not fix classifier errors or implement the architecture's
uncertainty/review workflow. A prompt misclassified as safe still passes. Score
calibration, review rules, HTTP error mapping, and Manuel's response-interface
approval remain integration decisions, not claims made by this implementation.
