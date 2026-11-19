# Model recommendation and routing — issue #21

## Findings

On the cleaned test set (4,883 prompts), the v2 classifier and label policy pass
4,560 prompts and block 323. Routing the passed prompts by risk score and
post-compression token count gives:

| Tier | Model | Prompts | Share of passed | Routed input tokens | Estimated input cost (USD) |
|---|---|---:|---:|---:|---:|
| small | `gpt-6-luna` | 3,385 | 74.23% | 40,430 | $0.004043 |
| medium | `gpt-6.1-terra` | 950 | 20.83% | 61,892 | $0.061892 |
| large | `gpt-6.2-astra` | 225 | 4.93% | 53,306 | $0.426448 |
| **Total** | | **4,560** | | **155,628** | **$0.492383** |

Compared with sending the same 4,560 passed prompts to the largest tier
(`gpt-6.2-astra`, $1.245024), routing lowers the estimated **input** cost by
**$0.752641 (60.45%)**.

The large tier receives 4.93% of passed prompts but accounts for 86.6% of routed
input cost, because it is 80× the small tier's input price and receives the
longest prompts.

Why prompts were routed where they were:

| Reason | Tier | Prompts |
|---|---|---:|
| `short_low_risk` | small | 3,385 |
| `medium_length` | medium | 755 |
| `elevated_risk` | medium | 195 |
| `long_prompt` | large | 158 |
| `high_risk` | large | 67 |

By source label (dataset labels, not predictions):

| Source label | Blocked | small | medium | large |
|---|---:|---:|---:|---:|
| safe (4,549) | 88 | 3,350 | 914 | 197 |
| toxic (257) | 167 | 35 | 31 | 24 |
| jailbreak (77) | 68 | 0 | 5 | 4 |

Risk escalation moves some classifier misses to a larger model, but 35 toxic
prompts that passed governance still scored low enough to go to the small tier.
Routing is a cost decision, not a safety control; it does not compensate for
classifier mistakes.

## Interface

`scripts/route_model.py` implements the router interface proposed in
[PR #32](https://github.com/Break-Through-Tech/AI-Governance-1C-governed-prompt-gateway/pull/32)
(issue #22):

```python
from scripts.route_model import ModelRouter, recommend

recommend('What is your refund policy?', risk_score=0.0, token_count=7)
# {'tier': 'small', 'model': 'gpt-6-luna', 'estimated_cost_usd': 7e-07,
#  'estimated_cost_usd_exact': '0.000000700000', 'cost_scope': 'input_tokens_only',
#  'reason': 'short_low_risk', 'pricing_checked': '2026-11-18'}
```

- `risk_score`: v2's P(toxic) + P(jailbreak), a probability in [0, 1].
- `token_count`: the #8 tiktoken count of the text sent downstream (after
  compression), or `None` if counting failed.
- `ModelRouter()` is the object the gateway plugs in as `components.router`;
  the module-level `recommend()` uses the default thresholds.
- Invalid input (non-text prompt, risk outside [0, 1] or non-finite, negative or
  non-integer token count, booleans) raises `ValueError`. The gateway treats
  router errors as fail-closed (503).
- Only passed prompts are routed. Blocked prompts never reach this stage.
- The prompt text is accepted for the agreed interface but not inspected.

PR #32 currently keeps only `tier` and `estimated_cost_usd` from the router and
labels routing `is_stub: true`. Integrating this router there means replacing
`StubRouter()` with `ModelRouter()` and deriving `is_stub` from the router type.
Agreement with Manuel on the final interface is still pending.

## Routing rules

Rules are checked in order; the first match wins.

| # | Condition | Tier | Reason |
|---|---|---|---|
| 1 | `token_count` is `None` | large | `token_count_unknown` (cost `None`) |
| 2 | `risk_score ≥ 0.10` | large | `high_risk` |
| 3 | `token_count > 200` | large | `long_prompt` |
| 4 | `risk_score ≥ 0.001` | medium | `elevated_risk` |
| 5 | `token_count > 30` | medium | `medium_length` |
| 6 | otherwise | small | `short_low_risk` |

Design choices:

- **Unknown length goes large.** Without a token count the prompt's complexity
  is unknown, so the router chooses the most capable tier rather than guess low.
  The cost is reported as `None`, not zero.
- **Borderline prompts go to stronger models.** A passed prompt with an elevated
  risk score is one the classifier was less sure about. Sending it to a more
  capable model is a design assumption that stronger models handle ambiguous
  requests more reliably; it was **not** measured here.
- **Length is a proxy for complexity.** Token count is the only complexity
  signal available; it does not detect short prompts that need deep reasoning.

### Where the thresholds come from

All thresholds were derived from training data only; the test set was used once,
for the results above. `scripts/derive_routing_thresholds.py` reproduces them and
writes [`data/routing/routing_thresholds.json`](../data/routing/routing_thresholds.json).

- **Token thresholds (30, 200):** P75 (29) and P95 (195.15) of `o200k_base`
  token counts on the 4,278-row training split, rounded.
- **Risk thresholds (0.001, 0.10):** P95 (0.0013) and P99 (0.1077) of
  **out-of-fold** v2 risk scores among the 4,687 train + validation prompts the
  fold models label safe, rounded. Five folds, grouped by identical prompt text,
  seed 42, using v2's features, class weights, and selected C (1000). Out-of-fold
  scores are required: v2 is trained on these rows, so in-sample scores are
  near zero for 95% of them and say nothing about new prompts.

v2 scores are concentrated near zero (out-of-fold median ≈ 1e-9), which is why
the elevated-risk threshold is so low. **The risk thresholds depend on the
trained model.** After any retrain, rerun the derivation script and update
`ELEVATED_RISK` and `HIGH_RISK`. The local retrain used here selected `C=1000`
(test macro F1 0.792), not the `C=100` (0.780) recorded in
[classifier-v2.md](classifier-v2.md), so v2 training is not exactly
reproducible across environments.

## Pricing source and reference models

Prices are OpenAI standard-tier, uncached, short-context text rates from the
[OpenAI pricing page](https://developers.openai.com/api/docs/pricing),
**checked November 18, 2026**. Tiers follow the
[models page](https://developers.openai.com/api/docs/models), which lists these
as current models; GPT-4o, GPT-4.1, and GPT-4.1 mini are marked deprecated there,
so they were not used.

| Tier | Model | Input (USD / 1M tokens) | Output (USD / 1M tokens) |
|---|---|---:|---:|
| small | `gpt-6-luna` | $0.10 | $0.50 |
| medium | `gpt-6.1-terra` | $1.00 | $5.00 |
| large | `gpt-6.2-astra` | $8.00 | $40.00 |

These are reference models for cost estimation, not a deployment decision. The
#8 baseline's reference model (`gpt-4o-mini`, $0.15 / 1M input) is unchanged.

## Cost scope and caveats

- **Input tokens only**, matching #8: `tokens × input price / 1,000,000`, in
  `Decimal`, reported to 12 decimal places. Output, system instructions, chat
  framing, cached-input discounts, and gateway compute are excluded.
- **Output costs are not estimated.** Output length is unknown before
  generation. For illustration only: every tier's output price is 5× its input
  price, so if each passed prompt produced the same number of output tokens
  regardless of tier, routing would lower output cost by 91.53% versus always
  using the largest tier. Real output lengths vary by prompt and model.
- **Token counts are approximate for these models.** tiktoken 0.12.0 does not
  map the GPT-6 family, so counts use the #8 `o200k_base` encoding. The true
  tokenizer may count differently.
- **Quality is not measured.** No model was called. The savings assume the
  smaller tiers answer routed prompts acceptably; that was not evaluated.
- **Compression barely changes counts here.** Routed tokens use #19's compressor
  exactly as the gateway does: 186 passed prompts were compressed, and passed
  prompts total 155,940 tokens before and 155,628 after (0.20%). These use v2
  governance decisions, so they are not directly comparable to #19's figures,
  which used simulated governance.

## Reproduce

From the repository root, with trained v2 artifacts (see [classifier-v2.md](classifier-v2.md)):

```bash
python -m scripts.derive_routing_thresholds   # training data only
python -m scripts.evaluate_routing            # test set, writes data/routing/
python -m unittest tests.test_route_model -v
```

| File | Purpose |
|---|---|
| `scripts/route_model.py` | Routing rules, tiers, pricing, `recommend()` |
| `scripts/derive_routing_thresholds.py` | Threshold derivation from training data |
| `scripts/evaluate_routing.py` | Test-set run: classify, govern, compress, route, compare |
| `data/routing/routing_thresholds.json` | Percentiles behind the thresholds |
| `data/routing/routing_results.csv` | One row per test observation (no prompt text) |
| `data/routing/routing_summary.json` | Distribution, reasons, label breakdown, cost comparison |
| `tests/test_route_model.py` | Rules, boundaries, validation, cost arithmetic, alignment, saved-result consistency |

Rows align with #8: the test file checksum must match the token baseline, each
row's `prompt_sha256` is verified, and duplicates are kept. The evaluation's
governance decisions go to `data/processed/routing/routing_decisions.jsonl`
(gitignored), not the live gateway audit log.
