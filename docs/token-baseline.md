# Token usage and baseline input cost — issue #8

## Findings

We counted every `user_input` in the existing cleaned, held-out test set, without
changing text, labels, row order, or duplicate observations. This is the same
4,883-row test set used by the split/vectorization and baseline classifier
pipeline. This establishes the **before-compression** baseline; it does not
measure achieved savings.

| Measure | Result |
|---|---:|
| Test prompts | 4,883 |
| Total input tokens | 181,855 |
| Average tokens per prompt | 37.2425 |
| Median tokens per prompt | 13 |
| Minimum / maximum tokens | 4 / 924 |
| Total baseline input cost (USD) | $0.02727825 |
| Average baseline input cost per prompt (USD) | $0.000005586371 |

GPT-4o mini (`gpt-4o-mini-2024-07-18`) is the reference cost model, not a claim
about the model the final gateway will deploy. Its published standard uncached
input price is **$0.15 per million tokens**, checked October 7, 2026 against
[OpenAI's model page](https://developers.openai.com/api/docs/models/gpt-4o-mini).
The calculation is `input_tokens × 0.15 / 1,000,000`. Decimal arithmetic avoids
rounding each small cost to zero. Average cost is rounded to 12 decimal places.

| Source label | Prompts | Total tokens | Average tokens | Median tokens |
|---|---:|---:|---:|---:|
| safe | 4,549 | 156,189 | 34.3348 | 13 |
| toxic | 257 | 12,425 | 48.3463 | 18 |
| jailbreak | 77 | 13,241 | 171.9610 | 149 |

These are existing dataset labels, not live gateway predictions or pass/block
decisions. No unsafe prompts were sent to an LLM; measurement runs locally.

## Tokenizer and scope

Use `tiktoken==0.12.0` with `o200k_base`, the encoding associated with GPT-4o mini
in [OpenAI's tokenizer mapping](https://github.com/openai/tiktoken/blob/0.12.0/tiktoken/model.py).
`encode_ordinary` treats any special-token-looking strings as literal prompt
text. Count only `user_input`, not labels, model responses, or TF-IDF features.
TF-IDF features and word counts are not LLM token counts.

This estimate excludes generated output, system instructions, chat framing,
tool schemas, cache discounts, compression overhead, and all other gateway
costs. It is a reproducible prompt-text input estimate, not an API invoice or
full end-to-end cost. No paid API calls are required.

The source is **already cleaned**, including the earlier 2,000-character
truncation. “Original” here means before October compression, not the raw,
untruncated source prompt. Character limits do not guarantee a 512-token limit;
the maximum is 924 tokens with this tokenizer. We preserve it as-is.

## Reproduce and saved outputs

From the repository root, in the project environment:

```bash
python -m pip install -r requirements.txt
python scripts/measure_token_baseline.py
python -m unittest discover -s tests -p test_token_baseline.py -v
```

The first tokenizer load may download its public encoding vocabulary; subsequent
runs can use the local cache. Reusing the output directory replaces baseline
outputs. `--test-input` and `--output-dir` support alternative paths.

| File | Purpose |
|---|---|
| `scripts/measure_token_baseline.py` | Repeatable local token counting and cost calculation |
| `data/token_baseline/token_baseline.csv` | One result per cleaned test observation |
| `data/token_baseline/token_baseline_summary.json` | Statistics, price source/date, tokenizer version, input checksum |
| `tests/test_token_baseline.py` | Text/row preservation, duplicate IDs, literal special strings, cost arithmetic, invalid input checks |

CSV columns are `prompt_id`, zero-based `test_row`, `prompt_sha256`, `label`,
`original_tokens`, `baseline_input_cost_usd`, `reference_model`, and `encoding`.
Full prompt text is not duplicated into the output. Row IDs distinguish repeated
observations; hashes verify the exact UTF-8 text.

Input SHA-256:
`4f01feab0aa828f4486a20048ef3cd3ecabf26ea24722e86b21e3e802339975e`

Repository source revision:
`1120e7b145adbeff79af37744cdb5a71cb1a98f6`

## Handoff for October compression

Join later results using the **input checksum and test_row**, then verify
`prompt_sha256`. A row ID alone is not stable across a changed test file. Retain
all observations, including duplicates, to match classifier evaluation.
Count compressed text with the same tokenizer and encoding.

For each eligible prompt:

`reduction_percent = 100 × (original_tokens − compressed_tokens) / original_tokens`

Report mean and median per-prompt reduction as requested in issue #19, plus
aggregate reduction `100 × (sum(before) − sum(after)) / sum(before)` over the
**same eligible rows**. Aggregate and average per-prompt reductions are different
measures. Negative reductions must remain visible. Do not treat blocked prompts
as zero-token compression successes; report blocking savings separately.

The 30–50% figure is a target. Of these prompts, 3,207 (65.68%) contain fewer than
20 tokens. Short prompts may offer little room for compression without changing
meaning. If every test prompt could be reduced by 30–50%, the hypothetical
prompt-only savings would be $0.00818–$0.01364 per test-set pass. Actual results
must use the gateway's eligible subset and account for optimization overhead.
