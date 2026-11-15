# Milestone 2: Midpoint evaluation

**Project:** Governed Prompt Gateway, Break Through Tech AI Studio  
**Prepared by:** SinhSinh An  
**Evidence reviewed:** November 14, 2026  
**Assignment:** [Issue #23](https://github.com/Break-Through-Tech/AI-Governance-1C-governed-prompt-gateway/issues/23)  
**Source snapshot:** [`e3d3a4291c088289027a5b0d3bb21508d33c8b2f`](https://github.com/Break-Through-Tech/AI-Governance-1C-governed-prompt-gateway/tree/e3d3a4291c088289027a5b0d3bb21508d33c8b2f)

## Summary

The prototype has working classification, governance, compression, and semantic-cache components. The v2 classifier reports **78.0% macro F1**, compared with the documented SVM baseline of **75.9%**. Conservative prompt compression saves **0.2004%** of eligible input tokens under simulated governance. The cache reports a **6.5% hit rate** at similarity threshold 0.85 in a replay of the test set.

Neither the **80% macro F1** target nor the **30-50% token-reduction** target is met. Full gateway latency and routing cost savings are unavailable. This is a component evaluation report, with simulated results labeled explicitly. It is ready for review, but the final integrated evaluation remains pending #22.

| Measure | Available result | Assessment |
|---|---:|---|
| Classification macro F1 | Baseline 0.759; v2 0.780 | Reported improvement of 2.1 percentage points; below 0.80 |
| Mean / median per-prompt token reduction | 0.5841% / 0% | Simulated governance; below target |
| Aggregate eligible token reduction | 0.2004% | 313 tokens saved across 4,549 eligible prompts |
| Cache-hit rate | 6.5% at threshold 0.85 | Historical replay, without governance gating |
| Cache lookup latency | Notebook mean 5.59 ms; p95 12.07 ms | Component timing only |
| Full gateway latency | Not measured | Awaiting #22 benchmark |
| Routing savings | Not measured | #21 remains open |

## 1. Evaluation data and evidence

All reported component evaluations use the cleaned **4,883-row** test set: 4,549 safe, 257 toxic, and 77 jailbreak prompts. Task #5 created 4,278 training and 754 validation rows, fitting TF-IDF on training text only. v2 fits its final classifier on training plus validation and tunes C using an internal split of those rows.

This report consolidates existing results. No classifier was retrained and no model-based cache benchmark was rerun for this report. Token and compression figures were independently checked against the committed CSVs and JSON summaries. Classifier results are attributed to the source write-ups; cache results are cross-checked against the saved notebook output.

The token and compression artifacts both identify the exact test CSV with SHA-256:

`4f01feab0aa828f4486a20048ef3cd3ecabf26ea24722e86b21e3e802339975e`

**Evidence discrepancies:** The baseline write-up records SVM macro F1 0.759 and names it the official baseline. The saved baseline notebook instead shows 0.756 and an unset official baseline. We use the explicitly recorded 0.759 baseline for the comparison and flag the discrepancy for reconciliation. The cache write-up reports mean/median/p95/max lookup times of 5.64/4.52/12.50/19.80 ms, while the saved notebook shows 5.59/4.58/12.07/18.36 ms. We use the identifiable notebook run for timing. These runs must not be silently combined. The detailed generated evaluation JSONs for these components are gitignored and absent from this checkout.

Sources: [data split](split-and-vectorize.md), [baseline write-up](baseline_eval.md), [baseline notebook](../notebooks/05_model_evaluation.ipynb), [v2 write-up](classifier-v2.md), [cache notebook](../notebooks/06_semantic_cache.ipynb).

## 2. Safety classification

The baseline is an untuned LinearSVC using TF-IDF. v2 uses Logistic Regression with concatenated TF-IDF and `all-MiniLM-L6-v2` sentence embeddings. Its risk score is P(toxic) + P(jailbreak). These probability outputs have not been independently calibrated.

| Class | Support | Baseline precision | Baseline recall | Baseline F1 | v2 precision | v2 recall | v2 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Safe | 4,549 | 0.967 | 0.987 | 0.977 | 0.980 | 0.976 | 0.978 |
| Toxic | 257 | 0.678 | 0.475 | 0.558 | 0.626 | 0.646 | 0.636 |
| Jailbreak | 77 | 0.862 | 0.649 | 0.741 | 0.670 | 0.792 | 0.726 |
| Macro average | 4,883 total | 0.836 | 0.704 | 0.759 | ~0.759 | ~0.805 | 0.780 |

v2 macro precision and recall above are calculated from the rounded per-class values. Other values are the source write-ups' reported test results.

**Interpretation:** Toxic F1 improves by 7.8 percentage points, and jailbreak recall improves by 14.3 points. Jailbreak F1 falls by 1.5 points because precision drops. The tradeoff is more unsafe prompts detected, with more false alarms on legitimate requests. Toxic remains the weakest class. Macro F1 gives all three classes equal weight, avoiding the misleading impression from the dominant safe class.

**Evaluation limitation:** The v2 write-up describes inspecting test errors and comparing test scores across three feature approaches before adopting the combined approach. Although C is selected on an internal training split, this feature-development history makes the reported test gain exploratory. It is not an independent confirmation on a previously untouched final test set. Freeze the model and evaluate on new held-out data before claiming generalization or a confirmed target result.

Sources: [#7 baseline](baseline_eval.md), [#18 v2 classifier](classifier-v2.md).

## 3. Token reduction and estimated input cost

The #8 baseline counts cleaned user-message text with `tiktoken==0.12.0`, encoding `o200k_base`. Across all test prompts it totals **181,855 tokens**, with mean **37.2425**, median **13**, and range **4-924**.

The #19 compressor removes a supported leading request wrapper, such as “Can you,” while retaining the task body. It skips protected wording and keeps the original when compression is unsuitable. The saved run uses dataset labels to simulate governance: safe passes, toxic and jailbreak block. The governance implementation now exists, but these saved results were not generated from its classifier-driven decisions.

| Measure | Result |
|---|---:|
| Eligible / blocked prompts | 4,549 / 334 |
| Compressed / unchanged eligible prompts | 187 / 4,362 |
| Eligible tokens before / after | 156,189 / 155,876 |
| Mean per-prompt reduction, including unchanged prompts | 0.5841% |
| Median per-prompt reduction | 0% |
| Aggregate reduction | 0.2004% |
| Estimated eligible input cost before / after | $0.02342835 / $0.02338140 |
| Estimated input savings per replay | $0.00004695 |

Mean reduction averages each eligible prompt's `100 * (before - after) / before`. Aggregate reduction applies the same formula to summed token counts. Both use the same 4,549 eligible rows. Blocked requests are excluded, rather than counted as compression successes.

Costs retain #8's recorded **$0.15 per million input tokens** reference rate, checked there on October 7, 2026. This is a historical comparison assumption, not a fresh price quote. The full test-set baseline is $0.02727825. Estimates exclude output tokens, system instructions, request framing, caching discounts, and compression compute. Net gateway cost savings have not been measured.

**Meaning check:** The saved compressor requires exact ordered task-body lexical equivalence, threshold 1.0, after removing at most one supported wrapper. This guards task wording but shares the compressor's assumptions. It does not establish independent semantic preservation or response quality.

**Conclusion:** Wrapper removal alone misses the 30-50% target. Most inputs are already short or remain unchanged. [Issue #30](https://github.com/Break-Through-Tech/AI-Governance-1C-governed-prompt-gateway/issues/30) proposes a separately labeled simulated customer-service workload with system instructions, context, and history. It is unassigned and has no committed results in this snapshot.

Sources: [token baseline](token-baseline.md), [baseline JSON](../data/token_baseline/token_baseline_summary.json), [compression write-up](prompt-compression.md), [compression JSON](../data/compression_demo/compression_summary.json), [per-prompt compression CSV](../data/compression_demo/compression_results.csv).

## 4. Semantic cache

The cache uses normalized sentence embeddings and a FAISS inner-product index. The streaming evaluation starts with an empty cache, replays all 4,883 test prompts in their original order, checks for a match, and stores responses on misses. Hit rate is hits divided by lookups. Responses are recovered from the source dataset, not generated by a live LLM.

| Similarity threshold | Streaming hit rate | Duplicate-match precision | Duplicate-match recall |
|---:|---:|---:|---:|
| 0.70 | 12.0% | 0.243 | 1.000 |
| 0.75 | 9.7% | 0.301 | 1.000 |
| 0.80 | 7.8% | 0.358 | 1.000 |
| 0.85 | 6.5% | 0.429 | 1.000 |
| 0.90 | 5.3% | 0.512 | 1.000 |
| 0.95 | 4.1% | 0.644 | 1.000 |

Duplicate validation separately checks the nearest other prompt against normalized text-duplicate groups. Its precision and recall describe that proxy task, not answer correctness or streaming hit quality. All 115 duplicate groups are near-identical normalized text, so recall 1.000 does not prove robustness to reworded paraphrases. The write-up reports 20 reasonable sample hits at 0.85, which is a limited manual check. Threshold 0.85 is a working default, not a validated optimum.

The saved notebook's CPU lookup run includes embedding and search, one prompt at a time: **mean 5.59 ms, median 4.58 ms, p95 12.07 ms, max 18.36 ms**. This is cache lookup timing, not full request latency. The replay includes all labels without governance gating, so its hit rate cannot be presented as an allowed-only gateway rate. #29 tracks that safety requirement. The source also flags 24 rows with ambiguous recovered responses.

Sources: [cache write-up](semantic-cache.md), [saved cache notebook](../notebooks/06_semantic_cache.ipynb), [evaluation method](../scripts/evaluate_semantic_cache.py).

## 5. Integration, routing, and remaining work

At review time, [#22](https://github.com/Break-Through-Tech/AI-Governance-1C-governed-prompt-gateway/issues/22) remains open. Its draft [PR #32](https://github.com/Break-Through-Tech/AI-Governance-1C-governed-prompt-gateway/pull/32), head `4bcebd1334f64359d94087ad7424a97683669ba4`, contains schemas, adapters, and an orchestrator. The PR explicitly lists the FastAPI app, automated tests, and benchmark as unfinished. It uses a stub router and simulated generator, and its classifier adapter still references the SVM rather than adopted v2.

**Full gateway mean, median, and p95 latency: unavailable.** Do not substitute cache lookup timing or sum independently measured component times. **Routing distribution and savings versus the largest-model baseline: unavailable**, since #21 remains open. No claim about paid inference latency or measured production savings is supported.

| Next step for November | Existing owner / issue | Evidence needed |
|---|---|---|
| Complete and benchmark the integrated gateway | Manuel, #22 | Revision, classifier used, hardware, workload, cold/warm behavior, timing boundaries, mean/median/p95, failures, and cache state |
| Reconcile baseline artifacts and confirm v2 independently | Barsat #7; Angelina #18 | Aligned reports with versions and a new held-out evaluation after freezing the approach |
| Re-evaluate compression with actual governance decisions | SinhSinh, follow-up to #19 | Classifier/version, aligned decisions and before/after CSV; retain the simulation separately |
| Calibrate and validate allowed-only caching | #29, currently unassigned | Zero cache responses for blocked prompts and reviewed answer quality |
| Measure the proposed customer-service simulation | #30, currently unassigned | Fixed workload, source-level savings, meaning checks, honest target comparison |
| Add routing results if delivered | Prince, #21 | Model distribution, dated prices, largest-model comparison |
| Finalize this report and submit to Canvas | SinhSinh, #23 | Add integrated latency and resolve material evidence discrepancies |

Issue #30's references to #4 as the report and #6 as the dashboard do not match the repository's current issue titles. The evaluation report is #23; no dashboard issue appears in the reviewed list. Those handoff references need correction before relying on them.

## 6. Reproduction and completion status

Existing component commands, run from the repository root in the documented Python environment:

```bash
python -m pip install -r requirements.txt
python scripts/split_and_vectorize.py
python scripts/train_baseline_models.py
python scripts/evaluate_baseline.py
python scripts/train_classifier_v2.py
python scripts/measure_token_baseline.py
python scripts/compress_prompts.py --demo-governance --output-dir data/compression_demo
python scripts/recover_test_responses.py
python scripts/evaluate_semantic_cache.py
```

These commands regenerate component artifacts, may download the embedding model, and do not produce an integrated gateway benchmark. Generated model/cache reports are under gitignored `data/processed/`. Preserve each run's environment, model revision, input hashes, and outputs when reconciling discrepancies.

For this report, checks confirmed the test-file checksum, all 4,883 baseline/compression row IDs and prompt hashes, token totals, reduction statistics, and cost arithmetic against the committed data. No application behavior changes are included.

**#23 status:** Report, comparison tables, cache methodology, limitations, and November next steps are prepared. Full gateway latency is still a required missing dependency. Keep #23 open until those measurements are added. Routing is optional under its acceptance criteria if #21 is unfinished. Canvas submission is a separate action after the report is finalized.
