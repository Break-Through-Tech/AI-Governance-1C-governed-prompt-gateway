# Baseline model evaluation (performance floor)

This is the handoff for [issue #7](https://github.com/Break-Through-Tech/AI-Governance-1C-governed-prompt-gateway/issues/7).
It takes the two baseline models issue #6 selected on the **validation** split
and evaluates them, for the first time, on the **held-out test set**. The
numbers here are the project's **baseline performance floor**: the minimum bar
any future model (October's work) must beat to count as progress.

Reproduce with:

```bash
python scripts/split_and_vectorize.py      # if data/processed/tfidf/ is missing
python scripts/train_baseline_models.py    # if data/processed/models/ is missing
python scripts/evaluate_baseline.py
python -m unittest discover -s tests -v
```

The same run is available in
[`05_model_evaluation.ipynb`](../notebooks/05_model_evaluation.ipynb).

## Why the test set, and why only now

Issue #6 compared four classifiers (KNN, Logistic Regression, SVM, Naive Bayes)
on **validation** and used those scores to pick winners. The test set was never
read — see that task's `load_splits()`, which only opens the train and
validation files. Reusing validation to also *report* final numbers would
flatter the models, because validation already influenced which models were
chosen. So the test set is held back for exactly one purpose: a single, honest
measurement, taken here.

Two models carried over from issue #6, each the winner of its own comparison:

| Candidate | Model | Selected on validation by |
|---|---|---|
| `untuned` | SVM (`LinearSVC`, `C=1.0`) | best untuned validation macro F1 (0.732) |
| `tuned` | Logistic Regression (`C=10.0`) | best tuned validation macro F1 (0.738) |

Both are evaluated on test below. **This document does not silently pick one as
"the" baseline** — see [The baseline decision](#the-baseline-decision).

## Test set

4,883 rows, read from `data/processed/tfidf/X_test.npz` / `y_test.npy`:

| Class | Support |
|---|---:|
| safe | 4,549 |
| toxic | 257 |
| jailbreak | 77 |

The test `jailbreak` support (77) is far larger than validation's (16), so these
per-class numbers are more stable than issue #6's validation ones.

## Results (test set)

Macro = unweighted mean across the three classes (each counts equally; the
selection metric). Weighted = mean weighted by class size (dominated by the 93%
`safe` class). Accuracy = overall fraction correct.

| Candidate | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| **untuned — SVM** | 0.955 | **0.836** | 0.704 | **0.759** | 0.951 |
| tuned — Logistic Regression | 0.945 | 0.736 | **0.746** | 0.741 | 0.945 |

Per-class precision / recall / F1 (support in parentheses):

| Class | SVM P | SVM R | SVM F1 | LogReg P | LogReg R | LogReg F1 |
|---|---:|---:|---:|---:|---:|---:|
| safe (4,549) | 0.967 | 0.987 | 0.977 | 0.972 | 0.971 | 0.971 |
| toxic (257) | 0.678 | 0.475 | 0.558 | 0.557 | 0.553 | 0.555 |
| jailbreak (77) | 0.862 | 0.649 | 0.741 | 0.679 | 0.714 | 0.696 |

## Confusion matrices

Rows are the true class, columns the predicted class. The off-diagonal cells are
the misclassifications; the one that matters most for a safety gateway is
**toxic or jailbreak predicted as `safe`** (an unsafe prompt let through).

**Untuned SVM**

| true \ pred | jailbreak | safe | toxic |
|---|---:|---:|---:|
| jailbreak | 50 | 22 | 5 |
| safe | 4 | 4492 | 53 |
| toxic | 4 | 131 | 122 |

**Tuned Logistic Regression**

| true \ pred | jailbreak | safe | toxic |
|---|---:|---:|---:|
| jailbreak | 55 | 17 | 5 |
| safe | 23 | 4418 | 108 |
| toxic | 3 | 112 | 142 |

What the matrices show:

- The dominant error for **both** models is `toxic` → `safe`: SVM lets **131 of
  257** toxic prompts through as safe, Logistic Regression **112 of 257**. This
  is the single biggest weakness of the baseline, and the reason the `toxic` F1
  (~0.56 for both) drags macro F1 well below the 80% target.
- Logistic Regression leaks fewer unsafe prompts to `safe` (112 + 17 = 129 vs
  SVM's 131 + 22 = 153 toxic/jailbreak rows misread as safe), consistent with
  its higher macro recall — but it raises more false alarms on `safe`
  (131 safe rows flagged toxic/jailbreak vs SVM's 57), consistent with its lower
  precision.

## Validation vs test: the ranking flips

| Candidate | Validation macro F1 | Test macro F1 |
|---|---:|---:|
| untuned — SVM | 0.732 | **0.759** |
| tuned — Logistic Regression | **0.738** | 0.741 |

On validation, tuned Logistic Regression edged out SVM. On test, **SVM is
higher**. Both models actually scored *better* on test than on validation —
largely because the test set has many more `jailbreak` rows (77 vs 16), so that
class's F1 is less noisy. The flip is small (0.759 vs 0.741) and does not by
itself settle the baseline choice; see below.

## Against the project's success criterion

The project target is **80%+ F1** on prompt-risk classification. On the held-out
test set, the best candidate reaches **macro F1 0.759** (SVM) — **the target is
not met**. The gap is driven almost entirely by the `toxic` class (F1 ~0.56 for
both models), whose prompts are most often misread as `safe`. This is an honest
baseline gap for October's work to close, not a result to round up.

Note weighted F1 (~0.95 for both) *does* clear 80%, but that number is dominated
by the easy 93% `safe` class and is not the project's stated metric — it is
reported only for context.

## Performance floor

These are the numbers October's improvements must beat to count as progress,
recorded from the held-out test set. The authoritative machine-readable copy is
`data/processed/eval/test_evaluation.json`.

| Metric (test set) | Untuned SVM | Tuned Logistic Regression |
|---|---:|---:|
| Macro F1 **(primary)** | **0.759** | 0.741 |
| Macro precision | 0.836 | 0.736 |
| Macro recall | 0.704 | 0.746 |
| Weighted F1 | 0.951 | 0.945 |
| Accuracy | 0.955 | 0.945 |
| toxic F1 | 0.558 | 0.555 |
| jailbreak F1 | 0.741 | 0.696 |
| toxic recall (unsafe caught) | 0.475 | 0.553 |

## The baseline decision

Issue #7 requires one model to be marked as "the baseline." The two candidates
trade off against each other rather than one dominating, so this is a policy
call for the team, not a purely numeric one:

- **Untuned SVM** — higher macro F1 (0.759) and much higher macro precision
  (0.836). When it flags a prompt as toxic/jailbreak it is right more often, so
  fewer false alarms on legitimate `safe` traffic. Caveat: `LinearSVC` has no
  `predict_proba`, so a calibrated confidence score for the gateway would need
  `CalibratedClassifierCV` or a different model.
- **Tuned Logistic Regression** — higher macro recall (0.746) and higher
  `toxic`/`jailbreak` recall, i.e. it lets fewer unsafe prompts slip through as
  `safe`. For a safety-first gateway, missing a toxic prompt may be worse than a
  false alarm. It also natively provides `predict_proba`, useful if the gateway
  later needs confidence thresholds.

> **TEAM DECISION.** Official baseline: **Untuned SVM** (test macro F1: 0.759). Rationale: highest macro F1 on the held-out test set. October's governance work will target the toxic class recall gap as the primary improvement area.

## Limitations

- **Neither model meets the 80% F1 target**; the `toxic` class is the bottleneck
  and the main target for October.
- **`jailbreak` support is still modest** (77 test rows), so its per-class F1,
  while more stable than validation's 16, can still move with a handful of
  predictions.
- **Only the two validation-selected winners are evaluated on test**, not all
  four models from issue #6. That is deliberate: evaluating models that were
  already rejected on validation against the held-out set would erode the
  test set's role as an untouched final check. The full four-model validation
  comparison lives in [`baseline-classifier.md`](baseline-classifier.md).
- **This is a single fixed test split**, not cross-validation, so the numbers
  carry the usual single-split variance.

## Output files

`data/processed/eval/` (gitignored, like the other `data/processed/` outputs):

| File | Contents |
|---|---|
| `test_evaluation.json` | Full report: labels, seed, selection rule, 80% criterion, per-candidate test accuracy + macro/weighted/per-label precision/recall/F1 + confusion matrix, each candidate's validation metrics for comparison, and `best_by_test_macro_f1`. `official_baseline` is **untuned SVM**. |

Use `--data-dir`, `--models-dir`, or `--output-dir` to change a run.
