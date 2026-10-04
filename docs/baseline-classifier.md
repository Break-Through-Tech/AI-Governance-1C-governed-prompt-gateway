# Baseline classifier comparison

This is the handoff for issue #6. It trains and compares baseline models on
the TF-IDF features from task 5 (`data/processed/tfidf/`) and selects the
best one by **validation** macro F1. 

It reports two comparisons side by
side — **untuned** (default hyperparameters) and **tuned** (grid search per
model) — and keeps both; it does not collapse them into a single "the"
baseline. See [Hyperparameter tuning](#hyperparameter-tuning) below.

This task is validation-only by design. The held-out test set, a confusion
matrix, and the official "performance floor" documentation for comparing
against future models are issue #7's scope, not this one's — see
`scripts/train_baseline_models.py`'s `load_splits()`, which only reads
`X_train`/`y_train`/`X_val`/`y_val` and never opens `X_test.npz`/`y_test.npy`.

## Why these four models

Issue #6 names KNN, Logistic Regression, SVM, and Naive Bayes directly, and
that's a reasonable set for a *baseline* step, for a few reasons:

- **They're classic, well-established text-classification baselines**, not
  an arbitrary pick. Naive Bayes in particular has been a standard choice
  for text classification (e.g. spam filtering) for decades; linear SVMs
  and logistic regression are the standard comparison points once TF-IDF
  features are involved.
- **They cover genuinely different algorithmic approaches**, not four
  variations on the same idea — useful for a first baseline, since a weak
  result from one family (e.g. instance-based KNN) doesn't tell you much
  about whether a different family (e.g. a linear model) would also
  struggle:
  - KNN — instance-based, no real training step, classifies by local
    similarity to stored examples.
  - Logistic Regression — a linear, probabilistic model.
  - SVM (LinearSVC) — also linear, but margin-maximizing rather than
    probability-based.
  - Naive Bayes — a generative, probabilistic model built around word
    frequencies, which is why `MultinomialNB` specifically (built for
    count/frequency features) fits TF-IDF input.
- **All four are fast to train at this data size** 
- **All four ship in scikit-learn**, already a project dependency via
  `requirements.txt` — no new libraries needed to run this comparison.


## Run it

```bash
python scripts/split_and_vectorize.py      # if data/processed/tfidf/ is missing
python scripts/train_baseline_models.py
python -m unittest discover -s tests -v
```

The same preparation is available in
[`04_baseline_classifier.ipynb`](../notebooks/04_baseline_classifier.ipynb).

## Models

Four candidates, all trained on `X_train`/`y_train` from task 5:

| Model | Library class | Class imbalance handling |
|---|---|---|
| KNN | `KNeighborsClassifier(n_neighbors=5)` | None — scikit-learn's KNN has no `class_weight` or `sample_weight` fit parameter, so this model sees the raw 93%/5%/2% split |
| Logistic Regression | `LogisticRegression(class_weight=class_weights, max_iter=2000)` | `class_weight` set to task 5's computed balanced weights |
| SVM | `LinearSVC(class_weight=class_weights, max_iter=5000)` | Same `class_weight` dict. LinearSVC rather than kernel `SVC`: standard choice for high-dimensional sparse TF-IDF text features, and far faster at this scale |
| Naive Bayes | `MultinomialNB()` | No `class_weight` parameter, but `fit` accepts `sample_weight`; each training row is weighted by its class's weight as a substitute |

Each model is fit once on `train` and evaluated on `validation`. The
selected model's trained object is saved for later use.

## Hyperparameter tuning

Beyond the untuned defaults above, each model is also grid-searched over a
small set of its own hyperparameters, fit on `train`, scored on `validation`
macro F1 — same split, same metric, same rules as the untuned comparison.

| Model | Hyperparameters searched | Values |
|---|---|---|
| KNN | `n_neighbors`, `metric`, `weights` | `n_neighbors`: 3, 5, 7, 9, 15, 25 · `metric`: minkowski, cosine · `weights`: uniform, distance |
| Logistic Regression | `C` (inverse regularization strength) | 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0 |
| SVM | `C` | 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0 |
| Naive Bayes | `alpha` (Laplace smoothing) | 1.0, 0.1, 0.01, 0.001, 0.0001 |

`class_weight`/`sample_weight` handling is left exactly as in the untuned
comparison — only the listed hyperparameters vary. **Every grid includes the
untuned default** (e.g. KNN's grid includes `n_neighbors=5, metric=minkowski,
weights=uniform`), so a tuned model's validation score can never be worse
than its untuned counterpart's, only equal or better. Every candidate's
score is recorded in `model_comparison.json`'s `tuned.grid_results`, not
just the winner's.

## Why macro F1

Precision/recall/F1 are computed three ways: 
**macro** - unweighted mean across safe/toxic/jailbreak — each class counts equally
**weighted** - mean weighted by class size — dominated by the 93% safe class
**per-label** - each class on its own
**Macro F1 is the selection metric.** Given the
project's purpose — catching toxic and jailbreak prompts, not just safe ones
— a metric dominated by how well a model predicts "safe" would reward a
model for ignoring the two classes that actually matter. Weighted and
per-label numbers are recorded too, for context.

## Results (validation set)

| Model | Macro P | Macro R | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| KNN | 0.559 | 0.396 | 0.421 | 0.895 |
| Logistic Regression | 0.659 | 0.759 | 0.700 | 0.929 |
| **SVM** | **0.832** | 0.668 | **0.732** | 0.945 |
| Naive Bayes | 0.478 | 0.786 | 0.522 | 0.841 |

Per-label F1 on validation (support in parentheses):

| Model | jailbreak (16) | safe (696) | toxic (42) |
|---|---:|---:|---:|
| KNN | 0.300 | 0.963 | 0.000 |
| Logistic Regression | 0.611 | 0.961 | 0.529 |
| SVM | 0.692 | 0.976 | 0.528 |
| Naive Bayes | 0.265 | 0.879 | 0.420 |

KNN's weighted F1 (0.895) looks competitive despite a macro F1 of 0.421 —
it predicts `safe` well (96% of the data) and almost never predicts
`toxic` correctly (F1 0.000), which is exactly the failure mode macro F1
is meant to expose.

## Tuned results (validation set)

| Model | Best params | Untuned Macro F1 | Tuned Macro F1 | Δ |
|---|---|---:|---:|---:|
| KNN | `metric=cosine, n_neighbors=3, weights=distance` | 0.421 | 0.678 | +0.257 |
| Logistic Regression | `C=10.0` | 0.700 | 0.738 | +0.038 |
| **SVM** | `C=1.0` (= untuned default) | 0.732 | 0.732 | +0.000 |
| Naive Bayes | `alpha=0.001` | 0.522 | 0.657 | +0.135 |

Tuning helps every model except SVM, whose untuned default was already the
best value in its own search range. KNN benefits the most by far — almost
entirely from switching its distance metric from `minkowski` (Euclidean,
the scikit-learn default) to `cosine`, which is the standard distance for
high-dimensional sparse TF-IDF text data; Euclidean distance is known to
behave poorly in that setting.

Full tuned per-label F1 on validation:

| Model | jailbreak (16) | safe (696) | toxic (42) |
|---|---:|---:|---:|
| KNN | 0.615 | 0.967 | 0.453 |
| Logistic Regression | 0.667 | 0.970 | 0.578 |
| SVM | 0.692 | 0.976 | 0.528 |
| Naive Bayes | 0.465 | 0.953 | 0.552 |

## Best model: untuned SVM, tuned Logistic Regression — not the same model

**Untuned:** SVM, by highest validation macro F1 (0.732). SVM also has the
highest macro precision (0.832) — when it flags a prompt as toxic or
jailbreak, it is right more often than the alternatives. Logistic Regression
has higher macro recall (0.759 vs 0.668) and is a reasonable second choice
if missing a jailbreak is considered worse than a false alarm; that
trade-off is a policy decision, not purely a modeling one.

**Tuned:** Logistic Regression (`C=10.0`), by highest validation macro F1
(0.738) — it **overtakes** untuned SVM once tuned, which stayed at 0.732
since its own best `C` was already its default. Tuned Logistic Regression
also has noticeably better balance between precision and recall (0.743 vs
0.737) than its untuned version did (0.659 vs 0.759), and a meaningfully
higher `toxic` F1 (0.578 vs untuned SVM's 0.528).

**This document deliberately does not pick one of these two as "the"
baseline.** Both `best_model.pkl` (untuned SVM) and `best_model_tuned.pkl`
(tuned Logistic Regression) are saved; `model_comparison.json` records both
in full. Issue #7 decides which one to carry forward for the held-out
test-set evaluation.

## Against the project's success criterion

The project overview's target is 80%+ F1 on prompt-risk classification.
Neither comparison reaches it on **validation**: untuned SVM is at 0.732,
tuned Logistic Regression at 0.738. Both gaps are driven mainly by the
`toxic` class (F1 0.528–0.578 across the two) and by `jailbreak`'s small
support (16 rows) making its F1 noisy. This is a real gap in both cases,
not glossed over here, and tuning alone does not close it.



## Limitations

- **KNN's weak minority-class performance partly reflects a tooling limit**,
  not only the algorithm: scikit-learn's `KNeighborsClassifier` has no
  reweighting mechanism, so it never saw the class imbalance during fit.
- **LinearSVC has no `predict_proba`** (only `decision_function`). If the
  governance gateway later needs a calibrated confidence score rather than
  a hard label, this model would need `CalibratedClassifierCV` wrapped
  around it, or a different model chosen.
- **Validation `jailbreak` support is small** (16 rows), so its F1 can swing
  with a handful of predictions. The test set has more jailbreak rows, so
  issue #7's numbers should be more stable
- **Naive Bayes's `sample_weight` substitute is not equivalent to a true
  `class_weight`** — it weights the loss contribution of each row but
  doesn't change the class priors the same way `class_weight` does for the
  other two weighted models.
- **Tuning was grid search over a hand-picked, small set of values**, not
  an exhaustive or randomized search, and it was scored on the same fixed
  validation split used for model selection (not k-fold cross-validation).
  That's consistent with how this pipeline already uses validation
  elsewhere, but it means the tuned numbers above carry the same small-
  validation-set noise as the untuned ones, especially for `jailbreak`
  (16 rows).
- **`C`/`alpha` grids stop at 1000/0.0001 respectively.** Logistic
  Regression's best value (`C=10`) and SVM's (`C=1`) were both comfortably
  inside the searched range, not at an edge, so there's no strong reason to
  believe extending the range further would help. Naive Bayes's best
  `alpha` (0.001) was one step from the grid's edge (0.0001) — a wider
  search in that direction wasn't tried.

## Output files

`data/processed/models/` (gitignored, same as `data/processed/tfidf/`):

| File | Contents |
|---|---|
| `best_model.pkl` | The untuned best model (SVM), loadable with `pickle.load`. Not yet evaluated on test — that's issue #7 |
| `best_model_tuned.pkl` | The tuned best model (Logistic Regression, `C=10.0`), loadable with `pickle.load`. Also not yet evaluated on test |
| `model_comparison.json` | Full report: seed, class weights, selection rule, every model's untuned validation metrics (macro/weighted/per-label), and — under `tuned` — every model's best hyperparameters, best validation metrics, and the full per-combination grid search results |

Use `--data-dir`, `--output-dir`, or `--seed` to change a run.
