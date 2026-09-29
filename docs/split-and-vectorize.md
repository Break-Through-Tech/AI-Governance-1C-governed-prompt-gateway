# Dataset split and TF-IDF features

This is the handoff from [issue #5](https://github.com/Break-Through-Tech/AI-Governance-1C-governed-prompt-gateway/issues/5)
to the baseline classifier task. It reads `data/train_clean.csv` and
`data/test_clean.csv`, the task 4 cleanup outputs committed in PR #15. These are
the files referred to as `clean_train.csv` and `clean_test.csv` in the review.
Only `user_input` is vectorized, and the lowercase `safe`, `toxic`, and
`jailbreak` labels are kept unchanged.

## Run it

Use Python 3.12 or later. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/split_and_vectorize.py
python -m unittest discover -s tests -v
```

Both cleaned CSVs are already committed under `data/`. No earlier notebook needs
to run first. This pipeline does not read `combined_risk_dataset.csv` or the raw
archives. The same preparation is available in
[`03_split_and_vectorize.ipynb`](../notebooks/03_split_and_vectorize.ipynb); run it
with this environment's Python kernel in your notebook editor.

## Split decision

The task 4 notebook recommends holding out 15% of cleaned training data for
validation and keeping the cleaned test set for final evaluation. This follows
that handoff and the review request to use the cleaned CSVs.

- Start with the 5,082 rows in `train_clean.csv`.
- Exclude 50 training rows matching a cleaned test prompt after Unicode NFKC
  normalization, case folding, and whitespace collapse. Cleanup removed exact
  overlap, but these variants remain. The split report records the exclusion.
- Split the remaining **duplicate groups** 85/15, stratified by label, with seed
  42. Every copy stays in one split, so row proportions differ slightly from
  group proportions. A conflicting label within a training group raises an error.
- Keep all 4,883 rows in `test_clean.csv`, including their text, labels, and order.
  Source CSVs are never modified.

The default run on the committed dataset produces:

| Split | safe | toxic | jailbreak | Total rows | Prompt groups |
|---|---:|---:|---:|---:|---:|
| Train | 3,957 | 226 | 95 | 4,278 | 4,182 |
| Validation | 696 | 42 | 16 | 754 | 739 |
| Test | 4,549 | 257 | 77 | 4,883 | 4,764 |

There are no shared duplicate groups across these splits. Duplicates within a
split still count as separate observations, which matters when interpreting
metrics. Validation has only 16 jailbreak examples, so per-class scores will be
sensitive to a small number of mistakes.

## Vectorization and saved files

TF-IDF uses up to 10,000 features, word unigrams and bigrams, and sublinear term
frequency. The vocabulary and IDF weights are fitted on training prompts only;
validation and test use `transform`. Cleaned text is used exactly as saved,
including the cleanup notebook's 2,000-character truncation. Normalization is
used only for duplicate grouping. Labels are never included in the feature text.

Outputs go to `data/processed/tfidf/`, which is ignored by Git:

| Files | Contents |
|---|---|
| `train.csv`, `val.csv`, `test.csv` | Cleaned user_input and label rows in feature-matrix order |
| `X_train.npz`, `X_val.npz`, `X_test.npz` | Sparse TF-IDF matrices |
| `y_train.npy`, `y_val.npy`, `y_test.npy` | String labels in the same row order |
| `tfidf_vectorizer.joblib` | Fitted vectorizer for new prompts |
| `class_weights.json` | Balanced weights calculated from the final training split only |
| `split_report.json` | Both input checksums, excluded overlap count, seed, package versions, shapes and counts |

The default matrices have 10,000 columns. Some prompts yield no known features
(11 train, 5 validation, 34 test rows in this run). They remain as zero vectors
so the original row/label alignment is preserved; counts appear in the report.

Use `--seed`, `--validation-size`, `--train-input`, `--test-input`, or
`--output-dir` to change a run. `--validation-size` is a fraction of training
groups after excluding test overlap. Reusing an output directory replaces its generated artifacts.

## Load the inputs for task 6

From the repository root:

```python
from pathlib import Path
import json
import joblib
import numpy as np
from scipy.sparse import load_npz

data = Path('data/processed/tfidf')
X_train = load_npz(data / 'X_train.npz')
X_val = load_npz(data / 'X_val.npz')
y_train = np.load(data / 'y_train.npy', allow_pickle=False)
y_val = np.load(data / 'y_val.npy', allow_pickle=False)
class_weights = json.loads((data / 'class_weights.json').read_text())
vectorizer = joblib.load(data / 'tfidf_vectorizer.joblib')
X_new = vectorizer.transform(['Explain how a solar panel works.'])
```

Compare classifiers on validation data, then load `X_test.npz` and `y_test.npy`
for final evaluation. Use the newly saved class weights where the classifier
supports them; weights from before the validation split no longer describe the
training data. If adding cross-validation, refit TF-IDF within each training fold.
The supplied matrices are for the fixed train/validation protocol above.
