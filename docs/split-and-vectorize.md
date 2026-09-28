# Dataset split and TF-IDF features

This is the handoff from [issue #5](https://github.com/Break-Through-Tech/AI-Governance-1C-governed-prompt-gateway/issues/5)
to the baseline classifier task. It reads `data/combined_risk_dataset.csv` from
task 2, using `prompt` as the only feature and keeping the `Safe`, `Toxic`, and
`Jailbreak` labels unchanged.

## Run it

Use Python 3.12 or later. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/build_risk_dataset.py --check
python scripts/split_and_vectorize.py
python -m unittest discover -s tests -v
```

The input CSV is already committed. If it is missing or stale, regenerate it with
`python scripts/build_risk_dataset.py`. You do not need to run the EDA or cleanup
notebooks first. The same preparation is available in
[`03_split_and_vectorize.ipynb`](../notebooks/03_split_and_vectorize.ipynb); run it
with this environment's Python kernel in your notebook editor.

## Split decision

The issue suggests 70/15/15 or 80/10/10. However, the
[task 2 dataset contract](risk-taxonomy.md#split-preservation-and-downstream-handoff)
requires preserving Toxic Chat's original test set. The task 4 notebook also
recommends keeping that test set and holding out 15% of training for validation.
This implementation follows those instructions instead of reshuffling the
combined dataset to force an overall ratio.

- Only `training_eligible=true` rows can enter training or validation. The task 2
  quality flags exclude conflicting labels and overlap with reserved data.
- Split the eligible **duplicate groups** 85/15, stratified by label, with seed 42.
  Every copy of a normalized prompt stays in one split. Original rows are kept,
  so the row ratio can differ slightly from the group ratio.
- Keep all 5,083 original Toxic Chat test rows, in their original order.
- Leave JBB behaviors and judge-comparison records out of these classifier
  artifacts. They remain separate benchmark/evaluation data in task 2.

The default run on the committed dataset produces:

| Split | Safe | Toxic | Jailbreak | Total rows | Prompt groups |
|---|---:|---:|---:|---:|---:|
| Train | 3,819 | 217 | 81 | 4,117 | 4,023 |
| Validation | 669 | 37 | 14 | 720 | 711 |
| Test | 4,721 | 271 | 91 | 5,083 | 4,951 |

There are no shared duplicate groups across these splits. The 745 unused rows
are 245 ineligible Toxic Chat training rows and 500 JBB records. Duplicates within
a split still count as separate observations, which matters when interpreting
metrics. Validation has only 14 jailbreak examples, so per-class scores will be
sensitive to a small number of mistakes.

## Vectorization and saved files

TF-IDF uses up to 10,000 features, word unigrams and bigrams, and sublinear term
frequency. The vocabulary and IDF weights are fitted on training prompts only;
validation and test use `transform`. Full prompt text is retained: this pipeline
does not apply the cleanup notebook's 2,000-character truncation. TF-IDF does not
require that limit, and preserving text also preserves task 2's duplicate groups.
Metadata, annotations, and labels are never included in the feature text.

Outputs go to `data/processed/tfidf/`, which is ignored by Git:

| Files | Contents |
|---|---|
| `train.csv`, `val.csv`, `test.csv` | Original rows and provenance in feature-matrix order |
| `X_train.npz`, `X_val.npz`, `X_test.npz` | Sparse TF-IDF matrices |
| `y_train.npy`, `y_val.npy`, `y_test.npy` | String labels in the same row order |
| `tfidf_vectorizer.joblib` | Fitted vectorizer for new prompts |
| `class_weights.json` | Balanced weights calculated from the final training split only |
| `split_report.json` | Input checksum, seed, package versions, settings, shapes and counts |

The default matrices have 10,000 columns. Some prompts yield no known features
(10 train, 4 validation, 33 test rows in this run). They remain as zero vectors
so the original row/label alignment is preserved; counts appear in the report.

Use `--seed`, `--validation-size`, `--input`, or `--output-dir` to change a run.
`--validation-size` is a fraction of eligible training groups, not of the entire
combined dataset. Reusing an output directory replaces its generated artifacts.

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
