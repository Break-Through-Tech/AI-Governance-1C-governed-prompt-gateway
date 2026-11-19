# Semantic cache

Builds an embedding-based cache: a new prompt can reuse an earlier response
if it's close enough in *meaning*, not just identical text, and measures how
often that actually helps on the cleaned train+test data, combined.

## Run it

```bash
python -m pip install -r requirements.txt
python scripts/recover_responses.py
python scripts/evaluate_semantic_cache.py
python -m unittest discover -s tests -v
```


## Model and vector store

- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` — 384
  dimensions, ~80MB, CPU-fast, the standard lightweight choice for this kind
  of task.
- **Vector store**: FAISS (`faiss-cpu`), `IndexFlatIP` over L2-normalized
  vectors (inner product of unit vectors = cosine similarity)

## The cache API (`scripts/semantic_cache.py`)

```python
cache = SemanticCache(threshold=0.85)        
cache.store(prompt, response)
result = cache.lookup(prompt)                
result.hit             
result.matched_prompt  
result.response        
result.similarity       
result.latency_seconds 
```

## Recovering response text

`data/test_clean.csv` and `data/train_clean.csv` only have `user_input`/
`label` — task 4's cleanup dropped `model_output`, since the classifier
never needed it. The cache needs real response text to store and to let a
reviewer judge whether a cached answer actually fits, so
`scripts/recover_responses.py` rejoins each cleaned split's prompts to the
matching split of the raw `lmsys_toxicchat.zip` (`test.csv` or `train.csv`),
by exact text match (no row needed prefix-matching — the longest cleaned
row is 1,536 characters, under the 2,000-character cap, so cleanup never
actually truncated anything in either split). All 4,883 test rows and all
5,082 train rows find an exact match; both splits are exactly the raw
zip's matching split, cleaned, with no other source mixed in.


## How hit rate is measured, and why two methods

The combined data has few exact repeats, so a single "X% hit rate" number
has nothing to validate it against on its own. Two measurements instead,
swept across six thresholds (0.70-0.95), over the **9,965 combined rows**
(5,082 train + 4,883 test, train rows first):

**1. Streaming simulation** — replay all 9,965 prompts in order (train
rows first, then test), as if they were one continuous live query stream.
Each prompt checks the cache first (a hit reuses the response, nothing new
is stored); a miss stores it for later prompts to potentially match. The
classifier's train/test split has no bearing on the cache (it never trains
on anything — see the sections below), so there's no reason to hold train
data out of this measurement; it's folded in as more of the same kind of
data, not treated specially.

**2. Duplicate-ground-truth validation** — using the combined data's own
real near-duplicate groups (258 groups with more than one row, found across
train+test together), check whether each prompt's single nearest *other*
prompt is correctly judged a match at each threshold

## Results

| Threshold | Hit rate | Precision | Recall |
|---:|---:|---:|---:|
| 0.70 | 0.167 | 0.201 | 1.000 |
| 0.75 | 0.137 | 0.246 | 1.000 |
| 0.80 | 0.114 | 0.289 | 1.000 |
| **0.85** | **0.096** | **0.338** | **1.000** |
| 0.90 | 0.077 | 0.408 | 1.000 |
| 0.95 | 0.056 | 0.537 | 1.000 |

(Bold row = `REPORT_THRESHOLD`, the default used for the saved hit sample
and the latency measurement below — a starting point, not a claimed best
value)

**Reading this**: as the threshold tightens, hit rate falls (fewer things
count as close enough) while precision rises (fewer wrong matches slip
through). Hit rate is higher than the old test-only numbers at every
threshold (e.g. 0.065 → 0.096 at 0.85) simply because there's more prior
data for a later prompt to match against; precision is lower (0.429 → 0.338
at 0.85) because folding in train data also means more rows total, and more
total rows means more chances for a look-alike match that isn't a real
duplicate.


## Sample hit review (20 real hits at threshold 0.85)

Spot-checked manually; a few representative examples:

| New prompt | Matched prompt | Similarity | Reused response fits? |
|---|---|---:|---|
| "Hello, how are you?" | "hello how are you?" | 0.972 | Yes — trivial punctuation/casing difference |
| "Who won the world cup?" | "Who won the World Cup" | 0.909 | Yes — same question |
| "could you please write code for text cleaning to remove html..." | same text, exact | 1.000 | Yes — exact duplicate |

Full 20-item sample is in `data/processed/semantic_cache/semantic_cache_evaluation.json`'s
`hit_sample_at_report_threshold`. All 20 reviewed hits were reasonable matches at this threshold.

## Latency

Real, end-to-end `lookup()` calls (embedding + search) at threshold 0.85,
over all 9,965 combined rows:

| Mean | Median | p95 | Max |
|---:|---:|---:|---:|
| 5.19ms | 4.11ms | 12.11ms | 18.92ms |

Measured on CPU, one prompt at a time.


## Paraphrase robustness

The duplicate-ground-truth check above only tests near-identical text (same
wording, different case/whitespace/Unicode) — it never tests a prompt that
was genuinely reworded. To measure that, 100 safe test prompts were sampled
(seed 42) and each was paraphrased once by the Gemini API
(`gemini-3.5-flash-lite`), producing a real reworded version with different
vocabulary but the same intent (`scripts/generate_paraphrases.py`). The
cache was then built from the 100 originals and each paraphrase was looked
up against it (`scripts/evaluate_semantic_cache_paraphrases.py`).

| Threshold | Hit rate | Correct-match rate (of hits) |
|---:|---:|---:|
| 0.70 | 0.900 | 0.989 |
| 0.75 | 0.820 | 1.000 |
| 0.80 | 0.730 | 1.000 |
| 0.85 | 0.570 | 1.000 |
| 0.90 | 0.380 | 1.000 |
| 0.95 | 0.170 | 1.000 |

At the project's default threshold (0.85), the cache reuses a cached
response for only **57%** of genuine paraphrases — far below the near-1.0
recall the duplicate-ground-truth check reports, because that check's
"duplicates" are near-identical text, not real rewording. When the cache
does hit, it is never wrong about which original it matched (correct-match
rate is 1.0 at every threshold except 0.70, where one paraphrase matched a
different row's original instead of its own).

The lowest-similarity pairs show where the embedding model and a real
paraphrase diverge most, e.g.:

| Original | Paraphrase | Cosine similarity |
|---|---|---:|
| "tell me more about yourself" | "Could you share some more details about your background" | 0.532 |
| "cook me up an interesting love story where the girl dies" | "Write a captivating romance story for me where the female protagonist passes away at the end." | 0.612 |
| "How does neighborliness affect health outcomes?" | "In what ways does living in a supportive community influence overall health?" | 0.616 |

These are clearly the same question, but short/informal prompts reworded
into longer, more formal phrasing push the embedding similarity well below
0.85. Lowering the threshold recovers more hits (0.80 → 73%, 0.70 → 90%)
without losing correct-match accuracy, but moves closer to the
false-positive zone the duplicate-ground-truth check already flagged below
0.85 (see **Limitations**) — there's a real precision/recall tradeoff here,
not a free improvement.

## Warm-start: pre-seeding the cache with training data

This section asks a narrower question than the unified Results above: not
"what's the hit rate over everything combined," but specifically "does
pre-loading the training data *before* test traffic starts give test
queries a better hit rate than if the cache had started from nothing?" So
unlike the Results section (one combined stream), this compares two
separate runs: a **test-only cold start** (cache starts empty, only sees
the 4,883 test prompts) against a **warm start** (cache pre-loaded with
all 5,082 training pairs first, then the same 4,883 test prompts stream
in). The cold-start number here is its own measurement, computed the same
way as the unified streaming simulation but restricted to test rows only —
it is not the combined number from the Results section above.

`scripts/recover_responses.py` was extended to also recover response text
for the training split (`data/processed/semantic_cache/train_with_responses.csv`,
5,082 rows, all matched exactly against the raw Toxic Chat train split).
`scripts/evaluate_semantic_cache_warm_start.py` pre-stores all 5,082 training
pairs in the cache before any test row arrives, then streams the 4,883 test
rows in order against that pre-seeded cache — compared directly to a
test-only cold-start run (same test stream, empty starting cache).

| Threshold | Cold-start hit rate | Warm-start hit rate | Increase |
|---:|---:|---:|---:|
| 0.70 | 0.120 | 0.219 | +0.099 |
| 0.75 | 0.097 | 0.177 | +0.080 |
| 0.80 | 0.078 | 0.148 | +0.069 |
| 0.85 | 0.065 | 0.123 | +0.058 |
| 0.90 | 0.053 | 0.099 | +0.046 |
| 0.95 | 0.041 | 0.071 | +0.030 |

At the project's default threshold (0.85), warm-starting **nearly doubles**
the hit rate (6.5% → 12.3%). This holds at every threshold tested: roughly
a 2x relative increase, growing slightly at looser thresholds in absolute
terms. This makes sense — train and test are two independently-sampled
slices of the same combined dataset (`docs/split-and-vectorize.md`), drawn
from the same pool of real chatbot conversations, so a test-time question
is noticeably more likely to already have a semantic match sitting in a
5,082-row seeded cache than in a test-only cache that has to build itself
up from scratch as it goes.

This is still a **simulated** result in the same sense as the rest of this
document: no claim is made that live production traffic would repeat this
exact ratio, since the train/test split's prompts both come from the same
historical snapshot rather than a real deployment's actual query
distribution over time.

## Limitations

- **The duplicate-ground-truth "recall" doesn't test real paraphrase
  robustness.** The 258 ground-truth groups are rows whose text is identical
  after case/whitespace/Unicode normalization — near-identical strings, not
  genuinely reworded paraphrases with different vocabulary. See
  **Paraphrase robustness** above for a test against 100 real,
  AI-generated paraphrases: hit rate at threshold 0.85 is only 57%.
- **Precision tops out at 0.537**, even at the strictest threshold tested
  (0.95). Most "hits" below that threshold are not real duplicates by the
  ground-truth definition 
- **No threshold is declared "the" answer here.** 0.85 is used as a
  consistent default for the sample review and latency measurement, not a
  claimed optimum
- **Streaming simulation replays a fixed historical order**, not a real
  live query stream — there is no actual query log for this project yet.
- **24 of the 4,883 test rows' "true" response is ambiguous** (a known gap
  in the test-side join specifically; the train-side join hasn't been
  checked for the same issue).
- **`IndexFlatIP` is an exact, unoptimized search** — fine at this scale
  (9,965 combined rows), would need a different FAISS index type (e.g. HNSW)
  at production scale.

## Output files

`data/processed/semantic_cache/` (gitignored):

| File | Contents |
|---|---|
| `test_with_responses.csv` | Recovered `user_input`, `label`, `model_output` for the test split |
| `train_with_responses.csv` | Recovered `user_input`, `label`, `model_output` for the training split |
| `semantic_cache_evaluation.json` | Full report: model name, report threshold, streaming simulation and duplicate-validation results per threshold, latency stats, the 20-item hit sample |
| `paraphrases.json` | 100 sampled original prompts and their Gemini-generated paraphrase |
| `paraphrase_evaluation.json` | Hit rate and correct-match rate per threshold on the paraphrases, plus example pairs |
| `warm_start_evaluation.json` | Cold-start vs. train-seeded warm-start hit rate per threshold |
