# Semantic cache

Builds an embedding-based cache: a new prompt can reuse an earlier response
if it's close enough in *meaning*, not just identical text, and measures how
often that actually helps on the cleaned test set.

## Run it

```bash
python -m pip install -r requirements.txt
python scripts/recover_test_responses.py
python scripts/evaluate_semantic_cache.py
python -m unittest discover -s tests -v
```

## Why an embedding model, not the TF-IDF already in this repo

TF-IDF (`split_and_vectorize.py`) measures literal word overlap. Two
sentences phrased differently but meaning the same thing can look almost
unrelated to it, and two unrelated sentences sharing common words ("how do
I...") can look *more* similar than a real paraphrase. Measured directly:

| Pair | TF-IDF cosine | Embedding cosine |
|---|---:|---:|
| "How do I bake a chocolate cake?" / "What is the recipe for making a chocolate cake?" (paraphrase) | 0.208 | **0.852** |
| "How do I bake a chocolate cake?" / "How do I fix my broken laptop?" (unrelated) | 0.247 | **0.088** |

TF-IDF rates the unrelated pair *higher* than the real paraphrase. A cache
meant to catch paraphrases needs the embedding's property, not TF-IDF's.

## Model and vector store

- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` — 384
  dimensions, ~80MB, CPU-fast, the standard lightweight choice for this kind
  of task. This is a materially heavier dependency than anything else in
  this repo so far (every prior script used only `scikit-learn`/`numpy`/
  `scipy`/`tiktoken`); it is added here because the task explicitly calls
  for an embedding, unlike the classifier work, which stayed on TF-IDF.
- **Vector store**: FAISS (`faiss-cpu`), `IndexFlatIP` over L2-normalized
  vectors (inner product of unit vectors = cosine similarity). Chosen over
  ChromaDB for consistency with this repo's existing minimalist style (a
  library, not an embedded database/server); prompt/response text is kept
  as a plain parallel Python list alongside the index, since FAISS itself
  only stores vectors.

## The cache API (`scripts/semantic_cache.py`)

```python
cache = SemanticCache(threshold=0.85)        # the threshold is fixed per cache instance
cache.store(prompt, response)
result = cache.lookup(prompt)                # always uses this cache's threshold
result.hit             # bool
result.matched_prompt  # the stored prompt that matched, if hit
result.response         # its response, if hit
result.similarity       # cosine similarity of the best match (or None if cache empty)
result.latency_seconds  # wall-clock time of this call
```

## Recovering response text

`data/test_clean.csv` only has `user_input`/`label` — task 4's cleanup
dropped `model_output`, since the classifier never needed it. The cache
needs real response text to store and to let a reviewer judge whether a
cached answer actually fits (see below), so
`scripts/recover_test_responses.py` rejoins `test_clean.csv`'s prompts to
the raw `lmsys_toxicchat.zip` test split's `model_output`, by exact text
match (no row needed prefix-matching — the longest cleaned row is 1,536
characters, under the 2,000-character cap, so cleanup never actually
truncated anything in this split).

**Caveat**: 206 of 4,883 rows have prompt text that recurs more than once in
the raw file; for 107 distinct such texts the response was the same every
time, but for 24 of them the model gave genuinely different responses on
different occurrences (`conv_id`, which could have disambiguated this, was
dropped during cleanup). The script deterministically takes the first
occurrence in the raw file's original order. For those 24 prompts, an
equally valid different response could have been joined instead — a known,
documented limitation, not a correctness guarantee.

## How hit rate is measured, and why two methods

The test set has few exact repeats, so a single "X% hit rate" number has
nothing to validate it against on its own. Two measurements instead, swept
across six thresholds (0.70-0.95):

**1. Streaming simulation** — replay the 4,883 test prompts in their
original row order, as if they were a live query stream. Each prompt checks
the cache first (a hit reuses the response, nothing new is stored); a miss
stores it for later prompts to potentially match. This is the simplest
available proxy for live behavior, since there's no real query log.

**2. Duplicate-ground-truth validation** — using the test set's own real
near-duplicate groups (the same normalized-text method `split_and_vectorize.py`
uses: Unicode NFKC + casefold + whitespace collapse — 115 such groups exist
with more than one row), check whether each prompt's single nearest *other*
prompt is correctly judged a match at each threshold. This is what actually
justifies a threshold choice, rather than reporting a bare hit rate.

## Results

| Threshold | Hit rate | Precision | Recall |
|---:|---:|---:|---:|
| 0.70 | 0.120 | 0.243 | 1.000 |
| 0.75 | 0.097 | 0.301 | 1.000 |
| 0.80 | 0.078 | 0.358 | 1.000 |
| **0.85** | **0.065** | **0.429** | **1.000** |
| 0.90 | 0.053 | 0.512 | 1.000 |
| 0.95 | 0.041 | 0.644 | 1.000 |

(Bold row = `REPORT_THRESHOLD`, the default used for the saved hit sample
and the latency measurement below — a starting point, not a claimed best
value; see Limitations.)

**Reading this**: as the threshold tightens, hit rate falls (fewer things
count as close enough) while precision rises (fewer wrong matches slip
through) — the expected trade-off. Recall is 239/239 — **every** real
near-duplicate pair gets correctly matched, at every threshold tested, down
to 0.70.

**This 1.000 recall is not as strong a result as it looks** — see
Limitations. It means the model never misses one of these 115 groups, but
it does not mean it's good at catching real paraphrases.

## Sample hit review (20 real hits at threshold 0.85)

Spot-checked manually; a few representative examples:

| New prompt | Matched prompt | Similarity | Reused response fits? |
|---|---|---:|---|
| "what is the weather today?" | "whats the weather like today" | 0.852 | Yes — both ask the same thing; the cached "I don't have real-time weather access" reply is correct for either |
| "Can you code in python?" | "can you write python code" | 0.948 | Yes — same request, same valid answer |
| "what is your name?" | "what is your name" (no "?") | 0.870 | Yes — trivial punctuation difference |

Full 20-item sample is in `data/processed/semantic_cache/semantic_cache_evaluation.json`'s
`hit_sample_at_report_threshold`. All 20 reviewed hits were reasonable matches at this threshold.

## Latency

Real, end-to-end `lookup()` calls (embedding + search) at threshold 0.85,
over all 4,883 rows:

| Mean | Median | p95 | Max |
|---:|---:|---:|---:|
| 5.64ms | 4.52ms | 12.5ms | 19.8ms |

Measured on CPU, one prompt at a time (no batching) — a real caller's cost.
The other five thresholds reuse precomputed embeddings for speed (see the
script's module docstring); only this one run re-embeds through the normal
`lookup()` path specifically so this latency number is honest.

## Limitations

- **The duplicate-ground-truth "recall" doesn't test real paraphrase
  robustness.** The 115 ground-truth groups are rows whose text is identical
  after case/whitespace/Unicode normalization — near-identical strings, not
  genuinely reworded paraphrases with different vocabulary. Catching these
  is a low bar; a model (or even careful exact-match logic) could likely
  score well on this specific test without being good at real paraphrases
  like the "bake a cake" / "recipe for cake" example above. The 1.000 recall
  figure should not be read as "catches all paraphrases."
- **Precision tops out at 0.644**, even at the strictest threshold tested
  (0.95). Most "hits" below that threshold are not real duplicates by the
  ground-truth definition — a real deployment would want to understand this
  trade-off, not just pick the row with the best-looking number.
- **No threshold is declared "the" answer here.** 0.85 is used as a
  consistent default for the sample review and latency measurement, not a
  claimed optimum — picking the operating point is a product decision
  (how many false cache hits is acceptable vs. how much reuse is wanted),
  left open deliberately, consistent with how `official_baseline` was
  handled in issue #7.
- **Streaming simulation replays a fixed historical order**, not a real
  live query stream — there is no actual query log for this project yet.
- **24 of 4,883 rows' "true" response is ambiguous** (see Recovering
  response text above).
- **`IndexFlatIP` is an exact, unoptimized search** — fine at 4,883 rows,
  would need a different FAISS index type (e.g. HNSW) at production scale.

## Output files

`data/processed/semantic_cache/` (gitignored, like other `data/processed/`
outputs):

| File | Contents |
|---|---|
| `test_with_responses.csv` | Recovered `user_input`, `label`, `model_output` |
| `semantic_cache_evaluation.json` | Full report: model name, report threshold, streaming simulation and duplicate-validation results per threshold, latency stats, the 20-item hit sample |

Use `--responses`/`--output-dir` to change where the evaluation script
reads from or writes to. The recovery script's paths are fixed constants
at the top of `scripts/recover_test_responses.py` (no CLI flags) — edit
those directly if a different run is needed.
