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

`data/test_clean.csv` only has `user_input`/`label` — task 4's cleanup
dropped `model_output`, since the classifier never needed it. The cache
needs real response text to store and to let a reviewer judge whether a
cached answer actually fits, so
`scripts/recover_test_responses.py` rejoins `test_clean.csv`'s prompts to
the raw `lmsys_toxicchat.zip` test split's `model_output`, by exact text
match (no row needed prefix-matching — the longest cleaned row is 1,536
characters, under the 2,000-character cap, so cleanup never actually
truncated anything in this split).


## How hit rate is measured, and why two methods

The test set has few exact repeats, so a single "X% hit rate" number has
nothing to validate it against on its own. Two measurements instead, swept
across six thresholds (0.70-0.95):

**1. Streaming simulation** — replay the 4,883 test prompts in their
original row order, as if they were a live query stream. Each prompt checks
the cache first (a hit reuses the response, nothing new is stored); a miss
stores it for later prompts to potentially match

**2. Duplicate-ground-truth validation** — using the test set's own real
near-duplicate groups, check whether each prompt's single nearest *other*
prompt is correctly judged a match at each threshold

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
value)

**Reading this**: as the threshold tightens, hit rate falls (fewer things
count as close enough) while precision rises (fewer wrong matches slip
through)


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

Measured on CPU, one prompt at a time.


## Limitations

- **The duplicate-ground-truth "recall" doesn't test real paraphrase
  robustness.** The 115 ground-truth groups are rows whose text is identical
  after case/whitespace/Unicode normalization — near-identical strings, not
  genuinely reworded paraphrases with different vocabulary.
- **Precision tops out at 0.644**, even at the strictest threshold tested
  (0.95). Most "hits" below that threshold are not real duplicates by the
  ground-truth definition 
- **No threshold is declared "the" answer here.** 0.85 is used as a
  consistent default for the sample review and latency measurement, not a
  claimed optimum
- **Streaming simulation replays a fixed historical order**, not a real
  live query stream — there is no actual query log for this project yet.
- **24 of 4,883 rows' "true" response is ambiguous**.
- **`IndexFlatIP` is an exact, unoptimized search** — fine at 4,883 rows,
  would need a different FAISS index type (e.g. HNSW) at production scale.

## Output files

`data/processed/semantic_cache/` (gitignored):

| File | Contents |
|---|---|
| `test_with_responses.csv` | Recovered `user_input`, `label`, `model_output` |
| `semantic_cache_evaluation.json` | Full report: model name, report threshold, streaming simulation and duplicate-validation results per threshold, latency stats, the 20-item hit sample |
