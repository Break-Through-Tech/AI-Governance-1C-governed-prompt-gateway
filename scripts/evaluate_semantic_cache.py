"""Evaluate the semantic cache's hit rate and latency on the cleaned,
response-recovered test set (scripts/recover_test_responses.py)"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import random
import statistics
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.semantic_cache import MODEL_NAME, SemanticCache

THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
REPORT_THRESHOLD = 0.85  # used for the saved hit-sample review and headline latency
SAMPLE_SIZE = 20
SEED = 42


def normalize(text: str) -> str:
    """Same normalization split_and_vectorize.py uses for duplicate groups."""
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def build_duplicate_groups(prompts: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, prompt in enumerate(prompts):
        groups[normalize(prompt)].append(i)
    return groups


def embed_all(model: SentenceTransformer, prompts: list[str]) -> np.ndarray:
    return np.asarray(
        model.encode(prompts, batch_size=64, normalize_embeddings=True, show_progress_bar=False),
        dtype='float32',
    )


def streaming_simulation(prompts, responses, embeddings, model, threshold, measure_real_latency=False):
    """Replay the test set once, in order, as described in the module
    docstring. Uses precomputed `embeddings` unless `measure_real_latency`,
    in which case the cache re-embeds each prompt through its normal path so
    the reported latency includes embedding time, not just vector search."""
    cache = SemanticCache(threshold=threshold, model=model)
    hits = 0
    latencies = []
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        row_embedding = None if measure_real_latency else embeddings[i:i + 1]
        result = cache.lookup(prompt, embedding=row_embedding)
        latencies.append(result.latency_seconds)
        if result.hit:
            hits += 1
        else:
            cache.store(prompt, response, embedding=row_embedding)
    return {
        'threshold': threshold,
        'hits': hits,
        'total': len(prompts),
        'hit_rate': hits / len(prompts),
        'final_cache_size': len(cache),
    }, latencies


def nearest_other_for_every_prompt(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One batched top-2 search over all prompts at once: for each row, its
    nearest neighbor is always itself (similarity 1.0); return the SECOND
    nearest's similarity and index for every row. Threshold-independent, so
    computed exactly once regardless of how many thresholds are evaluated."""
    import faiss
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    similarities, indices = index.search(embeddings, 2)
    return similarities[:, 1], indices[:, 1]


def classify_duplicate_predictions(prompts, groups, nearest_similarity, nearest_index, threshold):
    """Cheap per-threshold classification against the precomputed nearest-
    other-prompt result — see duplicate-ground-truth validation in the
    module docstring."""
    group_of = [normalize(p) for p in prompts]
    tp = fp = tn = fn = 0
    for i in range(len(prompts)):
        has_real_duplicate = len(groups[group_of[i]]) > 1
        predicted_duplicate = bool(nearest_similarity[i] >= threshold)
        if predicted_duplicate:
            if group_of[int(nearest_index[i])] == group_of[i]:
                tp += 1
            else:
                fp += 1
        elif has_real_duplicate:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {
        'threshold': threshold, 'true_positive': tp, 'false_positive': fp,
        'true_negative': tn, 'false_negative': fn,
        'precision': precision, 'recall': recall,
    }


def sample_hits_for_review(prompts, responses, embeddings, model, threshold, sample_size, seed):
    cache = SemanticCache(threshold=threshold, model=model)
    hit_records = []
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        row_embedding = embeddings[i:i + 1]
        result = cache.lookup(prompt, embedding=row_embedding)
        if result.hit:
            hit_records.append({
                'new_prompt': prompt, 'matched_prompt': result.matched_prompt,
                'reused_response': result.response, 'similarity': result.similarity,
            })
        else:
            cache.store(prompt, response, embedding=row_embedding)
    rng = random.Random(seed)
    return rng.sample(hit_records, min(sample_size, len(hit_records)))


def latency_stats(latencies: list[float]) -> dict:
    sorted_lat = sorted(latencies)
    p95_index = min(len(sorted_lat) - 1, int(round(0.95 * (len(sorted_lat) - 1))))
    return {
        'count': len(latencies),
        'mean_seconds': statistics.mean(latencies),
        'median_seconds': statistics.median(latencies),
        'p95_seconds': sorted_lat[p95_index],
        'max_seconds': max(latencies),
    }


def evaluate(responses_path: Path, output_dir: Path) -> dict:
    data = pd.read_csv(responses_path, encoding='utf-8-sig')
    if list(data.columns[:3]) != ['user_input', 'label', 'model_output']:
        raise ValueError('Expected user_input,label,model_output,... columns')
    prompts = data['user_input'].tolist()
    responses = data['model_output'].tolist()

    model = SentenceTransformer(MODEL_NAME)
    embeddings = embed_all(model, prompts)
    groups = build_duplicate_groups(prompts)
    nearest_similarity, nearest_index = nearest_other_for_every_prompt(embeddings)

    streaming_results, duplicate_results = {}, {}
    report_latencies = None
    for threshold in THRESHOLDS:
        measure_real_latency = (threshold == REPORT_THRESHOLD)
        streaming_results[threshold], latencies = streaming_simulation(
            prompts, responses, embeddings, model, threshold, measure_real_latency)
        duplicate_results[threshold] = classify_duplicate_predictions(
            prompts, groups, nearest_similarity, nearest_index, threshold)
        if measure_real_latency:
            report_latencies = latencies

    sample = sample_hits_for_review(prompts, responses, embeddings, model, REPORT_THRESHOLD, SAMPLE_SIZE, SEED)

    report = {
        'model_name': MODEL_NAME,
        'report_threshold': REPORT_THRESHOLD,
        'test_rows': len(prompts),
        'normalized_duplicate_groups_with_more_than_one_row':
            sum(1 for idxs in groups.values() if len(idxs) > 1),
        'streaming_simulation': {str(t): r for t, r in streaming_results.items()},
        'duplicate_ground_truth_validation': {str(t): r for t, r in duplicate_results.items()},
        'latency_at_report_threshold': latency_stats(report_latencies),
        'hit_sample_at_report_threshold': sample,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'semantic_cache_evaluation.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--responses', type=Path,
                        default=ROOT / 'data/processed/semantic_cache/test_with_responses.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed/semantic_cache')
    args = parser.parse_args()
    report = evaluate(args.responses, args.output_dir)

    print(f"Model: {report['model_name']}  |  test rows: {report['test_rows']}")
    print(f"Normalized duplicate groups with >1 row: "
          f"{report['normalized_duplicate_groups_with_more_than_one_row']}\n")
    print(f"{'threshold':>9}  {'hit_rate':>9}  {'precision':>9}  {'recall':>9}")
    for t in THRESHOLDS:
        s = report['streaming_simulation'][str(t)]
        d = report['duplicate_ground_truth_validation'][str(t)]
        p = f"{d['precision']:.3f}" if d['precision'] is not None else 'n/a'
        r = f"{d['recall']:.3f}" if d['recall'] is not None else 'n/a'
        print(f"{t:>9.2f}  {s['hit_rate']:>9.3f}  {p:>9}  {r:>9}")

    lat = report['latency_at_report_threshold']
    print(f"\nLookup latency at threshold {REPORT_THRESHOLD} (real, end-to-end incl. embedding): "
          f"mean={lat['mean_seconds']*1000:.2f}ms median={lat['median_seconds']*1000:.2f}ms "
          f"p95={lat['p95_seconds']*1000:.2f}ms")
    print(f"Saved {len(report['hit_sample_at_report_threshold'])}-item hit sample and full "
          f"report to {args.output_dir}")


if __name__ == '__main__':
    main()
