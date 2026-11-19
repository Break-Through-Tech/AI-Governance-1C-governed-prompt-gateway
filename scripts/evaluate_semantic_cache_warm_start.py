"""Test whether pre-populating ("warm-starting") the semantic cache with
the training set's prompt/response pairs improves hit rate on test-time
traffic, compared to the cold-start cache measured in
scripts/evaluate_semantic_cache.py (built purely from the test stream
itself, with nothing stored beforehand).

The classifier's train/test split has no bearing on the cache -- the cache
does not train on anything and never reads train_clean.csv for its own
measurements (see docs/semantic-cache.md). The real question this answers
is practical, not about leakage: if a deployed cache were seeded with
already-known Q&A pairs from training time before any live traffic
arrives, does that actually raise the hit rate once test queries (standing
in for live queries) start coming in?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_semantic_cache import THRESHOLDS, embed_all, streaming_simulation
from scripts.semantic_cache import MODEL_NAME, SemanticCache

TEST_RESPONSES_PATH = ROOT / 'data/processed/semantic_cache/test_with_responses.csv'
TRAIN_RESPONSES_PATH = ROOT / 'data/processed/semantic_cache/train_with_responses.csv'
OUTPUT_PATH = ROOT / 'data/processed/semantic_cache/warm_start_evaluation.json'


def load_split(path):
    data = pd.read_csv(path, encoding='utf-8-sig')
    if list(data.columns[:3]) != ['user_input', 'label', 'model_output']:
        raise ValueError(f'Expected user_input,label,model_output,... columns in {path.name}')
    return data['user_input'].tolist(), data['model_output'].tolist()


def warm_start_hit_rate(train_prompts, train_responses, train_embeddings,
                         test_prompts, test_responses, test_embeddings, model, threshold):
    """Pre-seed the cache with every training prompt/response pair, then
    stream the test prompts against that already-populated cache."""
    cache = SemanticCache(threshold=threshold, model=model)
    for i, (prompt, response) in enumerate(zip(train_prompts, train_responses)):
        cache.store(prompt, response, embedding=train_embeddings[i:i + 1])

    hits = 0
    for i, (prompt, response) in enumerate(zip(test_prompts, test_responses)):
        result = cache.lookup(prompt, embedding=test_embeddings[i:i + 1])
        if result.hit:
            hits += 1
        else:
            cache.store(prompt, response, embedding=test_embeddings[i:i + 1])
    return hits / len(test_prompts) if test_prompts else None


def evaluate(train_path=TRAIN_RESPONSES_PATH, test_path=TEST_RESPONSES_PATH, output_path=OUTPUT_PATH):
    train_prompts, train_responses = load_split(train_path)
    test_prompts, test_responses = load_split(test_path)

    model = SentenceTransformer(MODEL_NAME)
    train_embeddings = embed_all(model, train_prompts)
    test_embeddings = embed_all(model, test_prompts)

    results = {}
    for threshold in THRESHOLDS:
        cold_result, _ = streaming_simulation(
            test_prompts, test_responses, test_embeddings, model, threshold, measure_real_latency=False)
        warm_hit_rate = warm_start_hit_rate(
            train_prompts, train_responses, train_embeddings,
            test_prompts, test_responses, test_embeddings, model, threshold)
        results[threshold] = {
            'threshold': threshold,
            'cold_start_hit_rate': cold_result['hit_rate'],
            'warm_start_hit_rate': warm_hit_rate,
            'hit_rate_increase': warm_hit_rate - cold_result['hit_rate'],
        }

    report = {
        'model_name': MODEL_NAME,
        'train_rows_used_to_seed': len(train_prompts),
        'test_rows_evaluated': len(test_prompts),
        'results_by_threshold': {str(t): r for t, r in results.items()},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    report = evaluate()
    print(f"Model: {report['model_name']}  |  train rows seeded: {report['train_rows_used_to_seed']}  "
          f"|  test rows evaluated: {report['test_rows_evaluated']}\n")
    print(f"{'threshold':>9}  {'cold_hit_rate':>13}  {'warm_hit_rate':>13}  {'increase':>9}")
    for t in THRESHOLDS:
        r = report['results_by_threshold'][str(t)]
        print(f"{t:>9.2f}  {r['cold_start_hit_rate']:>13.3f}  {r['warm_start_hit_rate']:>13.3f}  "
              f"{r['hit_rate_increase']:>+9.3f}")
    print(f"\nSaved full report to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
