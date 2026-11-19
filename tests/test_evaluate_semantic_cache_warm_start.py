"""Check scripts/evaluate_semantic_cache_warm_start.py's warm_start_hit_rate:
pre-seeding the cache with a training pair can turn a test-time miss (cold
start, nothing stored yet) into a hit, using a fake embedding model with
fixed, hand-picked vectors so similarity is exact and predictable."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_semantic_cache_warm_start import warm_start_hit_rate
from scripts.evaluate_semantic_cache import streaming_simulation


class FakeModel:
    VECTORS = {
        'original A': [1.0, 0.0],
        'paraphrase of A': [0.99, 0.141],  # cos sim with "original A" ~ 0.99
        'unrelated query': [0.0, 1.0],     # cos sim with "original A" = 0.0
    }

    def encode(self, texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False):
        vectors = [self.VECTORS[t] for t in texts]
        arr = np.asarray(vectors, dtype='float32')
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / norms

    def get_sentence_embedding_dimension(self):
        return 2


class WarmStartTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.train_prompts = ['original A']
        self.train_responses = ['resp A']
        self.train_embeddings = self.model.encode(self.train_prompts)
        self.test_prompts = ['paraphrase of A', 'unrelated query']
        self.test_responses = ['x', 'y']
        self.test_embeddings = self.model.encode(self.test_prompts)

    def test_warm_start_hits_where_cold_start_would_miss(self):
        threshold = 0.95
        cold_result, _ = streaming_simulation(
            self.test_prompts, self.test_responses, self.test_embeddings, self.model, threshold)
        self.assertEqual(cold_result['hit_rate'], 0.0)  # nothing stored yet when "paraphrase of A" arrives

        warm_hit_rate = warm_start_hit_rate(
            self.train_prompts, self.train_responses, self.train_embeddings,
            self.test_prompts, self.test_responses, self.test_embeddings, self.model, threshold)
        self.assertEqual(warm_hit_rate, 0.5)  # "paraphrase of A" now matches the pre-seeded "original A"

    def test_low_threshold_warm_start_cannot_exceed_perfect_hit_rate(self):
        hit_rate = warm_start_hit_rate(
            self.train_prompts, self.train_responses, self.train_embeddings,
            self.test_prompts, self.test_responses, self.test_embeddings, self.model, threshold=0.0)
        self.assertEqual(hit_rate, 1.0)


if __name__ == '__main__':
    unittest.main()
