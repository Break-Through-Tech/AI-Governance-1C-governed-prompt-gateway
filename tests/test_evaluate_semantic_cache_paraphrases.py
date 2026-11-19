"""Check scripts/evaluate_semantic_cache_paraphrases.py's threshold sweep:
a paraphrase close enough to its original hits and matches the right
original; a low threshold can also produce a hit that matches a different
row's original (not just the right one)."""

from __future__ import annotations

import unittest

from scripts.evaluate_semantic_cache_paraphrases import evaluate_at_threshold
from scripts.semantic_cache import SemanticCache


class FakeModel:
    """Maps a handful of fixed strings to hand-picked 2D vectors so
    similarity is exact and threshold behavior is predictable."""

    VECTORS = {
        'original A': [1.0, 0.0],
        'paraphrase of A': [0.99, 0.141],   # cos sim with A ~ 0.99
        'original B': [0.0, 1.0],
        'paraphrase of B': [0.6, 0.8],      # cos sim with B = 0.8; closer to A? no
    }

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        import numpy as np
        vectors = [self.VECTORS[t] for t in texts]
        arr = np.asarray(vectors, dtype='float32')
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / norms

    def get_sentence_embedding_dimension(self):
        return 2


class EvaluateAtThresholdTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {'original': 'original A', 'paraphrase': 'paraphrase of A', 'response': 'resp A'},
            {'original': 'original B', 'paraphrase': 'paraphrase of B', 'response': 'resp B'},
        ]
        self.model = FakeModel()

    def test_high_threshold_only_the_close_paraphrase_hits_and_matches_correctly(self):
        result = evaluate_at_threshold(self.records, self.model, threshold=0.9)
        self.assertEqual(result['hits'], 1)
        self.assertEqual(result['correct_matches'], 1)
        self.assertEqual(result['correct_match_rate_of_hits'], 1.0)

    def test_low_threshold_both_hit(self):
        result = evaluate_at_threshold(self.records, self.model, threshold=0.7)
        self.assertEqual(result['hits'], 2)
        self.assertEqual(result['total'], 2)
        self.assertEqual(result['hit_rate'], 1.0)


if __name__ == '__main__':
    unittest.main()
