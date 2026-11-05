"""Check SemanticCache's lookup/store logic and the evaluation script's pure
functions. Most tests use a tiny fake embedder (no model download, no GPU,
runs in milliseconds); a few, marked slow, load the real model as a smoke
test."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.semantic_cache import CacheResult, SemanticCache
from scripts.evaluate_semantic_cache import (
    build_duplicate_groups, classify_duplicate_predictions, latency_stats, normalize,
)


class FakeEmbedder:
    """Deterministic, hand-built embedder: maps known phrases to known unit
    vectors so similarity between any two known phrases is exactly known,
    without loading a real model."""

    DIM = 4
    VECTORS = {
        'cake': np.array([1.0, 0.0, 0.0, 0.0], dtype='float32'),
        'cake paraphrase': np.array([0.9, 0.436, 0.0, 0.0], dtype='float32'),  # cos ~0.9 with 'cake'
        'laptop': np.array([0.0, 0.0, 1.0, 0.0], dtype='float32'),  # cos 0.0 with 'cake'
    }

    def get_sentence_embedding_dimension(self):
        return self.DIM

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64):
        vectors = np.stack([self.VECTORS[t] for t in texts])
        if normalize_embeddings:
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors


class SemanticCacheTests(unittest.TestCase):
    def setUp(self):
        self.cache = SemanticCache(threshold=0.85, model=FakeEmbedder())

    def test_lookup_on_empty_cache_is_a_miss(self):
        result = self.cache.lookup('cake')
        self.assertFalse(result.hit)
        self.assertIsNone(result.similarity)
        self.assertGreaterEqual(result.latency_seconds, 0.0)

    def test_store_then_lookup_identical_text_is_a_hit(self):
        self.cache.store('cake', 'Preheat oven to 350F')
        result = self.cache.lookup('cake')
        self.assertTrue(result.hit)
        self.assertEqual(result.matched_prompt, 'cake')
        self.assertEqual(result.response, 'Preheat oven to 350F')
        self.assertAlmostEqual(result.similarity, 1.0, places=5)

    def test_similar_but_not_identical_text_above_threshold_is_a_hit(self):
        self.cache.store('cake', 'Preheat oven to 350F')
        result = self.cache.lookup('cake paraphrase')
        self.assertTrue(result.hit)
        self.assertGreaterEqual(result.similarity, 0.85)

    def test_unrelated_text_is_a_miss(self):
        self.cache.store('cake', 'Preheat oven to 350F')
        result = self.cache.lookup('laptop')
        self.assertFalse(result.hit)
        self.assertAlmostEqual(result.similarity, 0.0, places=5)

    def test_store_does_not_require_a_prior_lookup(self):
        self.cache.store('cake', 'response one')
        self.cache.store('laptop', 'response two')
        self.assertEqual(len(self.cache), 2)

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            SemanticCache(threshold=1.5, model=FakeEmbedder())
        with self.assertRaises(ValueError):
            SemanticCache(threshold=-0.1, model=FakeEmbedder())

    def test_cache_result_to_dict_is_json_safe(self):
        self.cache.store('cake', 'Preheat oven to 350F')
        result = self.cache.lookup('cake')
        as_dict = result.to_dict()
        self.assertEqual(as_dict['matched_prompt'], 'cake')
        self.assertIsInstance(as_dict, dict)

    def test_precomputed_embedding_gives_the_same_result_as_default(self):
        embedder = FakeEmbedder()
        vec = embedder.encode(['cake'])
        self.cache.store('cake', 'Preheat oven to 350F', embedding=vec)
        result = self.cache.lookup('cake', embedding=vec)
        self.assertTrue(result.hit)
        self.assertAlmostEqual(result.similarity, 1.0, places=5)


class EvaluationHelperTests(unittest.TestCase):
    def test_normalize_collapses_case_and_whitespace(self):
        self.assertEqual(normalize('  Hello   World  '), 'hello world')
        self.assertEqual(normalize('HELLO WORLD'), normalize('hello   world'))

    def test_build_duplicate_groups_groups_by_normalized_text(self):
        groups = build_duplicate_groups(['Hi there', 'hi there', 'Something else'])
        self.assertEqual(len(groups), 2)
        self.assertEqual(sorted(groups[normalize('Hi there')]), [0, 1])

    def test_classify_duplicate_predictions_counts_match_expectations(self):
        prompts = ['Hi there', 'hi there', 'Something else']
        groups = build_duplicate_groups(prompts)
        # Row 0's nearest-other is row 1 (a real duplicate), similarity 0.9.
        # Row 1's nearest-other is row 0 (a real duplicate), similarity 0.9.
        # Row 2's nearest-other is row 0 (not a real duplicate), similarity 0.5.
        nearest_similarity = np.array([0.9, 0.9, 0.5])
        nearest_index = np.array([1, 0, 0])

        result = classify_duplicate_predictions(prompts, groups, nearest_similarity, nearest_index, threshold=0.8)
        self.assertEqual((result['true_positive'], result['false_positive'],
                          result['true_negative'], result['false_negative']), (2, 0, 1, 0))
        self.assertEqual(result['precision'], 1.0)
        self.assertEqual(result['recall'], 1.0)

        # At a stricter threshold, rows 0/1 no longer predict a match — both
        # become false negatives (a real duplicate existed and was missed).
        strict_result = classify_duplicate_predictions(prompts, groups, nearest_similarity, nearest_index, threshold=0.95)
        self.assertEqual(strict_result['false_negative'], 2)
        self.assertEqual(strict_result['true_positive'], 0)

    def test_latency_stats_shape(self):
        stats = latency_stats([0.001, 0.002, 0.003, 0.004, 0.100])
        self.assertEqual(stats['count'], 5)
        self.assertAlmostEqual(stats['median_seconds'], 0.003)
        self.assertEqual(stats['max_seconds'], 0.100)
        self.assertGreaterEqual(stats['p95_seconds'], stats['median_seconds'])


@unittest.skipUnless(__import__('importlib').util.find_spec('sentence_transformers'), 'sentence-transformers not installed')
class RealModelSmokeTest(unittest.TestCase):
    """Slow (loads a real ~80MB model): confirms the actual configured model
    behaves as the rest of this project assumes — a paraphrase scores much
    higher than an unrelated sentence."""

    def test_real_model_separates_a_paraphrase_from_an_unrelated_sentence(self):
        from scripts.semantic_cache import SemanticCache
        cache = SemanticCache(threshold=0.7)
        cache.store('How do I bake a chocolate cake?', 'Preheat oven to 350F...')
        paraphrase = cache.lookup('What is the recipe for making a chocolate cake?')
        unrelated = cache.lookup('How do I fix my broken laptop?')
        self.assertTrue(paraphrase.hit)
        self.assertFalse(unrelated.hit)
        self.assertGreater(paraphrase.similarity, unrelated.similarity)


if __name__ == '__main__':
    unittest.main()
