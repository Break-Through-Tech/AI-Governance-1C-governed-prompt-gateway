"""Routing rules, cost arithmetic, and test-set evaluation checks for issue #21."""
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_routing import evaluate, load_test_set, summarize
from scripts.route_model import (ELEVATED_RISK, HIGH_RISK, LONG_TOKENS, SHORT_TOKENS, TIERS,
                                 ModelRouter, recommend)


class RoutingRuleTests(unittest.TestCase):
    def tier(self, risk, tokens):
        return recommend('prompt', risk, tokens)['tier']

    def test_short_low_risk_goes_small(self):
        result = recommend('What is your refund policy?', 0.0, SHORT_TOKENS)
        self.assertEqual((result['tier'], result['model'], result['reason']),
                         ('small', 'gpt-6-luna', 'short_low_risk'))

    def test_length_boundaries(self):
        self.assertEqual(self.tier(0.0, SHORT_TOKENS), 'small')
        self.assertEqual(self.tier(0.0, SHORT_TOKENS + 1), 'medium')
        self.assertEqual(self.tier(0.0, LONG_TOKENS), 'medium')
        self.assertEqual(self.tier(0.0, LONG_TOKENS + 1), 'large')

    def test_risk_boundaries_override_length(self):
        just_below = math.nextafter(ELEVATED_RISK, 0)
        self.assertEqual(self.tier(just_below, 5), 'small')
        self.assertEqual(self.tier(ELEVATED_RISK, 5), 'medium')
        self.assertEqual(self.tier(HIGH_RISK, 5), 'large')
        self.assertEqual(recommend('p', HIGH_RISK, 5)['reason'], 'high_risk')
        self.assertEqual(recommend('p', ELEVATED_RISK, 5)['reason'], 'elevated_risk')

    def test_long_prompt_reason(self):
        self.assertEqual(recommend('p', 0.0, LONG_TOKENS + 1)['reason'], 'long_prompt')

    def test_unknown_token_count_routes_large_without_cost(self):
        result = recommend('p', 0.0, None)
        self.assertEqual(result['tier'], 'large')
        self.assertEqual(result['reason'], 'token_count_unknown')
        self.assertIsNone(result['estimated_cost_usd'])
        self.assertIsNone(result['estimated_cost_usd_exact'])

    def test_input_cost_arithmetic(self):
        self.assertEqual(recommend('p', 0.0, 10)['estimated_cost_usd_exact'], '0.000001000000')
        self.assertEqual(recommend('p', 0.0, 100)['estimated_cost_usd_exact'], '0.000100000000')
        self.assertEqual(recommend('p', 0.0, 1000)['estimated_cost_usd_exact'], '0.008000000000')
        self.assertEqual(recommend('p', 0.0, 0)['estimated_cost_usd'], 0.0)
        self.assertAlmostEqual(recommend('p', 0.0, 1000)['estimated_cost_usd'], 0.008)
        self.assertEqual(recommend('p', 0.0, 1)['cost_scope'], 'input_tokens_only')

    def test_tiers_increase_in_price(self):
        prices = [TIERS[name].input_usd_per_million for name in ('small', 'medium', 'large')]
        self.assertEqual(prices, sorted(prices))
        self.assertEqual(len(set(prices)), 3)

    def test_rejects_invalid_inputs(self):
        for risk in (True, -0.01, 1.01, float('nan'), float('inf'), '0.5', None):
            with self.subTest(risk=risk), self.assertRaises(ValueError):
                recommend('p', risk, 10)
        for tokens in (-1, 1.5, True, '10'):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                recommend('p', 0.0, tokens)
        with self.assertRaises(ValueError):
            recommend(None, 0.0, 10)

    def test_custom_thresholds_and_validation(self):
        router = ModelRouter(short_tokens=5, long_tokens=10, elevated_risk=0.3, high_risk=0.6)
        self.assertEqual(router.recommend('p', 0.0, 6)['tier'], 'medium')
        self.assertEqual(router.recommend('p', 0.5, 1)['tier'], 'medium')
        for kwargs in ({'short_tokens': 10, 'long_tokens': 10},
                       {'elevated_risk': 0.5, 'high_risk': 0.5},
                       {'high_risk': 1.5},
                       {'tiers': {'small': TIERS['small']}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ModelRouter(**kwargs)

    def test_matches_gateway_router_interface(self):
        """PR #32's pipeline keeps tier and estimated_cost_usd from recommend()."""
        result = ModelRouter().recommend('prompt', 0.2, 40)
        self.assertIsInstance(result['tier'], str)
        self.assertIsInstance(result['estimated_cost_usd'], float)


def write_fixture(directory: Path, prompts, labels, *, tamper_hash=False):
    source = directory / 'test.csv'
    with source.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['user_input', 'label'])
        writer.writerows(zip(prompts, labels))
    baseline = directory / 'baseline'
    baseline.mkdir()
    (baseline / 'token_baseline_summary.json').write_text(json.dumps(
        {'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}))
    with (baseline / 'token_baseline.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt_id', 'test_row', 'prompt_sha256', 'original_tokens'])
        for i, prompt in enumerate(prompts):
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            writer.writerow([f'test-{i:06d}', i, '0' * 64 if tamper_hash else digest, 5])
    return source, baseline


class EvaluationTests(unittest.TestCase):
    def test_blocked_rows_are_logged_but_never_routed(self):
        prompts = ['What is your refund policy?', 'You are useless', 'What is your refund policy?']
        classifications = [
            {'label': 'safe', 'risk_score': 0.0, 'score_type': 'test'},
            {'label': 'toxic', 'risk_score': 0.99, 'score_type': 'test'},
            {'label': 'safe', 'risk_score': HIGH_RISK, 'score_type': 'test'},
        ]
        with tempfile.TemporaryDirectory() as d:
            source, baseline = write_fixture(Path(d), prompts, ['safe', 'toxic', 'safe'])
            rows, sha = load_test_set(source, baseline)
            log = Path(d) / 'log.jsonl'
            records = evaluate(rows, classifications, log)
            summary = summarize(records, sha)
            self.assertEqual(len(log.read_text().splitlines()), 3)

        self.assertEqual([r['action'] for r in records], ['pass', 'block', 'pass'])
        self.assertEqual([r['tier'] for r in records], ['small', '', 'large'])
        self.assertEqual(records[1]['estimated_cost_usd'], '')
        self.assertEqual(summary['governance'], {'passed': 2, 'blocked': 1})
        self.assertEqual(summary['source_label_by_outcome']['toxic'], {'blocked': 1})
        comparison = summary['cost_comparison']
        tokens = records[0]['routed_tokens']
        expected_routed = TIERS['small'].input_cost(tokens) + TIERS['large'].input_cost(tokens)
        self.assertEqual(Decimal(comparison['routed_cost_usd']), expected_routed)
        self.assertEqual(Decimal(comparison['always_largest_cost_usd']),
                         2 * TIERS['large'].input_cost(tokens))
        self.assertGreater(comparison['savings_percent'], 0)

    def test_rejects_misaligned_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            source, baseline = write_fixture(Path(d), ['hello there'], ['safe'], tamper_hash=True)
            with self.assertRaisesRegex(ValueError, 'does not match'):
                load_test_set(source, baseline)

    def test_rejects_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            source, baseline = write_fixture(Path(d), ['hello there'], ['safe'])
            source.write_text('user_input,label\nchanged,safe\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_test_set(source, baseline)


class SavedResultsTests(unittest.TestCase):
    """The committed results must agree with each other and the #8 baseline."""
    ROOT = Path(__file__).resolve().parents[1]

    def test_committed_summary_matches_rows(self):
        results = self.ROOT / 'data/routing/routing_results.csv'
        if not results.exists():
            self.skipTest('routing results not generated')
        with results.open(encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        summary = json.loads((self.ROOT / 'data/routing/routing_summary.json').read_text())
        baseline = json.loads((self.ROOT / 'data/token_baseline/token_baseline_summary.json').read_text())
        self.assertEqual(summary['input_sha256'], baseline['input_sha256'])
        self.assertEqual(len(rows), baseline['prompt_count'])
        passed = [r for r in rows if r['action'] == 'pass']
        self.assertEqual(summary['governance']['passed'], len(passed))
        self.assertTrue(all(r['tier'] == '' for r in rows if r['action'] == 'block'))
        self.assertEqual(sum(Decimal(r['estimated_cost_usd']) for r in passed),
                         Decimal(summary['cost_comparison']['routed_cost_usd']))
        router = ModelRouter(**{k: v for k, v in summary['thresholds'].items()})
        for r in passed:
            route = router.recommend('p', float(r['risk_score']), int(r['routed_tokens']))
            self.assertEqual((route['tier'], route['reason']), (r['tier'], r['reason']))


if __name__ == '__main__':
    unittest.main()
