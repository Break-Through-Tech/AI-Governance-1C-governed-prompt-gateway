"""Governance gates, preservation, and baseline-aligned reporting for issue #19."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.compress_prompts import compress_prompt, digest, evaluate, preservation_score
from scripts.measure_token_baseline import measure


class CompressionTests(unittest.TestCase):
    def test_blocked_never_reaches_compression_or_returns_a_prompt(self):
        with patch('scripts.compress_prompts.count_tokens', side_effect=AssertionError):
            result = compress_prompt('Please explain something', 'block')
        self.assertIsNone(result.compressed_prompt)
        self.assertIsNone(result.reduction_percent)

    def test_missing_or_unknown_action_fails_closed(self):
        for action in ('allow', '', None, 'PASS'):
            with self.subTest(action=action), self.assertRaises(ValueError):
                compress_prompt('Hello', action)

    def test_preserves_constraints_numbers_and_negations(self):
        prompt = 'Could you please list 3 options under $50, excluding flights?'
        result = compress_prompt(prompt, 'pass')
        self.assertEqual(result.compressed_prompt, 'list 3 options under $50, excluding flights?')
        self.assertGreater(result.reduction_percent, 0)
        self.assertEqual(result.preservation_score, 1)
        self.assertEqual(preservation_score('Please explain why not to spend $50',
                                            'explain why to spend $50'), 0)
        self.assertEqual(preservation_score('Please list A before B', 'list B before A'), 0)

    def test_literal_formatting_multilingual_and_nonrequest_prompts_unchanged(self):
        prompts = ['Please explain "Please explain".', 'Please write code to print 3.',
                   'Please write a poem about cats.', 'Please explain this:\n  a  b',
                   'Can you swim?', '你好世界', 'Please do not disclose this.',
                   'Please explain the exact wording.', 'Please', '<|endoftext|>',
                   'Please list the first two words of this prompt.']
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                result = compress_prompt(prompt, 'pass')
                self.assertEqual(result.compressed_prompt, prompt)
                self.assertEqual(result.reduction_percent, 0)

    def test_failed_preservation_falls_back_to_original(self):
        prompt = 'Please explain gravity'
        with patch('scripts.compress_prompts.preservation_score', return_value=0):
            result = compress_prompt(prompt, 'pass')
        self.assertEqual(result.status, 'rejected_preservation')
        self.assertEqual(result.compressed_prompt, prompt)
        self.assertEqual(result.reduction_percent, 0)

    def test_empty_pass_is_rejected(self):
        for prompt in ('', '   ', None):
            with self.assertRaises(ValueError):
                compress_prompt(prompt, 'pass')


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / 'test.csv'
        self.baseline = self.root / 'baseline'
        self.output = self.root / 'output'
        self.prompts = ['Please explain gravity', 'Please explain gravity', 'hello', 'blocked']
        with self.input.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['user_input', 'label'])
            writer.writerows(zip(self.prompts, ['safe', 'safe', 'safe', 'toxic']))
        self.metadata = measure(self.input, self.baseline)
        self.decisions = self.root / 'decisions.json'
        self.payload = {'source': 'unit-test governance stub',
                        'input_sha256': self.metadata['input_sha256'],
                        'decisions': [{'prompt_id': f'test-{i:06d}', 'prompt_sha256': digest(p),
                                       'action': 'block' if i == 0 else 'pass'}
                                      for i, p in enumerate(self.prompts)]}
        self.save_decisions()

    def save_decisions(self):
        self.decisions.write_text(json.dumps(self.payload))

    def run_eval(self):
        return evaluate(self.input, self.baseline, self.output, self.decisions)

    def test_explicit_decisions_override_labels_and_keep_duplicate_rows(self):
        before = self.input.read_bytes()
        report = self.run_eval()
        self.assertEqual(before, self.input.read_bytes())
        self.assertEqual(report['eligible_prompts'], 3)
        self.assertFalse(report['simulated_governance'])
        with (self.output / 'compression_results.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]['status'], 'blocked')
        self.assertEqual(rows[0]['compressed_prompt'], '')
        self.assertEqual(rows[0]['compressed_tokens'], '')
        self.assertEqual(rows[3]['action'], 'pass')  # Labels do not decide eligibility.
        self.assertEqual(len({r['prompt_id'] for r in rows}), 4)
        eligible = rows[1:]
        reductions = [float(r['reduction_percent']) for r in eligible]
        self.assertAlmostEqual(report['mean_reduction_percent'], sum(reductions) / 3)
        self.assertEqual(report['median_reduction_percent'], 0)
        original = sum(int(r['original_tokens']) for r in eligible)
        final = sum(int(r['compressed_tokens']) for r in eligible)
        self.assertAlmostEqual(report['aggregate_reduction_percent'], 100 * (original - final) / original)

    def test_all_blocked_has_null_metrics(self):
        for row in self.payload['decisions']:
            row['action'] = 'block'
        self.save_decisions()
        report = self.run_eval()
        self.assertIsNone(report['mean_reduction_percent'])
        self.assertIsNone(report['median_reduction_percent'])
        self.assertIsNone(report['aggregate_reduction_percent'])
        self.assertEqual(report['tokens_saved'], 0)

    def test_demo_is_explicit_and_labeled(self):
        with self.assertRaises(ValueError):
            evaluate(self.input, self.baseline, self.output)
        with self.assertRaises(ValueError):
            evaluate(self.input, self.baseline, self.output, self.decisions, demo_governance=True)
        report = evaluate(self.input, self.baseline, self.output, demo_governance=True)
        self.assertTrue(report['simulated_governance'])
        self.assertEqual(report['eligible_prompts'], 3)

    def test_bad_governance_snapshots_fail_before_writing(self):
        original = json.loads(json.dumps(self.payload))
        for mutation in ('missing', 'duplicate', 'hash', 'unknown', 'action', 'checksum', 'source'):
            self.payload = json.loads(json.dumps(original))
            rows = self.payload['decisions']
            if mutation == 'missing':
                rows.pop()
            elif mutation == 'duplicate':
                rows[-1] = rows[0]
            elif mutation == 'hash':
                rows[0]['prompt_sha256'] = 'bad'
            elif mutation == 'unknown':
                rows[0]['prompt_id'] = 'unknown'
            elif mutation == 'action':
                rows[0]['action'] = 'allow'
            elif mutation == 'checksum':
                self.payload['input_sha256'] = 'bad'
            else:
                self.payload['source'] = ''
            self.save_decisions()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.run_eval()
            self.assertFalse(self.output.exists())

    def test_changed_test_input_is_rejected(self):
        self.input.write_text(self.input.read_text().replace('gravity', 'relativity'))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.run_eval()
        self.assertFalse(self.output.exists())

    def test_shuffled_governance_decisions_join_by_id(self):
        expected = self.run_eval()
        self.payload['decisions'].reverse()
        self.save_decisions()
        self.assertEqual(self.run_eval(), expected)

    def test_tokenizer_drift_is_rejected(self):
        path = self.baseline / 'token_baseline_summary.json'
        metadata = json.loads(path.read_text())
        metadata['encoding'] = 'cl100k_base'
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'Tokenizer'):
            self.run_eval()
        self.assertFalse(self.output.exists())

        path = self.baseline / 'token_baseline.csv'
        with path.open() as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        rows[0]['original_tokens'] = '999'
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, 'alignment/count'):
            self.run_eval()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
