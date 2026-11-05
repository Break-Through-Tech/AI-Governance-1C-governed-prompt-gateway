"""Policy, response contract, invalid input, and persisted audit records."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.governance import POLICY_VERSION, apply_governance
from scripts.governance_demo import run_demo, stub_classifier


class GovernanceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.log = Path(temp.name) / 'audit/decisions.jsonl'

    def decide(self, output, prompt_id='request-1'):
        return apply_governance(output, prompt_id=prompt_id, log_path=self.log)

    def test_policy_and_response_structure(self):
        for label, action, code, reason in (
            ('safe', 'pass', 200, 'allowed_label'),
            ('toxic', 'block', 403, 'blocked_label'),
            ('jailbreak', 'block', 403, 'blocked_label'),
        ):
            with self.subTest(label=label):
                result = self.decide({'label': label, 'risk_score': 0.4})
                self.assertEqual(set(result), {'timestamp', 'prompt_id', 'policy_version',
                                               'label', 'risk_score', 'score_type', 'action',
                                               'status_code', 'reason', 'message'})
                self.assertEqual((result['action'], result['status_code'], result['reason']),
                                 (action, code, reason))
                self.assertEqual(result['label'], label)
                self.assertEqual(result['risk_score'], 0.4)
                self.assertEqual(result['policy_version'], POLICY_VERSION)
                self.assertEqual(result['score_type'], 'classifier_score')
                self.assertEqual(result['message'], 'Request blocked by the prompt safety policy.'
                                 if action == 'block' else 'Request passed the prompt safety policy.')
                self.assertNotIn('prompt', result)

    def test_scores_do_not_change_the_label_policy(self):
        for score in (-12.5, -0.1, 0, 0.5, 1, 28.0):
            for label, action in (('safe', 'pass'), ('toxic', 'block'), ('jailbreak', 'block')):
                with self.subTest(score=score, label=label):
                    result = self.decide({'label': label, 'risk_score': score})
                    self.assertEqual(result['action'], action)
                    self.assertEqual(result['risk_score'], score)

    def test_invalid_classifier_outputs_are_rejected_without_logging(self):
        invalid = [None, [], 'safe', {}, {'label': 'safe'}, {'risk_score': 0.2}]
        invalid += [{'label': label, 'risk_score': 0.2}
                    for label in ('unknown', 'harmful', 'Safe', ' safe ', '', None, [], 1)]
        invalid += [{'label': 'safe', 'risk_score': score}
                    for score in (None, '0.9', True, False, [], {}, complex(1, 2),
                                  float('nan'), float('inf'), -float('inf'), 10 ** 1000)]
        invalid += [{'label': 'safe', 'risk_score': 0.2, 'score_type': value}
                    for value in ('', ' ', None, 1)]
        for output in invalid:
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.decide(output)
        self.assertFalse(self.log.exists())

    def test_prompt_id_is_required(self):
        for prompt_id in ('', '  ', None, 123):
            with self.subTest(prompt_id=prompt_id), self.assertRaises(ValueError):
                self.decide({'label': 'safe', 'risk_score': 0.2}, prompt_id)
        self.assertFalse(self.log.exists())

    def test_every_returned_decision_is_appended_with_utc_timestamp(self):
        start = datetime.now().astimezone()
        expected = [self.decide({'label': label, 'risk_score': score, 'score_type': 'stub_fixture'},
                               prompt_id='request-雪\n1')
                    for label, score in [('safe', -0.2), ('toxic', 1.4), ('jailbreak', 0.8)]]
        lines = self.log.read_text(encoding='utf-8').splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual([json.loads(line) for line in lines], expected)
        for result in expected:
            timestamp = datetime.fromisoformat(result['timestamp'])
            self.assertEqual(timestamp.utcoffset(), timedelta(0))
            self.assertGreaterEqual(timestamp, start)
            self.assertLessEqual(timestamp, datetime.now().astimezone())
        before = self.log.read_bytes()
        with self.assertRaises(ValueError):
            self.decide({'label': 'unknown', 'risk_score': 1})
        self.assertEqual(self.log.read_bytes(), before)

    def test_logging_failure_never_returns_a_pass(self):
        with patch('scripts.governance.Path.open', side_effect=PermissionError('read only')):
            with self.assertRaises(PermissionError):
                self.decide({'label': 'safe', 'risk_score': 1})
        with patch('scripts.governance.Path.open') as open_log:
            open_log.return_value.__enter__.return_value.write.side_effect = OSError('disk full')
            with self.assertRaises(OSError):
                self.decide({'label': 'safe', 'risk_score': 1})

    def test_stub_demo_and_saved_log(self):
        decisions = run_demo(self.log)
        self.assertEqual([d['action'] for d in decisions], ['pass', 'block', 'block'])
        self.assertTrue(all(d['score_type'] == 'stub_fixture' for d in decisions))
        self.assertEqual([json.loads(line) for line in self.log.read_text().splitlines()], decisions)
        with self.assertRaises(ValueError):
            stub_classifier('An unsupported sample')


if __name__ == '__main__':
    unittest.main()
