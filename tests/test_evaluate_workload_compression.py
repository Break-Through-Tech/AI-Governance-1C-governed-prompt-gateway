"""Check scripts/evaluate_workload_compression.py's per-request/per-source
measurement and the real governance wiring. classify() is mocked (no need
to load the real model for these); apply_governance() and compress_prompt()
run for real, since they're fast and pure."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_workload_compression import evaluate_request, render_context, render_history

DESIGN = {
    'system_prompt': 'Full verbose instructions for the assistant, repeated at length.',
    'tightened_system_prompt': 'Short instructions.',
    'context_pool': {
        'a': 'Context snippet A.', 'b': 'Context snippet B.',
        'c': 'Context snippet C.', 'd': 'Context snippet D.',
    },
    'context_per_query': 3,
    'context_kept_after_compression': 1,
    'chat_history': [
        ('customer', 'Hi, I need help.'), ('assistant', 'Sure, how can I help?'),
        ('customer', 'My actual question.'), ('assistant', 'Let me look into that.'),
    ],
    'history_turns_kept_after_compression': 2,
}


class RenderHelperTests(unittest.TestCase):
    def test_render_history_joins_role_and_text_in_order(self):
        rendered = render_history(DESIGN['chat_history'][:2])
        self.assertEqual(rendered, 'customer: Hi, I need help.\nassistant: Sure, how can I help?')

    def test_render_context_joins_pool_entries_by_id_in_order(self):
        rendered = render_context(DESIGN, ['b', 'a'])
        self.assertEqual(rendered, 'Context snippet B.\nContext snippet A.')


class EvaluateRequestTests(unittest.TestCase):
    def test_blocked_request_skips_compression_and_records_the_classifier_label(self):
        query = {'test_row': 1, 'user_message': 'some prompt', 'retrieved_context_ids': ['a', 'b', 'c']}
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / 'audit.jsonl'
            with patch('scripts.evaluate_workload_compression.classify',
                      return_value={'label': 'toxic', 'risk_score': 0.9, 'score_type': 'test'}):
                result = evaluate_request(DESIGN, query, log_path)
            self.assertTrue(log_path.is_file())
        self.assertEqual(result['action'], 'block')
        self.assertEqual(result['classifier_label'], 'toxic')
        self.assertNotIn('before_by_source', result)

    def test_passed_request_measures_every_source_before_and_after(self):
        query = {'test_row': 2, 'user_message': 'Can you explain this?',
                 'retrieved_context_ids': ['a', 'b', 'c']}
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / 'audit.jsonl'
            with patch('scripts.evaluate_workload_compression.classify',
                      return_value={'label': 'safe', 'risk_score': 0.01, 'score_type': 'test'}):
                result = evaluate_request(DESIGN, query, log_path)

        self.assertEqual(result['action'], 'pass')
        self.assertEqual(set(result['before_by_source']), {'system_prompt', 'context', 'history', 'user_message'})
        for source in result['before_by_source']:
            self.assertGreaterEqual(result['before_by_source'][source], result['after_by_source'][source])
        self.assertTrue(result['user_message_preserved'])
        # "Can you explain this?" has a supported wrapper before "explain" -> compressed.
        self.assertEqual(result['user_message_status'], 'compressed')

    def test_context_is_trimmed_to_the_kept_count_and_history_to_the_kept_turns(self):
        query = {'test_row': 3, 'user_message': 'Tell me about your return policy.',
                 'retrieved_context_ids': ['d', 'c', 'b']}
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / 'audit.jsonl'
            with patch('scripts.evaluate_workload_compression.classify',
                      return_value={'label': 'safe', 'risk_score': 0.0, 'score_type': 'test'}):
                result = evaluate_request(DESIGN, query, log_path)

        # 1 kept context snippet ("d", the first of the simulated-ranked list)
        # and 2 kept history turns (the last exchange) by DESIGN above.
        self.assertEqual(result['after_by_source']['context'],
                         render_tokens(DESIGN['context_pool']['d']))
        self.assertEqual(result['after_by_source']['history'],
                         render_tokens('customer: My actual question.\nassistant: Let me look into that.'))


def render_tokens(text):
    from scripts.measure_token_baseline import count_tokens
    return count_tokens(text)


if __name__ == '__main__':
    unittest.main()
