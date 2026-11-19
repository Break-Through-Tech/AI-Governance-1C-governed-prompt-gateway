"""Check the HTTP layer with FastAPI's TestClient: status codes, response
shapes, validation, and /health's closed/open rule. Components are injected,
so no model artifacts are needed."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.components import (GatewayComponents, LoggedGovernance, PromptCompressor,
                                SimulatedGenerator, StubRouter, TiktokenCounter)
from tests.test_gateway_pipeline import JAILBREAK, SAFE, TOXIC, FakeClassifier

COMPLETED_KEYS = {'status', 'request_id', 'policy_version', 'response', 'governance', 'cache',
                  'tokens', 'routing', 'timings_ms'}
BLOCKED_KEYS = {'status', 'request_id', 'policy_version', 'response', 'message', 'governance',
                'timings_ms'}


class DictCache:
    """Exact-match stand-in for the semantic cache."""

    def __init__(self):
        self.entries = {}

    def lookup(self, prompt):
        response = self.entries.get(prompt)
        return type('Lookup', (), {'hit': response is not None, 'response': response,
                                   'similarity': 1.0 if response else None})()

    def store(self, prompt, response):
        self.entries[prompt] = response


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.log_path = Path(directory.name) / 'decisions.jsonl'

    def client(self, *, classifier='fake', cache='dict', problems=None):
        components = GatewayComponents(
            classifier=FakeClassifier() if classifier == 'fake' else classifier,
            governance=LoggedGovernance(self.log_path),
            cache=DictCache() if cache == 'dict' else cache,
            compressor=PromptCompressor(), token_counter=TiktokenCounter(),
            router=StubRouter(), generator=SimulatedGenerator(), problems=problems or {})
        client = TestClient(create_app(components))
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client


class PromptEndpointTests(ApiTestCase):
    def test_safe_prompt_returns_200_with_completed_shape(self):
        response = self.client().post('/v1/prompt', json={'prompt': SAFE, 'request_id': 'req-1'})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body), COMPLETED_KEYS)
        self.assertEqual((body['status'], body['request_id']), ('completed', 'req-1'))
        self.assertTrue(body['routing']['is_stub'])

    def test_blocked_prompts_return_403_with_blocked_shape_only(self):
        client = self.client()
        for prompt, label in ((TOXIC, 'toxic'), (JAILBREAK, 'jailbreak')):
            with self.subTest(label=label):
                response = client.post('/v1/prompt', json={'prompt': prompt})
                self.assertEqual(response.status_code, 403)
                body = response.json()
                self.assertEqual(set(body), BLOCKED_KEYS)
                self.assertEqual(body['status'], 'blocked')
                self.assertIsNone(body['response'])
                self.assertEqual(body['governance']['label'], label)
                self.assertEqual(body['governance']['action'], 'block')
                for field in ('cache', 'tokens', 'routing'):
                    self.assertNotIn(field, body)

    def test_repeat_is_a_cache_hit_over_http(self):
        client = self.client()
        client.post('/v1/prompt', json={'prompt': SAFE})
        body = client.post('/v1/prompt', json={'prompt': SAFE}).json()
        self.assertEqual(body['cache']['status'], 'hit')
        self.assertIsNone(body['routing'])

    def test_request_id_is_generated_when_absent(self):
        body = self.client().post('/v1/prompt', json={'prompt': SAFE}).json()
        uuid.UUID(body['request_id'])  # Raises if not a UUID.

    def test_missing_classifier_returns_503_error_shape(self):
        with self.assertLogs('gateway', 'ERROR'):
            response = self.client(classifier=None).post('/v1/prompt', json={'prompt': SAFE})
        self.assertEqual(response.status_code, 503)
        body = response.json()
        self.assertEqual((body['status'], body['stage']), ('error', 'classify'))
        self.assertIsNone(body['response'])


class ValidationTests(ApiTestCase):
    def test_invalid_requests_return_422(self):
        client = self.client()
        cases = {
            'empty prompt': {'prompt': ''},
            'whitespace prompt': {'prompt': '   \n\t'},
            'missing prompt': {},
            'non-string prompt': {'prompt': 123},
            'unknown field': {'prompt': SAFE, 'admin': True},
            'bad request_id': {'prompt': SAFE, 'request_id': 'has spaces'},
            'empty request_id': {'prompt': SAFE, 'request_id': ''},
        }
        for name, payload in cases.items():
            with self.subTest(name):
                self.assertEqual(client.post('/v1/prompt', json=payload).status_code, 422)
        self.assertFalse(self.log_path.exists())  # Rejected requests are never governed.

    def test_max_prompt_length_is_configurable(self):
        with patch.dict(os.environ, {'GATEWAY_MAX_PROMPT_CHARS': '10'}):
            client = self.client()
        self.assertEqual(client.post('/v1/prompt', json={'prompt': 'x' * 11}).status_code, 422)


class HealthTests(ApiTestCase):
    def test_all_components_up_is_200_ok(self):
        response = self.client().get('/health')
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['status'], 'ok')
        self.assertEqual(body['components']['router'], 'stub (#21)')
        self.assertEqual(body['components']['generator'], 'simulated')

    def test_fail_open_problems_are_200_degraded(self):
        for name, kwargs in (('cache missing', {'cache': None}),
                             ('token counter problem', {'problems': {'token_counter': 'x'}})):
            with self.subTest(name):
                response = self.client(**kwargs).get('/health')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()['status'], 'degraded')

    def test_fail_closed_problems_are_503_unavailable(self):
        for name, kwargs in (('classifier missing', {'classifier': None}),
                             ('classifier problem', {'problems': {'classifier': 'x'}}),
                             ('governance problem', {'problems': {'governance': 'x'}})):
            with self.subTest(name):
                response = self.client(**kwargs).get('/health')
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()['status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
