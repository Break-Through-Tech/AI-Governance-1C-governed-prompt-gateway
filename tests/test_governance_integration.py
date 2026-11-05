"""Check the existing SVM and compression handoffs without saved model artifacts."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from scripts.compress_prompts import compress_prompt
from scripts.governance import apply_governance, classify_with_svm


class SVMAdapterTests(unittest.TestCase):
    def test_real_three_class_svm_interface(self):
        # A tiny fixture checks the API, not model accuracy or the trained baseline.
        texts = ['weather rain', 'weather sun', 'insult rude', 'insult cruel',
                 'override rules', 'override restrictions']
        labels = ['safe', 'safe', 'toxic', 'toxic', 'jailbreak', 'jailbreak']
        vectorizer = TfidfVectorizer()
        features = vectorizer.fit_transform(texts)
        model = LinearSVC(random_state=42).fit(features, labels)
        for prompt in texts:
            output = classify_with_svm(prompt, model=model, vectorizer=vectorizer)
            row = vectorizer.transform([prompt])
            label = model.predict(row)[0]
            index = list(model.classes_).index(label)
            self.assertEqual(output['label'], label)
            self.assertAlmostEqual(output['risk_score'], model.decision_function(row)[0, index])
            self.assertEqual(output['score_type'], 'svm_decision_function')

    def test_adapter_uses_model_class_order_and_keeps_signed_scores(self):
        model = Mock(classes_=np.array(['toxic', 'jailbreak', 'safe']))
        model.predict.return_value = ['jailbreak']
        model.decision_function.return_value = [[-2.1, -0.2, -3.5]]
        vectorizer = Mock()
        output = classify_with_svm('sample', model=model, vectorizer=vectorizer)
        self.assertEqual(output['label'], 'jailbreak')
        self.assertEqual(output['risk_score'], -0.2)
        vectorizer.transform.assert_called_once_with(['sample'])

    def test_invalid_adapter_inputs(self):
        model = Mock(classes_=np.array(['jailbreak', 'safe', 'toxic']))
        model.predict.return_value = ['safe']
        vectorizer = Mock()
        for scores in ([1, 2, 3], [[1, 2]], [[1, np.nan, 3]], [[1, np.inf, 3]]):
            model.decision_function.return_value = scores
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                classify_with_svm('sample', model=model, vectorizer=vectorizer)
        model.classes_ = ['safe', 'toxic']
        with self.assertRaises(ValueError):
            classify_with_svm('sample', model=model, vectorizer=vectorizer)
        with self.assertRaises(ValueError):
            classify_with_svm('', model=model, vectorizer=vectorizer)


class CompressionHandoffTests(unittest.TestCase):
    def test_blocked_decision_stops_existing_compressor(self):
        with tempfile.TemporaryDirectory() as directory:
            for label in ('toxic', 'jailbreak'):
                decision = apply_governance(
                    {'label': label, 'risk_score': -0.1}, prompt_id=label,
                    log_path=Path(directory) / 'audit.jsonl')
                with patch('scripts.compress_prompts.count_tokens', side_effect=AssertionError):
                    result = compress_prompt('Please explain something', action=decision['action'])
                self.assertEqual(result.status, 'blocked')
                self.assertIsNone(result.compressed_prompt)

    def test_passed_decision_reaches_existing_compressor(self):
        with tempfile.TemporaryDirectory() as directory:
            decision = apply_governance(
                {'label': 'safe', 'risk_score': 2.5}, prompt_id='safe-request',
                log_path=Path(directory) / 'audit.jsonl')
            result = compress_prompt('Could you please list 3 options under $50, excluding flights?',
                                     action=decision['action'])
            self.assertEqual(result.action, 'pass')
            self.assertEqual(result.compressed_prompt, 'list 3 options under $50, excluding flights?')


if __name__ == '__main__':
    unittest.main()
