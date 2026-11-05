"""Label-based governance with a required JSONL audit trail (issue #17)."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
from numbers import Real
from pathlib import Path

POLICY_VERSION = 'label-policy-v1'
ACTIONS = {'safe': 'pass', 'toxic': 'block', 'jailbreak': 'block'}


def _nonempty_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be nonempty text')


def _classification(output):
    if not isinstance(output, Mapping):
        raise ValueError('Classifier output must be a mapping with label and risk_score')
    label = output.get('label')
    if not isinstance(label, str) or label not in ACTIONS:
        raise ValueError('label must be exactly safe, toxic, or jailbreak')
    score = output.get('risk_score')
    if isinstance(score, bool) or not isinstance(score, Real):
        raise ValueError('risk_score must be a finite real number, not a boolean')
    try:
        score = float(score)
    except (OverflowError, ValueError) as error:
        raise ValueError('risk_score must be a finite real number') from error
    if not math.isfinite(score):
        raise ValueError('risk_score must be a finite real number')
    score_type = output.get('score_type', 'classifier_score')
    _nonempty_text(score_type, 'score_type')
    return label, score, score_type


def apply_governance(classifier_output, *, prompt_id: str, log_path: Path | str) -> dict:
    """Return a logged pass/block decision; validation and I/O errors propagate.

    Only the label determines the action. risk_score is recorded as supplied;
    it has no probability interpretation or threshold in this policy.
    """
    _nonempty_text(prompt_id, 'prompt_id')
    label, score, score_type = _classification(classifier_output)
    action = ACTIONS[label]
    blocked = action == 'block'
    decision = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'prompt_id': prompt_id,
        'policy_version': POLICY_VERSION,
        'label': label,
        'risk_score': score,
        'score_type': score_type,
        'action': action,
        'status_code': 403 if blocked else 200,
        'reason': 'blocked_label' if blocked else 'allowed_label',
        'message': ('Request blocked by the prompt safety policy.' if blocked
                    else 'Request passed the prompt safety policy.'),
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(decision, ensure_ascii=False, allow_nan=False) + '\n')
    # No decision is returned if opening, writing, or closing the log fails.
    return decision


def classify_with_svm(prompt: str, *, model, vectorizer) -> dict:
    """Adapt the existing three-class LinearSVC interface; never calibrate scores.

    risk_score is the predicted class's decision_function column, indexed using
    model.classes_. It is a signed decision score, not a probability of harm.
    """
    import numpy as np  # The policy and stub example need only the standard library.

    _nonempty_text(prompt, 'prompt')
    classes = list(model.classes_)
    if len(classes) != 3 or set(classes) != set(ACTIONS):
        raise ValueError('Expected the three-class safe/toxic/jailbreak SVM')
    features = vectorizer.transform([prompt])
    predicted = model.predict(features)
    scores = np.asarray(model.decision_function(features))
    if (len(predicted) != 1 or predicted[0] not in classes
            or scores.shape != (1, 3)):
        raise ValueError('Expected one prediction and one score per SVM class')
    if not np.isfinite(scores).all():
        raise ValueError('SVM returned nonfinite decision scores')
    label = str(predicted[0])
    output = {'label': label, 'risk_score': float(scores[0, classes.index(label)]),
              'score_type': 'svm_decision_function'}
    _classification(output)
    return output
