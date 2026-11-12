"""Callable risk-classification interface for the gateway pipeline (issue
#18). Loads the v2 Logistic Regression model
(scripts/train_classifier_v2.py), task 5's TF-IDF vectorizer, and the
sentence embedding model once, lazily, and classifies a raw prompt string.

v2's features are TF-IDF concatenated with a sentence embedding, in that
column order -- see scripts/train_classifier_v2.py's
build_combined_features(). A single prompt must be featurized the exact
same way for the model's coefficients to line up with the right columns.

Output is ready to pass straight to scripts.governance.apply_governance():
{'label', 'risk_score', 'score_type'} -- the same shape governance.py's
existing classify_with_svm() stub adapter returns, and the real classifier
that stub was always a placeholder for.

risk_score = P(toxic) + P(jailbreak), not confidence in the predicted
label: a confident "safe" prediction must score LOW risk, not high, so this
can't just be the predicted class's own probability (max(predict_proba)).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer

from scripts.train_classifier_v2 import MODEL_NAME

ROOT = Path(__file__).resolve().parents[1]
VECTORIZER_PATH = ROOT / 'data/processed/tfidf/tfidf_vectorizer.joblib'
MODEL_PATH = ROOT / 'data/processed/classifier_v2/classifier_v2.joblib'
RISK_LABELS = ('toxic', 'jailbreak')
SCORE_TYPE = 'logistic_regression_probability'

_vectorizer = None
_embedder = None
_model = None


def _load():
    global _vectorizer, _embedder, _model
    if _vectorizer is None:
        _vectorizer = joblib.load(VECTORIZER_PATH)
    if _embedder is None:
        _embedder = SentenceTransformer(MODEL_NAME)
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _vectorizer, _embedder, _model


def _featurize(prompt, vectorizer, embedder):
    tfidf = vectorizer.transform([prompt])
    embedding = np.asarray(
        embedder.encode([prompt], normalize_embeddings=True, show_progress_bar=False), dtype='float32')
    return hstack([tfidf, csr_matrix(embedding)], format='csr')


def classify(prompt: str) -> dict:
    """Classify one prompt. Returns {'label', 'risk_score', 'score_type'}.

    risk_score is P(toxic) + P(jailbreak) from the model's own predicted
    probabilities, indexed by model.classes_ rather than assumed order."""
    vectorizer, embedder, model = _load()
    features = _featurize(prompt, vectorizer, embedder)
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)
    label = classes[probabilities.argmax()]
    risk_score = float(sum(probabilities[classes.index(risk_label)] for risk_label in RISK_LABELS))
    return {'label': str(label), 'risk_score': risk_score, 'score_type': SCORE_TYPE}
