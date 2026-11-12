"""Stage interfaces, adapters around teammates' modules, and placeholder stubs.

Each stage is a Protocol so #18 (classifier) and #21 (router) owners can drop
in a replacement without touching the pipeline. Adapters only wrap existing
modules; they do not change their behavior.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
import logging
import os
from pathlib import Path
import pickle
from typing import Protocol

import joblib

from scripts.compress_prompts import CompressionResult, compress_prompt
from scripts.governance import apply_governance, classify_with_svm
from scripts.measure_token_baseline import MODEL, RATE, ROOT, count_tokens

logger = logging.getLogger('gateway')

DEFAULT_MODEL_PATH = ROOT / 'data/processed/models/best_model.pkl'
DEFAULT_VECTORIZER_PATH = ROOT / 'data/processed/tfidf/tfidf_vectorizer.joblib'
DEFAULT_LOG_PATH = ROOT / 'data/processed/gateway/decisions.jsonl'
DEFAULT_CACHE_THRESHOLD = 0.85


# Stage interfaces -----------------------------------------------------------

class Classifier(Protocol):
    def classify(self, prompt: str) -> Mapping:
        """Return {label, risk_score, score_type}; raise on any failure."""


class Governance(Protocol):
    def decide(self, classification: Mapping, request_id: str) -> dict:
        """Return a logged pass/block decision; raise if it cannot be logged."""


class CacheLookup(Protocol):
    hit: bool
    response: str | None
    similarity: float | None


class Cache(Protocol):
    def lookup(self, prompt: str) -> CacheLookup: ...
    def store(self, prompt: str, response: str) -> None: ...


class Compressor(Protocol):
    def compress(self, prompt: str, action: str) -> CompressionResult: ...


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class Router(Protocol):
    def recommend(self, prompt: str, risk_score: float, token_count: int) -> dict:
        """Return {tier, estimated_cost_usd}."""


class Generator(Protocol):
    def generate(self, prompt: str, tier: str) -> str: ...


# Adapters around existing modules ---------------------------------------------

class SvmClassifier:
    """Baseline LinearSVC (#7) via scripts.governance.classify_with_svm."""

    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

    @classmethod
    def load(cls, model_path: Path | str, vectorizer_path: Path | str) -> SvmClassifier:
        # pickle executes code on load: only point this at locally trained, trusted artifacts.
        with Path(model_path).open('rb') as stream:
            model = pickle.load(stream)
        return cls(model, joblib.load(vectorizer_path))

    def classify(self, prompt: str) -> Mapping:
        return classify_with_svm(prompt, model=self.model, vectorizer=self.vectorizer)


class LoggedGovernance:
    """scripts.governance.apply_governance with a fixed JSONL audit log."""

    def __init__(self, log_path: Path | str):
        self.log_path = Path(log_path)

    def decide(self, classification: Mapping, request_id: str) -> dict:
        return apply_governance(classification, prompt_id=request_id, log_path=self.log_path)


class SemanticCacheAdapter:
    """scripts.semantic_cache.SemanticCache, imported lazily (faiss + torch are heavy)."""

    def __init__(self, threshold: float = DEFAULT_CACHE_THRESHOLD, model=None):
        from scripts.semantic_cache import SemanticCache
        self.cache = SemanticCache(threshold=threshold, model=model)

    def lookup(self, prompt: str) -> CacheLookup:
        return self.cache.lookup(prompt)

    def store(self, prompt: str, response: str) -> None:
        self.cache.store(prompt, response)


class PromptCompressor:
    """scripts.compress_prompts.compress_prompt (#19)."""

    def compress(self, prompt: str, action: str) -> CompressionResult:
        return compress_prompt(prompt, action)


class TiktokenCounter:
    """The #8 token counter (tiktoken o200k_base)."""

    def count(self, text: str) -> int:
        return count_tokens(text)


# Placeholders -----------------------------------------------------------------

class StubRouter:
    """PLACEHOLDER for #21. Always picks the #8 reference model; no routing logic.

    Cost is input tokens only, at the #8 reference price. risk_score is accepted
    to fix the interface but ignored: it is a signed SVM score, not a probability.
    """
    TIER = 'reference'

    def recommend(self, prompt: str, risk_score: float, token_count: int) -> dict:
        cost = Decimal(token_count) * RATE / Decimal(1_000_000)
        return {'tier': self.TIER, 'estimated_cost_usd': float(cost)}


class SimulatedGenerator:
    """PLACEHOLDER: no LLM is called. Output depends only on the prompt and tier."""

    def generate(self, prompt: str, tier: str) -> str:
        tag = hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:12]
        return f'[simulated response | tier={tier} | prompt={tag}] No LLM was called.'


# Wiring -----------------------------------------------------------------------

@dataclass(frozen=True)
class GatewayConfig:
    model_path: Path = DEFAULT_MODEL_PATH
    vectorizer_path: Path = DEFAULT_VECTORIZER_PATH
    log_path: Path = DEFAULT_LOG_PATH
    cache_threshold: float = DEFAULT_CACHE_THRESHOLD
    max_prompt_chars: int = 4000

    @classmethod
    def from_env(cls) -> GatewayConfig:
        env = os.environ.get
        return cls(
            model_path=Path(env('GATEWAY_MODEL_PATH', DEFAULT_MODEL_PATH)),
            vectorizer_path=Path(env('GATEWAY_VECTORIZER_PATH', DEFAULT_VECTORIZER_PATH)),
            log_path=Path(env('GATEWAY_LOG_PATH', DEFAULT_LOG_PATH)),
            cache_threshold=float(env('GATEWAY_CACHE_THRESHOLD', DEFAULT_CACHE_THRESHOLD)),
            max_prompt_chars=int(env('GATEWAY_MAX_PROMPT_CHARS', 4000)),
        )


@dataclass
class GatewayComponents:
    classifier: Classifier | None  # None when artifacts are missing: requests get 503.
    governance: Governance
    cache: Cache | None            # None when the cache failed to start: every lookup is a miss.
    compressor: Compressor
    token_counter: TokenCounter
    router: Router
    generator: Generator
    problems: dict[str, str] = field(default_factory=dict)  # stage -> reason, for /health.


def build_components(config: GatewayConfig) -> GatewayComponents:
    """Load the real components; record (never raise) startup problems."""
    problems = {}
    classifier = None
    try:
        classifier = SvmClassifier.load(config.model_path, config.vectorizer_path)
    except FileNotFoundError as error:
        problems['classifier'] = f'artifact missing: {Path(error.filename).name}'
        logger.error('Classifier artifacts missing; /v1/prompt will return 503')
    except Exception as error:
        problems['classifier'] = f'failed to load: {type(error).__name__}'
        logger.exception('Classifier failed to load')
    cache = None
    try:
        cache = SemanticCacheAdapter(threshold=config.cache_threshold)
    except Exception as error:
        problems['cache'] = f'failed to start: {type(error).__name__}'
        logger.exception('Semantic cache failed to start; all lookups will miss')
    return GatewayComponents(
        classifier=classifier, governance=LoggedGovernance(config.log_path), cache=cache,
        compressor=PromptCompressor(), token_counter=TiktokenCounter(),
        router=StubRouter(), generator=SimulatedGenerator(), problems=problems)
