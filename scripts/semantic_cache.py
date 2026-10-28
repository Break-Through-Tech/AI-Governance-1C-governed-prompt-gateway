"""Embedding-based semantic cache: reuse an earlier response for a prompt
that means roughly the same thing, even if the wording differs"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'all-MiniLM-L6-v2'
DEFAULT_THRESHOLD = 0.85


@dataclass
class CacheResult:
    hit: bool
    matched_prompt: str | None
    response: str | None
    similarity: float | None #cosine similarity of the matched prompt, if any
    latency_seconds: float #how long the lookup took, in seconds

    def to_dict(self) -> dict:
        return asdict(self)


class SemanticCache:
    """lookup(prompt) returns a cached response if a sufficiently similar
    prompt was stored before; 
    store(prompt, response) adds a new pair."""

    def __init__(self, model_name: str = MODEL_NAME, threshold: float = DEFAULT_THRESHOLD, model: SentenceTransformer | None = None):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError('threshold must be between 0 and 1')

        self.model = model if model is not None else SentenceTransformer(model_name)
        self.threshold = threshold
        self.dim = self.model.get_sentence_embedding_dimension()

        # store normalized embeddings in an inner FAISS index for fast similarity search
        self.index = faiss.IndexFlatIP(self.dim)
        self.prompts: list[str] = []
        self.responses: list[str] = []

    def embed(self, text: str) -> np.ndarray:
        """Returns a normalized embedding vector for the given text"""
        vec = self.model.encode([text], normalize_embeddings=True)
        return np.asarray(vec, dtype='float32')

    def lookup(self, prompt: str, embedding: np.ndarray | None = None) -> CacheResult:
        """Lookup a prompt in the semantic cache, using this cache's threshold."""

        t0 = time.perf_counter()
        if self.index.ntotal == 0:
            return CacheResult(False, None, None, None, time.perf_counter() - t0)

        vec = self.embed(prompt) if embedding is None else embedding
        similarities, indices = self.index.search(vec, 1)
        similarity = float(similarities[0][0])
        idx = int(indices[0][0])
        latency = time.perf_counter() - t0
        if similarity >= self.threshold:
            return CacheResult(True, self.prompts[idx], self.responses[idx], similarity, latency)
        return CacheResult(False, None, None, similarity, latency)


    def store(self, prompt: str, response: str, embedding: np.ndarray | None = None) -> None:
        vec = self.embed(prompt) if embedding is None else embedding
        self.index.add(vec)
        self.prompts.append(prompt)
        self.responses.append(response)

    def __len__(self) -> int:
        return self.index.ntotal
