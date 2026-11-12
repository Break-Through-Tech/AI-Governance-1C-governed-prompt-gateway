"""Gateway orchestrator: plain Python, no web framework, testable on its own.

Order (issue #29 supersedes #22's original text):
classify -> governance (logged) -> block: stop
                                -> pass: cache lookup -> hit: cached response
                                                      -> miss: compress -> route -> generate -> store
Fail closed: a classifier or governance failure is an error, never a pass.
Logs carry request IDs, stage names, and exception types, never prompt text.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import time
import uuid

from scripts.governance import POLICY_VERSION

from .components import GatewayComponents
from .schemas import (BlockedResponse, CacheInfo, CompletedResponse, ErrorResponse,
                      GovernanceInfo, RoutingInfo, TokenInfo)

logger = logging.getLogger('gateway')


@dataclass(frozen=True)
class GatewayResult:
    status_code: int  # 200 completed, 403 blocked, 503 component failure.
    body: CompletedResponse | BlockedResponse | ErrorResponse


class StageError(Exception):
    """A required stage failed; the request ends with a 503."""

    def __init__(self, stage: str):
        super().__init__(stage)
        self.stage = stage


class Gateway:
    def __init__(self, components: GatewayComponents, *, clock=time.perf_counter):
        self.c = components
        self.clock = clock

    def handle(self, prompt: str, request_id: str | None = None) -> GatewayResult:
        request_id = request_id or str(uuid.uuid4())
        timings: dict[str, float] = {}
        start = self.clock()
        try:
            result = self._run(prompt, request_id, timings)
        except StageError as error:
            result = GatewayResult(503, ErrorResponse(
                request_id=request_id, policy_version=POLICY_VERSION, stage=error.stage,
                error=f'The {error.stage} stage is unavailable; the request was not processed.',
                timings_ms=timings))
        # Pydantic copied the dict at construction, so add the total to the model's own copy.
        result.body.timings_ms['total'] = (self.clock() - start) * 1000
        return result

    def _stage(self, name: str, timings: dict, request_id: str, call):
        """Run a required stage; any exception becomes a StageError (fail closed)."""
        t0 = self.clock()
        try:
            return call()
        except Exception as error:
            logger.error('request=%s stage=%s failed: %s', request_id, name, type(error).__name__)
            raise StageError(name) from error
        finally:
            timings[name] = (self.clock() - t0) * 1000

    def _run(self, prompt: str, request_id: str, timings: dict) -> GatewayResult:
        c = self.c
        if c.classifier is None:
            logger.error('request=%s stage=classify unavailable: no classifier loaded', request_id)
            raise StageError('classify')
        classification = self._stage('classify', timings, request_id,
                                     lambda: c.classifier.classify(prompt))
        decision = self._stage('governance', timings, request_id,
                               lambda: c.governance.decide(classification, request_id))
        try:
            governance = GovernanceInfo(**{k: decision[k] for k in
                                           ('label', 'risk_score', 'score_type', 'action', 'reason')})
        except Exception as error:
            logger.error('request=%s stage=governance bad decision: %s', request_id, type(error).__name__)
            raise StageError('governance') from error
        policy_version = decision.get('policy_version', POLICY_VERSION)

        if governance.action == 'block':
            return GatewayResult(403, BlockedResponse(
                request_id=request_id, policy_version=policy_version,
                message=decision.get('message', 'Request blocked by the prompt safety policy.'),
                governance=governance, timings_ms=timings))
        # GovernanceInfo only admits pass or block, so everything below is a pass.

        lookup = self._lookup(prompt, request_id, timings)
        if lookup is not None and lookup.hit:
            original = self._stage('count_tokens', timings, request_id,
                                   lambda: c.token_counter.count(prompt))
            return GatewayResult(200, CompletedResponse(
                request_id=request_id, policy_version=policy_version, response=lookup.response,
                governance=governance,
                cache=CacheInfo(status='hit', similarity=lookup.similarity, store='not_attempted'),
                tokens=TokenInfo(original=original, compressed=None,
                                 compression_status='skipped_cache_hit'),
                routing=None, timings_ms=timings))

        compression = self._stage('compress', timings, request_id,
                                  lambda: c.compressor.compress(prompt, governance.action))
        if compression.compressed_prompt is None:
            logger.error('request=%s stage=compress returned no prompt for a pass', request_id)
            raise StageError('compress')
        downstream = compression.compressed_prompt
        routing = self._stage('route', timings, request_id, lambda: RoutingInfo(**{
            k: v for k, v in c.router.recommend(
                downstream, governance.risk_score, compression.compressed_tokens).items()
            if k in ('tier', 'estimated_cost_usd')}))
        response = self._stage('generate', timings, request_id,
                               lambda: c.generator.generate(downstream, routing.tier))
        store = self._store(prompt, response, request_id, timings)
        return GatewayResult(200, CompletedResponse(
            request_id=request_id, policy_version=policy_version, response=response,
            governance=governance,
            cache=CacheInfo(status='miss',
                            similarity=None if lookup is None else lookup.similarity,
                            lookup_failed=lookup is None, store=store),
            tokens=TokenInfo(original=compression.original_tokens,
                             compressed=compression.compressed_tokens,
                             compression_status=compression.status),
            routing=routing, timings_ms=timings))

    def _lookup(self, prompt: str, request_id: str, timings: dict):
        """Return the cache result, or None when the cache is unavailable or raised (a miss)."""
        t0 = self.clock()
        try:
            if self.c.cache is None:
                logger.warning('request=%s stage=cache_lookup skipped: cache unavailable', request_id)
                return None
            return self.c.cache.lookup(prompt)
        except Exception as error:
            logger.warning('request=%s stage=cache_lookup failed, treating as miss: %s',
                           request_id, type(error).__name__)
            return None
        finally:
            timings['cache_lookup'] = (self.clock() - t0) * 1000

    def _store(self, prompt: str, response: str, request_id: str, timings: dict) -> str:
        """Best effort: the response is returned even if storing it fails."""
        if self.c.cache is None:
            return 'not_attempted'
        t0 = self.clock()
        try:
            self.c.cache.store(prompt, response)
            return 'stored'
        except Exception as error:
            logger.warning('request=%s stage=cache_store failed: %s', request_id, type(error).__name__)
            return 'failed'
        finally:
            timings['cache_store'] = (self.clock() - t0) * 1000
