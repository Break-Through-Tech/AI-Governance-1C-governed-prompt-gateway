"""Request and response contract for the gateway (Pydantic v2).

Completed, blocked, and error responses are separate types joined by a
`status` discriminator. Each forbids extra fields, so a blocked response
cannot carry cache, token, or routing metadata.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, create_model, field_validator

DEFAULT_MAX_PROMPT_CHARS = 4000

RequestId = Annotated[str, StringConstraints(min_length=1, max_length=128,
                                             pattern=r'^[A-Za-z0-9._:-]+$')]


class _Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class PromptRequest(_Strict):
    prompt: str = Field(min_length=1, max_length=DEFAULT_MAX_PROMPT_CHARS)
    request_id: RequestId | None = None

    @field_validator('prompt')
    @classmethod
    def _not_blank(cls, prompt: str) -> str:
        # Validate only; the original text is kept because compression is whitespace-sensitive.
        if not prompt.strip():
            raise ValueError('prompt must contain non-whitespace text')
        return prompt


def request_model(max_prompt_chars: int) -> type[PromptRequest]:
    """PromptRequest with a configurable maximum prompt length."""
    if max_prompt_chars < 1:
        raise ValueError('max_prompt_chars must be positive')
    return create_model('PromptRequest', __base__=PromptRequest,
                        prompt=(str, Field(min_length=1, max_length=max_prompt_chars)))


class GovernanceInfo(_Strict):
    label: Literal['safe', 'toxic', 'jailbreak']
    risk_score: float
    score_type: str
    action: Literal['pass', 'block']
    reason: str


class CacheInfo(_Strict):
    status: Literal['hit', 'miss']
    similarity: float | None
    lookup_failed: bool = False  # True when lookup raised and was treated as a miss.
    store: Literal['stored', 'failed', 'not_attempted']


class TokenInfo(_Strict):
    original: int
    compressed: int | None  # None on a cache hit: compression does not run.
    compression_status: str


class RoutingInfo(_Strict):
    tier: str
    estimated_cost_usd: float
    is_stub: Literal[True] = True  # Placeholder until issue #21 lands.


class CompletedResponse(_Strict):
    status: Literal['completed'] = 'completed'
    request_id: str
    policy_version: str
    response: str
    governance: GovernanceInfo
    cache: CacheInfo
    tokens: TokenInfo
    routing: RoutingInfo | None  # None on a cache hit: no model is selected or called.
    timings_ms: dict[str, float]


class BlockedResponse(_Strict):
    status: Literal['blocked'] = 'blocked'
    request_id: str
    policy_version: str
    response: None = None
    message: str
    governance: GovernanceInfo
    timings_ms: dict[str, float]


class ErrorResponse(_Strict):
    status: Literal['error'] = 'error'
    request_id: str
    policy_version: str
    response: None = None
    stage: str
    error: str  # Generic text only; exception details stay in server logs.
    timings_ms: dict[str, float]


GatewayResponse = Annotated[Union[CompletedResponse, BlockedResponse, ErrorResponse],
                            Field(discriminator='status')]


class HealthResponse(_Strict):
    status: Literal['ok', 'degraded']
    policy_version: str
    components: dict[str, str]
