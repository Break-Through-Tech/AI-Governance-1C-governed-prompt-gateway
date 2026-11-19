"""Recommend a model tier and estimate input cost per prompt (issue #21).

Implements the router interface proposed in PR #32 (issue #22):

    recommend(prompt, risk_score, token_count) -> {'tier', 'estimated_cost_usd', ...}

risk_score is the v2 classifier's P(toxic) + P(jailbreak), a probability in
[0, 1] (scripts/classify_prompt.py). token_count is the #8 tiktoken count of the
text that will be sent downstream (after compression), or None if counting
failed. Only prompts that governance passed are routed; blocked prompts never
reach this stage.

Rules, first match wins (thresholds and their sources: docs/model-routing.md):
1. token_count unknown            -> large  (complexity unknown; cost unknown)
2. risk_score >= HIGH_RISK        -> large  (borderline prompt that passed)
3. token_count >  LONG_TOKENS     -> large  (long prompt)
4. risk_score >= ELEVATED_RISK    -> medium
5. token_count >  SHORT_TOKENS    -> medium
6. otherwise                      -> small  (short, low-risk)

estimated_cost_usd covers input tokens only, matching the #8 baseline; output
length is unknown before generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
from numbers import Integral, Real

PRICING_SOURCE = 'https://developers.openai.com/api/docs/pricing'
PRICING_CHECKED = '2026-11-18'
PRICING_MODE = 'standard tier, uncached input, short context'

# Token-length thresholds: training-split percentiles of o200k_base counts
# (P75 = 29, P95 = 195), rounded. See docs/model-routing.md.
SHORT_TOKENS = 30
LONG_TOKENS = 200
# Risk thresholds: out-of-fold v2 risk_score percentiles on train + validation among
# prompts labeled safe (P95 = 0.0013, P99 = 0.108), rounded. They depend on the
# trained model: re-derive with scripts/derive_routing_thresholds.py after retraining.
ELEVATED_RISK = 0.001
HIGH_RISK = 0.10


@dataclass(frozen=True)
class Tier:
    name: str
    model: str
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal

    def input_cost(self, tokens: int) -> Decimal:
        return Decimal(tokens) * self.input_usd_per_million / Decimal(1_000_000)


TIERS = {
    'small': Tier('small', 'gpt-6-luna', Decimal('0.10'), Decimal('0.50')),
    'medium': Tier('medium', 'gpt-6.1-terra', Decimal('1.00'), Decimal('5.00')),
    'large': Tier('large', 'gpt-6.2-astra', Decimal('8.00'), Decimal('40.00')),
}
LARGEST = 'large'


def usd(value: Decimal) -> str:
    """Fixed 12-decimal USD string, the #8 baseline's format."""
    return format(value.quantize(Decimal('0.000000000001')), 'f')


def _check_inputs(prompt, risk_score, token_count) -> tuple[float, int | None]:
    if not isinstance(prompt, str):
        raise ValueError('prompt must be text')
    if isinstance(risk_score, bool) or not isinstance(risk_score, Real):
        raise ValueError('risk_score must be a real number in [0, 1]')
    risk = float(risk_score)
    if not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
        raise ValueError('risk_score must be a real number in [0, 1]')
    if token_count is None:
        return risk, None
    if isinstance(token_count, bool) or not isinstance(token_count, Integral) or token_count < 0:
        raise ValueError('token_count must be a nonnegative integer or None')
    return risk, int(token_count)


class ModelRouter:
    """Rule-based router. Thresholds are constructor arguments so they can be tuned
    without code changes; defaults are the documented values above."""

    def __init__(self, *, short_tokens: int = SHORT_TOKENS, long_tokens: int = LONG_TOKENS,
                 elevated_risk: float = ELEVATED_RISK, high_risk: float = HIGH_RISK,
                 tiers: dict[str, Tier] = TIERS):
        if not 0 <= short_tokens < long_tokens:
            raise ValueError('need 0 <= short_tokens < long_tokens')
        if not 0.0 <= elevated_risk < high_risk <= 1.0:
            raise ValueError('need 0 <= elevated_risk < high_risk <= 1')
        if set(tiers) != set(TIERS):
            raise ValueError(f'tiers must define exactly {sorted(TIERS)}')
        self.short_tokens, self.long_tokens = short_tokens, long_tokens
        self.elevated_risk, self.high_risk = elevated_risk, high_risk
        self.tiers = dict(tiers)

    def rule(self, risk_score: float, token_count: int | None) -> tuple[str, str]:
        """Return (tier name, reason) for already-validated inputs."""
        if token_count is None:
            return 'large', 'token_count_unknown'
        if risk_score >= self.high_risk:
            return 'large', 'high_risk'
        if token_count > self.long_tokens:
            return 'large', 'long_prompt'
        if risk_score >= self.elevated_risk:
            return 'medium', 'elevated_risk'
        if token_count > self.short_tokens:
            return 'medium', 'medium_length'
        return 'small', 'short_low_risk'

    def recommend(self, prompt: str, risk_score: float, token_count: int | None) -> dict:
        """Return {tier, model, estimated_cost_usd, reason, ...}; raise ValueError on
        invalid input. estimated_cost_usd is input-only and None if token_count is None.
        The prompt text is accepted for the agreed interface but not inspected."""
        risk, tokens = _check_inputs(prompt, risk_score, token_count)
        name, reason = self.rule(risk, tokens)
        tier = self.tiers[name]
        cost = None if tokens is None else tier.input_cost(tokens)
        return {
            'tier': name,
            'model': tier.model,
            'estimated_cost_usd': None if cost is None else float(cost),
            'estimated_cost_usd_exact': None if cost is None else usd(cost),
            'cost_scope': 'input_tokens_only',
            'reason': reason,
            'pricing_checked': PRICING_CHECKED,
        }


_default_router = ModelRouter()


def recommend(prompt: str, risk_score: float, token_count: int | None) -> dict:
    """Module-level shortcut using the default thresholds and tiers."""
    return _default_router.recommend(prompt, risk_score, token_count)
