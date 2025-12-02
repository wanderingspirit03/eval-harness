"""Model pricing for cost calculation.

Prices are in USD per 1M tokens as of November 2025.
Sources:
- OpenAI: https://openai.com/api/pricing/
- Anthropic: https://www.anthropic.com/pricing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Pricing per 1M tokens for a model."""

    input_price: float  # USD per 1M input tokens
    output_price: float  # USD per 1M output tokens
    cached_input_price: float | None = None  # USD per 1M cached input tokens


# Model pricing (USD per 1M tokens, November 2025)
MODEL_PRICING: dict[str, ModelPricing] = {
    # ==========================================================================
    # ANTHROPIC CLAUDE MODELS (used by agent-cluster via LiteLLM)
    # ==========================================================================
    # Claude Sonnet 4.5 - balanced performance/cost
    "claude-sonnet-4-5": ModelPricing(3.00, 15.00, 0.30),
    "claude-sonnet-4-5-20250929": ModelPricing(3.00, 15.00, 0.30),
    # Claude Opus 4.5 - highest capability (Nov 2025 release)
    "claude-opus-4-5": ModelPricing(5.00, 25.00, 0.50),
    "claude-opus-4-5-20251124": ModelPricing(5.00, 25.00, 0.50),
    # Claude Opus 4.1 - previous flagship
    "claude-opus-4-1": ModelPricing(15.00, 75.00, 1.50),
    "claude-opus-4-1-20250805": ModelPricing(15.00, 75.00, 1.50),
    # Claude Haiku 4.5 - fast/cheap
    "claude-haiku-4-5": ModelPricing(1.00, 5.00, 0.10),
    "claude-haiku-4-5-20251001": ModelPricing(1.00, 5.00, 0.10),
    # ==========================================================================
    # OPENAI GPT MODELS
    # ==========================================================================
    # GPT-5 series (August 2025 release)
    "gpt-5": ModelPricing(1.25, 10.00, 0.125),
    "gpt-5-2025-08-07": ModelPricing(1.25, 10.00, 0.125),
    "gpt-5-mini": ModelPricing(0.25, 2.00, 0.025),
    "gpt-5-mini-2025-08-07": ModelPricing(0.25, 2.00, 0.025),
    "gpt-5-nano": ModelPricing(0.05, 0.40, 0.005),
    "gpt-5-nano-2025-08-07": ModelPricing(0.05, 0.40, 0.005),
    "gpt-5-codex": ModelPricing(1.25, 10.00, 0.125),
    # GPT-5.1 series (November 2025)
    "gpt-5.1": ModelPricing(1.25, 10.00, 0.125),
    "gpt-5.1-2025-11-13": ModelPricing(1.25, 10.00, 0.125),
    "gpt-5.1-mini": ModelPricing(0.25, 2.00, 0.025),
    "gpt-5.1-mini-2025-11-13": ModelPricing(0.25, 2.00, 0.025),
    # GPT-4o series
    "gpt-4o": ModelPricing(2.50, 10.00, 1.25),
    "gpt-4o-2024-11-20": ModelPricing(2.50, 10.00, 1.25),
    "gpt-4o-2024-08-06": ModelPricing(2.50, 10.00, 1.25),
    "gpt-4o-2024-05-13": ModelPricing(5.00, 15.00, 2.50),
    # GPT-4o mini series
    "gpt-4o-mini": ModelPricing(0.15, 0.60, 0.075),
    "gpt-4o-mini-2024-07-18": ModelPricing(0.15, 0.60, 0.075),
    # o1 series (reasoning models)
    "o1": ModelPricing(15.00, 60.00, 7.50),
    "o1-2024-12-17": ModelPricing(15.00, 60.00, 7.50),
    "o1-preview": ModelPricing(15.00, 60.00, 7.50),
    "o1-mini": ModelPricing(3.00, 12.00, 1.50),
    # o3 series (reasoning models)
    "o3": ModelPricing(10.00, 40.00, 2.50),
    "o3-mini": ModelPricing(1.10, 4.40, 0.55),
    "o3-mini-2025-01-31": ModelPricing(1.10, 4.40, 0.55),
    # o4 series (reasoning models)
    "o4-mini": ModelPricing(1.10, 4.40, 0.275),
    "o4-mini-2025-04-16": ModelPricing(1.10, 4.40, 0.275),
    # Legacy models
    "gpt-4": ModelPricing(30.00, 60.00, None),
    "gpt-4-turbo": ModelPricing(10.00, 30.00, None),
    "gpt-3.5-turbo": ModelPricing(0.50, 1.50, 0.25),
}

# Default pricing for unknown models (use claude-haiku-4-5 as default - cheap and fast)
DEFAULT_PRICING = ModelPricing(1.00, 5.00, 0.10)


def get_model_pricing(model: str) -> ModelPricing:
    """Get pricing for a model, with fallback to default.

    Args:
        model: Model name (e.g., "gpt-4o-mini", "o1-mini")

    Returns:
        ModelPricing with input/output prices per 1M tokens
    """
    # Try exact match first
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    # Try prefix matching for versioned models
    model_lower = model.lower()
    for known_model, pricing in MODEL_PRICING.items():
        if model_lower.startswith(known_model.lower()):
            return pricing

    return DEFAULT_PRICING


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> dict[str, Any]:
    """Calculate USD cost for a model invocation.

    Args:
        model: Model name
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        cached_tokens: Number of cached input tokens (counted separately at discounted rate)

    Returns:
        Dict with:
            - usd: Total cost in USD
            - model: Model name used for pricing lookup
            - breakdown: Detailed cost breakdown
            - tokens: Token counts
    """
    pricing = get_model_pricing(model)

    # Calculate non-cached input tokens
    non_cached_input = max(0, input_tokens - cached_tokens)

    # Cost calculation (prices are per 1M tokens)
    input_cost = (non_cached_input / 1_000_000) * pricing.input_price
    output_cost = (output_tokens / 1_000_000) * pricing.output_price

    cached_cost = 0.0
    if cached_tokens > 0 and pricing.cached_input_price is not None:
        cached_cost = (cached_tokens / 1_000_000) * pricing.cached_input_price

    total_cost = input_cost + output_cost + cached_cost

    return {
        "usd": round(total_cost, 8),  # Keep precision for small costs
        "model": model,
        "pricing_per_1m": {
            "input": pricing.input_price,
            "output": pricing.output_price,
            "cached_input": pricing.cached_input_price,
        },
        "breakdown": {
            "input_cost": round(input_cost, 8),
            "output_cost": round(output_cost, 8),
            "cached_cost": round(cached_cost, 8),
        },
        "tokens": {
            "prompt": input_tokens,
            "completion": output_tokens,
            "cached": cached_tokens,
            "total": input_tokens + output_tokens,
        },
    }


__all__ = [
    "ModelPricing",
    "MODEL_PRICING",
    "DEFAULT_PRICING",
    "get_model_pricing",
    "calculate_cost",
]
