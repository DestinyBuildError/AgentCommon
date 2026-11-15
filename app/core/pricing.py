
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class ModelPricing:

    input_per_mtok: float
    output_per_mtok: float
    cache_read_multiplier: float = 0.1
    cache_write_multiplier: float = 1.25


PRICING: dict[str, ModelPricing] = {
    "claude-opus-4-8": ModelPricing(input_per_mtok=5.0, output_per_mtok=25.0),
}


def register_pricing(model: str, pricing: ModelPricing) -> None:
    PRICING[model.strip().lower()] = pricing


def get_pricing(model: str | None) -> ModelPricing | None:
    if not model:
        return None
    return PRICING.get(model.strip().lower())


def cost_usd(
    model: str | None,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    pricing = get_pricing(model)
    if pricing is None:
        return 0.0
    total = (
        input_tokens * pricing.input_per_mtok
        + output_tokens * pricing.output_per_mtok
        + cache_read_tokens * pricing.input_per_mtok * pricing.cache_read_multiplier
        + cache_creation_tokens * pricing.input_per_mtok * pricing.cache_write_multiplier
    ) / 1_000_000
    return round(total, 6)
