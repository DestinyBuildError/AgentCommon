"""模型定价表与成本计算（）。

口径（Anthropic 公开定价，per MTok）：
- claude-opus-4-8: 输入 $5、输出 $25
- cache 读 = 输入价 × 0.1（$0.50/MTok）；cache 写 = 输入价 × 1.25（$6.25/MTok）

注意 usage 口径：Anthropic API 的 input_tokens 不含 cache 部分——
cache_read / cache_creation 是独立字段（见 llm_client.UsageData），
所以四项直接按各自单价相加，不做去重。

未知模型 → 返回 0.0（宁可少记不可虚报），并留 register_pricing 供扩展。
金额统一 round 到 6 位小数（亚分级精度，消除浮点尾噪）。
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class ModelPricing:
    """单个模型的定价（USD per megatok）。"""

    input_per_mtok: float
    output_per_mtok: float
    cache_read_multiplier: float = 0.1    # 相对输入价
    cache_write_multiplier: float = 1.25  # 相对输入价


# Anthropic 公开定价。键名小写；查询时 strip+lower 归一。
PRICING: dict[str, ModelPricing] = {
    "claude-opus-4-8": ModelPricing(input_per_mtok=5.0, output_per_mtok=25.0),
}


def register_pricing(model: str, pricing: ModelPricing) -> None:
    """注册/覆盖一个模型的定价（新模型接入时用，不改代码结构）。"""
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
    """按定价表计算一轮（或一段累计）用量的美元成本。

    未知模型返回 0.0。四项分别计价：
        input × input_per_mtok
        output × output_per_mtok
        cache_read × input_per_mtok × cache_read_multiplier
        cache_creation × input_per_mtok × cache_write_multiplier
    """
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
