"""定价表与成本计算测试（）：claude-opus-4-8 基准价、
cache 读 0.1x / 写 1.25x 权重、未知模型 0.0、register_pricing 扩展。
"""

from __future__ import annotations

import pytest

from app.core.pricing import (
    PRICING,
    ModelPricing,
    cost_usd,
    get_pricing,
    register_pricing,
)


def test_opus_pricing_table_public_values() -> None:
    """Anthropic 公开定价：输入 $5 / 输出 $25 per MTok；cache 权重 0.1x / 1.25x。"""
    pricing = get_pricing("claude-opus-4-8")
    assert pricing == ModelPricing(
        input_per_mtok=5.0, output_per_mtok=25.0,
        cache_read_multiplier=0.1, cache_write_multiplier=1.25,
    )
    # 键名归一：大小写/空白不敏感
    assert get_pricing(" Claude-Opus-4-8 ") is pricing


def test_cost_base_input_output() -> None:
    """纯输入输出：1M 输入 + 0.2M 输出 = 5 + 0.2*25 = $10。"""
    cost = cost_usd(
        "claude-opus-4-8", input_tokens=1_000_000, output_tokens=200_000
    )
    assert cost == pytest.approx(10.0)


def test_cost_cache_read_at_01x() -> None:
    """cache 读按输入价 0.1x：1M cache_read = $0.5。"""
    cost = cost_usd("claude-opus-4-8", cache_read_tokens=1_000_000)
    assert cost == pytest.approx(0.5)


def test_cost_cache_write_at_125x() -> None:
    """cache 写按输入价 1.25x：1M cache_creation = $6.25。"""
    cost = cost_usd("claude-opus-4-8", cache_creation_tokens=1_000_000)
    assert cost == pytest.approx(6.25)


def test_cost_combined_with_cache_weights() -> None:
    """组合口径（一轮真实用量形态）：
    input 10k@5 + output 2k@25 + cache_read 100k@0.5 + cache_creation 8k@6.25（每 MTok）
    = 0.05 + 0.05 + 0.05 + 0.05 = 0.2
    """
    cost = cost_usd(
        "claude-opus-4-8",
        input_tokens=10_000,
        output_tokens=2_000,
        cache_read_tokens=100_000,
        cache_creation_tokens=8_000,
    )
    expected = (
        10_000 * 5.0 + 2_000 * 25.0 + 100_000 * 0.5 + 8_000 * 6.25
    ) / 1_000_000
    assert cost == pytest.approx(expected)
    assert cost == pytest.approx(0.2)


def test_unknown_model_costs_zero() -> None:
    """未知模型宁可少记不可虚报 → 0.0。"""
    assert cost_usd("gpt-99", input_tokens=1_000_000) == 0.0
    assert cost_usd(None, input_tokens=1_000_000) == 0.0
    assert cost_usd("") == 0.0


def test_zero_usage_costs_zero() -> None:
    assert cost_usd("claude-opus-4-8") == 0.0


def test_register_pricing_extends_table() -> None:
    """新模型接入：注册后立即可计价；不污染原有条目。"""
    model = "test-model-x"
    try:
        register_pricing(model, ModelPricing(input_per_mtok=1.0, output_per_mtok=2.0))
        assert cost_usd(model, input_tokens=1_000_000, output_tokens=1_000_000) == 3.0
        # cache 权重继承默认 0.1x / 1.25x
        assert cost_usd(model, cache_read_tokens=1_000_000) == pytest.approx(0.1)
        assert get_pricing("claude-opus-4-8") is not None
    finally:
        PRICING.pop(model, None)
