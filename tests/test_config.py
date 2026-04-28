"""
Tests for Config defaults, model pricing, and UsageTracker.

No LLM calls — purely unit tests against static values and arithmetic.
"""

import pytest
from src.llm.client import Config, UsageTracker, MODEL_PRICING


# ── Config defaults ────────────────────────────────────────────────────────────

def test_default_decision_model():
    assert Config().DECISION_MODEL == "claude-sonnet-4-6"


def test_default_reflection_model_is_haiku():
    assert Config().REFLECTION_MODEL == "claude-haiku-4-5-20251001"


def test_default_judge_model_is_opus():
    assert Config().JUDGE_MODEL == "claude-opus-4-6"


def test_reflection_question_max_tokens_exists():
    assert hasattr(Config(), "REFLECTION_QUESTION_MAX_TOKENS")


def test_reflection_question_max_tokens_value():
    assert Config().REFLECTION_QUESTION_MAX_TOKENS == 2048


def test_reflection_max_tokens_value():
    assert Config().REFLECTION_MAX_TOKENS == 512


def test_decision_max_tokens_value():
    assert Config().DECISION_MAX_TOKENS == 1024


def test_concise_output_default_false():
    assert Config().CONCISE_OUTPUT is False


def test_config_override_decision_model():
    cfg = Config(DECISION_MODEL="claude-haiku-4-5-20251001")
    assert cfg.DECISION_MODEL == "claude-haiku-4-5-20251001"


def test_config_override_reflection_model():
    cfg = Config(REFLECTION_MODEL="claude-sonnet-4-6")
    assert cfg.REFLECTION_MODEL == "claude-sonnet-4-6"


def test_config_override_does_not_affect_defaults():
    cfg = Config(DECISION_MODEL="claude-haiku-4-5-20251001")
    assert cfg.REFLECTION_MODEL == "claude-haiku-4-5-20251001"  # default unchanged


def test_config_concise_output_override():
    cfg = Config(CONCISE_OUTPUT=True)
    assert cfg.CONCISE_OUTPUT is True


# ── Model pricing ──────────────────────────────────────────────────────────────

def test_sonnet_pricing():
    assert MODEL_PRICING["claude-sonnet-4-6"] == (3.00, 15.00)


def test_opus_pricing_correct():
    # Was wrong in code ($15/$75), verified against Anthropic docs: $5/$25
    assert MODEL_PRICING["claude-opus-4-6"] == (5.00, 25.00)


def test_haiku_pricing_correct():
    # Was wrong in code ($0.80/$4.00), actual: $1/$5
    assert MODEL_PRICING["claude-haiku-4-5-20251001"] == (1.00, 5.00)


def test_all_pricing_entries_are_positive():
    for model, (inp, out) in MODEL_PRICING.items():
        assert inp > 0, f"{model} input price not positive"
        assert out > 0, f"{model} output price not positive"


def test_all_pricing_entries_are_floats():
    for model, (inp, out) in MODEL_PRICING.items():
        assert isinstance(inp, float), f"{model} input price not float"
        assert isinstance(out, float), f"{model} output price not float"


# ── UsageTracker ───────────────────────────────────────────────────────────────

def test_tracker_starts_at_zero():
    t = UsageTracker()
    assert t.agent_tokens_in == 0
    assert t.agent_tokens_out == 0
    assert t.judge_tokens_in == 0
    assert t.judge_tokens_out == 0


def test_tracker_add_agent():
    t = UsageTracker()
    t.add("agent", 1000, 200)
    assert t.agent_tokens_in == 1000
    assert t.agent_tokens_out == 200
    assert t.judge_tokens_in == 0


def test_tracker_add_judge():
    t = UsageTracker()
    t.add("judge", 500, 100)
    assert t.judge_tokens_in == 500
    assert t.judge_tokens_out == 100
    assert t.agent_tokens_in == 0


def test_tracker_add_accumulates():
    t = UsageTracker()
    t.add("agent", 1000, 200)
    t.add("agent", 500, 50)
    assert t.agent_tokens_in == 1500
    assert t.agent_tokens_out == 250


def test_tracker_reset():
    t = UsageTracker()
    t.add("agent", 1000, 200)
    t.add("judge", 500, 100)
    t.reset()
    assert t.agent_tokens_in == 0
    assert t.agent_tokens_out == 0
    assert t.judge_tokens_in == 0
    assert t.judge_tokens_out == 0


def test_tracker_cost_calculation_sonnet():
    t = UsageTracker()
    t.add("agent", 1_000_000, 0)  # 1M input tokens at $3/M
    result = t.to_dict(agent_model="claude-sonnet-4-6")
    assert abs(result["agent_cost_usd"] - 3.00) < 0.001


def test_tracker_cost_calculation_haiku_output():
    t = UsageTracker()
    t.add("agent", 0, 1_000_000)  # 1M output tokens at $5/M
    result = t.to_dict(agent_model="claude-haiku-4-5-20251001")
    assert abs(result["agent_cost_usd"] - 5.00) < 0.001


def test_tracker_cost_calculation_opus_judge():
    t = UsageTracker()
    t.add("judge", 1_000_000, 0)  # 1M input at $5/M
    result = t.to_dict(agent_model="claude-sonnet-4-6", judge_model="claude-opus-4-6")
    assert abs(result["judge_cost_usd"] - 5.00) < 0.001


def test_tracker_total_cost_is_sum():
    t = UsageTracker()
    t.add("agent", 100_000, 10_000)
    t.add("judge", 50_000, 5_000)
    result = t.to_dict(agent_model="claude-sonnet-4-6", judge_model="claude-opus-4-6")
    assert abs(result["total_cost_usd"] - (result["agent_cost_usd"] + result["judge_cost_usd"])) < 1e-9


def test_tracker_unknown_model_zero_cost():
    t = UsageTracker()
    t.add("agent", 1_000_000, 1_000_000)
    result = t.to_dict(agent_model="unknown/model-xyz")
    assert result["agent_cost_usd"] == 0.0


def test_tracker_to_dict_has_all_fields():
    t = UsageTracker()
    result = t.to_dict(agent_model="claude-sonnet-4-6")
    for key in ("agent_model", "agent_tokens_in", "agent_tokens_out", "agent_cost_usd",
                "judge_tokens_in", "judge_tokens_out", "judge_cost_usd", "total_cost_usd"):
        assert key in result, f"Missing key: {key}"


def test_tracker_to_dict_agent_model_recorded():
    t = UsageTracker()
    result = t.to_dict(agent_model="claude-haiku-4-5-20251001")
    assert result["agent_model"] == "claude-haiku-4-5-20251001"
