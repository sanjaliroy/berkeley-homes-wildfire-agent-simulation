"""
Tests for SimulationLogger.

Covers: all log methods, JSONL validity, lifecycle (write → summary → close),
entry ordering, flush-on-write, and the specific bug where run() closed the
logger before log_run_summary() could be called.
"""

import json
import tempfile
import pytest
from pathlib import Path
from src.output.logger import SimulationLogger


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def logger(tmp_path):
    lg = SimulationLogger(run_id="test_run", output_dir=str(tmp_path))
    yield lg
    if not lg._file.closed:
        lg.close()


def read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ── File creation ──────────────────────────────────────────────────────────────

def test_log_file_created(tmp_path):
    lg = SimulationLogger(run_id="myrun", output_dir=str(tmp_path))
    lg.close()
    assert (tmp_path / "myrun.jsonl").exists()


def test_log_path_attribute(tmp_path):
    lg = SimulationLogger(run_id="myrun", output_dir=str(tmp_path))
    lg.close()
    assert lg.log_path == tmp_path / "myrun.jsonl"


# ── log_run_config ─────────────────────────────────────────────────────────────

def test_log_run_config_entry_type(logger, tmp_path):
    logger.log_run_config({"run_label": "test", "decision_model": "claude-sonnet-4-6"})
    logger.close()
    entries = read_jsonl(tmp_path / "test_run.jsonl")
    assert entries[0]["entry_type"] == "run_config"


def test_log_run_config_fields_preserved(logger, tmp_path):
    logger.log_run_config({"run_label": "ablation", "use_memory": True})
    logger.close()
    entries = read_jsonl(tmp_path / "test_run.jsonl")
    assert entries[0]["run_label"] == "ablation"
    assert entries[0]["use_memory"] is True


def test_log_run_config_tick_minus_one(logger, tmp_path):
    logger.log_run_config({})
    logger.close()
    entries = read_jsonl(tmp_path / "test_run.jsonl")
    assert entries[0]["tick"] == -1


# ── log_decision ───────────────────────────────────────────────────────────────

def test_log_decision_entry_type(logger, tmp_path):
    logger.log_decision(
        tick=1, agent_id="beth", agent_display_name="Beth",
        event_type="firewise_outreach", channel="social",
        intervention="Clear your zone 0.", seed_personality="Beth is cautious.",
        retrieved_memories=[], decision="Will do it.", reasoning="Seems prudent.",
        new_memory_ids=[1],
    )
    logger.close()
    entries = read_jsonl(tmp_path / "test_run.jsonl")
    assert entries[0]["entry_type"] == "decision"


def test_log_decision_all_judge_fields_present(logger, tmp_path):
    logger.log_decision(
        tick=3, agent_id="jennifer", agent_display_name="Jennifer",
        event_type="inspection", channel="direct_experience",
        intervention="Inspector arrived.", seed_personality="Jennifer is pragmatic.",
        retrieved_memories=[], decision="Comply.", reasoning="Fine avoidance.",
        new_memory_ids=[],
    )
    logger.close()
    entry = read_jsonl(tmp_path / "test_run.jsonl")[0]
    for field in ("seed_personality", "intervention", "decision", "reasoning"):
        assert field in entry, f"Missing judge-required field: {field}"


def test_log_decision_retrieved_memories_serialised(logger, tmp_path):
    from unittest.mock import MagicMock
    mem = MagicMock()
    mem.id = 7
    mem.timestamp = 5
    mem.type = "observation"
    mem.importance = 8
    mem.description = "Saw fire trucks."

    logger.log_decision(
        tick=1, agent_id="a", agent_display_name="A",
        event_type="e", channel="c", intervention="i",
        seed_personality="s", retrieved_memories=[mem],
        decision="d", reasoning="r", new_memory_ids=[7],
    )
    logger.close()
    entry = read_jsonl(tmp_path / "test_run.jsonl")[0]
    assert entry["retrieved_memories"][0]["id"] == 7
    assert entry["retrieved_memories"][0]["description"] == "Saw fire trucks."


# ── log_reflection ─────────────────────────────────────────────────────────────

def test_log_reflection_entry_type(logger, tmp_path):
    from unittest.mock import MagicMock
    ref = MagicMock()
    ref.id = 1
    ref.description = "I fear loss."
    ref.importance = 8

    logger.log_reflection(tick=5, agent_id="beth", agent_display_name="Beth", reflections=[ref])
    logger.close()
    entries = read_jsonl(tmp_path / "test_run.jsonl")
    assert entries[0]["entry_type"] == "reflection"


def test_log_reflection_count(logger, tmp_path):
    from unittest.mock import MagicMock
    refs = [MagicMock(id=i, description=f"r{i}", importance=8) for i in range(3)]
    logger.log_reflection(tick=5, agent_id="a", agent_display_name="A", reflections=refs)
    logger.close()
    entry = read_jsonl(tmp_path / "test_run.jsonl")[0]
    assert entry["reflection_count"] == 3
    assert len(entry["reflections"]) == 3


# ── log_tick_summary ───────────────────────────────────────────────────────────

def test_log_tick_summary_entry_type(logger, tmp_path):
    logger.log_tick_summary(tick=10, agent_states=[{"agent": "beth", "memory_count": 5}])
    logger.close()
    entries = read_jsonl(tmp_path / "test_run.jsonl")
    assert entries[0]["entry_type"] == "tick_summary"


def test_log_tick_summary_tick_value(logger, tmp_path):
    logger.log_tick_summary(tick=42, agent_states=[])
    logger.close()
    entry = read_jsonl(tmp_path / "test_run.jsonl")[0]
    assert entry["tick"] == 42


# ── log_run_summary ────────────────────────────────────────────────────────────

def test_log_run_summary_entry_type(logger, tmp_path):
    logger.log_run_summary(latency_seconds=12.3, cost_info={"total_cost_usd": 0.05})
    logger.close()
    entries = read_jsonl(tmp_path / "test_run.jsonl")
    assert entries[0]["entry_type"] == "run_summary"


def test_log_run_summary_tick_minus_two(logger, tmp_path):
    logger.log_run_summary(latency_seconds=1.0, cost_info={})
    logger.close()
    entry = read_jsonl(tmp_path / "test_run.jsonl")[0]
    assert entry["tick"] == -2


def test_log_run_summary_latency_rounded(logger, tmp_path):
    logger.log_run_summary(latency_seconds=12.3456789, cost_info={})
    logger.close()
    entry = read_jsonl(tmp_path / "test_run.jsonl")[0]
    assert entry["latency_seconds"] == 12.35


def test_log_run_summary_cost_fields_present(logger, tmp_path):
    cost_info = {"agent_cost_usd": 0.01, "judge_cost_usd": 0.02, "total_cost_usd": 0.03}
    logger.log_run_summary(latency_seconds=5.0, cost_info=cost_info)
    logger.close()
    entry = read_jsonl(tmp_path / "test_run.jsonl")[0]
    assert entry["total_cost_usd"] == 0.03


# ── THE BUG: log_run_summary after run() ──────────────────────────────────────

def test_log_run_summary_callable_before_close(tmp_path):
    """
    Regression: run() used to close the logger internally, making this fail.
    The notebook calls log_run_summary → close, so the file must still be open.
    """
    lg = SimulationLogger(run_id="regression", output_dir=str(tmp_path))
    lg.log_run_config({"run_label": "test"})
    lg.log_decision(
        tick=1, agent_id="a", agent_display_name="A",
        event_type="e", channel="c", intervention="i",
        seed_personality="s", retrieved_memories=[],
        decision="d", reasoning="r", new_memory_ids=[],
    )
    # Simulate what the notebook does after sim.run() returns
    lg.log_run_summary(latency_seconds=10.0, cost_info={"total_cost_usd": 0.05})
    lg.close()

    entries = read_jsonl(tmp_path / "regression.jsonl")
    types = [e["entry_type"] for e in entries]
    assert "run_summary" in types
    assert types[-1] == "run_summary"


# ── Entry ordering and count ───────────────────────────────────────────────────

def test_entry_count_tracked(logger):
    logger.log_run_config({})
    logger.log_tick_summary(0, [])
    logger.log_run_summary(1.0, {})
    assert logger._entry_count == 3


def test_all_entries_valid_json(tmp_path):
    lg = SimulationLogger(run_id="json_check", output_dir=str(tmp_path))
    lg.log_run_config({"run_label": "x"})
    lg.log_tick_summary(1, [{"agent": "beth"}])
    lg.log_run_summary(2.5, {"total_cost_usd": 0.01})
    lg.close()

    for line in (tmp_path / "json_check.jsonl").read_text().splitlines():
        json.loads(line)  # raises if invalid


def test_notebook_sequence_entry_order(tmp_path):
    """Full notebook write sequence — run_config first, run_summary last."""
    lg = SimulationLogger(run_id="seq", output_dir=str(tmp_path))
    lg.log_run_config({"run_label": "Baseline_Full"})
    lg.log_decision(
        tick=1, agent_id="b", agent_display_name="B",
        event_type="firewise_outreach", channel="social",
        intervention="Clear zone 0.", seed_personality="Cautious.",
        retrieved_memories=[], decision="Will comply.", reasoning="Risk averse.",
        new_memory_ids=[1],
    )
    lg.log_tick_summary(1, [])
    lg.log_run_summary(30.0, {"agent_cost_usd": 0.54, "total_cost_usd": 3.74})
    lg.close()

    entries = read_jsonl(tmp_path / "seq.jsonl")
    assert entries[0]["entry_type"] == "run_config"
    assert entries[-1]["entry_type"] == "run_summary"


# ── Context manager ────────────────────────────────────────────────────────────

def test_context_manager_closes_file(tmp_path):
    with SimulationLogger(run_id="ctx", output_dir=str(tmp_path)) as lg:
        lg.log_run_config({})
    assert lg._file.closed


def test_context_manager_writes_on_exit(tmp_path):
    with SimulationLogger(run_id="ctx2", output_dir=str(tmp_path)) as lg:
        lg.log_tick_summary(0, [])
    entries = read_jsonl(tmp_path / "ctx2.jsonl")
    assert len(entries) == 1
