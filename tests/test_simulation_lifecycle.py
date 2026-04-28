"""
Tests for Simulation lifecycle — no LLM calls.

Patches Agent and EventScheduler so we can exercise the run() → log_run_summary()
→ close() sequence that was broken, plus SimulationConfig behaviour.

The core invariant: run() must NOT close the logger. The notebook owns the
logger lifecycle: it calls log_run_summary() then close() after run() returns.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.engine.simulation import Simulation, SimulationConfig
from src.llm.client import Config
from src.agents.retrieval import RetrievalConfig
from src.agents.reflection import ReflectionConfig


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fake_cognition_result(agent_id="test_agent", display_name="TestAgent"):
    return {
        "agent_id": agent_id,
        "agent_display_name": display_name,
        "seed_personality": "A cautious homeowner.",
        "retrieved_memories": [],
        "decision": "Will comply.",
        "reasoning": "Risk reduction.",
        "new_memory_ids": [1],
        "new_reflections": [],
    }


def _make_mock_agent(agent_id="test_agent", display_name="TestAgent"):
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.display_name = display_name
    agent.state_snapshot.return_value = {"agent_id": agent_id, "memory_count": 0}
    agent.run_cognition_cycle.return_value = _fake_cognition_result(agent_id, display_name)
    return agent


def _make_mock_scheduler(duration_days=2, events_per_tick=None):
    """Returns a scheduler mock. events_per_tick: dict of {tick: [events]}."""
    scheduler = MagicMock()
    scheduler.duration_days = duration_days

    def get_events(tick):
        if events_per_tick and tick in events_per_tick:
            return events_per_tick[tick]
        return []

    scheduler.get_events.side_effect = get_events
    return scheduler


def _make_simulation(tmp_path, duration_days=2, with_event=False):
    """Build a Simulation with mocked Agent and EventScheduler."""
    cfg = SimulationConfig(
        run_label="Test_Run",
        scenario_path="config/scenarios/baseline.yaml",
        agent_yaml_paths=["config/agents/selected/beth.yaml"],
        llm_config=Config(),
        output_dir=str(tmp_path),
    )

    mock_agent = _make_mock_agent()
    events_per_tick = {}
    if with_event:
        event = MagicMock()
        event.type = "firewise_outreach"
        event.channel = "social"
        event.content = "Clear your zone 0."
        event.target_agents = "all"
        events_per_tick[1] = [event]

    mock_scheduler = _make_mock_scheduler(duration_days=duration_days, events_per_tick=events_per_tick)

    with patch("src.engine.simulation.Agent") as MockAgent, \
         patch("src.engine.simulation.EventScheduler") as MockScheduler:
        MockAgent.return_value = mock_agent
        MockScheduler.return_value = mock_scheduler
        sim = Simulation(
            sim_config=cfg,
            client_anthropic=MagicMock(),
            client_openrouter=None,
        )

    # Inject mocks directly so step() uses them
    sim.agents = {mock_agent.display_name: mock_agent}
    sim.scheduler = mock_scheduler
    return sim


# ── SimulationConfig ───────────────────────────────────────────────────────────

def test_run_id_auto_generated():
    cfg = SimulationConfig(
        scenario_path="config/scenarios/baseline.yaml",
        agent_yaml_paths=[],
    )
    assert cfg.run_id is not None
    assert cfg.run_label in cfg.run_id
    assert len(cfg.run_id) > 15  # label + _ + YYYYMMDD_HHMMSS


def test_run_label_auto_full():
    cfg = SimulationConfig(
        scenario_path="s", agent_yaml_paths=[],
        use_memory=True, use_reflection=True,
    )
    assert cfg.run_label == "Full"


def test_run_label_auto_no_reflection():
    cfg = SimulationConfig(
        scenario_path="s", agent_yaml_paths=[],
        use_memory=True, use_reflection=False,
    )
    assert cfg.run_label == "No_Reflection"


def test_run_label_auto_no_memory():
    cfg = SimulationConfig(
        scenario_path="s", agent_yaml_paths=[],
        use_memory=False, use_reflection=False,
    )
    assert cfg.run_label == "No_Memory_No_Retrieval_No_Reflection"


def test_run_label_explicit():
    cfg = SimulationConfig(
        scenario_path="s", agent_yaml_paths=[],
        run_label="Baseline_Full",
    )
    assert cfg.run_label == "Baseline_Full"


def test_describe_contains_run_label():
    cfg = SimulationConfig(
        scenario_path="config/scenarios/baseline.yaml",
        agent_yaml_paths=["config/agents/selected/beth.yaml"],
        run_label="Baseline_Full",
    )
    assert "Baseline_Full" in cfg.describe()


def test_describe_contains_model_names():
    cfg = SimulationConfig(
        scenario_path="s", agent_yaml_paths=[],
        llm_config=Config(DECISION_MODEL="claude-sonnet-4-6", REFLECTION_MODEL="claude-haiku-4-5-20251001"),
    )
    desc = cfg.describe()
    assert "claude-sonnet-4-6" in desc
    assert "claude-haiku-4-5-20251001" in desc


# ── Logger lifecycle — THE BUG ─────────────────────────────────────────────────

def test_run_does_not_close_logger(tmp_path):
    """run() must not close the logger — notebook calls log_run_summary() afterwards."""
    sim = _make_simulation(tmp_path, duration_days=2)
    sim.run(verbose=False)
    assert not sim.logger._file.closed


def test_log_run_summary_callable_after_run(tmp_path):
    """The exact notebook sequence that was failing."""
    sim = _make_simulation(tmp_path, duration_days=2)
    sim.run(verbose=False)
    # This should not raise ValueError: I/O operation on closed file
    sim.logger.log_run_summary(latency_seconds=5.0, cost_info={"total_cost_usd": 0.01})
    sim.logger.close()


def test_close_after_run_closes_file(tmp_path):
    sim = _make_simulation(tmp_path, duration_days=2)
    sim.run(verbose=False)
    sim.close()
    assert sim.logger._file.closed


def test_run_summary_appears_in_jsonl(tmp_path):
    import json
    sim = _make_simulation(tmp_path, duration_days=2)
    sim.run(verbose=False)
    sim.logger.log_run_summary(latency_seconds=10.0, cost_info={"total_cost_usd": 0.54})
    sim.close()

    entries = [json.loads(l) for l in sim.logger.log_path.read_text().splitlines() if l.strip()]
    summary = next((e for e in entries if e["entry_type"] == "run_summary"), None)
    assert summary is not None
    assert summary["total_cost_usd"] == 0.54


# ── run() tick loop ────────────────────────────────────────────────────────────

def test_run_returns_tick_results(tmp_path):
    sim = _make_simulation(tmp_path, duration_days=3)
    results = sim.run(verbose=False)
    assert isinstance(results, list)
    assert len(results) > 0


def test_run_advances_tick(tmp_path):
    sim = _make_simulation(tmp_path, duration_days=3)
    sim.run(verbose=False)
    assert sim.current_tick > 0


def test_run_sets_done_flag(tmp_path):
    sim = _make_simulation(tmp_path, duration_days=2)
    sim.run(verbose=False)
    assert sim._done is True


def test_run_processes_events(tmp_path):
    sim = _make_simulation(tmp_path, duration_days=3, with_event=True)
    sim.run(verbose=False)
    events_seen = [
        er
        for tr in sim.tick_results
        for er in tr["events_processed"]
    ]
    assert len(events_seen) == 1
    assert events_seen[0]["event_type"] == "firewise_outreach"


def test_step_returns_none_when_done(tmp_path):
    sim = _make_simulation(tmp_path, duration_days=1)
    # Exhaust the simulation
    while sim.step() is not None:
        pass
    assert sim.step() is None


def test_step_logs_tick_summary(tmp_path):
    import json
    sim = _make_simulation(tmp_path, duration_days=1)
    while sim.step() is not None:
        pass
    sim.close()

    entries = [json.loads(l) for l in sim.logger.log_path.read_text().splitlines() if l.strip()]
    tick_summaries = [e for e in entries if e["entry_type"] == "tick_summary"]
    assert len(tick_summaries) > 0


# ── run_config logged at init ──────────────────────────────────────────────────

def test_run_config_logged_on_init(tmp_path):
    import json
    sim = _make_simulation(tmp_path, duration_days=1)
    sim.close()

    entries = [json.loads(l) for l in sim.logger.log_path.read_text().splitlines() if l.strip()]
    run_config = next((e for e in entries if e["entry_type"] == "run_config"), None)
    assert run_config is not None


def test_run_config_contains_run_label(tmp_path):
    import json
    sim = _make_simulation(tmp_path, duration_days=1)
    sim.close()

    entries = [json.loads(l) for l in sim.logger.log_path.read_text().splitlines() if l.strip()]
    run_config = next(e for e in entries if e["entry_type"] == "run_config")
    assert run_config["run_label"] == "Test_Run"
