"""
logger.py — Structured JSONL logging for simulation runs.

Writes one JSON object per line to a .jsonl file. Each entry has an entry_type
field so you can filter by type when analysing results:

  "decision"     — one per agent per event; contains all fields the judge needs
  "reflection"   — fired when cumulative importance exceeds threshold
  "tick_summary" — end-of-tick state snapshot for all agents

The four fields required for judge_simulation() calls are always present in
every "decision" entry:
  seed_personality   — the agent's full seed paragraph
  intervention       — the exact event content (channel is also logged separately)
  decision           — what the agent decided to do
  reasoning          — why (the agent's internal logic)
"""

import json
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.memory import Memory


class SimulationLogger:
    """
    Writes structured JSONL output for one simulation run.

    Use as a context manager so the file is always closed cleanly:

        with SimulationLogger(run_id="my_run") as logger:
            logger.log_decision(...)

    Or call close() manually after the run completes.
    """

    def __init__(self, run_id: str, output_dir: str = "outputs/runs"):
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.output_dir / f"{run_id}.jsonl"
        self._file = open(self.log_path, "w", encoding="utf-8")
        self._entry_count = 0

        print(f"[logger] Run '{run_id}' → {self.log_path}")

    # ── Primary log methods ────────────────────────────────────────────────────

    def log_decision(
        self,
        tick: int,
        agent_id: str,
        agent_display_name: str,
        event_type: str,
        channel: str,
        intervention: str,           # raw event content (judge reads this)
        seed_personality: str,       # agent's full seed narrative (judge reads this)
        retrieved_memories: List["Memory"],
        decision: str,               # what the agent does (judge reads this)
        reasoning: str,              # why (judge reads this)
        new_memory_ids: List[int],
    ):
        """
        Log one agent decision. This is the primary record for judge scoring.

        All four judge-required fields (seed_personality, intervention, decision,
        reasoning) are always written here, even if some are empty strings.
        """
        entry = {
            "entry_type": "decision",
            "tick": tick,
            "agent_id": agent_id,
            "agent_display_name": agent_display_name,
            "event_type": event_type,
            "channel": channel,
            # ── Judge-required fields ──────────────────────────────────────────
            "seed_personality": seed_personality,
            "intervention": intervention,
            "decision": decision,
            "reasoning": reasoning,
            # ── Supporting context ─────────────────────────────────────────────
            "retrieved_memories": [
                {
                    "id": m.id,
                    "timestamp": m.timestamp,
                    "type": m.type,
                    "importance": m.importance,
                    "description": m.description,
                }
                for m in retrieved_memories
            ],
            "new_memory_ids": new_memory_ids,
        }
        self._write(entry)

    def log_reflection(
        self,
        tick: int,
        agent_id: str,
        agent_display_name: str,
        reflections: List["Memory"],
    ):
        """
        Log a reflection event — fired when cumulative importance exceeds threshold.
        Reflections become high-importance memories that re-enter the retrieval stream.
        """
        entry = {
            "entry_type": "reflection",
            "tick": tick,
            "agent_id": agent_id,
            "agent_display_name": agent_display_name,
            "reflections": [
                {
                    "id": m.id,
                    "description": m.description,
                    "importance": m.importance,
                }
                for m in reflections
            ],
            "reflection_count": len(reflections),
        }
        self._write(entry)

    def log_tick_summary(self, tick: int, agent_states: List[dict]):
        """
        Log an end-of-tick state snapshot for all agents.
        Useful for tracking how agent state evolves across the simulation.
        """
        entry = {
            "entry_type": "tick_summary",
            "tick": tick,
            "agent_states": agent_states,
        }
        self._write(entry)

    def log_run_config(self, config_dict: dict):
        """Log the full run configuration as the first entry — enables reproducibility."""
        entry = {
            "entry_type": "run_config",
            "tick": -1,
            **config_dict,
        }
        self._write(entry)

    def log_run_summary(self, latency_seconds: float, cost_info: dict):
        """
        Log end-of-run cost and latency as the final JSONL entry.

        latency_seconds: wall-clock time for the full simulation run.
        cost_info: dict from UsageTracker.to_dict() — agent/judge token counts and costs.
        """
        entry = {
            "entry_type": "run_summary",
            "tick": -2,
            "latency_seconds": round(latency_seconds, 2),
            **cost_info,
        }
        self._write(entry)

    # ── Internal write ─────────────────────────────────────────────────────────

    def _write(self, entry: dict):
        """Serialise entry to JSON and append to the log file."""
        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._file.flush()   # flush after every write so partial runs are readable
        self._entry_count += 1

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def close(self):
        self._file.close()
        print(f"[logger] Closed. {self._entry_count} entries written to {self.log_path}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
