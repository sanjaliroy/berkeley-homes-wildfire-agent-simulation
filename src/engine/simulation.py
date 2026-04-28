"""
simulation.py — Tick-based simulation loop.

Thin orchestrator: per tick it asks the scheduler for due events, routes each
event to the right agents (by display_name or 'all'), runs the full cognition
cycle, and logs everything.

All heavy cognition logic lives in agent.py. simulation.py just coordinates:
    scheduler → channels → agents → logger

SimulationConfig is the single place to set the scenario, agents, models, and
ablation flags for one run. The notebook creates a SimulationConfig, passes it
to Simulation(), then either calls run() (full run) or step() (one tick at a
time, for the interactive notebook).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from src.agents.agent import Agent
from src.agents.retrieval import RetrievalConfig
from src.agents.reflection import ReflectionConfig
from src.engine.scheduler import EventScheduler
from src.llm.client import Config
from src.output.logger import SimulationLogger


# ── Run configuration ─────────────────────────────────────────────────────────

@dataclass
class SimulationConfig:
    """
    All knobs for one simulation run — create one of these in the notebook
    and pass it to Simulation().

    The agent_yaml_paths list controls which agents are loaded. Each path
    should point to a YAML file with id, display_name, seed_narrative, and
    memory_seeds fields (same format as beth.yaml / eleanor_v2.yaml).

    Ablation variants (Experiment 1):
        Variant 1: use_memory=True,  use_reflection=True  — Full system
        Variant 2: use_memory=True,  use_reflection=False — Memory + retrieval, no reflection
        Variant 3: use_memory=False, use_reflection=False — Seed personality only (no memory, retrieval, or reflection)

    run_label is the human-readable name used in evaluation exports, e.g.:
        "Premium_Full", "Baseline_Full", "Baseline_No_Reflection",
        "Baseline_No_Memory_No_Retrieval_No_Reflection"
    """

    scenario_path: str                         # path to scenario YAML
    agent_yaml_paths: List[str]                # one path per agent

    # Model and retrieval configuration
    llm_config: Config = field(default_factory=Config)
    retrieval_config: RetrievalConfig = field(default_factory=RetrievalConfig)
    reflection_config: ReflectionConfig = field(default_factory=ReflectionConfig)

    # Ablation flags (Experiment 1)
    use_memory: bool = True
    use_reflection: bool = True

    # Output
    output_dir: str = "outputs/runs"
    run_id: Optional[str] = None       # auto-generated from timestamp if not set
    run_label: Optional[str] = None    # human-readable label for eval exports

    def __post_init__(self):
        if self.run_label is None:
            # Auto-derive label from flags if not provided
            if self.use_memory and self.use_reflection:
                self.run_label = "Full"
            elif self.use_memory:
                self.run_label = "No_Reflection"
            else:
                self.run_label = "No_Memory_No_Retrieval_No_Reflection"
        if self.run_id is None:
            self.run_id = f"{self.run_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def describe(self) -> str:
        """Human-readable summary for notebook display."""
        if self.use_memory and self.use_reflection:
            variant = "Variant 1 — Full system (memory + retrieval + reflection)"
        elif self.use_memory:
            variant = "Variant 2 — Memory + retrieval, no reflection"
        else:
            variant = "Variant 3 — Seed personality only (no memory, no retrieval, no reflection)"
        return (
            f"Run ID:    {self.run_id}\n"
            f"Run label: {self.run_label}\n"
            f"Agents:    {[p.split('/')[-1] for p in self.agent_yaml_paths]}\n"
            f"Scenario:  {self.scenario_path.split('/')[-1]}\n"
            f"Ablation:  {variant}\n"
            f"Decision model:   {self.llm_config.DECISION_MODEL}\n"
            f"Reflection model: {self.llm_config.REFLECTION_MODEL}\n"
            f"Concise output:   {self.llm_config.CONCISE_OUTPUT}"
        )


# ── Simulation ────────────────────────────────────────────────────────────────

class Simulation:
    """
    Runs the tick loop and coordinates agents, scheduler, and logger.

    Two ways to run from the notebook:
        sim.run()     — full run, prints a summary after each event-filled tick
        sim.step()    — advance exactly one day; returns the tick result dict
                        (use this for step-by-step inspection in the notebook)

    After the run, sim.tick_results holds every tick result in memory for
    the notebook analysis helpers (compare_agents, inspect_memory, etc.).
    """

    def __init__(
        self,
        sim_config: SimulationConfig,
        client_anthropic,
        client_openrouter=None,
    ):
        self.sim_config = sim_config
        self.client_anthropic = client_anthropic
        self.client_openrouter = client_openrouter

        print(f"\n{'='*60}")
        print(f"  Initialising simulation: {sim_config.run_id}")
        print(f"{'='*60}")

        # Scheduler reads the scenario YAML
        self.scheduler = EventScheduler(sim_config.scenario_path)

        # Load agents — keyed by display_name for target_agents matching
        self.agents: Dict[str, Agent] = {}
        for yaml_path in sim_config.agent_yaml_paths:
            agent = Agent(
                yaml_path=yaml_path,
                client_anthropic=client_anthropic,
                llm_config=sim_config.llm_config,
                retrieval_config=sim_config.retrieval_config,
                reflection_config=sim_config.reflection_config,
                use_memory=sim_config.use_memory,
                use_reflection=sim_config.use_reflection,
                client_openrouter=client_openrouter,
            )
            self.agents[agent.display_name] = agent
            agent.pretty_print()

        # Logger
        self.logger = SimulationLogger(
            run_id=sim_config.run_id,
            output_dir=sim_config.output_dir,
        )
        # Log the run configuration as the first JSONL entry for reproducibility
        self.logger.log_run_config({
            "run_label":       sim_config.run_label,
            "scenario_path":   sim_config.scenario_path,
            "agents":          list(self.agents.keys()),
            "use_memory":      sim_config.use_memory,
            "use_reflection":  sim_config.use_reflection,
            "decision_model":  sim_config.llm_config.DECISION_MODEL,
            "reflection_model": sim_config.llm_config.REFLECTION_MODEL,
            "concise_output":  sim_config.llm_config.CONCISE_OUTPUT,
        })

        # Runtime state
        self.current_tick: int = 0
        self.tick_results: List[dict] = []   # in-memory for notebook inspection
        self._done: bool = False

        print(f"\n  Agents loaded: {list(self.agents.keys())}")
        print(f"  Duration: {self.scheduler.duration_days} days")
        print(f"{'='*60}\n")

    # ── Event routing ─────────────────────────────────────────────────────────

    def _resolve_targets(self, target_agents) -> List[Agent]:
        """
        Map the target_agents field from the scenario YAML to loaded Agent objects.

        'all'  → every loaded agent
        'Name' → the agent whose display_name matches (case-sensitive)
        'A, B' → multiple agents by display_name
        """
        target_str = str(target_agents).strip()
        if target_str.lower() == "all":
            return list(self.agents.values())

        names = [n.strip() for n in target_str.split(",")]
        targets = []
        for name in names:
            if name in self.agents:
                targets.append(self.agents[name])
            else:
                print(
                    f"  [sim] Warning: target '{name}' not loaded. "
                    f"Available: {list(self.agents.keys())}"
                )
        return targets

    # ── Tick execution ────────────────────────────────────────────────────────

    def step(self) -> Optional[dict]:
        """
        Advance exactly one simulation day.

        Returns a tick_result dict for notebook inspection if there were events
        this tick, or a lightweight dict with just tick and agent_states otherwise.
        Returns None when the simulation is complete.

        This is the method to call from the notebook for step-by-step viewing —
        call it in a loop or cell-by-cell to watch the simulation unfold.
        """
        if self._done or self.current_tick > self.scheduler.duration_days:
            self._done = True
            return None

        tick = self.current_tick
        events = self.scheduler.get_events(tick)

        tick_result: dict = {
            "tick": tick,
            "events_processed": [],
            "agent_states": [],
        }

        for event in events:
            targets = self._resolve_targets(event.target_agents)

            event_result = {
                "event_type": event.type,
                "channel": event.channel,
                "content": event.content,
                "target_display_names": [a.display_name for a in targets],
                "agent_responses": [],
            }

            for agent in targets:
                # Full cognition cycle — perceive → retrieve → decide → store → reflect
                result = agent.run_cognition_cycle(event.__dict__, tick)

                # Write decision entry to JSONL
                self.logger.log_decision(
                    tick=tick,
                    agent_id=result["agent_id"],
                    agent_display_name=result["agent_display_name"],
                    event_type=event.type,
                    channel=event.channel,
                    intervention=event.content,
                    seed_personality=result["seed_personality"],
                    retrieved_memories=result["retrieved_memories"],
                    decision=result["decision"],
                    reasoning=result["reasoning"],
                    new_memory_ids=result["new_memory_ids"],
                )

                # Write reflection entry if one fired this cycle
                if result["new_reflections"]:
                    self.logger.log_reflection(
                        tick=tick,
                        agent_id=result["agent_id"],
                        agent_display_name=result["agent_display_name"],
                        reflections=result["new_reflections"],
                    )

                event_result["agent_responses"].append(result)

            tick_result["events_processed"].append(event_result)

        # End-of-tick state snapshot for all agents
        agent_states = [a.state_snapshot() for a in self.agents.values()]
        self.logger.log_tick_summary(tick, agent_states)
        tick_result["agent_states"] = agent_states

        self.tick_results.append(tick_result)
        self.current_tick += 1

        return tick_result

    def run(self, verbose: bool = True) -> List[dict]:
        """
        Run the full simulation from current tick to end.

        Use step() from the notebook for interactive step-through.
        Use run() when you want a complete background run (e.g. for ablation experiments).

        Returns the list of tick results (same as self.tick_results).
        """
        while not self._done:
            tick_result = self.step()
            if tick_result is None:
                break
            if verbose and tick_result["events_processed"]:
                _print_tick_summary(tick_result)

        print(f"\n  Simulation '{self.sim_config.run_id}' complete.")
        print(f"  {self.current_tick} days simulated, "
              f"{len(self.tick_results)} ticks with events.")
        return self.tick_results

    def close(self):
        """Close the logger. Call after run() completes and any final log entries are written."""
        self.logger.close()


# ── Display helpers ───────────────────────────────────────────────────────────

def _print_tick_summary(tick_result: dict):
    """Print a readable summary of one tick — used by run() verbose mode."""
    tick = tick_result["tick"]
    for er in tick_result["events_processed"]:
        print(f"\n{'─'*60}")
        print(f"  Day {tick} | {er['event_type'].upper()} | channel: {er['channel']}")
        print(f"  → {er['content'][:100]}...")
        for resp in er["agent_responses"]:
            name = resp["agent_display_name"]
            dec = resp["decision"] or "(no structured decision parsed)"
            rea = resp["reasoning"]
            refs = resp["new_reflections"]
            print(f"\n  [{name}]")
            print(f"    DECISION:  {dec[:110]}")
            print(f"    REASONING: {rea[:110]}")
            if refs:
                print(f"    ★ REFLECTION fired — {len(refs)} new insight(s)")
