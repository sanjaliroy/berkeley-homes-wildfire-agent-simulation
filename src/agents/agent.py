"""
agent.py — Agent class: the full Park et al. (2023) cognition cycle for one homeowner.

Each Agent wraps:
  - A MemoryStream (append-only, grows through the simulation)
  - Retrieval (top-K scoring via retrieval.py)
  - Reflection (periodic belief synthesis via reflection.py)
  - Decision-making (LLM call via client.py, structured JSON output)

The cognition cycle per event is:
    perceive → retrieve → decide → store → maybe_reflect

Ablation flags let you disable memory and/or reflection for Experiment 1:
    use_memory=False      → agent gets no retrieved memories (Variant 3)
    use_reflection=False  → reflection step is skipped (Variant 2)
    Both True             → full system (Variant 1)
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

import yaml

from src.agents.memory import MemoryStream
from src.agents.retrieval import RetrievalConfig, retrieve_memories
from src.agents.reflection import ReflectionConfig, maybe_reflect
from src.agents.prompts import build_system_prompt, build_decision_prompt
from src.environment.channels import frame_event
from src.llm.client import Config, decide, embed, score_importance

if TYPE_CHECKING:
    import numpy as np
    from src.agents.memory import Memory

# Memory types allowed by MemoryStream — used to sanitise YAML seeds
_VALID_MEMORY_TYPES = {"observation", "decision", "reflection", "conversation"}


class Agent:
    """
    A single simulated Berkeley Hills homeowner.

    Loads identity + memory seeds from a YAML file (same format as beth.yaml /
    eleanor_v2.yaml). Runs the full cognition cycle when given an event.

    Args:
        yaml_path:          Path to the agent's YAML file.
        client_anthropic:   Anthropic client (from client.init_clients()).
        llm_config:         Config dataclass controlling which models to use.
        retrieval_config:   RetrievalConfig (top-K weights etc.). Defaults applied if None.
        reflection_config:  ReflectionConfig (threshold, num_questions). Defaults if None.
        use_memory:         Ablation flag — False disables retrieval (Experiment 1 Variant 3).
        use_reflection:     Ablation flag — False disables reflection (Experiment 1 Variant 2).
        client_openrouter:  Optional OpenRouter client for non-Claude models.
    """

    def __init__(
        self,
        yaml_path: str,
        client_anthropic,
        llm_config: Config,
        retrieval_config: Optional[RetrievalConfig] = None,
        reflection_config: Optional[ReflectionConfig] = None,
        use_memory: bool = True,
        use_reflection: bool = True,
        client_openrouter=None,
    ):
        self.client_anthropic = client_anthropic
        self.client_openrouter = client_openrouter
        self.llm_config = llm_config
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.reflection_config = reflection_config or ReflectionConfig()
        self.use_memory = use_memory
        self.use_reflection = use_reflection

        # Load agent data from YAML
        data = _load_agent_yaml(yaml_path)
        self.id: str = data["id"]
        self.display_name: str = data["display_name"]
        self.seed_narrative: str = data["seed_narrative"]
        self._raw_data: dict = data   # kept for reference (persona, key_concerns etc.)

        # Runtime state — updated as the simulation progresses
        self.memory = MemoryStream(agent_name=self.id)
        self._last_reflection_index: int = 0   # tracks where last reflection ended
        self.compliance_status: str = data.get("compliance_status", "unknown")
        self.attitude: str = "neutral"
        self.actions_taken: List[str] = []

        # Seed memory stream from YAML
        self._load_seeds(data.get("memory_seeds", []))

    # ── Initialisation ────────────────────────────────────────────────────────

    def _load_seeds(self, seeds: List[dict]):
        """
        Pre-load memory seeds from the agent YAML into the memory stream.

        Seeds provide the agent's background knowledge before the simulation starts.
        If a seed has no importance score, the LLM is called to score it.
        Memory types that aren't in the allowed set (e.g. 'direct_experience' which
        is actually a source field) are mapped to 'observation'.
        """
        if not seeds:
            print(f"[{self.id}] Warning: no memory seeds found.")
            return

        print(f"[{self.id}] Seeding {len(seeds)} memories...")
        for seed in seeds:
            description = seed["description"]

            # Sanitise memory type — 'direct_experience' etc. are source fields, not types
            raw_type = seed.get("type", "observation")
            memory_type = raw_type if raw_type in _VALID_MEMORY_TYPES else "observation"

            importance = seed.get("importance")
            if importance is None:
                importance = score_importance(
                    self.client_anthropic, description, self.seed_narrative,
                    client_openrouter=self.client_openrouter,
                )

            self.memory.add(
                description=description,
                importance=importance,
                embedding=embed(description),
                memory_type=memory_type,
                timestamp=0,   # seeds exist before the simulation starts (day 0)
            )

        print(f"[{self.id}] Done — {self.memory.count()} memories seeded.\n")

    # ── Cognition cycle ───────────────────────────────────────────────────────

    def perceive(self, event: dict) -> Tuple[str, "np.ndarray", str]:
        """
        Frame the event through its channel and embed the content.

        The framing language (official_mail, news_media, social, direct_experience)
        shapes which memories the agent retrieves. We embed the raw content rather
        than the framed text so the semantic query isn't skewed by framing words.

        Returns:
            situation    — framed text passed to the decision prompt
            query_embed  — embedding of the raw content (for retrieval)
            query_text   — raw content (for sparse/hybrid retrieval)
        """
        situation = frame_event(event["channel"], event["content"])
        query_text = event["content"]
        query_embed = embed(query_text)
        return situation, query_embed, query_text

    def retrieve(
        self,
        query_embed: "np.ndarray",
        query_text: str,
        current_day: int,
    ) -> List["Memory"]:
        """
        Return the top-K most relevant memories for this query.

        Returns an empty list if use_memory=False (ablation Variant 3 — the agent
        gets only its seed personality, no episodic context).
        """
        if not self.use_memory:
            return []
        return retrieve_memories(
            self.memory, query_embed, query_text, self.retrieval_config, current_day
        )

    def _parse_response(self, raw: str) -> Tuple[str, str]:
        """
        Parse the LLM's response into (decision, reasoning).

        Expects JSON: {"decision": "...", "reasoning": "..."}.
        If parsing fails (model didn't follow the format), the whole response
        is stored as reasoning with an empty decision — so nothing is lost.
        """
        text = raw.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            if text.startswith("json"):
                text = text[4:]
        try:
            parsed = json.loads(text.strip())
            return (
                str(parsed.get("decision", "")).strip(),
                str(parsed.get("reasoning", raw)).strip(),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback: keep the full response as reasoning
            return "", raw.strip()

    def make_decision(
        self,
        situation: str,
        retrieved_memories: List["Memory"],
    ) -> Tuple[str, str]:
        """
        Build the full prompt and call the LLM for a structured decision.

        Asks the model to return JSON with separate 'decision' and 'reasoning'
        fields. The separation is required so the judge can score them
        independently and so the JSONL log captures both cleanly.

        Returns:
            decision  — what the agent does (actions, calls, choices)
            reasoning — why (internal logic, feelings, memories that shaped it)
        """
        system_prompt = build_system_prompt(self.seed_narrative, retrieved_memories)

        decision_question = (
            "Given your personality, memories, and the situation above, respond with "
            "JSON in this exact format (no extra text outside the JSON):\n"
            '{"decision": "<specific actions you take — what you do, who you contact, '
            'what you decide>", '
            '"reasoning": "<your internal logic — how your personality, memories, and '
            'feelings shape this response>"}\n\n'
            "Be specific. Stay in character. Do not break the JSON format."
        )
        user_prompt = build_decision_prompt(situation, decision_question)

        raw = decide(
            self.client_anthropic,
            system_prompt,
            user_prompt,
            client_openrouter=self.client_openrouter,
        )
        return self._parse_response(raw)

    def store(
        self,
        description: str,
        memory_type: str,
        current_day: int,
        importance: Optional[int] = None,
    ) -> "Memory":
        """
        Score importance (if not provided) and store a new memory.

        Called after each decision to record both what happened (observation)
        and what the agent decided (decision memory).
        """
        if importance is None:
            importance = score_importance(
                self.client_anthropic, description, self.seed_narrative,
                client_openrouter=self.client_openrouter,
            )
        return self.memory.add(
            description=description,
            importance=importance,
            embedding=embed(description),
            memory_type=memory_type,
            timestamp=current_day,
        )

    def maybe_reflect(self, current_day: int) -> List["Memory"]:
        """
        Trigger the two-step reflection process if the importance threshold is met.

        If use_reflection=False (ablation Variant 2), always returns empty list.
        Otherwise delegates to reflection.py's maybe_reflect().
        """
        if not self.use_reflection:
            return []
        new_reflections, self._last_reflection_index = maybe_reflect(
            stream=self.memory,
            client_anthropic=self.client_anthropic,
            config=self.llm_config,
            reflection_config=self.reflection_config,
            agent_seed=self.seed_narrative,
            current_day=current_day,
            last_reflection_index=self._last_reflection_index,
            client_openrouter=self.client_openrouter,
        )
        return new_reflections

    def run_cognition_cycle(self, event: dict, current_day: int) -> dict:
        """
        Run the full perceive → retrieve → decide → store → reflect cycle.

        This is called once per event per targeted agent by simulation.py.

        Returns a structured result dict that simulation.py passes directly to
        the logger — every field needed for logging and judge scoring is here.
        """
        # 1. Perceive — frame the event and embed the content
        situation, query_embed, query_text = self.perceive(event)

        # 2. Retrieve — top-K relevant memories (empty if use_memory=False)
        retrieved = self.retrieve(query_embed, query_text, current_day)

        # 3. Decide — call LLM, parse into decision + reasoning
        decision, reasoning = self.make_decision(situation, retrieved)

        # 4. Store — record what happened (observation) and what was decided
        obs_mem = self.store(
            description=f"[Day {current_day}] {event['channel']}: {event['content']}",
            memory_type="observation",
            current_day=current_day,
        )
        dec_mem = self.store(
            description=f"[Day {current_day}] My response to {event['type']}: {decision}",
            memory_type="decision",
            current_day=current_day,
        )

        # Track the action in runtime state
        if decision:
            self.actions_taken.append(decision)

        # 5. Reflect — fires only if cumulative importance exceeds threshold
        new_reflections = self.maybe_reflect(current_day)

        return {
            "agent_id": self.id,
            "agent_display_name": self.display_name,
            "seed_personality": self.seed_narrative,
            "situation": situation,
            "retrieved_memories": retrieved,
            "decision": decision,
            "reasoning": reasoning,
            "new_memory_ids": [obs_mem.id, dec_mem.id],
            "new_reflections": new_reflections,
        }

    # ── State inspection ──────────────────────────────────────────────────────

    def state_snapshot(self) -> dict:
        """
        Return a lightweight state dict for end-of-tick logging.
        Also used by compare_agents() in the notebook.
        """
        return {
            "agent_id": self.id,
            "display_name": self.display_name,
            "memory_count": self.memory.count(),
            "compliance_status": self.compliance_status,
            "attitude": self.attitude,
            "actions_taken_count": len(self.actions_taken),
            "last_action_preview": self.actions_taken[-1] if self.actions_taken else None,
        }

    def pretty_print(self):
        """Print a human-readable summary of this agent's current state."""
        print(f"\n{'='*60}")
        print(f"  Agent: {self.display_name}  (id: {self.id})")
        print(f"  Memories: {self.memory.count()}  |  "
              f"Actions taken: {len(self.actions_taken)}")
        print(f"  Compliance: {self.compliance_status}  |  Attitude: {self.attitude}")
        print(f"  Ablation: memory={'ON' if self.use_memory else 'OFF'}  "
              f"reflection={'ON' if self.use_reflection else 'OFF'}")
        if self.actions_taken:
            print(f"  Last decision: {self.actions_taken[-1][:80]}...")
        print(f"{'='*60}\n")


# ── YAML loading helper ───────────────────────────────────────────────────────

def _load_agent_yaml(yaml_path: str) -> dict:
    """
    Load agent data from a YAML file.

    Handles two formats:
      - Wrapped:  agents: [{id: ..., seed_narrative: ..., ...}]  (beth.yaml / eleanor_v2.yaml)
      - Flat:     id: ...  seed_narrative: ...  (older prototype format)

    For wrapped files with multiple agents, loads the first entry only.
    """
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if "agents" in raw and isinstance(raw["agents"], list):
        data = raw["agents"][0]
    else:
        data = raw

    # Validate required fields
    for required in ("id", "display_name", "seed_narrative"):
        if required not in data:
            raise ValueError(
                f"Agent YAML at '{yaml_path}' is missing required field '{required}'. "
                f"Expected fields: id, display_name, seed_narrative, memory_seeds."
            )
    return data
