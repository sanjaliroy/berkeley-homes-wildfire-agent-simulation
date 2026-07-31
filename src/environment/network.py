"""
network.py: which residents an event reaches.

The Environment layer answers two separate questions about an incoming event.
`channels.py` answers HOW it arrives — the framing language that shapes which
memories the resident retrieves. This module answers WHO it arrives for.

Audience selection is driven by the `target_agents` field on each scenario event:

    all               -> every loaded agent
    Margaret          -> the agent whose display_name matches (case-sensitive)
    Margaret, Laura   -> several agents, comma-separated

Scope, stated plainly: the baseline scenario targets `all` on every event, so in
the reported runs each resident receives the same six interventions and residents
do not interact with one another. This module is the seam where that changes —
targeted mailings, neighbour-scoped social pressure, or a partial-coverage
campaign can be expressed in the scenario YAML without touching the engine.
"""

from typing import Callable, Dict, List, Optional, TypeVar

# Agent is imported only for typing in callers; keep this module dependency-free
# so it can be tested without constructing agents or LLM clients.
AgentT = TypeVar("AgentT")

ALL = "all"


def parse_target_spec(target_agents) -> Optional[List[str]]:
    """
    Normalise a scenario `target_agents` value into a list of display names.

    Returns None for the broadcast case ("all"), which callers treat as
    "every loaded agent" — distinct from an empty list, which means "a specific
    audience was requested and nobody matched".

        parse_target_spec("all")              -> None
        parse_target_spec("Margaret")         -> ["Margaret"]
        parse_target_spec("Margaret, Laura")  -> ["Margaret", "Laura"]
    """
    spec = str(target_agents).strip()
    if spec.lower() == ALL:
        return None
    return [name.strip() for name in spec.split(",") if name.strip()]


def resolve_audience(
    target_agents,
    agents_by_name: Dict[str, AgentT],
    on_missing: Optional[Callable[[str, List[str]], None]] = None,
) -> List[AgentT]:
    """
    Map a scenario event's `target_agents` field to the agents that receive it.

    Args:
        target_agents:  the raw field from the scenario YAML.
        agents_by_name: loaded agents keyed by display_name.
        on_missing:     called as on_missing(name, available) for each requested
                        name that is not loaded. Unknown names are skipped rather
                        than raising, so one typo in a scenario does not abort a
                        run that is otherwise valid.

    Returns:
        The agents to deliver this event to, in scenario order for a named
        audience and in load order for a broadcast.
    """
    names = parse_target_spec(target_agents)
    if names is None:
        return list(agents_by_name.values())

    audience: List[AgentT] = []
    for name in names:
        if name in agents_by_name:
            audience.append(agents_by_name[name])
        elif on_missing is not None:
            on_missing(name, list(agents_by_name))
    return audience


def describe_audience(target_agents, agents_by_name: Dict[str, AgentT]) -> str:
    """Short human-readable audience label for notebook and log display."""
    names = parse_target_spec(target_agents)
    if names is None:
        return f"all residents ({len(agents_by_name)})"
    return ", ".join(names)
