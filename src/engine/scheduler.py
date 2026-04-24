"""
scheduler.py — Intervention event scheduler.

Loads the scenario YAML into a min-heap (priority queue) sorted by day.
simulation.py calls get_events(tick) each day to retrieve whatever is due.

Using a heap means the scheduler is O(k log n) per tick (where k = events due
that day) rather than scanning the full list — this matters when you have
hundreds of events across a long simulation.
"""

import heapq
import yaml
from dataclasses import dataclass
from typing import List


@dataclass
class Event:
    """
    A single intervention event, loaded from the scenario YAML.

    Fields match the scenario YAML structure:
        day           — simulation day this event fires
        type          — event category (e.g. 'insurance_non_renewal', 'wildfire_news')
        channel       — delivery channel ('official_mail', 'news_media', 'social', 'direct_experience')
        target_agents — 'all', a display_name, or comma-separated display names
        content       — the raw event text delivered to the agent
    """
    day: int
    type: str
    channel: str
    target_agents: str
    content: str

    # Required so heapq can sort Events when days are equal
    def __lt__(self, other: "Event") -> bool:
        return self.day < other.day


class EventScheduler:
    """
    Loads a scenario YAML and serves events day by day.

    Usage:
        scheduler = EventScheduler("config/scenarios/baseline.yaml")
        events = scheduler.get_events(tick=10)  # returns all events on day 10

    Events are consumed as they're returned — call peek_all() before the
    simulation starts to see the full event timeline without consuming anything.
    """

    def __init__(self, scenario_path: str):
        self._heap: List[tuple] = []   # (day, Event) pairs — kept as min-heap
        self.scenario_name: str = ""
        self.description: str = ""
        self.duration_days: int = 60

        self._load(scenario_path)

    def _load(self, path: str):
        """Parse the scenario YAML and push all events onto the heap."""
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        self.scenario_name = raw.get("scenario_name", "unnamed")
        self.description = raw.get("description", "")
        self.duration_days = raw.get("duration_days", 60)

        for raw_event in raw.get("events", []):
            event = Event(
                day=int(raw_event["day"]),
                type=raw_event["type"],
                channel=raw_event["channel"],
                target_agents=str(raw_event.get("target_agents", "all")),
                content=str(raw_event["content"]).strip(),
            )
            # heapq is a min-heap: (day, event) tuples sort by day first
            heapq.heappush(self._heap, (event.day, event))

        print(
            f"[scheduler] Loaded '{self.scenario_name}' — "
            f"{len(self._heap)} events over {self.duration_days} days."
        )

    def get_events(self, tick: int) -> List[Event]:
        """
        Return all events scheduled for this exact day and remove them from the queue.

        Returns an empty list if no events are scheduled for this tick.
        """
        events = []
        while self._heap and self._heap[0][0] == tick:
            _, event = heapq.heappop(self._heap)
            events.append(event)
        return events

    def peek_all(self) -> List[Event]:
        """
        Return all remaining events in day order without consuming them.

        Use this in the notebook to preview the full event timeline before
        starting the simulation.
        """
        return [event for _, event in sorted(self._heap)]

    def remaining_count(self) -> int:
        """Number of events still in the queue (not yet delivered)."""
        return len(self._heap)

    def has_events_on(self, tick: int) -> bool:
        """Quick check whether any events are scheduled for a given day."""
        return bool(self._heap) and self._heap[0][0] == tick
