"""
Audience resolution (src/environment/network.py).

These lock in the behaviour the engine relied on before the logic moved out of
simulation.py, so a future change to routing cannot silently alter which
residents receive an intervention.
"""

from src.environment.network import parse_target_spec, resolve_audience, describe_audience

AGENTS = {"Laura": 1, "Linda": 2, "Walter": 3, "Margaret": 4, "Miriam Voss": 5}


# parse_target_spec

def test_broadcast_parses_to_none():
    assert parse_target_spec("all") is None


def test_broadcast_is_case_and_whitespace_insensitive():
    assert parse_target_spec(" ALL ") is None


def test_single_name_parses_to_one_element():
    assert parse_target_spec("Margaret") == ["Margaret"]


def test_comma_separated_names_are_stripped():
    assert parse_target_spec("Margaret, Laura") == ["Margaret", "Laura"]


def test_empty_names_are_discarded():
    assert parse_target_spec("Laura,,Linda") == ["Laura", "Linda"]


# resolve_audience

def test_all_returns_every_agent_in_load_order():
    assert resolve_audience("all", AGENTS) == [1, 2, 3, 4, 5]


def test_named_audience_preserves_scenario_order():
    assert resolve_audience("Margaret, Laura", AGENTS) == [4, 1]


def test_display_names_with_spaces_resolve():
    assert resolve_audience("Miriam Voss", AGENTS) == [5]


def test_matching_is_case_sensitive():
    # 'margaret' must not silently resolve to 'Margaret'
    assert resolve_audience("margaret", AGENTS) == []


def test_unknown_name_is_skipped_not_raised():
    assert resolve_audience("Margaret, Nobody", AGENTS) == [4]


def test_unknown_name_invokes_on_missing_with_available_names():
    seen = []
    resolve_audience("Nobody", AGENTS, on_missing=lambda n, a: seen.append((n, a)))
    assert seen == [("Nobody", list(AGENTS))]


def test_on_missing_is_optional():
    assert resolve_audience("Nobody", AGENTS) == []


def test_no_agents_loaded_yields_empty_broadcast():
    assert resolve_audience("all", {}) == []


def test_empty_audience_is_distinct_from_broadcast():
    # A named audience that matches nobody must not fall back to everyone.
    assert resolve_audience("Nobody", AGENTS) != resolve_audience("all", AGENTS)


# describe_audience

def test_describe_broadcast_reports_count():
    assert describe_audience("all", AGENTS) == "all residents (5)"


def test_describe_named_audience_lists_names():
    assert describe_audience("Margaret, Laura", AGENTS) == "Margaret, Laura"
