"""
export_agents_json.py — Convert agents_extracted.yaml to JSON for Adaptation Labs.

Reads all extracted agent profiles and writes a clean JSON array suitable for
upload to the Adaptive Data Platform (accepts JSON files directly).

Strips held_out_responses — those are private test/eval data and should not
be used as synthetic generation examples.

Usage:
    python scripts/export_agents_json.py

Output:
    outputs/synthetic/agents_for_synthesis.json
"""

import json
import yaml
from pathlib import Path

INPUT_YAML = Path("config/agents/preprocessing_output/agents_extracted.yaml")
OUTPUT_JSON = Path("outputs/synthetic/agents_for_synthesis.json")

# Fields to remove before export — test data or internal-only fields
STRIP_FIELDS = {"held_out_responses"}


def clean_agent(agent: dict) -> dict:
    """Remove internal fields and normalise nulls for export."""
    return {k: v for k, v in agent.items() if k not in STRIP_FIELDS}


def main():
    with open(INPUT_YAML, "r") as f:
        data = yaml.safe_load(f)

    agents = data.get("agents", [])
    cleaned = [clean_agent(a) for a in agents]

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(cleaned, f, indent=2, default=str)

    print(f"Exported {len(cleaned)} agents → {OUTPUT_JSON}")
    for a in cleaned:
        print(f"  {a.get('id', '?'):30s}  {a.get('archetype', '')}")


if __name__ == "__main__":
    main()
