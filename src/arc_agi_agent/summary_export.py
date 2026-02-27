from __future__ import annotations

import json
from typing import Any, Dict, List


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_markdown(path: str, summary: Dict[str, Any]) -> None:
    lines: List[str] = []
    run_summary = summary.get("run_summary", {})
    lessons = summary.get("lessons", {})

    lines.append("# Run Summary")
    lines.append(f"- game_id: {run_summary.get('game_id')}")
    lines.append(f"- seed: {run_summary.get('seed')}")
    lines.append(f"- steps: {run_summary.get('steps')}")
    lines.append(f"- unique_states: {run_summary.get('unique_states')}")
    lines.append(f"- unique_transitions: {run_summary.get('unique_transitions')}")
    lines.append(f"- terminal_reached: {run_summary.get('terminal_reached')}")
    lines.append(f"- reward_total: {run_summary.get('reward_total')}")

    lines.append("")
    lines.append("# Top Action Efficacy (Best/Worst)")
    action_efficacy = lessons.get("action_efficacy", {})
    best = action_efficacy.get("best_actions", [])
    worst = action_efficacy.get("worst_actions", [])
    lines.append(f"- best_actions: {best}")
    lines.append(f"- worst_actions: {worst}")

    lines.append("")
    lines.append("# Loops")
    loops = lessons.get("loop_analysis", {}).get("loops", [])
    for loop in loops:
        lines.append(
            f"- {loop.get('type')} {loop.get('likely_cause')} steps {loop.get('start_step')}-{loop.get('end_step')}"
        )

    lines.append("")
    lines.append("# Invariants")
    invariants = lessons.get("discovered_invariants", {})
    lines.append(f"- static_cells: {len(invariants.get('static_cells', []))}")
    lines.append(f"- always_present_objects: {len(invariants.get('always_present_objects', []))}")
    lines.append(f"- never_used_actions: {invariants.get('never_used_actions', [])}")

    lines.append("")
    lines.append("# Keyframes")
    for frame in lessons.get("state_keyframes", {}).get("keyframes", []):
        lines.append(f"- step {frame.get('step_idx')}: {frame.get('why_selected')}")

    lines.append("")
    lines.append("# Hypothesis/Mechanic Outcomes")
    lines.append(f"- hypotheses: {bool(lessons.get('hypothesis_outcomes'))}")
    lines.append(f"- mechanics: {bool(lessons.get('mechanic_outcomes'))}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
