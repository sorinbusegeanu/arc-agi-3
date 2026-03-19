from __future__ import annotations

from v3_1.eval.planner_probe_escalation_metrics import compute_planner_probe_escalation_metrics


def run_probe_escalation_eval(
    *,
    seeding_fix_only_events: list[dict],
    probe_escalation_events: list[dict],
    games: list[str],
    seeds: list[int],
    round_budget: int,
    graph_settings: dict | None = None,
    hypothesis_settings: dict | None = None,
) -> dict:
    total_rounds = int(round_budget) * max(1, len(list(games or [])) * len(list(seeds or [])))
    return {
        "games": list(games),
        "seeds": list(seeds),
        "round_budget": int(round_budget),
        "graph_settings": dict(graph_settings or {}),
        "hypothesis_settings": dict(hypothesis_settings or {}),
        "comparison": {
            "detector_backed_seeding_fix_only": compute_planner_probe_escalation_metrics(seeding_fix_only_events, total_rounds=total_rounds),
            "detector_backed_seeding_plus_probe_escalation": compute_planner_probe_escalation_metrics(probe_escalation_events, total_rounds=total_rounds),
        },
    }
