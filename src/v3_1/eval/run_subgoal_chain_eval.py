from __future__ import annotations

from v3_1.eval.subgoal_chain_metrics import compute_subgoal_chain_metrics


def run_subgoal_chain_eval(
    *,
    metadata_only_chain_events: list[dict],
    executable_subgoal_chain_events: list[dict],
    games: list[str],
    seeds: list[int],
    round_budget: int,
    graph_settings: dict | None = None,
    hypothesis_settings: dict | None = None,
) -> dict:
    return {
        "games": list(games),
        "seeds": list(seeds),
        "round_budget": int(round_budget),
        "graph_settings": dict(graph_settings or {}),
        "hypothesis_settings": dict(hypothesis_settings or {}),
        "comparison": {
            "metadata_only_chain_mode": compute_subgoal_chain_metrics(metadata_only_chain_events, total_rounds=round_budget * max(1, len(games) * len(seeds))),
            "executable_subgoal_chain_mode": compute_subgoal_chain_metrics(executable_subgoal_chain_events, total_rounds=round_budget * max(1, len(games) * len(seeds))),
        },
    }
