from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .agent import ZodAgent


def run_eval(
    games: list[str],
    seed: int = 0,
    max_actions: int = 80,
    **agent_kwargs: Any,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for game in games:
        agent = ZodAgent(game_id=game, seed=seed, max_actions=max_actions, **agent_kwargs)
        out = agent.run_episode()
        results.append(asdict(out))
    return results
