from __future__ import annotations

import ray

from v3_1.world.blackboard import BlackboardState


@ray.remote
class BlackboardAgent:
    def __init__(self, session_id: str, game_id: str) -> None:
        self.state = BlackboardState(session_id=session_id, game_id=game_id)

    def merge(self, *, round_id: int, pass_id: int, deltas: list[dict]):
        return self.state.merge(round_id=round_id, pass_id=pass_id, deltas=deltas)

    def snapshot(self, *, round_id: int, pass_id: int, material_change: bool):
        return self.state.snapshot(round_id=round_id, pass_id=pass_id, material_change=material_change)

    def get_state(self) -> dict:
        return dict(self.state.state)
