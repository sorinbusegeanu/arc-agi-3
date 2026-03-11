from __future__ import annotations

from dataclasses import dataclass, field

from v3_1.contracts.snapshots import BlackboardSnapshot
from v3_1.contracts.versions import next_blackboard_version
from v3_1.utils.ids import make_handle
from v3_1.world.merge import apply_delta


@dataclass
class BlackboardState:
    session_id: str
    game_id: str
    revision: int = 0
    state: dict = field(default_factory=lambda: {
        "areas": {},
        "entities": {},
        "consequences": {},
        "trigger_zones": {},
        "topology_nodes": {},
        "topology_edges": {},
        "indexes": {},
    })

    def snapshot(self, *, round_id: int, pass_id: int, material_change: bool) -> BlackboardSnapshot:
        version = next_blackboard_version(self.session_id, round_id, self.revision)
        payload = {"version": version, "revision": self.revision, "state": self.state}
        return BlackboardSnapshot(
            snapshot_handle=make_handle("snapshot:blackboard", payload),
            blackboard_version=version,
            created_round_id=round_id,
            created_pass_id=pass_id,
            material_change=material_change,
            state=self.state,
            indexes=self.state.get("indexes", {}),
        )

    def merge(self, *, round_id: int, pass_id: int, deltas: list[dict]) -> BlackboardSnapshot:
        material_change = False
        next_state = self.state
        for delta in deltas:
            next_state, changed = apply_delta(next_state, delta)
            material_change = material_change or changed
        self.revision += 1
        self.state = next_state
        return self.snapshot(round_id=round_id, pass_id=pass_id, material_change=material_change)

