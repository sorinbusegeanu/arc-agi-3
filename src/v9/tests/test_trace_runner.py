from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass

from v9.environments.contract import BoundaryEvent, BoundaryScope, TaskProgress
from v9.environments.schemas import ActionSchema, EnvironmentIdentity, ObservationSchema
from v9.runtime.trace_runner import run_trace_bundle


@dataclass(frozen=True)
class FakeSpec:
    adapter: str = "fake"
    game_id: str = "Fake-v0"

    @property
    def display_name(self) -> str:
        return self.game_id


class FakeAdapter:
    def __init__(self) -> None:
        self.value = 0
        self._boundary = BoundaryEvent()

    def identity(self):
        return EnvironmentIdentity("fake", "Fake-v0", "trace", "seed=0")

    def observation_schema(self):
        return ObservationSchema("scalar", "fake")

    def action_schema(self):
        return ActionSchema("discrete", "n=2")

    def observe(self):
        return {"value": self.value, "text": f"state-{self.value}"}

    def encode_observation(self, observation):
        return int(observation["value"])

    def encode_action(self, action):
        return int(action)

    def available_actions(self):
        return (0, 1)

    def optional_symbol_stream(self):
        return tuple(b"mission")

    def trace_action_labels(self, actions):
        return {int(action): ("stay" if int(action) == 0 else "advance") for action in actions}

    def step(self, action):
        self.value += int(action) + 1
        terminal = self.value >= 5
        self._boundary = BoundaryEvent(
            BoundaryScope.EPISODE if terminal else BoundaryScope.NONE,
            1 if terminal else 0,
            not terminal,
        )
        return self.observe()

    def boundary_event(self):
        return self._boundary

    def task_progress(self):
        return TaskProgress(
            game_id="Fake-v0",
            level_id="level-1",
            level_index=1,
            levels_completed=int(self.value >= 5),
            terminal=not self._boundary.continuation,
            success=self._boundary.primary_valence > 0,
            failure=False,
            truncated=False,
            score=float(self.value),
        )

    def reset(self):
        self.value = 0
        self._boundary = BoundaryEvent()
        return self.observe()

    def close(self):
        return None


def test_trace_bundle_contains_100_steps_and_readable_fields(tmp_path) -> None:
    def factory(spec, **kwargs):
        return FakeAdapter()

    bundle = run_trace_bundle(
        (FakeSpec(),),
        root=tmp_path,
        seed=0,
        env_root=None,
        alfred_backend_factory=None,
        make_adapter=factory,
        steps_per_game=100,
    )
    assert bundle.exists()
    with zipfile.ZipFile(bundle) as archive:
        assert "manifest.json" in archive.namelist()
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["steps_per_game"] == 100
        assert manifest["games"][0]["steps"] == 100
        trace_name = manifest["games"][0]["file"]
        rows = [json.loads(line) for line in archive.read(trace_name).decode("utf-8").splitlines()]
    assert len(rows) == 100
    assert rows[0]["symbols_before"] == "mission"
    assert rows[0]["available_actions_before"][1]["semantic"] == "advance"
    assert rows[0]["chosen_action"]["semantic"] in {"stay", "advance"}
    assert "before_observation" in rows[0]
    assert "after_observation" in rows[0]
    assert "task_progress" in rows[0]
