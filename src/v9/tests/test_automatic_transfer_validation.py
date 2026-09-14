from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from v9.environments.contract import BoundaryEvent, BoundaryScope
from v9.environments.schemas import ActionSchema, EnvironmentIdentity, ObservationSchema
from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig, ScientificConfig
from v9.runtime.transfer_validation import run_transfer_validation_interval


def _runtime_with_concept(root: Path) -> ContinuousMemoryRuntime:
    scientific = ScientificConfig(
        transfer_validation_mode="validation_budgeted",
        transfer_minimum_trials=2,
        transfer_validation_trials_per_interval=4,
    )
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(root, restore=False, enable_snapshots=True, scientific=scientific)
    )
    for index in range(2):
        runtime.submit(
            runtime.make_experience(
                producer_id=1,
                producer_sequence=index + 1,
                environment_instance_id=7,
                global_step=index,
                context_signature=3,
                action_id=2,
                outcome_signature=4,
                family_signature=5,
                primary_valence=1,
            )
        )
    assert runtime._m4
    return runtime


def test_positive_held_out_validation_forms_and_restores_m5_m6_m7(tmp_path: Path) -> None:
    runtime = _runtime_with_concept(tmp_path)
    concept_uid = next(iter(runtime._m4))
    candidate = runtime.transfer_validation_candidates(limit=1)[0]
    assert candidate["formation_scope"] == (7,)
    assert candidate["actions"][0] == 2

    runtime.record_transfer_validation(
        concept_uid,
        target_environment_id=9,
        target_native_action=2,
        enabled_metric=1.0,
        ablated_metric=0.0,
    )
    runtime.record_transfer_validation(
        concept_uid,
        target_environment_id=10,
        target_native_action=2,
        enabled_metric=1.0,
        ablated_metric=0.0,
    )
    levels = runtime.metrics()["memory_levels"]
    assert levels["M5"] > 0
    assert levels["M6"] > 0
    assert levels["M7"] > 0
    assert runtime._m5 and runtime._m6 and runtime._m7

    runtime.close()
    restored = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, scientific=runtime.config.scientific))
    assert restored._m5 and restored._m6 and restored._m7
    restored_levels = restored.metrics()["memory_levels"]
    assert restored_levels["M5"] == levels["M5"]
    assert restored_levels["M6"] == levels["M6"]
    assert restored_levels["M7"] == levels["M7"]


class _FakeTransferAdapter:
    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.value = 0
        self._boundary = BoundaryEvent()
        self._identity = EnvironmentIdentity("fake", "target", "default", f"seed={seed}")
        self._observation_schema = ObservationSchema("scalar", "fake")
        self._action_schema = ActionSchema("environment-local", "fake")

    def identity(self):
        return self._identity

    def observation_schema(self):
        return self._observation_schema

    def action_schema(self):
        return self._action_schema

    def observe(self):
        return self.value

    def encode_observation(self, observation):
        return int(observation)

    def available_actions(self):
        return (0, 2)

    def boundary_event(self):
        return self._boundary

    def step(self, action):
        if int(action) == 2:
            self.value += 1
            self._boundary = BoundaryEvent(BoundaryScope.SUBEPISODE, 1, True)
        else:
            self.value -= 1
            self._boundary = BoundaryEvent()
        return self.value

    def capture_state(self):
        return self.value

    def restore_state(self, state):
        self.value = int(state)
        self._boundary = BoundaryEvent()

    def close(self):
        return None


def test_epoch_transfer_interval_uses_matched_intervention_and_unlocks_higher_memory(tmp_path: Path) -> None:
    runtime = _runtime_with_concept(tmp_path)
    args = SimpleNamespace(seed=0, steps_per_game=2, env_root=None, alfred_backend_factory=None)
    spec = SimpleNamespace(game_id="target")

    def factory(_spec, *, seed, env_root, alfred_backend_factory=None):
        del env_root, alfred_backend_factory
        return _FakeTransferAdapter(seed)

    stats = run_transfer_validation_interval(runtime, (spec,), args, epoch=1, adapter_factory=factory)
    assert stats.completed >= 2
    assert stats.passed >= 2
    assert stats.validated_concepts >= 1
    levels = runtime.metrics()["memory_levels"]
    assert levels["M5"] > 0
    assert levels["M6"] > 0
    assert levels["M7"] > 0
