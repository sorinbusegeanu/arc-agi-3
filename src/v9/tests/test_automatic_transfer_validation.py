from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import v9.cli as cli
from v9.environments.contract import BoundaryEvent, BoundaryScope
from v9.environments.schemas import ActionSchema, EnvironmentIdentity, ObservationSchema
from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig, ScientificConfig
from v9.runtime.developmental_cut import DevelopmentalWorkStatus
from v9.runtime.epoch_runner import _run_epoch_transfer_validation
from v9.runtime.transfer_validation import (
    TransferValidationStats,
    _balance_transfer_candidates,
    run_transfer_validation_interval,
)


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
    assert stats.completed == 8
    assert stats.passed >= 2
    assert stats.validated_concepts >= 1
    levels = runtime.metrics()["memory_levels"]
    assert levels["M5"] > 0
    assert levels["M6"] > 0
    assert levels["M7"] > 0


def test_repeated_success_after_validation_does_not_rematerialize_higher_memory(tmp_path: Path) -> None:
    runtime = _runtime_with_concept(tmp_path)
    concept_uid = next(iter(runtime._m4))
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
    before = runtime.metrics()["memory_levels"].copy()
    assert runtime.is_concept_validated(concept_uid)

    runtime.record_transfer_validation(
        concept_uid,
        target_environment_id=11,
        target_native_action=2,
        enabled_metric=2.0,
        ablated_metric=0.0,
    )
    after = runtime.metrics()["memory_levels"]
    assert after["M5"] == before["M5"]
    assert after["M6"] == before["M6"]
    assert after["M7"] == before["M7"]


def test_transfer_candidates_are_round_robin_balanced_by_environment_type() -> None:
    rows = (
        {"id": "a1", "source_environment_types": ("blackjack",)},
        {"id": "a2", "source_environment_types": ("blackjack",)},
        {"id": "a3", "source_environment_types": ("blackjack",)},
        {"id": "b1", "source_environment_types": ("frozenlake",)},
        {"id": "b2", "source_environment_types": ("frozenlake",)},
        {"id": "c1", "source_environment_types": ("cartpole",)},
    )
    balanced = _balance_transfer_candidates(rows)
    assert [row["id"] for row in balanced] == ["a1", "c1", "b1", "a2", "b2", "a3"]


def test_transfer_validation_cli_restores_validates_persists_and_exits(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "persisted"
    snapshot = root / "snapshots" / "snapshot-1"
    snapshot.mkdir(parents=True)
    (snapshot / "COMPLETE").write_text("complete\n", encoding="utf-8")
    (root / "v9_run_summary.json").write_text(
        json.dumps({"epochs": [{"epoch": 7}]}) + "\n",
        encoding="utf-8",
    )
    model_dir = root / "models"
    model_dir.mkdir()
    manifest = model_dir / "hgt_manifest.json"
    manifest.write_text('{"accepted_model_version":"hgt-000007"}\n', encoding="utf-8")
    manifest_before = manifest.read_bytes()
    calls: dict[str, object] = {}

    class FakeRuntime:
        def __init__(self, config):
            calls["config"] = config
            self.config = config
            self.unified_telemetry = SimpleNamespace(model_version="hgt-000007")
            self.closed = False

        def start(self):
            calls["started"] = True

        def wait_quiescent(self, timeout):
            calls["quiescent_timeout"] = timeout

        def metrics(self):
            return {"validation_only": True}

        def close(self, *, normal=True, timeout=300.0):
            calls["close"] = (normal, timeout)
            self.closed = True
            return None

        def replay_once(self):
            raise AssertionError("validation-only mode invoked replay")

    def validate(runtime, specs, args, *, epoch, adapter_factory):
        calls["validation"] = (runtime, specs, args, epoch, adapter_factory)
        return TransferValidationStats(3, 2, 1, 1, None)

    def unexpected_epochs(*_args, **_kwargs):
        raise AssertionError("validation-only mode invoked sampling/training epochs")

    monkeypatch.setattr(cli, "ContinuousMemoryRuntime", FakeRuntime)
    monkeypatch.setattr(cli, "run_transfer_validation_interval", validate)
    monkeypatch.setattr(cli, "run_epochs", unexpected_epochs)
    monkeypatch.setattr(cli, "latest_snapshot", lambda _root: snapshot)

    args = cli.build_parser().parse_args(
        [
            "continuous-run",
            "--root",
            str(root),
            "--games",
            "synthetic",
            "--transfer-validation",
            "--no-dashboard",
        ]
    )
    assert cli.run_continuous(args) == 0

    config = calls["config"]
    assert config.restore is True
    assert calls["started"] is True
    assert calls["validation"][3] == 7
    assert calls["validation"][4] is cli.make_adapter
    assert calls["close"] == (True, args.final_save_timeout)
    assert manifest.read_bytes() == manifest_before
    report = json.loads((root / "transfer_validation_summary.json").read_text(encoding="utf-8"))
    assert report["model_version"] == "hgt-000007"
    assert report["validation"] == {
        "attempted": 3,
        "blocker": None,
        "completed": 2,
        "passed": 1,
        "validated_concepts": 1,
    }



def test_epoch_runner_automatically_executes_transfer_validation(monkeypatch) -> None:
    calls: dict[str, object] = {}
    expected = TransferValidationStats(8, 8, 4, 1, None)

    def validate(runtime, specs, args, *, epoch, adapter_factory):
        calls["validation"] = (runtime, specs, args, epoch, adapter_factory)
        return expected

    monkeypatch.setattr("v9.runtime.epoch_runner.run_transfer_validation_interval", validate)
    runtime = object()
    specs = (object(),)
    args = SimpleNamespace()
    adapter_factory = object()

    result = _run_epoch_transfer_validation(
        runtime,
        specs,
        args,
        epoch=3,
        adapter_factory=adapter_factory,
        developmental_session=None,
    )

    assert result == expected
    assert calls["validation"] == (runtime, specs, args, 3, adapter_factory)


def test_epoch_transfer_validation_runs_inside_matched_developmental_cut(monkeypatch) -> None:
    calls: dict[str, object] = {}
    expected = TransferValidationStats(4, 4, 2, 1, None)

    def validate(runtime, specs, args, *, epoch, adapter_factory):
        calls["validation"] = (runtime, specs, args, epoch, adapter_factory)
        return expected

    class Session:
        def run(self, operator, operation, *, stable_key, status):
            calls["operator"] = operator
            calls["stable_key"] = stable_key
            value = operation()
            calls["status"] = status(value)
            return value

    monkeypatch.setattr("v9.runtime.epoch_runner.run_transfer_validation_interval", validate)
    result = _run_epoch_transfer_validation(
        object(),
        (object(),),
        SimpleNamespace(),
        epoch=5,
        adapter_factory=object(),
        developmental_session=Session(),
    )

    assert result == expected
    assert calls["operator"] == "transfer_validation"
    assert calls["stable_key"] == "epoch:5:transfer-validation"
    assert calls["status"] is DevelopmentalWorkStatus.APPLIED
