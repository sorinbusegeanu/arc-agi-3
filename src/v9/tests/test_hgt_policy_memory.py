from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

from v9.runtime import hgt_policy_memory as policy_memory


class FakeRuntime:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._hgt_action_scores = {1: {2: 0.5}}
        self._hgt_context_action_scores = {1: {7: {2: 0.75}}}
        self._actor_policy_generation = 0
        self.unified_telemetry = SimpleNamespace(model_version="hgt-000001")
        self.gauges = {}

    def set_telemetry_gauge(self, key, value) -> None:
        self.gauges[key] = value


def test_shallow_capture_keeps_policy_maps_without_deep_copy() -> None:
    runtime = FakeRuntime()
    captured = policy_memory._capture_policy_state_shallow(runtime)
    assert captured["scores"] is runtime._hgt_action_scores
    assert captured["context_scores"] is runtime._hgt_context_action_scores

    original_scores = captured["scores"]
    runtime._hgt_action_scores = {9: {3: 1.0}}
    assert captured["scores"] is original_scores
    assert 1 in captured["scores"]


def test_policy_sidecar_roundtrip_assigns_loaded_maps_directly(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    policy_memory._write_policy_sidecar(
        tmp_path,
        "hgt-000002",
        {4: {1: 0.25}},
        {4: {11: {1: 0.5}}},
    )
    restored = policy_memory.load_hgt_policy_version(
        runtime, root=tmp_path, model_version="hgt-000002"
    )
    assert restored == "hgt-000002"
    assert runtime._hgt_action_scores == {4: {1: 0.25}}
    assert runtime._hgt_context_action_scores == {4: {11: {1: 0.5}}}
    assert runtime.unified_telemetry.model_version == "hgt-000002"


def test_current_manifest_bootstraps_sidecar_without_checkpoint_load(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "hgt_manifest.json").write_text(
        json.dumps(
            {
                "current_model_version": "hgt-000003",
                "action_scores": {"8": {"2": 0.2}},
                "context_action_scores": {"8": {"13": {"2": 0.4}}},
            }
        ),
        encoding="utf-8",
    )
    runtime = FakeRuntime()
    restored = policy_memory.load_hgt_policy_version(
        runtime, root=tmp_path, model_version="hgt-000003"
    )
    assert restored == "hgt-000003"
    assert runtime._hgt_action_scores == {8: {2: 0.2}}
    assert (model_dir / "hgt-000003.policy.pkl").exists()


def test_behavior_rejection_rolls_back_from_sidecar(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    policy_memory._write_policy_sidecar(
        tmp_path, "hgt-000004", {5: {1: 0.9}}, {5: {17: {1: 0.8}}}
    )
    manifest = {
        "current_model_version": "hgt-000005",
        "candidate_model_version": "hgt-000005",
        "candidate_status": "TESTING_PENDING_BEHAVIOR",
        "parent_model_version": "hgt-000004",
        "last_accepted_model_version": "hgt-000004",
    }
    (model_dir / "hgt_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    runtime = FakeRuntime()
    runtime.unified_telemetry.model_version = "hgt-000005"

    restored = policy_memory.resolve_hgt_behavior_test(
        runtime, root=tmp_path, accepted=False
    )
    assert restored == "hgt-000004"
    assert runtime._hgt_action_scores == {5: {1: 0.9}}
    saved = json.loads((model_dir / "hgt_manifest.json").read_text(encoding="utf-8"))
    assert saved["candidate_status"] == "REJECTED_BEHAVIOR_GATE"
    assert saved["current_model_version"] == "hgt-000004"


def test_legacy_checkpoint_fallback_is_memory_mapped() -> None:
    source = Path(policy_memory.__file__).read_text(encoding="utf-8")
    assert 'torch.load(checkpoint_path, map_location="cpu", mmap=True)' in source
