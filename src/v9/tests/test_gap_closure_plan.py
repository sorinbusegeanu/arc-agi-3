from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from v9 import ContinuousMemoryRuntime
from v9.memory import CanonicalNode, MemoryLevel, MemoryType
from v9.runtime.adaptive_exploration import viability_adjusted_epsilon
from v9.runtime.config import RuntimeConfig
from v9.runtime.environment_viability import EnvironmentViabilityController, EnvironmentViabilityProfile, ViabilityState
from v9.runtime.pressure_control import pressure_budget
from v9.runtime.memory_governor import MemoryGovernorState
from v9.runtime.viability_confidence import _row_confidence


def _transition(
    index: int,
    *,
    success: bool = False,
    failure: bool = False,
    truncated: bool = False,
    levels_completed: int = 0,
    valence: int = 0,
    branching: int = 4,
    after: int | None = None,
) -> SimpleNamespace:
    before = index % 8
    return SimpleNamespace(
        environment_identity=("synthetic", "gap-test", "default", "instance-1"),
        game_scenario="gap-test",
        episode_id=index // 4,
        before_signature=before,
        action_id=index % branching,
        after_signature=before if after is None else int(after),
        available_actions_after=branching,
        primary_valence=valence,
        task_success=success,
        task_failure=failure,
        task_truncated=truncated,
        levels_completed=levels_completed,
    )


def test_authoritative_runtime_has_active_bounded_indexes(tmp_path: Path) -> None:
    runtime = ContinuousMemoryRuntime(RuntimeConfig(tmp_path / "run", enable_snapshots=False, restore=False))
    try:
        assert getattr(runtime.graph, "_bounded_recent_by_level", None) is not None
        assert getattr(runtime.graph, "_bounded_edges_by_uid", None) is not None
        assert getattr(runtime.graph, "_bounded_compaction_queue", None) is not None
        for index in range(200):
            node = CanonicalNode.build(MemoryLevel.M2, MemoryType.FAMILY, (index,), index + 1)
            runtime._publish(node, {"parents": []}, ())
        view = runtime.graph.bounded_view(max_nodes=12, max_edges=32)
        assert len(view.nodes) <= 12
        assert runtime.graph._bounded_view_node_scan <= max(12 * 8, 12 + 64)
        assert runtime.graph._bounded_view_edge_scan <= max(32 * 4, 12 * 8, 256)
    finally:
        runtime.close()


def test_pressure_policy_shrinks_working_sets_and_pauses_only_at_hard_pressure() -> None:
    normal = pressure_budget(MemoryGovernorState.NORMAL)
    compacting = pressure_budget(MemoryGovernorState.COMPACTING)
    hard = pressure_budget(MemoryGovernorState.HARD_PRESSURE_DRAIN)
    recovering = pressure_budget(MemoryGovernorState.RECOVERING)
    assert normal.view_scale == normal.hgt_scale == 1.0
    assert 0.0 < hard.view_scale < compacting.view_scale < normal.view_scale
    assert 0.0 < hard.hgt_scale < recovering.hgt_scale < normal.hgt_scale
    assert hard.producer_paused
    assert not normal.producer_paused
    assert not compacting.producer_paused
    assert not recovering.producer_paused


def test_viability_profiles_detect_anomaly_recover_and_roundtrip(tmp_path: Path) -> None:
    controller = EnvironmentViabilityController(tmp_path)
    profile = None
    for index in range(64):
        terminal = index % 4 == 3
        profile = controller.observe(
            _transition(index, failure=terminal, branching=4, after=(index + 1) % 8),
            watermark=index + 1,
            policy_scores={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        )
    assert profile is not None
    assert profile.state in {ViabilityState.LOW_EVIDENCE, ViabilityState.VIABILITY_ANOMALY}
    # Force substantial context-local action coverage and repeated identical terminal failures.
    for index in range(64, 160):
        terminal = index % 4 == 3
        profile = controller.observe(
            _transition(index, failure=terminal, branching=4, after=(index + 1) % 8),
            watermark=index + 1,
            policy_scores={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        )
    assert profile.state in {ViabilityState.LOW_EVIDENCE, ViabilityState.VIABILITY_ANOMALY}
    confidence_before = profile.viability_confidence

    # Direct progress remains authoritative and moves anomalous/weak profiles toward viability.
    for index in range(160, 168):
        profile = controller.observe(
            _transition(index, success=True, levels_completed=1, valence=1, after=(index + 1) % 8),
            watermark=index + 1,
            policy_scores={0: 0.8, 1: 0.1, 2: 0.0, 3: -0.1},
        )
    assert profile.state in {ViabilityState.RECOVERING, ViabilityState.VIABLE}
    assert profile.viability_confidence >= confidence_before

    state = controller.state_dict()
    restored = EnvironmentViabilityController(tmp_path / "restored")
    restored.load_state(state)
    restored_profile = restored.profiles[profile.environment_id]
    assert restored_profile.state is profile.state
    assert restored_profile.observations == profile.observations
    assert restored_profile.action_coverage == profile.action_coverage


def test_viability_state_changes_actor_exploration_pressure() -> None:
    viable = viability_adjusted_epsilon(
        0.05,
        8,
        state="VIABLE",
        coverage=1.0,
        uncertainty=0.0,
        learning_progress=1.0,
    )
    probing = viability_adjusted_epsilon(
        0.05,
        8,
        state="PROBING",
        coverage=0.1,
        uncertainty=1.0,
        learning_progress=0.0,
    )
    anomaly = viability_adjusted_epsilon(
        0.05,
        8,
        state="VIABILITY_ANOMALY",
        coverage=0.8,
        uncertainty=0.8,
        learning_progress=0.0,
    )
    assert viable < probing <= 0.50
    assert probing <= anomaly <= 0.50


def test_viability_confidence_preserves_direct_positive_evidence() -> None:
    runtime = SimpleNamespace(_environment_evidence_confidence={7: 0.20})
    negative = _row_confidence(runtime, {"environment_instance_id": 7, "primary_valence": 0})
    positive = _row_confidence(runtime, {"environment_instance_id": 7, "task_success": True, "primary_valence": 1})
    assert negative == 0.20
    assert positive == 1.0
