from __future__ import annotations

from random import Random

import pytest

from v9.cognition.action_selection import choose_action
from v9.hgt import training
from v9.memory import CanonicalNode, MemoryLevel, MemoryType
from v9.memory.relations import RelationEdge, RelationType
from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig
from v9.runtime.actor_policy import ActorPolicySnapshot
from v9.runtime.read_view import ReadView


class _FakeTensor:
    def t(self):
        return self

    def contiguous(self):
        return self


class _FakeTorch:
    float32 = "float32"
    long = "long"
    bool = "bool"

    @staticmethod
    def tensor(_values, *, dtype):
        del dtype
        return _FakeTensor()

    @staticmethod
    def stack(_values, *, dim):
        del dim
        return _FakeTensor()


def test_learned_hgt_scores_influence_action_choice(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False))
    runtime.set_hgt_action_scores({7: {1: 0.2, 2: 0.9}})
    action = choose_action(
        runtime.read_view,
        (1, 2),
        rng=Random(0),
        epsilon=0.0,
        learned_scores=runtime.hgt_action_scores(7, (1, 2)),
        target_environment_id=7,
    )
    assert action == 2


def test_context_hgt_scores_override_environment_average(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False))
    runtime.set_hgt_action_scores(
        {7: {1: 0.8, 2: 0.2}},
        context_action_scores={7: {99: {1: -0.5, 2: 0.9}}},
    )
    snapshot = runtime.actor_policy_snapshot()
    assert snapshot.learned_scores(7, (1, 2), context_signature=99) == {1: -0.5, 2: 0.9}
    assert snapshot.learned_scores(7, (1, 2), context_signature=100) == {1: 0.8, 2: 0.2}


def test_hgt_action_scores_persist_in_snapshot(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False))
    runtime.set_hgt_action_scores({9: {3: 0.75}}, context_action_scores={9: {12: {3: 0.9}}})
    runtime.close(normal=True)
    restored = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path))
    assert restored.hgt_action_scores(9, (3, 4)) == {3: 0.75, 4: 0.0}
    assert restored.actor_policy_snapshot().learned_scores(9, (3, 4), context_signature=12) == {3: 0.9, 4: 0.0}


def test_hgt_graph_builder_accepts_immutable_read_view_edges(monkeypatch) -> None:
    source = CanonicalNode.build(MemoryLevel.M0, MemoryType.EPISODE, (1,), 1)
    target = CanonicalNode.build(MemoryLevel.M1, MemoryType.NORMALIZED_RELATION, (2,), 2)
    edge = RelationEdge(source.uid, RelationType.PROVENANCE, target.uid)
    read_view = ReadView.build(1, {source.uid: source, target.uid: target}, {source.uid: {}, target.uid: {}}, {edge.key: edge}, {})
    monkeypatch.setattr(training, "_require_torch", lambda: (_FakeTorch, None, None))
    _x, edge_indexes, _y, _targets, _masks, _meta = training.build_hgt_graph(read_view)
    assert isinstance(read_view.edges, tuple)
    assert ("M0_EPISODE", "PROVENANCE", "M1_NORMALIZED_RELATION") in edge_indexes


def test_hgt_behavior_rollback_restores_older_schema_parent_policy(tmp_path) -> None:
    import json
    torch = pytest.importorskip("torch")

    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False))
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    parent = "hgt-000001"
    current = "hgt-000002"
    torch.save(
        {
            "model_schema_version": training.MODEL_SCHEMA_VERSION - 2,
            "validation_loss": 0.5,
            "validation_accuracy": 0.7,
            "action_scores": {7: {1: 0.9, 2: 0.1}},
            "context_action_scores": {7: {11: {1: 0.8, 2: 0.2}}},
        },
        models / f"{parent}.pt",
    )
    (models / "hgt_manifest.json").write_text(
        json.dumps(
            {
                "model_schema_version": training.MODEL_SCHEMA_VERSION,
                "current_model_version": current,
                "current_checkpoint": f"models/{current}.pt",
                "parent_model_version": parent,
                "validation_loss": 0.6,
                "validation_accuracy": 0.6,
            }
        ),
        encoding="utf-8",
    )

    restored = training.rollback_hgt_model(runtime, root=tmp_path)
    assert restored == parent
    assert runtime.hgt_action_scores(7, (1, 2)) == {1: 0.9, 2: 0.1}
    manifest = json.loads((models / "hgt_manifest.json").read_text(encoding="utf-8"))
    assert manifest["current_model_version"] == parent


def test_unseen_actions_respect_epsilon_instead_of_forcing_exploration() -> None:
    snapshot = ActorPolicySnapshot.build(
        generation=1,
        normalized_action_supports={},
        hgt_action_scores={},
        model_version="test",
    )
    action = choose_action(
        snapshot,
        (1, 2),
        rng=Random(0),
        epsilon=0.0,
        learned_scores={1: 0.1, 2: 0.9},
        target_environment_id=7,
    )
    assert action == 2


def test_first_hgt_model_can_rollback_to_untrained(tmp_path) -> None:
    import json
    pytest.importorskip("torch")

    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False))
    runtime.set_hgt_action_scores({7: {1: 0.9}}, context_action_scores={7: {11: {1: 0.8}}})
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    (models / "hgt_manifest.json").write_text(
        json.dumps(
            {
                "model_schema_version": training.MODEL_SCHEMA_VERSION,
                "version_index": 1,
                "current_model_version": "hgt-000001",
                "current_checkpoint": "models/hgt-000001.pt",
                "parent_model_version": None,
            }
        ),
        encoding="utf-8",
    )

    restored = training.rollback_hgt_model(runtime, root=tmp_path)
    assert restored == "untrained"
    assert runtime.hgt_action_scores(7, (1,)) == {1: 0.0}
    assert runtime.actor_policy_snapshot().learned_scores(7, (1,), context_signature=11) == {1: 0.0}


def test_hgt_policy_scores_are_clamped_to_training_target_range() -> None:
    snapshot = ActorPolicySnapshot.build(
        generation=1,
        normalized_action_supports={},
        hgt_action_scores={7: {1: 4.0, 2: -3.0}},
        hgt_context_action_scores={7: {9: {1: 2.5, 2: -2.5}}},
        hgt_action_scores_by_type={"target": {1: 5.0, 2: -5.0}},
        model_version="test",
    )
    assert snapshot.learned_scores(7, (1, 2)) == {1: 1.0, 2: -1.0}
    assert snapshot.learned_scores(7, (1, 2), context_signature=9) == {1: 1.0, 2: -1.0}
