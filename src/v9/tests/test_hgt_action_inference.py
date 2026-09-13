from __future__ import annotations

from random import Random

from v9.cognition.action_selection import choose_action
from v9.hgt import training
from v9.memory import CanonicalNode, MemoryLevel, MemoryType
from v9.memory.relations import RelationEdge, RelationType
from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig
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
    assert ("MEMORY", "PROVENANCE", "MEMORY") in edge_indexes


def test_hgt_graph_sampling_keeps_relation_endpoints(monkeypatch) -> None:
    old = CanonicalNode.build(MemoryLevel.M1, MemoryType.NORMALIZED_RELATION, (1,), 1)
    recent = [CanonicalNode.build(MemoryLevel.M0, MemoryType.EPISODE, (100 + index,), 100 + index) for index in range(20)]
    source = recent[-1]
    edge = RelationEdge(source.uid, RelationType.PROVENANCE, old.uid)
    nodes = {old.uid: old, **{node.uid: node for node in recent}}
    payloads = {uid: {} for uid in nodes}
    payloads[source.uid] = {"action_id": 1, "environment_instance_id": 7, "context_signature": 9, "episode_id": 3, "primary_valence": 1}
    read_view = ReadView.build(1, nodes, payloads, {edge.key: edge}, {})
    monkeypatch.setattr(training, "_require_torch", lambda: (_FakeTorch, None, None))
    _x, edge_indexes, _y, _targets, _masks, _meta = training.build_hgt_graph(read_view, max_nodes=4)
    assert ("MEMORY", "PROVENANCE", "MEMORY") in edge_indexes


def test_hgt_promotion_compares_candidate_to_parent_on_same_current_graph() -> None:
    assert training._should_promote("hgt-000001", 0.80, 0.76)
    assert training._should_promote("hgt-000001", 0.80, 0.805)
    assert not training._should_promote("hgt-000001", 0.80, 0.82)
    assert training._should_promote(None, float("inf"), 1.20)
