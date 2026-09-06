from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from v8 import incremental_peer_drain_v862 as drain
from v8 import information_flow_diagnostics as flow
from v8.model import MemoryLevel, MemoryType, MemoryUid, RelationType
from v8.normalized_memory_v086 import _M2N_MARKER
from v8.peers_v82 import V82DevelopmentalPeerSupervisor
from v8.structural_events import NormalizedPrimitive, StructuralFact
from v8.tests.test_v8_2_semantics import edge, node


@pytest.fixture
def developmental_graph(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_AGI3_V8_ROOT", str(tmp_path))
    parents = tuple(
        node(MemoryLevel.M1, MemoryType.CONTINGENCY,
             (StructuralFact(NormalizedPrimitive.COMPONENT_RELOCATED, i).token,),
             support=6, future=1.0)
        for i in (1, 2)
    )
    family = node(MemoryLevel.M2, MemoryType.FAMILY,
                  (_M2N_MARKER | int(NormalizedPrimitive.COMPONENT_RELOCATED), 0),
                  support=12)
    rows = {row.uid: row for row in (*parents, family)}
    edges = [edge(family.uid, RelationType.EXPLAINS, parent.uid) for parent in parents]
    edges.extend(edge(parent.uid, RelationType.GAME_PROVENANCE, MemoryUid(0, i))
                 for i, parent in enumerate(parents, 1))
    pending = []
    view = SimpleNamespace(
        node_records=lambda **kwargs: tuple(row for row in rows.values()
            if kwargs.get("level") is None or int(row.level) == int(kwargs["level"])),
        edge_records=lambda: tuple(edges),
        source_games=lambda uid: frozenset({1, 2}),
    )
    supervisor = V82DevelopmentalPeerSupervisor(
        read_view=view, submit_proposal=pending.append,
        watermark=lambda: 10, generation=lambda: 1, interval_seconds=100.0,
    )

    def commit():
        for proposal in pending:
            row = rows.get(proposal.uid)
            if row is None:
                row = node(proposal.level, proposal.memory_type, proposal.key_parts,
                           support=0, significance=0, learning=0)
                row = replace(row, score_weight=0)
            rows[proposal.uid] = replace(
                row, support_count=row.support_count + proposal.support_delta,
                explanatory_sum=row.explanatory_sum + proposal.explanatory_sum,
                transfer_prior_sum=row.transfer_prior_sum + proposal.transfer_prior_sum,
                future_option_sum=row.future_option_sum + proposal.future_option_sum,
                score_weight=row.score_weight + proposal.score_weight,
            )
            if not proposal.parent_uid.is_zero:
                edges.append(edge(proposal.uid, proposal.relation_type, proposal.parent_uid))
        pending.clear()

    yield supervisor, rows, pending, commit
    supervisor.close()
    supervisor.ledger.close()


def test_normalized_generations_commit_between_immutable_cuts(
    developmental_graph, monkeypatch, tmp_path,
):
    supervisor, rows, pending, commit = developmental_graph
    monkeypatch.setenv("ARC_AGI3_V8_ROOT", str(tmp_path))
    flow.begin_run(tmp_path)
    cuts = []
    correspondence_cuts = []
    base = supervisor._process_formation
    correspondence = supervisor._process_correspondence

    def observe(cut, frozen):
        assert not pending, "the preceding generation must be committed"
        cuts.append(cut)
        base(cut, frozen)
        assert tuple(frozen.node_records()) == cut.nodes
        assert not any(p.uid in {row.uid for row in frozen.node_records()}
                       for p in pending if p.uid not in rows)

    monkeypatch.setattr(supervisor, "_process_formation", observe)

    def observe_correspondence(cut, frozen):
        correspondence_cuts.append(cut)
        assert cut is cuts[-1]
        correspondence(cut, frozen)

    monkeypatch.setattr(supervisor, "_process_correspondence", observe_correspondence)
    result = supervisor.run_until_stable(commit_proposals=commit)
    kinds = [{(int(row.level), int(row.memory_type)) for row in cut.nodes} for cut in cuts]
    carrier = (int(MemoryLevel.M3), int(MemoryType.CARRIER))
    role = (int(MemoryLevel.M3), int(MemoryType.ROLE))
    concept = (int(MemoryLevel.M4), int(MemoryType.CONCEPT))
    assert carrier not in kinds[0] and role not in kinds[0]
    assert carrier in kinds[1] and role not in kinds[1]
    assert role in kinds[2] and concept not in kinds[2]
    assert concept in kinds[3]
    assert correspondence_cuts == cuts
    assert result == "stable"
    assert not pending
    records = [json.loads(line) for line in (tmp_path / flow.LOG_NAME).read_text().splitlines()]
    records = [row for row in records if row["stage"] == "stabilization"]
    assert records[0]["new_m3_carrier_count"] == 2
    assert records[1]["new_m3_role_count"] >= 1
    assert records[2]["new_m4_count"] >= 1
    assert records[-1]["stop_reason"] == "stable"
    assert records[-1]["output_count"] == 0


def test_stabilization_hard_cap_and_cancellation_restore(developmental_graph, monkeypatch):
    supervisor, rows, pending, commit = developmental_graph
    supervisor.pause()
    supervisor._v841_peer_cancel.set()
    cycles = []

    def grow():
        from v8.developmental_cut import capture_developmental_cut
        supervisor._last_developmental_cut = capture_developmental_cut(
            supervisor.read_view, generation=1, watermark=10)
        row = node(MemoryLevel.M2, MemoryType.FAMILY, (len(cycles), 99))
        supervisor._submit(supervisor._existing_proposal(row))
        supervisor._cycles += 1
        cycles.append(row)

    monkeypatch.setattr(supervisor, "run_once", grow)
    assert supervisor.run_until_stable(100, commit_proposals=commit) == "max_cycles"
    assert len(cycles) == 8
    assert not pending
    assert supervisor._pause.is_set()
    assert supervisor._v841_peer_cancel.is_set()


def test_commit_failure_does_not_capture_another_cut(developmental_graph, monkeypatch):
    supervisor, rows, pending, commit = developmental_graph
    calls = []

    def fail():
        calls.append(1)
        if len(calls) > 1:
            raise TimeoutError("commit failed")
        commit()

    with pytest.raises(TimeoutError, match="commit failed"):
        supervisor.run_until_stable(commit_proposals=fail)
    assert len(calls) == 2
    assert supervisor.submit_proposal == pending.append
    assert not supervisor._v82_stabilizing


def test_incomplete_cycle_cannot_report_stability(developmental_graph, monkeypatch):
    from v8.developmental_cut import capture_developmental_cut

    supervisor, rows, pending, commit = developmental_graph

    def incomplete():
        supervisor._last_developmental_cut = capture_developmental_cut(
            supervisor.read_view, generation=1, watermark=10)

    monkeypatch.setattr(supervisor, "run_once", incomplete)
    with pytest.raises(RuntimeError, match="did not complete"):
        supervisor.run_until_stable(commit_proposals=commit)


def test_final_drain_stabilizes_once_and_keeps_peers_paused(developmental_graph, monkeypatch):
    supervisor, rows, pending, commit = developmental_graph
    runtime = SimpleNamespace(_sampling_complete=True, peers=supervisor)
    barriers = []

    def canonical_wait(runtime, **kwargs):
        assert not kwargs["settle_peers"] and not kwargs["resume_peers"]
        barriers.append(1)
        commit()

    monkeypatch.setattr(drain, "_BASE_RUNTIME_WAIT", canonical_wait)
    drain._runtime_wait_quiescent_v862(runtime)
    assert any(int(row.level) == int(MemoryLevel.M4) for row in rows.values())
    assert runtime._v82_developmental_finalized
    assert supervisor._pause.is_set()
    count = len(barriers)
    drain._runtime_wait_quiescent_v862(runtime)
    assert len(barriers) == count + 1


def test_runtime_final_drain_commits_normalized_generations(developmental_graph, tmp_path):
    from v8.model import proposal_fingerprint
    from v8.runtime import ContinuousMemoryRuntime, V8RuntimeConfig

    source, rows, pending, commit = developmental_graph
    rows = {uid: replace(row, fingerprint=proposal_fingerprint(
        row.level, row.memory_type, row.key_parts)) for uid, row in rows.items()}
    runtime = ContinuousMemoryRuntime(V8RuntimeConfig(
        root=tmp_path / "runtime", shards=1, stage_workers=1,
        stage_ring_capacity=256, shard_ring_capacity=1024,
        node_capacity_per_shard=5000, edge_capacity_per_shard=15000,
        action_capacity_per_shard=512, enable_snapshots=False, restore=False,
        peer_interval_seconds=100.0,
    ))
    try:
        runtime.peers.pause()
        runtime.start()
        for row in rows.values():
            proposal = runtime.peers._existing_proposal(row)
            runtime.submit_proposal(replace(
                proposal,
                support_delta=row.support_count, future_option_sum=row.future_option_sum,
                score_weight=row.score_weight, cognitive_state=row.cognitive_state,
                validation_state=row.validation_state,
            ))
        for relation in source.read_view.edge_records():
            runtime.submit_proposal(runtime.peers._existing_proposal(
                rows[relation.source_uid], parent_uid=relation.target_uid,
                relation_type=RelationType(relation.relation_type),
            ))
        runtime._sampling_complete = True
        runtime.peers._v841_peer_cancel.set()
        runtime.wait_quiescent(timeout=30)
        assert runtime._v82_developmental_finalized
        assert runtime.peers._pause.is_set()
        assert runtime.peers._v841_peer_cancel.is_set()
        assert runtime._is_quiescent()
        assert any(int(row.memory_type) == int(MemoryType.CONCEPT)
                   for row in runtime.read_view.node_records(level=MemoryLevel.M4))
    finally:
        runtime.close(normal=False)
