from __future__ import annotations

import sqlite3

from v7.memory.candidates.cpu import IndexedCpuCandidateProvider
from v7.memory.durable_store import DurableGenerationStore
from v7.memory.evidence_store import EvidenceRecord, EvidenceStore
from v7.memory.ids import MemoryIdAllocator, MemoryLevel
from v7.memory.models import EdgeMutation, NodeMutation, ScoreMutation
from v7.memory.writer import CanonicalMemoryWriter


def test_writer_coalesces_and_publishes_immutable_generation(tmp_path) -> None:
    ids = MemoryIdAllocator()
    source_id, target_id = ids.allocate_many(2)
    writer = CanonicalMemoryWriter()

    applied = writer.apply_mutation_batch(
        [
            NodeMutation(source_id, MemoryLevel.M1, 10, support_delta=1),
            NodeMutation(source_id, MemoryLevel.M1, 10, support_delta=2),
            NodeMutation(target_id, MemoryLevel.M2, 20, support_delta=1),
        ]
    )
    assert applied == 2
    writer.apply_edge_batch(
        [
            EdgeMutation(source_id, 7, target_id, support_delta=1),
            EdgeMutation(source_id, 7, target_id, support_delta=3),
        ]
    )
    writer.apply_score_batch(
        [
            ScoreMutation(source_id, significance=0.2),
            ScoreMutation(source_id, future_option_delta=1.5),
        ]
    )
    writer.observe_global_step(100)
    state, view = writer.commit_generation()

    assert int(state.generation_id) == 1
    assert view.nodes[source_id].support_count == 3
    assert view.scores[source_id].significance == 0.2
    assert view.scores[source_id].future_option_delta == 1.5
    assert view.neighbors([source_id], 7) == ((target_id,),)
    assert writer.dirty_counts == {"nodes": 0, "scores": 0, "edges": 0}

    durable = DurableGenerationStore(tmp_path / "state.sqlite")
    try:
        durable.persist_generation(state, view, writer.edge_support_snapshot())
        row = durable.connection.execute(
            "SELECT committed FROM generations WHERE generation_id=1"
        ).fetchone()
        assert row == (1,)
    finally:
        durable.close()


def test_evidence_is_separate_from_active_memory(tmp_path) -> None:
    memory_id = MemoryIdAllocator().allocate()
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    try:
        count = store.append_evidence_batch(
            [
                EvidenceRecord(
                    memory_id=memory_id,
                    evidence_type=3,
                    generation_id=1,
                    payload={"support": "observed"},
                    source_game="tt01",
                    source_context="ctx",
                    source_global_step=10,
                )
            ]
        )
        assert count == 1
        row = store.connection.execute(
            "SELECT memory_id, evidence_type, generation_id FROM evidence_records"
        ).fetchone()
        assert row == (int(memory_id), 3, 1)
    finally:
        store.close()


def test_candidate_provider_is_bounded() -> None:
    ids = MemoryIdAllocator()
    role_ids = ids.allocate_many(4)
    concept_ids = ids.allocate_many(4)
    key = (11, 2, 99)
    provider = IndexedCpuCandidateProvider(
        role_index={key: role_ids},
        concept_index={role_ids[0]: concept_ids},
    )

    assert provider.role_candidates([key], limit=2) == (role_ids[:2],)
    assert provider.concept_candidates([role_ids[0]], limit=3) == (concept_ids[:3],)


def test_v7_schema_has_no_v6_tables(tmp_path) -> None:
    store = DurableGenerationStore(tmp_path / "state.sqlite")
    try:
        tables = {
            row[0]
            for row in store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "memory_nodes" in tables
        assert "stable_contingencies" not in tables
        assert "transformation_families" not in tables
    finally:
        store.close()
