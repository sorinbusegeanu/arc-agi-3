from __future__ import annotations

import json
import sqlite3

from v7.memory.evidence_lifecycle import EvidenceLifecycleStore, ProvenanceRecord
from v7.memory.ids import MemoryId


def _gate_row(memory_id: int) -> tuple[object, ...]:
    return (
        memory_id,
        4,
        1,
        "target",
        f"context-{memory_id}",
        1,
        0.2,
        0.3,
        0.1,
        0.1,
        0.1,
        1.0,
        0.0,
        "decision_score_ablation",
        f"pair-{memory_id}",
        1,
        1,
        0.42,
        "{}",
        2,
    )


def test_bulk_lifecycle_queries_respect_sqlite_variable_limit(tmp_path) -> None:
    store = EvidenceLifecycleStore(tmp_path / "lifecycle.sqlite")
    try:
        store.connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 8)
        memory_ids = tuple(MemoryId(value) for value in range(1, 25))

        assert store.transfer_summary(memory_ids) == {}
        assert store.heldout_transfer_summary(memory_ids) == {}
        assert store.contradiction_summary(memory_ids) == {}
        assert store.gate_trial_summary(memory_ids) == {}
    finally:
        store.close()


def test_chunked_lifecycle_queries_merge_results_across_chunks(tmp_path) -> None:
    store = EvidenceLifecycleStore(tmp_path / "lifecycle.sqlite")
    try:
        selected = (1, 9, 17)
        store.append_provenance(
            ProvenanceRecord(
                memory_id=MemoryId(memory_id),
                generation_id=1,
                source_game="source",
                source_context=f"source-{memory_id}",
            )
            for memory_id in selected
        )
        with store.connection:
            store.connection.executemany(
                "INSERT INTO transfer_trials(memory_id,source_game,target_game,success,score,payload_json,generation_id) VALUES (?,?,?,?,?,?,?)",
                [
                    (
                        memory_id,
                        "source",
                        "target",
                        1,
                        0.5 + memory_id / 100.0,
                        json.dumps({"source_global_step": memory_id}),
                        2,
                    )
                    for memory_id in selected
                ],
            )
            store.connection.executemany(
                "INSERT INTO contradiction_records(memory_id,severity,source_game,source_context,source_global_step,payload_json,generation_id) VALUES (?,?,?,?,?,?,?)",
                [
                    (
                        memory_id,
                        memory_id / 100.0,
                        "source",
                        f"context-{memory_id}",
                        memory_id,
                        "{}",
                        2,
                    )
                    for memory_id in selected
                ],
            )
            store.connection.executemany(
                """
                INSERT INTO gate_trials(
                    memory_id,gate_id,candidate_generation,target_game,target_context,
                    participated,contribution,causal_gain,prediction_gain,planning_gain,
                    future_option_gain,terminal_gain,efficiency_gain,intervention_type,
                    paired_trial_id,genuine,success,transfer_score,payload_json,generation_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [_gate_row(memory_id) for memory_id in selected],
            )

        store.connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 8)
        memory_ids = tuple(MemoryId(value) for value in range(1, 25))

        transfer = store.transfer_summary(memory_ids)
        heldout = store.heldout_transfer_summary(memory_ids)
        contradiction = store.contradiction_summary(memory_ids)
        gates = store.gate_trial_summary(memory_ids)

        assert set(map(int, transfer)) == set(selected)
        assert set(map(int, heldout)) == set(selected)
        assert set(map(int, contradiction)) == set(selected)
        assert set(map(int, gates)) == set(selected)
        for memory_id in selected:
            mid = MemoryId(memory_id)
            assert transfer[mid][0:2] == (1, 1)
            assert heldout[mid][0:2] == (1, 1)
            assert contradiction[mid] == (1, memory_id / 100.0)
            assert gates[mid].trials == 1
            assert gates[mid].successes == 1
            assert gates[mid].mean_causal_gain == 0.3
    finally:
        store.close()
