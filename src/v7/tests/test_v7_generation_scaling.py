from __future__ import annotations

from v7.derivation.online_runtime import OnlineHierarchyBuilder
from v7.derivation.pipeline import MemoryLearningPipeline
from v7.developmental_v707 import (
    CarrierPersistenceRuntime,
    TrajectoryEfficiencyRuntime,
)
from v7.memory.evidence_lifecycle import EvidenceLifecycleStore
from v7.memory.evidence_store import EvidenceRecord, EvidenceStore
from v7.memory.evidence_types import EvidenceType
from v7.memory.structural_gate_runtime import StructuralGateRuntime
from v7.memory.writer import CanonicalMemoryWriter


def _episode(
    *,
    step: int,
    carrier: int,
    outcome: int = 7,
) -> EvidenceRecord:
    return EvidenceRecord(
        memory_id=None,
        evidence_type=int(EvidenceType.EPISODE),
        generation_id=1,
        source_game="g",
        source_global_step=step,
        payload={
            "carrier_signature": carrier,
            "outcome_signature": outcome,
            "context_signatures": [step],
            "next_context_signatures": [step + 1],
        },
    )


def _trajectory(step: int) -> EvidenceRecord:
    return EvidenceRecord(
        memory_id=None,
        evidence_type=int(EvidenceType.TRAJECTORY),
        generation_id=1,
        source_game="g",
        source_context=f"level-{step}",
        source_global_step=step,
        payload={
            "level_key": f"level-{step}",
            "action_sequence": [1, 2, step],
            "context_sequence": [10, 11, 10 + step],
            "future_option_sum": 2.0,
            "raw_action_option_sum": 0.0,
        },
    )


def test_carrier_canonicalization_scans_link_table_once_per_load(tmp_path) -> None:
    evidence = EvidenceStore(tmp_path / "evidence.sqlite")
    lifecycle = EvidenceLifecycleStore(tmp_path / "lifecycle.sqlite")
    try:
        runtime = CarrierPersistenceRuntime(lifecycle, evidence)
        with lifecycle.connection:
            lifecycle.connection.executemany(
                "INSERT INTO carrier_persistence_links("
                "carrier_a,carrier_b,support_count,first_generation,last_generation,predictive_gain) "
                "VALUES (?,?,?,?,?,?)",
                (
                    (11, 22, 2, 1, 1, 1.0),
                    (22, 33, 2, 1, 1, 1.0),
                ),
            )
        evidence.append_evidence_batch(
            _episode(step=step, carrier=11 if step % 2 else 33)
            for step in range(1, 201)
        )
        writer = CanonicalMemoryWriter()
        pipeline = MemoryLearningPipeline(writer, lifecycle, evidence)
        hierarchy = OnlineHierarchyBuilder(writer, pipeline, evidence, lifecycle)
        statements: list[str] = []
        lifecycle.connection.set_trace_callback(statements.append)
        rows = hierarchy._load(EvidenceType.EPISODE)
        lifecycle.connection.set_trace_callback(None)
        link_scans = [
            statement
            for statement in statements
            if "FROM carrier_persistence_links" in statement
        ]
        assert len(link_scans) == 1
        assert len({int(row["carrier_signature"]) for row in rows}) == 1
        assert runtime.canonical_carrier_signature(11) == runtime.canonical_carrier_signature(33)
    finally:
        evidence.close()
        lifecycle.close()


def test_carrier_persistence_processes_only_new_episode_suffix(tmp_path) -> None:
    evidence = EvidenceStore(tmp_path / "evidence.sqlite")
    lifecycle = EvidenceLifecycleStore(tmp_path / "lifecycle.sqlite")
    try:
        runtime = CarrierPersistenceRuntime(lifecycle, evidence)
        writer = CanonicalMemoryWriter()
        evidence.append_evidence_batch(
            (_episode(step=1, carrier=11), _episode(step=2, carrier=22))
        )
        assert runtime.run(writer=writer) == ()
        evidence.append_evidence_batch((_episode(step=3, carrier=11),))
        links = runtime.run(writer=writer)
        assert len(links) == 1
        assert links[0].support_count == 2
        assert runtime.run(writer=writer) == ()
        assert lifecycle.connection.execute(
            "SELECT support_count FROM carrier_persistence_links"
        ).fetchone() == (2,)
    finally:
        evidence.close()
        lifecycle.close()


def test_trajectory_efficiency_processes_only_new_evidence(tmp_path) -> None:
    evidence = EvidenceStore(tmp_path / "evidence.sqlite")
    lifecycle = EvidenceLifecycleStore(tmp_path / "lifecycle.sqlite")
    try:
        runtime = TrajectoryEfficiencyRuntime(lifecycle, evidence)
        writer = CanonicalMemoryWriter()
        evidence.append_evidence_batch((_trajectory(1),))
        assert len(runtime.run(writer=writer)) == 1
        assert runtime.run(writer=writer) == ()
        evidence.append_evidence_batch((_trajectory(2),))
        assert len(runtime.run(writer=writer)) == 1
        assert lifecycle.connection.execute(
            "SELECT COUNT(*) FROM trajectory_efficiency_trials"
        ).fetchone() == (2,)
    finally:
        evidence.close()
        lifecycle.close()


def test_structural_gate_runtime_reads_only_new_episode_suffix(tmp_path) -> None:
    evidence = EvidenceStore(tmp_path / "evidence.sqlite")
    lifecycle = EvidenceLifecycleStore(tmp_path / "lifecycle.sqlite")
    try:
        runtime = StructuralGateRuntime(
            evidence_store=evidence,
            lifecycle_store=lifecycle,
        )
        writer = CanonicalMemoryWriter()
        evidence.append_evidence_batch(
            _episode(step=step, carrier=step % 3) for step in range(1, 101)
        )
        runtime.run(writer)
        evidence.append_evidence_batch(
            _episode(step=step, carrier=step % 3) for step in range(101, 106)
        )
        statements: list[str] = []
        evidence.connection.set_trace_callback(statements.append)
        runtime.run(writer)
        evidence.connection.set_trace_callback(None)
        reads = [
            statement
            for statement in statements
            if "FROM evidence_records" in statement
        ]
        assert len(reads) == 1
        assert "evidence_id>100" in reads[0].replace(" ", "")
        assert runtime._last_evidence_id == 105
    finally:
        evidence.close()
        lifecycle.close()


def test_lifecycle_windows_can_commit_as_one_writer_batch(tmp_path) -> None:
    lifecycle = EvidenceLifecycleStore(tmp_path / "lifecycle.sqlite")
    try:
        statements: list[str] = []
        lifecycle.connection.set_trace_callback(statements.append)
        with lifecycle.connection:
            for memory_id in range(1, 51):
                lifecycle.update_lifecycle_window(
                    memory_id,
                    generation_id=1,
                    utility=0.5,
                    harm=False,
                    commit=False,
                )
        lifecycle.connection.set_trace_callback(None)
        assert sum(statement == "COMMIT" for statement in statements) == 1
        assert lifecycle.connection.execute(
            "SELECT COUNT(*) FROM lifecycle_windows"
        ).fetchone() == (50,)
    finally:
        lifecycle.close()
