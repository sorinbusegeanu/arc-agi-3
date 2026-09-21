from __future__ import annotations

import json

from v9.runtime.training_evidence import TrainingEvidenceKind, TrainingEvidenceRecord
from v9.memory.identity import MemoryUid
from v9.memory.model import CanonicalNode, MemoryLevel, MemoryType
from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig
from v9.tests.test_live_canonical_durability import _proposal


def test_training_evidence_identity_is_deterministic() -> None:
    kwargs = dict(
        kind=TrainingEvidenceKind.GROUNDING,
        source_wal_lsn=3,
        scientific_provenance={"experiment": "e", "producer": 7},
        schema_versions={"label": 1},
        label_payload={"grounded": True},
    )
    left = TrainingEvidenceRecord.create(**kwargs)
    right = TrainingEvidenceRecord.create(**kwargs)
    assert left == right
    assert TrainingEvidenceRecord.from_dict(left.as_dict()) == left


def test_scientific_evidence_identity_does_not_depend_on_wal_scheduling() -> None:
    kwargs = {
        "kind": "interaction",
        "scientific_provenance": {"producer": 7, "sequence": 11},
        "schema_versions": {"record": 1},
        "label_payload": {"action": 3},
    }
    early = TrainingEvidenceRecord.create(source_wal_lsn=4, **kwargs)
    delayed = TrainingEvidenceRecord.create(source_wal_lsn=19, **kwargs)
    assert early.evidence_id == delayed.evidence_id
    assert early.checksum != delayed.checksum


def test_canonical_updates_cover_every_training_evidence_kind(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(tmp_path, restore=False, enable_snapshots=False)
    )
    specifications = (
        (MemoryLevel.M0, MemoryType.EPISODE, {"producer_sequence": 1, "actor_id": 1, "symbol_identity": None}),
        (MemoryLevel.M1, MemoryType.GROUNDED_CONTINGENCY, {}),
        (MemoryLevel.M1, MemoryType.NORMALIZED_RELATION, {"heldout_transfer": True}),
        (MemoryLevel.M2, MemoryType.FAMILY, {}),
        (MemoryLevel.M3, MemoryType.ROLE, {}),
        (MemoryLevel.M4, MemoryType.CONCEPT, {"validated": True}),
        (MemoryLevel.M5, MemoryType.CONSEQUENCE, {}),
        (MemoryLevel.M6, MemoryType.OUTCOME, {}),
        (MemoryLevel.M7, MemoryType.STRATEGY, {}),
    )
    updates = {}
    for ordinal, (level, memory_type, payload) in enumerate(specifications, start=1):
        uid = MemoryUid(90, ordinal)
        updates[uid] = (CanonicalNode(uid, level, memory_type, (ordinal,), ordinal), payload)
    retired = MemoryUid(91, 1)
    records = runtime._training_records_for_canonical_update(
        target_lsn=3,
        proposal=_proposal(runtime.graph.partition_count),
        node_updates=updates,
        node_deletes={
            retired: CanonicalNode(retired, MemoryLevel.M0, MemoryType.EPISODE, (1,), 1)
        },
    )

    assert {row.kind for row in records} == set(TrainingEvidenceKind)
    assert max(len(json.dumps(row.as_dict()).encode()) for row in records) <= 4096
