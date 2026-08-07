from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.memory.migrations.v61 import migrate_connection
from v6.memory.substrate import (
    MemoryEdge,
    MemoryNode,
    MemoryPromotion,
    MemoryScore,
    MemorySubstrate,
)


def test_v61_additive_memory_schema_and_identity() -> None:
    connection = sqlite3.connect(":memory:")
    memory = MemorySubstrate(connection)

    memory.upsert_node(
        MemoryNode(
            node_id="M4:concept:test",
            memory_level="M4",
            node_type="ConceptMemory",
            canonical_key="test",
            created_epoch=1,
            updated_epoch=2,
            status="active",
        ),
        step=10,
    )
    node = memory.get_node("M4:concept:test")
    assert node is not None
    assert node["schema_version"] == "v6.1"
    assert node["evidence_version"] == "v1"
    assert node["created_epoch"] == 1
    assert node["updated_epoch"] == 2
    assert node["status"] == "active"


def test_edge_lifecycle_fields_and_validation() -> None:
    connection = sqlite3.connect(":memory:")
    memory = MemorySubstrate(connection)
    memory.upsert_edge(
        MemoryEdge(
            "M1:a",
            "M2:b",
            "promoted_from",
            edge_status="candidate",
            edge_confidence=0.4,
            edge_source="fixture",
            specificity_score=0.5,
        )
    )
    memory.validate_edge(
        "M1:a",
        "M2:b",
        "promoted_from",
        accepted=True,
        confidence=0.9,
        specificity_score=0.8,
        epoch=3,
    )
    edge = memory.edges_from("M1:a")[0]
    assert edge["edge_status"] == "accepted"
    assert edge["edge_confidence"] == 0.9
    assert edge["last_validated_epoch"] == 3


def test_promotion_and_lifecycle_audit_rows() -> None:
    connection = sqlite3.connect(":memory:")
    memory = MemorySubstrate(connection)
    memory.record_promotion(
        MemoryPromotion(
            promotion_id="promotion:test",
            source_node_id="M3:role:a",
            target_node_id="M4:concept:b",
            promotion_type="M3_ROLE_M4_CONCEPT",
            evidence_count=4,
            promotion_score=0.8,
            status="recorded",
            source_memory_ids=("M3:role:a", "M3:role:c"),
            compression_gain=0.2,
            prediction_lift=0.1,
            transfer_score=0.7,
            explanatory_reach=5.0,
            epoch=2,
            global_step=100,
        )
    )
    row = connection.execute(
        """
        SELECT source_memory_ids_json, compression_gain,
               prediction_lift, transfer_score,
               explanatory_reach, epoch, global_step
        FROM memory_promotions
        WHERE promotion_id='promotion:test'
        """
    ).fetchone()
    assert json.loads(row[0]) == ["M3:role:a", "M3:role:c"]
    assert tuple(row[1:]) == (0.2, 0.1, 0.7, 5.0, 2, 100)
    lifecycle_count = connection.execute(
        """
        SELECT COUNT(*) FROM memory_lifecycle_events
        WHERE memory_id='M4:concept:b'
          AND event_type='promoted'
        """
    ).fetchone()[0]
    assert lifecycle_count == 1


def test_memory_score_status_change_creates_lifecycle_event() -> None:
    connection = sqlite3.connect(":memory:")
    memory = MemorySubstrate(connection)
    memory.upsert_score(
        MemoryScore(
            node_id="M0:interaction:g1",
            memory_state="active",
            stored_epoch=1,
        ),
        step=1,
    )
    memory.upsert_score(
        MemoryScore(
            node_id="M0:interaction:g1",
            memory_state="forgotten",
            forgetting_reason="low_utility",
            stored_epoch=2,
        ),
        step=2,
    )
    rows = connection.execute(
        """
        SELECT event_type, previous_status, new_status
        FROM memory_lifecycle_events
        WHERE memory_id='M0:interaction:g1'
        ORDER BY created_at
        """
    ).fetchall()
    assert rows[-1] == ("forgotten", "active", "forgotten")


def test_migration_is_idempotent() -> None:
    connection = sqlite3.connect(":memory:")
    MemorySubstrate(connection)
    first = migrate_connection(connection)
    second = migrate_connection(connection)
    assert first["schema_version"] == "v6.1"
    assert second["schema_version"] == "v6.1"
