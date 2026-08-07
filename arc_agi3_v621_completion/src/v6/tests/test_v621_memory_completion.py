from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from v6.memory.migrations.v621 import migrate_connection
from v6.memory.query_engine import MemoryActionScore, MemoryPrediction
from v6.memory.substrate import MemoryNode, MemoryScore, MemorySubstrate
from v6.memory.v621_compact import merge_v621_state_connections
from v6.memory.v621_runtime import (
    CachedAbstractionFutureOptionEstimator,
    HierarchicalMemoryLifecycleEngine,
    V621MemoryController,
    V621PromotionEngine,
    V621SnapshotMemoryQueryEngine,
)


def _encoded(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=int), allow_pickle=False)
    return buffer.getvalue()


def test_v621_migration_adds_completion_tables() -> None:
    connection = sqlite3.connect(":memory:")
    memory = MemorySubstrate(connection)
    result = migrate_connection(connection)
    assert result["schema_version"] == "v6.2.1"
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {
        "concept_transfer_attempts_v621",
        "world_model_relations_v621",
        "memory_level_lifecycle_v621",
        "memory_runtime_audit_v621",
    }.issubset(tables)


def test_cached_future_options_refresh_incrementally() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE interactions(
            id INTEGER PRIMARY KEY,
            action INTEGER,
            state_hash_before TEXT,
            state_hash_after TEXT,
            outcome_state TEXT,
            observation_before BLOB,
            observation_after BLOB
        )
        """
    )
    a = np.array([[0, 1], [0, 0]])
    b = np.array([[0, 1], [1, 0]])
    c = np.array([[1, 1], [1, 0]])
    connection.executemany(
        "INSERT INTO interactions VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 1, "s0", "s1", "NOT_FINISHED", _encoded(a), _encoded(b)),
            (2, 2, "s1", "s2", "WIN", _encoded(b), _encoded(c)),
        ],
    )
    estimator = CachedAbstractionFutureOptionEstimator(
        connection,
        refresh_interval=1,
    )
    estimator._state_hash = lambda value: "s0"
    result = estimator.estimate_option_set(
        a,
        depth=2,
        available_actions=[1, 2],
    )
    assert len(result.reachable_signatures) >= 2
    first_rows = estimator.rows_loaded

    connection.execute(
        "INSERT INTO interactions VALUES (?, ?, ?, ?, ?, ?, ?)",
        (3, 3, "s2", "s3", "WIN", _encoded(c), _encoded(c)),
    )
    connection.commit()
    estimator.force_refresh()
    assert estimator.rows_loaded == first_rows + 1


def test_higher_level_lifecycle_can_forget_low_value_memory() -> None:
    connection = sqlite3.connect(":memory:")
    memory = MemorySubstrate(connection)
    migrate_connection(connection)
    memory.upsert_node(
        MemoryNode(
            node_id="M4:concept:old",
            memory_level="M4",
            node_type="ConceptMemory",
            attrs={
                "promotion_status": "accepted",
                "transfer_tests": 6,
            },
        ),
        step=1,
    )
    memory.upsert_score(
        MemoryScore(
            node_id="M4:concept:old",
            isf_total=0.05,
            replay_priority=0.05,
        ),
        step=1,
    )
    summary = HierarchicalMemoryLifecycleEngine(memory).apply(step=1000)
    row = connection.execute(
        """
        SELECT memory_state
        FROM memory_scores
        WHERE node_id='M4:concept:old'
        """
    ).fetchone()
    assert summary["forgotten"] >= 1
    assert row[0] == "forgotten"


def test_concept_requires_direct_transfer_for_acceptance() -> None:
    connection = sqlite3.connect(":memory:")
    memory = MemorySubstrate(connection)
    migrate_connection(connection)
    memory.upsert_node(
        MemoryNode(
            node_id="M4:concept:v621",
            memory_level="M4",
            node_type="ConceptMemory",
            attrs={
                "source_roles": ["M3:role:a", "M3:role:b"],
                "structural_overlap_score": 0.5,
                "cross_game_evidence": True,
                "compression_gain": 0.5,
                "concept_version": "v621_relational_abstraction_v1",
            },
        )
    )
    engine = V621PromotionEngine(memory)
    first = engine._validate_levels({"M4"}, step=1)
    assert first["candidate"] == 1

    connection.executemany(
        """
        INSERT INTO concept_transfer_attempts_v621(
            attempt_id, concept_id, success,
            evidence_source, created_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [
            ("a1", "M4:concept:v621", 1, "test", 1.0),
            ("a2", "M4:concept:v621", 1, "test", 2.0),
        ],
    )
    second = engine._validate_levels({"M4"}, step=2)
    node = memory.get_node("M4:concept:v621")
    assert second["accepted"] == 1
    assert node["attrs"]["promotion_status"] == "accepted"


def test_m6_requires_equivalence_cost_and_reuse() -> None:
    connection = sqlite3.connect(":memory:")
    memory = MemorySubstrate(connection)
    migrate_connection(connection)
    memory.upsert_node(
        MemoryNode(
            node_id="M6:strategy:test",
            memory_level="M6",
            node_type="EfficientStrategyMemory",
            attrs={
                "success_rate": 1.0,
                "cost": 5.0,
                "best_known_length": 5,
                "effects": {"outcome": "win"},
                "reuse_count": 0,
            },
        )
    )
    engine = V621PromotionEngine(memory)
    first = engine._validate_levels({"M6"}, step=1)
    assert first["candidate"] == 1

    node = memory.get_node("M6:strategy:test")
    attrs = dict(node["attrs"])
    attrs["reuse_count"] = 1
    memory.update_node_support_and_attrs(
        "M6:strategy:test",
        attrs,
        support_increment=0,
    )
    second = engine._validate_levels({"M6"}, step=2)
    assert second["accepted"] == 1


def test_compact_extension_merge_preserves_v621_fields() -> None:
    source = sqlite3.connect(":memory:")
    target = sqlite3.connect(":memory:")
    source_memory = MemorySubstrate(source)
    target_memory = MemorySubstrate(target)
    migrate_connection(source)
    migrate_connection(target)

    for memory in (source_memory, target_memory):
        memory.upsert_node(
            MemoryNode(
                node_id="M4:concept:x",
                memory_level="M4",
                node_type="ConceptMemory",
            )
        )
        memory.upsert_score(
            MemoryScore(
                node_id="M4:concept:x",
                isf_total=0.5,
            )
        )
    source.execute(
        """
        UPDATE memory_scores
        SET hierarchical_score=0.91,
            developmental_stage='concept_transfer',
            score_version='v621'
        WHERE node_id='M4:concept:x'
        """
    )
    source.execute(
        """
        INSERT INTO memory_level_lifecycle_v621(
            memory_id, memory_level, memory_state,
            replay_priority, retention_score, forgetting_score,
            updated_at
        ) VALUES ('M4:concept:x', 'M4', 'protected',
                  0.91, 0.91, 0.09, 1.0)
        """
    )
    source.commit()

    merge_v621_state_connections(source, target)
    score = target.execute(
        """
        SELECT hierarchical_score, developmental_stage, score_version
        FROM memory_scores
        WHERE node_id='M4:concept:x'
        """
    ).fetchone()
    lifecycle = target.execute(
        """
        SELECT memory_state
        FROM memory_level_lifecycle_v621
        WHERE memory_id='M4:concept:x'
        """
    ).fetchone()
    assert score == (0.91, "concept_transfer", "v621")
    assert lifecycle[0] == "protected"


class _FakeSnapshotBase:
    def __init__(self, source_memory_dir: str) -> None:
        self.snapshot = SimpleNamespace(
            source_memory_dir=source_memory_dir,
            concept_ids_by_role={},
        )
        self.overlay = SimpleNamespace()

    def predict_family(
        self,
        context_signatures,
        action,
        *,
        record_query=False,
    ):
        return MemoryPrediction(None, 0.0, "none", [])

    def find_similar_roles(self, context_signature, action):
        return []

    def find_concept_matches(
        self,
        context_signature,
        action,
        *,
        role_matches=None,
    ):
        return []

    def _future_and_failure(self, context, action):
        return (
            {
                "expected_future_option_delta": 0.0,
                "completion_likelihood": 0.0,
                "sources": [],
            },
            {
                "failure_risk": 0.0,
                "contradiction_evidence": False,
                "sources": [],
            },
        )

    def _best_context_signature(self, context_signatures, action):
        return tuple(context_signatures[max(context_signatures)])

    def score_action(
        self,
        context_signatures,
        action,
        available_actions,
        *,
        record_query=False,
    ):
        return MemoryActionScore(
            action=int(action),
            score=0.0,
            predicted_family=None,
            expected_future_option_delta=0.0,
            failure_risk=0.0,
            completion_likelihood=0.0,
            evidence_sources=[],
        )

    def rank_actions(self, context_signatures_by_action, available_actions):
        return [
            self.score_action(
                context_signatures_by_action[action],
                action,
                available_actions,
            )
            for action in available_actions
        ]


def test_snapshot_wrapper_loads_m6_into_ram(tmp_path: Path) -> None:
    database = tmp_path / "current_state.sqlite"
    connection = sqlite3.connect(database)
    memory = MemorySubstrate(connection)
    migrate_connection(connection)
    memory.upsert_node(
        MemoryNode(
            node_id="M6:strategy:one",
            memory_level="M6",
            node_type="EfficientStrategyMemory",
            attrs={
                "promotion_status": "accepted",
                "action_sequence": [1, 2],
                "success_rate": 1.0,
                "cost": 2.0,
                "best_known_length": 2,
            },
        )
    )
    connection.commit()
    connection.close()

    wrapped = V621SnapshotMemoryQueryEngine(
        _FakeSnapshotBase(str(tmp_path))
    )
    result = wrapped.score_action(
        {0: (1,)},
        1,
        [1, 2],
    )
    assert result.score > 0.0
    assert "M6_strategy_memory_v621" in result.evidence_sources


class _Ranker:
    def __init__(self) -> None:
        self.last_strategy_by_action = {}

    def rank_actions(self, contexts, actions):
        return [
            MemoryActionScore(
                action=2,
                score=0.9,
                predicted_family=None,
                expected_future_option_delta=0.0,
                failure_risk=0.0,
                completion_likelihood=0.0,
                evidence_sources=["memory"],
            ),
            MemoryActionScore(
                action=1,
                score=0.2,
                predicted_family=None,
                expected_future_option_delta=0.0,
                failure_risk=0.0,
                completion_likelihood=0.0,
                evidence_sources=[],
            ),
        ]


def test_sampler_action_can_be_overridden_by_strong_memory_prior() -> None:
    connection = sqlite3.connect(":memory:")
    memory = MemorySubstrate(connection)
    controller = V621MemoryController(memory)
    controller.query_engine = _Ranker()
    selected = controller.choose_with_sampler_prior(
        context_signatures_by_action={
            1: {0: (1,)},
            2: {0: (2,)},
        },
        available_actions=[1, 2],
        sampler_action=1,
        override_margin=0.15,
    )
    assert selected == 2
