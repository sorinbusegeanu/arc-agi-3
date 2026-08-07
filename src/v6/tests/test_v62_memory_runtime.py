from __future__ import annotations

import sqlite3

from v6.memory.migrations.v62 import migrate_connection
from v6.memory.substrate import MemoryEdge, MemoryNode, MemoryScore, MemorySubstrate
from v6.memory.v62_runtime import HierarchicalSignificanceEngine, LearnedFutureOptionEstimator, V62PromotionEngine


def test_v62_migration_adds_hierarchical_fields() -> None:
    conn = sqlite3.connect(":memory:")
    memory = MemorySubstrate(conn)
    migrate_connection(conn)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memory_scores)")}
    assert {"hierarchical_score", "developmental_stage", "source_score_count", "score_version"}.issubset(columns)


def test_learned_future_options_use_observed_transition_graph() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE interactions(id INTEGER PRIMARY KEY, state_hash_before TEXT, action INTEGER, state_hash_after TEXT, outcome_state TEXT)")
    conn.executemany(
        "INSERT INTO interactions VALUES (?, ?, ?, ?, ?)",
        [(1, "s0", 1, "s1", "NOT_FINISHED"), (2, "s1", 2, "s2", "WIN")],
    )
    estimator = LearnedFutureOptionEstimator(conn)
    estimator._state_hash = lambda _: "s0"  # isolate reachability semantics
    option_set = estimator.estimate_option_set(object(), depth=2, available_actions=[1, 2])
    assert len(option_set.reachable_signatures) == 2
    assert option_set.option_set_id.startswith("fos:v62:")


def test_hierarchical_significance_propagates_lower_level_isf() -> None:
    conn = sqlite3.connect(":memory:")
    memory = MemorySubstrate(conn)
    migrate_connection(conn)
    memory.upsert_node(MemoryNode("M0:interaction:1", "M0", "InteractionMemory"))
    memory.upsert_node(MemoryNode("M1:contingency:x", "M1", "ContingencyMemory", attrs={"support_count": 3, "confidence": 0.9, "promotion_status": "accepted"}))
    memory.upsert_edge(MemoryEdge("M0:interaction:1", "M1:contingency:x", "supports"))
    memory.upsert_score(MemoryScore(node_id="M0:interaction:1", isf_total=0.8))
    result = HierarchicalSignificanceEngine(memory).rescore_all(step=1)
    row = conn.execute("SELECT hierarchical_score FROM memory_scores WHERE node_id='M1:contingency:x'").fetchone()
    assert result["scored"] >= 1
    assert row is not None and float(row[0]) > 0.0


def test_single_role_concept_fails_v62_policy() -> None:
    conn = sqlite3.connect(":memory:")
    memory = MemorySubstrate(conn)
    migrate_connection(conn)
    memory.upsert_node(
        MemoryNode(
            "M4:concept:single",
            "M4",
            "ConceptMemory",
            attrs={
                "source_roles": ["M3:role:one"],
                "transfer_tests": 4,
                "transfer_success_count": 4,
                "explanatory_reach": 4,
            },
        )
    )
    engine = V62PromotionEngine(memory)
    summary = engine._validate_levels({"M4"}, step=1)
    node = memory.get_node("M4:concept:single")
    assert summary["rejected"] == 1
    assert node["attrs"]["promotion_status"] == "rejected"
