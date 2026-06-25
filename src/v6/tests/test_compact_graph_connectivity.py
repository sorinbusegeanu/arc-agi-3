from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.contingency.contingency_learner import Contingency
from v6.graph.graph_manager import GraphManager
from v6.main import V6Config, V6System
from v6.memory.compact_memory import CompactMemoryFoldConfig, ensure_memory_layout, fold_epoch_raw_into_compact_memory, fold_live_system_into_compact_memory
from v6.transformation.transformation_clusterer import TransformationFamily


def _write_raw_db(path: Path, *, family_id: int, centroid_vector: list[float], support_count: int = 25) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE contingencies (
                id INTEGER PRIMARY KEY,
                context_signature TEXT,
                action INTEGER,
                transformation_family INTEGER,
                support_count INTEGER
            );
            CREATE TABLE prediction_results (
                interaction_id INTEGER,
                global_step INTEGER,
                context_signature TEXT,
                action INTEGER,
                predicted_family INTEGER,
                actual_family INTEGER,
                prediction_error INTEGER,
                isf_prediction_error REAL,
                memory_replay_priority REAL,
                context_contradiction INTEGER
            );
            CREATE TABLE interactions (
                id INTEGER PRIMARY KEY,
                global_step INTEGER,
                memory_replay_priority REAL,
                memory_replay_candidate INTEGER,
                carrier_signature TEXT,
                context_depth_used INTEGER
            );
            CREATE TABLE transformation_families (
                id INTEGER PRIMARY KEY,
                centroid_vector TEXT,
                support_count INTEGER,
                member_delta_ids TEXT
            );
            """
        )
        connection.execute("INSERT INTO contingencies VALUES (1, ?, 0, ?, ?)", ('["ctx",1]', family_id, support_count))
        connection.execute(
            "INSERT INTO prediction_results VALUES (1, 1, ?, 0, ?, ?, 1, 1.0, 0.9, 1)",
            ('["ctx",1]', family_id, family_id),
        )
        connection.execute("INSERT INTO interactions VALUES (1, 1, 0.9, 1, 'carrier-a', 1)")
        connection.execute(
            "INSERT INTO transformation_families VALUES (?, ?, ?, ?)",
            (family_id, json.dumps(centroid_vector), support_count, json.dumps([1])),
        )
        connection.commit()
    path.with_name("carrier_candidates.json").write_text(
        json.dumps(
            [
                {
                    "carrier_id": "carrier-a",
                    "carrier_signature": "carrier-a",
                    "carrier_source": "object",
                    "support_count": 4,
                    "distinct_family_count": 2,
                    "prediction_lift": 0.4,
                    "status": "emergent_carrier",
                    "context_signature": '["ctx",1]',
                    "family_id": family_id,
                }
            ]
        ),
        encoding="utf-8",
    )
    path.with_name("live_graph_compact.json").write_text(
        json.dumps(
            {
                "nodes": [{"node_id": "context:[\"ctx\",1]", "node_type": "context", "canonical_key": '["ctx",1]', "support_count": 1, "attrs_json": "{}"}],
                "edges": [{"source_node_id": "carrier:carrier-a", "target_node_id": "family:centroid:[2,1,0,0,0]", "edge_type": "explains", "weight": 1.0, "support_count": 1, "evidence_json": "{}"}],
            }
        ),
        encoding="utf-8",
    )


def test_compact_fold_uses_semantic_family_signatures_and_graph_connectivity(tmp_path: Path) -> None:
    raw_dir = tmp_path / "epoch_0001" / "raw"
    db_path = raw_dir / "sampling_v05c" / "tt01" / "mixed" / "steps_10" / "seed_0.sqlite"
    _write_raw_db(db_path, family_id=7, centroid_vector=[2, 1, 0, 0, 0])
    (raw_dir / "interaction_sampling_v05c_report.json").write_text(json.dumps({"validation": {"memory_record_count": 1}}), encoding="utf-8")
    memory = ensure_memory_layout(tmp_path / "memory")

    fold_epoch_raw_into_compact_memory(
        epoch_raw_dir=raw_dir,
        memory_dir=memory.root,
        fold_config=CompactMemoryFoldConfig(global_step_start=1, global_step_end=10),
    )

    with sqlite3.connect(memory.current_state) as connection:
        family_signatures = {row[0] for row in connection.execute("SELECT canonical_signature FROM transformation_families").fetchall()}
        assert "7" not in family_signatures
        assert any(signature.startswith("centroid:") for signature in family_signatures)
    with sqlite3.connect(memory.graph) as connection:
        node_types = {row[0] for row in connection.execute("SELECT node_id FROM graph_nodes").fetchall()}
        edge_types = {(row[0], row[1], row[2]) for row in connection.execute("SELECT source_node_id, target_node_id, edge_type FROM graph_edges").fetchall()}
        assert any(node.startswith("context:") for node in node_types)
        assert any(node.startswith("action:") for node in node_types)
        assert any(node.startswith("effect:") for node in node_types)
        assert any(node.startswith("contingency:") for node in node_types)
        assert any(node.startswith("family:") for node in node_types)
        assert any(node.startswith("carrier:") for node in node_types)
        assert any(edge[2] == "member_of" for edge in edge_types)
        assert any(edge[2] == "explains" for edge in edge_types)
        assert any(edge[2] == "appears_in" for edge in edge_types)
    with sqlite3.connect(memory.current_state) as connection:
        links = {(row[0], row[1], row[2]) for row in connection.execute("SELECT carrier_signature, linked_type, linked_key FROM carrier_links").fetchall()}
        assert any(link[1] == "family" for link in links)
        assert any(link[1] == "context" for link in links)


def test_same_centroid_merges_and_different_centroid_does_not(tmp_path: Path) -> None:
    raw_dir = tmp_path / "epoch_0001" / "raw"
    _write_raw_db(raw_dir / "sampling_v05c" / "tt01" / "mixed" / "steps_10" / "seed_0.sqlite", family_id=7, centroid_vector=[2, 1, 0, 0, 0])
    _write_raw_db(raw_dir / "sampling_v05c" / "pb02" / "mixed" / "steps_10" / "seed_0.sqlite", family_id=7, centroid_vector=[2, 1, 0, 0, 0])
    _write_raw_db(raw_dir / "sampling_v05c" / "gr01" / "mixed" / "steps_10" / "seed_0.sqlite", family_id=7, centroid_vector=[9, 9, 0, 0, 0])
    (raw_dir / "interaction_sampling_v05c_report.json").write_text(json.dumps({"validation": {"memory_record_count": 3}}), encoding="utf-8")
    memory = ensure_memory_layout(tmp_path / "memory")

    fold_epoch_raw_into_compact_memory(
        epoch_raw_dir=raw_dir,
        memory_dir=memory.root,
        fold_config=CompactMemoryFoldConfig(global_step_start=1, global_step_end=10),
    )

    with sqlite3.connect(memory.current_state) as connection:
        family_signatures = [row[0] for row in connection.execute("SELECT canonical_signature FROM transformation_families ORDER BY canonical_signature").fetchall()]
        assert len(family_signatures) == 2
        assert family_signatures[0] != family_signatures[1]
    with sqlite3.connect(memory.graph) as connection:
        explain_targets = [row[0] for row in connection.execute("SELECT target_node_id FROM graph_edges WHERE source_node_id = 'carrier:carrier-a' AND edge_type = 'explains'").fetchall()]
        assert len(set(explain_targets)) >= 2
        assert all(target != "family:family_id:1" for target in explain_targets)


def test_replay_and_contradiction_graph_connectivity(tmp_path: Path) -> None:
    raw_dir = tmp_path / "epoch_0001" / "raw"
    db_path = raw_dir / "sampling_v05c" / "tt01" / "mixed" / "steps_10" / "seed_0.sqlite"
    _write_raw_db(db_path, family_id=3, centroid_vector=[2, 1, 0, 0, 0])
    with sqlite3.connect(db_path) as connection:
        connection.execute("ALTER TABLE prediction_results ADD COLUMN context_contradiction_key TEXT")
        connection.execute("UPDATE prediction_results SET context_contradiction_key = 'contradiction-a'")
        connection.commit()
    (raw_dir / "interaction_sampling_v05c_report.json").write_text(json.dumps({"validation": {"memory_record_count": 1}}), encoding="utf-8")
    memory = ensure_memory_layout(tmp_path / "memory")

    fold_epoch_raw_into_compact_memory(
        epoch_raw_dir=raw_dir,
        memory_dir=memory.root,
        fold_config=CompactMemoryFoldConfig(global_step_start=1, global_step_end=10),
    )

    with sqlite3.connect(memory.graph) as connection:
        edges = {(row[0], row[1], row[2]) for row in connection.execute("SELECT source_node_id, target_node_id, edge_type FROM graph_edges").fetchall()}
        assert any(edge[0].startswith("contradiction:") and edge[1].startswith("replay:") and edge[2] == "prioritizes" for edge in edges)
        assert any(edge[0].startswith("replay:") and edge[1].startswith("context:") and edge[2] == "replays" for edge in edges)
        assert any(edge[0].startswith("replay:") and edge[1].startswith("family:") and edge[2] == "replays" for edge in edges)
        assert any(edge[1].startswith("contradiction:") and edge[2] == "contradicts" for edge in edges)
        assert any(edge[2] == "contradicted_by" for edge in edges)
        assert any(edge[2] == "challenges" for edge in edges)


def test_live_graph_export_writes_compact_rows() -> None:
    graph = GraphManager()
    graph.import_node("carrier:carrier-a", "carrier", "carrier-a")
    graph.import_node("family:family-a", "family", "family-a")
    graph.import_edge("carrier:carrier-a", "family:family-a", "explains", support_count=2)

    payload = graph.export_compact_rows()

    assert payload["nodes"]
    assert payload["edges"]


def test_live_system_fold_uses_semantic_family_signature(tmp_path: Path) -> None:
    class _Env:
        def __init__(self) -> None:
            import numpy as np

            self.grid = np.zeros((2, 2), dtype=int)

        def observe(self):
            return self.grid.copy()

        def step(self, action: int):
            return self.grid.copy()

        def available_actions(self) -> list[int]:
            return [0]

    system = V6System(env=_Env(), config=V6Config(database_path=":memory:"))
    system.clusterer.import_family(TransformationFamily(id=5, centroid_vector=[2.0, 1.0, 0.0, 0.0, 0.0], support_count=10, member_delta_ids=[]))
    system.contingency_learner.import_contingency(
        Contingency(id=1, context_level=1, context_signature=("ctx",), action=0, transformation_family=5, support_count=25, confidence=1.0)
    )
    try:
        fold_live_system_into_compact_memory(system, tmp_path / "memory")
    finally:
        system.close()
    with sqlite3.connect(tmp_path / "memory" / "current_state.sqlite") as connection:
        row = connection.execute("SELECT effect_signature FROM stable_contingencies").fetchone()
        assert row is not None
        assert str(row[0]).startswith("centroid:")
        assert str(row[0]) != "family_sig:5"
