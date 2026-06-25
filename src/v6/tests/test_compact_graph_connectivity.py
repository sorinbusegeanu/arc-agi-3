from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.graph.graph_manager import GraphManager
from v6.memory.compact_memory import CompactMemoryFoldConfig, ensure_memory_layout, fold_epoch_raw_into_compact_memory


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


def test_live_graph_export_writes_compact_rows() -> None:
    graph = GraphManager()
    graph.import_node("carrier:carrier-a", "carrier", "carrier-a")
    graph.import_node("family:family-a", "family", "family-a")
    graph.import_edge("carrier:carrier-a", "family:family-a", "explains", support_count=2)

    payload = graph.export_compact_rows()

    assert payload["nodes"]
    assert payload["edges"]
