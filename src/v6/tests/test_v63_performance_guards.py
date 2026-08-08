from __future__ import annotations

import sqlite3
from pathlib import Path

from v6.memory.migrations.v61 import migrate_connection as migrate_v61
from v6.memory.migrations.v621 import migrate_connection as migrate_v621
from v6.memory.migrations.v63 import migrate_connection as migrate_v63
from v6.memory.substrate import MemorySubstrate
from v6.reporting.evidence_snapshot import memory_fingerprint, read_only_evidence_snapshot


def test_v61_migration_preserves_newer_schema_version() -> None:
    connection = sqlite3.connect(":memory:")
    MemorySubstrate(connection)
    connection.execute(
        "INSERT OR REPLACE INTO memory_versions(key, value) VALUES ('memory_substrate_schema', 'v6.3')"
    )
    migrate_v61(connection)
    row = connection.execute(
        "SELECT value FROM memory_versions WHERE key='memory_substrate_schema'"
    ).fetchone()
    assert row is not None
    assert row[0] == "v6.3"
    second = migrate_v61(connection)
    assert second["migration_applied"] is False


def test_v621_migration_fast_exits_for_newer_schema() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE memory_versions (key TEXT PRIMARY KEY, value TEXT)")
    connection.execute(
        "INSERT INTO memory_versions(key, value) VALUES ('memory_substrate_schema', 'v6.3')"
    )
    result = migrate_v621(connection)
    assert result["migration_applied"] is False
    assert result["schema_version"] == "v6.3"


def test_v63_installs_sampling_hot_path_index() -> None:
    connection = sqlite3.connect(":memory:")
    MemorySubstrate(connection)
    migrate_v63(connection)
    indexes = {
        row[1]
        for row in connection.execute("PRAGMA index_list(concept_transfer_attempts_v621)").fetchall()
    }
    assert "idx_v63_concept_transfer_step_game_created" in indexes
    assert "idx_v63_concept_transfer_evidence" in indexes
    second = migrate_v63(connection)
    assert second["migration_applied"] is False


def test_evidence_snapshot_excludes_noncanonical_recursive_artifacts(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    for name in ("current_state.sqlite", "graph.sqlite", "replay_queue.sqlite"):
        with sqlite3.connect(memory_dir / name) as connection:
            connection.execute("CREATE TABLE evidence(value INTEGER)")
            connection.execute("INSERT INTO evidence(value) VALUES (1)")
    (memory_dir / "memory_summary.json").write_text('{"ok": true}', encoding="utf-8")
    nested = memory_dir / "old_shards"
    nested.mkdir()
    with sqlite3.connect(nested / "stale.sqlite") as connection:
        connection.execute("CREATE TABLE stale(value INTEGER)")
    (nested / "bulk.parquet").write_bytes(b"not-report-evidence")

    with read_only_evidence_snapshot(memory_dir) as snapshot:
        assert snapshot is not None
        assert (snapshot / "current_state.sqlite").exists()
        assert (snapshot / "graph.sqlite").exists()
        assert (snapshot / "replay_queue.sqlite").exists()
        assert (snapshot / "memory_summary.json").exists()
        assert not (snapshot / "old_shards" / "stale.sqlite").exists()
        assert not (snapshot / "old_shards" / "bulk.parquet").exists()


def test_memory_fingerprint_tracks_metadata_changes(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    summary = memory_dir / "memory_summary.json"
    summary.write_text("{}", encoding="utf-8")
    before = memory_fingerprint(memory_dir)
    summary.write_text('{"changed": true}', encoding="utf-8")
    after = memory_fingerprint(memory_dir)
    assert before != after
