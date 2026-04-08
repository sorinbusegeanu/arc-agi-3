from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Iterator

from v4_5.benchmark.db.store import utc_now_text
from v4_5.memory.levelMemoryJson import (
    deserialize_memory_region,
    deserialize_memory_regions,
    serialize_memory_region,
    serialize_memory_regions,
)
from v4_5.memory.levelMemoryMerge import merge_level_memory
from v4_5.memory.levelMemorySchema import create_level_memory_schema
from v4_5.memory.levelMemoryTypes import LevelMemoryRecord


def initialize_level_memory_schema(db_path: Path | str) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    try:
        create_level_memory_schema(connection)
    finally:
        connection.close()


@dataclass
class LevelMemoryStore:
    db_path: Path | str

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        initialize_level_memory_schema(self.db_path)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def get_level_memory(self, game_id: str, level_id: str) -> LevelMemoryRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM level_memory WHERE game_id = ? AND level_id = ?",
                (game_id, level_id),
            ).fetchone()
        if row is None:
            return None
        return LevelMemoryRecord(
            game_id=row["game_id"],
            level_id=row["level_id"],
            memory_state=row["memory_state"] if "memory_state" in row.keys() else "hypothesis",
            avatar=deserialize_memory_region(row["avatar_json"]),
            hud_regions=deserialize_memory_regions(row["hud_regions_json"]),
            life_regions=deserialize_memory_regions(row["life_regions_json"]),
            progress_regions=deserialize_memory_regions(row["progress_regions_json"]),
            pois=deserialize_memory_regions(row["pois_json"]),
            exit_regions=deserialize_memory_regions(row["exit_regions_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            schema_version=row["schema_version"],
        )

    def has_level_memory(self, game_id: str, level_id: str) -> bool:
        return self.get_level_memory(game_id, level_id) is not None

    def upsert_level_memory(self, record: LevelMemoryRecord) -> LevelMemoryRecord:
        existing = self.get_level_memory(record.game_id, record.level_id)
        timestamp = utc_now_text()
        merged = merge_level_memory(
            existing,
            LevelMemoryRecord(
                game_id=record.game_id,
                level_id=record.level_id,
                memory_state=record.memory_state,
                avatar=record.avatar,
                hud_regions=record.hud_regions,
                life_regions=record.life_regions,
                progress_regions=record.progress_regions,
                pois=record.pois,
                exit_regions=record.exit_regions,
                created_at=(record.created_at or timestamp) if existing is None else existing.created_at,
                updated_at=timestamp,
                schema_version=record.schema_version,
            ),
        )
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO level_memory (
                    game_id, level_id, avatar_json, hud_regions_json, life_regions_json,
                    progress_regions_json, pois_json, exit_regions_json, memory_state, created_at, updated_at, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id, level_id) DO UPDATE SET
                    avatar_json = excluded.avatar_json,
                    hud_regions_json = excluded.hud_regions_json,
                    life_regions_json = excluded.life_regions_json,
                    progress_regions_json = excluded.progress_regions_json,
                    pois_json = excluded.pois_json,
                    exit_regions_json = excluded.exit_regions_json,
                    memory_state = excluded.memory_state,
                    updated_at = excluded.updated_at,
                    schema_version = excluded.schema_version
                """,
                (
                    merged.game_id,
                    merged.level_id,
                    serialize_memory_region(merged.avatar),
                    serialize_memory_regions(merged.hud_regions),
                    serialize_memory_regions(merged.life_regions),
                    serialize_memory_regions(merged.progress_regions),
                    serialize_memory_regions(merged.pois),
                    serialize_memory_regions(merged.exit_regions),
                    merged.memory_state,
                    merged.created_at,
                    merged.updated_at,
                    merged.schema_version,
                ),
            )
        return merged

    def list_game_levels(self, game_id: str) -> tuple[str, ...]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT level_id FROM level_memory WHERE game_id = ? ORDER BY level_id",
                (game_id,),
            ).fetchall()
        return tuple(row["level_id"] for row in rows)
