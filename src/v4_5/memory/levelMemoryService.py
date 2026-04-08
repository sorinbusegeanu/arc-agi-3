from __future__ import annotations

import json
from pathlib import Path

from v4_5.benchmark.db.store import REPO_ROOT, utc_now_text
from v4_5.logging import BoundAgentLogger
from v4_5.memory.levelMemoryStore import LevelMemoryStore, initialize_level_memory_schema
from v4_5.memory.levelMemoryTypes import LevelMemoryRecord


DEFAULT_LEVEL_MEMORY_DB = REPO_ROOT / "artifacts" / "v4_5" / "level_memory" / "level_memory.sqlite"


class LevelMemoryService:
    def __init__(self, db_path: str | None = None, logger: BoundAgentLogger | None = None) -> None:
        self.db_path = db_path or self._resolve_db_path()
        self.logger = logger
        self.store = LevelMemoryStore(self.db_path)

    def ensure_initialized(self) -> None:
        initialize_level_memory_schema(self.db_path)

    def load_level_memory(self, game_id: str, level_id: str) -> LevelMemoryRecord | None:
        if self.logger is not None:
            self.logger.info(game_id, "loading level memory", level_index=level_id)
        record = self.store.get_level_memory(game_id, level_id)
        if record is None and self.logger is not None:
            self.logger.info(game_id, "no stored level memory found", level_index=level_id)
        return record

    def save_level_memory(self, record: LevelMemoryRecord) -> LevelMemoryRecord:
        existing = self.store.get_level_memory(record.game_id, record.level_id)
        if self.logger is not None:
            self.logger.info(
                record.game_id,
                "updating stored level memory" if existing is not None else "saving level memory",
                level_index=record.level_id,
            )
        return self.store.upsert_level_memory(record)

    def save_hypothesis_level_memory(self, record: LevelMemoryRecord) -> LevelMemoryRecord:
        if self.logger is not None:
            self.logger.info(record.game_id, "saving hypothesis level memory", level_index=record.level_id)
        return self.save_level_memory(
            LevelMemoryRecord(
                game_id=record.game_id,
                level_id=record.level_id,
                memory_state="hypothesis",
                avatar=record.avatar,
                hud_regions=record.hud_regions,
                life_regions=record.life_regions,
                progress_regions=record.progress_regions,
                pois=record.pois,
                exit_regions=record.exit_regions,
                created_at=record.created_at,
                updated_at=record.updated_at,
                schema_version=record.schema_version,
            )
        )

    def promote_level_memory_to_validated(self, game_id: str, level_id: str) -> LevelMemoryRecord | None:
        record = self.store.get_level_memory(game_id, level_id)
        if record is None:
            return None
        if self.logger is not None:
            self.logger.info(game_id, "promoting level memory to validated", level_index=level_id)
        return self.store.upsert_level_memory(
            LevelMemoryRecord(
                game_id=record.game_id,
                level_id=record.level_id,
                memory_state="validated",
                avatar=record.avatar,
                hud_regions=record.hud_regions,
                life_regions=record.life_regions,
                progress_regions=record.progress_regions,
                pois=record.pois,
                exit_regions=record.exit_regions,
                created_at=record.created_at,
                updated_at=record.updated_at,
                schema_version=record.schema_version,
            )
        )

    def _resolve_db_path(self) -> str:
        config_path = REPO_ROOT / "src" / "v4_5" / "config" / "agents_config.json"
        if config_path.exists():
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            sqlite_path = payload.get("level_memory", {}).get("sqlite_path")
            if sqlite_path:
                return str((REPO_ROOT / sqlite_path).resolve()) if not Path(sqlite_path).is_absolute() else sqlite_path
        return str(DEFAULT_LEVEL_MEMORY_DB)
