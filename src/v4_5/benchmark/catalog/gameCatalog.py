from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from v4_5.benchmark.catalog.gameCatalogSeed import GAME_CATALOG_SEED
from v4_5.benchmark.db.query import list_active_benchmark_games, list_all_games
from v4_5.benchmark.db.store import BenchmarkStore


@dataclass
class GameCatalog:
    store: BenchmarkStore

    def sync_catalog_from_seed(self) -> None:
        self.store.upsert_games(entry.to_row() for entry in GAME_CATALOG_SEED)

    def initialize_catalog_if_empty(self) -> None:
        if self.store.is_catalog_empty():
            self.sync_catalog_from_seed()

    def list_all_games(self) -> list[dict[str, Any]]:
        return list_all_games(self.store)

    def list_active_benchmark_games(self) -> list[dict[str, Any]]:
        return list_active_benchmark_games(self.store)

    def update_benchmark_participation_flag(self, game_id: str, in_benchmark: bool) -> None:
        self.store.update_game_benchmark_flag(game_id, in_benchmark)

    def update_game_metadata(
        self,
        game_id: str,
        *,
        family: str | None = None,
        description: str | None = None,
        notes: str | None = None,
    ) -> None:
        self.store.update_game_metadata(game_id, family=family, description=description, notes=notes)

    def get_game(self, game_id: str) -> dict[str, Any] | None:
        return self.store.fetch_one("SELECT * FROM games WHERE game_id = ?", (game_id,))
