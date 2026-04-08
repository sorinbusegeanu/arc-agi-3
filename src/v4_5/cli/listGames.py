from __future__ import annotations

from pathlib import Path

from v4_5.benchmark.catalog.gameCatalog import GameCatalog
from v4_5.benchmark.db.store import BenchmarkStore, default_db_path


def list_games(*, db_path: str | None = None) -> list[dict]:
    store = BenchmarkStore(db_path=Path(db_path) if db_path else default_db_path())
    catalog = GameCatalog(store)
    catalog.sync_catalog_from_seed()
    rows = catalog.list_all_games()
    return [
        {
            "game_id": row["game_id"],
            "title": row["title"],
            "family": row["family"],
            "in_benchmark": row["in_benchmark"],
        }
        for row in rows
    ]
