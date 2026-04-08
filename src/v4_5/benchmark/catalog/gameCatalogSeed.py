from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from v4_5.benchmark.db.store import utc_now_text


@dataclass(frozen=True)
class GameCatalogSeedEntry:
    game_id: str
    title: str
    description: str
    family: str
    in_benchmark: bool
    notes: str | None

    def to_row(self) -> dict[str, object]:
        timestamp = utc_now_text()
        return {
            "game_id": self.game_id,
            "title": self.title,
            "description": self.description,
            "family": self.family,
            "in_benchmark": self.in_benchmark,
            "notes": self.notes,
            "created_at": timestamp,
            "updated_at": timestamp,
        }


REPO_ROOT = Path(__file__).resolve().parents[4]
GAME_REFERENCE_PATH = REPO_ROOT / "docs" / "v4" / "reference" / "arc_interactive_games.md"
DEFAULT_ACTIVE_BENCHMARK_GAMES = {"ez01", "ez02", "ez03", "ez04"}


def _split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _parse_reference_catalog() -> tuple[GameCatalogSeedEntry, ...]:
    rows: list[GameCatalogSeedEntry] = []
    for raw_line in GAME_REFERENCE_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("| "):
            continue
        if line.startswith("| Game ") or line.startswith("|------"):
            continue
        cells = _split_markdown_row(line)
        if len(cells) < 7:
            continue
        game_id = cells[0]
        category = cells[1]
        grid = cells[2]
        level_count = cells[3]
        description = cells[4]
        actions = cells[6]
        rows.append(
            GameCatalogSeedEntry(
                game_id=game_id,
                title=game_id.upper(),
                description=description,
                family=game_id,
                in_benchmark=game_id in DEFAULT_ACTIVE_BENCHMARK_GAMES,
                notes=f"category={category};grid={grid};levels={level_count};actions={actions}",
            )
        )
    return tuple(rows)


GAME_CATALOG_SEED: tuple[GameCatalogSeedEntry, ...] = _parse_reference_catalog()
