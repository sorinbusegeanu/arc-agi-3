from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GameSetManifest:
    name: str
    games: tuple[str, ...]
    families: dict[str, tuple[str, ...]]
    purpose: str
    core7_anchors: tuple[str, ...] = ()


def load_game_set_manifest(
    *,
    manifest_path: str | None = None,
    game_set_name: str | None = None,
    fallback_games: tuple[str, ...] = (),
) -> GameSetManifest:
    path = resolve_manifest_path(manifest_path=manifest_path, game_set_name=game_set_name)
    if path is None or not path.exists():
        return GameSetManifest(
            name=game_set_name or "adhoc",
            games=tuple(fallback_games),
            families={},
            purpose="ad hoc game selection",
            core7_anchors=(),
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return GameSetManifest(
        name=str(raw.get("name", path.stem)),
        games=tuple(str(game) for game in raw.get("games", [])),
        families={str(key): tuple(str(game) for game in values) for key, values in raw.get("families", {}).items()},
        purpose=str(raw.get("purpose", "")),
        core7_anchors=tuple(str(game) for game in raw.get("core7_anchors", [])),
    )


def resolve_manifest_path(*, manifest_path: str | None, game_set_name: str | None) -> Path | None:
    if manifest_path:
        return Path(manifest_path)
    if game_set_name:
        return Path("runs/v6/game_sets") / f"{game_set_name}.json"
    return None


def parquet_games_present(parquet_root: str | Path) -> tuple[str, ...]:
    root = Path(parquet_root)
    if not root.exists():
        return ()
    return tuple(sorted(path.name.split("=", 1)[1] for path in root.glob("game=*") if "=" in path.name))
