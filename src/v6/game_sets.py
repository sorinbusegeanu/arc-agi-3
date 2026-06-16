from __future__ import annotations

import json
from collections import defaultdict
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
    if path is None and fallback_games:
        path = infer_manifest_path_from_games(fallback_games)
    if path is None or not path.exists():
        return inferred_game_set_manifest(game_set_name=game_set_name, fallback_games=fallback_games)
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


def infer_manifest_path_from_games(games: tuple[str, ...]) -> Path | None:
    requested = set(games)
    if not requested:
        return None
    root = Path("runs/v6/game_sets")
    if not root.exists():
        return None

    best_path: Path | None = None
    best_extra: int | None = None
    for path in sorted(root.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        manifest_games = {str(game) for game in raw.get("games", [])}
        if not requested.issubset(manifest_games):
            continue
        extra = len(manifest_games - requested)
        if best_extra is None or extra < best_extra:
            best_path = path
            best_extra = extra
    return best_path


def inferred_game_set_manifest(*, game_set_name: str | None, fallback_games: tuple[str, ...]) -> GameSetManifest:
    families = infer_families_from_games(fallback_games)
    return GameSetManifest(
        name=game_set_name or "adhoc",
        games=tuple(fallback_games),
        families=families,
        purpose="auto-inferred game selection",
        core7_anchors=(),
    )


def infer_families_from_games(games: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for game in sorted(set(games)):
        family = infer_family_name(game)
        groups[family].append(game)
    return {family: tuple(items) for family, items in sorted(groups.items())}


def infer_family_name(game: str) -> str:
    prefixes = {
        "tt": "collection",
        "pb": "push_crate",
        "sk": "push_crate",
        "fs": "switch_unlock",
        "ul": "switch_unlock",
        "tp": "teleport_warp",
        "wa": "teleport_warp",
        "va": "coverage_path",
        "bd": "coverage_path",
        "gr": "movement_modifier",
        "mo": "movement_modifier",
        "nw": "vector_delay",
        "dl": "vector_delay",
        "zq": "hazard_timing",
        "hd": "hazard_timing",
        "sv": "survival_resource",
        "wm": "click_timing",
        "gp": "paint_pattern",
        "lo": "toggle_pattern",
        "rp": "graph_logic",
        "pu": "graph_logic",
        "mm": "memory_hidden",
        "ms": "memory_hidden",
        "dd": "logistics",
        "as": "logistics",
        "tb": "build_craft",
        "wl": "build_craft",
    }
    return prefixes.get(game[:2], game)


def parquet_games_present(parquet_root: str | Path) -> tuple[str, ...]:
    root = Path(parquet_root)
    if not root.exists():
        return ()
    return tuple(sorted(path.name.split("=", 1)[1] for path in root.glob("game=*") if "=" in path.name))
