from __future__ import annotations

from v7.environment.arc_adapter import registered_game_ids

FAILED_REPRESENTATIVES = ("tt01", "pb02", "fs02", "tp02", "gr01")
PASSING_REFERENCES = ("va02", "mo01")
BROAD_GAMES = FAILED_REPRESENTATIVES + PASSING_REFERENCES
FOUNDATION_GAMES = ("ez01", "ez02", "ez03", "ez04", "ul01", "tt01", "pb01", "fs01", "ic01", "va01")
TRANSFORMATION_GAMES = ("pb01", "pb02", "pb03", "sk01", "sk02", "sk03", "ci01", "op01", "rz01", "mb01", "tk01", "ic01", "ic02", "ic03", "fs01", "fs02", "fs03", "tp01", "tp02", "tp03", "ml01", "ml02", "ml03", "tb01", "tb02", "tb03", "cr01", "rn01", "wl01", "dr01", "dg01", "mx01")
CONTEXT_GAMES = ("ul01", "ul02", "ul03", "fs01", "fs02", "fs03", "tp01", "tp02", "tp03", "ic01", "ic02", "ic03", "nw01", "nw02", "nw03", "rs01", "rs02", "rs03", "zq01", "zq02", "zq03", "ex01", "ex02", "ex03", "va01", "va02", "va03", "bd01", "hm01", "gl01", "tr01", "vp01", "cf01")
ROLE_TRANSFER_GAMES = ("ul01", "ul02", "ul03", "fs01", "fs02", "fs03", "co01", "ex01", "ex02", "ex03", "tb01", "tb02", "tb03", "cr01", "rn01", "wl01", "fi01", "fw01", "hd01", "bp01", "dd01", "as01")
FUTURE_ENABLE_GAMES = ("ul01", "ul02", "ul03", "fs01", "fs02", "fs03", "co01", "tb01", "tb02", "tb03", "cr01", "rn01", "wl01", "fi01", "fw01", "mx01", "dr01", "dg01", "ex01", "ex02", "ex03", "bp01")
FUTURE_BLOCK_GAMES = ("pb01", "pb02", "pb03", "sk01", "sk02", "sk03", "ci01", "rz01", "op01", "mb01", "fb01", "va02", "bd01", "hm01", "vp01", "gl01", "tr01", "cf01", "in01", "wk01", "lf01", "rh01", "hz01", "zq01", "zq02", "zq03")
FUTURE_REVERSIBLE_GAMES = ("pb01", "fs01", "tp01", "tp02", "ic01", "ic02", "ic03", "nw01", "nw02", "nw03", "rs01", "rs02", "rs03", "ex01", "ex02", "ex03", "rc01", "bl01", "sw01", "dv01", "dp01", "wr01")
FUTURE_TERMINATE_GAMES = ("tt01", "tt02", "tt03", "wm01", "wm02", "wm03", "sv01", "sv02", "sv03", "st01", "tg01", "hs01", "sc01", "vi01", "fw01", "fi01", "fb01", "av01", "sb01", "rh01", "lf01", "hz01", "zq01", "zq02", "zq03")
BRIDGE_GAMES = ("ul01", "ul02", "ul03", "fs01", "fs02", "fs03", "co01", "tb01", "tb02", "tb03", "cr01", "rn01", "wl01", "fi01", "fw01", "pb01", "pb02", "pb03", "sk01", "sk02", "sk03", "ci01", "ic01", "ic02", "ic03", "tp01", "tp02", "tp03", "ex01", "ex02", "ex03", "bp01", "dd01", "as01")
TRANSFER_VALIDATION_GAMES = ("ul01", "ul02", "ul03", "fs01", "fs02", "fs03", "co01", "pb01", "pb02", "pb03", "sk01", "sk02", "sk03", "ci01", "tb01", "tb02", "tb03", "cr01", "rn01", "wl01", "ex01", "ex02", "ex03", "tp01", "tp02", "tp03", "ml01", "ml02", "ml03", "bp01", "dd01", "as01", "fi01", "fw01")
FALSIFICATION_GAMES = ("ul01", "ul02", "ul03", "fs01", "fs02", "fs03", "co01", "pb01", "pb03", "sk02", "sk03", "ci01", "tp02", "tp03", "tb02", "tb03", "cr01", "rn01", "ex02", "ex03", "ml03", "nw03", "rs03", "zq03", "dr01", "dg01", "mx01", "fb01", "vi01")
DIVERSE_GAMES = ("ez01", "ul01", "pb01", "fs01", "tp01", "ic01", "tb01", "ex01", "bp01", "fw01")

V7_GAME_PRESETS = {
    "failed_representatives": FAILED_REPRESENTATIVES,
    "passing_references": PASSING_REFERENCES,
    "broad": BROAD_GAMES,
    "foundation": FOUNDATION_GAMES,
    "transformation": TRANSFORMATION_GAMES,
    "context": CONTEXT_GAMES,
    "role_transfer": ROLE_TRANSFER_GAMES,
    "future_enable": FUTURE_ENABLE_GAMES,
    "future_block": FUTURE_BLOCK_GAMES,
    "future_reversible": FUTURE_REVERSIBLE_GAMES,
    "future_terminate": FUTURE_TERMINATE_GAMES,
    "bridge": BRIDGE_GAMES,
    "transfer_validation": TRANSFER_VALIDATION_GAMES,
    "falsification": FALSIFICATION_GAMES,
    "diverse": DIVERSE_GAMES,
}


def resolve_game_selector(selector: str, env_root: str | None = None) -> tuple[str, ...]:
    value = str(selector).strip()
    available = tuple(sorted(registered_game_ids(env_root)))
    if value == "all":
        if not available:
            raise ValueError("no registered game ids were found in the project environment registry")
        return tuple(game_id for game_id in available if game_id != "gc01")
    games = tuple(dict.fromkeys(V7_GAME_PRESETS[value])) if value in V7_GAME_PRESETS else tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not games:
        raise ValueError("games selection is empty; pass a v6-compatible preset, 'all', or game ids")
    if available:
        invalid = tuple(game_id for game_id in games if game_id not in available)
        if invalid:
            raise ValueError(f"game selector '{value}' contains invalid game ids: {', '.join(invalid)}. Valid installed IDs: {', '.join(available)}")
    return games
