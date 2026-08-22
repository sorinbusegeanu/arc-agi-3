from __future__ import annotations

"""Target-grounded M1N transfer ranking and failure backoff for v8.56."""

import math

from v8.model import MemoryLevel, MemoryType, RelationType, signed_u64, stable_u64


_INSTALLED = False
_BASE_GROUNDED_TRANSFER = None
_BASE_CROSS_GAME = None
_BASE_OBSERVE_TRANSFER = None
_MAX_FAILURES = 6
_BACKOFF_LIMIT = 4
_TARGET_M1N_FAILURES: dict[tuple[str, int, int], int] = {}
_LINEAGE = {
    int(RelationType.PROVENANCE),
    int(RelationType.EXPLAINS),
    int(RelationType.LEADS_TO),
    int(RelationType.CONTEXT_REFINES),
}


def _key(game_id: str, uid) -> tuple[str, int, int]:
    return str(game_id), int(uid.hi), int(uid.lo)


def _record(game_id: str, uid, *, success: bool) -> None:
    key = _key(game_id, uid)
    if success:
        _TARGET_M1N_FAILURES.pop(key, None)
        return
    _TARGET_M1N_FAILURES[key] = min(
        _MAX_FAILURES,
        max(0, int(_TARGET_M1N_FAILURES.get(key, 0))) + 1,
    )


def _is_grounded(row) -> bool:
    return bool(
        row is not None
        and int(getattr(row, "level", -1)) == int(MemoryLevel.M1)
        and int(getattr(row, "memory_type", -1)) == int(MemoryType.CONTINGENCY)
        and len(getattr(row, "key_parts", ())) >= 4
    )


def _provenance_resolver(view):
    direct = {}
    lineage = {}
    try:
        edges = tuple(view.edge_records())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        edges = ()
    for edge in edges:
        relation = int(edge.relation_type)
        if relation == int(RelationType.GAME_PROVENANCE) and int(edge.target_uid.hi) == 0:
            direct.setdefault(edge.source_uid, set()).add(int(edge.target_uid.lo))
        elif relation in _LINEAGE:
            lineage.setdefault(edge.source_uid, set()).add(edge.target_uid)
    cache = {}

    def graph_games(uid):
        if uid in cache:
            return cache[uid]
        found = set(direct.get(uid, ()))
        frontier, visited = {uid}, {uid}
        for _depth in range(8):
            following = set()
            for current in frontier:
                for parent in lineage.get(current, ()):
                    found.update(direct.get(parent, ()))
                    if parent not in visited:
                        visited.add(parent)
                        following.add(parent)
            if not following:
                break
            frontier = following
        cache[uid] = frozenset(found)
        return cache[uid]

    source_games = getattr(view, "source_games", None)

    def resolve(uid):
        if callable(source_games):
            try:
                games = frozenset(int(value) for value in source_games(uid))
            except (RuntimeError, TypeError, ValueError):
                games = frozenset()
            if games:
                return games
        return graph_games(uid)

    return resolve


def _raw_m1n_index(view, game_id: str):
    from v8 import normalized_memory_v086 as normalized

    refresh = getattr(view, "_refresh_strategy_cache", None)
    if callable(refresh):
        refresh()
    current_game = int(stable_u64(str(game_id), person=b"v8-game"))
    version = tuple(getattr(view, "_strategy_version", ()))
    cache_key = (version, current_game)
    if getattr(view, "_v856_m1n_index_key", None) == cache_key:
        return getattr(view, "_v856_m1n_index", {})

    nodes = dict(getattr(view, "_node_by_uid", {}))
    parents = getattr(view, "_parents", {})
    games_for = _provenance_resolver(view)
    by_action: dict[int, list[tuple[float, object, str]]] = {}
    for row in nodes.values():
        if not normalized.is_normalized_contingency(row):
            continue
        grounded = []
        for parent_uid in parents.get(row.uid, ()):
            parent = nodes.get(parent_uid)
            if not _is_grounded(parent):
                continue
            games = games_for(parent_uid)
            if games:
                grounded.append((parent, games))
        if not grounded:
            continue
        if not any(any(game != current_game for game in games) for _parent, games in grounded):
            continue

        support_bonus = 0.05 * math.log1p(max(1, int(getattr(row, "support_count", 1))))
        score = max(
            0.0,
            float(getattr(row, "significance", 0.0)),
            float(getattr(row, "learning_value", 0.0)),
        ) + support_bonus
        for parent, games in grounded:
            if current_game not in games or len(parent.key_parts) < 2:
                continue
            action = int(signed_u64(int(parent.key_parts[1])))
            by_action.setdefault(action, []).append(
                (float(score), row.uid, "M1N_GROUNDED")
            )

    result = {
        int(action): tuple(sorted(rows, key=lambda item: (-float(item[0]), item[1])))
        for action, rows in by_action.items()
    }
    view._v856_m1n_index_key = cache_key
    view._v856_m1n_index = result
    return result


def _adjust_m1n(game_id: str, rows):
    adjusted = {}
    for action, candidates in rows.items():
        kept = []
        for score, uid, origin in candidates:
            failures = max(0, int(_TARGET_M1N_FAILURES.get(_key(game_id, uid), 0)))
            if failures >= _BACKOFF_LIMIT:
                continue
            kept.append((float(score) - 0.15 * failures, uid, origin))
        if kept:
            kept.sort(key=lambda item: (-float(item[0]), item[1]))
            adjusted[int(action)] = tuple(kept)
    return adjusted


def _grounded_transfer_m1n_v856(view, game_id: str):
    m7, _legacy_m1n = _BASE_GROUNDED_TRANSFER(view, game_id)
    return m7, _adjust_m1n(str(game_id), _raw_m1n_index(view, str(game_id)))


def _cross_game_m1n_v856(sampler, actions):
    selected = _BASE_CROSS_GAME(sampler, actions)
    if selected is None:
        sampler._v856_m1n_selected = None
        return None
    action, origin, uid = selected
    if str(origin) == "M1N_GROUNDED" and uid is not None:
        sampler._v856_m1n_selected = (str(sampler.game_id), int(action), uid)
    else:
        sampler._v856_m1n_selected = None
    return selected


def _observe_transfer_m1n_v856(self, intervention, **kwargs):
    selected = getattr(self, "_v856_m1n_selected", None)
    result = _BASE_OBSERVE_TRANSFER(self, intervention, **kwargs)
    if selected is None:
        return result
    game_id, selected_action, uid = selected
    action = int(kwargs.get("action", getattr(intervention, "action", selected_action)))
    if action != int(selected_action):
        self._v856_m1n_selected = None
        return result

    from v8 import environment_neutrality_v837 as v837

    semantics = v837._transition_semantics(kwargs)
    success = bool(semantics.successful_boundary or semantics.productive)
    _record(str(game_id), uid, success=success)
    self._v856_m1n_selected = None
    return result


def install_adaptive_memory_transfer_m1n_v856() -> None:
    global _INSTALLED, _BASE_GROUNDED_TRANSFER, _BASE_CROSS_GAME
    global _BASE_OBSERVE_TRANSFER
    if _INSTALLED:
        return

    from v8 import adaptive_memory_transfer_grounding_v856 as grounding
    from v8 import sampling_transfer_v833 as transfer

    # Compose beneath the public grounding authority so v8.56 M7 backoff remains
    # the single final grounded hook and M1N enrichment is an implementation detail.
    _BASE_GROUNDED_TRANSFER = grounding._BASE_GROUNDED_TRANSFER
    grounding._BASE_GROUNDED_TRANSFER = _grounded_transfer_m1n_v856

    _BASE_CROSS_GAME = transfer._cross_game_transfer_action
    transfer._cross_game_transfer_action = _cross_game_m1n_v856

    _BASE_OBSERVE_TRANSFER = transfer._observe_transfer_v833
    transfer._observe_transfer_v833 = _observe_transfer_m1n_v856

    _INSTALLED = True
