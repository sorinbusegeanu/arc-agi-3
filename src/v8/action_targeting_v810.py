from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
from hashlib import blake2b
from typing import Iterable

from v8.model import MemoryLevel, signed_u64, stable_u64
from v8.publication import ActionScore, PlannedAction


_INSTALLED = False
_CLICK_ACTION_ID = 6
_STRUCTURAL_CLICK_MARKER = 1 << 30
_STRUCTURAL_DESCRIPTOR_MASK = (1 << 22) - 1
_MAX_STRUCTURAL_CLICK_TARGETS = 96
_MAX_LEGACY_EXACT_CLICK_TARGETS = 16


@dataclass(frozen=True, slots=True)
class ClickTarget:
    token: int
    x: int
    y: int
    kind: str
    descriptor: tuple[int, ...]
    priority: int


def is_structural_click_token(token: int) -> bool:
    value = int(token)
    return bool(
        value >= 0
        and (value & _STRUCTURAL_CLICK_MARKER)
        and (value & 0xFF) == _CLICK_ACTION_ID
    )


def _legacy_coordinate_payload(token: int):
    if is_structural_click_token(token):
        return None
    from v8.learning_blockers_v055 import unpack_action_choice

    try:
        action, data = unpack_action_choice(int(token))
    except (TypeError, ValueError):
        return None
    if int(action) != _CLICK_ACTION_ID or data is None:
        return None
    return data


def native_action_id(token: int) -> int:
    value = int(token)
    if is_structural_click_token(value) or _legacy_coordinate_payload(value) is not None:
        return _CLICK_ACTION_ID
    return value


def _shape_signature(cells: tuple[tuple[int, int], ...]) -> int:
    min_x = min(x for x, _y in cells)
    min_y = min(y for _x, y in cells)
    normalized = tuple(sorted((x - min_x, y - min_y) for x, y in cells))
    max_x = max(x for x, _y in normalized)
    max_y = max(y for _x, y in normalized)
    digest = blake2b(digest_size=8, person=b"v810-shape")
    digest.update(int(max_x + 1).to_bytes(2, "little"))
    digest.update(int(max_y + 1).to_bytes(2, "little"))
    digest.update(int(len(normalized)).to_bytes(2, "little"))
    for x, y in normalized:
        digest.update(int(x).to_bytes(2, "little"))
        digest.update(int(y).to_bytes(2, "little"))
    return int.from_bytes(digest.digest(), "little")


def _descriptor_token(descriptor: tuple[int, ...], used: dict[int, tuple[int, ...]]) -> int:
    salt = 0
    while True:
        code = int(
            stable_u64(*descriptor, salt, person=b"v810-target")
            & _STRUCTURAL_DESCRIPTOR_MASK
        )
        if code == 0:
            code = 1
        token = int(_STRUCTURAL_CLICK_MARKER | (code << 8) | _CLICK_ACTION_ID)
        prior = used.get(token)
        if prior is None or prior == descriptor:
            used[token] = descriptor
            return token
        salt += 1


def _background_value(array) -> int:
    counts = Counter(int(value) for value in array.ravel())
    if not counts:
        return 0
    maximum = max(counts.values())
    tied = sorted(value for value, count in counts.items() if count == maximum)
    return 0 if 0 in tied else int(tied[0])


def _components(array, background: int) -> list[dict[str, object]]:
    height, width = array.shape
    visited: set[tuple[int, int]] = set()
    result: list[dict[str, object]] = []
    for y in range(height):
        for x in range(width):
            if (x, y) in visited or int(array[y, x]) == int(background):
                continue
            color = int(array[y, x])
            queue = deque([(x, y)])
            visited.add((x, y))
            cells: list[tuple[int, int]] = []
            while queue:
                cx, cy = queue.popleft()
                cells.append((cx, cy))
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    if (nx, ny) in visited or int(array[ny, nx]) != color:
                        continue
                    visited.add((nx, ny))
                    queue.append((nx, ny))
            ordered = tuple(sorted(cells, key=lambda point: (point[1], point[0])))
            result.append(
                {
                    "cells": ordered,
                    "shape": _shape_signature(ordered),
                    "min_x": min(x for x, _y in ordered),
                    "min_y": min(y for _x, y in ordered),
                }
            )
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for component in result:
        grouped[int(component["shape"])].append(component)
    for siblings in grouped.values():
        siblings.sort(key=lambda item: (int(item["min_y"]), int(item["min_x"])))
        for rank, component in enumerate(siblings):
            component["rank"] = int(rank)
    return result


def structural_click_targets(
    grid,
    *,
    last_changed: Iterable[tuple[int, int]] = (),
    limit: int = _MAX_STRUCTURAL_CLICK_TARGETS,
) -> tuple[ClickTarget, ...]:
    """Derive bounded, color-invariant click targets from observable structure.

    Tokens describe a target's role inside a connected component rather than an
    absolute coordinate.  The token is therefore reusable when the corresponding
    object moves or changes color; execution resolves the descriptor back to the
    current grid coordinate.
    """
    import numpy as np

    array = np.asarray(grid, dtype=np.int64)
    if array.ndim != 2 or array.size == 0:
        return ()
    height, width = array.shape
    if height > 64 or width > 64:
        raise ValueError("coordinate actions support grids up to 64x64")

    changed = {(int(x), int(y)) for x, y in last_changed}
    background = _background_value(array)
    components = _components(array, background)
    used: dict[int, tuple[int, ...]] = {}
    rows: list[ClickTarget] = []

    def add(
        x: int,
        y: int,
        *,
        kind: str,
        kind_code: int,
        shape: int,
        rank: int,
        role_a: int = 0,
        role_b: int = 0,
        priority: int,
    ) -> None:
        x_i, y_i = int(x), int(y)
        if not (0 <= x_i < width and 0 <= y_i < height):
            return
        descriptor = (
            int(kind_code),
            int(shape),
            int(rank),
            int(role_a),
            int(role_b),
        )
        token = _descriptor_token(descriptor, used)
        if any(row.token == token for row in rows):
            return
        rows.append(ClickTarget(token, x_i, y_i, kind, descriptor, int(priority)))

    for component in components:
        cells = tuple(component["cells"])
        shape = int(component["shape"])
        rank = int(component.get("rank", 0))
        min_x = min(x for x, _y in cells)
        min_y = min(y for _x, y in cells)
        mean_x = sum(x for x, _y in cells) / len(cells)
        mean_y = sum(y for _x, y in cells) / len(cells)
        center = min(
            cells,
            key=lambda point: (
                (point[0] - mean_x) ** 2 + (point[1] - mean_y) ** 2,
                point[1],
                point[0],
            ),
        )
        changed_cells = tuple(point for point in cells if point in changed)
        if changed_cells:
            target = min(
                changed_cells,
                key=lambda point: (
                    (point[0] - mean_x) ** 2 + (point[1] - mean_y) ** 2,
                    point[1],
                    point[0],
                ),
            )
            add(
                *target,
                kind="changed_component",
                kind_code=7,
                shape=shape,
                rank=rank,
                role_a=target[0] - min_x,
                role_b=target[1] - min_y,
                priority=0,
            )

        add(*center, kind="component_center", kind_code=1, shape=shape, rank=rank, priority=10)
        extrema = (
            (2, "component_top", min(cells, key=lambda point: (point[1], point[0]))),
            (3, "component_bottom", max(cells, key=lambda point: (point[1], -point[0]))),
            (4, "component_left", min(cells, key=lambda point: (point[0], point[1]))),
            (5, "component_right", max(cells, key=lambda point: (point[0], -point[1]))),
        )
        for code, kind, point in extrema:
            add(*point, kind=kind, kind_code=code, shape=shape, rank=rank, priority=20)

        if len(cells) <= 8:
            for point in cells:
                add(
                    *point,
                    kind="component_member",
                    kind_code=6,
                    shape=shape,
                    rank=rank,
                    role_a=point[0] - min_x,
                    role_b=point[1] - min_y,
                    priority=30,
                )

    anchors = (
        (0, 0),
        (max(0, width - 1), 0),
        (0, max(0, height - 1)),
        (max(0, width - 1), max(0, height - 1)),
        ((width - 1) // 2, (height - 1) // 2),
    )
    for index, point in enumerate(anchors):
        add(
            *point,
            kind="grid_anchor",
            kind_code=8,
            shape=0,
            rank=index,
            priority=50,
        )

    rows.sort(key=lambda row: (row.priority, row.token, row.y, row.x))
    return tuple(rows[: max(1, int(limit))])


def _native_environment_actions(env) -> tuple[int, ...]:
    raw = getattr(env, "_last_raw", None)
    actions = getattr(raw, "available_actions", None)
    if actions:
        return tuple(sorted(set(int(action) for action in actions)))
    method = getattr(getattr(env, "env", None), "available_actions", None)
    if method is None:
        return ()
    return tuple(sorted(set(int(action) for action in method())))


def _install_structural_click_environment() -> None:
    from v7.environment import arc_adapter as adapter
    from v8.learning_blockers_v055 import pack_action_choice

    base_step = adapter.ArcGridEnvironment.step
    base_reset = adapter.ArcGridEnvironment.reset

    def available_actions(self):
        native = _native_environment_actions(self)
        rows = [value for value in native if int(value) != _CLICK_ACTION_ID]
        if _CLICK_ACTION_ID in native:
            targets = structural_click_targets(
                self._last_grid,
                last_changed=getattr(self, "_v810_last_changed", ()),
            )
            self._v810_click_targets = {target.token: target for target in targets}
            rows.extend(target.token for target in targets)
        return rows

    def step(self, action):
        import numpy as np

        token = int(action)
        before = np.asarray(self._last_grid).copy()
        if is_structural_click_token(token):
            target = getattr(self, "_v810_click_targets", {}).get(token)
            if target is None:
                targets = structural_click_targets(
                    before,
                    last_changed=getattr(self, "_v810_last_changed", ()),
                )
                target = next((row for row in targets if row.token == token), None)
            if target is None:
                raise ValueError("structural click target is not resolvable in the current grid")
            executed = pack_action_choice(_CLICK_ACTION_ID, target.x, target.y)
            result = base_step(self, executed)
        elif token == _CLICK_ACTION_ID:
            targets = structural_click_targets(
                before,
                last_changed=getattr(self, "_v810_last_changed", ()),
            )
            if not targets:
                raise ValueError("click action has no observable target")
            target = targets[0]
            executed = pack_action_choice(_CLICK_ACTION_ID, target.x, target.y)
            result = base_step(self, executed)
        else:
            result = base_step(self, token)

        after = np.asarray(self._last_grid)
        if bool(getattr(self, "last_step_was_reset_boundary", False)) or before.shape != after.shape:
            self._v810_last_changed = ()
        else:
            ys, xs = np.nonzero(before != after)
            self._v810_last_changed = tuple((int(x), int(y)) for y, x in zip(ys, xs, strict=True))
        return result

    def reset(self, *args, **kwargs):
        result = base_reset(self, *args, **kwargs)
        self._v810_last_changed = ()
        self._v810_click_targets = {}
        return result

    adapter.ArcGridEnvironment.available_actions = available_actions
    adapter.ArcGridEnvironment.step = step
    adapter.ArcGridEnvironment.reset = reset


def _legacy_exact_click_actions(view, context_signature: int) -> tuple[int, ...]:
    view._refresh_strategy_cache()
    rows = []
    for row in getattr(view, "_node_by_uid", {}).values():
        if int(row.level) != int(MemoryLevel.M1) or len(row.key_parts) < 2:
            continue
        if int(row.key_parts[0]) != int(context_signature):
            continue
        action = signed_u64(int(row.key_parts[1]))
        if _legacy_coordinate_payload(action) is None:
            continue
        valence = float(getattr(row, "expected_primary_valence", 0.0)) * float(
            getattr(row, "primary_valence_confidence", 0.0)
        )
        rows.append((action, valence, int(row.support_count)))
    rows.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return tuple(item[0] for item in rows[:_MAX_LEGACY_EXACT_CLICK_TARGETS])


def _one_plan_per_native_type(plans: Iterable[PlannedAction]) -> tuple[PlannedAction, ...]:
    selected: dict[int, PlannedAction] = {}
    for plan in sorted(plans, key=lambda row: (-row.score, row.action_id, row.strategy_uid)):
        selected.setdefault(native_action_id(plan.action_id), plan)
    return tuple(sorted(selected.values(), key=lambda row: (-row.score, row.action_id, row.strategy_uid)))


def _one_score_per_native_type(rows: Iterable[ActionScore], *, rng=None) -> tuple[ActionScore, ...]:
    groups: dict[int, list[ActionScore]] = defaultdict(list)
    for row in rows:
        groups[native_action_id(row.action_id)].append(row)
    selected: list[ActionScore] = []
    for action_type, members in groups.items():
        seen = [row for row in members if int(row.support_count) > 0]
        if seen:
            choice = min(
                seen,
                key=lambda row: (-float(row.score), -int(row.support_count), int(row.action_id)),
            )
        elif action_type == _CLICK_ACTION_ID and rng is not None and members:
            choice = members[rng.randrange(len(members))]
        else:
            choice = min(members, key=lambda row: int(row.action_id))
        selected.append(choice)
    selected.sort(key=lambda row: int(native_action_id(row.action_id)))
    return tuple(selected)


def prefer_persisted_scores(
    rows: Iterable[ActionScore],
    *,
    force_random: bool = False,
) -> tuple[ActionScore, ...]:
    """Prevent mandatory unseen-action exploration from hiding restored memory.

    Once a context has a positively scored remembered action, unseen alternatives
    become neutral epsilon candidates instead of mandatory choices. Explicit actor
    epsilon exploration still returns truly unseen rows because force_random=True.
    """
    values = tuple(rows)
    if force_random:
        return values
    remembered = [row for row in values if int(row.support_count) > 0]
    if not remembered or max(float(row.score) for row in remembered) <= 0.0:
        return values
    return tuple(
        row
        if int(row.support_count) > 0
        else replace(row, support_count=1, score=0.0)
        for row in values
    )


def _install_hierarchical_memory_selection() -> None:
    from v8.publication import LiveReadView

    base_plan = LiveReadView.plan_candidates
    base_score = LiveReadView.score_actions

    def plan_candidates(self, context_signature, action_ids, **kwargs):
        logical = tuple(sorted(set(int(value) for value in action_ids)))
        legacy = _legacy_exact_click_actions(self, int(context_signature))
        expanded = tuple(sorted(set((*logical, *legacy))))
        plans = base_plan(self, context_signature, expanded, **kwargs)
        return _one_plan_per_native_type(plans)

    def score_actions(self, context_signature, action_ids):
        logical = tuple(sorted(set(int(value) for value in action_ids)))
        forced_random = bool(getattr(self, "_behavior_force_random", False))
        legacy = () if forced_random else _legacy_exact_click_actions(self, int(context_signature))
        expanded = tuple(sorted(set((*logical, *legacy))))
        rows = base_score(self, context_signature, expanded)
        rng = getattr(self, "_behavior_rng", None) if forced_random else None
        grouped = _one_score_per_native_type(rows, rng=rng)
        return prefer_persisted_scores(grouped, force_random=forced_random)

    LiveReadView.plan_candidates = plan_candidates
    LiveReadView.score_actions = score_actions


def _install_compact_progress_format() -> None:
    from v8 import diagnostics as diagnostics_module

    def format_game_rate_line(rows) -> str:
        rows = tuple(rows)
        win_rate, level_rate, solved_games, games = diagnostics_module.game_summary(rows)
        grouped = diagnostics_module._group_games(rows)
        details = []
        for game_id, lane_rows in sorted(grouped.items()):
            solved_rows = [row for row in lane_rows if int(getattr(row, "wins", 0)) > 0]
            if not solved_rows:
                continue
            best_values = [int(getattr(row, "best_win_steps", 0) or 0) for row in solved_rows]
            first_values = [int(getattr(row, "first_win_step", 0) or 0) for row in solved_rows]
            best = min((value for value in best_values if value > 0), default=0)
            if best <= 0:
                best = min(
                    (value for value in first_values if value > 0),
                    default=min((int(getattr(row, "steps", 0) or 0) for row in solved_rows), default=0),
                )
            latest = max(solved_rows, key=lambda row: int(getattr(row, "steps", 0) or 0))
            last = int(getattr(latest, "last_win_steps", 0) or 0) or best
            details.append(f"{game_id}:B={best},L={last}")
        suffix = "" if not details else " (" + "; ".join(details) + ")"
        return (
            f"current_run_wins={win_rate:.1f}% current_run_levels_solved={level_rate:.1f}% "
            f"current_run_solved_games={solved_games}/{games}{suffix}"
        )

    diagnostics_module.format_game_rate_line = format_game_rate_line


def install_action_targeting_v810() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_structural_click_environment()
    _install_hierarchical_memory_selection()
    _install_compact_progress_format()
    _INSTALLED = True
