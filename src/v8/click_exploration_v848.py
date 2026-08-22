from __future__ import annotations

"""v8.48 complete, target-specific click exploration.

This layer repairs three click-specific learning blockers without changing the
native ARC action contract:

* structural click targets are augmented by a bounded exact-coordinate coverage
  page, so every grid cell is eventually executable without publishing the whole
  grid as one action branch;
* distinct ACTION6 targets remain distinct through planning/scoring and one
  remembered click no longer suppresses unseen click targets;
* visually unchanged click probes are evidence against that target, not automatic
  path-specific hidden-state frontier nodes. Delayed click mechanics remain
  discoverable through the existing random rollout path and successful trajectories.

Unsolved click games receive a data-derived sampling multiplier based on their
observed branching factor relative to non-click games. The multiplier is tempered
by a square root so action-space complexity matters without linearly starving the
rest of the run.
"""

import math
import statistics
from collections import defaultdict
from typing import Iterable


_INSTALLED = False
_BASE_ENV_INIT = None
_BASE_ENV_AVAILABLE = None
_BASE_ENV_STEP = None
_BASE_ENV_RESET = None
_BASE_RECORD_EXPANSION = None
_BASE_BEST_EXPANSION = None
_BASE_SELECT_EXPANDABLE_ACTION = None
_BASE_REGISTER_GAMES = None
_BASE_SAMPLING_WEIGHT = None


def _targeting():
    from v8 import action_targeting_v810 as targeting

    return targeting


def _is_click_token(action: int) -> bool:
    return int(_targeting().native_action_id(int(action))) == 6


def _is_exact_click_token(action: int) -> bool:
    return _targeting()._legacy_coordinate_payload(int(action)) is not None


def _coverage_page_size() -> int:
    from v8 import learning_blockers_v055 as blockers

    return max(1, int(getattr(blockers, "_COORDINATE_PAGE_SIZE", 64)))


def _coverage_page_count(grid) -> int:
    import numpy as np

    array = np.asarray(grid)
    if array.ndim != 2 or array.size <= 0:
        return 0
    total = int(array.shape[0]) * int(array.shape[1])
    return max(1, (total + _coverage_page_size() - 1) // _coverage_page_size())


def exact_click_coverage_page(grid, page: int) -> tuple[int, ...]:
    """Return one bounded exact-coordinate page covering every cell over all pages."""
    import numpy as np

    from v8.learning_blockers_v055 import pack_action_choice

    array = np.asarray(grid)
    if array.ndim != 2 or array.size <= 0:
        return ()
    height, width = (int(array.shape[0]), int(array.shape[1]))
    if height > 64 or width > 64:
        raise ValueError("coordinate actions support grids up to 64x64")
    total = height * width
    size = _coverage_page_size()
    pages = max(1, (total + size - 1) // size)
    page_index = int(page) % pages
    start = page_index * size
    stop = min(total, start + size)
    return tuple(
        pack_action_choice(6, index % width, index // width)
        for index in range(start, stop)
    )


def _valid_exact_click(token: int, grid) -> bool:
    import numpy as np

    payload = _targeting()._legacy_coordinate_payload(int(token))
    if payload is None:
        return False
    array = np.asarray(grid)
    if array.ndim != 2 or array.size <= 0:
        return False
    return bool(
        0 <= int(payload["x"]) < int(array.shape[1])
        and 0 <= int(payload["y"]) < int(array.shape[0])
    )


def _ensure_env_click_state(env) -> None:
    if not hasattr(env, "_v848_click_seed"):
        env._v848_click_seed = 0
    if not hasattr(env, "_v848_click_page"):
        env._v848_click_page = None
    if not hasattr(env, "_v848_click_page_tried"):
        env._v848_click_page_tried = set()
    if not hasattr(env, "_v848_replayable_clicks"):
        env._v848_replayable_clicks = set()


def _env_init_v848(self, *args, **kwargs) -> None:
    seed = int(kwargs.get("seed", 0) or 0)
    _BASE_ENV_INIT(self, *args, **kwargs)
    self._v848_click_seed = seed
    self._v848_click_page = None
    self._v848_click_page_tried = set()
    self._v848_replayable_clicks = set()


def _env_available_v848(self):
    _ensure_env_click_state(self)
    base = tuple(sorted({int(value) for value in _BASE_ENV_AVAILABLE(self)}))
    if not any(_is_click_token(value) for value in base):
        return list(base)

    pages = _coverage_page_count(self._last_grid)
    if pages <= 0:
        return list(base)
    if self._v848_click_page is None:
        self._v848_click_page = int(self._v848_click_seed) % int(pages)
    else:
        self._v848_click_page = int(self._v848_click_page) % int(pages)

    coverage = exact_click_coverage_page(self._last_grid, int(self._v848_click_page))
    replayable = tuple(
        int(value)
        for value in getattr(self, "_v848_replayable_clicks", ())
        if _valid_exact_click(int(value), self._last_grid)
    )
    return list(sorted(set((*base, *coverage, *replayable))))


def _env_step_v848(self, action):
    import numpy as np

    _ensure_env_click_state(self)
    token = int(action)
    exact = _is_exact_click_token(token)
    before = np.asarray(self._last_grid).copy()
    before_levels = int(getattr(self, "last_levels_completed", 0))
    pages = _coverage_page_count(before)
    if pages > 0 and self._v848_click_page is None:
        self._v848_click_page = int(self._v848_click_seed) % int(pages)
    page_tokens = set(
        exact_click_coverage_page(before, int(self._v848_click_page or 0))
        if pages > 0
        else ()
    )

    result = _BASE_ENV_STEP(self, token)

    after = np.asarray(self._last_grid)
    if exact and token in page_tokens:
        self._v848_click_page_tried.add(token)

    level_progress = bool(
        int(getattr(self, "last_levels_completed", 0)) > before_levels
        or bool(getattr(self, "level_completed_event", False))
        or str(getattr(self, "last_outcome_state", "")) == "WIN"
    )
    structural_change = bool(
        before.shape != after.shape
        or (before.shape == after.shape and bool(np.any(before != after)))
    )
    if exact and (level_progress or structural_change):
        self._v848_replayable_clicks.add(token)

    # Advance only after every exact coordinate in the exposed page has actually
    # been tried. This prevents an ever-moving target set from leaving untested
    # actions behind in the evidence frontier.
    if page_tokens and page_tokens.issubset(self._v848_click_page_tried):
        self._v848_click_page = (int(self._v848_click_page or 0) + 1) % max(1, pages)
        self._v848_click_page_tried.clear()
    return result


def _env_reset_v848(self, *args, **kwargs):
    _ensure_env_click_state(self)
    # Coverage state deliberately survives explicit/replay resets within the same
    # lease. A reset is not evidence that already-tested coordinates became new.
    return _BASE_ENV_RESET(self, *args, **kwargs)


def _selection_group(action: int) -> tuple[int, int]:
    native = int(_targeting().native_action_id(int(action)))
    if native == 6:
        return native, int(action)
    return native, 0


def _one_plan_per_target_v848(plans: Iterable[object]) -> tuple[object, ...]:
    selected: dict[tuple[int, int], object] = {}
    for plan in sorted(
        plans,
        key=lambda row: (-float(row.score), int(row.action_id), row.strategy_uid),
    ):
        selected.setdefault(_selection_group(int(plan.action_id)), plan)
    return tuple(
        sorted(
            selected.values(),
            key=lambda row: (-float(row.score), int(row.action_id), row.strategy_uid),
        )
    )


def _one_score_per_target_v848(rows: Iterable[object], *, rng=None) -> tuple[object, ...]:
    groups: dict[tuple[int, int], list[object]] = defaultdict(list)
    for row in rows:
        groups[_selection_group(int(row.action_id))].append(row)

    selected = []
    for (_native, _target), members in groups.items():
        seen = [row for row in members if int(row.support_count) > 0]
        if seen:
            choice = min(
                seen,
                key=lambda row: (
                    -float(row.score),
                    -int(row.support_count),
                    int(row.action_id),
                ),
            )
        elif rng is not None and members:
            choice = members[rng.randrange(len(members))]
        else:
            choice = min(members, key=lambda row: int(row.action_id))
        selected.append(choice)
    selected.sort(key=lambda row: (int(_targeting().native_action_id(row.action_id)), int(row.action_id)))
    return tuple(selected)


def _prefer_persisted_scores_v848(rows: Iterable[object], *, force_random: bool = False):
    """Remembered movement may suppress unseen movement; one click never hides another."""
    from dataclasses import replace

    values = tuple(rows)
    if force_random:
        return values
    remembered = [row for row in values if int(row.support_count) > 0]
    if not remembered or max(float(row.score) for row in remembered) <= 0.0:
        return values
    result = []
    for row in values:
        if int(row.support_count) > 0 or _is_click_token(int(row.action_id)):
            result.append(row)
        else:
            result.append(replace(row, support_count=1, score=0.0))
    return tuple(result)


def _suppressed_click_latent(node) -> bool:
    if not bool(getattr(node, "latent", False)):
        return False
    anchor = tuple(int(value) for value in getattr(node, "anchor", ()))
    return bool(anchor and _is_click_token(anchor[-1]))


def _select_expandable_action_v848(actions: tuple[int, ...]) -> int:
    """Give a click-capable frontier a click before inert movement prefixes."""
    values = tuple(int(value) for value in actions)
    clicks = tuple(value for value in values if _is_click_token(value))
    if clicks:
        return min(clicks)
    return int(_BASE_SELECT_EXPANDABLE_ACTION(values))


def _best_expansion_v848(sampler):
    from v8 import sampling_evidence_frontier_v847 as frontier

    nodes = frontier._ensure_state_v847(sampler)
    candidates = [
        node
        for node in nodes.values()
        if frontier._expandable_actions(node) and not _suppressed_click_latent(node)
    ]
    if not candidates:
        return None
    node = max(candidates, key=frontier._priority_key_v847)
    action = _select_expandable_action_v848(frontier._expandable_actions(node))
    return node, action


def _record_expansion_v848(sampler, **kwargs):
    from v8 import sampling_evidence_frontier_v847 as frontier

    destination = _BASE_RECORD_EXPANSION(sampler, **kwargs)
    action = int(kwargs.get("action", 0))
    before_level = int(kwargs.get("before_level", 0))
    after_level = int(kwargs.get("after_level", before_level))
    before_context = int(kwargs.get("before_context", 0))
    after_context = int(kwargs.get("after_context", before_context))
    changed_cells = int(kwargs.get("changed_cells", 0))
    future_delta = float(kwargs.get("future_delta", 0.0))
    terminal_state = str(kwargs.get("terminal_state", ""))
    click_noop = bool(
        _is_click_token(action)
        and terminal_state != "GAME_OVER"
        and after_level == before_level
        and after_context == before_context
        and changed_cells <= 0
        and future_delta == 0.0
    )
    if not click_noop:
        return destination

    nodes = frontier._ensure_state_v847(sampler)
    source = nodes.get(str(kwargs.get("source_node_id", "")))
    if source is not None:
        source.failures += 1
        sampler._v847_dirty = True
    if destination is not None and bool(getattr(destination, "latent", False)):
        nodes.pop(str(destination.node_id), None)
        sampler._v847_dirty = True
    # A wrong click is a tested target at the same actionable state. Do not turn
    # it into an automatically high-priority hidden-state prefix. Random rollouts
    # remain free to discover genuinely delayed click mechanics.
    return None


def _probe_game_action_space(game_id: str) -> tuple[bool, int] | None:
    try:
        from v7.environment.arc_adapter import ArcGridEnvironment

        env = ArcGridEnvironment(game_id=str(game_id))
        try:
            actions = tuple(sorted({int(value) for value in env.available_actions()}))
            return bool(any(_is_click_token(value) for value in actions)), len(actions)
        finally:
            close = getattr(getattr(env, "env", None), "close", None)
            if callable(close):
                close()
    except Exception:
        return None


def _register_games_v848(self, games) -> None:
    values = tuple(dict.fromkeys(str(game) for game in games))
    _BASE_REGISTER_GAMES(self, values)
    with self._lock:
        known = getattr(self, "_v848_action_spaces", None)
        if known is None:
            known = {}
            self._v848_action_spaces = known
        missing = tuple(game for game in values if game not in known)
    for game in missing:
        measured = _probe_game_action_space(game)
        with self._lock:
            known = self._v848_action_spaces
            known.setdefault(game, measured)


def _click_complexity_multiplier(self, game_id: str) -> float:
    with self._lock:
        spaces = getattr(self, "_v848_action_spaces", {})
        measured = spaces.get(str(game_id))
        if not measured or not bool(measured[0]):
            return 1.0
        click_branching = max(1, int(measured[1]))
        references = [
            max(1, int(value[1]))
            for value in spaces.values()
            if value is not None and not bool(value[0]) and int(value[1]) > 0
        ]
    if not references:
        return 1.0
    reference = max(1.0, float(statistics.median(references)))
    return math.sqrt(max(1.0, float(click_branching) / reference))


def _sampling_weight_v848(self, game_id: str) -> float:
    from v8 import adaptive_learning_allocation_v819 as allocation

    base = float(_BASE_SAMPLING_WEIGHT(self, str(game_id)))
    if self.game_state(str(game_id)) != allocation.GameLearningState.UNSOLVED:
        return base
    return base * float(_click_complexity_multiplier(self, str(game_id)))


def install_click_exploration_v848() -> None:
    global _INSTALLED
    global _BASE_ENV_INIT, _BASE_ENV_AVAILABLE, _BASE_ENV_STEP, _BASE_ENV_RESET
    global _BASE_RECORD_EXPANSION, _BASE_BEST_EXPANSION, _BASE_SELECT_EXPANDABLE_ACTION
    global _BASE_REGISTER_GAMES, _BASE_SAMPLING_WEIGHT
    if _INSTALLED:
        return

    from v7.environment import arc_adapter as adapter
    from v8 import action_targeting_v810 as targeting
    from v8 import adaptive_learning_allocation_v819 as allocation
    from v8 import sampling_evidence_frontier_v847 as frontier

    _BASE_ENV_INIT = adapter.ArcGridEnvironment.__init__
    _BASE_ENV_AVAILABLE = adapter.ArcGridEnvironment.available_actions
    _BASE_ENV_STEP = adapter.ArcGridEnvironment.step
    _BASE_ENV_RESET = adapter.ArcGridEnvironment.reset
    adapter.ArcGridEnvironment.__init__ = _env_init_v848
    adapter.ArcGridEnvironment.available_actions = _env_available_v848
    adapter.ArcGridEnvironment.step = _env_step_v848
    adapter.ArcGridEnvironment.reset = _env_reset_v848

    # The installed v8.10 wrappers resolve these helpers at call time, so replacing
    # the helpers preserves the public LiveReadView method identities.
    targeting._one_plan_per_native_type = _one_plan_per_target_v848
    targeting._one_score_per_native_type = _one_score_per_target_v848
    targeting.prefer_persisted_scores = _prefer_persisted_scores_v848

    _BASE_RECORD_EXPANSION = frontier._record_expansion_v847
    _BASE_BEST_EXPANSION = frontier._best_expansion_v847
    _BASE_SELECT_EXPANDABLE_ACTION = frontier._select_expandable_action_v847
    frontier._record_expansion_v847 = _record_expansion_v848
    frontier._best_expansion_v847 = _best_expansion_v848
    frontier._select_expandable_action_v847 = _select_expandable_action_v848

    _BASE_REGISTER_GAMES = allocation.AdaptiveLearningCoordinator.register_games
    _BASE_SAMPLING_WEIGHT = allocation.AdaptiveLearningCoordinator.sampling_weight
    allocation.AdaptiveLearningCoordinator.register_games = _register_games_v848
    allocation.AdaptiveLearningCoordinator.sampling_weight = _sampling_weight_v848
    _INSTALLED = True
