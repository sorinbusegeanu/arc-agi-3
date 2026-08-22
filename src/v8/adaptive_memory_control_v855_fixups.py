from __future__ import annotations

"""Post-review correctness fixes for v8.55 adaptive M7 arbitration."""

import os

from v8.model import stable_u64


_INSTALLED = False
_BASE_V829_PLAN = None
_BASE_RESET = None
_MAX_STARVATION_KEYS = 4096
_CONSECUTIVE: dict[tuple[str, int, int, int, int], int] = {}


def _same_world_v855_fixup(view, strategy_uid, game_hash: int) -> bool:
    """Fail closed when strategy provenance is absent or unreadable."""
    try:
        games = frozenset(int(value) for value in view.source_games(strategy_uid))
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return False
    return bool(games) and int(game_hash) in games


def _starvation_key(game: str, level: int, context: int, strategy_uid) -> tuple[str, int, int, int, int]:
    return (
        str(game),
        int(level),
        int(context),
        int(strategy_uid.hi),
        int(strategy_uid.lo),
    )


def _consecutive(key: tuple[str, int, int, int, int]) -> int:
    return max(0, int(_CONSECUTIVE.get(key, 0)))


def _set_consecutive(key: tuple[str, int, int, int, int], value: int) -> int:
    value = max(0, int(value))
    if value <= 0:
        _CONSECUTIVE.pop(key, None)
        return 0
    if key not in _CONSECUTIVE and len(_CONSECUTIVE) >= _MAX_STARVATION_KEYS:
        _CONSECUTIVE.pop(next(iter(_CONSECUTIVE)), None)
    _CONSECUTIVE[key] = value
    return value


def _plan_chain_v855_fixup(self, context_signature, action_ids, **kwargs):
    """Arbitrate exact same-world M7 with context-local starvation state."""
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import adaptive_memory_control_v855 as v855
    from v8 import sampling_portfolio_v831 as portfolio
    from v8 import sampling_progress_control_v829 as v829

    mode = str(
        os.environ.get(
            v819._SAMPLING_MODE_ENV,
            v819.SamplingMode.DISCOVERY.value,
        )
    )
    if mode != v819.SamplingMode.DISCOVERY.value:
        return v855._BASE_PLAN_CHAIN(self, context_signature, action_ids, **kwargs)

    game = str(getattr(v829._CONTROL_STATE, "game_id", ""))
    if not game:
        return v855._BASE_PLAN_CHAIN(self, context_signature, action_ids, **kwargs)

    available = tuple(sorted({int(value) for value in action_ids}))
    if not available:
        return ()

    state = v829._state_key(context_signature)
    progress_action = None if state is None else v829._PROGRESS_ACTION.get(state)
    if progress_action in available:
        v829._set_selection(context_signature, "PROGRESS_REPLAY", (progress_action,))
        return ()

    portfolio_mode = str(getattr(portfolio._PORTFOLIO_STATE, "mode", "MEMORY"))
    if portfolio_mode == "MEMORY":
        return v855._BASE_PLAN_CHAIN(self, context_signature, available, **kwargs)
    if portfolio_mode == "RANDOM":
        return ()

    stats = v855._stats(game)
    stats["calls"] += 1.0
    level = int(getattr(v829._CONTROL_STATE, "level", 0))
    warm = v855._game_is_warm(game, level)
    plans, probationary = v855._exact_m7_candidates(
        self,
        context_signature,
        available,
        **kwargs,
    )
    if not plans:
        stats["no_candidate"] += 1.0
        stats["consecutive_exploration"] = 0.0
        return ()

    game_hash = int(stable_u64(game, person=b"v8-game"))
    plans = tuple(
        plan
        for plan in plans
        if _same_world_v855_fixup(self, plan.strategy_uid, game_hash)
    )
    if not plans:
        stats["no_candidate"] += 1.0
        stats["consecutive_exploration"] = 0.0
        return ()

    top = plans[0]
    node = getattr(self, "_node_by_uid", {}).get(top.strategy_uid)
    reliability = (
        float(getattr(node, "strategy_reliability", 0.0))
        if node is not None
        else 0.0
    )
    attempts = (
        float(getattr(node, "attempt_weight", 0.0))
        if node is not None
        else 0.0
    )
    if not probationary and (
        attempts < 3.0 or reliability < v855._MEDIUM_RELIABILITY
    ):
        stats["low_confidence"] += 1.0
        stats["consecutive_exploration"] = 0.0
        return ()

    failures = 0
    if state is not None:
        failures = max(
            0,
            int(v829._NO_PROGRESS.get((*state, int(top.action_id)), 0)),
        )
    decision = v855.adaptive_m7_probability_v855(
        reliability=reliability,
        warm=warm,
        failures=failures,
        probationary=probationary,
    )
    stats["eligible"] += 1.0
    stats["last_floor"] = float(decision.exploration_floor)
    if failures:
        stats["failure_backoff"] += 1.0

    key = _starvation_key(game, level, int(context_signature), top.strategy_uid)
    consecutive = _consecutive(key)
    starvation_release = (
        consecutive >= v855._MAX_CONSECUTIVE_ARBITRATION_EXPLORATION
        and not probationary
        and reliability >= v855._STRONG_RELIABILITY
        and failures < 6
    )
    draw = float(getattr(self, "_behavior_rng").random())
    if starvation_release or draw < float(decision.memory_probability):
        stats["selected"] += 1.0
        stats["consecutive_exploration"] = 0.0
        _set_consecutive(key, 0)
        if starvation_release:
            stats["starvation_release"] += 1.0
        v829._set_selection(
            context_signature,
            "M7_ADAPTIVE",
            (row.action_id for row in plans),
        )
        return plans

    stats["exploration_floor"] += 1.0
    stats["consecutive_exploration"] = float(
        _set_consecutive(key, consecutive + 1)
    )
    return ()


def _plan_chain_v829_fixup(self, context_signature, action_ids, **kwargs):
    """Give v8.55 one adaptive opportunity before v8.29 exhaustive coverage."""
    from v8 import sampling_portfolio_v831 as portfolio
    from v8 import sampling_progress_control_v829 as v829

    if not v829._discovery_mode():
        return _BASE_V829_PLAN(self, context_signature, action_ids, **kwargs)

    available = tuple(sorted({int(value) for value in action_ids}))
    state = v829._state_key(context_signature)
    if state is None or not available:
        return _BASE_V829_PLAN(self, context_signature, action_ids, **kwargs)

    progress_action = v829._PROGRESS_ACTION.get(state)
    if progress_action in available:
        return _BASE_V829_PLAN(self, context_signature, available, **kwargs)

    tested = v829._TESTED.setdefault(state, set())
    coverage_incomplete = any(action not in tested for action in available)
    portfolio_mode = str(getattr(portfolio._PORTFOLIO_STATE, "mode", "MEMORY"))
    if coverage_incomplete and portfolio_mode not in {"MEMORY", "RANDOM"}:
        rows = tuple(
            _plan_chain_v855_fixup(
                self,
                context_signature,
                available,
                **kwargs,
            )
        )
        if rows:
            return rows

    return _BASE_V829_PLAN(self, context_signature, available, **kwargs)


def _reset_adaptive_memory_control_v855_fixup() -> None:
    _BASE_RESET()
    _CONSECUTIVE.clear()


def install_adaptive_memory_control_v855_fixups() -> None:
    global _INSTALLED, _BASE_V829_PLAN, _BASE_RESET
    if _INSTALLED:
        return

    from v8 import adaptive_memory_control_v855 as v855
    from v8 import learning_performance_repair_v824 as v824
    from v8 import sampling_portfolio_v831 as portfolio
    from v8 import sampling_progress_control_v829 as v829

    _BASE_V829_PLAN = v829._plan_chain_v829
    _BASE_RESET = v855._reset_adaptive_memory_control_v855

    v855._same_world = _same_world_v855_fixup
    v855._plan_chain_v855 = _plan_chain_v855_fixup
    v855._reset_adaptive_memory_control_v855 = _reset_adaptive_memory_control_v855_fixup

    portfolio._plan_chain_v831 = v855._plan_chain_v855
    v829._BASE_PLAN_CHAIN = portfolio._plan_chain_v831

    # Preserve the historical public identity relation while inserting a
    # pre-coverage adaptive opportunity above the original v8.29 gate.
    v829._plan_chain_v829 = _plan_chain_v829_fixup
    v824._BASE_PLAN_CHAIN = v829._plan_chain_v829

    _INSTALLED = True
