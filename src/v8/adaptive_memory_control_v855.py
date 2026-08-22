from __future__ import annotations

"""v8.55 adaptive M7 control arbitration.

The sampling portfolio remains the exploration authority, but DISCOVERY no longer
hard-disables every learned strategy outside the MEMORY slot. Reliable exact-context
M7 strategies compete with exploration while an independent exploration floor,
uncertainty, and repeated no-progress evidence prevent memory from monopolizing
behavior. Explicit VERIFY/ALTERNATIVE/TRANSFER modes retain their existing policy.
"""

from dataclasses import dataclass

from v8.model import stable_u64


_INSTALLED = False
_BASE_PLAN_CHAIN = None

_COLD_EXPLORATION_FLOOR = 0.25
_WARM_EXPLORATION_FLOOR = 0.15
_MAX_EXPLORATION = 0.70
_STRONG_RELIABILITY = 0.75
_MEDIUM_RELIABILITY = 0.50
_PROBE_MEMORY_PROBABILITY = 0.10
_MAX_CONSECUTIVE_ARBITRATION_EXPLORATION = 8

_WARM_GAMES: set[str] = set()
_CONTROL: dict[str, dict[str, float]] = {}


@dataclass(frozen=True, slots=True)
class M7ArbitrationDecision:
    memory_probability: float
    exploration_probability: float
    exploration_floor: float
    reliability: float
    failures: int
    probationary: bool
    progress_replay: bool


def adaptive_m7_probability_v855(
    *,
    reliability: float,
    warm: bool,
    failures: int = 0,
    probationary: bool = False,
    progress_replay: bool = False,
) -> M7ArbitrationDecision:
    """Return bounded M7/exploration probabilities for one exact-context candidate."""

    reliability = max(0.0, min(1.0, float(reliability)))
    failures = max(0, int(failures))
    floor = _WARM_EXPLORATION_FLOOR if bool(warm) else _COLD_EXPLORATION_FLOOR

    uncertainty_bonus = max(0.0, _STRONG_RELIABILITY - reliability) * 0.40
    failure_bonus = min(0.30, 0.05 * failures)
    probation_bonus = 0.20 if probationary else 0.0
    explore = min(
        _MAX_EXPLORATION,
        max(floor, floor + uncertainty_bonus + failure_bonus + probation_bonus),
    )
    memory = max(0.0, 1.0 - explore)

    if probationary:
        memory = min(memory, _PROBE_MEMORY_PROBABILITY)
    elif reliability < _STRONG_RELIABILITY:
        memory = min(memory, 0.50)

    if progress_replay and not probationary:
        memory = max(memory, min(1.0 - floor, 0.85))

    # Repeated no-progress evidence backs memory off even when an old reliability
    # estimate remains high. The hard floor is reapplied after the reduction.
    if failures:
        memory *= 1.0 / (1.0 + 0.25 * failures)

    memory = max(0.0, min(memory, 1.0 - floor))
    explore = max(floor, min(1.0, 1.0 - memory))
    return M7ArbitrationDecision(
        memory,
        explore,
        floor,
        reliability,
        failures,
        bool(probationary),
        bool(progress_replay),
    )


def _stats(game_id: str) -> dict[str, float]:
    return _CONTROL.setdefault(
        str(game_id),
        {
            "calls": 0.0,
            "eligible": 0.0,
            "selected": 0.0,
            "exploration_floor": 0.0,
            "low_confidence": 0.0,
            "failure_backoff": 0.0,
            "no_candidate": 0.0,
            "starvation_release": 0.0,
            "consecutive_exploration": 0.0,
            "last_floor": _COLD_EXPLORATION_FLOOR,
        },
    )


def adaptive_memory_control_telemetry_v855(game_id: str) -> dict[str, float]:
    return dict(_stats(str(game_id)))


def _game_is_warm(game_id: str, level: int) -> bool:
    game = str(game_id)
    if int(level) > 0 or game in _WARM_GAMES:
        _WARM_GAMES.add(game)
        return True
    try:
        from v8 import sampling_progress_control_v829 as v829

        if any(str(key[0]) == game for key in v829._PROGRESS_ACTION):
            _WARM_GAMES.add(game)
            return True
    except (AttributeError, TypeError, ValueError):
        pass
    return False


def _same_world(view, strategy_uid, game_hash: int) -> bool:
    try:
        games = set(int(value) for value in view.source_games(strategy_uid))
    except (AttributeError, TypeError, ValueError):
        games = set()
    # Unknown provenance is not treated as foreign. Explicitly foreign-only M7 is
    # left to the existing TRANSFER mode and its target-scoped validation policy.
    return not games or int(game_hash) in games


def _exact_m7_candidates(view, context_signature: int, action_ids, **kwargs):
    from v8 import behavior_recovery as behavior

    view._refresh_strategy_cache()
    available = {int(value) for value in action_ids}
    bucket = stable_u64(int(context_signature), person=b"v8-context")
    exact = list(getattr(view, "_strategy_by_context", {}).get(bucket, ()))
    outcome_uid = kwargs.get("outcome_uid")
    required_ancestor = kwargs.get("required_ancestor")
    excluded = kwargs.get("excluded_strategies", frozenset())
    ignore_preference = bool(kwargs.get("ignore_preference", False))

    controls = [
        row
        for row in exact
        if behavior.strategy_can_control(view, row.strategy_uid, row.outcome_uid)
    ]
    probes = []
    if not controls:
        probes = [
            row
            for row in exact
            if behavior._strategy_can_probe(view, row.strategy_uid, row.outcome_uid)
        ]
    rows = controls if controls else probes
    if not rows:
        return (), False
    plans = behavior._score_strategy_rows(
        view,
        rows,
        available=available,
        outcome_uid=outcome_uid,
        required_ancestor=required_ancestor,
        excluded_strategies=excluded,
        ignore_preference=ignore_preference,
        cross_context=False,
    )
    return tuple(plans), bool(not controls and probes)


def _plan_chain_v855(self, context_signature, action_ids, **kwargs):
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import sampling_portfolio_v831 as portfolio
    from v8 import sampling_progress_control_v829 as v829

    mode = str(
        __import__("os").environ.get(
            v819._SAMPLING_MODE_ENV,
            v819.SamplingMode.DISCOVERY.value,
        )
    )
    if mode != v819.SamplingMode.DISCOVERY.value:
        return _BASE_PLAN_CHAIN(self, context_signature, action_ids, **kwargs)

    game = str(getattr(v829._CONTROL_STATE, "game_id", ""))
    if not game:
        return _BASE_PLAN_CHAIN(self, context_signature, action_ids, **kwargs)

    available = tuple(sorted({int(value) for value in action_ids}))
    if not available:
        return ()

    state = v829._state_key(context_signature)
    progress_action = None if state is None else v829._PROGRESS_ACTION.get(state)
    if progress_action in available:
        # Proven progress replay remains authoritative; adaptive M7 must not displace
        # a directly observed successful action for this exact state.
        v829._set_selection(context_signature, "PROGRESS_REPLAY", (progress_action,))
        return ()

    portfolio_mode = str(getattr(portfolio._PORTFOLIO_STATE, "mode", "MEMORY"))
    if portfolio_mode == "RANDOM":
        # Existing unconditional random portfolio slots remain a hard exploration
        # channel independent of learned-memory confidence.
        return ()

    stats = _stats(game)
    stats["calls"] += 1.0
    level = int(getattr(v829._CONTROL_STATE, "level", 0))
    warm = _game_is_warm(game, level)
    plans, probationary = _exact_m7_candidates(
        self,
        context_signature,
        available,
        **kwargs,
    )
    if not plans:
        stats["no_candidate"] += 1.0
        stats["consecutive_exploration"] += 1.0
        return ()

    game_hash = int(stable_u64(game, person=b"v8-game"))
    plans = tuple(
        plan for plan in plans if _same_world(self, plan.strategy_uid, game_hash)
    )
    if not plans:
        stats["no_candidate"] += 1.0
        stats["consecutive_exploration"] += 1.0
        return ()

    top = plans[0]
    node = getattr(self, "_node_by_uid", {}).get(top.strategy_uid)
    reliability = float(getattr(node, "strategy_reliability", 0.0)) if node is not None else 0.0
    attempts = float(getattr(node, "attempt_weight", 0.0)) if node is not None else 0.0
    if not probationary and (attempts < 3.0 or reliability < _MEDIUM_RELIABILITY):
        stats["low_confidence"] += 1.0
        stats["consecutive_exploration"] += 1.0
        return ()

    failures = 0
    if state is not None:
        failures = max(
            0,
            int(v829._NO_PROGRESS.get((*state, int(top.action_id)), 0)),
        )
    historical_progress = bool(
        state is not None and v829._PROGRESS_ACTION.get(state) == int(top.action_id)
    )
    decision = adaptive_m7_probability_v855(
        reliability=reliability,
        warm=warm,
        failures=failures,
        probationary=probationary,
        progress_replay=historical_progress,
    )
    stats["eligible"] += 1.0
    stats["last_floor"] = float(decision.exploration_floor)
    if failures:
        stats["failure_backoff"] += 1.0

    consecutive = int(stats.get("consecutive_exploration", 0.0))
    starvation_release = (
        consecutive >= _MAX_CONSECUTIVE_ARBITRATION_EXPLORATION
        and not probationary
        and reliability >= _STRONG_RELIABILITY
        and failures < 6
    )
    draw = float(getattr(self, "_behavior_rng").random())
    if starvation_release or draw < float(decision.memory_probability):
        stats["selected"] += 1.0
        stats["consecutive_exploration"] = 0.0
        if starvation_release:
            stats["starvation_release"] += 1.0
        v829._set_selection(context_signature, "M7_ADAPTIVE", (row.action_id for row in plans))
        return plans

    stats["exploration_floor"] += 1.0
    stats["consecutive_exploration"] += 1.0
    return ()


def _reset_adaptive_memory_control_v855() -> None:
    _WARM_GAMES.clear()
    _CONTROL.clear()


def install_adaptive_memory_control_v855() -> None:
    global _INSTALLED, _BASE_PLAN_CHAIN
    if _INSTALLED:
        return

    from v8 import learning_performance_repair_v824 as v824

    # v824 is the public planner wrapper. Later layers v8.29/v8.31/v8.54 compose
    # through this delegate, so inserting here relaxes only the DISCOVERY hard gate
    # while keeping those historical public function identities unchanged.
    _BASE_PLAN_CHAIN = v824._BASE_PLAN_CHAIN
    v824._BASE_PLAN_CHAIN = _plan_chain_v855
    _INSTALLED = True
