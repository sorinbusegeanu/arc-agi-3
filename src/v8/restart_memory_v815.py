from __future__ import annotations

import math
import os
import queue
import time
from collections import defaultdict
from dataclasses import replace
from typing import Iterable

from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, RelationType, stable_u64


_INSTALLED = False
_CONTROL_SCOPE_ENV = "ARC_AGI3_V8_CONTROL_SCOPE"
_SESSION_GAMMA = 0.995
_SESSION_HORIZON = 512
_CURRENT_LEVELS_COMPLETED = 0

_BASE_VIEW_INIT = None
_BASE_VIEW_REFRESH = None
_BASE_SCORE_ACTIONS = None
_BASE_PLAN_CANDIDATES = None
_BASE_ENV_STEP = None
_BASE_ENV_RESET = None
_BASE_REPORTING_WORKER = None
_BASE_FORMAT_GAME_RATE_LINE = None


def _current_game_id() -> str:
    try:
        from v8 import trajectory_optimizer_v814 as optimizer

        source = str(getattr(optimizer, "_CAPTURE_SOURCE_ID", ""))
        if source:
            return source
    except BaseException:
        pass
    return str(os.environ.get(_CONTROL_SCOPE_ENV, ""))


def _current_game_hash() -> int:
    game_id = _current_game_id()
    return 0 if not game_id else int(stable_u64(game_id, person=b"v8-game"))


def _is_grounded_m1(row) -> bool:
    return bool(
        int(row.level) == int(MemoryLevel.M1)
        and int(row.memory_type) == int(MemoryType.CONTINGENCY)
        and len(row.key_parts) >= 4
    )


def _is_normalized_m1(row) -> bool:
    try:
        from v8.normalized_memory_v086 import is_normalized_contingency

        return bool(is_normalized_contingency(row))
    except BaseException:
        return False


def _positive_memory_value(row) -> float:
    valence = float(getattr(row, "expected_primary_valence", 0.0))
    confidence = float(getattr(row, "primary_valence_confidence", 0.0))
    future = max(-1.0, min(1.0, float(getattr(row, "future_option_delta", 0.0))))
    transfer = max(0.0, float(getattr(row, "transfer_prior", 0.0)))
    reliability = max(0.0, min(1.0, float(getattr(row, "strategy_reliability", 0.0))))
    value = valence * confidence + 0.15 * max(0.0, future) + 0.10 * transfer
    if int(getattr(row, "level", -1)) == int(MemoryLevel.M7):
        value += 0.25 * reliability
    return float(max(0.0, value))


def _finalize_prior(raw: dict[int, list[float]], *, discount: float) -> dict[int, tuple[int, float]]:
    result: dict[int, tuple[int, float]] = {}
    for action, values in raw.items():
        support = max(0, int(values[0]))
        weight = max(0.0, float(values[2]))
        if support <= 0 or weight <= 0.0:
            continue
        score = max(0.0, float(values[1]) / weight) * float(discount)
        if score <= 0.0:
            continue
        result[int(action)] = (support, score)
    return result


def _build_restart_indexes(view) -> None:
    version = tuple(getattr(view, "_strategy_version", ()))
    game_hash = _current_game_hash()
    cache_key = (version, int(game_hash))
    if getattr(view, "_v815_restart_index_key", None) == cache_key:
        return

    nodes = tuple(getattr(view, "_node_by_uid", {}).values())
    by_uid = {row.uid: row for row in nodes}
    edges = tuple(view.edge_records())

    games_by_uid: dict[MemoryUid, set[int]] = defaultdict(set)
    normalized_by_grounded: dict[MemoryUid, set[MemoryUid]] = defaultdict(set)
    for edge in edges:
        relation = int(edge.relation_type)
        if relation == int(RelationType.GAME_PROVENANCE) and int(edge.target_uid.hi) == 0:
            games_by_uid[edge.source_uid].add(int(edge.target_uid.lo))
        elif relation == int(RelationType.EXPLAINS):
            source = by_uid.get(edge.source_uid)
            target = by_uid.get(edge.target_uid)
            if source is not None and target is not None and _is_normalized_m1(source) and _is_grounded_m1(target):
                normalized_by_grounded[target.uid].add(source.uid)

    same_game_raw: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    normalized_raw: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])

    for row in nodes:
        if not _is_grounded_m1(row):
            continue
        action = int(row.key_parts[1])
        support = max(1, int(row.support_count))
        value = _positive_memory_value(row)
        if value <= 0.0:
            continue

        if game_hash and game_hash in games_by_uid.get(row.uid, set()):
            bucket = same_game_raw[action]
            bucket[0] += support
            bucket[1] += value * support
            bucket[2] += support

        normalized_sources = normalized_by_grounded.get(row.uid, set())
        if normalized_sources:
            normalized_value = value
            normalized_support = support
            for uid in normalized_sources:
                source = by_uid.get(uid)
                if source is None:
                    continue
                normalized_value = max(normalized_value, _positive_memory_value(source))
                normalized_support += max(1, int(source.support_count))
            if normalized_value > 0.0:
                bucket = normalized_raw[action]
                bucket[0] += normalized_support
                bucket[1] += normalized_value * normalized_support
                bucket[2] += normalized_support

    same_game_strategies = []
    seen_strategy_uids: set[MemoryUid] = set()
    dependencies = getattr(view, "_behavior_strategy_dependencies", {})
    strategy_rows = []
    for rows in getattr(view, "_strategy_by_context", {}).values():
        strategy_rows.extend(rows)
    strategy_rows.extend(getattr(view, "_strategy_fallback", ()))
    for strategy in strategy_rows:
        if strategy.strategy_uid in seen_strategy_uids:
            continue
        seen_strategy_uids.add(strategy.strategy_uid)
        strategy_games = set(games_by_uid.get(strategy.strategy_uid, set()))
        for dependency_uid in dependencies.get(strategy.strategy_uid, set()):
            strategy_games.update(games_by_uid.get(dependency_uid, set()))
        if game_hash and game_hash in strategy_games:
            same_game_strategies.append(strategy)

    view._v815_same_game_action_priors = _finalize_prior(same_game_raw, discount=0.70)
    view._v815_normalized_action_priors = _finalize_prior(normalized_raw, discount=0.40)
    view._v815_same_game_strategies = tuple(same_game_strategies)
    view._v815_restart_index_key = cache_key
    view._v815_restart_memory_counts = {
        "same_game_actions": len(view._v815_same_game_action_priors),
        "normalized_actions": len(view._v815_normalized_action_priors),
        "same_game_strategies": len(view._v815_same_game_strategies),
    }


def _session_prior(view, action: int) -> tuple[int, float] | None:
    raw = getattr(view, "_v815_session_action_priors", {}).get(int(action))
    if raw is None:
        return None
    support = max(0, int(raw[0]))
    weight = max(0.0, float(raw[2]))
    if support <= 0 or weight <= 0.0:
        return None
    score = float(raw[1]) / weight
    return None if score <= 0.0 else (support, score)


def _best_prior(view, action: int) -> tuple[int, float, str] | None:
    session = _session_prior(view, action)
    if session is not None:
        return session[0], session[1], "session"
    same = getattr(view, "_v815_same_game_action_priors", {}).get(int(action))
    if same is not None:
        return same[0], same[1], "same_game_m1"
    normalized = getattr(view, "_v815_normalized_action_priors", {}).get(int(action))
    if normalized is not None:
        return normalized[0], normalized[1], "normalized_m1"
    return None


def _augment_scores(view, rows: Iterable[object], *, force_random: bool = False) -> tuple[object, ...]:
    rows = tuple(rows)
    if force_random:
        view._v815_score_origins = {}
        return rows
    _build_restart_indexes(view)
    origins: dict[int, str] = {}
    result = []
    for row in rows:
        if int(row.support_count) > 0 and float(row.score) > 0.0:
            origins[int(row.action_id)] = "exact_m1"
            result.append(row)
            continue
        prior = _best_prior(view, int(row.action_id))
        if prior is None:
            result.append(row)
            continue
        support, score, origin = prior
        origins[int(row.action_id)] = origin
        result.append(
            replace(
                row,
                support_count=max(int(row.support_count), int(support)),
                score=max(float(row.score), float(score)),
            )
        )
    view._v815_score_origins = origins
    return tuple(result)


def _same_game_plans(view, action_ids, **kwargs):
    from v8 import behavior_recovery as behavior

    _build_restart_indexes(view)
    rows = [
        row
        for row in getattr(view, "_v815_same_game_strategies", ())
        if behavior.strategy_can_control(view, row.strategy_uid, row.outcome_uid)
    ]
    if not rows:
        return ()
    return behavior._score_strategy_rows(
        view,
        rows,
        available={int(value) for value in action_ids},
        outcome_uid=kwargs.get("outcome_uid"),
        required_ancestor=kwargs.get("required_ancestor"),
        excluded_strategies=kwargs.get("excluded_strategies", frozenset()),
        ignore_preference=bool(kwargs.get("ignore_preference", False)),
        cross_context=False,
    )


def _inferred_variant_start_level(row) -> int:
    completed = max(0, int(getattr(row.target, "levels_completed", 0)))
    return max(0, completed - 1)


def _phase_variant(view, action_ids):
    try:
        from v8 import trajectory_optimizer_v814 as optimizer
    except BaseException:
        return None

    optimizer._refresh_view_variants(view)
    source_id = _current_game_id()
    if not source_id:
        return None
    attempted = set(getattr(view, "_v814_attempted_variants", set()))
    rows = [
        row
        for row in tuple(getattr(view, "_v814_variants", ()))
        if row.variant_id not in attempted
        and row.anchor.source_id == source_id
        and _inferred_variant_start_level(row) == int(_CURRENT_LEVELS_COMPLETED)
        and row.actions
    ]
    if not rows:
        return None
    current_seed = int(getattr(optimizer, "_CAPTURE_SEED", 0))
    history = tuple(getattr(optimizer, "_ACTOR_ACTION_HISTORY", ()))
    rows.sort(
        key=lambda row: (
            0 if int(row.anchor.seed) == current_seed else 1,
            0 if tuple(row.anchor.prefix_actions) == history else 1,
            -int(row.saved_actions),
            int(row.cost),
            row.variant_id,
        )
    )
    available = {int(value) for value in action_ids}
    for row in rows:
        if int(row.actions[0]) in available:
            return row
    return None


def _is_exact_base_plan(view, plan, context_signature: int) -> bool:
    if float(getattr(plan, "score", 0.0)) >= 100_000.0:
        return True
    row = getattr(view, "_node_by_uid", {}).get(plan.strategy_uid)
    if row is None or len(row.key_parts) < 4:
        return False
    try:
        from v8.learning_blockers_v055 import is_composite_strategy

        if is_composite_strategy(row):
            return True
    except BaseException:
        pass
    exact_bucket = stable_u64(int(context_signature), person=b"v8-context")
    return int(row.key_parts[3]) == int(exact_bucket)


def _credit_session(view, *, success: bool, failure: bool) -> None:
    trajectory = list(getattr(view, "_v815_session_trajectory", ()))
    if not trajectory:
        return
    if success:
        priors = getattr(view, "_v815_session_action_priors", {})
        for distance, (_context, action) in enumerate(reversed(trajectory[-_SESSION_HORIZON:])):
            credit = _SESSION_GAMMA ** distance
            bucket = priors.setdefault(int(action), [0.0, 0.0, 0.0])
            bucket[0] += 1.0
            bucket[1] += float(credit)
            bucket[2] += 1.0
        view._v815_session_action_priors = priors
    if success or failure:
        view._v815_session_trajectory = []


def _invalidate_actor_view() -> None:
    try:
        from v8 import behavior_recovery as behavior

        view = getattr(behavior, "_CURRENT_ACTOR_VIEW", None)
        if view is not None:
            view.invalidate_strategy_cache()
            view._v815_restart_index_key = None
    except BaseException:
        return


def _environment_step_v815(self, action):
    global _CURRENT_LEVELS_COMPLETED
    result = _BASE_ENV_STEP(self, action)
    try:
        from v8 import behavior_recovery as behavior

        view = getattr(behavior, "_CURRENT_ACTOR_VIEW", None)
    except BaseException:
        view = None
    if view is not None:
        last = getattr(view, "_behavior_last_action", None)
        if last is not None:
            view._v815_session_trajectory.append((int(last[0]), int(last[1])))
        state = str(getattr(self, "last_outcome_state", ""))
        level_success = bool(getattr(self, "level_completed_event", False))
        success = bool(level_success or state == "WIN")
        failure = state == "GAME_OVER"
        _credit_session(view, success=success, failure=failure)
        if success or failure or bool(getattr(self, "last_step_was_reset_boundary", False)):
            view.invalidate_strategy_cache()
            view._v815_restart_index_key = None
    _CURRENT_LEVELS_COMPLETED = max(0, int(getattr(self, "last_levels_completed", 0)))
    return result


def _environment_reset_v815(self, *args, **kwargs):
    global _CURRENT_LEVELS_COMPLETED
    result = _BASE_ENV_RESET(self, *args, **kwargs)
    _CURRENT_LEVELS_COMPLETED = max(0, int(getattr(self, "last_levels_completed", 0)))
    _invalidate_actor_view()
    return result


def _reporting_worker_v815(
    *,
    event_queue,
    stop_event,
    watermark,
    actors,
    interval_seconds: float,
    output_queue=None,
) -> None:
    del watermark
    from v8.actor import ActorProgress
    from v8.reporter import _emit_line
    from v8.diagnostics import format_game_rate_line

    latest = {
        int(actor_id): ActorProgress(int(actor_id), str(game_id), 0, 0, 0, 0)
        for actor_id, game_id in actors
    }
    expected = set(latest)
    seen: set[int] = set()
    next_report = time.monotonic() + float(interval_seconds)
    while not stop_event.is_set():
        now = time.monotonic()
        timeout = max(0.0, min(0.25, next_report - now))
        try:
            row = event_queue.get(timeout=timeout)
        except queue.Empty:
            row = None
        if isinstance(row, ActorProgress):
            actor_id = int(row.actor_id)
            latest[actor_id] = row
            seen.add(actor_id)
        now = time.monotonic()
        if now < next_report:
            continue
        if expected and expected.issubset(seen):
            rows = tuple(latest[key] for key in sorted(latest))
            _emit_line(format_game_rate_line(rows), output_queue)
        while next_report <= now:
            next_report += float(interval_seconds)


def install_restart_memory_v815() -> None:
    global _INSTALLED
    global _BASE_VIEW_INIT, _BASE_VIEW_REFRESH, _BASE_SCORE_ACTIONS, _BASE_PLAN_CANDIDATES
    global _BASE_ENV_STEP, _BASE_ENV_RESET, _BASE_REPORTING_WORKER
    if _INSTALLED:
        return

    from v7.environment.arc_adapter import ArcGridEnvironment
    from v8.publication import LiveReadView, PlannedAction
    from v8 import reporter as reporter_module

    _BASE_VIEW_INIT = LiveReadView.__init__
    _BASE_VIEW_REFRESH = LiveReadView._refresh_strategy_cache
    _BASE_SCORE_ACTIONS = LiveReadView.score_actions
    _BASE_PLAN_CANDIDATES = LiveReadView.plan_candidates
    _BASE_ENV_STEP = ArcGridEnvironment.step
    _BASE_ENV_RESET = ArcGridEnvironment.reset
    _BASE_REPORTING_WORKER = reporter_module.reporting_worker

    def view_init(self, *args, **kwargs):
        _BASE_VIEW_INIT(self, *args, **kwargs)
        self._v815_restart_index_key = None
        self._v815_same_game_action_priors = {}
        self._v815_normalized_action_priors = {}
        self._v815_same_game_strategies = ()
        self._v815_restart_memory_counts = {}
        self._v815_session_action_priors = {}
        self._v815_session_trajectory = []
        self._v815_score_origins = {}
        self._v815_last_plan_origin = ""
        if bool(getattr(self, "_behavior_actor_mode", False)):
            self._refresh_strategy_cache()

    def refresh(self):
        _BASE_VIEW_REFRESH(self)
        _build_restart_indexes(self)

    def score_actions(self, context_signature, action_ids):
        forced = bool(getattr(self, "_behavior_force_random", False))
        escape = int(getattr(self, "_v055_escape_budget", 0)) > 0
        rows = _BASE_SCORE_ACTIONS(self, context_signature, action_ids)
        if forced or escape:
            self._v815_score_origins = {}
            return rows
        return _augment_scores(self, rows, force_random=False)

    def plan_candidates(self, context_signature, action_ids, **kwargs):
        base = tuple(_BASE_PLAN_CANDIDATES(self, context_signature, action_ids, **kwargs))
        if bool(getattr(self, "_behavior_force_random", False)) or int(
            getattr(self, "_v055_escape_budget", 0)
        ) > 0:
            self._v815_last_plan_origin = "exploration"
            return base
        if base and any(_is_exact_base_plan(self, plan, context_signature) for plan in base):
            self._v815_last_plan_origin = "exact_or_sequence"
            return base

        phase = _phase_variant(self, action_ids)
        if phase is not None:
            self._v814_attempted_variants.add(phase.variant_id)
            self._v814_active_variant = phase
            self._v814_active_actions = tuple(phase.actions[1:])
            plan = PlannedAction(
                int(phase.actions[0]),
                phase.target_outcome_uid,
                phase.strategy_uid,
                900_000.0,
                False,
            )
            self._behavior_last_plans = (plan,)
            self._v815_last_plan_origin = "phase_trajectory"
            return (plan,)

        same_game = tuple(_same_game_plans(self, action_ids, **kwargs))
        if same_game:
            self._behavior_last_plans = same_game
            self._v815_last_plan_origin = "same_game_m7"
            return same_game

        self._v815_last_plan_origin = "cross_context" if base else ""
        return base

    LiveReadView.__init__ = view_init
    LiveReadView._refresh_strategy_cache = refresh
    LiveReadView.score_actions = score_actions
    LiveReadView.plan_candidates = plan_candidates
    ArcGridEnvironment.step = _environment_step_v815
    ArcGridEnvironment.reset = _environment_reset_v815
    reporter_module.reporting_worker = _reporting_worker_v815
    _INSTALLED = True
