from __future__ import annotations

"""Target-scoped learning authority for cross-game M7 execution."""

from dataclasses import is_dataclass, replace

from v8.model import MemoryLevel, stable_u64


_INSTALLED = False
_BASE_RECORD_RESULTS = None
_BASE_OBSERVED = None


def _foreign_strategy(runtime, game_hash: int, uid, cache: dict) -> bool:
    if uid in cache:
        return bool(cache[uid])
    try:
        games = frozenset(int(value) for value in runtime.read_view.source_games(uid))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        games = frozenset()
    # Missing provenance is not enough evidence to rewrite intrinsic learning. The
    # transfer control path itself already fails closed when provenance is absent.
    foreign = bool(games) and int(game_hash) not in games
    cache[uid] = bool(foreign)
    return bool(foreign)


def _replace_row(row, **changes):
    if not changes:
        return row
    if is_dataclass(row):
        return replace(row, **changes)
    try:
        from types import SimpleNamespace

        values = dict(vars(row))
        values.update(changes)
        return SimpleNamespace(**values)
    except (TypeError, ValueError):
        return row


def _filter_target_scoped_learning(runtime, row):
    """Prevent target-world execution from mutating source-intrinsic M7 quality."""
    game_id = str(getattr(row, "game_id", ""))
    if not game_id:
        return row
    game_hash = int(stable_u64(game_id, person=b"v8-game"))
    cache = {}
    changes = {}

    stats = getattr(row, "strategy_stats", None)
    if stats is not None:
        kept = tuple(
            stat
            for stat in stats
            if not _foreign_strategy(runtime, game_hash, stat.strategy_uid, cache)
        )
        if kept != tuple(stats):
            changes["strategy_stats"] = kept

    credits = getattr(row, "primary_valence_credits", None)
    if credits is not None:
        kept = tuple(
            credit
            for credit in credits
            if not (
                int(getattr(credit, "level", -1)) == int(MemoryLevel.M7)
                and _foreign_strategy(runtime, game_hash, credit.uid, cache)
            )
        )
        if kept != tuple(credits):
            changes["primary_valence_credits"] = kept

    trials = getattr(row, "replanning_trials", None)
    if trials is not None:
        kept = tuple(
            trial
            for trial in trials
            if not (
                _foreign_strategy(
                    runtime,
                    game_hash,
                    trial.primary_strategy_uid,
                    cache,
                )
                or _foreign_strategy(
                    runtime,
                    game_hash,
                    trial.alternative_strategy_uid,
                    cache,
                )
            )
        )
        if kept != tuple(trials):
            changes["replanning_trials"] = kept
            if hasattr(row, "replans"):
                changes["replans"] = len(kept)

    pending = getattr(row, "pending_learning", None)
    if pending is not None:
        filtered_pending = _filter_target_scoped_learning(runtime, pending)
        if filtered_pending is not pending:
            changes["pending_learning"] = filtered_pending

    return _replace_row(row, **changes)


def _record_actor_results_v856(self, results) -> None:
    filtered = tuple(_filter_target_scoped_learning(self, row) for row in results)
    return _BASE_RECORD_RESULTS(self, filtered)


def _capture_plan_state(view):
    if view is None:
        return None
    last_action = getattr(view, "_behavior_last_action", None)
    plans = tuple(getattr(view, "_behavior_last_plans", ()))
    if last_action is None or not plans:
        return None
    action = int(last_action[1])
    plan = next((row for row in plans if int(row.action_id) == action), None)
    if plan is None:
        return None
    try:
        games = frozenset(int(value) for value in view.source_games(plan.strategy_uid))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        games = frozenset()
    try:
        from v8 import sampling_progress_control_v829 as v829

        game = str(getattr(v829._CONTROL_STATE, "game_id", ""))
    except (AttributeError, ImportError):
        game = ""
    if not game or not games:
        return None
    current = int(stable_u64(game, person=b"v8-game"))
    if current in games:
        return None
    return game, int(last_action[0]), plan


def _observed_target_scope_v856(**kwargs):
    """Neutral target mismatch is inconclusive, not evidence against source M7."""
    from v8 import adaptive_memory_transfer_integrity_v856 as v856

    try:
        from v8 import behavior_recovery as behavior

        view = getattr(behavior, "_CURRENT_ACTOR_VIEW", None)
    except (AttributeError, ImportError):
        view = None
    captured = _capture_plan_state(view)
    prior_local = prior_target = None
    local_key = target_key = None
    if captured is not None:
        game, context, plan = captured
        local_key = v856._failure_key(game, context, plan.strategy_uid)
        target_key = v856._transfer_failure_key(game, plan.strategy_uid)
        prior_local = v856._RECENT_FAILURES.get(local_key)
        prior_target = v856._TARGET_TRANSFER_FAILURES.get(target_key)

    result = _BASE_OBSERVED(**kwargs)

    if captured is None:
        return result
    _game, _context, plan = captured
    observed = {uid for uid in result if not uid.is_zero}
    represented_success = plan.outcome_uid in observed
    polarity = int(kwargs.get("terminal_polarity", 0))
    if represented_success or polarity != 0:
        return result

    # A foreign strategy points at a source-world M6 identity. A neutral target
    # transition can therefore be useful without matching that exact UID. Restore
    # the pre-step operational counters; only positive evidence resets them and a
    # hard negative primitive outcome increases target-local backoff.
    if prior_local is None:
        v856._RECENT_FAILURES.pop(local_key, None)
    else:
        v856._RECENT_FAILURES[local_key] = int(prior_local)
    if prior_target is None:
        v856._TARGET_TRANSFER_FAILURES.pop(target_key, None)
    else:
        v856._TARGET_TRANSFER_FAILURES[target_key] = int(prior_target)
    return result


def install_adaptive_memory_transfer_scope_v856() -> None:
    global _INSTALLED, _BASE_RECORD_RESULTS, _BASE_OBSERVED
    if _INSTALLED:
        return

    from v8 import actor as actor_module
    from v8.runtime import ContinuousMemoryRuntime

    _BASE_RECORD_RESULTS = ContinuousMemoryRuntime.record_actor_results
    ContinuousMemoryRuntime.record_actor_results = _record_actor_results_v856

    _BASE_OBSERVED = actor_module._observed_outcome_uids
    actor_module._observed_outcome_uids = _observed_target_scope_v856

    _INSTALLED = True
