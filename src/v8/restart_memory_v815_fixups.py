from __future__ import annotations

import queue
import time


_INSTALLED = False
_BASE_GRAPH_CHECK = None
_BASE_SCORE = None
_BASE_PLAN = None
_BASE_STEP = None
_BASE_INIT = None


def _arena_versions(view) -> tuple[int, ...]:
    return tuple(
        int(arena.sequence)
        for arena in (*getattr(view, "_nodes", ()), *getattr(view, "_edges", ()))
    )


def _arm_refresh_retry(view, *, seconds: float = 2.0) -> None:
    view._v815_wait_version = _arena_versions(view)
    view._v815_wait_until = time.monotonic() + max(0.1, float(seconds))
    view._v815_wait_next = 0.0


def _refresh_if_published(view) -> None:
    prior = tuple(getattr(view, "_v815_wait_version", ()))
    if not prior:
        return
    now = time.monotonic()
    if now > float(getattr(view, "_v815_wait_until", 0.0)):
        view._v815_wait_version = ()
        return
    if now < float(getattr(view, "_v815_wait_next", 0.0)):
        return
    view._v815_wait_next = now + 0.05
    current = _arena_versions(view)
    if current != prior and not any(value & 1 for value in current):
        view.invalidate_strategy_cache()
        view._v815_restart_index_key = None
        view._v815_wait_version = ()


def _reporting_worker_after_first_progress(
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
    from v8.diagnostics import format_game_rate_line
    from v8.reporter import _emit_line

    latest = {
        int(actor_id): ActorProgress(int(actor_id), str(game_id), 0, 0, 0, 0)
        for actor_id, game_id in actors
    }
    seen_progress = False
    next_report = time.monotonic() + float(interval_seconds)
    while not stop_event.is_set():
        now = time.monotonic()
        timeout = max(0.0, min(0.25, next_report - now))
        try:
            row = event_queue.get(timeout=timeout)
        except queue.Empty:
            row = None
        if isinstance(row, ActorProgress):
            latest[int(row.actor_id)] = row
            seen_progress = True
        now = time.monotonic()
        if now < next_report:
            continue
        if seen_progress:
            rows = tuple(latest[key] for key in sorted(latest))
            _emit_line(format_game_rate_line(rows), output_queue)
        while next_report <= now:
            next_report += float(interval_seconds)


def install_restart_memory_v815_fixups() -> None:
    global _INSTALLED
    global _BASE_GRAPH_CHECK, _BASE_SCORE, _BASE_PLAN, _BASE_STEP, _BASE_INIT
    if _INSTALLED:
        return

    from v7.environment.arc_adapter import ArcGridEnvironment
    from v8 import actor as actor_module
    from v8 import behavior_recovery as behavior
    from v8 import reporter as reporter_module
    from v8.publication import LiveReadView

    _BASE_GRAPH_CHECK = actor_module._refresh_actor_graph_if_due
    _BASE_SCORE = LiveReadView.score_actions
    _BASE_PLAN = LiveReadView.plan_candidates
    _BASE_STEP = ArcGridEnvironment.step
    _BASE_INIT = LiveReadView.__init__

    def init(self, *args, **kwargs):
        _BASE_INIT(self, *args, **kwargs)
        self._v815_wait_version = ()
        self._v815_wait_until = 0.0
        self._v815_wait_next = 0.0

    def score_actions(self, context_signature, action_ids):
        _refresh_if_published(self)
        return _BASE_SCORE(self, context_signature, action_ids)

    def plan_candidates(self, context_signature, action_ids, **kwargs):
        _refresh_if_published(self)
        return _BASE_PLAN(self, context_signature, action_ids, **kwargs)

    def step(self, action):
        result = _BASE_STEP(self, action)
        view = getattr(behavior, "_CURRENT_ACTOR_VIEW", None)
        if view is not None and (
            bool(getattr(self, "level_completed_event", False))
            or str(getattr(self, "last_outcome_state", "")) in {"WIN", "GAME_OVER"}
            or bool(getattr(self, "last_step_was_reset_boundary", False))
        ):
            _arm_refresh_retry(view)
        return result

    def refresh_actor_graph_if_due(
        read_view,
        *,
        completed_steps: int,
        next_check_step: int,
        check_interval_steps: int = actor_module._ACTOR_GRAPH_CHECK_INTERVAL_STEPS,
    ) -> int:
        # The actor calls this before taking the next action. Use the impending
        # accepted step for the boundary test so a 1000-step job with a 1000-step
        # graph interval performs one shared-graph check instead of zero.
        return _BASE_GRAPH_CHECK(
            read_view,
            completed_steps=int(completed_steps) + 1,
            next_check_step=int(next_check_step),
            check_interval_steps=int(check_interval_steps),
        )

    LiveReadView.__init__ = init
    LiveReadView.score_actions = score_actions
    LiveReadView.plan_candidates = plan_candidates
    ArcGridEnvironment.step = step
    actor_module._refresh_actor_graph_if_due = refresh_actor_graph_if_due
    reporter_module.reporting_worker = _reporting_worker_after_first_progress
    _INSTALLED = True
