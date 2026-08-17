from __future__ import annotations

"""v8.17 safety fix for restart-memory control.

v8.15 added useful restart publication/refresh behavior, but its actor policy
fallbacks collapsed context: same-game M1 priors, same-game M7 strategies, and
same-level trajectory variants could control an actor outside the context in
which they were learned.  Restore the pre-v8.15 policy stack for actor control
while retaining the v8.15 refresh/reporting infrastructure.
"""

_INSTALLED = False
_BASE_REFRESH = None
_BASE_SCORE = None
_BASE_PLAN = None


def install_restart_control_safety_v817() -> None:
    global _INSTALLED, _BASE_REFRESH, _BASE_SCORE, _BASE_PLAN
    if _INSTALLED:
        return

    from v8.publication import LiveReadView
    from v8 import restart_memory_v815 as restart
    from v8 import restart_memory_v815_fixups as refresh_fixups

    _BASE_REFRESH = LiveReadView._refresh_strategy_cache
    _BASE_SCORE = LiveReadView.score_actions
    _BASE_PLAN = LiveReadView.plan_candidates

    def refresh(self):
        if bool(getattr(self, "_behavior_actor_mode", False)):
            # Use the policy/index refresh that existed immediately before v8.15.
            # This preserves exact-context M1/M7, structural policy layers, and
            # v8.14 exact trajectory matching without constructing context-free
            # same-game control indexes over the full restored graph.
            return restart._BASE_VIEW_REFRESH(self)
        return _BASE_REFRESH(self)

    def score_actions(self, context_signature, action_ids):
        refresh_fixups._refresh_if_published(self)
        if bool(getattr(self, "_behavior_actor_mode", False)):
            # Pre-v8.15 scoring is context-sensitive.  In particular, an action
            # successful elsewhere in the same game must remain unseen here when
            # the current context has no supporting contingency.
            return restart._BASE_SCORE_ACTIONS(self, context_signature, action_ids)
        return _BASE_SCORE(self, context_signature, action_ids)

    def plan_candidates(self, context_signature, action_ids, **kwargs):
        refresh_fixups._refresh_if_published(self)
        if bool(getattr(self, "_behavior_actor_mode", False)):
            # Bypass v8.15's same-game M7 and same-level trajectory fallbacks.
            # The retained v8.14 layer still permits a validated trajectory when
            # its original seed/prefix anchor matches exactly.
            return restart._BASE_PLAN_CANDIDATES(
                self, context_signature, action_ids, **kwargs
            )
        return _BASE_PLAN(self, context_signature, action_ids, **kwargs)

    LiveReadView._refresh_strategy_cache = refresh
    LiveReadView.score_actions = score_actions
    LiveReadView.plan_candidates = plan_candidates
    _INSTALLED = True
