from __future__ import annotations

from v8.model import stable_u64


_INSTALLED = False


def _install_cross_context_probe_fallback() -> None:
    from v8 import behavior_recovery as behavior_module
    from v8 import learning_blockers_v055 as blocker_module
    from v8.publication import LiveReadView

    current_plan_candidates = LiveReadView.plan_candidates

    def plan_candidates(self, context_signature, action_ids, **kwargs):
        plans = tuple(current_plan_candidates(self, context_signature, action_ids, **kwargs))
        required_ancestor = kwargs.get("required_ancestor")

        # v8.5 composite planning admitted probationary composites directly.  Keep
        # those available for an explicit validation probe, but never let them
        # bypass the normal behavioral control gate.
        if required_ancestor is None:
            by_uid = getattr(self, "_node_by_uid", {})
            return tuple(
                plan
                for plan in plans
                if not blocker_module.is_composite_strategy(by_uid.get(plan.strategy_uid))
                or behavior_module.strategy_can_control(
                    self, plan.strategy_uid, plan.outcome_uid
                )
            )

        if plans:
            return plans

        self._refresh_strategy_cache()
        context_bucket = stable_u64(int(context_signature), person=b"v8-context")
        available = {int(value) for value in action_ids}
        exact = list(getattr(self, "_strategy_by_context", {}).get(context_bucket, ()))
        exact_uids = {row.strategy_uid for row in exact}
        fallback = [
            row
            for row in getattr(self, "_strategy_fallback", ())
            if row.strategy_uid not in exact_uids
            and row.action_id in available
            and self.strategy_has_ancestor(row.strategy_uid, required_ancestor)
            and behavior_module._strategy_can_probe(self, row.strategy_uid, row.outcome_uid)
        ]
        if not fallback:
            return ()
        return tuple(
            behavior_module._score_strategy_rows(
                self,
                fallback,
                available=available,
                outcome_uid=kwargs.get("outcome_uid"),
                required_ancestor=required_ancestor,
                excluded_strategies=kwargs.get("excluded_strategies", frozenset()),
                ignore_preference=True,
                cross_context=True,
            )
        )

    LiveReadView.plan_candidates = plan_candidates


def install_learning_fixes_v088_fixups() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_cross_context_probe_fallback()
    _INSTALLED = True
