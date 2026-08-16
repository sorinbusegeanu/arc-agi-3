from __future__ import annotations

from dataclasses import replace

from v8.model import stable_u64


_INSTALLED = False
_RELATIVE_EFFICIENCY_WEIGHT = 0.15


def _install_outcome_conditioned_efficiency() -> None:
    """Keep efficiency comparative inside one outcome/context cohort."""
    from v8 import behavior_recovery as behavior_module
    from v8 import learning_blockers_v055 as blocker_module

    # v8.8 initially restored the pre-v8.4 absolute cost term to regain search
    # pressure. Search is now explicit, so remove that term again and retain the
    # v8.4 outcome-conditioned relative efficiency semantics.
    current_score_rows = behavior_module._score_strategy_rows

    def score_rows(view, rows, **kwargs):
        rows = tuple(rows)
        plans = list(current_score_rows(view, rows, **kwargs))
        by_uid = {row.strategy_uid: row for row in rows}
        adjusted = []
        for plan in plans:
            row = by_uid.get(plan.strategy_uid)
            if row is None:
                adjusted.append(plan)
                continue
            absolute = 0.10 / max(1.0, float(row.mean_cost))
            adjusted.append(replace(plan, score=float(plan.score) - absolute))
        adjusted.sort(key=lambda item: (-item.score, item.action_id, item.strategy_uid))
        return tuple(adjusted)

    behavior_module._score_strategy_rows = score_rows

    # v8.5 composite plans also carried an unconditional 1/path-length bonus, and
    # v8.8 initially added empirical absolute cost on top. Remove both and add a
    # relative term only when at least two procedures target the same M6 outcome.
    current_composites = blocker_module._composite_plans

    def composite_plans(view, context_signature, action_ids):
        plans = list(current_composites(view, context_signature, action_ids))
        by_uid = getattr(view, "_node_by_uid", {})
        rows = []
        grouped: dict[object, list[int]] = {}
        for index, plan in enumerate(plans):
            row = by_uid.get(plan.strategy_uid)
            if row is None:
                rows.append((plan, float(plan.score), 1.0))
                grouped.setdefault(plan.outcome_uid, []).append(index)
                continue
            path = blocker_module._path_for_composite(view, row, int(context_signature))
            path_cost = float(max(1, len(path)))
            base_score = float(plan.score) - 0.10 / path_cost
            attempts = float(getattr(row, "attempt_weight", 0.0))
            if attempts > 0.0:
                empirical_cost = max(1.0, float(getattr(row, "strategy_mean_cost", 1.0)))
                base_score -= 0.10 / empirical_cost
                cost = empirical_cost
            else:
                cost = path_cost
            rows.append((plan, base_score, cost))
            grouped.setdefault(plan.outcome_uid, []).append(index)

        relative: dict[int, float] = {}
        for members in grouped.values():
            if len(members) < 2:
                continue
            best = min(rows[index][2] for index in members)
            for index in members:
                relative[index] = max(0.0, min(1.0, best / max(1.0, rows[index][2])))

        adjusted = [
            replace(
                plan,
                score=base_score + _RELATIVE_EFFICIENCY_WEIGHT * relative.get(index, 0.0),
            )
            for index, (plan, base_score, _cost) in enumerate(rows)
        ]
        adjusted.sort(key=lambda item: (-item.score, item.action_id, item.strategy_uid))
        return tuple(adjusted)

    blocker_module._composite_plans = composite_plans


def _install_cross_context_probe_fallback() -> None:
    from v8 import behavior_recovery as behavior_module
    from v8 import learning_blockers_v055 as blocker_module
    from v8.publication import LiveReadView

    current_plan_candidates = LiveReadView.plan_candidates

    def plan_candidates(self, context_signature, action_ids, **kwargs):
        plans = tuple(current_plan_candidates(self, context_signature, action_ids, **kwargs))
        required_ancestor = kwargs.get("required_ancestor")

        if required_ancestor is None:
            # Preserve actor epsilon/efficiency-search randomization even when the
            # v8.5 composite planner found a replayable procedure after the base
            # behavior layer deliberately requested random exploration.
            if bool(getattr(self, "_behavior_force_random", False)):
                self._behavior_last_plans = ()
                return ()

            # v8.5 composite planning admitted probationary composites directly.
            # Keep those available for an explicit validation probe, but never let
            # them bypass the normal behavioral control gate.
            by_uid = getattr(self, "_node_by_uid", {})
            admitted = []
            for plan in plans:
                row = by_uid.get(plan.strategy_uid)
                if row is None or not blocker_module.is_composite_strategy(row):
                    admitted.append(plan)
                    continue
                if behavior_module.strategy_can_control(
                    self, plan.strategy_uid, plan.outcome_uid
                ):
                    admitted.append(plan)
            return tuple(admitted)

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
    _install_outcome_conditioned_efficiency()
    _install_cross_context_probe_fallback()
    _INSTALLED = True
