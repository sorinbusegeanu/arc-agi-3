from __future__ import annotations

from dataclasses import replace

from v8.model import stable_u64
from v8.learning_fixes_v088_target_grounding_fix import (
    install_target_local_transfer_grounding_fix,
)
from v8.learning_fixes_v088_target_lineage_grounding_fix import (
    install_target_lineage_grounding_fix,
)
from v8.learning_fixes_v088_target_ephemeral_production_grounding_fix import (
    install_target_ephemeral_production_grounding_fix,
)
from v8.learning_fixes_v088_target_mapping_lifetime_fix import (
    install_target_mapping_lifetime_fix,
)
from v8.learning_fixes_v088_immediate_structural_transfer_fix import (
    install_immediate_structural_transfer_fix,
)
from v8.learning_fixes_v088_transfer_pass_evidence_fix import (
    install_transfer_pass_evidence_fix,
)


_INSTALLED = False
_RELATIVE_EFFICIENCY_WEIGHT = 0.15


def _install_outcome_conditioned_efficiency() -> None:
    """Keep efficiency comparative inside one outcome/context cohort."""
    from v8 import behavior_recovery as behavior_module
    from v8 import learning_blockers_v055 as blocker_module

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
            if bool(getattr(self, "_behavior_force_random", False)):
                self._behavior_last_plans = ()
                return ()

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


def _install_fresh_transfer_experiment_cut() -> None:
    """Refresh the coherent read cut before final held-out transfer discovery."""
    from v8 import learning_fixes_v088 as learning_module

    current_cut = learning_module._coherent_cached_transfer_cut

    def coherent_cached_transfer_cut(view):
        invalidate = getattr(view, "invalidate_strategy_cache", None)
        refresh = getattr(view, "_refresh_strategy_cache", None)
        if callable(invalidate) and callable(refresh):
            invalidate()
            refresh()
        return current_cut(view)

    learning_module._coherent_cached_transfer_cut = coherent_cached_transfer_cut


def install_learning_fixes_v088_fixups() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_outcome_conditioned_efficiency()
    _install_cross_context_probe_fallback()
    _install_fresh_transfer_experiment_cut()
    install_target_local_transfer_grounding_fix()
    install_target_lineage_grounding_fix()
    install_target_ephemeral_production_grounding_fix()
    install_target_mapping_lifetime_fix()
    install_immediate_structural_transfer_fix()
    install_transfer_pass_evidence_fix()
    _INSTALLED = True
