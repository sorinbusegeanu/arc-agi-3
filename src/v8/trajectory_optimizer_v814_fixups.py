from __future__ import annotations


_INSTALLED = False


def install_trajectory_optimizer_v814_fixups() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import trajectory_optimizer_v814 as optimizer
    from v8.publication import LiveReadView, PlannedAction

    def plan_candidates(self, context_signature, action_ids, **kwargs):
        if int(getattr(self, "_v814_reset_epoch", -1)) != int(
            optimizer._ACTOR_RESET_EPOCH
        ):
            self._v814_reset_epoch = int(optimizer._ACTOR_RESET_EPOCH)
            self._v814_active_variant = None
            self._v814_active_actions = ()
            self._v814_attempted_variants = set()

        active = getattr(self, "_v814_active_variant", None)
        remaining = tuple(getattr(self, "_v814_active_actions", ()))
        if active is not None and remaining:
            action = int(remaining[0])
            self._v814_active_actions = remaining[1:]
            plan = PlannedAction(
                action,
                active.target_outcome_uid,
                active.strategy_uid,
                1_000_000.0,
                False,
            )
            self._behavior_last_plans = (plan,)
            return (plan,)

        optimizer._refresh_view_variants(self)
        selected = optimizer.select_validated_variant(
            tuple(getattr(self, "_v814_variants", ())),
            source_id=optimizer._CAPTURE_SOURCE_ID,
            seed=optimizer._CAPTURE_SEED,
            action_history=tuple(optimizer._ACTOR_ACTION_HISTORY),
            attempted=set(getattr(self, "_v814_attempted_variants", set())),
        )
        if selected is not None:
            self._v814_attempted_variants.add(selected.variant_id)
            self._v814_active_variant = selected
            self._v814_active_actions = tuple(selected.actions[1:])
            plan = PlannedAction(
                int(selected.actions[0]),
                selected.target_outcome_uid,
                selected.strategy_uid,
                1_000_000.0,
                False,
            )
            self._behavior_last_plans = (plan,)
            return (plan,)

        return optimizer._BASE_PLAN_CANDIDATES(
            self, context_signature, action_ids, **kwargs
        )

    LiveReadView.plan_candidates = plan_candidates
    _INSTALLED = True
