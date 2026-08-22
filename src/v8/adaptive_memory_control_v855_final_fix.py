from __future__ import annotations

"""Final authority fix for v8.55 adaptive M7 arbitration."""

_INSTALLED = False
_BASE_ADAPTIVE_PLAN = None
_BASE_ARM_COMPOSITE = None


def _adaptive_view_capable(view) -> bool:
    """Identify a real adaptive actor view without constraining historical hook tests."""
    return bool(
        hasattr(view, "_behavior_rng")
        and hasattr(view, "_node_by_uid")
        and callable(getattr(view, "source_games", None))
    )


def _plan_chain_v855_compat(self, context_signature, action_ids, **kwargs):
    """Keep the historical v8.31 standalone hook contract for lightweight callers."""
    from v8 import adaptive_memory_control_v855 as v855

    if not _adaptive_view_capable(self):
        return v855._BASE_PLAN_CHAIN(self, context_signature, action_ids, **kwargs)
    return _BASE_ADAPTIVE_PLAN(self, context_signature, action_ids, **kwargs)


def _arm_composite_safe(self, context_signature, plan) -> None:
    """Ordinary M7 test doubles and legacy rows are never assumed to be composite."""
    try:
        return _BASE_ARM_COMPOSITE(self, context_signature, plan)
    except (AttributeError, TypeError, ValueError):
        return None


def _plan_chain_v855_final(self, context_signature, action_ids, **kwargs):
    """Run adaptive M7 exactly once per eligible DISCOVERY decision."""
    from v8 import adaptive_memory_control_v855_fixups as fixups
    from v8 import sampling_portfolio_v831 as portfolio
    from v8 import sampling_progress_control_v829 as v829

    if not _adaptive_view_capable(self):
        return fixups._BASE_V829_PLAN(self, context_signature, action_ids, **kwargs)

    if not v829._discovery_mode():
        return fixups._BASE_V829_PLAN(self, context_signature, action_ids, **kwargs)

    available = tuple(sorted({int(value) for value in action_ids}))
    state = v829._state_key(context_signature)
    if state is None or not available:
        return fixups._BASE_V829_PLAN(self, context_signature, action_ids, **kwargs)

    progress_action = v829._PROGRESS_ACTION.get(state)
    if progress_action in available:
        return fixups._BASE_V829_PLAN(self, context_signature, available, **kwargs)

    portfolio_mode = str(getattr(portfolio._PORTFOLIO_STATE, "mode", "MEMORY"))
    if portfolio_mode == "RANDOM":
        return fixups._BASE_V829_PLAN(self, context_signature, available, **kwargs)

    # The adaptive layer owns the M7 decision. If it declines, exploration owns
    # the action; do not call the historical planner a second time and defeat the
    # exploration floor or reapply generic per-action stagnation to M7.
    v829._set_selection(context_signature, "DISCOVERY")
    return tuple(
        fixups._plan_chain_v855_fixup(
            self,
            context_signature,
            available,
            **kwargs,
        )
    )


def install_adaptive_memory_control_v855_final_fix() -> None:
    global _INSTALLED, _BASE_ADAPTIVE_PLAN, _BASE_ARM_COMPOSITE
    if _INSTALLED:
        return

    from v8 import adaptive_memory_control_v855 as v855
    from v8 import adaptive_memory_control_v855_fixups as fixups
    from v8 import learning_performance_repair_v824 as v824
    from v8 import sampling_portfolio_v831 as portfolio
    from v8 import sampling_progress_control_v829 as v829

    _BASE_ADAPTIVE_PLAN = v855._plan_chain_v855
    _BASE_ARM_COMPOSITE = fixups._arm_composite

    # Historical direct-hook tests and inspection helpers may pass lightweight
    # objects. Preserve v8.31 semantics for those callers while production actor
    # views use the adaptive policy.
    v855._plan_chain_v855 = _plan_chain_v855_compat
    portfolio._plan_chain_v831 = v855._plan_chain_v855
    v829._BASE_PLAN_CHAIN = portfolio._plan_chain_v831

    # Make composite arming tolerant of ordinary/legacy M7 rows that do not carry
    # the full current NodeRecord shape.
    fixups._arm_composite = _arm_composite_safe

    v829._plan_chain_v829 = _plan_chain_v855_final
    v824._BASE_PLAN_CHAIN = v829._plan_chain_v829
    _INSTALLED = True
