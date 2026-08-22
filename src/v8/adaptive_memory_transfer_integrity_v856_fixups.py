from __future__ import annotations

"""Authority-preserving fixups for v8.56 M7 credit and transfer integrity."""

_INSTALLED = False
_BASE_OBSERVED = None


def _publish_plans_safe(view, plans) -> tuple:
    """Publish only to mutable actor views; historical lightweight callers stay valid."""
    rows = tuple(plans)
    if view is not None:
        try:
            view._behavior_last_plans = rows
        except (AttributeError, TypeError):
            pass
    return rows


def _observed_outcomes_clear_v856(**kwargs):
    """Clear the just-consumed plan after all current-step credit observers run."""
    try:
        return _BASE_OBSERVED(**kwargs)
    finally:
        try:
            from v8 import behavior_recovery as behavior

            view = getattr(behavior, "_CURRENT_ACTOR_VIEW", None)
            if view is not None:
                view._behavior_last_plans = ()
        except (AttributeError, ImportError, TypeError):
            pass


def install_adaptive_memory_transfer_integrity_v856_fixups() -> None:
    global _INSTALLED, _BASE_OBSERVED
    if _INSTALLED:
        return

    from v8 import adaptive_memory_transfer_integrity_v856 as v856
    from v8 import actor as actor_module
    from v8 import click_exploration_v848 as v848
    from v8 import sampling_persistence_v832 as persistence

    # Direct historical v8.29 hook tests may pass immutable plain objects.
    v856._publish_plans = _publish_plans_safe

    # Do not replace v8.32/v8.48's established forced-action authority chain merely
    # to clear credit state. The current action's plan is still needed by experience
    # and outcome observers, so clear it only after observation has consumed it.
    persistence._BASE_FORCED_ACTION = v848._sampler_forced_action_v848

    _BASE_OBSERVED = actor_module._observed_outcome_uids
    actor_module._observed_outcome_uids = _observed_outcomes_clear_v856

    _INSTALLED = True
