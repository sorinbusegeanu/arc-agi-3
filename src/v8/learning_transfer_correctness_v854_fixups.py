from __future__ import annotations

"""Authority-preserving integration for v8.54 learning/transfer correctness."""

import inspect

_INSTALLED = False
_PRIOR_RESTART_BASE_RESET = None


def _clear_view_state(view) -> None:
    if view is None:
        return
    view._v854_session_transitions = []
    view._v815_session_trajectory = []
    active = getattr(view, "_v854_transfer_active", None)
    if isinstance(active, dict):
        active.clear()


def _restart_base_reset_v854(self, *args, **kwargs):
    try:
        from v8 import behavior_recovery as behavior

        _clear_view_state(getattr(behavior, "_CURRENT_ACTOR_VIEW", None))
    except BaseException:
        pass
    return _PRIOR_RESTART_BASE_RESET(self, *args, **kwargs)


def _contextual_cross_game_transfer(sampler, actions):
    """Use the actual v8.33 caller context without replacing its public hooks."""
    from v8 import behavior_recovery as behavior
    from v8 import learning_transfer_correctness_v854 as v854

    available = {int(value) for value in actions}
    view = getattr(behavior, "_CURRENT_ACTOR_VIEW", None)
    if view is not None and available:
        caller = inspect.currentframe().f_back
        context = None if caller is None else caller.f_locals.get("context")
        if context is not None:
            ordered = v854._ordered_action(
                view,
                str(getattr(sampler, "game_id", "")),
                int(context),
                available,
            )
            if ordered is not None:
                return ordered
    return v854._BASE_CROSS_GAME(sampler, tuple(sorted(available)))


def _locked_record_transfer_trial(self, *args, **kwargs):
    """Keep v8.45 snapshot serialization authority around v8.54 mutation."""
    from v8 import learning_transfer_correctness_v854 as v854

    lock = getattr(self, "_v845_state_lock", None)
    if lock is None:
        return v854._record_trial_v854(self, *args, **kwargs)
    with lock:
        return v854._record_trial_v854(self, *args, **kwargs)


def install_learning_transfer_correctness_v854_fixups() -> None:
    global _INSTALLED, _PRIOR_RESTART_BASE_RESET
    if _INSTALLED:
        return

    from v7.environment.arc_adapter import ArcGridEnvironment
    from v8 import learning_transfer_correctness_v854 as v854
    from v8 import restart_memory_v815 as restart
    from v8 import sampling_transfer_v833 as transfer_sampling
    from v8.peers_v82 import V82DevelopmentalPeerSupervisor

    # Restore historical public sampler identities. Their global transfer helper is
    # dynamic, so ordered v8.54 transfer can compose underneath without changing the
    # v8.33/v8.47 authority chain.
    transfer_sampling._forced_action_v833 = v854._BASE_FORCED
    transfer_sampling._discovery_action_v833 = v854._BASE_DISCOVERY
    transfer_sampling._cross_game_transfer_action = _contextual_cross_game_transfer

    # Restore v8.22 as the public reset authority. v8.15's reset wrapper calls its
    # module-global base dynamically, which is the safe point for v8.54 cleanup.
    ArcGridEnvironment.reset = v854._BASE_RESET
    _PRIOR_RESTART_BASE_RESET = restart._BASE_ENV_RESET
    restart._BASE_ENV_RESET = _restart_base_reset_v854

    # v8.54 replaced the peer method after v8.45 was installed. Reapply the same
    # serialization lock around the corrected relation-scoped mutation.
    V82DevelopmentalPeerSupervisor.record_transfer_trial = _locked_record_transfer_trial

    _INSTALLED = True
