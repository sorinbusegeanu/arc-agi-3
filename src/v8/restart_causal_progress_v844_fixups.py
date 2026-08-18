from __future__ import annotations

"""Compatibility wiring for v8.44 beneath established final authorities."""

_INSTALLED = False
_BASE_V834_GAME_STATE = None
_BASE_V843_CHOOSE_MODE = None


def _game_state_lower_v844(self, game_id: str):
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import restart_causal_progress_v844 as v844

    state = _BASE_V834_GAME_STATE(self, game_id)
    if state != v819.GameLearningState.UNSOLVED:
        return state
    game = str(game_id)
    if game in set(getattr(self, "_v844_durable_complete_games", set())):
        if v844._durable_missing_identity(self, game):
            return v819.GameLearningState.SOLVED_OPTIMIZING
    return state


def _choose_mode_lower_v844(self, game_id: str):
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import restart_causal_progress_v844 as v844

    if v844._durable_missing_identity(self, str(game_id)):
        return v819.SamplingMode.VERIFY
    return _BASE_V843_CHOOSE_MODE(self, game_id)


def install_restart_causal_progress_v844_fixups() -> None:
    global _INSTALLED, _BASE_V834_GAME_STATE, _BASE_V843_CHOOSE_MODE
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import lease_dispatch_lifecycle_v843 as v843
    from v8 import restart_causal_progress_v844 as v844
    from v8 import runtime_win_optimization_v834 as v834
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    # v8.18/v8.37 already own seedless, environment-neutral trajectory identity
    # and seedless activation. Restore those final semantics instead of duplicating
    # them in v8.44.
    optimizer._anchor_hash = v844._BASE_ANCHOR_HASH
    optimizer.select_validated_variant = v844._BASE_SELECT_VALIDATED

    # Keep v8.34 as the public game-state authority. Durable sidecar recovery is a
    # lower source of solved state, so runtime-win markers remain a fallback rather
    # than a restart dependency.
    _BASE_V834_GAME_STATE = v834._BASE_GAME_STATE
    v834._BASE_GAME_STATE = _game_state_lower_v844
    v819.AdaptiveLearningCoordinator.game_state = v834._game_state_v834

    # Keep v8.43 as the public choose-mode authority. Once the reconciler has made
    # the raw state solved, v8.43 delegates here and a missing canonical identity is
    # routed to explicit VERIFY replay.
    _BASE_V843_CHOOSE_MODE = v843._BASE_CHOOSE_MODE
    v843._BASE_CHOOSE_MODE = _choose_mode_lower_v844
    v819.AdaptiveLearningCoordinator.choose_mode = v843._choose_mode_v843

    # scientific_semantics_version is the paper/schema capability marker, not a
    # chronological runtime patch number.
    V82ContinuousMemoryRuntime.scientific_semantics_version = "v8.5-learning-capability"

    _INSTALLED = True
