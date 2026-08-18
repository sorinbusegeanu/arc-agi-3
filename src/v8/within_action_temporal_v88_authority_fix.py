from __future__ import annotations

"""Place v8.8 temporal capture below the historical v8.29/v8.21/v8.22 hook chain."""

_INSTALLED = False


def install_within_action_temporal_v88_authority_fix() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v7.environment.arc_adapter import ArcGridEnvironment
    from v8 import runtime_repair_v822 as v822
    from v8 import sampling_progress_control_v829 as repair
    from v8 import solved_game_recovery_v821 as recovery
    from v8 import within_action_temporal_v88 as temporal

    # Preserve the exact historical public chain:
    # ArcGridEnvironment.step -> v8.22 -> v8.21 -> v8.29 -> v8.8 -> earlier adapter.
    lower_step = repair._BASE_ENV_STEP
    temporal._BASE_ARC_STEP = lower_step
    repair._BASE_ENV_STEP = temporal._adapter_step_v88
    recovery._BASE_ENV_STEP = repair._env_step_v829
    v822._BASE_ENV_STEP = recovery._tracked_env_step
    ArcGridEnvironment.step = v822._runtime_env_step

    lower_reset = repair._BASE_ENV_RESET
    temporal._BASE_ARC_RESET = lower_reset
    repair._BASE_ENV_RESET = temporal._adapter_reset_v88
    recovery._BASE_ENV_RESET = repair._env_reset_v829
    v822._BASE_ENV_RESET = recovery._tracked_env_reset
    ArcGridEnvironment.reset = v822._runtime_env_reset

    _INSTALLED = True
