from __future__ import annotations

"""Authority-preserving composition fixups for v8.56 click trajectory audit."""


_INSTALLED = False


def _show_best_trajectory_v856(root, game_id: str) -> int:
    from v8 import lifecycle_competence_integration_v827 as lifecycle
    from v8 import trajectory_click_audit_v856 as audit

    game = str(game_id)
    record = lifecycle._best_visible_solution_v827(root, game)
    if record is None:
        available = lifecycle._available_solution_games(root)
        suffix = "" if not available else "; available=" + ",".join(available)
        print(f"game={game} no successful trajectory found{suffix}", flush=True)
        return 1
    for line in audit._format_best_trajectory_lines_v856(game, record):
        print(line, flush=True)
    return 0


def install_trajectory_click_audit_v856_fixups() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import runtime_repair_v822 as repair
    from v8 import solved_game_recovery_v821 as recovery
    from v8 import trajectory_click_audit_v856 as audit
    from v8 import trajectory_inspection_v819 as inspection

    # v8.56 initially wrapped the v8.22 delegate. Restore the exact historical
    # authority chain required by v8.22, then instrument one layer deeper:
    #
    # ArcGridEnvironment.step
    #   -> v8.22 _runtime_env_step
    #   -> v8.21 _tracked_env_step
    #   -> v8.56 audit
    #   -> pre-v8.21 execution chain
    #
    # No published or historically asserted function identity changes.
    inner_step = recovery._BASE_ENV_STEP
    inner_reset = recovery._BASE_ENV_RESET

    repair._BASE_ENV_STEP = recovery._tracked_env_step
    repair._BASE_ENV_RESET = recovery._tracked_env_reset

    audit._BASE_ENV_STEP = inner_step
    audit._BASE_ENV_RESET = inner_reset
    recovery._BASE_ENV_STEP = audit._env_step_v856
    recovery._BASE_ENV_RESET = audit._env_reset_v856

    # Read through v8.27's durable visibility authority so solutions_history keeps
    # working after inbox files are consumed.
    inspection.show_best_trajectory = _show_best_trajectory_v856
    _INSTALLED = True
