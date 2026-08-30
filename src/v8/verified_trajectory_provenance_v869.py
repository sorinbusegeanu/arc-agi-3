from __future__ import annotations

"""v8.69 preserve generic episode provenance in verified trajectories."""


_INSTALLED = False
_BASE_VERIFIED_PROXY_STEP = None
_BASE_VERIFIED_EXPORT_RECORD = None
_BASE_FORMAT_BEST_TRAJECTORY_LINES = None


def _gym_episode_seed(self) -> int:
    return int(self.seed) + max(0, int(self._episode) - 1)


def _chess_episode_seed(self) -> int:
    return int(self.seed) + max(0, int(self._episode) - 1)


def _sudoku_episode_seed(self) -> int:
    return int(self.seed) + max(0, int(self._episode) - 1) * 104729


def _active_episode_seed(inner, fallback: int) -> int:
    getter = getattr(inner, "cognitive_episode_seed", None)
    if callable(getter):
        try:
            return int(getter())
        except (TypeError, ValueError):
            pass
    return int(fallback)


def _verified_proxy_step_v869(self, action):
    from v8 import verified_success_metrics_v866 as verified

    self._steps += 1
    self._actions.append(int(action))
    result = self._inner.step(action)
    boundary = self._inner.cognitive_boundary_event()
    if not bool(boundary.continuation) and int(boundary.primary_valence) > 0:
        verified.record_verified_success_v866(
            game_id=self._game_id,
            seed=_active_episode_seed(self._inner, self._seed),
            terminal_state="WIN",
            levels_completed=1,
            actions=tuple(self._actions),
            capture_step=self._steps,
        )
    return result


def _verified_export_record_v869(row: dict[str, object]) -> dict[str, object]:
    record = dict(_BASE_VERIFIED_EXPORT_RECORD(row))
    try:
        record["seed"] = int(row["seed"])
    except (KeyError, TypeError, ValueError):
        pass
    return record


def _format_best_trajectory_lines_v869(
    game_id: str,
    record: dict[str, object],
) -> tuple[str, ...]:
    lines = list(_BASE_FORMAT_BEST_TRAJECTORY_LINES(game_id, record))
    if lines and "seed" in record:
        try:
            lines[0] = f"{lines[0]} seed={int(record['seed'])}"
        except (TypeError, ValueError):
            pass
    return tuple(lines)


def install_verified_trajectory_provenance_v869() -> None:
    global _INSTALLED
    global _BASE_VERIFIED_PROXY_STEP, _BASE_VERIFIED_EXPORT_RECORD
    global _BASE_FORMAT_BEST_TRAJECTORY_LINES
    if _INSTALLED:
        return

    from v8.environments.chess_env import ChessAdapter
    from v8.environments.gym_adapter import GymDiscreteAdapter
    from v8.environments.sudoku_env import SudokuAdapter
    from v8 import trajectory_inspection_v819 as inspection
    from v8 import verified_success_metrics_v866 as verified
    from v8 import verified_trajectory_export_v868 as export

    GymDiscreteAdapter.cognitive_episode_seed = _gym_episode_seed
    ChessAdapter.cognitive_episode_seed = _chess_episode_seed
    SudokuAdapter.cognitive_episode_seed = _sudoku_episode_seed

    _BASE_VERIFIED_PROXY_STEP = verified._VerifiedAdapterProxy.step
    verified._VerifiedAdapterProxy.step = _verified_proxy_step_v869

    _BASE_VERIFIED_EXPORT_RECORD = export._verified_export_record
    export._verified_export_record = _verified_export_record_v869

    _BASE_FORMAT_BEST_TRAJECTORY_LINES = inspection._format_best_trajectory_lines
    inspection._format_best_trajectory_lines = _format_best_trajectory_lines_v869

    _INSTALLED = True
