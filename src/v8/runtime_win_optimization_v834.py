from __future__ import annotations

import json
import os
import time
from pathlib import Path


_INSTALLED = False
_BASE_RESULT_PUT = None
_BASE_GAME_STATE = None
_BASE_PUBLISH_RUNTIME_LEVELS = None
_BASE_PREFIX_FOR = None
_BASE_RUNTIME_INIT = None


def _marker_path(game_id: str) -> Path | None:
    from v8 import trajectory_optimizer_v814 as optimizer

    root = os.environ.get(optimizer._TRAJECTORY_ROOT_ENV)
    if not root:
        return None
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(game_id))
    if not safe:
        return None
    return Path(root) / "runtime_wins" / f"{safe}.json"


def _write_runtime_win_marker(game_id: str, result) -> None:
    if int(getattr(result, "wins", 0)) <= 0:
        return
    path = _marker_path(game_id)
    if path is None:
        return
    from v8 import trajectory_optimizer_v814 as optimizer

    optimizer._atomic_json(
        path,
        {
            "game_id": str(game_id),
            "levels_completed": max(1, int(getattr(result, "levels_completed", 0))),
            "steps": max(1, int(getattr(result, "steps", 0))),
            "time_ns": time.time_ns(),
        },
    )


def _result_adapter_put_v834(self, row) -> None:
    _write_runtime_win_marker(str(self.lease.game_id), row)
    return _BASE_RESULT_PUT(self, row)


def _promote_runtime_win_if_present(coordinator, game_id: str) -> bool:
    from v8 import adaptive_learning_allocation_v819 as v819

    path = _marker_path(game_id)
    if path is None or not path.is_file():
        return False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return False
    game = str(game_id)
    if str(raw.get("game_id", "")) != game:
        return False
    level = max(1, int(raw.get("levels_completed", 1)))
    runtime = getattr(coordinator, "_v834_runtime", None)
    generation = max(1, int(getattr(runtime, "generation", 1)))
    with coordinator._lock:
        coordinator.register_games((game,))
        row = coordinator._record(game, level)
        previous = row.state
        coordinator._game_won[game] = True
        if row.first_success_generation <= 0:
            row.first_success_generation = generation
        row.last_success_generation = generation
        row.last_frontier_improvement_generation = max(
            int(row.last_frontier_improvement_generation), generation
        )
        row.optimizer_exhausted_version = -1
        if row.state == v819.GameLearningState.UNSOLVED:
            row.state = v819.GameLearningState.SOLVED_OPTIMIZING
        if previous != row.state:
            coordinator._emit(
                f"learning state game={game} level={level} "
                f"{previous.value}->{row.state.value} observed_runtime_win=1"
            )
    return True


def _game_state_v834(self, game_id: str):
    from v8 import adaptive_learning_allocation_v819 as v819

    state = _BASE_GAME_STATE(self, game_id)
    if state != v819.GameLearningState.UNSOLVED:
        return state
    if _promote_runtime_win_if_present(self, str(game_id)):
        return v819.GameLearningState.SOLVED_OPTIMIZING
    return state


def _publish_complete_optimizer_source(game_id: str, levels) -> None:
    from v8 import trajectory_optimizer_v814 as optimizer

    try:
        canonical = tuple(tuple(int(value) for value in level) for level in levels)
    except (TypeError, ValueError):
        return
    if not str(game_id) or not canonical or any(not row for row in canonical):
        return
    actions = tuple(value for row in canonical for value in row)
    if not actions:
        return
    anchor = optimizer.ReplayAnchor(
        str(game_id),
        int(getattr(optimizer, "_CAPTURE_SEED", 0)),
        (),
        getattr(optimizer, "_CAPTURE_ENV_ROOT", None),
    )
    target = optimizer.TrajectoryTarget(len(canonical), "WIN")
    row = optimizer.SuccessfulTrajectory(
        optimizer._trajectory_id(anchor, target, actions),
        anchor,
        target,
        actions,
    )
    optimizer._write_successful_trajectory(row)


def _publish_runtime_levels_v834(game_id: str, levels) -> None:
    result = _BASE_PUBLISH_RUNTIME_LEVELS(game_id, levels)
    _publish_complete_optimizer_source(game_id, levels)
    return result


def _prefix_for_v834(service, candidate) -> tuple[int, ...]:
    source = candidate.source
    if (
        str(getattr(source.target, "terminal_state", "")) == "WIN"
        and not tuple(getattr(source.anchor, "prefix_actions", ()))
    ):
        return ()
    return _BASE_PREFIX_FOR(service, candidate)


def _runtime_init_v834(self, *args, **kwargs) -> None:
    _BASE_RUNTIME_INIT(self, *args, **kwargs)
    coordinator = getattr(self, "_v819_adaptive_learning", None)
    if coordinator is not None:
        coordinator._v834_runtime = self


def install_runtime_win_optimization_v834() -> None:
    global _INSTALLED
    global _BASE_RESULT_PUT, _BASE_GAME_STATE, _BASE_PUBLISH_RUNTIME_LEVELS
    global _BASE_PREFIX_FOR, _BASE_RUNTIME_INIT
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import solved_game_recovery_v821 as recovery
    from v8 import trajectory_optimizer_v818 as v818
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    _BASE_RESULT_PUT = v819._ResultAdapter.put
    _BASE_GAME_STATE = v819.AdaptiveLearningCoordinator.game_state
    _BASE_PUBLISH_RUNTIME_LEVELS = recovery._publish_runtime_levels
    _BASE_PREFIX_FOR = v818._prefix_for
    _BASE_RUNTIME_INIT = V82ContinuousMemoryRuntime.__init__

    v819._ResultAdapter.put = _result_adapter_put_v834
    v819.AdaptiveLearningCoordinator.game_state = _game_state_v834
    recovery._publish_runtime_levels = _publish_runtime_levels_v834
    v818._prefix_for = _prefix_for_v834
    V82ContinuousMemoryRuntime.__init__ = _runtime_init_v834
    _INSTALLED = True
