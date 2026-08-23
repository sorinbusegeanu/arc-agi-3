from __future__ import annotations

"""v8.27 lifecycle authority across canonical and auxiliary competence stores.

Canonical M6/M7 lifecycle state is the behavioral authority. Optimizer sidecars and
adaptive frontier rows remain durable evidence, but may not silently bypass
QUARANTINED/RETIRE_PENDING/RETIRED canonical memory.
"""

import json
import os
import time
from pathlib import Path


_INSTALLED = False
_BASE_REFRESH_VIEW_VARIANTS = None
_BASE_GAME_STATE = None
_BASE_CHOOSE_MODE = None
_BASE_ALTERNATIVE_EXCLUSION = None
_BASE_VISIBLE_SOLUTION = None
_BASE_SHOW_BEST = None
_BASE_WRITE_COMPLETE_OBSERVED_SOLUTION = None


def _lifecycle_class(row) -> str:
    from v8.model import CognitiveState

    if row is None:
        return "BLOCKED"
    state = int(getattr(row, "cognitive_state", int(CognitiveState.RETIRED)))
    if state in {
        int(CognitiveState.ACTIVE),
        int(CognitiveState.VALIDATED),
        int(CognitiveState.REACTIVATED),
        int(CognitiveState.CANDIDATE),
        int(CognitiveState.PROBATION),
    }:
        return "ACTIVE"
    if state == int(CognitiveState.QUARANTINED):
        return "QUARANTINED"
    return "BLOCKED"


def _uid_class(index, uid) -> str:
    if uid is None or bool(getattr(uid, "is_zero", True)):
        # Autonomous sidecar/frontier control requires a canonical identity.
        return "BLOCKED"
    return _lifecycle_class(index.get(uid))


def _pair_class(index, strategy_uid, outcome_uid) -> str:
    values = (_uid_class(index, strategy_uid), _uid_class(index, outcome_uid))
    if "BLOCKED" in values:
        return "BLOCKED"
    if "QUARANTINED" in values:
        return "QUARANTINED"
    return "ACTIVE"


def _lifecycle_index(read_view):
    if read_view is None:
        return None
    try:
        read_view._refresh_strategy_cache()
        return getattr(read_view, "_node_by_uid", None)
    except BaseException:
        return None


def _frontier_lifecycle_class(coordinator, game_id: str) -> str:
    view = getattr(coordinator, "_v827_read_view", None)
    index = _lifecycle_index(view)
    return _frontier_lifecycle_class_from_index(coordinator, game_id, index)


def _cached_frontier_lifecycle_class(coordinator, game_id: str) -> str:
    """Classify a frontier from the last published graph index without rebuilding it."""

    if not bool(getattr(coordinator, "_v827_lifecycle_authority", True)):
        return "UNKNOWN"
    view = getattr(coordinator, "_v827_read_view", None)
    index = None if view is None else getattr(view, "_node_by_uid", None)
    return _frontier_lifecycle_class_from_index(coordinator, game_id, index)


def _frontier_lifecycle_class_from_index(coordinator, game_id: str, index) -> str:
    if index is None:
        return "UNKNOWN"

    game = str(game_id)
    saw_quarantined = False
    with coordinator._lock:
        scopes = tuple(coordinator.frontier.scopes())
        for scope in scopes:
            if str(scope.game_id) != game:
                continue
            for candidate in coordinator.frontier.candidates(scope):
                state = _pair_class(index, candidate.strategy_uid, scope.outcome_uid)
                if state == "ACTIVE":
                    return "ACTIVE"
                if state == "QUARANTINED":
                    saw_quarantined = True
    return "QUARANTINED" if saw_quarantined else "BLOCKED"


def _best_active_frontier_uid(coordinator, game_id: str):
    from v8.model import MemoryUid

    index = _lifecycle_index(getattr(coordinator, "_v827_read_view", None))
    if index is None:
        return None
    choices = []
    with coordinator._lock:
        for scope in coordinator.frontier.scopes():
            if str(scope.game_id) != str(game_id):
                continue
            for candidate in coordinator.frontier.candidates(scope):
                if _pair_class(index, candidate.strategy_uid, scope.outcome_uid) != "ACTIVE":
                    continue
                choices.append((scope, candidate))
    if not choices:
        return MemoryUid.zero()
    _scope, winner = min(
        choices,
        key=lambda item: (
            -int(item[0].level),
            int(item[1].cost),
            -float(item[1].reliability),
            item[1].trajectory_id,
        ),
    )
    return winner.strategy_uid


def _refresh_view_variants_v827(self) -> None:
    from v8 import adaptive_learning_allocation_v819 as v819

    mode = str(os.environ.get(v819._SAMPLING_MODE_ENV, v819.SamplingMode.DISCOVERY.value))
    prior_mode = getattr(self, "_v827_variant_lifecycle_mode", None)
    if prior_mode != mode:
        # A DISCOVERY filter must not hide a quarantined row from an explicit VERIFY
        # call until the base sidecar cache happens to refresh on its one-second timer.
        self._v814_next_refresh = 0.0
        self._v827_variant_lifecycle_mode = mode

    _BASE_REFRESH_VIEW_VARIANTS(self)
    index = _lifecycle_index(self)
    if index is None:
        self._v814_variants = ()
        return

    allow_quarantined = mode == v819.SamplingMode.VERIFY.value
    kept = []
    for row in tuple(getattr(self, "_v814_variants", ())):
        state = _pair_class(index, row.strategy_uid, row.target_outcome_uid)
        if state == "ACTIVE" or (allow_quarantined and state == "QUARANTINED"):
            kept.append(row)
    self._v814_variants = tuple(kept)


def _game_state_v827(self, game_id: str):
    from v8 import adaptive_learning_allocation_v819 as v819

    state = _BASE_GAME_STATE(self, game_id)
    if state == v819.GameLearningState.UNSOLVED:
        return state
    lifecycle = _frontier_lifecycle_class(self, str(game_id))
    if lifecycle in {"UNKNOWN", "ACTIVE"}:
        return state
    if lifecycle == "QUARANTINED":
        # Keep a quarantined solved game in explicit verification, never STABLE.
        return v819.GameLearningState.SOLVED_OPTIMIZING
    # RETIRE_PENDING, RETIRED, missing/compacted identity: competence is no longer
    # behaviorally available, so sampling must rediscover/rebuild it.
    return v819.GameLearningState.UNSOLVED


def _choose_mode_v827(self, game_id: str):
    from v8 import adaptive_learning_allocation_v819 as v819

    if _frontier_lifecycle_class(self, str(game_id)) == "QUARANTINED":
        return v819.SamplingMode.VERIFY
    return _BASE_CHOOSE_MODE(self, game_id)


def _alternative_exclusion_v827(self, game_id: str):
    uid = _best_active_frontier_uid(self, str(game_id))
    if uid is None:
        return _BASE_ALTERNATIVE_EXCLUSION(self, game_id)
    return uid


def _candidate_complete_solution_before_write(row):
    """Mirror v8.19 chain validation before the base writer resets its state."""
    from v8 import trajectory_inspection_v819 as inspection

    game_id = str(row.anchor.source_id)
    prefix = tuple(int(value) for value in row.anchor.prefix_actions)
    actions = tuple(int(value) for value in row.actions)
    terminal_state = str(row.target.terminal_state)
    levels_completed = int(row.target.levels_completed)

    levels = list(inspection._OBSERVED_LEVELS)
    chain_valid = bool(inspection._OBSERVED_CHAIN_VALID)
    first_level = not prefix and levels_completed <= 1
    if first_level or game_id != str(inspection._OBSERVED_GAME_ID):
        levels = []
        chain_valid = bool(first_level)
    if not chain_valid or prefix != inspection._flatten_levels(levels):
        return None
    levels.append(actions)
    if terminal_state != "WIN":
        return None
    return inspection._observed_solution(row, tuple(levels))


def _write_complete_observed_solution_v827(row) -> None:
    solution = _candidate_complete_solution_before_write(row)
    _BASE_WRITE_COMPLETE_OBSERVED_SOLUTION(row)
    if solution is None:
        return
    root_raw = os.environ.get("ARC_AGI3_V8_TRAJECTORY_ROOT")
    if not root_raw:
        return

    from v8 import trajectory_optimizer_v814 as optimizer

    history = Path(root_raw) / "solutions_history"
    history.mkdir(parents=True, exist_ok=True)
    game = str(solution.get("game_id", "game"))
    trajectory = str(solution.get("trajectory_id", "trajectory"))
    target = history / f"{game}-{trajectory}-{os.getpid()}-{time.time_ns()}.json"
    optimizer._atomic_json(target, solution)


def _best_from_history(root: str | Path, game_id: str, best):
    from v8 import trajectory_inspection_v819 as inspection

    history = Path(root) / "trajectory_optimizer" / "solutions_history"
    try:
        paths = sorted(history.glob("*.json"))
    except OSError:
        paths = []
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        record = inspection._validated_solution_record(raw)
        if record is None or str(record.get("game_id", "")) != str(game_id):
            continue
        if inspection._is_better_solution(record, best):
            best = record
    return best


def _best_visible_solution_v827(root: str | Path, game_id: str):
    best = _BASE_VISIBLE_SOLUTION(root, game_id)
    return _best_from_history(root, str(game_id), best)


def _all_best_visible_solutions(root: str | Path) -> dict[str, dict[str, object]]:
    """Scan each durable trajectory source once and select one best row per game."""

    from v8 import trajectory_inspection_v819 as inspection
    from v8 import trajectory_inspection_v819_fixups as visibility
    from v8 import trajectory_optimizer_v814 as optimizer

    optimizer_root = Path(root) / "trajectory_optimizer"
    best = dict(
        inspection._load_best_successful(optimizer_root / "best_successful.json")
    )

    for directory in ("solutions_history", "solutions_inbox"):
        try:
            paths = tuple((optimizer_root / directory).glob("*.json"))
        except OSError:
            paths = ()
        for path in paths:
            try:
                record = inspection._validated_solution_record(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError, TypeError):
                record = None
            if record is None:
                continue
            game = str(record["game_id"])
            if inspection._is_better_solution(record, best.get(game)):
                best[game] = record

    rows_by_game: dict[str, list[object]] = {}
    try:
        optimizer_paths = tuple((optimizer_root / "inbox").glob("*.json"))
    except OSError:
        optimizer_paths = ()
    for path in optimizer_paths:
        try:
            row = optimizer.SuccessfulTrajectory.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, TypeError, KeyError):
            continue
        rows_by_game.setdefault(str(row.anchor.source_id), []).append(row)
    for game, rows in rows_by_game.items():
        for row in rows:
            if str(row.target.terminal_state) != "WIN":
                continue
            record = visibility._record_from_nested_optimizer_rows(rows, row)
            if record is not None and inspection._is_better_solution(
                record, best.get(game)
            ):
                best[game] = record
    return best


def _available_solution_games(root: str | Path) -> tuple[str, ...]:
    from v8 import trajectory_inspection_v819 as inspection
    from v8 import trajectory_optimizer_v814 as optimizer

    optimizer_root = Path(root) / "trajectory_optimizer"
    games = set(inspection._load_best_successful(optimizer_root / "best_successful.json"))

    for directory in ("solutions_history", "solutions_inbox"):
        try:
            paths = tuple((optimizer_root / directory).glob("*.json"))
        except OSError:
            paths = ()
        for path in paths:
            try:
                record = inspection._validated_solution_record(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError, TypeError):
                record = None
            if record is not None:
                games.add(str(record["game_id"]))

    for path in (optimizer_root / "validated.json",):
        for row in optimizer._load_validated_rows(path):
            if str(row.target.terminal_state) == "WIN":
                games.add(str(row.anchor.source_id))
    return tuple(sorted(game for game in games if game))


def _show_best_trajectory_v827(root: str | Path, game_id: str) -> int:
    from v8 import trajectory_inspection_v819 as inspection
    from v8.action_targeting_v810 import native_action_id

    game = str(game_id)
    record = _best_visible_solution_v827(root, game)
    if record is None:
        available = _available_solution_games(root)
        suffix = "" if not available else "; available=" + ",".join(available)
        print(f"game={game} no successful trajectory found{suffix}", flush=True)
        return 1

    reliability = float(record.get("reliability", 0.0))
    print(
        f"game={game} cost={int(record['total_cost'])} "
        f"source={record['source']} reliability={reliability:.3f}",
        flush=True,
    )
    levels = inspection._normalize_levels(record.get("levels")) or ()
    for index, actions in enumerate(levels):
        formatted = ",".join(f"A{int(native_action_id(action))}" for action in actions)
        print(f"L{index}: {formatted}", flush=True)
    return 0


def _save_best_trajectories_v827(root: str | Path, output_path: str | Path) -> int:
    from v8 import trajectory_inspection_v819 as inspection

    best = _all_best_visible_solutions(root)
    records = tuple((game, best[game]) for game in sorted(best))
    return inspection._save_best_trajectory_records(output_path, records)


def _format_game_rate_line_v827(rows) -> str:
    from v8 import diagnostics

    rows = tuple(rows)
    win_rate, level_rate, solved_games, games = diagnostics.game_summary(rows)
    grouped = diagnostics._group_games(rows)
    details = []
    for game_id, lane_rows in sorted(grouped.items()):
        solved_rows = [row for row in lane_rows if int(getattr(row, "wins", 0)) > 0]
        if not solved_rows:
            continue
        best_values = [int(getattr(row, "best_win_steps", 0) or 0) for row in solved_rows]
        best = min((value for value in best_values if value > 0), default=0)
        # Actor-local counters cannot establish a truthful "last" across concurrent
        # lanes. Normal adaptive runs have one lane per game, where L is exact.
        last = 0
        if len(solved_rows) == 1:
            last = int(getattr(solved_rows[0], "last_win_steps", 0) or 0)
        if best > 0 and last > 0:
            details.append(f"{game_id}:B={best},L={last}")
        elif best > 0:
            details.append(f"{game_id}:B={best}")
        else:
            details.append(f"{game_id}:win_observed")
    suffix = "" if not details else " (" + "; ".join(details) + ")"
    return (
        f"current_run_wins={win_rate:.1f}% current_run_levels_solved={level_rate:.1f}% "
        f"current_run_solved_games={solved_games}/{games}{suffix}"
    )


def install_lifecycle_competence_integration_v827() -> None:
    global _INSTALLED
    global _BASE_REFRESH_VIEW_VARIANTS, _BASE_GAME_STATE, _BASE_CHOOSE_MODE
    global _BASE_ALTERNATIVE_EXCLUSION, _BASE_VISIBLE_SOLUTION, _BASE_SHOW_BEST
    global _BASE_WRITE_COMPLETE_OBSERVED_SOLUTION
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import diagnostics
    from v8 import trajectory_inspection_v819 as inspection
    from v8 import trajectory_inspection_v819_fixups as visibility
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    _BASE_REFRESH_VIEW_VARIANTS = optimizer._refresh_view_variants
    optimizer._refresh_view_variants = _refresh_view_variants_v827

    _BASE_GAME_STATE = v819.AdaptiveLearningCoordinator.game_state
    _BASE_CHOOSE_MODE = v819.AdaptiveLearningCoordinator.choose_mode
    _BASE_ALTERNATIVE_EXCLUSION = v819.AdaptiveLearningCoordinator.alternative_exclusion
    v819.AdaptiveLearningCoordinator.game_state = _game_state_v827
    v819.AdaptiveLearningCoordinator.choose_mode = _choose_mode_v827
    v819.AdaptiveLearningCoordinator.alternative_exclusion = _alternative_exclusion_v827

    # Bind lifecycle reads to the same coherent graph view used by the runtime.
    base_runtime_init = V82ContinuousMemoryRuntime.__init__

    def runtime_init(self, *args, **kwargs):
        base_runtime_init(self, *args, **kwargs)
        coordinator = getattr(self, "_v819_adaptive_learning", None)
        if coordinator is not None:
            coordinator._v827_read_view = self.read_view

    V82ContinuousMemoryRuntime.__init__ = runtime_init

    _BASE_WRITE_COMPLETE_OBSERVED_SOLUTION = inspection._write_complete_observed_solution
    inspection._write_complete_observed_solution = _write_complete_observed_solution_v827

    _BASE_VISIBLE_SOLUTION = visibility._best_visible_solution
    visibility._best_visible_solution = _best_visible_solution_v827
    _BASE_SHOW_BEST = inspection.show_best_trajectory
    inspection.show_best_trajectory = _show_best_trajectory_v827
    inspection.save_best_trajectories = _save_best_trajectories_v827

    diagnostics.format_game_rate_line = _format_game_rate_line_v827
    try:
        from v8 import reporter

        reporter.format_game_rate_line = _format_game_rate_line_v827
    except BaseException:
        pass

    V82ContinuousMemoryRuntime.scientific_semantics_version = (
        "v8.27-lifecycle-competence-integration"
    )
    _INSTALLED = True
