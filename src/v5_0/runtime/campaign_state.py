from __future__ import annotations

from v5_0.contracts.avatar_types import CampaignLevelState, SavedLevelTrace
from v5_0.memory.trace_store import (
    get_best_trace_for_level,
    get_best_verified_trace_prefix,
    get_solved_levels_for_game,
    get_verified_solved_levels_for_game,
)


def load_or_initialize_campaign_state(
    *,
    game_id: str,
    level_sequence: tuple[str, ...] | list[str],
    trace_db_path: str | None = None,
) -> dict[str, CampaignLevelState]:
    ordered = tuple(str(item) for item in level_sequence)
    state = {
        level_id: CampaignLevelState(
            game_id=game_id,
            level_id=level_id,
            status="unknown",
            solved=False,
            solution_trace_path=None,
            best_step_count=None,
            attempt_count=0,
        )
        for level_id in ordered
    }
    try:
        solved_levels = set(get_solved_levels_for_game(db_path=trace_db_path, game_id=game_id))
        for level_id in ordered:
            if level_id in solved_levels:
                best = get_best_trace_for_level(db_path=trace_db_path, game_id=game_id, level_id=level_id)
                if best is not None:
                    state[level_id] = CampaignLevelState(
                        game_id=game_id,
                        level_id=level_id,
                        status="solved",
                        solved=True,
                        solution_trace_path=None,
                        best_step_count=int(best.step_count),
                        attempt_count=state[level_id].attempt_count,
                    )
    except Exception:
        pass
    return state


def update_campaign_state_after_level(
    *,
    state: dict[str, CampaignLevelState],
    level_id: str,
    solved: bool,
    trace_path: str | None,
    step_count: int | None,
) -> dict[str, CampaignLevelState]:
    current = state[str(level_id)]
    next_status = "solved" if solved else "failed"
    updated = CampaignLevelState(
        game_id=current.game_id,
        level_id=current.level_id,
        status=next_status,
        solved=bool(solved),
        solution_trace_path=(str(trace_path) if trace_path else current.solution_trace_path),
        best_step_count=(int(step_count) if solved and step_count is not None else current.best_step_count),
        attempt_count=int(current.attempt_count) + 1,
    )
    output = dict(state)
    output[str(level_id)] = updated
    return output


def get_frontier_level_id(
    *,
    state: dict[str, CampaignLevelState],
    level_sequence: tuple[str, ...] | list[str],
    trace_db_path: str | None = None,
    game_id: str | None = None,
    use_solutions: bool = False,
) -> str | None:
    ordered = tuple(str(item) for item in level_sequence)
    if not bool(use_solutions):
        for level_id in ordered:
            item = state.get(level_id)
            solved_in_current_run = (
                item is not None
                and bool(item.solved)
                and int(getattr(item, "attempt_count", 0) or 0) > 0
            )
            if not solved_in_current_run:
                return level_id
        return None

    if trace_db_path is None and game_id:
        solved_from_db = set(get_db_solved_levels_for_game(game_id=str(game_id), trace_db_path=trace_db_path))
        for level_id in ordered:
            if level_id not in solved_from_db:
                return level_id
        return None

    for level_id in ordered:
        item = state.get(level_id)
        if item is None or not bool(item.solved):
            return level_id
    return None


def get_db_solved_levels_for_game(
    *,
    game_id: str,
    trace_db_path: str | None = None,
) -> tuple[str, ...]:
    try:
        return get_solved_levels_for_game(db_path=trace_db_path, game_id=game_id)
    except Exception:
        return tuple()


def get_verified_prefix_traces(
    *,
    state: dict[str, CampaignLevelState],
    level_sequence: tuple[str, ...] | list[str],
    trace_db_path: str | None = None,
) -> tuple[SavedLevelTrace, ...]:
    resolved_game_id = next(iter(state.values())).game_id if state else ""
    explicit_level_ids = tuple(
        str(item)
        for item in level_sequence
        if bool(getattr(state.get(str(item)), "solved", False))
        and (
            int(getattr(state.get(str(item)), "attempt_count", 0) or 0) > 0
            or getattr(state.get(str(item)), "solution_trace_path", None) is not None
        )
    )
    ordered_level_ids = explicit_level_ids or tuple(
        str(item)
        for item in level_sequence
        if bool(getattr(state.get(str(item)), "solved", False))
    )
    return get_best_verified_trace_prefix(
        game_id=resolved_game_id,
        level_ids=ordered_level_ids,
        db_path=trace_db_path,
    )


def get_current_run_prefix_traces(
    *,
    level_sequence: tuple[str, ...] | list[str],
    current_run_traces: dict[str, SavedLevelTrace] | None = None,
    diagnostics: dict[str, object] | None = None,
) -> tuple[SavedLevelTrace, ...]:
    traces = dict(current_run_traces or {})
    ordered = tuple(str(item) for item in level_sequence)
    output: list[SavedLevelTrace] = []
    for level_id in ordered:
        entry = traces.get(level_id)
        if entry is None:
            continue
        if not validate_prefix_trace_entry(entry):
            if diagnostics is not None:
                warnings = list(diagnostics.get("prefix_validation_warnings", []))
                warnings.append(f"invalid_prefix_trace:{level_id}")
                diagnostics["prefix_validation_warnings"] = warnings
            continue
        output.append(entry)
    return tuple(output)


def validate_prefix_trace_entry(trace: SavedLevelTrace | None) -> bool:
    if trace is None:
        return False
    if not bool(getattr(trace, "solved", False)):
        return False
    if not bool(getattr(trace, "replay_verified", False)):
        return False
    if not tuple(getattr(trace, "action_trace", ())):
        return False
    if getattr(trace, "trace_id", None) is None and getattr(trace, "source_run_id", None) is None:
        return False
    return True
