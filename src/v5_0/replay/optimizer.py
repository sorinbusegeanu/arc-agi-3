from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from v5_0.memory.trace_store import (
    get_best_trace_for_level,
    get_best_verified_trace_prefix,
    get_global_trace_store_path,
    get_solved_levels_for_game,
    initialize_trace_store,
    mark_trace_optimized,
    save_level_trace,
)
from v5_0.contracts.avatar_types import (
    SavedLevelTrace,
    TraceOptimizationCandidate,
    TraceOptimizationReport,
)
from v5_0.replay.analyzer import (
    analyze_trace_for_redundancy,
    propose_shorter_trace_candidates,
    score_trace_redundancy,
)
from v5_0.replay.player import replay_saved_trace, replay_trace_at_frontier, verify_trace_matches_level
from v5_0.runtime.level_catalog import get_level_sequence_for_game


def optimize_saved_trace(
    *,
    game_id: str,
    level_id: str,
    saved_trace: SavedLevelTrace,
    trace_db_path: str | None = None,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
) -> TraceOptimizationReport:
    prefix_traces = _load_prefix_traces_for_optimization(
        game_id=game_id,
        level_id=level_id,
        trace_db_path=trace_db_path,
    )
    return optimize_level_trace(
        game_id=game_id,
        level_id=level_id,
        saved_trace=saved_trace,
        prefix_traces=tuple(prefix_traces),
        render_terminal=render_terminal,
        env_factory=env_factory,
        trace_db_path=trace_db_path,
    )


def optimize_level_trace(
    *,
    game_id: str,
    level_id: str,
    saved_trace: SavedLevelTrace,
    prefix_traces=(),
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
    trace_db_path: str | None = None,
) -> TraceOptimizationReport:
    trace_db_path = initialize_trace_store(trace_db_path or get_global_trace_store_path())
    baseline_steps = int(saved_trace.step_count)
    baseline_trace_id = getattr(saved_trace, "trace_id", None)
    analysis = analyze_trace_for_redundancy(saved_trace)
    redundancy_score = score_trace_redundancy(saved_trace)
    raw_candidates = propose_shorter_trace_candidates(saved_trace, analysis)

    verified: list[TraceOptimizationCandidate] = []
    seen_candidates = set(raw_candidates)
    for candidate in sorted(raw_candidates, key=lambda item: (len(item), tuple(item))):
        replay = replay_trace_at_frontier(
            game_id=game_id,
            level_id=level_id,
            prefix_traces=prefix_traces,
            frontier_trace=candidate,
            render_terminal=render_terminal,
            env_factory=env_factory,
        )
        is_verified = verify_trace_matches_level(replay_result=replay, intended_level_id=level_id)
        verified.append(
            TraceOptimizationCandidate(
                game_id=game_id,
                level_id=level_id,
                action_trace=tuple(candidate),
                step_count=len(candidate),
                improvement_vs_baseline=baseline_steps - len(candidate),
                verified=is_verified,
            )
        )

    # Greedy local shortening pass: remove short windows deterministically.
    baseline_actions = tuple(saved_trace.action_trace)
    for window in (1, 2, 3):
        for start in range(0, max(0, len(baseline_actions) - window + 1)):
            candidate = baseline_actions[:start] + baseline_actions[start + window :]
            if not candidate or candidate in seen_candidates:
                continue
            seen_candidates.add(candidate)
            replay = replay_trace_at_frontier(
                game_id=game_id,
                level_id=level_id,
                prefix_traces=prefix_traces,
                frontier_trace=candidate,
                render_terminal=render_terminal,
                env_factory=env_factory,
            )
            is_verified = verify_trace_matches_level(replay_result=replay, intended_level_id=level_id)
            verified.append(
                TraceOptimizationCandidate(
                    game_id=game_id,
                    level_id=level_id,
                    action_trace=tuple(candidate),
                    step_count=len(candidate),
                    improvement_vs_baseline=baseline_steps - len(candidate),
                    verified=is_verified,
                )
            )

    valid = [item for item in verified if item.verified and item.step_count <= baseline_steps]
    if valid:
        best = sorted(valid, key=lambda item: (item.step_count, tuple(item.action_trace)))[0]
    else:
        best = TraceOptimizationCandidate(
            game_id=game_id,
            level_id=level_id,
            action_trace=tuple(saved_trace.action_trace),
            step_count=baseline_steps,
            improvement_vs_baseline=0,
            verified=bool(saved_trace.replay_verified),
        )

    db_updated = False
    optimized_trace_id = None
    if best.verified and int(best.step_count) < baseline_steps:
        current_best = get_best_trace_for_level(db_path=trace_db_path, game_id=game_id, level_id=level_id)
        if current_best is None or int(best.step_count) < int(current_best.step_count):
            optimized_at = datetime.now(timezone.utc).isoformat()
            optimized_trace = SavedLevelTrace(
                game_id=game_id,
                level_id=level_id,
                solved=True,
                action_trace=tuple(best.action_trace),
                step_count=int(best.step_count),
                source_run_id=saved_trace.source_run_id,
                trace_version=int(saved_trace.trace_version) + 1,
                replay_verified=True,
                trace_id=None,
                optimized=True,
                optimized_at=optimized_at,
                parent_trace_id=baseline_trace_id,
            )
            optimized_trace_id = save_level_trace(db_path=trace_db_path, trace=optimized_trace)
            mark_trace_optimized(
                db_path=trace_db_path,
                trace_id=optimized_trace_id,
                optimized=True,
                optimized_at=optimized_at,
                notes=None,
            )
            db_updated = True

    return TraceOptimizationReport(
        game_id=game_id,
        level_id=level_id,
        baseline_trace=saved_trace,
        best_candidate=best,
        candidates=tuple(sorted(verified, key=lambda item: (item.step_count, tuple(item.action_trace)))),
        failure_reason=None,
        diagnostics={
            "candidate_count": len(verified),
            "verified_candidate_count": len([item for item in verified if item.verified]),
            "baseline_step_count": baseline_steps,
            "best_step_count": int(best.step_count),
            "redundancy_score": float(redundancy_score["redundancy_score"]),
            "baseline_trace_id": baseline_trace_id,
            "optimized_trace_id": optimized_trace_id,
            "db_updated": bool(db_updated),
        },
    )


def optimize_game_traces(
    *,
    game_id: str,
    solved_level_ids: tuple[str, ...] | list[str],
    trace_db_path: str | None = None,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
) -> tuple[TraceOptimizationReport, ...]:
    trace_db_path = initialize_trace_store(trace_db_path or get_global_trace_store_path())
    reports: list[TraceOptimizationReport] = []
    for level_id in tuple(str(item) for item in solved_level_ids):
        baseline = get_best_trace_for_level(db_path=trace_db_path, game_id=game_id, level_id=level_id)
        if baseline is None:
            continue
        earlier_levels = tuple(
            item for item in tuple(str(x) for x in solved_level_ids)
            if int(str(item).lstrip("L") or 0) < int(str(level_id).lstrip("L") or 0)
        )
        prefix = get_best_verified_trace_prefix(game_id=game_id, level_ids=earlier_levels, db_path=trace_db_path)
        report = optimize_level_trace(
            game_id=game_id,
            level_id=level_id,
            saved_trace=baseline,
            prefix_traces=tuple(prefix),
            render_terminal=render_terminal,
            env_factory=env_factory,
            trace_db_path=trace_db_path,
        )
        reports.append(report)
    return tuple(reports)


def optimize_game_traces_from_db(
    *,
    game_id: str,
    trace_db_path: str | None = None,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
) -> tuple[TraceOptimizationReport, ...]:
    trace_db_path = initialize_trace_store(trace_db_path or get_global_trace_store_path())
    solved_level_ids = get_solved_levels_for_game(db_path=trace_db_path, game_id=game_id)
    return optimize_game_traces(
        game_id=game_id,
        solved_level_ids=solved_level_ids,
        trace_db_path=trace_db_path,
        render_terminal=render_terminal,
        env_factory=env_factory,
    )


def _load_prefix_traces_for_optimization(
    *,
    game_id: str,
    level_id: str,
    trace_db_path: str | None = None,
) -> tuple[SavedLevelTrace, ...]:
    current_index = int(str(level_id).lstrip("L") or 0)
    if current_index <= 0:
        return tuple()
    try:
        ordered = tuple(get_level_sequence_for_game(game_id))
    except Exception:
        ordered = tuple(f"L{i}" for i in range(current_index + 1))
    earlier_levels = tuple(
        item
        for item in ordered
        if int(str(item).lstrip("L") or 0) < current_index
    )
    return get_best_verified_trace_prefix(
        game_id=game_id,
        level_ids=earlier_levels,
        db_path=trace_db_path,
    )
