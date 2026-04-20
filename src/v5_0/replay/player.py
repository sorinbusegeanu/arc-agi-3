from __future__ import annotations

from typing import Any, Callable

from v4_5.adapters.actionAdapter import ActionAdapter, ActionTranslationContext
from v4_5.runtime.sessionAdapter import SessionAdapter
from v5_0.memory.trace_store import get_best_verified_trace_prefix


def replay_saved_trace(
    *,
    game_id: str,
    action_trace: tuple[str, ...] | list[str],
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    session_adapter = SessionAdapter()
    action_adapter = ActionAdapter()
    session = session_adapter.create_session(
        game_id,
        seed=0,
        render_terminal=bool(render_terminal),
        env_factory=env_factory,
    )
    executed_count = 0
    divergence = False
    terminal_status = "none"
    final_level = 0
    try:
        for idx, action in enumerate(tuple(str(item) for item in action_trace)):
            observation = session_adapter.get_current_observation(session)
            context = ActionTranslationContext(
                available_action_ids=observation.available_actions,
                coordinate_action_id=session.environment_metadata.coordinate_action_id,
                coordinate_bounds=session.environment_metadata.coordinate_bounds,
            )
            try:
                translated = action_adapter.translate_token(action, context)
            except ValueError:
                divergence = True
                break
            try:
                executed = session_adapter.execute_action_prefix(session, (translated,), (action,))
            except Exception:
                divergence = True
                break
            executed_count = idx + 1
            post = session_adapter.get_current_observation(session)
            final_level = int(post.levels_completed)
            terminal_status = str(executed.terminal_status)
            if terminal_status in {"success", "failure"}:
                break
        success = not divergence
        return {
            "success": bool(success),
            "final_level_reached": int(final_level),
            "executed_action_count": int(executed_count),
            "divergence": bool(divergence),
            "terminal_status": terminal_status,
        }
    finally:
        session_adapter.close_session(session)


def replay_trace_at_frontier(
    *,
    game_id: str,
    level_id: str,
    prefix_traces,
    frontier_trace,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    prefix_actions = _compose_prefix_actions(tuple(prefix_traces or ()))
    frontier_actions = _extract_actions(frontier_trace)
    frontier_has_embedded_prefix = _trace_has_embedded_prefix(frontier_trace)

    expected_frontier = int(str(level_id).lstrip("L") or 0)
    if not frontier_has_embedded_prefix:
        prefix_result = replay_saved_trace(
            game_id=game_id,
            action_trace=tuple(prefix_actions),
            render_terminal=render_terminal,
            env_factory=env_factory,
        )
        frontier_reached = int(prefix_result.get("final_level_reached", 0)) >= expected_frontier
        if not frontier_reached:
            return {
                "success": False,
                "frontier_reached": False,
                "level_solved": False,
                "final_level_reached": int(prefix_result.get("final_level_reached", 0)),
                "divergence": bool(prefix_result.get("divergence", False)),
                "executed_action_count": int(prefix_result.get("executed_action_count", 0)),
            }
        if bool(prefix_result.get("divergence", False)):
            return {
                "success": False,
                "frontier_reached": False,
                "level_solved": False,
                "final_level_reached": int(prefix_result.get("final_level_reached", 0)),
                "divergence": True,
                "executed_action_count": int(prefix_result.get("executed_action_count", 0)),
            }

    combined = frontier_actions if frontier_has_embedded_prefix else (tuple(prefix_actions) + frontier_actions)
    result = replay_saved_trace(
        game_id=game_id,
        action_trace=combined,
        render_terminal=render_terminal,
        env_factory=env_factory,
    )
    final_level = int(result.get("final_level_reached", 0))
    expected_after = expected_frontier + 1
    level_solved = final_level >= expected_after
    verdict = verify_trace_matches_level(
        replay_result=result,
        intended_level_id=level_id,
    )
    return {
        "success": bool(verdict),
        "frontier_reached": bool(final_level >= expected_frontier),
        "level_solved": bool(level_solved),
        "final_level_reached": final_level,
        "divergence": bool(result.get("divergence", False)),
        "executed_action_count": int(result.get("executed_action_count", 0)),
    }


def replay_prefix_traces_to_frontier(
    *,
    game_id: str,
    prefix_traces,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    session_adapter = SessionAdapter()
    action_adapter = ActionAdapter()
    session = session_adapter.create_session(
        game_id,
        seed=0,
        render_terminal=bool(render_terminal),
        env_factory=env_factory,
    )
    traces = tuple(prefix_traces or ())
    expected_frontier = 0
    if traces:
        expected_frontier = max(int(str(getattr(trace, "level_id", "L0")).lstrip("L") or 0) + 1 for trace in traces)
    divergence = False
    executed_action_count = 0
    try:
        for action in _compose_prefix_actions(traces):
            observation = session_adapter.get_current_observation(session)
            context = ActionTranslationContext(
                available_action_ids=observation.available_actions,
                coordinate_action_id=session.environment_metadata.coordinate_action_id,
                coordinate_bounds=session.environment_metadata.coordinate_bounds,
            )
            try:
                translated = action_adapter.translate_token(str(action), context)
                session_adapter.execute_action_prefix(session, (translated,), (str(action),))
            except Exception:
                divergence = True
                break
            executed_action_count += 1

        final_observation = session_adapter.get_current_observation(session)
        final_level_index = int(final_observation.levels_completed)
        frontier_reached = (not divergence) and final_level_index >= expected_frontier
        if not frontier_reached:
            session_adapter.close_session(session)
            return {
                "session": None,
                "frontier_reached": False,
                "frontier_level_id": f"L{final_level_index}",
                "divergence": bool(divergence),
                "executed_action_count": int(executed_action_count),
            }
        return {
            "session": session,
            "frontier_reached": True,
            "frontier_level_id": f"L{final_level_index}",
            "divergence": bool(divergence),
            "executed_action_count": int(executed_action_count),
        }
    except Exception:
        try:
            session_adapter.close_session(session)
        except Exception:
            pass
        return {
            "session": None,
            "frontier_reached": False,
            "frontier_level_id": None,
            "divergence": True,
            "executed_action_count": int(executed_action_count),
        }


def replay_prefix_to_frontier(
    *,
    game_id: str,
    prefix_traces,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    replay = replay_prefix_traces_to_frontier(
        game_id=game_id,
        prefix_traces=tuple(prefix_traces or ()),
        render_terminal=bool(render_terminal),
        env_factory=env_factory,
    )
    if replay.get("session") is None:
        return {
            "session": None,
            "frontier_reached": False,
            "frontier_level_id": replay.get("frontier_level_id"),
            "failure_reason": "prefix_replay_failed",
        }
    if not bool(replay.get("frontier_reached", False)) or bool(replay.get("divergence", False)):
        try:
            SessionAdapter().close_session(replay.get("session"))
        except Exception:
            pass
        return {
            "session": None,
            "frontier_reached": False,
            "frontier_level_id": replay.get("frontier_level_id"),
            "failure_reason": "prefix_replay_failed",
        }
    return {
        "session": replay.get("session"),
        "frontier_reached": True,
        "frontier_level_id": replay.get("frontier_level_id"),
        "failure_reason": None,
    }


BOOTSTRAP_REPLAY_SOURCE = "bootstrap_replay"
SOLVED_PREFIX_REPLAY_SOURCE = "solved_prefix_replay"
FRONTIER_PREFIX_REPLAY_SOURCE = "frontier_prefix_replay"


def is_bootstrap_replay_source(source: str | None) -> bool:
    return str(source or "") == BOOTSTRAP_REPLAY_SOURCE


def is_solved_prefix_replay_source(source: str | None) -> bool:
    return str(source or "") == SOLVED_PREFIX_REPLAY_SOURCE


def is_frontier_prefix_replay_source(source: str | None) -> bool:
    return str(source or "") == FRONTIER_PREFIX_REPLAY_SOURCE


def trace_includes_bootstrap_prefix(saved_trace) -> bool:
    if saved_trace is None:
        return False
    if isinstance(saved_trace, dict):
        sources = saved_trace.get("action_sources")
        if isinstance(sources, (list, tuple)):
            return any(is_bootstrap_replay_source(str(item)) for item in sources)
    sources = getattr(saved_trace, "action_sources", None)
    if isinstance(sources, (list, tuple)):
        return any(is_bootstrap_replay_source(str(item)) for item in tuple(sources))
    actions = getattr(saved_trace, "action_trace", None)
    if isinstance(actions, (list, tuple)) and actions:
        first = actions[0]
        if isinstance(first, dict):
            return any(is_bootstrap_replay_source(str(item.get("source", ""))) for item in actions if isinstance(item, dict))
    return False


def verify_trace_matches_level(*, replay_result, intended_level_id: str) -> bool:
    if not isinstance(replay_result, dict):
        return False
    if bool(replay_result.get("divergence", False)):
        return False
    expected_frontier = int(str(intended_level_id).lstrip("L") or 0)
    expected_after = expected_frontier + 1
    final_level = int(replay_result.get("final_level_reached", 0))
    frontier_reached = final_level >= expected_frontier
    level_solved = final_level >= expected_after
    return bool(replay_result.get("success", False)) and frontier_reached and level_solved


def load_prefix_traces_for_level(
    *,
    game_id: str,
    level_id: str,
    level_ids: tuple[str, ...] | list[str] | None = None,
) -> tuple:
    if level_ids is None:
        try:
            from v5_0.runtime.level_catalog import get_level_sequence_for_game

            ordered = tuple(get_level_sequence_for_game(game_id))
        except Exception:
            ordered = tuple()
    else:
        ordered = tuple(str(item) for item in level_ids)
    if not ordered:
        return tuple()
    current_index = int(str(level_id).lstrip("L") or 0)
    earlier_levels = tuple(
        item
        for item in ordered
        if int(str(item).lstrip("L") or 0) < current_index
    )
    return get_best_verified_trace_prefix(game_id=game_id, level_ids=earlier_levels)


def _extract_actions(trace_like) -> tuple[str, ...]:
    if trace_like is None:
        return tuple()
    if isinstance(trace_like, dict):
        actions = tuple(str(item) for item in tuple(trace_like.get("action_trace", ()) or ()))
        sources = tuple(str(item) for item in tuple(trace_like.get("action_sources", ()) or ()))
        if sources and len(sources) == len(actions):
            return tuple(action for action, source in zip(actions, sources) if not is_bootstrap_replay_source(source))
        return actions
    if isinstance(trace_like, (list, tuple)):
        return tuple(str(item) for item in tuple(trace_like))
    actions = getattr(trace_like, "action_trace", ())
    actions_tuple = tuple(str(item) for item in tuple(actions))
    sources = tuple(str(item) for item in tuple(getattr(trace_like, "action_sources", ()) or ()))
    if sources and len(sources) == len(actions_tuple):
        return tuple(action for action, source in zip(actions_tuple, sources) if not is_bootstrap_replay_source(source))
    return actions_tuple


def _trace_has_embedded_prefix(trace_like) -> bool:
    if trace_like is None:
        return False
    if isinstance(trace_like, dict):
        sources = tuple(str(item) for item in tuple(trace_like.get("action_sources", ()) or ()))
    else:
        sources = tuple(str(item) for item in tuple(getattr(trace_like, "action_sources", ()) or ()))
    return any(is_solved_prefix_replay_source(source) for source in sources)


def _compose_prefix_actions(traces) -> tuple[str, ...]:
    combined: list[str] = []
    for trace in tuple(traces or ()):
        actions = _extract_actions(trace)
        combined.extend(actions)
    return tuple(combined)
