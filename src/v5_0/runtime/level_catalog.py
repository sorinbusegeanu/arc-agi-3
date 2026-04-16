from __future__ import annotations

from v4_5.runtime.sessionAdapter import SessionAdapter


def get_level_sequence_for_game(game_id: str, env_factory=None) -> tuple[str, ...]:
    return get_supported_level_ids_for_game(game_id, env_factory=env_factory)


def get_next_level_id(level_sequence: tuple[str, ...] | list[str], current_level_id: str) -> str | None:
    ordered = tuple(str(item) for item in level_sequence)
    current = str(current_level_id)
    try:
        idx = ordered.index(current)
    except ValueError:
        return None
    if idx + 1 >= len(ordered):
        return None
    return ordered[idx + 1]


def get_supported_level_ids_for_game(game_id: str, env_factory=None) -> tuple[str, ...]:
    session_adapter = SessionAdapter()
    session = session_adapter.create_session(game_id, seed=0, render_terminal=False, env_factory=env_factory)
    try:
        observation = session_adapter.get_current_observation(session)
        win_levels = int(getattr(observation, "win_levels", 0) or 0)
        if win_levels > 0:
            return tuple(f"L{idx}" for idx in range(win_levels))
        # Metadata fallback for environments that do not expose win_levels.
        game = getattr(session.env_session.env, "_game", None)
        maybe_levels = getattr(game, "levels", None)
        if isinstance(maybe_levels, (list, tuple)) and len(maybe_levels) > 0:
            return tuple(f"L{idx}" for idx in range(len(maybe_levels)))
        return ("L0",)
    finally:
        session_adapter.close_session(session)


def validate_level_id_for_game(game_id: str, level_id: str, env_factory=None) -> None:
    supported = get_supported_level_ids_for_game(game_id, env_factory=env_factory)
    if str(level_id) not in set(supported):
        raise ValueError(f"level_id {level_id!r} is not supported for game {game_id!r}; supported={supported}")


def engine_supports_direct_level_start(game_id: str, env_factory=None) -> bool:
    session_adapter = SessionAdapter()
    session = session_adapter.create_session(game_id, seed=0, render_terminal=False, env_factory=env_factory)
    try:
        env = getattr(session, "env_session", None)
        env_obj = getattr(env, "env", None)
        game_obj = getattr(env_obj, "_game", None)
        direct_methods = (
            hasattr(env_obj, "set_level")
            or hasattr(env_obj, "start_level")
            or hasattr(game_obj, "set_level")
            or hasattr(game_obj, "start_level")
        )
        return bool(direct_methods)
    finally:
        session_adapter.close_session(session)
