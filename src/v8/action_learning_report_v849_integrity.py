from __future__ import annotations

"""Preserve episode-wide productivity when a trajectory becomes mixed later."""

_INSTALLED = False
_BASE_RECORD_EPISODE_KIND = None
_BASE_RESET_EPISODE = None


def _record_episode_kind_integrity(
    env,
    kind: str,
    *,
    productive: bool,
    level_advanced: bool,
) -> None:
    any_productive = bool(
        getattr(env, "_v849_episode_any_productive", False) or productive
    )
    any_level = bool(
        getattr(env, "_v849_episode_any_level", False) or level_advanced
    )
    env._v849_episode_any_productive = any_productive
    env._v849_episode_any_level = any_level
    _BASE_RECORD_EPISODE_KIND(
        env,
        kind,
        productive=any_productive,
        level_advanced=any_level,
    )


def _reset_episode_integrity(env) -> None:
    _BASE_RESET_EPISODE(env)
    env._v849_episode_any_productive = False
    env._v849_episode_any_level = False


def install_action_learning_report_v849_integrity() -> None:
    global _INSTALLED, _BASE_RECORD_EPISODE_KIND, _BASE_RESET_EPISODE
    if _INSTALLED:
        return

    from v8 import action_learning_report_v849 as report

    _BASE_RECORD_EPISODE_KIND = report._record_episode_kind
    _BASE_RESET_EPISODE = report._reset_episode_metrics
    report._record_episode_kind = _record_episode_kind_integrity
    report._reset_episode_metrics = _reset_episode_integrity
    _INSTALLED = True
