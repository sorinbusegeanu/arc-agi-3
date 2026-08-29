from __future__ import annotations

"""Small runtime fixups for the v8.61 transition graph."""

_INSTALLED = False
_BASE_RECORD_TRANSITION = None


def _record_transition_v861_fix(sampler, intervention, kwargs) -> None:
    env = getattr(sampler, "_v861_env", None)
    if env is not None:
        sampler._v861_after_observation = getattr(env, "_last_grid", None)
    _BASE_RECORD_TRANSITION(sampler, intervention, kwargs)
    sampler._v861_current_observation = getattr(sampler, "_v861_after_observation", None)


def install_click_transition_graph_v861_fixups() -> None:
    global _INSTALLED, _BASE_RECORD_TRANSITION
    if _INSTALLED:
        return
    from v8 import click_transition_graph_v861 as graph

    _BASE_RECORD_TRANSITION = graph._record_transition
    graph._record_transition = _record_transition_v861_fix
    _INSTALLED = True
