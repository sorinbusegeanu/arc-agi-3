from __future__ import annotations

"""v8.60 state-transition-driven click exploration.

v8.57 made click coverage complete, but productive coordinates were still explored
only after the broad sweep and were repeated at most once. For multi-state click
mechanics (notably gp03), that spends most of a small interaction budget discovering
coordinates instead of characterizing the causal state machine of a coordinate that
has already produced evidence.

This layer composes beneath the established v8.32/v8.47/v8.48 authorities. A
productive CLICK_SCAN/CLICK_CHARACTERIZE transition immediately becomes a bounded
local characterization sequence. The same executable coordinate is re-clicked while
it continues to produce novel observable state transitions. Characterization stops
on no observable change, terminal/level progress, a repeated transition (cycle), or
the bounded repeat cap. Broad coverage then resumes from the existing frontier.
"""

import os
from dataclasses import dataclass


_INSTALLED = False
_BASE_BEGIN_LEASE = None
_BASE_PREPARE_STEP = None
_BASE_FORCED_ACTION = None
_BASE_OBSERVE_TRANSITION = None

_DEFAULT_REPEAT_CAP = 4
_REPEAT_CAP_ENV = "ARC_AGI3_V8_CLICK_CHARACTERIZE_CAP"


@dataclass(frozen=True, slots=True)
class ClickTransition:
    action: int
    before_context: int
    after_context: int


def _repeat_cap() -> int:
    raw = os.environ.get(_REPEAT_CAP_ENV)
    if raw is None:
        return _DEFAULT_REPEAT_CAP
    try:
        return max(1, min(16, int(raw)))
    except ValueError:
        return _DEFAULT_REPEAT_CAP


def _ensure_state(sampler) -> None:
    if not hasattr(sampler, "_v860_pending_action"):
        sampler._v860_pending_action = None
    if not hasattr(sampler, "_v860_repeat_depth"):
        sampler._v860_repeat_depth = {}
    if not hasattr(sampler, "_v860_seen_transitions"):
        sampler._v860_seen_transitions = set()
    if not hasattr(sampler, "_v860_transition_counts"):
        sampler._v860_transition_counts = {}


def _clear_pending(sampler) -> None:
    sampler._v860_pending_action = None


def _begin_lease_v860(self, seed: int) -> None:
    _BASE_BEGIN_LEASE(self, int(seed))
    _ensure_state(self)
    _clear_pending(self)


def _prepare_step_v860(self, env) -> bool:
    _ensure_state(self)
    if self._v860_pending_action is not None:
        self.pending_sequence = None
        self.base.pending_reset = None
        return False
    return bool(_BASE_PREPARE_STEP(self, env))


def _forced_action_v860(
    self,
    *,
    level: int,
    context: int,
    actions: tuple[int, ...],
    history: tuple[int, ...],
) -> int | None:
    _ensure_state(self)
    pending = self._v860_pending_action
    if pending is not None:
        action = int(pending)
        if action in {int(value) for value in actions}:
            from v8 import decision_point_sampling_v821 as sampling
            from v8 import sampling_portfolio_v831 as portfolio

            self._v860_pending_action = None
            self.base.current = sampling.Intervention(
                "CLICK_CHARACTERIZE",
                (int(level), int(context)),
                action,
                tuple(history),
            )
            portfolio._set_mode("NOVELTY")
            portfolio._set_source(context, "CLICK_CHARACTERIZE", (action,))
            return action
        self._v860_pending_action = None
    return _BASE_FORCED_ACTION(
        self,
        level=int(level),
        context=int(context),
        actions=tuple(actions),
        history=tuple(history),
    )


def _productive_click_transition(intervention, kwargs) -> ClickTransition | None:
    if intervention is None or str(getattr(intervention, "kind", "")) not in {
        "CLICK_SCAN",
        "CLICK_CHARACTERIZE",
    }:
        return None
    action = int(kwargs.get("action", getattr(intervention, "action", -1)))
    try:
        from v8 import click_exploration_v848 as click

        if not click._is_exact_click_token(action):
            return None
    except (AttributeError, TypeError, ValueError):
        return None
    before_level = int(kwargs.get("before_level", 0))
    after_level = int(kwargs.get("after_level", before_level))
    if bool(kwargs.get("level_advanced", False)) or after_level > before_level:
        return None
    terminal = str(kwargs.get("terminal_state", ""))
    if terminal in {"WIN", "GAME_OVER"}:
        return None
    if int(kwargs.get("changed_cells", 0)) <= 0:
        return None
    return ClickTransition(
        action=action,
        before_context=int(kwargs.get("before_context", 0)),
        after_context=int(kwargs.get("after_context", 0)),
    )


def _observe_transition_v860(self, **kwargs):
    _ensure_state(self)
    intervention = self.base.current
    transition = _productive_click_transition(intervention, kwargs)
    result = _BASE_OBSERVE_TRANSITION(self, **kwargs)

    if transition is None:
        if intervention is not None and str(getattr(intervention, "kind", "")) == "CLICK_CHARACTERIZE":
            _clear_pending(self)
        return result

    key = (
        int(transition.action),
        int(transition.before_context),
        int(transition.after_context),
    )
    prior_seen = key in self._v860_seen_transitions
    self._v860_seen_transitions.add(key)
    self._v860_transition_counts[key] = int(self._v860_transition_counts.get(key, 0)) + 1

    action = int(transition.action)
    depth = int(self._v860_repeat_depth.get(action, 0)) + 1
    self._v860_repeat_depth[action] = depth
    if prior_seen or depth >= _repeat_cap():
        _clear_pending(self)
        return result

    self._v860_pending_action = action
    self.pending_sequence = None
    self.base.pending_reset = None
    return result


def transition_telemetry_v860(sampler) -> dict[str, object]:
    _ensure_state(sampler)
    return {
        "pending_action": sampler._v860_pending_action,
        "repeat_depth": dict(sampler._v860_repeat_depth),
        "distinct_transitions": len(sampler._v860_seen_transitions),
        "transition_counts": dict(sampler._v860_transition_counts),
        "repeat_cap": _repeat_cap(),
    }


def install_click_transition_exploration_v860() -> None:
    global _INSTALLED
    global _BASE_BEGIN_LEASE, _BASE_PREPARE_STEP, _BASE_FORCED_ACTION, _BASE_OBSERVE_TRANSITION
    if _INSTALLED:
        return

    from v8 import click_exploration_v848 as click

    # v8.48 is already the established immediate lower delegate beneath the
    # historical public sampler authorities. Compose inside its saved bases so
    # both public and intermediate authority identities remain unchanged.
    _BASE_BEGIN_LEASE = click._BASE_SAMPLER_BEGIN_LEASE
    click._BASE_SAMPLER_BEGIN_LEASE = _begin_lease_v860

    _BASE_PREPARE_STEP = click._BASE_SAMPLER_PREPARE_STEP
    click._BASE_SAMPLER_PREPARE_STEP = _prepare_step_v860

    _BASE_FORCED_ACTION = click._BASE_SAMPLER_FORCED_ACTION
    click._BASE_SAMPLER_FORCED_ACTION = _forced_action_v860

    _BASE_OBSERVE_TRANSITION = click._BASE_SAMPLER_OBSERVE_TRANSITION
    click._BASE_SAMPLER_OBSERVE_TRANSITION = _observe_transition_v860
    _INSTALLED = True
