from __future__ import annotations

"""v8.32 productive-action persistence for the adaptive sampling portfolio.

v8.31 bounded arbitrary action combinations to depth four.  That is appropriate
for combinatorial branching, but it accidentally bounded repeated productive
runs too.  Movement tutorials expose the failure directly: later levels can
require six or seven repetitions of one action.

This layer keeps combinatorial sequence search bounded while allowing one action
that continues to change observable state to persist for a bounded rollout.  A
persistence rollout stops on a no-op/stall, GAME_OVER, action disappearance, or
a hard 64-step cap.  Level progress may carry the same action into the next level.
"""


_INSTALLED = False
_BASE_BEGIN_LEASE = None
_BASE_ON_EXTERNAL_RESET = None
_BASE_FORCED_ACTION = None
_BASE_OBSERVE_TRANSITION = None

_MAX_ACTION_PERSISTENCE = 64


def _clear_persistence_v832(sampler) -> None:
    sampler._v832_persist_action = None
    sampler._v832_persist_steps = 0
    sampler._v832_persist_origin = None


def _arm_persistence_v832(sampler, action: int, origin) -> None:
    sampler._v832_persist_action = int(action)
    sampler._v832_persist_steps = 0
    sampler._v832_persist_origin = origin


def _begin_lease_v832(self, seed: int) -> None:
    _BASE_BEGIN_LEASE(self, int(seed))
    _clear_persistence_v832(self)


def _on_external_reset_v832(self) -> None:
    _BASE_ON_EXTERNAL_RESET(self)
    _clear_persistence_v832(self)


def _forced_action_v832(
    self,
    *,
    level: int,
    context: int,
    actions: tuple[int, ...],
    history: tuple[int, ...],
) -> int | None:
    """Continue a productive rollout before selecting another portfolio branch."""
    from v8 import decision_point_sampling_v821 as sampling
    from v8 import sampling_portfolio_v831 as portfolio

    # Replay/verification and an already selected explicit short sequence remain
    # stronger authorities than persistence.
    if (
        self.base.replay_actions
        or self.base.replay_target is not None
        or self.base.verification is not None
        or self.active_sequence
    ):
        return _BASE_FORCED_ACTION(
            self,
            level=int(level),
            context=int(context),
            actions=tuple(actions),
            history=tuple(history),
        )

    action = getattr(self, "_v832_persist_action", None)
    steps = int(getattr(self, "_v832_persist_steps", 0))
    available = {int(value) for value in actions}
    if action is not None:
        action = int(action)
        if action in available and steps < _MAX_ACTION_PERSISTENCE:
            self._v832_persist_steps = steps + 1
            self.base.current = sampling.Intervention(
                "PERSIST",
                (int(level), int(context)),
                action,
                tuple(history),
            )
            self.mode_counts["PERSIST"] = int(self.mode_counts.get("PERSIST", 0)) + 1
            portfolio._set_mode("PROGRESS")
            portfolio._set_source(context, "ACTION_PERSISTENCE", (action,))
            return action
        _clear_persistence_v832(self)

    return _BASE_FORCED_ACTION(
        self,
        level=int(level),
        context=int(context),
        actions=tuple(actions),
        history=tuple(history),
    )


def _productive_transition(kwargs) -> bool:
    before_context = int(kwargs.get("before_context", 0))
    after_context = int(kwargs.get("after_context", before_context))
    changed_cells = int(kwargs.get("changed_cells", 0))
    return bool(after_context != before_context or changed_cells > 0)


def _observe_persist_v832(self, intervention, **kwargs) -> None:
    from v8 import sampling_portfolio_v831 as portfolio

    self.base.current = None
    before_level = int(kwargs.get("before_level", 0))
    after_level = int(kwargs.get("after_level", before_level))
    after_context = int(kwargs.get("after_context", 0))
    terminal_state = str(kwargs.get("terminal_state", ""))
    level_advanced = bool(kwargs.get("level_advanced", False))
    after_actions = tuple(int(value) for value in kwargs.get("after_actions", ()))
    history_after = tuple(int(value) for value in kwargs.get("history_after", ()))
    action = int(kwargs.get("action", intervention.action))
    success = bool(level_advanced or terminal_state == "WIN")
    productive = _productive_transition(kwargs)

    if terminal_state != "GAME_OVER" and after_actions:
        self.base.register_point(
            level=after_level,
            context=after_context,
            anchor=history_after,
            actions=after_actions,
            priority=6 if success else (4 if productive else 1),
        )

    if success:
        self.saw_progress = True
        self.base.transfer_action = action
        self.base.transfer_from_level = max(int(self.base.transfer_from_level), before_level)
        if terminal_state != "WIN" and action in after_actions:
            # A level boundary is a new branch.  Carry the productive action, but
            # never retain an old reset anchor across that boundary.
            _arm_persistence_v832(self, action, None)
        else:
            _clear_persistence_v832(self)
        portfolio._set_mode(None)
        return

    if (
        terminal_state != "GAME_OVER"
        and productive
        and action in after_actions
        and int(getattr(self, "_v832_persist_steps", 0)) < _MAX_ACTION_PERSISTENCE
    ):
        portfolio._set_mode(None)
        return

    origin = getattr(self, "_v832_persist_origin", None)
    _clear_persistence_v832(self)
    if terminal_state != "GAME_OVER" and origin is not None:
        self._schedule_next_sequence(origin)
    portfolio._set_mode(None)


def _observe_transition_v832(self, **kwargs) -> None:
    intervention = self.base.current
    if intervention is not None and str(intervention.kind) == "PERSIST":
        _observe_persist_v832(self, intervention, **kwargs)
        return

    sequence = bool(intervention is not None and str(intervention.kind) == "SEQUENCE")
    remaining = len(self.active_sequence) if sequence else -1
    origin = (
        self.active_point or intervention.point_key
        if sequence and intervention is not None
        else None
    )
    action = int(kwargs.get("action", intervention.action if intervention is not None else 0))
    before_level = int(kwargs.get("before_level", 0))
    after_level = int(kwargs.get("after_level", before_level))
    terminal_state = str(kwargs.get("terminal_state", ""))
    after_actions = tuple(int(value) for value in kwargs.get("after_actions", ()))
    success = bool(after_level > before_level or terminal_state == "WIN")
    productive = _productive_transition(kwargs)

    result = _BASE_OBSERVE_TRANSITION(self, **kwargs)

    if sequence:
        if success and terminal_state != "WIN" and action in after_actions:
            _arm_persistence_v832(self, action, None)
        elif (
            not success
            and terminal_state != "GAME_OVER"
            and remaining == 0
            and productive
            and action in after_actions
        ):
            # v8.31 would reset immediately after this singleton/finished prefix.
            # Keep walking the productive direction instead; if it later stalls,
            # resume the queued sequence frontier from the original anchor.
            self.pending_sequence = None
            _arm_persistence_v832(self, action, origin)
    elif success:
        # Another portfolio strategy found real progress.  Its own v8.21 logic
        # owns transfer/verification; discard any stale persistence rollout.
        _clear_persistence_v832(self)

    return result


def install_sampling_persistence_v832() -> None:
    global _INSTALLED, _BASE_BEGIN_LEASE, _BASE_ON_EXTERNAL_RESET
    global _BASE_FORCED_ACTION, _BASE_OBSERVE_TRANSITION
    if _INSTALLED:
        return

    from v8 import sampling_portfolio_v831 as portfolio

    cls = portfolio.PortfolioSampler
    _BASE_BEGIN_LEASE = cls.begin_lease
    _BASE_ON_EXTERNAL_RESET = cls.on_external_reset
    _BASE_FORCED_ACTION = cls.forced_action
    _BASE_OBSERVE_TRANSITION = cls.observe_transition

    cls.begin_lease = _begin_lease_v832
    cls.on_external_reset = _on_external_reset_v832
    cls.forced_action = _forced_action_v832
    cls.observe_transition = _observe_transition_v832
    _INSTALLED = True
