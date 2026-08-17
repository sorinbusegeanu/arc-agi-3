from __future__ import annotations

"""v8.32 variable-length productive-action rollouts for the sampling portfolio.

Combinatorial branch generation remains bounded, but execution length does not.
A productive action rollout is an A* macro: once selected, the same action keeps
executing until the level resolves, the environment reaches a stationary no-op,
terminal failure occurs, the action disappears, or the actor's existing sampling
budget is exhausted.  There is no second internal action-count limit.

A level boundary ends the rollout.  The successful action is retained only as a
high-priority transfer candidate for the next level; the next level still gets a
fresh portfolio decision and therefore preserves exploration authority.
"""


_INSTALLED = False
_BASE_BEGIN_LEASE = None
_BASE_ON_EXTERNAL_RESET = None
_BASE_FORCED_ACTION = None
_BASE_OBSERVE_TRANSITION = None


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
    """Continue a productive rollout until a semantic boundary or actor budget end."""
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
    available = {int(value) for value in actions}
    if action is not None:
        action = int(action)
        if action in available:
            self._v832_persist_steps = int(getattr(self, "_v832_persist_steps", 0)) + 1
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
        # The level boundary is the semantic end of this rollout.  Do not force
        # persistence into the next level; expose the action as transfer evidence.
        _clear_persistence_v832(self)
        portfolio._set_mode(None)
        return

    if terminal_state != "GAME_OVER" and productive and action in after_actions:
        # No fixed length cap.  The actor's for-range(job.steps) is the budget.
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
        if success:
            # v8.31 already records the successful action as transfer evidence.
            # A new level is a new sampling decision, not a continuation counter.
            _clear_persistence_v832(self)
        elif (
            terminal_state != "GAME_OVER"
            and remaining == 0
            and productive
            and action in after_actions
        ):
            # A finished short prefix that is still changing state becomes A*.
            # If it later stalls, resume the sequence frontier from this origin.
            self.pending_sequence = None
            _arm_persistence_v832(self, action, origin)
    elif success:
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
