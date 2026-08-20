from __future__ import annotations

"""Composition fixes for the v8.47 evidence-guided prefix frontier.

v8.47 sequence expansion is search, not learned action persistence.  This layer
routes sequence observations through the existing v8.33 -> v8.32 -> v8.44 stack
before recording frontier evidence, so v8.44 remains the authority that decides
whether an action has enough causal evidence to persist.

A newly reached same-level child may still be expanded immediately without a
reset, but only when that exact child is the globally preferred evidence frontier.
That is lazy best-first prefix expansion, not persistence: the next action is
selected from the child's untried actions and may differ from the prior action.
There is no internal sequence-length horizon.
"""


_INSTALLED = False
_BASE_FORCED_ACTION = None


def _clear_continuation_v847(self) -> None:
    self._v847_continuation_node_id = None


def _on_external_reset_v847_composed(self) -> None:
    from v8 import sampling_evidence_frontier_v847 as frontier

    _clear_continuation_v847(self)
    return frontier._on_external_reset_v847(self)


def _forced_action_v847_composed(
    self,
    *,
    level: int,
    context: int,
    actions: tuple[int, ...],
    history: tuple[int, ...],
) -> int | None:
    """Use all older forced authorities first, then expand the live best child."""

    forced = _BASE_FORCED_ACTION(
        self,
        level=int(level),
        context=int(context),
        actions=tuple(actions),
        history=tuple(history),
    )
    if forced is not None:
        return int(forced)

    from v8 import decision_point_sampling_v821 as sampling
    from v8 import sampling_evidence_frontier_v847 as frontier
    from v8 import sampling_portfolio_v831 as portfolio

    continuation = getattr(self, "_v847_continuation_node_id", None)
    if continuation is None:
        return None
    selected = frontier._best_expansion_v847(self)
    if selected is None:
        _clear_continuation_v847(self)
        return None
    node, action = selected
    if str(node.node_id) != str(continuation):
        # Another node has stronger evidence.  Leave it to the next SEQUENCE
        # portfolio slot, which can reset/replay its stored anchor.
        _clear_continuation_v847(self)
        return None
    if (
        int(node.level) != int(level)
        or int(node.context) != int(context)
        or tuple(int(value) for value in node.anchor) != tuple(int(value) for value in history)
    ):
        _clear_continuation_v847(self)
        return None
    available = {int(value) for value in actions}
    if int(action) not in available:
        node.available_actions.discard(int(action))
        self._v847_dirty = True
        _clear_continuation_v847(self)
        return None

    _clear_continuation_v847(self)
    self._v847_active_expansion = (str(node.node_id), int(action))
    self.active_point = (int(level), int(context))
    self.active_anchor = tuple(int(value) for value in history)
    self.active_sequence_full = (int(action),)
    self.active_sequence.clear()
    self.base.current = sampling.Intervention(
        "SEQUENCE",
        (int(level), int(context)),
        int(action),
        tuple(int(value) for value in history),
    )
    self.mode_counts["SEQUENCE_CONTINUE"] = int(
        self.mode_counts.get("SEQUENCE_CONTINUE", 0)
    ) + 1
    portfolio._set_mode("SEQUENCE")
    portfolio._set_source(context, "SEQUENCE_FRONTIER", (int(action),))
    return int(action)


def _observe_transition_v847_composed(self, **kwargs) -> None:
    from v8 import sampling_evidence_frontier_v847 as frontier

    intervention = self.base.current
    if intervention is None or str(intervention.kind) != "SEQUENCE":
        _clear_continuation_v847(self)
        return frontier._BASE_OBSERVE_TRANSITION(self, **kwargs)

    active = getattr(self, "_v847_active_expansion", None)
    before_level = int(kwargs.get("before_level", 0))
    before_context = int(kwargs.get("before_context", 0))
    after_level = int(kwargs.get("after_level", before_level))
    after_context = int(kwargs.get("after_context", before_context))
    action = int(kwargs.get("action", intervention.action))
    after_actions = tuple(int(value) for value in kwargs.get("after_actions", ()))
    history_after = tuple(int(value) for value in kwargs.get("history_after", ()))
    changed_cells = int(kwargs.get("changed_cells", 0))
    terminal_state = str(kwargs.get("terminal_state", ""))
    level_advanced = bool(kwargs.get("level_advanced", after_level > before_level))
    prediction_error = float(kwargs.get("prediction_error", 0.0))
    future_delta = float(kwargs.get("future_delta", 0.0))
    source_node_id = (
        str(active[0])
        if active is not None
        else frontier._canonical_id(before_level, before_context)
    )

    # Preserve the complete pre-v8.47 composition.  In particular, v8.32 may
    # request persistence, but v8.44's wrapped _arm_persistence_v832 remains the
    # causal-proof gate and rejects a merely productive singleton.
    result = frontier._BASE_OBSERVE_TRANSITION(self, **kwargs)

    destination = frontier._record_expansion_v847(
        self,
        source_node_id=source_node_id,
        action=action,
        before_level=before_level,
        before_context=before_context,
        after_level=after_level,
        after_context=after_context,
        after_actions=after_actions,
        history_after=history_after,
        changed_cells=changed_cells,
        terminal_state=terminal_state,
        level_advanced=level_advanced,
        prediction_error=prediction_error,
        future_delta=future_delta,
    )
    self._v847_active_expansion = None

    success = bool(level_advanced or terminal_state == "WIN")
    if destination is not None and not success and terminal_state != "GAME_OVER":
        self._v847_continuation_node_id = str(destination.node_id)
    else:
        _clear_continuation_v847(self)
    return result


def install_sampling_evidence_frontier_v847_fixups() -> None:
    global _INSTALLED, _BASE_FORCED_ACTION
    if _INSTALLED:
        return

    from v8 import sampling_evidence_frontier_v847 as frontier
    from v8 import sampling_portfolio_v831 as portfolio

    cls = portfolio.PortfolioSampler
    _BASE_FORCED_ACTION = cls.forced_action

    # Keep v8.47 as the reset implementation but clear the live continuation
    # marker before delegating through its v8.33/v8.32/v8.44 reset stack.
    cls.on_external_reset = _on_external_reset_v847_composed
    cls.forced_action = _forced_action_v847_composed
    cls.observe_transition = _observe_transition_v847_composed
    _INSTALLED = True
