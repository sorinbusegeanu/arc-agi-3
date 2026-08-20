from __future__ import annotations

"""Composition fixes for the v8.47 evidence-guided prefix frontier.

v8.47 sequence expansion is search, not learned action persistence.  The v8.32
public begin/reset/forced/observe methods remain authoritative.  This layer inserts
frontier behavior underneath those methods, preserving the existing
v8.33 -> v8.44 causal-proof composition.

A newly reached same-level child may be expanded immediately without a reset only
when that exact child is the globally preferred evidence frontier.  This is lazy
best-first prefix expansion, not persistence: the next action is selected from the
child's untried actions and may differ from the prior action.  There is no internal
sequence-length horizon.
"""


_INSTALLED = False
_BASE_LOWER_RESET = None
_BASE_LOWER_FORCED = None
_BASE_LOWER_OBSERVE = None


def _clear_continuation_v847(self) -> None:
    self._v847_continuation_node_id = None


def _lower_reset_v847(self) -> None:
    self._v847_active_expansion = None
    _clear_continuation_v847(self)
    return _BASE_LOWER_RESET(self)


def _lower_forced_v847(
    self,
    *,
    level: int,
    context: int,
    actions: tuple[int, ...],
    history: tuple[int, ...],
) -> int | None:
    """Use every older forced authority before expanding the live best child."""

    forced = _BASE_LOWER_FORCED(
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
        # Another node has stronger evidence. Leave it for the next SEQUENCE slot,
        # which can reset/replay that node's stored anchor.
        _clear_continuation_v847(self)
        return None
    if (
        int(node.level) != int(level)
        or int(node.context) != int(context)
        or tuple(int(value) for value in node.anchor)
        != tuple(int(value) for value in history)
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


def _lower_observe_v847(self, **kwargs) -> None:
    """Record sequence evidence while leaving v8.32 post-processing intact."""

    from v8 import sampling_evidence_frontier_v847 as frontier

    intervention = self.base.current
    if intervention is None or str(intervention.kind) != "SEQUENCE":
        _clear_continuation_v847(self)
        return _BASE_LOWER_OBSERVE(self, **kwargs)

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

    # This calls the pre-v8.47 lower composition (v8.44 beneath v8.32). v8.32
    # then resumes after this function and may request persistence; v8.44's wrapped
    # arm function remains the causal-proof gate and rejects mere motion.
    result = _BASE_LOWER_OBSERVE(self, **kwargs)

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
    global _INSTALLED
    global _BASE_LOWER_RESET, _BASE_LOWER_FORCED, _BASE_LOWER_OBSERVE
    if _INSTALLED:
        return

    from v8 import sampling_persistence_v832 as persistence
    from v8 import sampling_portfolio_v831 as portfolio

    cls = portfolio.PortfolioSampler
    _BASE_LOWER_RESET = persistence._BASE_ON_EXTERNAL_RESET
    _BASE_LOWER_FORCED = persistence._BASE_FORCED_ACTION
    _BASE_LOWER_OBSERVE = persistence._BASE_OBSERVE_TRANSITION

    persistence._BASE_ON_EXTERNAL_RESET = _lower_reset_v847
    persistence._BASE_FORCED_ACTION = _lower_forced_v847
    persistence._BASE_OBSERVE_TRANSITION = _lower_observe_v847

    # Preserve v8.32's public identities exactly as the v8.33/v8.44 compatibility
    # layers do. v8.47 owns prepare_step/discovery and the internal frontier only.
    cls.on_external_reset = persistence._on_external_reset_v832
    cls.forced_action = persistence._forced_action_v832
    cls.observe_transition = persistence._observe_transition_v832
    _INSTALLED = True
