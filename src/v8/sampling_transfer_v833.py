from __future__ import annotations

"""v8.33 cross-game transfer and true random-rollout sampling.

The portfolio's RANDOM and TRANSFER labels now denote rollouts rather than one-step
probes. Random rollout actions are sampled independently at every state until the
level resolves, terminal failure occurs, or the actor lease budget ends. Transfer
rollouts query accumulated foreign-game M7 knowledge first and normalized M1
knowledge second, continuing only while the transferred action remains productive.

No internal action-count horizon is introduced. The actor lease remains the budget
authority and the v8.31 10% random exploration floor remains intact.
"""

import math

from v8.persistent_identity import world_id

_INSTALLED = False
_BASE_BEGIN_LEASE = None
_BASE_ON_EXTERNAL_RESET = None
_BASE_FORCED_ACTION = None
_BASE_DISCOVERY_ACTION = None
_BASE_OBSERVE_TRANSITION = None

_COLD_MODES_V833 = (
    "TRANSFER", "SEQUENCE", "NOVELTY", "SEQUENCE", "RANDOM",
    "TRANSFER", "PROGRESS", "NOVELTY", "SEQUENCE", "MEMORY",
    "TRANSFER", "SEQUENCE", "RANDOM", "NOVELTY", "PROGRESS",
    "TRANSFER", "SEQUENCE", "NOVELTY", "SEQUENCE", "MEMORY",
)
_WARM_MODES_V833 = (
    "PROGRESS", "MEMORY", "TRANSFER", "PROGRESS", "RANDOM",
    "MEMORY", "PROGRESS", "TRANSFER", "SEQUENCE", "MEMORY",
    "PROGRESS", "NOVELTY", "RANDOM", "MEMORY", "PROGRESS",
    "TRANSFER", "SEQUENCE", "NOVELTY", "MEMORY", "PROGRESS",
)


def _clear_rollouts_v833(sampler) -> None:
    sampler._v833_random_rollout = False
    sampler._v833_transfer_rollout = False
    sampler._v833_transfer_origin = ""


def _begin_lease_v833(self, seed: int) -> None:
    _BASE_BEGIN_LEASE(self, int(seed))
    _clear_rollouts_v833(self)


def _on_external_reset_v833(self) -> None:
    _BASE_ON_EXTERNAL_RESET(self)
    _clear_rollouts_v833(self)


def _current_view():
    try:
        from v8 import behavior_recovery as behavior
        return getattr(behavior, "_CURRENT_ACTOR_VIEW", None)
    except BaseException:
        return None


def _lineage_transfer_index(view, game_id: str) -> dict[int, tuple[tuple[float, object, str], ...]]:
    from v8 import behavior_recovery as behavior
    from v8.model import RelationType

    view._refresh_strategy_cache()
    current_game = int(world_id(str(game_id)))
    version = tuple(getattr(view, "_strategy_version", ()))
    cache_key = (version, current_game)
    if getattr(view, "_v833_transfer_index_key", None) == cache_key:
        return getattr(view, "_v833_transfer_index", {})

    nodes = dict(getattr(view, "_node_by_uid", {}))
    parents = getattr(view, "_parents", {})
    direct_games: dict[object, set[int]] = {}
    correspondence: dict[object, list[tuple[object, float]]] = {}
    for edge in view.edge_records():
        relation = int(edge.relation_type)
        if relation == int(RelationType.GAME_PROVENANCE) and int(edge.target_uid.hi) == 0:
            direct_games.setdefault(edge.source_uid, set()).add(int(edge.target_uid.lo))
        elif relation in {
            int(RelationType.SIMILAR_TO),
            int(RelationType.TRANSFER_CORRESPONDENCE),
        }:
            score = max(0.0, float(getattr(edge, "score", 0.0)))
            correspondence.setdefault(edge.source_uid, []).append((edge.target_uid, score))
            correspondence.setdefault(edge.target_uid, []).append((edge.source_uid, score))

    lineage_cache: dict[object, tuple[frozenset[object], frozenset[int], float]] = {}

    def lineage(uid):
        cached = lineage_cache.get(uid)
        if cached is not None:
            return cached
        visited = {uid}
        frontier = {uid}
        games = set(direct_games.get(uid, ()))
        transfer_prior = 0.0
        for _depth in range(8):
            following = set()
            for current in frontier:
                row = nodes.get(current)
                if row is not None:
                    transfer_prior = max(
                        transfer_prior,
                        max(0.0, float(getattr(row, "transfer_prior", 0.0))),
                    )
                for parent in parents.get(current, ()):
                    games.update(direct_games.get(parent, ()))
                    if parent not in visited:
                        visited.add(parent)
                        following.add(parent)
            if not following:
                break
            frontier = following
        result = (frozenset(visited), frozenset(games), float(transfer_prior))
        lineage_cache[uid] = result
        return result

    by_action: dict[int, list[tuple[float, object, str]]] = {}
    seen = set()
    for strategy in tuple(getattr(view, "_strategy_fallback", ())):
        if strategy.strategy_uid in seen:
            continue
        seen.add(strategy.strategy_uid)
        if int(strategy.support) <= 0:
            continue

        ancestors, formation_games, transfer_prior = lineage(strategy.strategy_uid)
        if not any(int(game) != current_game for game in formation_games):
            continue
        if not (
            behavior.strategy_can_control(view, strategy.strategy_uid, strategy.outcome_uid)
            or behavior._strategy_can_probe(view, strategy.strategy_uid, strategy.outcome_uid)
        ):
            continue

        structural_match = 0.0
        for ancestor in ancestors:
            for other, score in correspondence.get(ancestor, ()):
                _other_lineage, other_games, other_prior = lineage(other)
                if current_game in other_games:
                    structural_match = max(structural_match, float(score), float(other_prior))

        node = nodes.get(strategy.strategy_uid)
        if node is not None:
            transfer_prior = max(
                transfer_prior,
                max(0.0, float(getattr(node, "transfer_prior", 0.0))),
            )
        support_prior = 0.05 * math.log1p(max(0, int(strategy.support)))
        efficiency = 1.0 / max(1e-9, float(strategy.mean_cost))
        probation_penalty = 0.20 if bool(strategy.probationary) else 0.0
        score = (
            float(strategy.reliability)
            + 0.10 * efficiency
            + support_prior
            + 0.50 * float(transfer_prior)
            + 0.75 * float(structural_match)
            - probation_penalty
        )
        by_action.setdefault(int(strategy.action_id), []).append(
            (float(score), strategy.strategy_uid, "M7")
        )

    result = {
        action: tuple(
            sorted(
                rows,
                key=lambda item: (
                    -float(item[0]),
                    int(item[1].hi),
                    int(item[1].lo),
                ),
            )
        )
        for action, rows in by_action.items()
    }
    view._v833_transfer_index_key = cache_key
    view._v833_transfer_index = result
    return result


def _cross_game_transfer_action(sampler, actions: tuple[int, ...]):
    view = _current_view()
    available = tuple(sorted({int(value) for value in actions}))
    if view is None or not available:
        return None

    index = _lineage_transfer_index(view, sampler.game_id)
    choices = []
    for action in available:
        rows = index.get(int(action), ())
        if rows:
            score, uid, origin = rows[0]
            choices.append((float(score), int(action), str(origin), uid))
    if choices:
        score, action, origin, uid = min(
            choices,
            key=lambda item: (
                -item[0],
                item[1],
                int(item[3].hi),
                int(item[3].lo),
            ),
        )
        return int(action), str(origin), uid

    try:
        from v8 import restart_memory_v815 as restart
        restart._build_restart_indexes(view)
        priors = getattr(view, "_v815_normalized_action_priors", {})
    except BaseException:
        priors = {}
    normalized = []
    for action in available:
        row = priors.get(int(action))
        if row is None:
            continue
        support, score = int(row[0]), float(row[1])
        if support > 0 and score > 0.0:
            normalized.append((score, support, int(action)))
    if normalized:
        score, support, action = min(
            normalized,
            key=lambda item: (-item[0], -item[1], item[2]),
        )
        return int(action), "NORMALIZED_M1", None
    return None


def _set_intervention(self, *, kind: str, level: int, context: int, action: int, history):
    from v8 import decision_point_sampling_v821 as sampling
    self.base.current = sampling.Intervention(
        str(kind),
        (int(level), int(context)),
        int(action),
        tuple(history),
    )


def _forced_action_v833(
    self,
    *,
    level: int,
    context: int,
    actions: tuple[int, ...],
    history: tuple[int, ...],
) -> int | None:
    from v8 import sampling_portfolio_v831 as portfolio

    if (
        self.base.replay_actions
        or self.base.replay_target is not None
        or self.base.verification is not None
        or self.active_sequence
        or getattr(self, "_v832_persist_action", None) is not None
    ):
        return _BASE_FORCED_ACTION(
            self,
            level=int(level),
            context=int(context),
            actions=tuple(actions),
            history=tuple(history),
        )

    available = tuple(sorted({int(value) for value in actions}))
    if not available:
        _clear_rollouts_v833(self)
        return None

    if bool(getattr(self, "_v833_random_rollout", False)):
        action = int(available[self.base.rng.randrange(len(available))])
        _set_intervention(
            self,
            kind="RANDOM_WALK",
            level=level,
            context=context,
            action=action,
            history=history,
        )
        self.mode_counts["RANDOM_WALK"] = int(self.mode_counts.get("RANDOM_WALK", 0)) + 1
        portfolio._set_mode("RANDOM")
        portfolio._set_source(context, "RANDOM_WALK", (action,))
        return action

    if bool(getattr(self, "_v833_transfer_rollout", False)):
        selected = _cross_game_transfer_action(self, available)
        if selected is not None:
            action, origin, _uid = selected
            _set_intervention(
                self,
                kind="CROSS_GAME_TRANSFER",
                level=level,
                context=context,
                action=action,
                history=history,
            )
            self._v833_transfer_origin = str(origin)
            key = f"CROSS_GAME_TRANSFER_{origin}"
            self.mode_counts[key] = int(self.mode_counts.get(key, 0)) + 1
            portfolio._set_mode("TRANSFER")
            portfolio._set_source(context, key, (action,))
            return int(action)
        self._v833_transfer_rollout = False
        self._v833_transfer_origin = ""

    return _BASE_FORCED_ACTION(
        self,
        level=int(level),
        context=int(context),
        actions=tuple(actions),
        history=tuple(history),
    )


def _discovery_action_v833(
    self,
    *,
    level: int,
    context: int,
    actions: tuple[int, ...],
    history: tuple[int, ...],
) -> int | None:
    from v8 import sampling_portfolio_v831 as portfolio

    available = tuple(sorted({int(value) for value in actions}))
    mode = str(getattr(portfolio._PORTFOLIO_STATE, "mode", "PROGRESS"))

    if mode == "TRANSFER":
        selected = _cross_game_transfer_action(self, available)
        if selected is not None:
            action, origin, _uid = selected
            self._v833_transfer_rollout = True
            self._v833_transfer_origin = str(origin)
            _set_intervention(
                self,
                kind="CROSS_GAME_TRANSFER",
                level=level,
                context=context,
                action=action,
                history=history,
            )
            key = f"CROSS_GAME_TRANSFER_{origin}"
            self.mode_counts[key] = int(self.mode_counts.get(key, 0)) + 1
            portfolio._set_source(context, key, (action,))
            return int(action)

    action = _BASE_DISCOVERY_ACTION(
        self,
        level=int(level),
        context=int(context),
        actions=available,
        history=tuple(history),
    )
    intervention = self.base.current
    if (
        action is not None
        and intervention is not None
        and str(intervention.kind) == "RANDOM"
    ):
        self._v833_random_rollout = True
    return action


def _productive_transition(kwargs) -> bool:
    before_context = int(kwargs.get("before_context", 0))
    after_context = int(kwargs.get("after_context", before_context))
    changed_cells = int(kwargs.get("changed_cells", 0))
    return bool(after_context != before_context or changed_cells > 0)


def _register_destination(self, *, kwargs, priority: int) -> None:
    terminal_state = str(kwargs.get("terminal_state", ""))
    after_actions = tuple(int(value) for value in kwargs.get("after_actions", ()))
    if terminal_state == "GAME_OVER" or not after_actions:
        return
    self.base.register_point(
        level=int(kwargs.get("after_level", 0)),
        context=int(kwargs.get("after_context", 0)),
        anchor=tuple(int(value) for value in kwargs.get("history_after", ())),
        actions=after_actions,
        priority=int(priority),
    )


def _observe_random_v833(self, intervention, **kwargs) -> None:
    from v8 import sampling_portfolio_v831 as portfolio

    self.base.current = None
    before_level = int(kwargs.get("before_level", 0))
    after_level = int(kwargs.get("after_level", before_level))
    terminal_state = str(kwargs.get("terminal_state", ""))
    success = bool(after_level > before_level or terminal_state == "WIN")
    productive = _productive_transition(kwargs)
    action = int(kwargs.get("action", intervention.action))

    _register_destination(self, kwargs=kwargs, priority=6 if success else (3 if productive else 1))
    if success:
        self.saw_progress = True
        self.base.transfer_action = action
        self.base.transfer_from_level = max(int(self.base.transfer_from_level), before_level)
        self._v833_random_rollout = False
    elif terminal_state == "GAME_OVER" or not tuple(kwargs.get("after_actions", ())):
        self._v833_random_rollout = False
    else:
        self._v833_random_rollout = True
    portfolio._set_mode(None)


def _observe_transfer_v833(self, intervention, **kwargs) -> None:
    from v8 import sampling_portfolio_v831 as portfolio

    self.base.current = None
    before_level = int(kwargs.get("before_level", 0))
    after_level = int(kwargs.get("after_level", before_level))
    terminal_state = str(kwargs.get("terminal_state", ""))
    success = bool(after_level > before_level or terminal_state == "WIN")
    productive = _productive_transition(kwargs)
    action = int(kwargs.get("action", intervention.action))

    _register_destination(self, kwargs=kwargs, priority=6 if success else (4 if productive else 1))
    if success:
        self.saw_progress = True
        self.base.transfer_action = action
        self.base.transfer_from_level = max(int(self.base.transfer_from_level), before_level)
        self._v833_transfer_rollout = False
        self._v833_transfer_origin = ""
    elif (
        terminal_state == "GAME_OVER"
        or not tuple(kwargs.get("after_actions", ()))
        or not productive
    ):
        self._v833_transfer_rollout = False
        self._v833_transfer_origin = ""
    else:
        self._v833_transfer_rollout = True
    portfolio._set_mode(None)


def _observe_transition_v833(self, **kwargs) -> None:
    intervention = self.base.current
    if intervention is not None:
        kind = str(intervention.kind)
        if kind in {"RANDOM", "RANDOM_WALK"}:
            _observe_random_v833(self, intervention, **kwargs)
            return
        if kind == "CROSS_GAME_TRANSFER":
            _observe_transfer_v833(self, intervention, **kwargs)
            return
    return _BASE_OBSERVE_TRANSITION(self, **kwargs)


def install_sampling_transfer_v833() -> None:
    global _INSTALLED
    global _BASE_BEGIN_LEASE, _BASE_ON_EXTERNAL_RESET, _BASE_FORCED_ACTION
    global _BASE_DISCOVERY_ACTION, _BASE_OBSERVE_TRANSITION
    if _INSTALLED:
        return

    from v8 import sampling_portfolio_v831 as portfolio

    cls = portfolio.PortfolioSampler
    _BASE_BEGIN_LEASE = cls.begin_lease
    _BASE_ON_EXTERNAL_RESET = cls.on_external_reset
    _BASE_FORCED_ACTION = cls.forced_action
    _BASE_DISCOVERY_ACTION = cls.discovery_action
    _BASE_OBSERVE_TRANSITION = cls.observe_transition

    cls.begin_lease = _begin_lease_v833
    cls.on_external_reset = _on_external_reset_v833
    cls.forced_action = _forced_action_v833
    cls.discovery_action = _discovery_action_v833
    cls.observe_transition = _observe_transition_v833

    portfolio._COLD_MODES = _COLD_MODES_V833
    portfolio._WARM_MODES = _WARM_MODES_V833
    _INSTALLED = True
