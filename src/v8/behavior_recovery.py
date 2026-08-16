from __future__ import annotations

import math
import os
from dataclasses import replace
from random import Random

from v8.arena import EdgeRecord, NodeRecord
from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
    signed_u64,
    stable_u64,
)
from v8.promotion import EvidenceGatedPromotionEngine as _BasePromotionEngine
from v8.promotion import FormationCandidate
from v8.publication import ActionScore, LiveReadView, PlannedAction


_ACTOR_MODE_ENV = "ARC_AGI3_V8_ACTOR_BEHAVIOR"
_ACTOR_EPSILON_ENV = "ARC_AGI3_V8_ACTOR_EPSILON"
_ACTOR_SEED_ENV = "ARC_AGI3_V8_ACTOR_SEED"
_MIN_CONTROL_ATTEMPTS = 3.0
_MIN_CONTROL_RELIABILITY = 0.50
_MAX_FAILED_PROBE_ATTEMPTS = 6.0
_FAILED_PROBE_RELIABILITY = 0.25
_LINEAGE_RELATIONS = {
    int(RelationType.EXPLAINS),
    int(RelationType.LEADS_TO),
    int(RelationType.CONTEXT_REFINES),
}
_ACTIVE_STATES = {
    int(CognitiveState.ACTIVE),
    int(CognitiveState.VALIDATED),
    int(CognitiveState.REACTIVATED),
}
_PROBE_STATES = _ACTIVE_STATES | {
    int(CognitiveState.CANDIDATE),
    int(CognitiveState.PROBATION),
}

_CURRENT_ACTOR_VIEW: LiveReadView | None = None
_INSTALLED = False


def future_bucket(value: float) -> int:
    return 1 if float(value) > 1e-9 else -1 if float(value) < -1e-9 else 0


def canonical_outcome_key(consequence: NodeRecord) -> tuple[int, int, int]:
    """Return the single v8.2 fine M6 descriptor for a learned consequence.

    The descriptor is terminal-label free and is the authority used by promotion.
    Actor-side outcome recognition resolves observed M1 transitions to these same
    canonical M6 nodes through graph lineage rather than inventing a second key.
    """
    if int(consequence.level) != int(MemoryLevel.M5) or len(consequence.key_parts) < 4:
        raise ValueError("canonical M6 outcome requires an M5 consequence")
    future = int(consequence.key_parts[3])
    consequence_bucket = int(consequence.key_parts[2]) & 0xFFFF
    context_variant = stable_u64(
        int(consequence.key_parts[0]),
        int(consequence.key_parts[1]),
        person=b"v8.2-outcome-context",
    ) & 0xF
    return (future, consequence_bucket, int(context_variant))


def canonical_outcome_uid(consequence: NodeRecord) -> MemoryUid:
    return MemoryUid.from_key(
        MemoryLevel.M6,
        MemoryType.OUTCOME,
        canonical_outcome_key(consequence),
    )


def _parent_map(edges: tuple[EdgeRecord, ...]) -> dict[MemoryUid, set[MemoryUid]]:
    parents: dict[MemoryUid, set[MemoryUid]] = {}
    for edge in edges:
        if int(edge.relation_type) in _LINEAGE_RELATIONS:
            parents.setdefault(edge.source_uid, set()).add(edge.target_uid)
    return parents


def causal_m1_ancestors(
    outcome_uid: MemoryUid,
    *,
    nodes: tuple[NodeRecord, ...],
    edges: tuple[EdgeRecord, ...],
    max_depth: int = 8,
) -> frozenset[MemoryUid]:
    """Return M1 contingencies actually present in an M6 outcome's lineage."""
    by_uid = {row.uid: row for row in nodes}
    parents = _parent_map(edges)
    result: set[MemoryUid] = set()
    frontier = {outcome_uid}
    visited = set(frontier)
    for _depth in range(max(1, int(max_depth))):
        following: set[MemoryUid] = set()
        for current in frontier:
            for parent in parents.get(current, ()):
                row = by_uid.get(parent)
                if row is not None and int(row.level) == int(MemoryLevel.M1):
                    result.add(parent)
                if parent not in visited:
                    visited.add(parent)
                    following.add(parent)
        if not following:
            break
        frontier = following
    return frozenset(result)


class CausalEvidenceGatedPromotionEngine(_BasePromotionEngine):
    """v8.2 formation with canonical outcomes and causal M7 strategy identity."""

    def _canonicalize_m6(
        self,
        candidate: FormationCandidate,
        by_uid: dict[MemoryUid, NodeRecord],
    ) -> FormationCandidate:
        if int(candidate.level) != int(MemoryLevel.M6) or not candidate.parents:
            return candidate
        consequence = by_uid.get(candidate.parents[0])
        if consequence is None or int(consequence.level) != int(MemoryLevel.M5):
            return candidate
        key = canonical_outcome_key(consequence)
        uid = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, key)
        return replace(candidate, uid=uid, key_parts=key)

    def _causal_strategies(
        self,
        nodes: tuple[NodeRecord, ...],
        edges: tuple[EdgeRecord, ...],
        *,
        limit: int,
    ) -> tuple[FormationCandidate, ...]:
        if limit <= 0:
            return ()
        by_uid = {row.uid: row for row in nodes}
        parents = _parent_map(edges)
        stable_m1 = {
            row.uid: row
            for row in nodes
            if int(row.level) == int(MemoryLevel.M1)
            and int(row.memory_type) == int(MemoryType.CONTINGENCY)
            and row.support_count >= self.min_contingency_support
            and len(row.key_parts) >= 4
            and self._admissible(row)
        }
        result: list[FormationCandidate] = []
        seen: set[MemoryUid] = set()
        outcomes = sorted(
            (
                row
                for row in nodes
                if int(row.level) == int(MemoryLevel.M6)
                and int(row.memory_type) == int(MemoryType.OUTCOME)
                and len(row.key_parts) >= 3
                and row.support_count >= 2
                and self._admissible(row)
            ),
            key=lambda row: row.uid,
        )
        for outcome in outcomes:
            outcome_future = int(outcome.key_parts[0])
            frontier = {outcome.uid}
            visited = set(frontier)
            ancestors: set[MemoryUid] = set()
            for _depth in range(8):
                following: set[MemoryUid] = set()
                for current in frontier:
                    for parent in parents.get(current, ()):
                        if parent in stable_m1:
                            ancestors.add(parent)
                        if parent not in visited:
                            visited.add(parent)
                            following.add(parent)
                if not following:
                    break
                frontier = following
            for contingency_uid in sorted(ancestors):
                contingency = stable_m1[contingency_uid]
                if future_bucket(contingency.future_option_delta) != outcome_future:
                    continue
                context_bucket = stable_u64(
                    int(contingency.key_parts[0]), person=b"v8-context"
                )
                key = (
                    int(contingency.key_parts[1]),
                    int(outcome.uid.hi),
                    int(outcome.uid.lo),
                    int(context_bucket),
                )
                uid = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, key)
                if uid in seen:
                    continue
                seen.add(uid)
                support = min(int(contingency.support_count), int(outcome.support_count))
                result.append(
                    FormationCandidate(
                        uid,
                        MemoryLevel.M7,
                        MemoryType.STRATEGY,
                        key,
                        (outcome.uid, contingency.uid),
                        max(1, support),
                        min(1.0, contingency.significance),
                        min(1.0, contingency.learning_value),
                        0.0,
                        1.0,
                        contingency.future_option_delta,
                        int(CognitiveState.PROBATION),
                        int(ValidationState.STRUCTURAL),
                        "strategy_reuse",
                        min(1.0, support / 4.0),
                    )
                )
                if len(result) >= limit:
                    return tuple(result)
        return tuple(result)

    def propose(
        self,
        nodes: tuple[NodeRecord, ...],
        edges: tuple[EdgeRecord, ...],
        *,
        budget: int = 256,
    ) -> tuple[FormationCandidate, ...]:
        limit = max(0, int(budget))
        if limit <= 0:
            return ()
        # Reserve a bounded fraction of each interval for M7 so lower-level churn
        # cannot permanently starve strategy formation.
        strategy_quota = max(1, limit // 4)
        lower_budget = max(1, limit - strategy_quota)
        by_uid = {row.uid: row for row in nodes}
        lower = super().propose(nodes, edges, budget=lower_budget)
        result = [
            self._canonicalize_m6(candidate, by_uid)
            for candidate in lower
            if int(candidate.level) != int(MemoryLevel.M7)
        ]
        remaining = max(0, limit - len(result))
        result.extend(self._causal_strategies(nodes, edges, limit=remaining))
        return tuple(result[:limit])


def _refresh_behavior_indexes(view: LiveReadView) -> None:
    version = tuple(getattr(view, "_strategy_version", ()))
    if version == getattr(view, "_behavior_index_version", None):
        return
    nodes = tuple(getattr(view, "_node_by_uid", {}).values())
    by_uid = {row.uid: row for row in nodes}
    parents = getattr(view, "_parents", {})
    edges = view.edge_records()

    aliases: dict[MemoryUid, set[MemoryUid]] = {}
    members: dict[MemoryUid, set[MemoryUid]] = {}
    direct_dependencies: dict[MemoryUid, set[MemoryUid]] = {}
    for edge in edges:
        relation = int(edge.relation_type)
        if relation == int(RelationType.SUPERSEDES):
            source = by_uid.get(edge.source_uid)
            target = by_uid.get(edge.target_uid)
            if (
                source is not None
                and target is not None
                and int(source.level) == int(MemoryLevel.M6)
                and int(target.level) == int(MemoryLevel.M6)
            ):
                aliases.setdefault(target.uid, set()).add(source.uid)
                members.setdefault(source.uid, set()).add(target.uid)
        elif relation == int(RelationType.DEPENDS_ON):
            target = by_uid.get(edge.target_uid)
            if target is not None and int(target.level) == int(MemoryLevel.M1):
                direct_dependencies.setdefault(edge.source_uid, set()).add(edge.target_uid)

    m1_by_outcome: dict[MemoryUid, set[MemoryUid]] = {}
    observed: dict[tuple[int, int, int], set[MemoryUid]] = {}
    fine_outcomes = [
        row
        for row in nodes
        if int(row.level) == int(MemoryLevel.M6)
        and int(row.memory_type) == int(MemoryType.OUTCOME)
        and len(row.key_parts) >= 3
        and int(row.cognitive_state)
        not in {
            int(CognitiveState.QUARANTINED),
            int(CognitiveState.RETIRE_PENDING),
            int(CognitiveState.RETIRED),
        }
    ]
    for outcome in fine_outcomes:
        frontier = {outcome.uid}
        visited = set(frontier)
        ancestors: set[MemoryUid] = set()
        for _depth in range(8):
            following: set[MemoryUid] = set()
            for current in frontier:
                for parent in parents.get(current, ()):
                    parent_row = by_uid.get(parent)
                    if parent_row is not None and int(parent_row.level) == int(MemoryLevel.M1):
                        ancestors.add(parent)
                    if parent not in visited:
                        visited.add(parent)
                        following.add(parent)
            if not following:
                break
            frontier = following
        m1_by_outcome[outcome.uid] = ancestors
        for uid in ancestors:
            row = by_uid.get(uid)
            if row is None or len(row.key_parts) < 3:
                continue
            key = (
                int(row.key_parts[0]),
                signed_u64(int(row.key_parts[1])),
                int(row.key_parts[2]),
            )
            observed.setdefault(key, set()).add(outcome.uid)

    # Coarse/merged outcomes inherit the exact observable lineage of their members.
    for _ in range(4):
        changed = False
        for member_uid, coarse_uids in tuple(aliases.items()):
            member_ancestors = m1_by_outcome.get(member_uid, set())
            for coarse_uid in coarse_uids:
                target = m1_by_outcome.setdefault(coarse_uid, set())
                before = len(target)
                target.update(member_ancestors)
                changed |= len(target) != before
        if not changed:
            break
    for outcome_uid, ancestors in m1_by_outcome.items():
        for uid in ancestors:
            row = by_uid.get(uid)
            if row is None or len(row.key_parts) < 3:
                continue
            key = (
                int(row.key_parts[0]),
                signed_u64(int(row.key_parts[1])),
                int(row.key_parts[2]),
            )
            observed.setdefault(key, set()).add(outcome_uid)

    # M7 copied to a coarse outcome inherits dependency evidence only from a
    # corresponding causal fine strategy with the same action/context identity.
    strategy_nodes = [
        row
        for row in nodes
        if int(row.level) == int(MemoryLevel.M7)
        and int(row.memory_type) == int(MemoryType.STRATEGY)
        and len(row.key_parts) >= 4
    ]
    fine_strategy: dict[tuple[int, MemoryUid, int], MemoryUid] = {}
    for row in strategy_nodes:
        outcome_uid = MemoryUid(int(row.key_parts[1]), int(row.key_parts[2]))
        fine_strategy[(int(row.key_parts[0]), outcome_uid, int(row.key_parts[3]))] = row.uid
    dependencies = {uid: set(values) for uid, values in direct_dependencies.items()}
    for row in strategy_nodes:
        if dependencies.get(row.uid):
            continue
        coarse_uid = MemoryUid(int(row.key_parts[1]), int(row.key_parts[2]))
        for member_uid in members.get(coarse_uid, ()):
            source_uid = fine_strategy.get(
                (int(row.key_parts[0]), member_uid, int(row.key_parts[3]))
            )
            if source_uid is not None:
                dependencies.setdefault(row.uid, set()).update(
                    direct_dependencies.get(source_uid, ())
                )

    view._behavior_observed_outcomes = observed
    view._behavior_m1_by_outcome = m1_by_outcome
    view._behavior_strategy_dependencies = dependencies
    view._behavior_index_version = version


def _strategy_is_causal(view: LiveReadView, strategy_uid: MemoryUid, outcome_uid: MemoryUid) -> bool:
    dependencies = getattr(view, "_behavior_strategy_dependencies", {}).get(strategy_uid, set())
    ancestors = getattr(view, "_behavior_m1_by_outcome", {}).get(outcome_uid, set())
    return bool(set(dependencies) & set(ancestors))


def strategy_can_control(view: LiveReadView, strategy_uid: MemoryUid, outcome_uid: MemoryUid) -> bool:
    row = getattr(view, "_node_by_uid", {}).get(strategy_uid)
    if row is None or int(row.cognitive_state) not in _ACTIVE_STATES:
        return False
    if not _strategy_is_causal(view, strategy_uid, outcome_uid):
        return False
    if float(row.attempt_weight) < _MIN_CONTROL_ATTEMPTS:
        return False
    return float(row.strategy_reliability) >= _MIN_CONTROL_RELIABILITY


def _strategy_can_probe(view: LiveReadView, strategy_uid: MemoryUid, outcome_uid: MemoryUid) -> bool:
    row = getattr(view, "_node_by_uid", {}).get(strategy_uid)
    if row is None or int(row.cognitive_state) not in _PROBE_STATES:
        return False
    if not _strategy_is_causal(view, strategy_uid, outcome_uid):
        return False
    if (
        float(row.attempt_weight) >= _MAX_FAILED_PROBE_ATTEMPTS
        and float(row.strategy_reliability) < _FAILED_PROBE_RELIABILITY
    ):
        return False
    return not strategy_can_control(view, strategy_uid, outcome_uid)


def _score_strategy_rows(
    view: LiveReadView,
    rows,
    *,
    available: set[int],
    outcome_uid: MemoryUid | None,
    required_ancestor: MemoryUid | None,
    excluded_strategies: frozenset[MemoryUid],
    ignore_preference: bool,
    cross_context: bool,
) -> tuple[PlannedAction, ...]:
    candidates: list[PlannedAction] = []
    for row in rows:
        if row.action_id not in available or row.support <= 0:
            continue
        if row.strategy_uid in excluded_strategies:
            continue
        if outcome_uid is not None and row.outcome_uid != outcome_uid:
            continue
        if required_ancestor is not None and not view.strategy_has_ancestor(
            row.strategy_uid, required_ancestor
        ):
            continue
        preference_influenced = (
            not ignore_preference and row.outcome_uid in view._preferred_outcomes
        )
        preference_bonus = 0.25 if preference_influenced else 0.0
        support_prior = 0.05 * math.log1p(max(0, row.support))
        efficiency = 1.0 / max(1e-9, row.mean_cost)
        transfer_penalty = 0.30 if cross_context else 0.0
        score = (
            row.reliability
            + 0.10 * efficiency
            + support_prior
            + preference_bonus
            - transfer_penalty
        )
        candidates.append(
            PlannedAction(
                row.action_id,
                row.outcome_uid,
                row.strategy_uid,
                float(score),
                preference_influenced,
            )
        )
    candidates.sort(key=lambda item: (-item.score, item.action_id, item.strategy_uid))
    return tuple(candidates)


_ORIGINAL_LIVE_INIT = LiveReadView.__init__
_ORIGINAL_LIVE_CLOSE = LiveReadView.close
_ORIGINAL_REFRESH = LiveReadView._refresh_strategy_cache
_ORIGINAL_SCORE_ACTIONS = LiveReadView.score_actions
_ORIGINAL_OUTCOME_DISTRIBUTION = LiveReadView.outcome_distribution


def _live_init(self: LiveReadView, *args, **kwargs) -> None:
    global _CURRENT_ACTOR_VIEW
    _ORIGINAL_LIVE_INIT(self, *args, **kwargs)
    self._behavior_actor_mode = os.environ.get(_ACTOR_MODE_ENV) == "1"
    self._behavior_epsilon = max(
        0.0, min(1.0, float(os.environ.get(_ACTOR_EPSILON_ENV, "0")))
    )
    self._behavior_rng = Random(int(os.environ.get(_ACTOR_SEED_ENV, "0")) ^ 0x5A17)
    self._behavior_force_random = False
    self._behavior_last_action: tuple[int, int] | None = None
    self._behavior_last_plans: tuple[PlannedAction, ...] = ()
    self._behavior_index_version = None
    self._behavior_observed_outcomes: dict[tuple[int, int, int], set[MemoryUid]] = {}
    self._behavior_m1_by_outcome: dict[MemoryUid, set[MemoryUid]] = {}
    self._behavior_strategy_dependencies: dict[MemoryUid, set[MemoryUid]] = {}
    if self._behavior_actor_mode:
        _CURRENT_ACTOR_VIEW = self


def _live_close(self: LiveReadView) -> None:
    global _CURRENT_ACTOR_VIEW
    if _CURRENT_ACTOR_VIEW is self:
        _CURRENT_ACTOR_VIEW = None
    _ORIGINAL_LIVE_CLOSE(self)


def _refresh(self: LiveReadView) -> None:
    _ORIGINAL_REFRESH(self)
    _refresh_behavior_indexes(self)


def _score_actions(self: LiveReadView, context_signature: int, action_ids) -> tuple[ActionScore, ...]:
    if getattr(self, "_behavior_force_random", False):
        self._behavior_force_random = False
        return tuple(ActionScore(int(action), 0, 0.0, 0) for action in action_ids)
    return _ORIGINAL_SCORE_ACTIONS(self, context_signature, action_ids)


def _outcome_distribution(self: LiveReadView, context_signature: int, action_id: int, **kwargs):
    self._behavior_last_action = (int(context_signature), int(action_id))
    return _ORIGINAL_OUTCOME_DISTRIBUTION(
        self, context_signature, action_id, **kwargs
    )


def _plan_candidates(
    self: LiveReadView,
    context_signature: int,
    action_ids,
    *,
    outcome_uid: MemoryUid | None = None,
    required_ancestor: MemoryUid | None = None,
    excluded_strategies: frozenset[MemoryUid] = frozenset(),
    ignore_preference: bool = False,
) -> tuple[PlannedAction, ...]:
    self._refresh_strategy_cache()
    context_bucket = stable_u64(int(context_signature), person=b"v8-context")
    available = {int(value) for value in action_ids}
    exact = list(self._strategy_by_context.get(context_bucket, ()))
    admitted_exact = [
        row
        for row in exact
        if strategy_can_control(self, row.strategy_uid, row.outcome_uid)
    ]
    control_rows = admitted_exact
    cross_context = False
    if not control_rows:
        control_rows = [
            row
            for row in self._strategy_fallback
            if strategy_can_control(self, row.strategy_uid, row.outcome_uid)
        ]
        cross_context = bool(control_rows)
    control = _score_strategy_rows(
        self,
        control_rows,
        available=available,
        outcome_uid=outcome_uid,
        required_ancestor=required_ancestor,
        excluded_strategies=excluded_strategies,
        ignore_preference=ignore_preference,
        cross_context=cross_context,
    )

    if getattr(self, "_behavior_actor_mode", False):
        rng = self._behavior_rng
        if rng.random() < float(self._behavior_epsilon):
            probe_rows = [
                row
                for row in exact
                if _strategy_can_probe(self, row.strategy_uid, row.outcome_uid)
            ]
            probes = _score_strategy_rows(
                self,
                probe_rows,
                available=available,
                outcome_uid=outcome_uid,
                required_ancestor=required_ancestor,
                excluded_strategies=excluded_strategies,
                ignore_preference=True,
                cross_context=False,
            )
            if probes and rng.random() < 0.5:
                chosen = probes[rng.randrange(min(8, len(probes)))]
                self._behavior_last_plans = (chosen,)
                return (chosen,)
            self._behavior_force_random = True
            self._behavior_last_plans = ()
            return ()

    self._behavior_last_plans = control
    return control


def observed_outcome_uids(
    view: LiveReadView,
    *,
    context_signature: int,
    action_id: int,
    outcome_signature: int,
) -> tuple[MemoryUid, ...]:
    view._refresh_strategy_cache()
    key = (int(context_signature), int(action_id), int(outcome_signature))
    return tuple(sorted(view._behavior_observed_outcomes.get(key, ())))


def _actor_observed_outcome_uids(
    *,
    outcome_signature: int,
    family_signature: int,
    future_delta: float,
    changed_cells: int,
    terminal_polarity: int = 0,
) -> tuple[MemoryUid, MemoryUid]:
    del family_signature, future_delta, changed_cells, terminal_polarity
    view = _CURRENT_ACTOR_VIEW
    if view is None or view._behavior_last_action is None:
        return (MemoryUid.zero(), MemoryUid.zero())
    context, action = view._behavior_last_action
    matches = list(
        observed_outcome_uids(
            view,
            context_signature=context,
            action_id=action,
            outcome_signature=outcome_signature,
        )
    )
    if not matches:
        return (MemoryUid.zero(), MemoryUid.zero())
    preferred = [
        plan.outcome_uid
        for plan in view._behavior_last_plans
        if int(plan.action_id) == int(action) and plan.outcome_uid in matches
    ]
    ordered: list[MemoryUid] = []
    for uid in (*preferred, *matches):
        if uid not in ordered:
            ordered.append(uid)
        if len(ordered) >= 2:
            break
    if len(ordered) == 1:
        ordered.append(ordered[0])
    return (ordered[0], ordered[1])


def _actor_worker_with_behavior(*, job, **kwargs) -> None:
    from v8 import actor as actor_module

    original = _ORIGINAL_ACTOR_WORKER
    prior_mode = os.environ.get(_ACTOR_MODE_ENV)
    prior_epsilon = os.environ.get(_ACTOR_EPSILON_ENV)
    prior_seed = os.environ.get(_ACTOR_SEED_ENV)
    os.environ[_ACTOR_MODE_ENV] = "1"
    os.environ[_ACTOR_EPSILON_ENV] = str(max(0.0, min(1.0, float(job.epsilon))))
    os.environ[_ACTOR_SEED_ENV] = str(int(job.seed))
    try:
        original(job=job, **kwargs)
    finally:
        for key, prior in (
            (_ACTOR_MODE_ENV, prior_mode),
            (_ACTOR_EPSILON_ENV, prior_epsilon),
            (_ACTOR_SEED_ENV, prior_seed),
        ):
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


_ORIGINAL_ACTOR_WORKER = None


def install_behavior_recovery() -> None:
    """Install the v8.2 behavioral corrections on the existing RAM runtime."""
    global _INSTALLED, _ORIGINAL_ACTOR_WORKER
    if _INSTALLED:
        return
    from v8 import actor as actor_module
    from v8 import peers_v82 as peers_v82_module
    from v8 import promotion as promotion_module

    _ORIGINAL_ACTOR_WORKER = actor_module.actor_worker
    LiveReadView.__init__ = _live_init
    LiveReadView.close = _live_close
    LiveReadView._refresh_strategy_cache = _refresh
    LiveReadView.score_actions = _score_actions
    LiveReadView.outcome_distribution = _outcome_distribution
    LiveReadView.plan_candidates = _plan_candidates
    LiveReadView.observed_outcome_uids = observed_outcome_uids
    actor_module._observed_outcome_uids = _actor_observed_outcome_uids
    actor_module.actor_worker = _actor_worker_with_behavior
    promotion_module.EvidenceGatedPromotionEngine = CausalEvidenceGatedPromotionEngine
    peers_v82_module.EvidenceGatedPromotionEngine = CausalEvidenceGatedPromotionEngine
    _INSTALLED = True
