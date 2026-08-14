from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from hashlib import blake2b
from random import Random
from typing import Iterable

import numpy as np

from v7.derivation.scientific import TYPE_ROLE
from v7.memory.concept_validation import ConceptValidationStatus
from v7.memory.developmental_policy import profile_for_view
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.read_view import MemoryReadView

_MASK63 = (1 << 63) - 1


def _hash_context(tag: bytes, *values: int) -> int:
    digest = blake2b(digest_size=8)
    digest.update(tag)
    for value in values:
        digest.update(int(value).to_bytes(8, "little", signed=False))
    return int.from_bytes(digest.digest(), "little") & _MASK63


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _signed_unit(value: float) -> float:
    return math.tanh(float(value))


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """General-to-specific context lattice for one decision state.

    New v7 generations use five signatures:

    C0 general
    C1 behavioral history
    C2 structural state
    C3 behavioral + structural
    C4 exact temporal specialization

    Older four-signature contexts remain readable for durable-run continuity.
    """

    signatures: tuple[int, ...]
    structural_signature: int
    exact_signature: int

    @property
    def behavioral_signature(self) -> int:
        if len(self.signatures) >= 5:
            return int(self.signatures[1])
        return int(self.signatures[0])

    @property
    def combined_signature(self) -> int:
        if len(self.signatures) >= 5:
            return int(self.signatures[3])
        if len(self.signatures) >= 3:
            return int(self.signatures[-2])
        return int(self.signatures[-1])

    @property
    def planning_signature(self) -> int:
        return self.combined_signature

    @property
    def exact_context_signature(self) -> int:
        return int(self.signatures[-1])


@dataclass(frozen=True, slots=True)
class DecisionSupport:
    context_signature: int
    role_ids: tuple[int, ...] = ()
    concept_ids: tuple[int, ...] = ()
    world_model_ids: tuple[int, ...] = ()
    strategy_ids: tuple[int, ...] = ()
    contextual_support: int = 0
    local_support: int = 0
    context_rank: int = 0


@dataclass(frozen=True, slots=True)
class ContextualActionDecision:
    action_id: int
    score: float
    support: DecisionSupport
    exploration_score: float
    failure_risk: float
    contradiction_risk: float
    future_reachability: float
    prediction_confidence: float = 0.0
    completion_likelihood: float = 0.0


@dataclass(slots=True)
class _LocalActionStats:
    outcome_counts: Counter[int]
    count: int = 0
    positive: int = 0
    negative: int = 0
    failures: int = 0
    contradictions: int = 0
    no_change: int = 0
    future_option_sum: float = 0.0

    @classmethod
    def empty(cls) -> "_LocalActionStats":
        return cls(Counter())

    @property
    def failure_risk(self) -> float:
        return 0.0 if self.count <= 0 else self.failures / self.count

    @property
    def contradiction_risk(self) -> float:
        return 0.0 if self.count <= 0 else self.contradictions / self.count

    @property
    def no_change_ratio(self) -> float:
        return 0.0 if self.count <= 0 else self.no_change / self.count

    @property
    def future_option_mean(self) -> float:
        return 0.0 if self.count <= 0 else self.future_option_sum / self.count

    @property
    def prediction_confidence(self) -> float:
        total = sum(self.outcome_counts.values())
        if total <= 0:
            return 0.0
        return max(self.outcome_counts.values()) / total

    @property
    def completion_likelihood(self) -> float:
        terminal = self.positive + self.negative
        return 0.0 if terminal <= 0 else self.positive / terminal

    @property
    def uncertainty(self) -> float:
        total = sum(self.outcome_counts.values())
        if total <= 1:
            return 1.0
        probabilities = [
            count / total for count in self.outcome_counts.values() if count > 0
        ]
        if len(probabilities) <= 1:
            return 0.0
        entropy = -sum(p * math.log(p) for p in probabilities)
        return _clamp01(entropy / math.log(len(probabilities)))


class LocalCognitionOverlay:
    """Worker-local online learning overlay; never mutates canonical memory."""

    def __init__(
        self,
        *,
        history_depth: int = 3,
        prediction_min_support: int = 2,
        stagnation_window: int = 100,
    ) -> None:
        self.history_depth = max(1, int(history_depth))
        self.prediction_min_support = max(1, int(prediction_min_support))
        self.stagnation_window = max(20, int(stagnation_window))
        self.recent_actions: deque[int] = deque(maxlen=self.history_depth)
        self.recent_outcomes: deque[int] = deque(maxlen=self.history_depth)
        self.recent_no_change: deque[int] = deque(maxlen=self.stagnation_window)
        self.recent_states: deque[int] = deque(maxlen=self.stagnation_window)
        self.action_counts: Counter[int] = Counter()
        self.stats: dict[tuple[int, int], _LocalActionStats] = {}
        self.transitions: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
        self.strategy_votes: Counter[tuple[int, int]] = Counter()
        self.strategy_failures: Counter[tuple[int, int]] = Counter()
        self._trajectory: list[tuple[tuple[int, ...], int]] = []
        self._last_contexts: tuple[int, ...] = ()
        self._consecutive_failures = 0

    def build_context(
        self,
        *,
        structural_signature: int,
        exact_signature: int,
    ) -> DecisionContext:
        history_values: list[int] = []
        for action, outcome in zip(self.recent_actions, self.recent_outcomes):
            history_values.extend((int(action), int(outcome)))
        general = _hash_context(b"v7-c0-general")
        behavioral = _hash_context(b"v7-c1-behavior", *history_values)
        structural = _hash_context(b"v7-c2-struct", int(structural_signature))
        combined = _hash_context(
            b"v7-c3-combined",
            int(structural_signature),
            *history_values,
        )
        specific = _hash_context(
            b"v7-c4-specific",
            int(exact_signature),
            *history_values,
        )
        signatures = (general, behavioral, structural, combined, specific)
        self._last_contexts = signatures
        return DecisionContext(signatures, int(structural_signature), int(exact_signature))

    def stats_for(self, context_signature: int, action_id: int) -> _LocalActionStats:
        return self.stats.get(
            (int(context_signature), int(action_id)),
            _LocalActionStats.empty(),
        )

    def prediction_error(
        self,
        contexts: Iterable[int],
        action_id: int,
        outcome_signature: int,
    ) -> float:
        for context in reversed(tuple(int(value) for value in contexts)):
            stats = self.stats.get((context, int(action_id)))
            if stats is None or stats.count < self.prediction_min_support:
                continue
            if not stats.outcome_counts:
                continue
            expected, _ = min(
                stats.outcome_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
            return 0.0 if int(expected) == int(outcome_signature) else 1.0
        return 0.0

    def record_step(
        self,
        *,
        contexts: tuple[int, ...],
        next_contexts: tuple[int, ...],
        action_id: int,
        outcome_signature: int,
        terminal_polarity: int,
        prediction_error: float,
        future_option_delta: float,
        changed: bool,
    ) -> None:
        action = int(action_id)
        outcome = int(outcome_signature)
        self.action_counts[action] += 1
        for context in contexts:
            key = (int(context), action)
            stats = self.stats.setdefault(key, _LocalActionStats.empty())
            stats.count += 1
            stats.outcome_counts[outcome] += 1
            stats.positive += int(terminal_polarity > 0)
            stats.negative += int(terminal_polarity < 0)
            stats.failures += int(terminal_polarity < 0)
            stats.contradictions += int(float(prediction_error) > 0.0)
            stats.no_change += int(not changed)
            stats.future_option_sum += float(future_option_delta)
        if terminal_polarity == 0 and next_contexts:
            for context, target in zip(contexts, next_contexts, strict=False):
                self.transitions[(int(context), action)][int(target)] += 1
        self.recent_actions.append(action)
        self.recent_outcomes.append(outcome)
        self.recent_no_change.append(int(not changed))
        if next_contexts:
            self.recent_states.append(int(next_contexts[-1]))
        planning_contexts = tuple(int(value) for value in contexts)
        self._trajectory.append((planning_contexts, action))
        if terminal_polarity > 0:
            self._promote_local_strategy()
            self._consecutive_failures = 0
            self.reset_episode_history(keep_statistics=True)
        elif terminal_polarity < 0:
            for context_values, prior_action in self._trajectory:
                for context in context_values:
                    self.strategy_failures[(context, prior_action)] += 1
            self._consecutive_failures += 1
            self.reset_episode_history(keep_statistics=True)

    def _promote_local_strategy(self) -> None:
        for contexts, action in self._trajectory:
            for context in contexts:
                self.strategy_votes[(int(context), int(action))] += 1

    def strategy_score(self, contexts: Iterable[int], action_id: int) -> float:
        best = 0.0
        action = int(action_id)
        for context in reversed(tuple(int(value) for value in contexts)):
            success = int(self.strategy_votes[(context, action)])
            failure = int(self.strategy_failures[(context, action)])
            total = success + failure
            if total <= 0:
                continue
            best = max(best, success / total)
        return best

    def exploration_score(self, contexts: Iterable[int], action_id: int) -> float:
        action = int(action_id)
        context_values = tuple(int(value) for value in contexts)
        if not context_values:
            return 1.0
        context = (
            context_values[-2]
            if len(context_values) >= 5
            else context_values[-1]
        )
        stats = self.stats_for(context, action)
        balance = 1.0 / math.sqrt(1.0 + self.action_counts[action])
        no_change_avoidance = max(0.05, 1.0 - stats.no_change_ratio)
        uncertainty = stats.uncertainty
        novelty = 1.0 / math.sqrt(1.0 + stats.count)
        return _clamp01(
            0.20 * balance
            + 0.25 * no_change_avoidance
            + 0.25 * uncertainty
            + 0.30 * novelty
        )

    def future_reachability(
        self,
        contexts: Iterable[int],
        action_id: int,
        *,
        depth: int = 2,
        max_nodes: int = 64,
    ) -> float:
        context_values = tuple(int(value) for value in contexts)
        if not context_values:
            return 0.0
        root = context_values[-2] if len(context_values) >= 2 else context_values[-1]
        first = tuple(self.transitions.get((root, int(action_id)), ()))
        if not first:
            return 0.0
        reached = set(int(value) for value in first)
        frontier = set(reached)
        for _ in range(max(0, int(depth) - 1)):
            next_frontier: set[int] = set()
            for context in sorted(frontier):
                actions = sorted(
                    action
                    for known_context, action in self.transitions
                    if int(known_context) == int(context)
                )
                for action in actions:
                    next_frontier.update(
                        int(value)
                        for value in self.transitions.get((context, action), ())
                    )
                    if len(reached) + len(next_frontier) >= max_nodes:
                        break
                if len(reached) + len(next_frontier) >= max_nodes:
                    break
            next_frontier.difference_update(reached)
            if not next_frontier:
                break
            reached.update(next_frontier)
            frontier = next_frontier
        return _clamp01(
            math.log1p(len(reached)) / math.log1p(max(2, int(max_nodes)))
        )

    def should_reset(self) -> bool:
        if len(self.recent_no_change) >= min(50, self.stagnation_window):
            if sum(self.recent_no_change) / len(self.recent_no_change) > 0.80:
                return True
        if len(self.recent_states) >= min(30, self.stagnation_window):
            counts = Counter(self.recent_states)
            if counts and counts.most_common(1)[0][1] / len(self.recent_states) > 0.80:
                return True
        return self._consecutive_failures >= 3

    def reset_episode_history(self, *, keep_statistics: bool = True) -> None:
        self.recent_actions.clear()
        self.recent_outcomes.clear()
        self.recent_no_change.clear()
        self.recent_states.clear()
        self._trajectory.clear()
        self._last_contexts = ()
        if not keep_statistics:
            self.action_counts.clear()
            self.stats.clear()
            self.transitions.clear()
            self.strategy_votes.clear()
            self.strategy_failures.clear()
            self._consecutive_failures = 0


@dataclass(frozen=True, slots=True)
class _ContextCandidate:
    rank: int
    context: int
    row: object
    local: _LocalActionStats
    support: int
    confidence: float
    contradiction: float
    roles: tuple[int, ...]
    worlds: tuple[int, ...]
    strategies: tuple[int, ...]


class ContextualActionScorer:
    """Context-first M1-M6 scorer with evidence-driven specialization/backoff."""

    def __init__(self, *, minimum_context_support: int = 2) -> None:
        self.minimum_context_support = max(1, int(minimum_context_support))

    def score_actions(
        self,
        *,
        view: MemoryReadView,
        contexts: DecisionContext,
        actions: Iterable[int],
        overlay: LocalCognitionOverlay,
    ) -> tuple[ContextualActionDecision, ...]:
        ordered = tuple(sorted(set(int(value) for value in actions)))
        if not ordered:
            return ()
        rows_by_context = {
            int(context): {
                int(row.action_id): row
                for row in view.score_inputs(
                    context_signature=int(context),
                    action_ids=ordered,
                )
            }
            for context in contexts.signatures
        }
        decisions = [
            self._score_one(
                view=view,
                contexts=contexts,
                action_id=action,
                overlay=overlay,
                rows_by_context=rows_by_context,
            )
            for action in ordered
        ]
        return tuple(sorted(decisions, key=lambda item: item.action_id))

    def _score_one(
        self,
        *,
        view: MemoryReadView,
        contexts: DecisionContext,
        action_id: int,
        overlay: LocalCognitionOverlay,
        rows_by_context,
    ) -> ContextualActionDecision:
        profile = profile_for_view(view)
        candidates: list[_ContextCandidate] = []
        for rank, context in enumerate(contexts.signatures):
            row = rows_by_context[int(context)][action_id]
            roles, worlds, strategies = self._split_decision_memories(
                view, row.role_ids
            )
            m1_support = sum(
                int(view.nodes[memory_id].support_count)
                for memory_id in row.contingency_ids
                if memory_id in view.nodes
            )
            local = overlay.stats_for(int(context), action_id)
            support = (
                m1_support
                + local.count
                + len(roles)
                + len(row.concept_ids)
                + len(worlds)
                + len(strategies)
            )
            m1 = self._m1_features(view, row.contingency_ids)
            confidence = max(m1[0], local.prediction_confidence)
            contradiction = max(m1[4], local.contradiction_risk)
            candidates.append(
                _ContextCandidate(
                    rank=rank,
                    context=int(context),
                    row=row,
                    local=local,
                    support=support,
                    confidence=confidence,
                    contradiction=contradiction,
                    roles=roles,
                    worlds=worlds,
                    strategies=strategies,
                )
            )

        selected = candidates[0]
        for candidate in candidates[1:]:
            if candidate.support < self.minimum_context_support:
                continue
            is_exact = candidate.rank == len(candidates) - 1 and len(candidates) >= 5
            if is_exact:
                exact_ready = (
                    candidate.support >= profile.exact_specialization_min_support
                    or selected.contradiction
                    >= profile.contradiction_specialization_threshold
                )
                if not exact_ready:
                    continue
            if selected.support < self.minimum_context_support:
                selected = candidate
                continue
            resolves_contradiction = (
                selected.contradiction
                >= profile.contradiction_specialization_threshold
                and candidate.contradiction + 0.05 < selected.contradiction
            )
            improves_prediction = candidate.confidence >= selected.confidence + 0.10
            substantially_supported = candidate.support >= max(
                self.minimum_context_support,
                2 * selected.support,
            )
            if resolves_contradiction or improves_prediction or substantially_supported:
                selected = candidate

        selected_row = selected.row
        selected_local = selected.local
        m1_confidence, m1_value, m1_future, m1_failure, m1_contradiction = (
            self._m1_features(view, selected_row.contingency_ids)
        )
        local_confidence = selected_local.prediction_confidence
        local_completion = selected_local.completion_likelihood
        local_future = _signed_unit(selected_local.future_option_mean)
        local_failure = selected_local.failure_risk
        local_contradiction = selected_local.contradiction_risk
        no_change = selected_local.no_change_ratio

        role_strength = self._memory_strength(view, selected.roles)
        concept_strength = self._memory_strength(
            view,
            tuple(int(v) for v in selected_row.concept_ids),
        )
        world_strength = self._memory_strength(view, selected.worlds)
        strategy_strength = max(
            self._memory_strength(view, selected.strategies),
            overlay.strategy_score(contexts.signatures, action_id),
        )
        reachability = overlay.future_reachability(
            contexts.signatures,
            action_id,
            depth=profile.planning_depth,
        )
        exploration = overlay.exploration_score(contexts.signatures, action_id)
        global_prior = self._global_prior(view, action_id)

        prediction = max(m1_confidence, local_confidence)
        future = max(m1_future, local_future, reachability)
        completion = max(m1_value, local_completion)
        failure = max(m1_failure, local_failure)
        contradiction = max(m1_contradiction, local_contradiction)

        score = (
            0.22 * prediction
            + 0.16 * completion
            + 0.16 * max(0.0, future)
            + 0.08 * role_strength
            + 0.08 * concept_strength
            + 0.10 * world_strength
            + 0.12 * strategy_strength
            + 0.08 * exploration
            + 0.04 * global_prior
            - 0.24 * failure
            - 0.10 * contradiction
            - 0.08 * no_change
        )
        support = DecisionSupport(
            context_signature=selected.context,
            role_ids=selected.roles,
            concept_ids=tuple(int(value) for value in selected_row.concept_ids),
            world_model_ids=selected.worlds,
            strategy_ids=selected.strategies,
            contextual_support=selected.support,
            local_support=selected_local.count,
            context_rank=selected.rank,
        )
        return ContextualActionDecision(
            action_id=int(action_id),
            score=float(score),
            support=support,
            exploration_score=exploration,
            failure_risk=failure,
            contradiction_risk=contradiction,
            future_reachability=reachability,
            prediction_confidence=prediction,
            completion_likelihood=completion,
        )

    @staticmethod
    def _split_decision_memories(
        view: MemoryReadView,
        memory_ids: Iterable[MemoryId],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        roles: list[int] = []
        worlds: list[int] = []
        strategies: list[int] = []
        for memory_id in memory_ids:
            node = view.nodes.get(memory_id)
            if node is None:
                continue
            if node.level == MemoryLevel.M3 and int(node.type_id) == TYPE_ROLE:
                roles.append(int(memory_id))
            elif node.level == MemoryLevel.M5:
                worlds.append(int(memory_id))
            elif node.level == MemoryLevel.M6:
                strategies.append(int(memory_id))
        return tuple(roles), tuple(worlds), tuple(strategies)

    @staticmethod
    def _m1_features(
        view: MemoryReadView,
        memory_ids: Iterable[MemoryId],
    ) -> tuple[float, float, float, float, float]:
        rows: list[tuple[int, float, float, float]] = []
        for memory_id in memory_ids:
            node = view.nodes.get(memory_id)
            if node is None or node.level != MemoryLevel.M1:
                continue
            score = view.scores.get(memory_id)
            rows.append(
                (
                    max(1, int(node.support_count)),
                    0.5 if score is None else float(score.significance),
                    0.0 if score is None else float(score.future_option_delta),
                    0.0 if score is None else float(score.prediction_error),
                )
            )
        total = sum(row[0] for row in rows)
        if total <= 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        confidence = max(row[0] for row in rows) / total
        value = (
            sum(
                weight * _clamp01(significance)
                for weight, significance, _, _ in rows
            )
            / total
        )
        future = (
            sum(
                weight * _signed_unit(option_delta)
                for weight, _, option_delta, _ in rows
            )
            / total
        )
        failure = (
            sum(
                weight
                for weight, significance, _, _ in rows
                if significance <= 0.10
            )
            / total
        )
        contradiction = (
            sum(
                weight * _clamp01(error)
                for weight, _, _, error in rows
            )
            / total
        )
        return confidence, value, future, failure, contradiction

    @staticmethod
    def _memory_strength(
        view: MemoryReadView,
        memory_ids: Iterable[int],
    ) -> float:
        best = 0.0
        for raw_memory_id in memory_ids:
            memory_id = MemoryId(int(raw_memory_id))
            node = view.nodes.get(memory_id)
            if node is None:
                continue
            score = view.scores.get(memory_id)
            support = 1.0 - math.exp(
                -max(0, int(node.support_count)) / 3.0
            )
            semantic = 0.0
            if score is not None:
                semantic = max(
                    float(score.significance),
                    float(score.learning_value),
                    float(score.transfer_prior),
                    float(score.explanatory_potential),
                )
            strength = _clamp01(0.5 * support + 0.5 * semantic)
            if node.level == MemoryLevel.M4:
                flags = int(node.status_flags)
                if flags & int(ConceptValidationStatus.TRANSFER_REJECTED):
                    strength = 0.0
                elif flags & int(ConceptValidationStatus.TRUSTED):
                    strength *= 1.0
                elif flags & int(ConceptValidationStatus.TRANSFER_VALIDATED):
                    strength *= 0.85
                elif flags & int(ConceptValidationStatus.TRANSFER_CANDIDATE):
                    strength *= 0.55
                elif flags & int(ConceptValidationStatus.STRUCTURAL_SUPPORTED):
                    strength *= 0.40
                elif flags & int(ConceptValidationStatus.CANDIDATE):
                    strength *= 0.20
                else:
                    strength *= 0.10
            best = max(best, strength)
        return best

    @staticmethod
    def _global_prior(view: MemoryReadView, action_id: int) -> float:
        aggregate = view.packed_cognition.action_aggregates.get(int(action_id))
        evidence = max(
            1,
            int(aggregate.future_option_count),
            int(aggregate.positive_count + aggregate.negative_count),
        )
        completion = aggregate.positive_count / max(
            1,
            aggregate.positive_count + aggregate.negative_count,
        )
        failure = aggregate.failure_count / evidence
        future = _signed_unit(aggregate.future_option_mean)
        return _clamp01(
            0.45 * completion
            + 0.35 * max(0.0, future)
            + 0.20 * (1.0 - failure)
        )


def choose_contextual_action(
    *,
    view: MemoryReadView,
    contexts: DecisionContext,
    actions: Iterable[int],
    overlay: LocalCognitionOverlay,
    rng: Random,
    epsilon: float,
    temperature: float = 0.35,
) -> tuple[int, ContextualActionDecision]:
    scorer = ContextualActionScorer()
    decisions = scorer.score_actions(
        view=view,
        contexts=contexts,
        actions=actions,
        overlay=overlay,
    )
    if not decisions:
        raise ValueError("environment returned no available actions")
    profile = profile_for_view(view)
    effective_epsilon = _clamp01(float(epsilon) * profile.exploration_multiplier)
    if rng.random() < effective_epsilon:
        selected = decisions[rng.randrange(len(decisions))]
        return selected.action_id, selected
    values = np.asarray([decision.score for decision in decisions], dtype=float)
    temp = max(1e-6, float(temperature))
    values = values / temp
    values -= np.max(values)
    weights = np.exp(values)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        selected = min(decisions, key=lambda item: (-item.score, item.action_id))
        return selected.action_id, selected
    probabilities = (weights / total).tolist()
    selected = rng.choices(list(decisions), weights=probabilities, k=1)[0]
    return selected.action_id, selected
