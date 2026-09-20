from __future__ import annotations

from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field, replace
from enum import Enum
import json
import math
from pathlib import Path
from statistics import median
from threading import RLock
from typing import Any, Iterable, Mapping

from v9.cognition.action_selection import adaptive_epsilon
from v9.environments.schemas import EnvironmentIdentity
from v9.runtime.actor_policy import ActorViabilityPolicy


class ViabilityState(str, Enum):
    PROBING = "PROBING"
    LOW_EVIDENCE = "LOW_EVIDENCE"
    VIABLE = "VIABLE"
    VIABILITY_ANOMALY = "VIABILITY_ANOMALY"
    RECOVERING = "RECOVERING"


_MAX_CONTEXTS = 4096
_MAX_ACTIONS_PER_CONTEXT = 128
_MAX_STATE_SIGNATURES = 4096
_MAX_TERMINAL_OUTCOMES = 256
_MAX_ACTIVE_EPISODES = 256
_HISTORY = 128
_EVALUATION_INTERVAL = 64
_PROBE_MIN_OBSERVATIONS = 32
_PROBE_MIN_EPISODES = 2
_ANOMALY_MIN_EPISODES = 8
_ANOMALY_MIN_COVERAGE = 0.70
_ANOMALY_MIN_CONTEXTS = 4
_ANOMALY_TERMINAL_CONCENTRATION = 0.80
_RECOVERY_CONFIRMATIONS = 2


def _bounded_set_add(rows: OrderedDict[int, None], value: int, limit: int) -> None:
    rows.pop(int(value), None)
    rows[int(value)] = None
    while len(rows) > max(1, int(limit)):
        rows.popitem(last=False)


def _uncertainty(scores: Mapping[int, float] | None) -> float:
    if not scores or len(scores) <= 1:
        return 1.0
    ranked = sorted((float(value) for value in scores.values()), reverse=True)
    margin = max(0.0, min(1.0, ranked[0] - ranked[1]))
    return 1.0 - margin


@dataclass(slots=True)
class EnvironmentViabilityProfile:
    environment_id: int
    identity: tuple[str, str, str, str]
    game_scenario: str
    state: ViabilityState = ViabilityState.PROBING
    observations: int = 0
    completed_episodes: int = 0
    successes: int = 0
    failures: int = 0
    truncations: int = 0
    progress_events: int = 0
    positive_valence_events: int = 0
    changed_transitions: int = 0
    last_watermark: int = 0
    last_evaluation_observations: int = 0
    recovery_evidence: int = 0
    viability_confidence: float = 0.25
    policy_uncertainty: float = 1.0
    learning_progress: float = 0.0
    effective_exploration_rate: float = 0.0
    anomaly_reasons: tuple[str, ...] = ()
    branching_history: deque[int] = field(default_factory=lambda: deque(maxlen=_HISTORY))
    episode_lengths: deque[int] = field(default_factory=lambda: deque(maxlen=_HISTORY))
    behavior_history: deque[float] = field(default_factory=lambda: deque(maxlen=_HISTORY))
    active_episode_steps: OrderedDict[int, int] = field(default_factory=OrderedDict)
    contexts: OrderedDict[int, None] = field(default_factory=OrderedDict)
    state_signatures: OrderedDict[int, None] = field(default_factory=OrderedDict)
    context_action_counts: OrderedDict[int, dict[int, int]] = field(default_factory=OrderedDict)
    context_branching: OrderedDict[int, int] = field(default_factory=OrderedDict)
    context_action_outcomes: OrderedDict[int, dict[int, OrderedDict[int, None]]] = field(default_factory=OrderedDict)
    terminal_outcomes: Counter[int] = field(default_factory=Counter)

    def _context_bucket(self, context: int) -> dict[int, int]:
        context = int(context)
        bucket = self.context_action_counts.pop(context, None)
        if bucket is None:
            bucket = {}
        self.context_action_counts[context] = bucket
        while len(self.context_action_counts) > _MAX_CONTEXTS:
            old, _ = self.context_action_counts.popitem(last=False)
            self.context_branching.pop(old, None)
            self.context_action_outcomes.pop(old, None)
            self.contexts.pop(old, None)
        return bucket

    def _record_action_outcome(self, context: int, action: int, after_signature: int) -> None:
        context = int(context)
        actions = self.context_action_outcomes.pop(context, None)
        if actions is None:
            actions = {}
        self.context_action_outcomes[context] = actions
        outcomes = actions.setdefault(int(action), OrderedDict())
        _bounded_set_add(outcomes, int(after_signature), 16)
        if len(actions) > _MAX_ACTIONS_PER_CONTEXT:
            for stale in sorted(actions, key=lambda key: sum(actions[key].values()) if isinstance(actions[key], Counter) else len(actions[key]))[: len(actions) - _MAX_ACTIONS_PER_CONTEXT]:
                actions.pop(stale, None)

    @property
    def mean_branching_factor(self) -> float:
        return sum(self.branching_history) / max(1, len(self.branching_history))

    @property
    def max_branching_factor(self) -> int:
        return max(self.branching_history, default=0)

    @property
    def median_episode_length(self) -> float:
        return float(median(self.episode_lengths)) if self.episode_lengths else 0.0

    @property
    def context_diversity(self) -> int:
        return len(self.contexts)

    @property
    def state_diversity(self) -> int:
        return len(self.state_signatures)

    @property
    def action_coverage(self) -> float:
        if not self.context_action_counts:
            return 0.0
        weighted = 0.0
        weight = 0.0
        for context, actions in self.context_action_counts.items():
            recurrence = max(1, sum(int(value) for value in actions.values()))
            observed_legal = max(len(actions), int(self.context_branching.get(context, len(actions) or 1)))
            coverage = min(1.0, len(actions) / max(1, observed_legal))
            local_weight = 1.0 + math.log2(1.0 + recurrence)
            weighted += coverage * local_weight
            weight += local_weight
        return weighted / max(1e-9, weight)

    @property
    def action_influence_score(self) -> float:
        comparable = 0
        influential = 0
        for actions in self.context_action_outcomes.values():
            if len(actions) < 2:
                continue
            comparable += 1
            outcome_sets = {tuple(rows.keys()) for rows in actions.values()}
            influential += int(len(outcome_sets) > 1)
        direct = self.changed_transitions / max(1, self.observations)
        comparative = influential / max(1, comparable)
        return max(0.0, min(1.0, 0.5 * direct + 0.5 * comparative))

    @property
    def terminal_outcome_diversity(self) -> int:
        return len(self.terminal_outcomes)

    @property
    def terminal_outcome_concentration(self) -> float:
        total = sum(self.terminal_outcomes.values())
        return max(self.terminal_outcomes.values(), default=0) / max(1, total)

    @property
    def reachability_observed(self) -> bool:
        return bool(self.successes or self.progress_events or self.positive_valence_events)

    @property
    def stagnation(self) -> float:
        if self.reachability_observed and self.learning_progress > 0.0:
            return 0.0
        coverage = self.action_coverage
        return max(0.0, min(1.0, coverage * (1.0 - max(0.0, self.learning_progress))))

    def _update_learning_progress(self) -> None:
        rows = tuple(self.behavior_history)
        if len(rows) < 8:
            self.learning_progress = 0.0
            return
        half = len(rows) // 2
        previous = sum(rows[:half]) / max(1, half)
        recent = sum(rows[half:]) / max(1, len(rows) - half)
        self.learning_progress = max(-1.0, min(1.0, recent - previous))

    def observe(
        self,
        transition: Any,
        *,
        watermark: int,
        policy_scores: Mapping[int, float] | None = None,
    ) -> tuple[ViabilityState, ViabilityState, bool]:
        previous_state = self.state
        self.observations += 1
        self.last_watermark = max(self.last_watermark, int(watermark))
        branching = max(1, int(getattr(transition, "available_actions_after", 1) or 1))
        self.branching_history.append(branching)
        self.policy_uncertainty = _uncertainty(policy_scores)

        context = int(getattr(transition, "before_signature", 0))
        action = int(getattr(transition, "action_id", 0))
        after = int(getattr(transition, "after_signature", context))
        _bounded_set_add(self.contexts, context, _MAX_CONTEXTS)
        _bounded_set_add(self.state_signatures, context, _MAX_STATE_SIGNATURES)
        _bounded_set_add(self.state_signatures, after, _MAX_STATE_SIGNATURES)
        bucket = self._context_bucket(context)
        bucket[action] = int(bucket.get(action, 0)) + 1
        if len(bucket) > _MAX_ACTIONS_PER_CONTEXT:
            for stale in sorted(bucket, key=lambda key: (bucket[key], key))[: len(bucket) - _MAX_ACTIONS_PER_CONTEXT]:
                bucket.pop(stale, None)
        self.context_branching[context] = max(int(self.context_branching.get(context, 0)), branching, len(bucket))
        self._record_action_outcome(context, action, after)
        self.changed_transitions += int(context != after)

        episode = int(getattr(transition, "episode_id", 0))
        steps = int(self.active_episode_steps.pop(episode, 0)) + 1
        self.active_episode_steps[episode] = steps
        while len(self.active_episode_steps) > _MAX_ACTIVE_EPISODES:
            self.active_episode_steps.popitem(last=False)

        success = bool(getattr(transition, "task_success", False))
        failure = bool(getattr(transition, "task_failure", False))
        truncated = bool(getattr(transition, "task_truncated", False))
        levels_completed = int(getattr(transition, "levels_completed", 0) or 0)
        valence = int(getattr(transition, "primary_valence", 0) or 0)
        progress = success or levels_completed > 0 or valence > 0
        self.successes += int(success)
        self.failures += int(failure)
        self.truncations += int(truncated)
        self.progress_events += int(progress)
        self.positive_valence_events += int(valence > 0)
        self.behavior_history.append(1.0 if progress else (-0.25 if failure else 0.0))
        self._update_learning_progress()

        terminal = success or failure or truncated
        if terminal:
            self.completed_episodes += 1
            self.episode_lengths.append(self.active_episode_steps.pop(episode, steps))
            terminal_signature = hash((after, valence, success, failure, truncated, levels_completed)) & ((1 << 63) - 1)
            self.terminal_outcomes[terminal_signature] += 1
            if len(self.terminal_outcomes) > _MAX_TERMINAL_OUTCOMES:
                for key, _count in self.terminal_outcomes.most_common()[_MAX_TERMINAL_OUTCOMES:]:
                    self.terminal_outcomes.pop(key, None)

        due = (
            previous_state is ViabilityState.PROBING
            or terminal
            or self.observations - self.last_evaluation_observations >= _EVALUATION_INTERVAL
        )
        changed = False
        if due:
            self.last_evaluation_observations = self.observations
            self._evaluate()
            changed = self.state is not previous_state
        return previous_state, self.state, changed

    def _evaluate(self) -> None:
        coverage = self.action_coverage
        evidence = min(1.0, self.observations / 128.0) * 0.30
        evidence += min(1.0, self.completed_episodes / 8.0) * 0.25
        evidence += coverage * 0.25
        evidence += min(1.0, self.context_diversity / 16.0) * 0.10
        evidence += min(1.0, len(self.branching_history) / 64.0) * 0.10
        evidence = max(0.0, min(1.0, evidence))

        if self.reachability_observed:
            self.anomaly_reasons = ()
            if self.state in {ViabilityState.VIABILITY_ANOMALY, ViabilityState.RECOVERING}:
                self.recovery_evidence += 1
                self.state = ViabilityState.VIABLE if self.recovery_evidence >= _RECOVERY_CONFIRMATIONS else ViabilityState.RECOVERING
            else:
                self.recovery_evidence = _RECOVERY_CONFIRMATIONS
                self.state = ViabilityState.VIABLE
            self.viability_confidence = max(0.75, evidence)
        elif self.observations < _PROBE_MIN_OBSERVATIONS or self.completed_episodes < _PROBE_MIN_EPISODES:
            self.state = ViabilityState.PROBING
            self.recovery_evidence = 0
            self.anomaly_reasons = ()
            self.viability_confidence = min(0.45, 0.20 + 0.25 * evidence)
        else:
            reasons: list[str] = []
            if self.completed_episodes >= _ANOMALY_MIN_EPISODES:
                reasons.append("many_complete_episodes")
            if coverage >= _ANOMALY_MIN_COVERAGE:
                reasons.append("substantial_action_coverage")
            if self.state_diversity >= _ANOMALY_MIN_CONTEXTS:
                reasons.append("intermediate_state_diversity")
            if self.successes == 0 and self.progress_events == 0:
                reasons.append("no_progress_or_success")
            if self.terminal_outcomes and self.terminal_outcome_concentration >= _ANOMALY_TERMINAL_CONCENTRATION:
                reasons.append("concentrated_terminal_outcomes")
            if self.learning_progress <= 0.02:
                reasons.append("negligible_behavioral_improvement")
            anomaly = {
                "many_complete_episodes",
                "substantial_action_coverage",
                "intermediate_state_diversity",
                "no_progress_or_success",
                "concentrated_terminal_outcomes",
                "negligible_behavioral_improvement",
            }.issubset(reasons)
            self.state = ViabilityState.VIABILITY_ANOMALY if anomaly else ViabilityState.LOW_EVIDENCE
            self.recovery_evidence = 0
            self.anomaly_reasons = tuple(reasons) if anomaly else ()
            self.viability_confidence = (
                max(0.10, min(0.25, 0.25 * (1.0 - evidence)))
                if anomaly
                else max(0.25, min(0.60, 0.25 + 0.35 * evidence))
            )

        self.effective_exploration_rate = adaptive_epsilon(
            0.10,
            max(1, round(self.mean_branching_factor)),
            coverage=self.action_coverage,
            uncertainty=self.policy_uncertainty,
            stagnation=self.stagnation,
        )

    def evidence_confidence_for(self, transition: Any) -> float:
        positive = bool(getattr(transition, "task_success", False))
        positive = positive or int(getattr(transition, "levels_completed", 0) or 0) > 0
        positive = positive or int(getattr(transition, "primary_valence", 0) or 0) > 0
        return 1.0 if positive else float(self.viability_confidence)

    def actor_policy(self) -> ActorViabilityPolicy:
        return ActorViabilityPolicy(
            int(self.environment_id),
            self.state.value,
            float(self.viability_confidence),
            float(self.mean_branching_factor),
            float(self.action_coverage),
            float(self.policy_uncertainty),
            float(self.learning_progress),
            float(self.effective_exploration_rate),
            tuple(self.anomaly_reasons),
        )

    def metrics(self) -> dict[str, Any]:
        return {
            "environment_id": int(self.environment_id),
            "game_scenario": self.game_scenario,
            "viability_state": self.state.value,
            "viability_confidence": float(self.viability_confidence),
            "complete_episodes": int(self.completed_episodes),
            "median_episode_length": float(self.median_episode_length),
            "mean_branching_factor": float(self.mean_branching_factor),
            "maximum_branching_factor": int(self.max_branching_factor),
            "action_coverage": float(self.action_coverage),
            "state_context_diversity": int(self.state_diversity),
            "action_influence_score": float(self.action_influence_score),
            "progress_event_count": int(self.progress_events),
            "successes": int(self.successes),
            "terminal_outcome_diversity": int(self.terminal_outcome_diversity),
            "policy_uncertainty": float(self.policy_uncertainty),
            "effective_exploration_rate": float(self.effective_exploration_rate),
            "behavioral_improvement": float(self.learning_progress),
            "anomaly_reasons": list(self.anomaly_reasons),
            "observations": int(self.observations),
            "last_watermark": int(self.last_watermark),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            **self.metrics(),
            "identity": list(self.identity),
            "failures": self.failures,
            "truncations": self.truncations,
            "positive_valence_events": self.positive_valence_events,
            "changed_transitions": self.changed_transitions,
            "last_evaluation_observations": self.last_evaluation_observations,
            "recovery_evidence": self.recovery_evidence,
            "branching_history": list(self.branching_history),
            "episode_lengths": list(self.episode_lengths),
            "behavior_history": list(self.behavior_history),
            "active_episode_steps": [[key, value] for key, value in self.active_episode_steps.items()],
            "contexts": list(self.contexts.keys()),
            "state_signatures": list(self.state_signatures.keys()),
            "context_action_counts": [[context, [[action, count] for action, count in actions.items()]] for context, actions in self.context_action_counts.items()],
            "context_branching": [[context, value] for context, value in self.context_branching.items()],
            "context_action_outcomes": [
                [context, [[action, list(outcomes.keys())] for action, outcomes in actions.items()]]
                for context, actions in self.context_action_outcomes.items()
            ],
            "terminal_outcomes": [[key, value] for key, value in self.terminal_outcomes.items()],
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "EnvironmentViabilityProfile":
        row = cls(
            int(state["environment_id"]),
            tuple(str(value) for value in state.get("identity", ("", "", "", ""))),
            str(state.get("game_scenario", "")),
            state=ViabilityState(str(state.get("viability_state", "PROBING"))),
            observations=int(state.get("observations", 0)),
            completed_episodes=int(state.get("complete_episodes", 0)),
            successes=int(state.get("successes", 0)),
            failures=int(state.get("failures", 0)),
            truncations=int(state.get("truncations", 0)),
            progress_events=int(state.get("progress_event_count", 0)),
            positive_valence_events=int(state.get("positive_valence_events", 0)),
            changed_transitions=int(state.get("changed_transitions", 0)),
            last_watermark=int(state.get("last_watermark", 0)),
            last_evaluation_observations=int(state.get("last_evaluation_observations", 0)),
            recovery_evidence=int(state.get("recovery_evidence", 0)),
            viability_confidence=float(state.get("viability_confidence", 0.25)),
            policy_uncertainty=float(state.get("policy_uncertainty", 1.0)),
            learning_progress=float(state.get("behavioral_improvement", 0.0)),
            effective_exploration_rate=float(state.get("effective_exploration_rate", 0.0)),
            anomaly_reasons=tuple(str(value) for value in state.get("anomaly_reasons", ())),
        )
        row.branching_history.extend(int(value) for value in state.get("branching_history", ()))
        row.episode_lengths.extend(int(value) for value in state.get("episode_lengths", ()))
        row.behavior_history.extend(float(value) for value in state.get("behavior_history", ()))
        row.active_episode_steps = OrderedDict((int(key), int(value)) for key, value in state.get("active_episode_steps", ()))
        row.contexts = OrderedDict((int(value), None) for value in state.get("contexts", ()))
        row.state_signatures = OrderedDict((int(value), None) for value in state.get("state_signatures", ()))
        row.context_action_counts = OrderedDict(
            (int(context), {int(action): int(count) for action, count in actions})
            for context, actions in state.get("context_action_counts", ())
        )
        row.context_branching = OrderedDict((int(context), int(value)) for context, value in state.get("context_branching", ()))
        row.context_action_outcomes = OrderedDict(
            (
                int(context),
                {
                    int(action): OrderedDict((int(outcome), None) for outcome in outcomes)
                    for action, outcomes in actions
                },
            )
            for context, actions in state.get("context_action_outcomes", ())
        )
        row.terminal_outcomes = Counter({int(key): int(value) for key, value in state.get("terminal_outcomes", ())})
        return row


class EnvironmentViabilityController:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.log_path = self.root / "environment_viability.log"
        self.profiles: dict[int, EnvironmentViabilityProfile] = {}
        self._lock = RLock()

    def _profile(self, transition: Any) -> EnvironmentViabilityProfile:
        identity_tuple = tuple(str(value) for value in transition.environment_identity)
        identity = EnvironmentIdentity(*identity_tuple)
        environment_id = int(identity.instance_id.value)
        profile = self.profiles.get(environment_id)
        if profile is None:
            profile = EnvironmentViabilityProfile(environment_id, identity_tuple, str(getattr(transition, "game_scenario", identity.environment_type)))
            self.profiles[environment_id] = profile
        return profile

    def observe(self, transition: Any, *, watermark: int, policy_scores: Mapping[int, float] | None = None, epoch: int = 0) -> EnvironmentViabilityProfile:
        with self._lock:
            profile = self._profile(transition)
            before, after, changed = profile.observe(transition, watermark=watermark, policy_scores=policy_scores)
            if changed or profile.observations % _EVALUATION_INTERVAL == 0:
                self._append_log(profile, epoch=epoch, previous_state=before.value if changed else after.value)
            return profile

    def _append_log(self, profile: EnvironmentViabilityProfile, *, epoch: int, previous_state: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": int(epoch),
            "game_environment": profile.game_scenario,
            "previous_state": str(previous_state),
            **profile.metrics(),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")

    def actor_policies(self) -> dict[int, ActorViabilityPolicy]:
        with self._lock:
            return {environment_id: profile.actor_policy() for environment_id, profile in self.profiles.items()}

    def confidence_by_environment(self) -> dict[int, float]:
        with self._lock:
            return {environment_id: float(profile.viability_confidence) for environment_id, profile in self.profiles.items()}

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            rows = tuple(self.profiles.values())
            states = Counter(profile.state.value for profile in rows)
            return {
                **{f"viability_{state.value.lower()}_environments": int(states.get(state.value, 0)) for state in ViabilityState},
                "viability_environment_count": len(rows),
                "viability_anomalies": int(states.get(ViabilityState.VIABILITY_ANOMALY.value, 0)),
                "viability_low_evidence_environments": int(states.get(ViabilityState.LOW_EVIDENCE.value, 0)),
                "viability_mean_confidence": sum(profile.viability_confidence for profile in rows) / max(1, len(rows)),
                "viability_min_confidence": min((profile.viability_confidence for profile in rows), default=1.0),
                "viability_mean_action_coverage": sum(profile.action_coverage for profile in rows) / max(1, len(rows)),
                "viability_mean_branching_factor": sum(profile.mean_branching_factor for profile in rows) / max(1, len(rows)),
                "viability_mean_policy_uncertainty": sum(profile.policy_uncertainty for profile in rows) / max(1, len(rows)),
                "viability_mean_effective_exploration_rate": sum(profile.effective_exploration_rate for profile in rows) / max(1, len(rows)),
            }

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {"schema_version": 1, "profiles": [profile.state_dict() for _, profile in sorted(self.profiles.items())]}

    def load_state(self, state: Mapping[str, Any]) -> None:
        with self._lock:
            self.profiles = {
                profile.environment_id: profile
                for profile in (EnvironmentViabilityProfile.from_state_dict(row) for row in state.get("profiles", ()))
            }


def install_environment_viability(runtime_cls: type, pipeline_cls: type) -> None:
    if getattr(runtime_cls, "_environment_viability_installed", False):
        return

    original_init = runtime_cls.__init__
    original_state_dict = runtime_cls.state_dict
    original_restore = runtime_cls._restore
    original_actor_policy_snapshot = runtime_cls.actor_policy_snapshot
    original_apply_batch = runtime_cls.apply_prepared_ingestion_batch
    original_metrics = runtime_cls.metrics
    original_pipeline_dispatch = pipeline_cls.dispatch_transition
    original_pipeline_dispatch_batch = pipeline_cls.dispatch_transitions_batch

    def runtime_init(self: Any, config: Any) -> None:
        self._environment_viability = EnvironmentViabilityController(config.root)
        self._environment_evidence_confidence = {}
        original_init(self, config)
        self._environment_evidence_confidence = self._environment_viability.confidence_by_environment()

    def restore(self: Any, snapshot: dict[str, Any], *, graph_override: Any | None = None) -> None:
        original_restore(self, snapshot, graph_override=graph_override)
        state = dict(snapshot.get("state", {}))
        controller = getattr(self, "_environment_viability", None)
        if controller is None:
            controller = EnvironmentViabilityController(self.root)
            self._environment_viability = controller
        controller.load_state(dict(state.get("environment_viability", {})))
        self._environment_evidence_confidence = controller.confidence_by_environment()

    def state_dict(self: Any) -> dict[str, Any]:
        state = original_state_dict(self)
        state["environment_viability"] = self._environment_viability.state_dict()
        return state

    def observe_environment_transition(self: Any, transition: Any, *, watermark: int | None = None) -> ActorViabilityPolicy:
        identity = EnvironmentIdentity(*tuple(str(value) for value in transition.environment_identity))
        environment_id = int(identity.instance_id.value)
        policy_scores = dict(getattr(self, "_hgt_action_scores", {}).get(environment_id, {}))
        epoch = int(getattr(getattr(self, "unified_telemetry", None), "gauges", {}).get("epoch", 0) or 0)
        profile = self._environment_viability.observe(
            transition,
            watermark=int(self.watermark + 1 if watermark is None else watermark),
            policy_scores=policy_scores,
            epoch=epoch,
        )
        previous = float(self._environment_evidence_confidence.get(environment_id, -1.0))
        self._environment_evidence_confidence[environment_id] = float(profile.viability_confidence)
        if abs(previous - float(profile.viability_confidence)) > 1e-9:
            self._actor_policy_generation += 1
        return profile.actor_policy()

    def actor_policy_snapshot(self: Any):
        snapshot = original_actor_policy_snapshot(self)
        return replace(snapshot, viability_by_environment=self._environment_viability.actor_policies())

    def apply_prepared_ingestion_batch(self: Any, rows: Iterable[Any]):
        prepared = tuple(rows)
        for row in prepared:
            self.observe_environment_transition(row.transition, watermark=int(self.watermark) + 1)
        return original_apply_batch(self, prepared)

    def metrics(self: Any) -> dict[str, Any]:
        result = dict(original_metrics(self))
        result.update(self._environment_viability.metrics())
        primary = result.get("primary_dashboard")
        if isinstance(primary, dict):
            primary.update({
                "viability_anomalies": result.get("viability_anomalies", 0),
                "viability_low_evidence_environments": result.get("viability_low_evidence_environments", 0),
                "viability_mean_action_coverage": result.get("viability_mean_action_coverage", 0.0),
                "viability_mean_effective_exploration_rate": result.get("viability_mean_effective_exploration_rate", 0.0),
            })
        return result

    def pipeline_dispatch(self: Any, transition: Any) -> None:
        observe = getattr(self.runtime, "observe_environment_transition", None)
        if callable(observe):
            observe(transition, watermark=int(self.watermark_cursor) + 1)
        return original_pipeline_dispatch(self, transition)

    def pipeline_dispatch_batch(
        self: Any,
        transitions: Iterable[Any],
        *,
        carried_bytes: int | None = None,
    ) -> int:
        rows = tuple(transitions)
        if not rows:
            return 0
        if carried_bytes is None:
            from v9.runtime.shared_batch_transport import encode_transport_rows

            measured_bytes = len(encode_transport_rows(rows))
        else:
            measured_bytes = int(carried_bytes)
        # Admission must precede viability mutation so a rejected publication
        # cannot leave actor-visible developmental evidence behind.
        self._check_ingest_admission(len(rows), measured_bytes)
        observe = getattr(self.runtime, "observe_environment_transition", None)
        if callable(observe):
            scientific = self.runtime.config.scientific
            grounded = bool(scientific.symbolic_grounding_enabled)
            symbol_limit = min(
                int(scientific.symbol_budget_per_window),
                int(scientific.max_symbol_facts_per_window),
            )
            cursor = int(self.watermark_cursor)
            for transition in rows:
                cursor += 1
                observe(transition, watermark=cursor)
                if grounded:
                    cursor += min(len(tuple(getattr(transition, "symbols", ()))), symbol_limit)
        return int(
            original_pipeline_dispatch_batch(
                self,
                rows,
                carried_bytes=measured_bytes,
            )
        )

    runtime_cls.__init__ = runtime_init
    runtime_cls._restore = restore
    runtime_cls.state_dict = state_dict
    runtime_cls.observe_environment_transition = observe_environment_transition
    runtime_cls.actor_policy_snapshot = actor_policy_snapshot
    runtime_cls.apply_prepared_ingestion_batch = apply_prepared_ingestion_batch
    runtime_cls.metrics = metrics
    pipeline_cls.dispatch_transition = pipeline_dispatch
    pipeline_cls.dispatch_transitions_batch = pipeline_dispatch_batch
    runtime_cls._environment_viability_installed = True
