from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Iterable


class NormalizationState(str, Enum):
    EMPTY = "EMPTY"
    PROVISIONAL = "PROVISIONAL"
    AUTHORITATIVE = "AUTHORITATIVE"
    RECALIBRATING = "RECALIBRATING"


@dataclass(frozen=True, slots=True)
class ScaleDescriptor:
    node_uid: int
    graph_generation: int
    object_version: int
    radius: int
    descriptor_version: int
    estimator_generation: int
    components: tuple[float, ...]

    def __post_init__(self) -> None:
        if int(self.radius) <= 0 or int(self.radius) & (int(self.radius) - 1):
            raise ValueError("radius must be a positive power of two")
        if not self.components:
            raise ValueError("descriptor components cannot be empty")
        if any(not math.isfinite(float(value)) for value in self.components):
            raise ValueError("descriptor components must be finite")


@dataclass(slots=True)
class StreamingComponentStats:
    state: NormalizationState = NormalizationState.EMPTY
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    provisional_observations_seen: int = 0
    provisional_samples_retained: int = 0
    provisional_samples_discarded: int = 0
    reservoir: list[float] = field(default_factory=list)

    def update(self, value: float, *, reservoir_limit: int) -> None:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("normalization sample must be finite")
        self.provisional_observations_seen += 1
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        if len(self.reservoir) < int(reservoir_limit):
            self.reservoir.append(value)
            self.provisional_samples_retained += 1
        else:
            self.provisional_samples_discarded += 1
        if self.state is NormalizationState.EMPTY:
            self.state = NormalizationState.PROVISIONAL

    @property
    def variance(self) -> float:
        return 0.0 if self.count < 2 else self.m2 / (self.count - 1)


@dataclass(frozen=True, slots=True)
class BootstrapCriteria:
    sample_count: int = 8
    stable_contingency_count: int = 2
    descriptor_coverage: int = 2
    minimum_observation_span: int = 4


class ScaleStatistics:
    STATE_VERSION = 1

    def __init__(self, *, reservoir_limit: int = 16, criteria: BootstrapCriteria | None = None) -> None:
        self.reservoir_limit = int(reservoir_limit)
        self.criteria = criteria or BootstrapCriteria()
        self.stats: dict[tuple[int, int], StreamingComponentStats] = {}
        self.observation_span: dict[int, tuple[int, int]] = {}
        self.stable_contingencies: dict[int, int] = {}
        self.descriptor_nodes: dict[int, set[int]] = {}

    def update(self, descriptor: ScaleDescriptor, *, stable_contingency: bool = False) -> None:
        radius = int(descriptor.radius)
        for index, value in enumerate(descriptor.components):
            self.stats.setdefault((radius, index), StreamingComponentStats()).update(value, reservoir_limit=self.reservoir_limit)
        low, high = self.observation_span.get(radius, (descriptor.graph_generation, descriptor.graph_generation))
        self.observation_span[radius] = (min(low, descriptor.graph_generation), max(high, descriptor.graph_generation))
        self.descriptor_nodes.setdefault(radius, set()).add(int(descriptor.node_uid))
        if stable_contingency:
            self.stable_contingencies[radius] = int(self.stable_contingencies.get(radius, 0)) + 1
        self._maybe_promote(radius)

    def _maybe_promote(self, radius: int) -> None:
        rows = [row for (r, _), row in self.stats.items() if r == int(radius)]
        if not rows:
            return
        low, high = self.observation_span.get(int(radius), (0, 0))
        ready = min(row.count for row in rows) >= int(self.criteria.sample_count) and int(self.stable_contingencies.get(int(radius), 0)) >= int(self.criteria.stable_contingency_count) and len(self.descriptor_nodes.get(int(radius), set())) >= int(self.criteria.descriptor_coverage) and int(high) - int(low) >= int(self.criteria.minimum_observation_span)
        if ready:
            for row in rows:
                row.state = NormalizationState.AUTHORITATIVE

    def state_for_radius(self, radius: int) -> NormalizationState:
        rows = [row.state for (r, _), row in self.stats.items() if r == int(radius)]
        if not rows:
            return NormalizationState.EMPTY
        return NormalizationState.AUTHORITATIVE if all(state is NormalizationState.AUTHORITATIVE for state in rows) else NormalizationState.PROVISIONAL

    def state_dict(self) -> dict[str, object]:
        return {"version": self.STATE_VERSION, "reservoir_limit": self.reservoir_limit, "criteria": asdict(self.criteria), "stats": [{"radius": r, "component": c, **asdict(row), "state": row.state.value} for (r, c), row in sorted(self.stats.items())], "observation_span": {str(k): list(v) for k, v in sorted(self.observation_span.items())}, "stable_contingencies": {str(k): v for k, v in sorted(self.stable_contingencies.items())}, "descriptor_nodes": {str(k): sorted(v) for k, v in sorted(self.descriptor_nodes.items())}}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "ScaleStatistics":
        if int(state.get("version", 0)) != cls.STATE_VERSION:
            raise ValueError("unsupported scale statistics state")
        criteria_raw = state.get("criteria", {})
        if not isinstance(criteria_raw, dict):
            raise ValueError("invalid bootstrap criteria")
        obj = cls(reservoir_limit=int(state.get("reservoir_limit", 16)), criteria=BootstrapCriteria(**{k: int(v) for k, v in criteria_raw.items()}))
        for raw in state.get("stats", []):
            if not isinstance(raw, dict):
                continue
            row = dict(raw); radius = int(row.pop("radius")); component = int(row.pop("component")); row["state"] = NormalizationState(str(row["state"]))
            obj.stats[(radius, component)] = StreamingComponentStats(**row)
        span = state.get("observation_span", {})
        if isinstance(span, dict): obj.observation_span = {int(k): (int(v[0]), int(v[1])) for k, v in span.items()}
        stable = state.get("stable_contingencies", {})
        if isinstance(stable, dict): obj.stable_contingencies = {int(k): int(v) for k, v in stable.items()}
        nodes = state.get("descriptor_nodes", {})
        if isinstance(nodes, dict): obj.descriptor_nodes = {int(k): {int(x) for x in v} for k, v in nodes.items()}
        return obj


def stable_distribution(scores: Iterable[float], beta: float) -> tuple[float, ...]:
    rows = tuple(float(score) for score in scores)
    if not rows: return ()
    if any(not math.isfinite(score) for score in rows) or not math.isfinite(float(beta)):
        raise ValueError("NaN/Inf invalidates structural comparison")
    if len(rows) == 1: return (1.0,)
    scaled = tuple(float(beta) * score for score in rows); top = max(scaled)
    exps = tuple(math.exp(value - top) for value in scaled); total = sum(exps)
    return tuple(value / total for value in exps)


def entropy(probabilities: Iterable[float]) -> float:
    return -sum(float(p) * math.log(float(p)) for p in probabilities if float(p) > 0.0)


def _similarity(a: ScaleDescriptor, b: ScaleDescriptor) -> float:
    if len(a.components) != len(b.components): raise ValueError("descriptor dimensions differ")
    distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(a.components, b.components)))
    return 1.0 / (1.0 + distance)


@dataclass(frozen=True, slots=True)
class ScaleSearchResult:
    radius: int
    candidate_uids: tuple[int, ...]
    scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    entropy: float
    entropy_delta: float
    top2_margin: float
    compute_cost: int


@dataclass(frozen=True, slots=True)
class StructuralEquivalenceSet:
    candidate_uids: tuple[int, ...]
    radius: int
    entropy: float


@dataclass(frozen=True, slots=True)
class ProgressiveSearchOutcome:
    scales: tuple[ScaleSearchResult, ...]
    winner_uid: int | None
    equivalence_set: StructuralEquivalenceSet | None


class ProgressiveSimilarityEngine:
    STATE_VERSION = 1

    def __init__(self, *, beta_by_radius: dict[int, float] | None = None, r_max: int = 8, symmetry_patience: int = 2, information_epsilon: float = 1e-3) -> None:
        self.beta_by_radius = dict(beta_by_radius or {1: 1.0, 2: 1.0, 4: 1.0, 8: 1.0})
        self.r_max = int(r_max); self.symmetry_patience = int(symmetry_patience); self.information_epsilon = float(information_epsilon)
        self.equivalence_sets: list[StructuralEquivalenceSet] = []

    @staticmethod
    def descriptor_is_stale(descriptor: ScaleDescriptor, *, graph_generation: int, object_version: int, estimator_generation: int) -> bool:
        return bool(descriptor.graph_generation != int(graph_generation) or descriptor.object_version != int(object_version) or descriptor.estimator_generation != int(estimator_generation))

    def search(self, query: dict[int, ScaleDescriptor], candidates: dict[int, dict[int, ScaleDescriptor]], *, compute_budget: int = 10_000) -> ProgressiveSearchOutcome:
        active = tuple(sorted(candidates)); scale_rows: list[ScaleSearchResult] = []; last_entropy: float | None = None; low_information_streak = 0; cost = 0
        for radius in sorted(r for r in query if r <= self.r_max):
            scored: list[tuple[int, float]] = []
            for uid in active:
                descriptor = candidates[uid].get(radius)
                if descriptor is None: continue
                cost += len(query[radius].components)
                if cost > int(compute_budget): break
                scored.append((uid, _similarity(query[radius], descriptor)))
            if not scored: break
            uids = tuple(uid for uid, _ in scored); scores = tuple(score for _, score in scored)
            probabilities = stable_distribution(scores, self.beta_by_radius.get(radius, 1.0)); current_entropy = entropy(probabilities)
            delta = 0.0 if last_entropy is None else last_entropy - current_entropy
            ordered_scores = sorted(scores, reverse=True); margin = ordered_scores[0] - ordered_scores[1] if len(ordered_scores) > 1 else 1.0
            scale_rows.append(ScaleSearchResult(radius, uids, scores, probabilities, current_entropy, delta, margin, cost))
            low_information_streak = low_information_streak + 1 if last_entropy is not None and abs(delta) <= self.information_epsilon else 0
            last_entropy = current_entropy; max_p = max(probabilities); best = tuple(uid for uid, p in zip(uids, probabilities) if abs(p - max_p) <= 1e-12)
            active = best if len(best) > 1 else uids
            if len(best) == 1 and margin > self.information_epsilon: return ProgressiveSearchOutcome(tuple(scale_rows), best[0], None)
            if len(best) > 1 and low_information_streak >= self.symmetry_patience:
                eq = StructuralEquivalenceSet(tuple(sorted(best)), radius, current_entropy); self.equivalence_sets.append(eq); return ProgressiveSearchOutcome(tuple(scale_rows), None, eq)
            if cost >= int(compute_budget): break
        if scale_rows:
            last = scale_rows[-1]; max_p = max(last.probabilities); best = tuple(uid for uid, p in zip(last.candidate_uids, last.probabilities) if abs(p - max_p) <= 1e-12)
            if len(best) == 1: return ProgressiveSearchOutcome(tuple(scale_rows), best[0], None)
            eq = StructuralEquivalenceSet(tuple(sorted(best)), last.radius, last.entropy); self.equivalence_sets.append(eq); return ProgressiveSearchOutcome(tuple(scale_rows), None, eq)
        return ProgressiveSearchOutcome((), None, None)

    def state_dict(self) -> dict[str, object]:
        return {"version": self.STATE_VERSION, "beta_by_radius": {str(k): v for k, v in sorted(self.beta_by_radius.items())}, "r_max": self.r_max, "symmetry_patience": self.symmetry_patience, "information_epsilon": self.information_epsilon, "equivalence_sets": [asdict(row) for row in self.equivalence_sets]}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "ProgressiveSimilarityEngine":
        if int(state.get("version", 0)) != cls.STATE_VERSION: raise ValueError("unsupported progressive similarity state")
        beta = state.get("beta_by_radius", {})
        if not isinstance(beta, dict): raise ValueError("invalid beta map")
        obj = cls(beta_by_radius={int(k): float(v) for k, v in beta.items()}, r_max=int(state.get("r_max", 8)), symmetry_patience=int(state.get("symmetry_patience", 2)), information_epsilon=float(state.get("information_epsilon", 1e-3)))
        rows = state.get("equivalence_sets", [])
        if isinstance(rows, list): obj.equivalence_sets = [StructuralEquivalenceSet(tuple(int(x) for x in raw["candidate_uids"]), int(raw["radius"]), float(raw["entropy"])) for raw in rows if isinstance(raw, dict)]
        return obj
