from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum

from v9.memory.identity import MemoryUid
from v9.memory.model import CanonicalNode


class NormalizationState(str, Enum):
    EMPTY = "EMPTY"
    PROVISIONAL = "PROVISIONAL"
    AUTHORITATIVE = "AUTHORITATIVE"
    RECALIBRATING = "RECALIBRATING"


@dataclass(frozen=True, slots=True)
class StructuralDescriptor:
    node_uid: int
    graph_generation: int
    object_version: int
    radius: int
    descriptor_version: int
    estimator_generation: int
    components: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.radius <= 0 or self.radius & (self.radius - 1):
            raise ValueError("radius must be a positive power of two")
        if not self.components or any(not math.isfinite(value) for value in self.components):
            raise ValueError("descriptor components must be finite and non-empty")


@dataclass(slots=True)
class ComponentStatistics:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    reservoir: list[float] = field(default_factory=list)

    def update(self, value: float, *, reservoir_limit: int) -> None:
        if not math.isfinite(value):
            raise ValueError("normalization observation must be finite")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        if len(self.reservoir) < reservoir_limit:
            self.reservoir.append(float(value))

    @property
    def variance(self) -> float:
        return self.m2 / (self.count - 1) if self.count > 1 else 0.0


class ScaleStatistics:
    def __init__(self, *, sample_threshold: int = 8, contingency_threshold: int = 2, reservoir_limit: int = 16, component_limit: int = 64, minimum_generation_span: int = 1, allowed_radii: tuple[int, ...] = (1, 2, 4, 8)) -> None:
        self.sample_threshold = int(sample_threshold)
        self.contingency_threshold = int(contingency_threshold)
        self.reservoir_limit = int(reservoir_limit)
        self.component_limit = int(component_limit)
        self.minimum_generation_span = int(minimum_generation_span)
        self.allowed_radii = tuple(int(value) for value in allowed_radii)
        if not self.allowed_radii:
            raise ValueError("normalization requires preregistered radii")
        self.estimator_generation = 1
        self._stats: dict[tuple[int, int], ComponentStatistics] = {}
        self._nodes: dict[int, set[int]] = {}
        self._stable_contingencies: dict[int, set[int]] = {}
        self._state: dict[int, NormalizationState] = {}
        self._generation_span: dict[int, tuple[int, int]] = {}

    def observe(self, descriptor: StructuralDescriptor, *, stable_contingency_uid: int | None = None, authoritative_evidence: bool = True) -> None:
        if not authoritative_evidence:
            return
        if descriptor.radius not in self.allowed_radii:
            raise ValueError("descriptor radius is not preregistered")
        if len(descriptor.components) > self.component_limit:
            raise ValueError("descriptor component budget exceeded")
        for index, value in enumerate(descriptor.components):
            self._stats.setdefault((descriptor.radius, index), ComponentStatistics()).update(float(value), reservoir_limit=self.reservoir_limit)
        nodes = self._nodes.setdefault(descriptor.radius, set())
        if len(nodes) < self.reservoir_limit:
            nodes.add(descriptor.node_uid)
        if stable_contingency_uid is not None:
            contingencies = self._stable_contingencies.setdefault(descriptor.radius, set())
            if len(contingencies) < self.reservoir_limit:
                contingencies.add(int(stable_contingency_uid))
        first, last = self._generation_span.get(descriptor.radius, (descriptor.graph_generation, descriptor.graph_generation))
        self._generation_span[descriptor.radius] = (min(first, descriptor.graph_generation), max(last, descriptor.graph_generation))
        rows = [row for (radius, _), row in self._stats.items() if radius == descriptor.radius]
        span = self._generation_span[descriptor.radius][1] - self._generation_span[descriptor.radius][0]
        ready = bool(rows) and min(row.count for row in rows) >= self.sample_threshold and len(self._stable_contingencies.get(descriptor.radius, set())) >= self.contingency_threshold and len(self._nodes[descriptor.radius]) >= 2 and span >= self.minimum_generation_span
        self._state[descriptor.radius] = NormalizationState.AUTHORITATIVE if ready else NormalizationState.PROVISIONAL

    def begin_recalibration(self, radius: int) -> None:
        if self.state(radius) is NormalizationState.EMPTY:
            raise ValueError("cannot recalibrate an empty estimator")
        self._state[int(radius)] = NormalizationState.RECALIBRATING
        self.estimator_generation += 1

    def state(self, radius: int) -> NormalizationState:
        return self._state.get(int(radius), NormalizationState.EMPTY)

    def normalize(self, radius: int, components: tuple[float, ...]) -> tuple[float, ...]:
        if self.state(radius) is not NormalizationState.AUTHORITATIVE:
            return tuple(float(value) for value in components)
        output = []
        for index, value in enumerate(components):
            row = self._stats[(int(radius), index)]
            deviation = math.sqrt(max(row.variance, 1e-12))
            output.append((float(value) - row.mean) / deviation)
        return tuple(output)

    def state_dict(self) -> dict[str, object]:
        return {
            "estimator_generation": self.estimator_generation,
            "sample_threshold": self.sample_threshold,
            "contingency_threshold": self.contingency_threshold,
            "reservoir_limit": self.reservoir_limit,
            "component_limit": self.component_limit,
            "minimum_generation_span": self.minimum_generation_span,
            "allowed_radii": list(self.allowed_radii),
            "states": {str(key): value.value for key, value in sorted(self._state.items())},
            "stats": [{"radius": radius, "component": component, **asdict(row)} for (radius, component), row in sorted(self._stats.items())],
            "nodes": {str(key): sorted(value) for key, value in sorted(self._nodes.items())},
            "stable_contingencies": {str(key): sorted(value) for key, value in sorted(self._stable_contingencies.items())},
            "generation_span": {str(key): list(value) for key, value in sorted(self._generation_span.items())},
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "ScaleStatistics":
        result = cls(sample_threshold=int(state["sample_threshold"]), contingency_threshold=int(state["contingency_threshold"]), reservoir_limit=int(state["reservoir_limit"]), component_limit=int(state.get("component_limit", 64)), minimum_generation_span=int(state.get("minimum_generation_span", 1)), allowed_radii=tuple(int(value) for value in state.get("allowed_radii", (1, 2, 4, 8))))
        result.estimator_generation = int(state.get("estimator_generation", 1))
        result._state = {int(key): NormalizationState(str(value)) for key, value in dict(state.get("states", {})).items()}
        for raw in state.get("stats", []):
            result._stats[(int(raw["radius"]), int(raw["component"]))] = ComponentStatistics(int(raw["count"]), float(raw["mean"]), float(raw["m2"]), [float(value) for value in raw.get("reservoir", [])])
        result._nodes = {int(key): {int(value) for value in values} for key, values in dict(state.get("nodes", {})).items()}
        result._stable_contingencies = {int(key): {int(value) for value in values} for key, values in dict(state.get("stable_contingencies", {})).items()}
        result._generation_span = {int(key): (int(values[0]), int(values[1])) for key, values in dict(state.get("generation_span", {})).items()}
        return result


def stable_distribution(scores: tuple[float, ...], beta: float) -> tuple[float, ...]:
    if not scores:
        return ()
    if not math.isfinite(beta) or any(not math.isfinite(value) for value in scores):
        raise ValueError("similarity distribution requires finite values")
    scaled = tuple(beta * value for value in scores)
    top = max(scaled)
    weights = tuple(math.exp(value - top) for value in scaled)
    total = sum(weights)
    return tuple(value / total for value in weights)


def entropy(probabilities: tuple[float, ...]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0)


@dataclass(frozen=True, slots=True)
class ScaleResult:
    radius: int
    candidate_uids: tuple[int, ...]
    scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    entropy: float
    information_gain: float
    top2_margin: float
    compute_cost: int


@dataclass(frozen=True, slots=True)
class StructuralEquivalenceSet:
    candidate_uids: tuple[int, ...]
    radius: int
    entropy: float
    estimator_generation: int


@dataclass(frozen=True, slots=True)
class SimilarityOutcome:
    scales: tuple[ScaleResult, ...]
    winner_uid: int | None = None
    equivalence_set: StructuralEquivalenceSet | None = None


@dataclass(frozen=True, slots=True, order=True)
class StructuralIndexKey:
    level: int
    memory_type: int
    signature_bucket: int


class StructuralCandidateIndex:
    """Incremental bounded discovery for M2-M4; never scans the graph."""

    def __init__(self, *, bucket_capacity: int, bucket_scan_limit: int) -> None:
        if min(bucket_capacity, bucket_scan_limit) <= 0:
            raise ValueError("structural index budgets must be positive")
        self.bucket_capacity = int(bucket_capacity)
        self.bucket_scan_limit = int(bucket_scan_limit)
        self.generation = 0
        self.buckets: dict[StructuralIndexKey, tuple[MemoryUid, ...]] = {}

    @staticmethod
    def keys(node: CanonicalNode) -> tuple[StructuralIndexKey, ...]:
        if not 2 <= int(node.level) <= 4:
            return ()
        signature = node.structural_key[0] if node.structural_key else 0
        return (
            StructuralIndexKey(int(node.level), int(node.memory_type), int(signature) & 0xFF),
            StructuralIndexKey(int(node.level), int(node.memory_type), (int(signature) >> 8) & 0xFF),
        )

    def add(self, node: CanonicalNode) -> None:
        changed = False
        for key in self.keys(node):
            existing = self.buckets.get(key, ())
            if node.uid in existing:
                continue
            self.buckets[key] = tuple(sorted(existing + (node.uid,)))[: self.bucket_capacity]
            changed = True
        if changed:
            self.generation += 1

    def retrieve(self, keys: tuple[StructuralIndexKey, ...], *, limit: int) -> tuple[MemoryUid, ...]:
        if limit <= 0:
            raise ValueError("candidate limit must be positive")
        selected: set[MemoryUid] = set()
        for key in tuple(sorted(set(keys)))[: self.bucket_scan_limit]:
            selected.update(self.buckets.get(key, ()))
            if len(selected) >= limit:
                break
        return tuple(sorted(selected))[:limit]

    def state_dict(self) -> dict[str, object]:
        return {
            "bucket_capacity": self.bucket_capacity,
            "bucket_scan_limit": self.bucket_scan_limit,
            "generation": self.generation,
            "buckets": [
                {"key": [key.level, key.memory_type, key.signature_bucket], "uids": [[uid.hi, uid.lo] for uid in values]}
                for key, values in sorted(self.buckets.items())
            ],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "StructuralCandidateIndex":
        result = cls(bucket_capacity=int(state["bucket_capacity"]), bucket_scan_limit=int(state["bucket_scan_limit"]))
        result.generation = int(state.get("generation", 0))
        for raw in state.get("buckets", []):
            key = StructuralIndexKey(*map(int, raw["key"]))
            result.buckets[key] = tuple(MemoryUid(int(uid[0]), int(uid[1])) for uid in raw["uids"])
        return result


class ProgressiveSimilarity:
    def __init__(self, *, beta_by_radius: dict[int, float], candidate_limit: int, equivalence_limit: int, maximum_radius: int, ambiguity_threshold: float, margin_threshold: float, information_threshold: float, symmetry_patience: int, statistics: ScaleStatistics, equivalence_history_limit: int = 1024) -> None:
        self.beta_by_radius = dict(beta_by_radius)
        self.candidate_limit = int(candidate_limit)
        self.equivalence_limit = int(equivalence_limit)
        self.maximum_radius = int(maximum_radius)
        self.ambiguity_threshold = float(ambiguity_threshold)
        self.margin_threshold = float(margin_threshold)
        self.information_threshold = float(information_threshold)
        self.symmetry_patience = int(symmetry_patience)
        self.statistics = statistics
        self.equivalence_history_limit = int(equivalence_history_limit)
        self.equivalence_sets: list[StructuralEquivalenceSet] = []

    @staticmethod
    def stale(descriptor: StructuralDescriptor, *, object_version: int, estimator_generation: int) -> bool:
        return descriptor.object_version != int(object_version) or descriptor.estimator_generation != int(estimator_generation)

    def search(self, query: dict[int, StructuralDescriptor], candidates: dict[int, dict[int, StructuralDescriptor]], *, compute_budget: int) -> SimilarityOutcome:
        if len(candidates) > self.candidate_limit:
            raise ValueError("candidate discovery must be bounded before progressive scoring")
        active = tuple(sorted(candidates))
        results: list[ScaleResult] = []
        previous_entropy: float | None = None
        low_gain_streak = 0
        cost = 0
        for radius in sorted(value for value in query if value <= self.maximum_radius and value in self.beta_by_radius):
            query_components = self.statistics.normalize(radius, query[radius].components)
            scored: list[tuple[int, float]] = []
            for uid in active:
                row = candidates[uid].get(radius)
                if row is None:
                    continue
                candidate_components = self.statistics.normalize(radius, row.components)
                if len(candidate_components) != len(query_components):
                    continue
                cost += len(query_components)
                if cost > compute_budget:
                    break
                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(query_components, candidate_components)))
                scored.append((uid, 1.0 / (1.0 + distance)))
            if not scored:
                break
            uids = tuple(row[0] for row in scored)
            scores = tuple(row[1] for row in scored)
            probabilities = stable_distribution(scores, self.beta_by_radius[radius])
            current_entropy = entropy(probabilities)
            gain = 0.0 if previous_entropy is None else previous_entropy - current_entropy
            ordered = sorted(probabilities, reverse=True)
            margin = ordered[0] - ordered[1] if len(ordered) > 1 else 1.0
            results.append(ScaleResult(radius, uids, scores, probabilities, current_entropy, gain, margin, cost))
            top = max(probabilities)
            best = tuple(uid for uid, probability in zip(uids, probabilities) if abs(probability - top) <= 1e-12)
            if len(best) == 1 and current_entropy <= self.ambiguity_threshold and margin >= self.margin_threshold:
                return SimilarityOutcome(tuple(results), winner_uid=best[0])
            low_gain_streak = low_gain_streak + 1 if previous_entropy is not None and gain <= self.information_threshold else 0
            if len(best) > 1 and low_gain_streak >= self.symmetry_patience:
                equivalence = StructuralEquivalenceSet(tuple(sorted(best)[: self.equivalence_limit]), radius, current_entropy, self.statistics.estimator_generation)
                self.equivalence_sets.append(equivalence)
                del self.equivalence_sets[:-self.equivalence_history_limit]
                return SimilarityOutcome(tuple(results), equivalence_set=equivalence)
            previous_entropy = current_entropy
            active = uids
            if cost >= compute_budget:
                break
        if not results:
            return SimilarityOutcome(())
        last = results[-1]
        top = max(last.probabilities)
        best = tuple(uid for uid, probability in zip(last.candidate_uids, last.probabilities) if abs(probability - top) <= 1e-12)
        if len(best) == 1:
            return SimilarityOutcome(tuple(results), winner_uid=best[0])
        equivalence = StructuralEquivalenceSet(tuple(sorted(best)[: self.equivalence_limit]), last.radius, last.entropy, self.statistics.estimator_generation)
        self.equivalence_sets.append(equivalence)
        del self.equivalence_sets[:-self.equivalence_history_limit]
        return SimilarityOutcome(tuple(results), equivalence_set=equivalence)

    def state_dict(self) -> dict[str, object]:
        return {"equivalence_history_limit": self.equivalence_history_limit, "equivalence_sets": [asdict(row) for row in self.equivalence_sets]}

    def load_state(self, state: dict[str, object]) -> None:
        self.equivalence_history_limit = int(state.get("equivalence_history_limit", self.equivalence_history_limit))
        self.equivalence_sets = [StructuralEquivalenceSet(tuple(int(value) for value in raw["candidate_uids"]), int(raw["radius"]), float(raw["entropy"]), int(raw["estimator_generation"])) for raw in state.get("equivalence_sets", [])]
