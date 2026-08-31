from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from v8.model import MemoryUid, stable_u64


@dataclass(frozen=True, slots=True)
class StructuralDescriptor:
    uid: MemoryUid
    graph_generation: int
    node_version: int
    radius: int
    descriptor_version: int
    estimator_generation: int
    components: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CandidateScore:
    uid: MemoryUid
    score: float


@dataclass(frozen=True, slots=True)
class RadiusTelemetry:
    radius: int
    candidate_count: int
    entropy: float
    entropy_delta: float
    top2_margin: float
    compute_cost: int


@dataclass(frozen=True, slots=True)
class StructuralEquivalenceSet:
    equivalence_uid: int
    source_uid: MemoryUid
    members: tuple[MemoryUid, ...]
    radius: int
    graph_generation: int


@dataclass(frozen=True, slots=True)
class ProgressiveSearchResult:
    winner: MemoryUid | None
    equivalence_set: StructuralEquivalenceSet | None
    telemetry: tuple[RadiusTelemetry, ...]
    candidates: tuple[CandidateScore, ...]


def stable_entropy(scores: tuple[float, ...], beta: float) -> float:
    if not scores or len(scores) == 1:
        return 0.0
    scaled = [float(beta) * float(v) for v in scores]
    if any(not math.isfinite(v) for v in scaled):
        raise ValueError("non-finite structural comparison")
    peak = max(scaled)
    weights = [math.exp(v - peak) for v in scaled]
    total = sum(weights)
    probabilities = [w / total for w in weights]
    entropy = -sum(p * math.log(max(p, 1e-300)) for p in probabilities)
    return entropy / math.log(len(probabilities))


class ProgressiveStructuralSearch:
    STATE_VERSION = 1

    def __init__(self, *, radii: tuple[int, ...] = (1, 2, 4, 8), r_max: int = 8, beta_by_radius: dict[int, float] | None = None, max_candidates_per_radius: int = 64, top_candidates: int = 16, ambiguity_threshold: float = 0.15, symmetry_information_threshold: float = 0.01, symmetry_patience: int = 2) -> None:
        self.radii = tuple(sorted({int(r) for r in radii if 0 < int(r) <= int(r_max)}))
        self.r_max = int(r_max)
        self.beta_by_radius = {int(r): float((beta_by_radius or {}).get(int(r), 1.0)) for r in self.radii}
        self.max_candidates_per_radius = int(max_candidates_per_radius)
        self.top_candidates = int(top_candidates)
        self.ambiguity_threshold = float(ambiguity_threshold)
        self.symmetry_information_threshold = float(symmetry_information_threshold)
        self.symmetry_patience = int(symmetry_patience)
        self.equivalence_sets: dict[int, StructuralEquivalenceSet] = {}
        self.search_count = 0
        self.compute_cost = 0

    @staticmethod
    def _score(left: StructuralDescriptor, right: StructuralDescriptor) -> float:
        a, b = set(left.components), set(right.components)
        if not a and not b: return 1.0
        return len(a & b) / max(1, len(a | b))

    def search(self, source_by_radius: dict[int, StructuralDescriptor], candidates_by_radius: dict[int, tuple[StructuralDescriptor, ...]], *, current_graph_generation: int | None = None) -> ProgressiveSearchResult:
        telemetry: list[RadiusTelemetry] = []
        latest: tuple[CandidateScore, ...] = ()
        prior_entropy = 1.0
        stagnant = 0
        self.search_count += 1
        for radius in self.radii:
            source = source_by_radius.get(radius)
            if source is None: continue
            if current_graph_generation is not None and source.graph_generation != int(current_graph_generation):
                raise ValueError("stale source descriptor")
            raw_candidates = candidates_by_radius.get(radius, ())[:self.max_candidates_per_radius]
            scored = []
            for candidate in raw_candidates:
                if current_graph_generation is not None and candidate.graph_generation != int(current_graph_generation):
                    continue
                scored.append(CandidateScore(candidate.uid, self._score(source, candidate)))
            scored.sort(key=lambda row: (-row.score, row.uid))
            latest = tuple(scored[:self.top_candidates])
            self.compute_cost += len(raw_candidates)
            entropy = stable_entropy(tuple(row.score for row in latest), self.beta_by_radius[radius])
            delta = prior_entropy - entropy
            margin = 1.0 if len(latest) < 2 else latest[0].score - latest[1].score
            telemetry.append(RadiusTelemetry(radius, len(latest), entropy, delta, margin, len(raw_candidates)))
            stagnant = stagnant + 1 if abs(delta) <= self.symmetry_information_threshold else 0
            prior_entropy = entropy
            if latest and (len(latest) == 1 or margin > self.ambiguity_threshold):
                return ProgressiveSearchResult(latest[0].uid, None, tuple(telemetry), latest)
            if latest and entropy >= 1.0 - self.ambiguity_threshold and stagnant >= self.symmetry_patience:
                members = tuple(row.uid for row in latest)
                eq_uid = stable_u64(source.uid.hi, source.uid.lo, radius, *(u.hi ^ u.lo for u in members), person=b"v9-equivalence")
                eq = StructuralEquivalenceSet(eq_uid, source.uid, members, radius, source.graph_generation)
                self.equivalence_sets[eq_uid] = eq
                return ProgressiveSearchResult(None, eq, tuple(telemetry), latest)
        winner = latest[0].uid if len(latest) == 1 else None
        return ProgressiveSearchResult(winner, None, tuple(telemetry), latest)

    def telemetry(self) -> dict[str, int]:
        return {"search_count": self.search_count, "candidate_search_compute": self.compute_cost, "equivalence_set_count": len(self.equivalence_sets)}

    def state_dict(self) -> dict[str, object]:
        return {
            "version": self.STATE_VERSION, "radii": list(self.radii), "r_max": self.r_max,
            "beta_by_radius": {str(k): v for k, v in sorted(self.beta_by_radius.items())},
            "max_candidates_per_radius": self.max_candidates_per_radius, "top_candidates": self.top_candidates,
            "ambiguity_threshold": self.ambiguity_threshold, "symmetry_information_threshold": self.symmetry_information_threshold, "symmetry_patience": self.symmetry_patience,
            "equivalence_sets": [{"uid": e.equivalence_uid, "source": [e.source_uid.hi,e.source_uid.lo], "members": [[u.hi,u.lo] for u in e.members], "radius": e.radius, "graph_generation": e.graph_generation} for e in self.equivalence_sets.values()],
            "search_count": self.search_count, "compute_cost": self.compute_cost,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "ProgressiveStructuralSearch":
        if int(state.get("version", 0)) != cls.STATE_VERSION: raise ValueError("unsupported progressive search state")
        obj = cls(radii=tuple(map(int,state.get("radii",(1,2,4,8)))), r_max=int(state.get("r_max",8)), beta_by_radius={int(k):float(v) for k,v in dict(state.get("beta_by_radius",{})).items()}, max_candidates_per_radius=int(state.get("max_candidates_per_radius",64)), top_candidates=int(state.get("top_candidates",16)), ambiguity_threshold=float(state.get("ambiguity_threshold",.15)), symmetry_information_threshold=float(state.get("symmetry_information_threshold",.01)), symmetry_patience=int(state.get("symmetry_patience",2)))
        for raw in state.get("equivalence_sets", []):
            if not isinstance(raw,dict): continue
            eq=StructuralEquivalenceSet(int(raw["uid"]),MemoryUid(*map(int,raw["source"])),tuple(MemoryUid(*map(int,p)) for p in raw["members"]),int(raw["radius"]),int(raw["graph_generation"])); obj.equivalence_sets[eq.equivalence_uid]=eq
        obj.search_count=int(state.get("search_count",0)); obj.compute_cost=int(state.get("compute_cost",0)); return obj
