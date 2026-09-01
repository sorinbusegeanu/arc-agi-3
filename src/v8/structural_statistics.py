from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
import math

from v8.model import stable_u64


class EstimatorState(IntEnum):
    EMPTY = 0
    PROVISIONAL = 1
    AUTHORITATIVE = 2
    RECALIBRATING = 3


@dataclass(slots=True)
class RunningSummary:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, value: float) -> None:
        x = float(value)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)
        self.minimum = min(self.minimum, x)
        self.maximum = max(self.maximum, x)

    @property
    def variance(self) -> float:
        return 0.0 if self.count < 2 else self.m2 / (self.count - 1)


@dataclass(slots=True)
class ScaleBucket:
    state: EstimatorState = EstimatorState.EMPTY
    estimator_generation: int = 0
    provisional: RunningSummary = field(default_factory=RunningSummary)
    bootstrap_eligible: RunningSummary = field(default_factory=RunningSummary)
    authoritative: RunningSummary = field(default_factory=RunningSummary)
    observations_seen: int = 0
    samples_discarded: int = 0
    stable_contingencies: set[int] = field(default_factory=set)
    descriptor_classes: set[int] = field(default_factory=set)
    first_watermark: int = 0
    last_watermark: int = 0
    reservoir: list[tuple[int, float]] = field(default_factory=list)


class ScaleStratifiedStatistics:
    STATE_VERSION = 1

    def __init__(self, *, n_bootstrap: int = 64, n_stable_bootstrap: int = 16, coverage_bootstrap: float = 0.50, span_bootstrap: int = 64, max_provisional_samples: int = 256) -> None:
        self.n_bootstrap = int(n_bootstrap)
        self.n_stable_bootstrap = int(n_stable_bootstrap)
        self.coverage_bootstrap = float(coverage_bootstrap)
        self.span_bootstrap = int(span_bootstrap)
        self.max_provisional_samples = int(max_provisional_samples)
        self.buckets: dict[tuple[str, int, int], ScaleBucket] = {}

    def observe(self, component: str, radius: int, value: float, *, watermark: int, stable: bool, contingency_uid: int = 0, descriptor_class: int = 0, schema_class: int = 0) -> ScaleBucket:
        key = (str(component), int(radius), int(schema_class))
        bucket = self.buckets.setdefault(key, ScaleBucket(state=EstimatorState.PROVISIONAL))
        bucket.observations_seen += 1
        bucket.provisional.update(value)
        if bucket.first_watermark == 0:
            bucket.first_watermark = int(watermark)
        bucket.last_watermark = max(bucket.last_watermark, int(watermark))
        priority = stable_u64(component, radius, schema_class, watermark, bucket.observations_seen, person=b"v9-stat-sample")
        if len(bucket.reservoir) < self.max_provisional_samples:
            bucket.reservoir.append((priority, float(value)))
        else:
            worst = max(range(len(bucket.reservoir)), key=lambda i: bucket.reservoir[i][0])
            if priority < bucket.reservoir[worst][0]:
                bucket.reservoir[worst] = (priority, float(value))
            else:
                bucket.samples_discarded += 1
        if stable:
            bucket.bootstrap_eligible.update(value)
            if contingency_uid:
                bucket.stable_contingencies.add(int(contingency_uid))
            bucket.descriptor_classes.add(int(descriptor_class))
        self._maybe_bootstrap(bucket)
        return bucket

    def _maybe_bootstrap(self, bucket: ScaleBucket) -> None:
        span = max(0, bucket.last_watermark - bucket.first_watermark)
        coverage = len(bucket.descriptor_classes) / max(1, self.n_stable_bootstrap)
        if (
            bucket.state is not EstimatorState.AUTHORITATIVE
            and bucket.bootstrap_eligible.count >= self.n_bootstrap
            and len(bucket.stable_contingencies) >= self.n_stable_bootstrap
            and coverage >= self.coverage_bootstrap
            and span >= self.span_bootstrap
        ):
            bucket.authoritative = RunningSummary(**asdict(bucket.bootstrap_eligible))
            bucket.estimator_generation += 1
            bucket.state = EstimatorState.AUTHORITATIVE

    def normalize(self, component: str, radius: int, value: float, *, schema_class: int = 0) -> float | None:
        bucket = self.buckets.get((str(component), int(radius), int(schema_class)))
        if bucket is None or bucket.state is not EstimatorState.AUTHORITATIVE:
            return None
        variance = max(0.0, bucket.authoritative.variance)
        if variance <= 1e-12:
            return 0.0
        return (float(value) - bucket.authoritative.mean) / math.sqrt(variance)

    def telemetry(self) -> dict[str, int]:
        return {
            "provisional_observations_seen": sum(b.observations_seen for b in self.buckets.values()),
            "provisional_samples_retained": sum(len(b.reservoir) for b in self.buckets.values()),
            "provisional_samples_discarded": sum(b.samples_discarded for b in self.buckets.values()),
            "authoritative_buckets": sum(b.state is EstimatorState.AUTHORITATIVE for b in self.buckets.values()),
        }

    def state_dict(self) -> dict[str, object]:
        rows = []
        for (component, radius, schema), bucket in sorted(self.buckets.items()):
            rows.append({
                "component": component, "radius": radius, "schema": schema,
                "state": int(bucket.state), "estimator_generation": bucket.estimator_generation,
                "provisional": asdict(bucket.provisional), "bootstrap_eligible": asdict(bucket.bootstrap_eligible), "authoritative": asdict(bucket.authoritative),
                "observations_seen": bucket.observations_seen, "samples_discarded": bucket.samples_discarded,
                "stable_contingencies": sorted(bucket.stable_contingencies), "descriptor_classes": sorted(bucket.descriptor_classes),
                "first_watermark": bucket.first_watermark, "last_watermark": bucket.last_watermark,
                "reservoir": [[p, v] for p, v in sorted(bucket.reservoir)],
            })
        return {
            "version": self.STATE_VERSION,
            "config": {"n_bootstrap": self.n_bootstrap, "n_stable_bootstrap": self.n_stable_bootstrap, "coverage_bootstrap": self.coverage_bootstrap, "span_bootstrap": self.span_bootstrap, "max_provisional_samples": self.max_provisional_samples},
            "buckets": rows,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "ScaleStratifiedStatistics":
        if int(state.get("version", 0)) != cls.STATE_VERSION:
            raise ValueError("unsupported structural statistics state")
        cfg = state.get("config", {}); obj = cls(**cfg) if isinstance(cfg, dict) else cls()
        for raw in state.get("buckets", []):
            if not isinstance(raw, dict): continue
            bucket = ScaleBucket(
                state=EstimatorState(int(raw["state"])), estimator_generation=int(raw.get("estimator_generation", 0)),
                provisional=RunningSummary(**raw["provisional"]), bootstrap_eligible=RunningSummary(**raw["bootstrap_eligible"]), authoritative=RunningSummary(**raw["authoritative"]),
                observations_seen=int(raw.get("observations_seen", 0)), samples_discarded=int(raw.get("samples_discarded", 0)),
                stable_contingencies=set(map(int, raw.get("stable_contingencies", []))), descriptor_classes=set(map(int, raw.get("descriptor_classes", []))),
                first_watermark=int(raw.get("first_watermark", 0)), last_watermark=int(raw.get("last_watermark", 0)),
                reservoir=[(int(p), float(v)) for p, v in raw.get("reservoir", [])],
            )
            obj.buckets[(str(raw["component"]), int(raw["radius"]), int(raw["schema"]))] = bucket
        return obj
