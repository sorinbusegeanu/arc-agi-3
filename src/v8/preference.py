from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

from v8.model import MemoryUid


@dataclass(frozen=True, slots=True)
class PreferenceProbe:
    outcome_a: MemoryUid
    outcome_b: MemoryUid
    context_bucket: int
    chosen_outcome: MemoryUid
    both_reachable: bool
    preference_influenced: bool


@dataclass(frozen=True, slots=True)
class PreferenceEvidence:
    preferred: MemoryUid
    other: MemoryUid
    context_bucket: int
    strength: float
    state: str
    clean_probe_count: int


class PreferenceEstimator:
    """Infer target-like preference only from causally clean comparable probes."""

    def __init__(self, *, support_threshold: int = 6, stable_margin: float = 0.30) -> None:
        self.support_threshold = int(support_threshold)
        self.stable_margin = float(stable_margin)
        self._probes: list[PreferenceProbe] = []

    def record_probe(
        self,
        *,
        outcome_a: MemoryUid,
        outcome_b: MemoryUid,
        context_bucket: int,
        chosen_outcome: MemoryUid,
        both_reachable: bool,
        preference_influenced: bool,
    ) -> bool:
        if outcome_a == outcome_b:
            return False
        if chosen_outcome not in {outcome_a, outcome_b}:
            return False
        if not both_reachable or preference_influenced:
            return False
        self._probes.append(
            PreferenceProbe(
                outcome_a,
                outcome_b,
                int(context_bucket),
                chosen_outcome,
                True,
                False,
            )
        )
        return True

    def evaluate(self) -> tuple[PreferenceEvidence, ...]:
        by_pair: dict[tuple[int, MemoryUid, MemoryUid], dict[MemoryUid, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for probe in self._probes:
            a, b = sorted((probe.outcome_a, probe.outcome_b))
            by_pair[(probe.context_bucket, a, b)][probe.chosen_outcome] += 1

        result: list[PreferenceEvidence] = []
        for (context, a, b), counts in by_pair.items():
            count_a = counts.get(a, 0)
            count_b = counts.get(b, 0)
            total = count_a + count_b
            if total == 0:
                continue
            if count_a >= count_b:
                preferred, other = a, b
                preferred_count, other_count = count_a, count_b
            else:
                preferred, other = b, a
                preferred_count, other_count = count_b, count_a
            strength = (preferred_count - other_count) / total
            if total < self.support_threshold:
                state = "CANDIDATE"
            else:
                state = "STABLE" if strength >= self.stable_margin else "PROBE_ONLY"
            result.append(
                PreferenceEvidence(
                    preferred,
                    other,
                    context,
                    float(strength),
                    state,
                    total,
                )
            )
        return tuple(result)

    def state_dict(self) -> dict[str, object]:
        rows = []
        for probe in self._probes:
            raw = asdict(probe)
            for key in ("outcome_a", "outcome_b", "chosen_outcome"):
                uid = raw[key]
                raw[key] = [uid.hi, uid.lo]
            rows.append(raw)
        return {
            "support_threshold": self.support_threshold,
            "stable_margin": self.stable_margin,
            "probes": rows,
        }

    def load_state(self, state: dict[str, object] | None) -> None:
        if not state:
            return
        for raw in state.get("probes", []):
            if not isinstance(raw, dict):
                continue
            a = raw.get("outcome_a", [0, 0])
            b = raw.get("outcome_b", [0, 0])
            chosen = raw.get("chosen_outcome", [0, 0])
            self._probes.append(
                PreferenceProbe(
                    MemoryUid(int(a[0]), int(a[1])),
                    MemoryUid(int(b[0]), int(b[1])),
                    int(raw.get("context_bucket", 0)),
                    MemoryUid(int(chosen[0]), int(chosen[1])),
                    bool(raw.get("both_reachable", True)),
                    bool(raw.get("preference_influenced", False)),
                )
            )
