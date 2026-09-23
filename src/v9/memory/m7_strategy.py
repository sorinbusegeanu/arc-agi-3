from __future__ import annotations

from dataclasses import dataclass

from .identity import MemoryUid
from .m6_outcome import M6Outcome
from .model import MemoryLevel, MemoryType
from .provenance import DerivationProvenance


@dataclass(frozen=True, slots=True)
class M7Strategy:
    uid: MemoryUid
    target_outcome: MemoryUid
    target_environment_id: int
    native_actions: tuple[int, ...]
    reliability_successes: int
    reliability_trials: int
    primary_valence_sum: int
    realized_cost_sum: int
    provenance: DerivationProvenance

    @property
    def reliability(self) -> float:
        return self.reliability_successes / self.reliability_trials if self.reliability_trials else 0.0

    @property
    def expected_cost(self) -> float | None:
        return self.realized_cost_sum / self.reliability_successes if self.reliability_successes else None

    @classmethod
    def form(cls, outcome: M6Outcome, *, target_environment_id: int, native_actions: tuple[int, ...], successes: int, trials: int, primary_valence_sum: int, realized_cost_sum: int) -> "M7Strategy":
        if not native_actions or trials <= 0 or not 0 <= successes <= trials:
            raise ValueError("strategy evidence is invalid")
        uid = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, (outcome.uid.hi, outcome.uid.lo, int(target_environment_id), *native_actions))
        return cls(uid, outcome.uid, int(target_environment_id), tuple(int(a) for a in native_actions), int(successes), int(trials), int(primary_valence_sum), int(realized_cost_sum), DerivationProvenance((outcome.uid,), outcome.provenance.evidence))

