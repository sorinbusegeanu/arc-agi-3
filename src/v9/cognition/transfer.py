from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from v9.memory.identity import MemoryUid

from .correspondence import StructuralCorrespondence


@dataclass(frozen=True, slots=True)
class TransferTrial:
    correspondence: StructuralCorrespondence
    target_environment_id: int
    target_native_action: int
    enabled_metric: float
    ablated_metric: float
    held_out: bool
    matched: bool

    @property
    def effect(self) -> float:
        return self.enabled_metric - self.ablated_metric


@dataclass(frozen=True, slots=True)
class TransferDecision:
    validated: bool
    target_native_action: int | None
    trials: tuple[TransferTrial, ...]


class ValidationMode(str, Enum):
    LEARNING_ONLY = "learning_only"
    VALIDATION_BUDGETED = "validation_budgeted"
    VALIDATION_FULL = "validation_full"


class TransferTrustState(str, Enum):
    ACTIVE = "ACTIVE"
    PROBATION = "PROBATION"
    FAILED = "FAILED"
    QUARANTINED = "QUARANTINED"


@dataclass(frozen=True, slots=True)
class TransferTrust:
    concept_uid: MemoryUid
    target_environment_id: int
    context_scope_id: int
    successes: int = 0
    failures: int = 0
    cumulative_effect: float = 0.0
    state: TransferTrustState = TransferTrustState.ACTIVE

    @property
    def score(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total else 0.5


class TransferTrustRegistry:
    """Scopes negative transfer to exactly the tested target and context."""

    def __init__(self, *, scope_limit: int = 8192) -> None:
        if scope_limit <= 0:
            raise ValueError("transfer trust scope limit must be positive")
        self.scope_limit = int(scope_limit)
        self.records: dict[tuple[MemoryUid, int, int], TransferTrust] = {}

    def observe(self, concept_uid: MemoryUid, *, target_environment_id: int, context_scope_id: int, effect: float, positive: bool) -> TransferTrust:
        key = (concept_uid, int(target_environment_id), int(context_scope_id))
        if key not in self.records and len(self.records) >= self.scope_limit:
            raise OverflowError("transfer trust scope budget exhausted")
        current = self.records.get(key, TransferTrust(*key))
        successes = current.successes + int(positive)
        failures = current.failures + int(not positive)
        state = TransferTrustState.ACTIVE
        if failures >= 2 and failures > successes:
            state = TransferTrustState.FAILED
        elif failures:
            state = TransferTrustState.PROBATION
        if failures >= 4 and failures >= 2 * max(1, successes):
            state = TransferTrustState.QUARANTINED
        row = TransferTrust(concept_uid, key[1], key[2], successes, failures, current.cumulative_effect + float(effect), state)
        self.records[key] = row
        return row

    def score_map(self) -> dict[tuple[MemoryUid, int, int], float]:
        return {key: row.score for key, row in self.records.items()}

    def state_dict(self) -> dict[str, object]:
        return {
            "scope_limit": self.scope_limit,
            "records": [
                {
                    "concept_uid": [row.concept_uid.hi, row.concept_uid.lo],
                    "target_environment_id": row.target_environment_id,
                    "context_scope_id": row.context_scope_id,
                    "successes": row.successes,
                    "failures": row.failures,
                    "cumulative_effect": row.cumulative_effect,
                    "state": row.state.value,
                }
                for row in sorted(self.records.values(), key=lambda value: (value.concept_uid, value.target_environment_id, value.context_scope_id))
            ]
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "TransferTrustRegistry":
        result = cls(scope_limit=int(state.get("scope_limit", 8192)))
        for raw in state.get("records", []):
            uid = MemoryUid(int(raw["concept_uid"][0]), int(raw["concept_uid"][1]))
            row = TransferTrust(uid, int(raw["target_environment_id"]), int(raw["context_scope_id"]), int(raw["successes"]), int(raw["failures"]), float(raw["cumulative_effect"]), TransferTrustState(str(raw["state"])))
            result.records[(uid, row.target_environment_id, row.context_scope_id)] = row
        return result


def validate_transfer(trials: tuple[TransferTrial, ...], *, minimum_trials: int, effect_threshold: float) -> TransferDecision:
    admissible = tuple(row for row in trials if row.held_out and row.matched and row.correspondence.target_environment_id == row.target_environment_id)
    valid = len(admissible) >= minimum_trials and sum(row.effect for row in admissible) / len(admissible) > effect_threshold
    actions = {row.target_native_action for row in admissible}
    return TransferDecision(bool(valid and len(actions) == 1), next(iter(actions)) if valid and len(actions) == 1 else None, admissible)
