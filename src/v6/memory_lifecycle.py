from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from v6.memory.v63_policy import unified_memory_fitness


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class MemoryRecord:
    interaction_id: str
    family_id: str | None
    context_signature: str | None
    action_signature: str | None
    carrier_signature: str | None
    isf_total: float
    prediction_error: float
    learning_value: float
    transfer_potential: float
    explanatory_potential: float
    context_contradiction: bool
    timestamp_step: int
    replay_count: int
    status: str
    retention_reason: str
    transfer_prior: float | None = None
    transfer_empirical_rate: float | None = None
    learning_value_realized: float | None = None
    explanatory_value_realized: float | None = None
    evidence_revision_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayCandidate:
    interaction_id: str
    replay_priority: float
    reason: str
    family_id: str | None
    context_signature: str | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_memory_fitness(
    *,
    isf_total: float,
    prediction_error: float,
    learning_value: float,
    transfer_potential: float,
    explanatory_potential: float,
    context_contradiction: bool,
    replay_count: int,
) -> float:
    # v6.3: PE/LV/TP/EP are already represented in ISF and must not be re-added
    # with a second arbitrary coefficient set. MF is the common monotone
    # aggregation over active normalized memory-level dimensions.
    fitness, _components = unified_memory_fitness(
        isf_score=clamp01(isf_total),
        explanatory_reach=clamp01(explanatory_potential),
        transfer_prior=clamp01(transfer_potential),
        transfer_empirical=None,
        recurrence_score=None,
        efficiency_score=None,
    )
    del prediction_error, learning_value, context_contradiction, replay_count
    return fitness


class MemoryLifecycleManager:
    def __init__(
        self,
        *,
        max_active_records: int = 50_000,
        replay_queue_size: int = 1_000,
        protect_isf_threshold: float = 0.70,
        forget_isf_threshold: float = 0.10,
        min_records_before_forgetting: int = 1_000,
    ) -> None:
        self.max_active_records = int(max_active_records)
        self.replay_queue_size = int(replay_queue_size)
        self.protect_isf_threshold = float(protect_isf_threshold)
        self.forget_isf_threshold = float(forget_isf_threshold)
        self.min_records_before_forgetting = int(min_records_before_forgetting)
        self.records: dict[str, MemoryRecord] = {}
        self.replay_candidates: dict[str, ReplayCandidate] = {}
        self.status_counts: Counter[str] = Counter()
        self.family_counts: Counter[str] = Counter()
        self.context_counts: Counter[str] = Counter()
        self.retention_reason_counts: Counter[str] = Counter()
        self.forgotten_count = 0
        self.compressed_count = 0
        self.protected_count = 0

    def register_interaction(
        self,
        *,
        interaction_id: str,
        family_id: str | None,
        context_signature: str | None,
        action_signature: str | None,
        carrier_signature: str | None,
        isf_total: float,
        prediction_error: float,
        learning_value: float,
        transfer_potential: float,
        explanatory_potential: float,
        context_contradiction: bool,
        timestamp_step: int,
    ) -> MemoryRecord:
        status = "protected" if float(isf_total) >= self.protect_isf_threshold else "active"
        retention_reason = _retention_reason(
            isf_total=float(isf_total),
            prediction_error=float(prediction_error),
            learning_value=float(learning_value),
            transfer_potential=float(transfer_potential),
            explanatory_potential=float(explanatory_potential),
            context_contradiction=bool(context_contradiction),
            status=status,
        )
        record = MemoryRecord(
            interaction_id=str(interaction_id),
            family_id=None if family_id is None else str(family_id),
            context_signature=None if context_signature is None else str(context_signature),
            action_signature=None if action_signature is None else str(action_signature),
            carrier_signature=None if carrier_signature is None else str(carrier_signature),
            isf_total=float(isf_total),
            prediction_error=float(prediction_error),
            learning_value=float(learning_value),
            transfer_potential=float(transfer_potential),
            explanatory_potential=float(explanatory_potential),
            context_contradiction=bool(context_contradiction),
            timestamp_step=int(timestamp_step),
            replay_count=0,
            status=status,
            retention_reason=retention_reason,
            transfer_prior=float(transfer_potential),
        )
        self.records[record.interaction_id] = record
        self._recompute_counters()
        if self._should_enter_replay(record):
            priority = self._memory_fitness(record)
            self.replay_candidates[record.interaction_id] = ReplayCandidate(
                interaction_id=record.interaction_id,
                replay_priority=priority,
                reason=retention_reason if retention_reason != "high_isf" else "prediction_error" if record.prediction_error >= 0.50 else retention_reason,
                family_id=record.family_id,
                context_signature=record.context_signature,
                status=record.status,
            )
            self._trim_replay_queue()
        if len(self.records) >= self.min_records_before_forgetting:
            self.apply_forgetting_policy()
        self._recompute_counters()
        return self.records[record.interaction_id]

    def apply_forgetting_policy(self) -> None:
        active_ids = [record.interaction_id for record in self.records.values() if record.status == "active"]
        overflow = len(active_ids) - self.max_active_records
        if overflow <= 0:
            return
        protected_ids = {record.interaction_id for record in self.records.values() if record.status == "protected"}
        replay_ids = set(self.replay_candidates)
        candidates = [
            (self._memory_fitness(record), record.timestamp_step, record.interaction_id)
            for record in self.records.values()
            if record.status == "active" and record.interaction_id not in protected_ids and record.interaction_id not in replay_ids
        ]
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        for _fitness, _step, interaction_id in candidates[: max(0, overflow)]:
            record = self.records[interaction_id]
            new_status = "forgotten" if float(record.isf_total) < self.forget_isf_threshold else "compressed"
            self.records[interaction_id] = MemoryRecord(**{**record.to_dict(), "status": new_status})
        self._recompute_counters()

    def active_record_count(self) -> int:
        return sum(1 for record in self.records.values() if record.status == "active")

    def get_replay_batch(self, limit: int = 32) -> list[ReplayCandidate]:
        ordered = sorted(self.replay_candidates.values(), key=lambda item: (-float(item.replay_priority), item.interaction_id))
        return ordered[: max(0, int(limit))]

    def mark_replayed(self, interaction_ids: Iterable[str]) -> None:
        for interaction_id in interaction_ids:
            record = self.records.get(str(interaction_id))
            if record is None:
                continue
            updated = MemoryRecord(**{**record.to_dict(), "replay_count": int(record.replay_count) + 1})
            self.records[str(interaction_id)] = updated
            if str(interaction_id) in self.replay_candidates:
                priority = self._memory_fitness(updated)
                if priority < 0.20:
                    self.replay_candidates.pop(str(interaction_id), None)
                else:
                    replay = self.replay_candidates[str(interaction_id)]
                    self.replay_candidates[str(interaction_id)] = ReplayCandidate(
                        interaction_id=replay.interaction_id,
                        replay_priority=priority,
                        reason=replay.reason,
                        family_id=replay.family_id,
                        context_signature=replay.context_signature,
                        status=updated.status,
                    )
        self._trim_replay_queue()
        self._recompute_counters()

    def summary(self) -> dict[str, Any]:
        priorities = [float(item.replay_priority) for item in self.replay_candidates.values()]
        return {
            "memory_record_count": len(self.records),
            "memory_active_count": sum(1 for record in self.records.values() if record.status == "active"),
            "memory_protected_count": sum(1 for record in self.records.values() if record.status == "protected"),
            "memory_compressed_count": sum(1 for record in self.records.values() if record.status == "compressed"),
            "memory_forgotten_count": sum(1 for record in self.records.values() if record.status == "forgotten"),
            "memory_replay_candidate_count": len(self.replay_candidates),
            "memory_max_replay_priority": max(priorities, default=0.0),
            "memory_mean_replay_priority": (sum(priorities) / len(priorities)) if priorities else 0.0,
            "memory_evidence_revision_count": sum(
                int(record.evidence_revision_count) for record in self.records.values()
            ),
            "memory_top_retention_reasons": [
                {"retention_reason": reason, "count": int(count)}
                for reason, count in self.retention_reason_counts.most_common(20)
            ],
        }

    def import_record(self, record: MemoryRecord) -> None:
        self.records[str(record.interaction_id)] = record
        self._recompute_counters()

    def import_replay_candidate(self, candidate: ReplayCandidate) -> None:
        self.replay_candidates[str(candidate.interaction_id)] = ReplayCandidate(
            interaction_id=str(candidate.interaction_id),
            replay_priority=float(candidate.replay_priority),
            reason=str(candidate.reason),
            family_id=None if candidate.family_id is None else str(candidate.family_id),
            context_signature=None if candidate.context_signature is None else str(candidate.context_signature),
            status=str(candidate.status),
        )
        self._trim_replay_queue()
        self._recompute_counters()

    def apply_post_factum_credit(
        self,
        interaction_id: str,
        *,
        learning_credit: float,
        reason: str,
    ) -> None:
        self.apply_retrospective_evidence(
            interaction_id,
            learning_value=learning_credit,
            reason=reason,
        )

    def apply_retrospective_evidence(
        self,
        interaction_id: str,
        *,
        learning_value: float | None = None,
        explanatory_value: float | None = None,
        transfer_empirical_rate: float | None = None,
        reason: str,
    ) -> None:
        interaction_id = str(interaction_id)
        record = self.records.get(interaction_id)
        if record is None:
            return
        updated = MemoryRecord(
            **{
                **record.to_dict(),
                "learning_value_realized": _max_optional(
                    record.learning_value_realized,
                    learning_value,
                ),
                "explanatory_value_realized": _max_optional(
                    record.explanatory_value_realized,
                    explanatory_value,
                ),
                "transfer_empirical_rate": (
                    record.transfer_empirical_rate
                    if transfer_empirical_rate is None
                    else clamp01(transfer_empirical_rate)
                ),
                "evidence_revision_count": int(record.evidence_revision_count) + 1,
            }
        )
        self.records[interaction_id] = updated

        realized_credit = max(
            0.0,
            float(learning_value or 0.0),
            float(explanatory_value or 0.0),
            float(transfer_empirical_rate or 0.0),
        )
        existing = self.replay_candidates.get(interaction_id)
        replay_priority = max(
            0.0 if existing is None else float(existing.replay_priority),
            clamp01(realized_credit),
            self._memory_fitness(updated),
        )
        merged_reason = _merge_reasons(existing.reason if existing is not None else None, reason)
        self.replay_candidates[interaction_id] = ReplayCandidate(
            interaction_id=interaction_id,
            replay_priority=replay_priority,
            reason=merged_reason,
            family_id=updated.family_id,
            context_signature=updated.context_signature,
            status=updated.status,
        )
        self._trim_replay_queue()
        self._recompute_counters()

    def _should_enter_replay(self, record: MemoryRecord) -> bool:
        transfer_prior = (
            record.transfer_prior
            if record.transfer_prior is not None
            else record.transfer_potential
        )
        return (
            float(record.prediction_error) >= 0.50
            or float(record.learning_value) >= 0.50
            or float(transfer_prior or 0.0) >= 0.40
            or float(record.explanatory_potential) >= 0.40
            or bool(record.context_contradiction)
        )

    def _trim_replay_queue(self) -> None:
        if len(self.replay_candidates) <= self.replay_queue_size:
            return
        keep = self.get_replay_batch(limit=self.replay_queue_size)
        keep_ids = {item.interaction_id for item in keep}
        self.replay_candidates = {key: value for key, value in self.replay_candidates.items() if key in keep_ids}

    def _memory_fitness(self, record: MemoryRecord) -> float:
        transfer_prior = (
            record.transfer_prior
            if record.transfer_prior is not None
            else record.transfer_potential
        )
        fitness, _components = unified_memory_fitness(
            isf_score=record.isf_total,
            explanatory_reach=(
                record.explanatory_value_realized
                if record.explanatory_value_realized is not None
                else record.explanatory_potential
            ),
            transfer_prior=transfer_prior,
            transfer_empirical=record.transfer_empirical_rate,
            recurrence_score=None,
            efficiency_score=None,
        )
        return fitness

    def _recompute_counters(self) -> None:
        self.status_counts = Counter(record.status for record in self.records.values())
        self.family_counts = Counter(record.family_id for record in self.records.values() if record.family_id is not None)
        self.context_counts = Counter(record.context_signature for record in self.records.values() if record.context_signature is not None)
        self.retention_reason_counts = Counter(record.retention_reason for record in self.records.values())
        self.forgotten_count = int(self.status_counts.get("forgotten", 0))
        self.compressed_count = int(self.status_counts.get("compressed", 0))
        self.protected_count = int(self.status_counts.get("protected", 0))


def _retention_reason(
    *,
    isf_total: float,
    prediction_error: float,
    learning_value: float,
    transfer_potential: float,
    explanatory_potential: float,
    context_contradiction: bool,
    status: str,
) -> str:
    if status == "protected" and float(isf_total) >= 0.70:
        return "high_isf"
    candidates = {
        "prediction_error": clamp01(prediction_error),
        "learning_value": clamp01(learning_value),
        "transfer_prior": clamp01(transfer_potential),
        "explanatory_potential": clamp01(explanatory_potential),
        "context_contradiction": 1.0 if context_contradiction else 0.0,
    }
    reason, score = max(candidates.items(), key=lambda item: item[1])
    return reason if score > 0.0 else "default_active"


def _merge_reasons(old: str | None, new: str) -> str:
    parts: list[str] = []
    for value in (old, new):
        if not value:
            continue
        for part in str(value).split("+"):
            if part and part not in parts:
                parts.append(part)
    return "+".join(parts)


def _max_optional(old: float | None, new: float | None) -> float | None:
    if new is None:
        return old
    if old is None:
        return clamp01(new)
    return max(clamp01(old), clamp01(new))
