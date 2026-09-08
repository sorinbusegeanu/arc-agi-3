from __future__ import annotations

from v8.arena import NodeRecord
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, ValidationState, stable_u64
from v8.promotion import EvidenceGatedPromotionEngine, FormationCandidate


def _strategy_candidates(
    engine: EvidenceGatedPromotionEngine,
    nodes: tuple[NodeRecord, ...],
    *,
    budget: int,
) -> tuple[FormationCandidate, ...]:
    limit = max(0, int(budget))
    if limit <= 0:
        return ()
    stable_m1 = [
        row
        for row in nodes
        if int(row.level) == int(MemoryLevel.M1)
        and int(row.memory_type) == int(MemoryType.CONTINGENCY)
        and row.support_count >= engine.min_contingency_support
        and len(row.key_parts) >= 4
        and engine._admissible(row)
    ]
    m6_rows = [
        row
        for row in nodes
        if int(row.level) == int(MemoryLevel.M6)
        and int(row.memory_type) == int(MemoryType.OUTCOME)
        and row.support_count >= 2
        and engine._admissible(row)
    ]
    result: list[FormationCandidate] = []
    for outcome in sorted(m6_rows, key=lambda row: row.uid):
        outcome_future = int(outcome.key_parts[0]) if outcome.key_parts else 0
        for contingency in stable_m1:
            if engine._future_bucket(contingency.future_option_delta) != outcome_future:
                continue
            context_bucket = stable_u64(
                int(contingency.key_parts[0]), person=b"v8-context"
            )
            key = (
                int(contingency.key_parts[1]),
                int(outcome.uid.hi),
                int(outcome.uid.lo),
                int(context_bucket),
            )
            uid = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, key)
            support = min(int(contingency.support_count), int(outcome.support_count))
            result.append(
                FormationCandidate(
                    uid,
                    MemoryLevel.M7,
                    MemoryType.STRATEGY,
                    key,
                    (outcome.uid, contingency.uid),
                    max(1, support),
                    min(1.0, contingency.significance),
                    min(1.0, contingency.learning_value),
                    0.0,
                    1.0,
                    contingency.future_option_delta,
                    int(CognitiveState.PROBATION),
                    int(ValidationState.STRUCTURAL),
                    "strategy_reuse",
                    min(1.0, support / 4.0),
                )
            )
            if len(result) >= limit:
                return tuple(result)
    return tuple(result)


def install_promotion_strategy_fairness_v828() -> None:
    if getattr(EvidenceGatedPromotionEngine, "_v828_strategy_fairness_installed", False):
        return
    original_propose = EvidenceGatedPromotionEngine.propose

    def propose(self, nodes, edges, *, budget=256):
        limit = max(0, int(budget))
        if limit <= 0:
            return ()
        reserve = min(64, max(8, limit // 4))
        strategy_rows = _strategy_candidates(self, tuple(nodes), budget=reserve)
        if not strategy_rows:
            return original_propose(self, nodes, edges, budget=limit)
        base_budget = max(1, limit - len(strategy_rows))
        base_rows = tuple(original_propose(self, nodes, edges, budget=base_budget))
        existing = {row.uid for row in base_rows}
        appended = tuple(row for row in strategy_rows if row.uid not in existing)
        return tuple((base_rows + appended)[:limit])

    EvidenceGatedPromotionEngine.propose = propose
    EvidenceGatedPromotionEngine._v828_strategy_fairness_installed = True
