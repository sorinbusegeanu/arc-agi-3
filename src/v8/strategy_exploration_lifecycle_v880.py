from __future__ import annotations

from v8.model import CognitiveState, MemoryLevel, MemoryType
from v8.peers_v82 import V82DevelopmentalPeerSupervisor


_PROBATIONARY = {
    int(CognitiveState.CANDIDATE),
    int(CognitiveState.PROBATION),
}


def _advance_probationary_m7(supervisor: V82DevelopmentalPeerSupervisor) -> int:
    """Give bounded lifecycle attention to M7 rows that the global scan can starve.

    Planning already permits probationary strategies as epsilon probes. This pass only
    ensures those strategies can receive lifecycle promotion decisions after evidence
    accumulates. It does not emit H12 efficiency evidence and does not relax the
    empirical same-outcome/same-context comparison gate.
    """
    limit = max(1, int(supervisor.candidate_budget) // 4)
    rows = tuple(
        row
        for row in supervisor.read_view.node_records(level=MemoryLevel.M7)
        if int(row.memory_type) == int(MemoryType.STRATEGY)
        and int(row.cognitive_state) in _PROBATIONARY
    )
    advanced = 0
    for row in rows[:limit]:
        decision = supervisor.lifecycle.decide(row)
        if decision is None:
            continue
        if int(decision.cognitive_state) not in {
            int(CognitiveState.ACTIVE),
            int(CognitiveState.VALIDATED),
            int(CognitiveState.REACTIVATED),
        }:
            continue
        if not supervisor._fresh(
            "m7_exploration_lifecycle", row.uid, row.updated_watermark
        ):
            continue
        supervisor._submit(
            supervisor._existing_proposal(
                row,
                cognitive_state=int(decision.cognitive_state),
                validation_state=int(decision.validation_state),
            )
        )
        advanced += 1
    return advanced


def install_strategy_exploration_lifecycle_v880() -> None:
    if getattr(V82DevelopmentalPeerSupervisor, "_v880_strategy_lifecycle_installed", False):
        return

    original_run_once = V82DevelopmentalPeerSupervisor.run_once

    def run_once(self: V82DevelopmentalPeerSupervisor):
        result = original_run_once(self)
        _advance_probationary_m7(self)
        return result

    V82DevelopmentalPeerSupervisor.run_once = run_once
    V82DevelopmentalPeerSupervisor._v880_strategy_lifecycle_installed = True
