from __future__ import annotations

from collections import Counter

from v5_0.contracts.avatar_types import MechanicDiagnostics, MechanicReport
from v5_0.mechanics.classifier import classify_poi_mechanics
from v5_0.mechanics.evidence_builder import build_mechanic_evidence
from v5_0.mechanics.memory import build_mechanic_memory


def build_mechanic_report(
    ranked_poi_candidates,
    contact_experiment_report=None,
    hud_targeting_report=None,
    solve_report=None,
    previous_mechanic_memory=None,
) -> MechanicReport:
    evidence = build_mechanic_evidence(
        ranked_poi_candidates,
        contact_experiment_report=contact_experiment_report,
        hud_targeting_report=hud_targeting_report,
        solve_report=solve_report,
    )
    states = classify_poi_mechanics(evidence)
    memory = build_mechanic_memory(states, previous_memory_state=previous_mechanic_memory)

    labels = Counter(str(item.mechanic_label) for item in memory.poi_states)
    diagnostics = MechanicDiagnostics(
        poi_count=len(memory.poi_states),
        target_count=int(labels.get("target", 0)),
        decoy_count=int(labels.get("decoy", 0)),
        hazard_count=int(labels.get("hazard", 0)),
        exit_count=int(labels.get("exit", 0)),
        door_or_switch_count=int(labels.get("door_or_switch", 0)),
        retarget_count=0,
        ambiguous_poi_count=sum(1 for item in memory.poi_states if float(item.confidence) < 0.35),
    )
    return MechanicReport(
        memory=memory,
        diagnostics=diagnostics,
        failure_reason=None if memory.poi_states else "no_poi_mechanic_evidence",
    )
