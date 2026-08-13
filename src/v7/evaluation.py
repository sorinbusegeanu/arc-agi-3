from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from v7.memory.concept_validation import ConceptValidationStatus
from v7.memory.ids import MemoryLevel
from v7.memory.lifecycle import MemoryStatus
from v7.runtime import V7Runtime, V7RuntimeConfig


@dataclass(frozen=True, slots=True)
class V7EvidenceSummary:
    generation: int
    total_memories: int
    level_counts: dict[str, int]
    prediction_error_memories: int
    future_option_memories: int
    promoted_memories: int
    demoted_memories: int
    concept_candidates: int
    transfer_validated_concepts: int
    transfer_trials: int
    successful_transfer_trials: int
    provenance_records: int
    contradiction_records: int


def collect_evidence(root: str | Path) -> V7EvidenceSummary:
    runtime = V7Runtime(V7RuntimeConfig.from_path(root, restore=True, derive_hierarchy=False))
    try:
        view = runtime.writer.published_view
        level_counts = {f"M{level}": sum(1 for node in view.nodes.values() if int(node.level) == level) for level in range(7)}
        concept_nodes = [node for node in view.nodes.values() if node.level == MemoryLevel.M4]
        connection = runtime.lifecycle_evidence.connection
        transfer_total, transfer_success = connection.execute("SELECT COUNT(*), COALESCE(SUM(success),0) FROM transfer_trials").fetchone()
        return V7EvidenceSummary(
            generation=int(view.generation_id),
            total_memories=len(view.nodes),
            level_counts=level_counts,
            prediction_error_memories=sum(1 for score in view.scores.values() if score.prediction_error > 0),
            future_option_memories=sum(1 for score in view.scores.values() if score.future_option_delta != 0),
            promoted_memories=sum(1 for node in view.nodes.values() if int(node.status_flags) & int(MemoryStatus.PROMOTED)),
            demoted_memories=sum(1 for node in view.nodes.values() if int(node.status_flags) & int(MemoryStatus.DEMOTED)),
            concept_candidates=sum(1 for node in concept_nodes if int(node.status_flags) & int(ConceptValidationStatus.CANDIDATE)),
            transfer_validated_concepts=sum(1 for node in concept_nodes if int(node.status_flags) & int(ConceptValidationStatus.TRANSFER_VALIDATED)),
            transfer_trials=int(transfer_total),
            successful_transfer_trials=int(transfer_success or 0),
            provenance_records=int(connection.execute("SELECT COUNT(*) FROM provenance_records").fetchone()[0]),
            contradiction_records=int(connection.execute("SELECT COUNT(*) FROM contradiction_records").fetchone()[0]),
        )
    finally:
        runtime.close()


def write_evidence_report(root: str | Path, output: str | Path | None = None) -> dict[str, object]:
    root_path = Path(root)
    summary = collect_evidence(root_path)
    payload: dict[str, object] = {"schema": "v7-experiment-evidence-v1", "summary": asdict(summary)}
    target = Path(output) if output is not None else root_path / "reports" / "v7_evidence.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
