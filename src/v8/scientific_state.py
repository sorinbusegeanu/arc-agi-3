from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScientificStateSnapshot:
    watermark: int
    graph_digest: str
    developmental_stage: int
    evidence_count: int
    replay_count: int
    attention_count: int
    invariant_violations: tuple[str, ...] = ()
