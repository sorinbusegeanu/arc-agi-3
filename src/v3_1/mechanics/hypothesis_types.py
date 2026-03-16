from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class HypothesisSupportRef:
    ref_id: str
    ref_kind: str
    evidence_tier: str
    provenance: str


@dataclass(frozen=True)
class HypothesisContradictionRef:
    ref_id: str
    ref_kind: str
    evidence_tier: str
    provenance: str


@dataclass(frozen=True)
class HypothesisEdgeProposal:
    proposal_id: str
    proposal_kind: str
    provenance: str
    authoritative: bool
    src_node_id: str
    dst_node_id: str
    edge_kind: str
    support_refs: tuple[HypothesisSupportRef, ...]
    contradiction_refs: tuple[HypothesisContradictionRef, ...]
    confidence: float
    novelty_score: float
    requires_validation: bool
    generation_version: str
    round_id: int
    episode_ids: tuple[str, ...]
    rule_id: str | None = None
    explanation: str | None = None
    validation_requirements: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HypothesisPathProposal:
    proposal_id: str
    proposal_kind: str
    provenance: str
    authoritative: bool
    src_node_id: str
    dst_node_id: str
    path_kind: str
    support_refs: tuple[HypothesisSupportRef, ...]
    contradiction_refs: tuple[HypothesisContradictionRef, ...]
    confidence: float
    novelty_score: float
    requires_validation: bool
    generation_version: str
    round_id: int
    episode_ids: tuple[str, ...]
    edge_kinds: tuple[str, ...] = ()
    explanation: str | None = None
    validation_requirements: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HypothesisTestProposal:
    proposal_id: str
    proposal_kind: str
    provenance: str
    authoritative: bool
    src_node_id: str
    dst_node_id: str
    path_kind: str
    support_refs: tuple[HypothesisSupportRef, ...]
    contradiction_refs: tuple[HypothesisContradictionRef, ...]
    confidence: float
    novelty_score: float
    requires_validation: bool
    generation_version: str
    round_id: int
    episode_ids: tuple[str, ...]
    test_id: str
    target_node_ids: tuple[str, ...]
    expected_edge_ids: tuple[str, ...]
    discriminates_between_proposal_ids: tuple[str, ...]
    priority: float
    estimated_cost: float
    expected_information_gain: float
    explanation: str | None = None
    validation_requirements: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HypothesisBundle:
    generation_version: str
    round_id: int
    episode_ids: tuple[str, ...]
    provenance: str
    edge_proposals: tuple[HypothesisEdgeProposal, ...] = ()
    path_proposals: tuple[HypothesisPathProposal, ...] = ()
    test_proposals: tuple[HypothesisTestProposal, ...] = ()
    support_summary: dict[str, Any] = field(default_factory=dict)
    contradiction_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
