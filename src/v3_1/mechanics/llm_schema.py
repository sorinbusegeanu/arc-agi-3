from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MAX_LLM_EDGE_PROPOSALS = 8
MAX_LLM_PATH_PROPOSALS = 6
MAX_LLM_TEST_PROPOSALS = 6
MAX_LLM_EXPLANATION_LENGTH = 240
ALLOWED_LLM_EDGE_KEYS = {
    "src_node_id",
    "dst_node_id",
    "edge_kind",
    "explanation",
    "validation_requirements",
    "confidence_estimate",
    "novelty_vs_deterministic",
}
ALLOWED_LLM_PATH_KEYS = {
    "src_node_id",
    "dst_node_id",
    "path_kind",
    "explanation",
    "validation_requirements",
    "confidence_estimate",
    "novelty_vs_deterministic",
    "node_ids",
}
ALLOWED_LLM_TEST_KEYS = {
    "src_node_id",
    "dst_node_id",
    "path_kind",
    "explanation",
    "validation_requirements",
    "confidence_estimate",
    "novelty_vs_deterministic",
    "target_node_ids",
}
ALLOWED_LLM_OUTPUT_KEYS = {"edge_proposals", "path_proposals", "test_proposals", "metadata"}


@dataclass(frozen=True)
class LLMEdgeProposal:
    src_node_id: str
    dst_node_id: str
    edge_kind: str
    explanation: str
    validation_requirements: tuple[str, ...]
    confidence_estimate: float
    novelty_vs_deterministic: float


@dataclass(frozen=True)
class LLMPathProposal:
    src_node_id: str
    dst_node_id: str
    path_kind: str
    explanation: str
    validation_requirements: tuple[str, ...]
    confidence_estimate: float
    novelty_vs_deterministic: float
    node_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LLMTestProposal:
    src_node_id: str
    dst_node_id: str
    path_kind: str
    explanation: str
    validation_requirements: tuple[str, ...]
    confidence_estimate: float
    novelty_vs_deterministic: float
    target_node_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LLMHypothesisInput:
    system_instruction: str
    graph_nodes: tuple[dict[str, Any], ...]
    graph_edges: tuple[dict[str, Any], ...]
    top_deterministic_edges: tuple[dict[str, Any], ...]
    top_deterministic_paths: tuple[dict[str, Any], ...]
    open_questions: tuple[str, ...]
    contradictions: tuple[dict[str, Any], ...]
    exit_attempts: tuple[dict[str, Any], ...]
    pattern_relations: tuple[dict[str, Any], ...]
    allowed_node_ids: tuple[str, ...]
    allowed_edge_kinds: tuple[str, ...]
    allowed_path_kinds: tuple[str, ...]


@dataclass(frozen=True)
class LLMHypothesisOutput:
    edge_proposals: tuple[LLMEdgeProposal, ...] = ()
    path_proposals: tuple[LLMPathProposal, ...] = ()
    test_proposals: tuple[LLMTestProposal, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
