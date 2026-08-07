from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TableRequirement:
    database: str
    table: str
    fields: tuple[str, ...] = ()
    alternatives: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceContract:
    hypothesis_id: str
    required_tables: tuple[TableRequirement, ...]
    required_report_fields: tuple[str, ...] = ("decision",)
    minimum_coverage: float | None = None
    dependencies: tuple[str, ...] = ()
    proxy_markers: tuple[str, ...] = ()
    allow_proxy_for_partial: bool = True
    notes: str = ""


CONTRACTS: Mapping[str, EvidenceContract] = {
    "H01": EvidenceContract(
        "H01",
        (
            TableRequirement(
                "current_state.sqlite",
                "stable_contingencies",
                ("canonical_key", "support_count", "stability_score"),
            ),
        ),
        minimum_coverage=0.80,
    ),
    "H02": EvidenceContract(
        "H02",
        (
            TableRequirement(
                "current_state.sqlite",
                "memory_scores",
                ("node_id", "replay_priority"),
            ),
            TableRequirement(
                "current_state.sqlite",
                "memory_edges",
                ("source_node_id", "edge_type"),
            ),
        ),
        minimum_coverage=0.80,
        proxy_markers=(
            "proxy_only",
            "raw_cleanup_prevents_direct_linkage",
        ),
    ),
    "H03": EvidenceContract(
        "H03",
        (
            TableRequirement(
                "current_state.sqlite",
                "transformation_families",
                ("canonical_signature", "support_count"),
            ),
            TableRequirement(
                "current_state.sqlite",
                "family_members",
                ("family_signature", "contingency_key"),
                alternatives=("transformation_family_members",),
            ),
        ),
        dependencies=("H01",),
        minimum_coverage=0.80,
    ),
    "H04": EvidenceContract(
        "H04",
        (
            TableRequirement(
                "current_state.sqlite",
                "carrier_candidates",
                ("carrier_signature", "support_count"),
            ),
        ),
        dependencies=("H03",),
        proxy_markers=("candidate_proxy_only",),
    ),
    "H05": EvidenceContract(
        "H05",
        (
            TableRequirement(
                "current_state.sqlite",
                "role_candidates",
                ("role_signature", "support_count"),
            ),
            TableRequirement(
                "current_state.sqlite",
                "role_links",
                ("role_signature", "linked_type", "linked_key"),
            ),
        ),
        dependencies=("H04",),
    ),
    "H06": EvidenceContract(
        "H06",
        (
            TableRequirement(
                "current_state.sqlite",
                "role_transfer_attempts",
                ("attempt_id", "reuse_success", "provenance_status"),
            ),
        ),
        dependencies=("H05",),
        proxy_markers=("proxy_transfer_evidence_only",),
    ),
    "H07": EvidenceContract(
        "H07",
        (
            TableRequirement(
                "current_state.sqlite",
                "concept_candidates",
                ("concept_signature", "promotion_score"),
            ),
            TableRequirement(
                "current_state.sqlite",
                "concept_promotion_validation_diagnostics",
                ("concept_signature", "payload_json"),
            ),
        ),
        dependencies=("H06",),
        proxy_markers=(
            "candidate_proxy_only",
            "promotion_retained_without_current_validation",
        ),
    ),
    "H08": EvidenceContract(
        "H08",
        (
            TableRequirement(
                "current_state.sqlite",
                "world_model_components",
                ("component_signature", "component_type", "linked_concept_count"),
            ),
            TableRequirement(
                "current_state.sqlite",
                "world_model_links",
                ("component_signature", "linked_type", "linked_key"),
            ),
        ),
        dependencies=("H06", "H07"),
        proxy_markers=(
            "candidate_proxy_only",
            "predicted_outcome_count_is_proxy_count",
        ),
    ),
    "H09": EvidenceContract(
        "H09",
        (
            TableRequirement(
                "current_state.sqlite",
                "future_option_events",
                ("event_id", "motif_type", "classification_provenance_status"),
            ),
            TableRequirement(
                "current_state.sqlite",
                "future_option_motifs",
                ("motif_signature", "motif_type", "provenance_status"),
            ),
            TableRequirement(
                "current_state.sqlite",
                "future_option_motif_observations",
                ("motif_signature", "event_id", "provenance_status"),
            ),
        ),
        proxy_markers=("proxy_only", "surrogate_only"),
    ),
    "H10": EvidenceContract(
        "H10",
        (
            TableRequirement(
                "current_state.sqlite",
                "future_option_attention_links",
                ("event_id", "motif_signature", "high_attention"),
                alternatives=("future_option_attention",),
            ),
        ),
        dependencies=("H09",),
        proxy_markers=("heuristic_only", "proxy_only"),
    ),
    "H11": EvidenceContract(
        "H11",
        (
            TableRequirement(
                "current_state.sqlite",
                "future_option_transfer_links",
                (
                    "motif_provenance_status",
                    "transfer_provenance_status",
                    "concept_validation_status",
                ),
            ),
        ),
        dependencies=("H06", "H09"),
        proxy_markers=("proxy_only", "surrogate_only"),
    ),
    "H12": EvidenceContract(
        "H12",
        (
            TableRequirement(
                "current_state.sqlite",
                "trajectory_efficiency",
                ("trajectory_id",),
                alternatives=("trajectory_efficiency_records",),
            ),
        ),
        proxy_markers=("reconstructed_trajectory_only",),
    ),
}


def get_contract(hypothesis_id: str) -> EvidenceContract:
    key = str(hypothesis_id).upper()
    if key not in CONTRACTS:
        raise KeyError(f"unknown hypothesis contract: {hypothesis_id}")
    return CONTRACTS[key]


def contract_manifest() -> dict[str, Any]:
    return {
        key: {
            "hypothesis_id": contract.hypothesis_id,
            "required_tables": [
                {
                    "database": item.database,
                    "table": item.table,
                    "fields": list(item.fields),
                    "alternatives": list(item.alternatives),
                }
                for item in contract.required_tables
            ],
            "required_report_fields": list(contract.required_report_fields),
            "minimum_coverage": contract.minimum_coverage,
            "dependencies": list(contract.dependencies),
            "proxy_markers": list(contract.proxy_markers),
            "allow_proxy_for_partial": contract.allow_proxy_for_partial,
            "notes": contract.notes,
        }
        for key, contract in CONTRACTS.items()
    }
