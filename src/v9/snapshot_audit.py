from __future__ import annotations

from dataclasses import dataclass


V9_AUXILIARY_SCHEMA_VERSION = 3
REQUIRED_V9_KEYS = frozenset(
    {
        "scientific_config_id",
        "environment_identity_registry",
        "symbol_codecs",
        "multimodal_schema_version",
        "multimodal_timeline",
        "multimodal_memory",
        "lineage_state",
        "object_versions",
        "normalization_state",
        "progressive_similarity_state",
        "beta_by_radius",
        "equivalence_sets",
        "grounding_state",
        "payload_state",
        "transfer_state",
    }
)


@dataclass(frozen=True, slots=True)
class SnapshotAuditResult:
    compatible: bool
    legacy_migration: bool
    missing_keys: tuple[str, ...]
    reason: str


def audit_snapshot(
    payload: dict[str, object],
    *,
    expected_scientific_config_id: str | None = None,
    allow_legacy_v8: bool = True,
) -> SnapshotAuditResult:
    version = payload.get("v9_auxiliary_state_version")
    if version is None:
        if allow_legacy_v8:
            return SnapshotAuditResult(
                True,
                True,
                tuple(sorted(REQUIRED_V9_KEYS)),
                "legacy v8 snapshot requires explicit default-state migration",
            )
        return SnapshotAuditResult(
            False,
            False,
            tuple(sorted(REQUIRED_V9_KEYS)),
            "v9 schema marker missing",
        )
    if int(version) > V9_AUXILIARY_SCHEMA_VERSION or int(version) <= 0:
        return SnapshotAuditResult(False, False, (), "unsupported v9 auxiliary schema")
    config_id = payload.get("scientific_config_id")
    if (
        expected_scientific_config_id is not None
        and str(config_id) != str(expected_scientific_config_id)
    ):
        return SnapshotAuditResult(False, False, (), "ScientificConfigId mismatch")
    missing = tuple(sorted(key for key in REQUIRED_V9_KEYS if key not in payload))
    if missing:
        return SnapshotAuditResult(
            False, False, missing, "required v9 persistence state missing"
        )
    return SnapshotAuditResult(True, False, (), "compatible")


def migrate_legacy_v8_auxiliary(
    payload: dict[str, object], *, scientific_config_id: str
) -> dict[str, object]:
    if "v9_auxiliary_state_version" in payload:
        raise ValueError("payload is already v9; do not reinterpret it as legacy")
    migrated = dict(payload)
    beta = {"1": 1.0, "2": 1.0, "4": 1.0, "8": 1.0}
    migrated.update(
        {
            "v9_auxiliary_state_version": V9_AUXILIARY_SCHEMA_VERSION,
            "scientific_config_id": str(scientific_config_id),
            "environment_identity_registry": {"version": 1, "identities": []},
            "symbol_codecs": [],
            "multimodal_schema_version": 1,
            "multimodal_timeline": {
                "version": 1,
                "max_symbols_per_window": 8,
                "max_symbol_payload_bytes": 4096,
                "max_pending_passive_events": 64,
                "pending_passive_count": 0,
                "telemetry": {
                    "symbol_observations_seen": 0,
                    "symbol_observations_admitted": 0,
                    "symbol_limit_dropped": 0,
                    "payload_limit_dropped": 0,
                    "pending_overflow_dropped": 0,
                    "committed_events": 0,
                    "committed_actions": 0,
                },
                "last_ordering_key": None,
                "pending_packets": [],
            },
            "multimodal_memory": {
                "version": 1,
                "budgets": {
                    "world_facts": 8,
                    "symbol_facts": 8,
                    "cross_modal_facts": 8,
                },
                "m0": [],
                "m1g": [],
                "m1n": [],
            },
            "lineage_state": {"version": 1, "overlays": []},
            "object_versions": {
                "version": 1,
                "graph_generation": int(payload.get("generation", 0)),
                "objects": [],
            },
            "normalization_state": {
                "version": 1,
                "reservoir_limit": 16,
                "criteria": {
                    "sample_count": 8,
                    "stable_contingency_count": 2,
                    "descriptor_coverage": 2,
                    "minimum_observation_span": 4,
                },
                "stats": [],
                "observation_span": {},
                "stable_contingencies": {},
                "descriptor_nodes": {},
            },
            "progressive_similarity_state": {
                "version": 1,
                "beta_by_radius": beta,
                "r_max": 8,
                "symmetry_patience": 2,
                "information_epsilon": 0.001,
                "equivalence_sets": [],
            },
            "beta_by_radius": beta,
            "equivalence_sets": [],
            "grounding_state": {"version": 1, "states": []},
            "payload_state": {
                "version": 1,
                "max_hot_payload_bytes": 1048576,
                "max_hot_payloads": 1024,
                "provenance": [],
                "payloads": {},
            },
            "transfer_state": {"trust": []},
            "legacy_v8_migration": True,
        }
    )
    return migrated
