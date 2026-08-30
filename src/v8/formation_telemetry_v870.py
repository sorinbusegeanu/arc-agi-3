from __future__ import annotations

"""v8.70 bounded telemetry for M1N->M2 and M3-carrier->role formation gates."""

from collections import defaultdict
from typing import Iterable

from v8.arena import EdgeRecord, NodeRecord
from v8.intelligence_loop_v087 import (
    CompressionProposal,
    V087GenerativeCompressionEstimator,
    V087RelationalRoleEstimator,
)
from v8.model import MemoryLevel, MemoryType, MemoryUid, RelationType
from v8.normalized_memory_v086 import (
    _M2N_MARKER,
    is_grounded_contingency,
    is_normalized_contingency,
)
from v8.roles import RoleCandidate
from v8.structural_events import normalized_family_key


_INSTALLED = False
_BASE_RUNTIME_METRICS = None
_BASE_KEY_RUN_SUMMARY = None
_MAX_REJECTED_EXAMPLES = 3


def _bounded_append(rows: list[dict[str, object]], payload: dict[str, object]) -> None:
    if len(rows) < _MAX_REJECTED_EXAMPLES:
        rows.append(payload)


class V870GenerativeCompressionEstimator(V087GenerativeCompressionEstimator):
    """Run the production M2 gates while retaining bounded aggregate diagnostics."""

    def discover(
        self,
        nodes: Iterable[NodeRecord],
        edges: Iterable[EdgeRecord] = (),
        *,
        budget: int = 256,
    ) -> tuple[CompressionProposal, ...]:
        del edges
        rows = tuple(nodes)
        limit = max(0, int(budget))

        m1g_count = 0
        m1n_count = 0
        m1n_cross_game = 0
        support_ge_3 = 0
        support_eligible = 0
        support_rejected = 0
        grouped: dict[tuple[int, int], list[NodeRecord]] = defaultdict(list)
        rejected_examples: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            if is_grounded_contingency(row):
                m1g_count += 1
            if not is_normalized_contingency(row):
                continue
            m1n_count += 1
            if int(row.game_mask).bit_count() >= 2:
                m1n_cross_game += 1
            if int(row.support_count) >= 3:
                support_ge_3 += 1
            if int(row.support_count) < self.min_support:
                support_rejected += 1
                _bounded_append(
                    rejected_examples["m1n_below_min_support"],
                    {
                        "support": int(row.support_count),
                        "required": int(self.min_support),
                        "key": [int(value) for value in row.key_parts[:2]],
                        "source_game_count": int(row.game_mask).bit_count(),
                    },
                )
                continue
            support_eligible += 1
            grouped[normalized_family_key(int(row.key_parts[0]))].append(row)

        insufficient_members = 0
        insufficient_benefit = 0
        eligible_groups = 0
        pair_opportunities = 0
        eligible_rows: list[tuple[tuple[int, int], list[NodeRecord], int, float]] = []
        for family_key, members in sorted(grouped.items()):
            members = list({row.uid: row for row in members}.values())
            pair_opportunities += len(members) * max(0, len(members) - 1) // 2
            if len(members) < self.min_members:
                insufficient_members += 1
                _bounded_append(
                    rejected_examples["group_insufficient_members"],
                    {
                        "family_key": [int(value) for value in family_key],
                        "members": len(members),
                        "required": int(self.min_members),
                        "total_support": sum(max(0, int(row.support_count)) for row in members),
                    },
                )
                continue
            total_support = sum(max(0, int(row.support_count)) for row in members)
            benefit = float(max(0, total_support - len(members)))
            if benefit <= self.min_benefit:
                insufficient_benefit += 1
                _bounded_append(
                    rejected_examples["group_insufficient_compression_benefit"],
                    {
                        "family_key": [int(value) for value in family_key],
                        "members": len(members),
                        "total_support": int(total_support),
                        "benefit": float(benefit),
                        "required_strictly_greater_than": float(self.min_benefit),
                    },
                )
                continue
            eligible_groups += 1
            eligible_rows.append((family_key, members, total_support, benefit))

        result: list[CompressionProposal] = []
        if limit > 0:
            for family_key, members, total_support, benefit in eligible_rows[:limit]:
                kind, variant = map(int, family_key)
                key = (int(_M2N_MARKER | kind), int(variant))
                uid = MemoryUid.from_key(MemoryLevel.M2, MemoryType.FAMILY, key)
                future = sum(
                    float(row.future_option_delta) * max(0, int(row.support_count))
                    for row in members
                ) / max(1, total_support)
                result.append(
                    CompressionProposal(
                        uid,
                        key,
                        tuple(sorted(row.uid for row in members)),
                        total_support,
                        benefit,
                        float(len(members)),
                        0.0,
                        float(future),
                    )
                )

        budget_limited = max(0, eligible_groups - len(result))
        if budget_limited:
            for family_key, members, total_support, benefit in eligible_rows[len(result):]:
                _bounded_append(
                    rejected_examples["budget_limited"],
                    {
                        "family_key": [int(value) for value in family_key],
                        "members": len(members),
                        "total_support": int(total_support),
                        "benefit": float(benefit),
                    },
                )

        self._v870_formation_telemetry = {
            "m1g_count": int(m1g_count),
            "m1n_count": int(m1n_count),
            "m1n_cross_game_count": int(m1n_cross_game),
            "stable_m1n_support_ge_3": int(support_ge_3),
            "m2_support_eligible_m1n": int(support_eligible),
            "m2_candidate_groups_considered": int(len(grouped)),
            "m2_within_group_pair_opportunities": int(pair_opportunities),
            "m2_min_support": int(self.min_support),
            "m2_min_members": int(self.min_members),
            "m2_min_compression_benefit": float(self.min_benefit),
            "m2_family_groups": int(len(grouped)),
            "eligible_m2_groups": int(eligible_groups),
            "m2_candidates_emitted": int(len(result)),
            "m2_rejections": {
                "m1n_below_min_support": int(support_rejected),
                "group_insufficient_members": int(insufficient_members),
                "group_insufficient_compression_benefit": int(insufficient_benefit),
                "budget_limited": int(budget_limited),
            },
            "m2_rejected_examples": {key: value for key, value in sorted(rejected_examples.items())},
            "m2_gate_note": "Counters reflect the actual production gates only; no synthetic context/valence/normalization rejection classes are inferred.",
        }
        return tuple(result)


class V870RelationalRoleEstimator(V087RelationalRoleEstimator):
    """Run the production relational-role gates with aggregate rejection counts."""

    def propose_relational(
        self,
        rows: Iterable[NodeRecord],
        edges: Iterable[EdgeRecord],
    ) -> tuple[RoleCandidate, ...]:
        rows, edges = tuple(rows), tuple(edges)
        by_uid = {row.uid: row for row in rows}
        grouped: dict[tuple[int, int, int, int], list[NodeRecord]] = defaultdict(list)
        carrier_count = 0
        existing_role_count = 0
        malformed_carriers = 0
        rejected_examples: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            if int(row.level) == int(MemoryLevel.M3) and int(row.memory_type) == int(MemoryType.ROLE):
                existing_role_count += 1
            if int(row.level) != int(MemoryLevel.M3) or int(row.memory_type) != int(MemoryType.CARRIER):
                continue
            carrier_count += 1
            if len(row.key_parts) < 3:
                malformed_carriers += 1
                _bounded_append(
                    rejected_examples["malformed_carrier_key"],
                    {"key_length": len(row.key_parts), "key": [int(value) for value in row.key_parts]},
                )
                continue
            grouped[self._descriptor(row, edges=edges, by_uid=by_uid)].append(row)

        groups_ge_2 = 0
        insufficient_carriers = 0
        insufficient_lower_support = 0
        result: list[RoleCandidate] = []
        for descriptor, members in sorted(grouped.items()):
            carriers = {int(row.key_parts[1]) for row in members}
            if len(carriers) >= 2:
                groups_ge_2 += 1
            if len(carriers) < self.min_carriers:
                insufficient_carriers += 1
                _bounded_append(
                    rejected_examples["group_insufficient_distinct_carriers"],
                    {
                        "descriptor": [int(value) for value in descriptor],
                        "members": len(members),
                        "distinct_carriers": len(carriers),
                        "required": int(self.min_carriers),
                    },
                )
                continue
            member_uids = {row.uid for row in members}
            lower_support = {
                edge.target_uid
                for edge in edges
                if edge.source_uid in member_uids
                and int(edge.relation_type) == int(RelationType.EXPLAINS)
                and edge.target_uid in by_uid
                and int(by_uid[edge.target_uid].level) < int(MemoryLevel.M3)
            }
            if len(lower_support) < 2:
                insufficient_lower_support += 1
                _bounded_append(
                    rejected_examples["group_insufficient_lower_support"],
                    {
                        "descriptor": [int(value) for value in descriptor],
                        "members": len(members),
                        "distinct_carriers": len(carriers),
                        "lower_support": len(lower_support),
                        "required": 2,
                    },
                )
                continue
            key = tuple(int(value) for value in descriptor)
            uid = MemoryUid.from_key(MemoryLevel.M3, MemoryType.ROLE, key)
            mask = 0
            for row in members:
                mask |= int(row.game_mask)
            result.append(
                RoleCandidate(
                    uid,
                    key,
                    tuple(sorted(row.uid for row in members)),
                    int(mask).bit_count(),
                )
            )

        self._v870_formation_telemetry = {
            "m3_carrier_count": int(carrier_count),
            "m3_existing_role_count": int(existing_role_count),
            "m3_role_groups_considered": int(len(grouped)),
            "m3_carrier_groups": int(len(grouped)),
            "carrier_groups_ge_2": int(groups_ge_2),
            "role_min_carriers": int(self.min_carriers),
            "role_candidates": int(len(result)),
            "role_rejections": {
                "malformed_carrier_key": int(malformed_carriers),
                "group_insufficient_distinct_carriers": int(insufficient_carriers),
                "group_insufficient_lower_support": int(insufficient_lower_support),
            },
            "role_rejected_examples": {key: value for key, value in sorted(rejected_examples.items())},
            "role_gate_note": "Role grouping uses the production relational descriptor; descriptor mismatches appear as separate groups rather than an invented rejection class.",
        }
        return tuple(result)


def _runtime_metrics_v870(self):
    payload = dict(_BASE_RUNTIME_METRICS(self))
    peers = getattr(self, "peers", None)
    merged: dict[str, object] = {}
    if peers is not None:
        for name in ("compression", "roles"):
            estimator = getattr(peers, name, None)
            telemetry = getattr(estimator, "_v870_formation_telemetry", {})
            if isinstance(telemetry, dict):
                merged.update(telemetry)
    payload["formation_telemetry"] = merged
    return payload


def _key_run_summary_v870(summary):
    payload = dict(_BASE_KEY_RUN_SUMMARY(summary))
    metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
    telemetry = metrics.get("formation_telemetry", {}) if isinstance(metrics, dict) else {}
    memory = dict(payload.get("memory", {}))
    memory["formation_telemetry"] = dict(telemetry) if isinstance(telemetry, dict) else {}
    payload["memory"] = memory

    optimizer = payload.get("trajectory_optimizer", {})
    if isinstance(optimizer, dict):
        optimizer = dict(optimizer)
        optimizer["counter_scope_note"] = (
            "generated, validation_successes, validated_variants, saved/frontier counters are stage-local and must not be assumed to be one-to-one."
        )
        payload["trajectory_optimizer"] = optimizer
    return payload


def install_formation_telemetry_v870() -> None:
    global _INSTALLED, _BASE_RUNTIME_METRICS, _BASE_KEY_RUN_SUMMARY
    if _INSTALLED:
        return

    from v8 import compression as compression_module
    from v8 import intelligence_loop_v087 as loop_module
    from v8 import peers as peers_module
    from v8 import roles as roles_module
    from v8 import runtime_v82
    from v8.research import researcher_packet

    loop_module.V087GenerativeCompressionEstimator = V870GenerativeCompressionEstimator
    compression_module.CompressionEstimator = V870GenerativeCompressionEstimator
    peers_module.CompressionEstimator = V870GenerativeCompressionEstimator

    loop_module.V087RelationalRoleEstimator = V870RelationalRoleEstimator
    roles_module.FunctionalRoleEstimator = V870RelationalRoleEstimator
    peers_module.FunctionalRoleEstimator = V870RelationalRoleEstimator

    _BASE_RUNTIME_METRICS = runtime_v82.V82ContinuousMemoryRuntime.metrics
    runtime_v82.V82ContinuousMemoryRuntime.metrics = _runtime_metrics_v870

    _BASE_KEY_RUN_SUMMARY = researcher_packet._key_run_summary
    researcher_packet._key_run_summary = _key_run_summary_v870
    _INSTALLED = True
