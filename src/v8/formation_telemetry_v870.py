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
_BASE_PARALLEL_ANALYSES = None
_BASE_RUNTIME_METRICS = None


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
        support_ge_3 = 0
        support_eligible = 0
        support_rejected = 0
        grouped: dict[tuple[int, int], list[NodeRecord]] = defaultdict(list)
        for row in rows:
            if is_grounded_contingency(row):
                m1g_count += 1
            if not is_normalized_contingency(row):
                continue
            m1n_count += 1
            if int(row.support_count) >= 3:
                support_ge_3 += 1
            if int(row.support_count) < self.min_support:
                support_rejected += 1
                continue
            support_eligible += 1
            grouped[normalized_family_key(int(row.key_parts[0]))].append(row)

        insufficient_members = 0
        insufficient_benefit = 0
        eligible_groups = 0
        eligible_rows: list[tuple[tuple[int, int], list[NodeRecord], int, float]] = []
        for family_key, members in sorted(grouped.items()):
            members = list({row.uid: row for row in members}.values())
            if len(members) < self.min_members:
                insufficient_members += 1
                continue
            total_support = sum(max(0, int(row.support_count)) for row in members)
            benefit = float(max(0, total_support - len(members)))
            if benefit <= self.min_benefit:
                insufficient_benefit += 1
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

        self._v870_formation_telemetry = {
            "m1g_count": int(m1g_count),
            "m1n_count": int(m1n_count),
            "stable_m1n_support_ge_3": int(support_ge_3),
            "m2_support_eligible_m1n": int(support_eligible),
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
                "budget_limited": int(max(0, eligible_groups - len(result))),
            },
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
        malformed_carriers = 0
        for row in rows:
            if int(row.level) != int(MemoryLevel.M3) or int(row.memory_type) != int(MemoryType.CARRIER):
                continue
            carrier_count += 1
            if len(row.key_parts) < 3:
                malformed_carriers += 1
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
            "m3_carrier_groups": int(len(grouped)),
            "carrier_groups_ge_2": int(groups_ge_2),
            "role_min_carriers": int(self.min_carriers),
            "role_candidates": int(len(result)),
            "role_rejections": {
                "malformed_carrier_key": int(malformed_carriers),
                "group_insufficient_distinct_carriers": int(insufficient_carriers),
                "group_insufficient_lower_support": int(insufficient_lower_support),
            },
        }
        return tuple(result)


def _parallel_analyses_v870(self, nodes, edges):
    analyses = _BASE_PARALLEL_ANALYSES(self, nodes, edges)
    compression = getattr(self.compression, "_v870_formation_telemetry", {})
    roles = getattr(self.roles, "_v870_formation_telemetry", {})
    merged: dict[str, object] = {}
    if isinstance(compression, dict):
        merged.update(compression)
    if isinstance(roles, dict):
        merged.update(roles)
    self._v870_formation_telemetry = merged
    return analyses


def _runtime_metrics_v870(self):
    payload = dict(_BASE_RUNTIME_METRICS(self))
    peers = getattr(self, "peers", None)
    telemetry = getattr(peers, "_v870_formation_telemetry", {}) if peers is not None else {}
    payload["formation_telemetry"] = dict(telemetry) if isinstance(telemetry, dict) else {}
    return payload


def install_formation_telemetry_v870() -> None:
    global _INSTALLED, _BASE_PARALLEL_ANALYSES, _BASE_RUNTIME_METRICS
    if _INSTALLED:
        return

    from v8 import compression as compression_module
    from v8 import intelligence_loop_v087 as loop_module
    from v8 import peers as peers_module
    from v8 import roles as roles_module
    from v8 import runtime_v82

    loop_module.V087GenerativeCompressionEstimator = V870GenerativeCompressionEstimator
    compression_module.CompressionEstimator = V870GenerativeCompressionEstimator
    peers_module.CompressionEstimator = V870GenerativeCompressionEstimator

    loop_module.V087RelationalRoleEstimator = V870RelationalRoleEstimator
    roles_module.FunctionalRoleEstimator = V870RelationalRoleEstimator
    peers_module.FunctionalRoleEstimator = V870RelationalRoleEstimator

    _BASE_PARALLEL_ANALYSES = peers_module.DevelopmentalPeerSupervisor._parallel_analyses
    peers_module.DevelopmentalPeerSupervisor._parallel_analyses = _parallel_analyses_v870

    _BASE_RUNTIME_METRICS = runtime_v82.V82ContinuousMemoryRuntime.metrics
    runtime_v82.V82ContinuousMemoryRuntime.metrics = _runtime_metrics_v870
    _INSTALLED = True
