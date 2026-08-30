from __future__ import annotations

"""v8.79 information-flow integrity across the developmental memory hierarchy.

Repairs five information-loss defects in the composed production runtime:

* M1N facts retain explicit temporal/magnitude family structure instead of forcing
  all normalized facts of one primitive kind into one M2 family;
* legacy one-part M1N rows cannot manufacture new coarse M2 families;
* M4 concept identity preserves the complete learned M3 role descriptor;
* formation, role, compression, and world-model provenance is no longer truncated
  to the first one/eight parents.

The layer does not relax support, compression, transfer, or validation thresholds.
"""

from collections import defaultdict
from dataclasses import replace
import math

from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryProposal,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
    proposal_fingerprint,
    stable_u64,
)


_INSTALLED = False
_BASE_PARALLEL_ANALYSES = None
_BASE_SUBMIT = None
_BASE_PROCESS_FORMATION = None

_M1N_MARKER = 1 << 63
_M1N_STRUCT_HASH_MASK = (1 << 49) - 1
_M1N_FAMILY_VERSION = 1 << 8
_LEGACY_FAMILY_MARKER = 1 << 61


def _structured_fact_token(self) -> int:
    """Encode reusable family dimensions explicitly while retaining hashed detail."""
    payload = stable_u64(
        int(self.structure_signature),
        int(self.relation_signature),
        person=b"v8.79-m1n-fact",
    )
    temporal = max(0, min(7, int(self.temporal_bucket)))
    magnitude = max(0, min(7, int(self.magnitude_bucket)))
    return int(
        _M1N_MARKER
        | ((payload & _M1N_STRUCT_HASH_MASK) << 14)
        | ((temporal & 0x7) << 11)
        | ((magnitude & 0x7) << 8)
        | (int(self.kind) & 0xFF)
    )


def normalized_family_variant_v879(token: int) -> int:
    """Return the semantic coarse family variant carried by a v8.79 M1N token."""
    raw = int(token)
    temporal = (raw >> 11) & 0x7
    magnitude = (raw >> 8) & 0x7
    return int(_M1N_FAMILY_VERSION | (temporal << 3) | magnitude)


def normalized_family_key_for_row_v879(row) -> tuple[int, int]:
    from v8.structural_events import normalized_fact_kind

    token = int(row.key_parts[0])
    kind = int(normalized_fact_kind(token))
    if len(row.key_parts) >= 2 and int(row.key_parts[1]) >= _M1N_FAMILY_VERSION:
        return kind, int(row.key_parts[1])
    # Legacy M1N stored only a cryptographic token. Its hidden payload has no metric
    # locality, so grouping it by primitive kind would recreate the old collapse.
    # Give every legacy normalized identity a stable private bucket instead.
    legacy = _LEGACY_FAMILY_MARKER | (
        stable_u64(token, person=b"v8.79-legacy-family") & (_LEGACY_FAMILY_MARKER - 1)
    )
    return kind, int(legacy)


def is_normalized_contingency_v879(row) -> bool:
    from v8.structural_events import is_normalized_fact_token

    return bool(
        int(row.level) == int(MemoryLevel.M1)
        and int(row.memory_type) == int(MemoryType.CONTINGENCY)
        and len(row.key_parts) in {1, 2}
        and is_normalized_fact_token(int(row.key_parts[0]))
    )


def derive_normalized_proposals_v879(pipeline, grounded) -> tuple[object, ...]:
    """Publish new M1N rows with an explicit reusable family descriptor."""
    from v8 import model as model_module
    from v8 import normalized_memory_v086 as normalized
    from v8.structural_events import is_normalized_fact_token

    facts = tuple(int(v) for v in pipeline.normalized_facts)
    if not facts:
        facts = normalized._fallback_normalized_facts(pipeline.experience)
    e = pipeline.experience
    multiplicity = max(1, int(pipeline.multiplicity))
    prediction = max(0.0, min(1.0, float(e.prediction_error)))
    option = min(1.0, abs(math.tanh(float(e.future_option_delta))))
    valence = 1.0 if int(e.terminal_polarity) != 0 else 0.0
    significance = min(1.0, 0.20 + 0.35 * prediction + 0.20 * option + 0.25 * valence)
    learning = min(1.0, 0.20 + 0.45 * prediction + 0.15 * option + 0.20 * valence)
    result = []
    seen = set()
    for token in facts[: normalized.MAX_NORMALIZED_FACTS_PER_EVENT]:
        if token in seen or not is_normalized_fact_token(token):
            continue
        seen.add(token)
        family_variant = normalized_family_variant_v879(token)
        key = (int(token), int(family_variant))
        uid = MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, key)
        result.append(
            replace(
                grounded,
                uid=uid,
                fingerprint=model_module.proposal_fingerprint(
                    MemoryLevel.M1, MemoryType.CONTINGENCY, key
                ),
                key_parts=key,
                support_delta=multiplicity,
                significance_sum=significance * multiplicity,
                prediction_error_sum=prediction * multiplicity,
                learning_value_sum=learning * multiplicity,
                transfer_prior_sum=0.0,
                explanatory_sum=0.0,
                future_option_sum=float(e.future_option_delta) * multiplicity,
                score_weight=float(multiplicity),
                parent_uid=grounded.uid,
                relation_type=RelationType.EXPLAINS,
                source_game_hash=int(e.source_game_hash),
                cognitive_state=int(CognitiveState.ACTIVE),
                validation_state=int(ValidationState.VALIDATED),
            )
        )
    return tuple(result)


def _bounded_append(rows: list[dict[str, object]], payload: dict[str, object]) -> None:
    if len(rows) < 3:
        rows.append(payload)


class V879GenerativeCompressionEstimator:
    """Factory shim; the production subclass is created at install time."""


def _compression_class(base_class):
    from v8.intelligence_loop_v087 import CompressionProposal
    from v8.normalized_memory_v086 import _M2N_MARKER, is_grounded_contingency

    class V879Compression(base_class):
        def discover(self, nodes, edges=(), *, budget: int = 256):
            del edges
            rows = tuple(nodes)
            limit = max(0, int(budget))
            m1g_count = m1n_count = m1n_cross_game = 0
            support_ge_3 = support_eligible = support_rejected = 0
            grouped = defaultdict(list)
            rejected_examples = defaultdict(list)
            for row in rows:
                if is_grounded_contingency(row):
                    m1g_count += 1
                if not is_normalized_contingency_v879(row):
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
                grouped[normalized_family_key_for_row_v879(row)].append(row)

            insufficient_members = insufficient_benefit = eligible_groups = 0
            pair_opportunities = 0
            eligible_rows = []
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

            result = []
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
                "m2_rejected_examples": {
                    key: value for key, value in sorted(rejected_examples.items())
                },
                "m2_gate_note": (
                    "v8.79 groups new M1N by observable primitive + temporal/magnitude "
                    "family descriptor; legacy opaque tokens are isolated rather than "
                    "collapsed by primitive kind."
                ),
            }
            return tuple(result)

    V879Compression.__name__ = "V879GenerativeCompressionEstimator"
    return V879Compression


def _promotion_class(base_class):
    class V879PromotionEngine(base_class):
        def propose(self, nodes, edges, *, budget: int = 256):
            rows = tuple(nodes)
            by_uid = {row.uid: row for row in rows}
            result = []
            for candidate in super().propose(rows, tuple(edges), budget=budget):
                if int(candidate.level) == int(MemoryLevel.M4) and candidate.parents:
                    role = by_uid.get(candidate.parents[0])
                    if (
                        role is not None
                        and int(role.level) == int(MemoryLevel.M3)
                        and int(role.memory_type) == int(MemoryType.ROLE)
                        and role.key_parts
                    ):
                        key = tuple(int(value) for value in role.key_parts)
                        uid = MemoryUid.from_key(MemoryLevel.M4, MemoryType.CONCEPT, key)
                        candidate = replace(candidate, uid=uid, key_parts=key)
                result.append(candidate)
            return tuple(result)

    V879PromotionEngine.__name__ = "V879InformationPreservingPromotionEngine"
    return V879PromotionEngine


def _process_formation_v879(self, cut, frozen) -> None:
    """Production formation with complete parent provenance."""
    by_uid = {row.uid: row for row in cut.nodes}
    for candidate in self.promotion.propose(
        cut.nodes,
        cut.edges,
        budget=self.candidate_budget,
    ):
        parent_watermark = max(
            (by_uid[uid].updated_watermark for uid in candidate.parents if uid in by_uid),
            default=cut.watermark,
        )
        freshness = f"v82-formation:{int(candidate.level)}:{int(candidate.memory_type)}"
        if not self._fresh(freshness, candidate.uid, parent_watermark):
            continue
        identity = self._formation_identity(candidate)
        weight = max(1.0, float(candidate.support))
        first_parent = candidate.parents[0] if candidate.parents else MemoryUid.zero()
        future_option_delta = self._formation_future_option(candidate, by_uid)
        proposal = MemoryProposal(
            uid=candidate.uid,
            fingerprint=identity.fingerprint,
            event_id=self._event_id(),
            watermark=int(cut.watermark),
            level=candidate.level,
            memory_type=candidate.memory_type,
            key_parts=candidate.key_parts,
            support_delta=max(1, int(candidate.support)),
            significance_sum=float(candidate.significance) * weight,
            learning_value_sum=float(candidate.learning_value) * weight,
            transfer_prior_sum=float(candidate.transfer_prior) * weight,
            explanatory_sum=float(candidate.explanatory_reach) * weight,
            future_option_sum=future_option_delta * weight,
            score_weight=weight,
            parent_uid=first_parent,
            relation_type=self._relation_for(candidate),
            cognitive_state=int(candidate.cognitive_state),
            validation_state=int(candidate.validation_state),
        )
        self._submit(proposal)
        for parent in candidate.parents[1:]:
            self._submit(
                self._existing_proposal(
                    identity,
                    parent_uid=parent,
                    relation_type=self._relation_for(candidate, extra_parent=True),
                )
            )
        provenance_games = set()
        for parent in candidate.parents:
            provenance_games.update(frozen.source_games(parent))
        self._append_evidence(
            candidate.evidence_kind,
            candidate,
            candidate.evidence_value,
            validation_state=int(candidate.validation_state),
            provenance_games=tuple(sorted(provenance_games)),
        )


def _parallel_analyses_v879(self, nodes, edges):
    analyses = _BASE_PARALLEL_ANALYSES(self, nodes, edges)
    self._v879_role_parents = {
        candidate.uid: tuple(candidate.carriers) for candidate in analyses.get("roles", ())
    }
    self._v879_world_parents = {
        component.uid: tuple(component.consequences) for component in analyses.get("world", ())
    }
    self._v879_compression_superseded = {
        evidence.uid: tuple(evidence.superseded) for evidence in analyses.get("compression", ())
    }
    return analyses


def _submit_extra_relation(self, proposal, parent_uid, relation_type=RelationType.EXPLAINS):
    extra = self._existing_proposal(
        proposal,
        parent_uid=parent_uid,
        relation_type=relation_type,
    )
    _BASE_SUBMIT(self, extra)


def _submit_v879(self, proposal) -> None:
    """Complete provenance omitted by the historical bounded peer writers."""
    _BASE_SUBMIT(self, proposal)
    parent = getattr(proposal, "parent_uid", MemoryUid.zero())

    if (
        int(proposal.level) == int(MemoryLevel.M3)
        and int(proposal.memory_type) == int(MemoryType.ROLE)
    ):
        carriers = tuple(getattr(self, "_v879_role_parents", {}).get(proposal.uid, ()))
        if carriers and parent == carriers[0]:
            for carrier in carriers[1:]:
                _submit_extra_relation(self, proposal, carrier)

    if (
        int(proposal.level) == int(MemoryLevel.M5)
        and int(proposal.memory_type) == int(MemoryType.WORLD_MODEL)
    ):
        consequences = tuple(getattr(self, "_v879_world_parents", {}).get(proposal.uid, ()))
        if consequences and parent == consequences[0]:
            for consequence in consequences[1:]:
                _submit_extra_relation(self, proposal, consequence)

    superseded = tuple(
        getattr(self, "_v879_compression_superseded", {}).get(proposal.uid, ())
    )
    if superseded and parent.is_zero:
        # The historical peer loop already writes the first eight supersession edges.
        for target in superseded[8:]:
            _submit_extra_relation(
                self,
                proposal,
                target,
                relation_type=RelationType.SUPERSEDES,
            )


def install_information_flow_integrity_v879() -> None:
    global _INSTALLED, _BASE_PARALLEL_ANALYSES, _BASE_SUBMIT, _BASE_PROCESS_FORMATION
    if _INSTALLED:
        return

    from v8 import compression as compression_module
    from v8 import formation_telemetry_v870 as telemetry
    from v8 import intelligence_loop_v087 as intelligence
    from v8 import normalized_memory_v086 as normalized
    from v8 import peers as peers_module
    from v8 import peers_v82
    from v8 import promotion as promotion_module
    from v8 import behavior_recovery
    from v8 import structural_events

    # New normalized observations expose semantic coarse family dimensions.
    structural_events.StructuralFact.token = property(_structured_fact_token)
    normalized.derive_normalized_proposals = derive_normalized_proposals_v879
    normalized.is_normalized_contingency = is_normalized_contingency_v879
    intelligence.is_normalized_contingency = is_normalized_contingency_v879
    telemetry.is_normalized_contingency = is_normalized_contingency_v879

    compression_class = _compression_class(telemetry.V870GenerativeCompressionEstimator)
    telemetry.V870GenerativeCompressionEstimator = compression_class
    intelligence.V087GenerativeCompressionEstimator = compression_class
    compression_module.CompressionEstimator = compression_class
    peers_module.CompressionEstimator = compression_class

    # Preserve all role identity dimensions when a concept is formed.
    active_engine = peers_v82.EvidenceGatedPromotionEngine
    promotion_class = _promotion_class(active_engine)
    promotion_module.EvidenceGatedPromotionEngine = promotion_class
    peers_v82.EvidenceGatedPromotionEngine = promotion_class
    behavior_recovery.CausalEvidenceGatedPromotionEngine = promotion_class

    # Complete provenance through every peer writer that historically bounded it.
    _BASE_PROCESS_FORMATION = peers_v82.V82DevelopmentalPeerSupervisor._process_formation
    peers_v82.V82DevelopmentalPeerSupervisor._process_formation = _process_formation_v879

    _BASE_PARALLEL_ANALYSES = peers_v82.V82DevelopmentalPeerSupervisor._parallel_analyses
    peers_v82.V82DevelopmentalPeerSupervisor._parallel_analyses = _parallel_analyses_v879
    _BASE_SUBMIT = peers_v82.V82DevelopmentalPeerSupervisor._submit
    peers_v82.V82DevelopmentalPeerSupervisor._submit = _submit_v879

    _INSTALLED = True
