from __future__ import annotations

"""v8.79 information-flow integrity across developmental memory layers.

The layer fixes two classes of information loss without replacing established public
runtime authorities or changing the one-part M1N schema:

* new non-temporal normalized facts expose their observable temporal/magnitude
  family buckets to M2 instead of collapsing every primitive kind into one family;
* complete parent/support relations are retained for formation, relational roles,
  compression supersession, and world-model components instead of silently keeping
  only the first one/eight relations.

Existing v8.8 temporal-family identities and legacy M1N tokens remain compatible.
No support, compression, transfer, lifecycle, or validation threshold is relaxed.
"""

from v8.model import MemoryLevel, MemoryType, MemoryUid, RelationType, stable_u64


_INSTALLED = False
_BASE_STRUCTURAL_TOKEN = None
_BASE_NORMALIZED_FAMILY_KEY = None
_BASE_PROMOTION_PROPOSE = None
_BASE_ROLE_PROPOSE = None
_BASE_WORLD_PROPOSE = None
_BASE_COMPRESSION_EVALUATE = None
_BASE_SUBMIT = None

_M1N_MARKER = 1 << 63
_V879_MAGIC = 0xB
_V879_HASH_MASK = (1 << 45) - 1
_V879_FAMILY_VERSION = 1 << 8


def _structured_fact_token_v879(self) -> int:
    """Keep one-part M1N identity while exposing coarse observable family buckets."""
    from v8.structural_events import NormalizedPrimitive

    if int(self.kind) == int(NormalizedPrimitive.AUTONOMOUS_CHANGE):
        return int(_BASE_STRUCTURAL_TOKEN(self))

    payload = stable_u64(
        int(self.structure_signature),
        int(self.relation_signature),
        int(self.temporal_bucket),
        int(self.magnitude_bucket),
        person=b"v8.79-m1n-fact",
    )
    temporal = max(0, min(7, int(self.temporal_bucket)))
    magnitude = max(0, min(7, int(self.magnitude_bucket)))
    return int(
        _M1N_MARKER
        | ((_V879_MAGIC & 0xF) << 59)
        | ((payload & _V879_HASH_MASK) << 14)
        | ((temporal & 0x7) << 11)
        | ((magnitude & 0x7) << 8)
        | (int(self.kind) & 0xFF)
    )


def normalized_family_key_v879(value: int) -> tuple[int, int]:
    """Use recoverable observable family dimensions, preserving legacy semantics."""
    from v8.structural_events import NormalizedPrimitive, normalized_fact_kind

    raw = int(value)
    kind = normalized_fact_kind(raw)
    if kind == NormalizedPrimitive.AUTONOMOUS_CHANGE:
        return int(kind), (raw >> 8) & ((1 << 55) - 1)
    if ((raw >> 59) & 0xF) == _V879_MAGIC:
        temporal = (raw >> 11) & 0x7
        magnitude = (raw >> 8) & 0x7
        return int(kind), int(_V879_FAMILY_VERSION | (temporal << 3) | magnitude)
    return tuple(_BASE_NORMALIZED_FAMILY_KEY(raw))


def _promotion_propose_v879(self, nodes, edges, *, budget: int = 256):
    result = tuple(_BASE_PROMOTION_PROPOSE(self, nodes, edges, budget=budget))
    self._v879_candidate_parents = {
        candidate.uid: tuple(candidate.parents)
        for candidate in result
        if getattr(candidate, "parents", ())
    }
    return result


def _role_propose_v879(self, rows, edges):
    result = tuple(_BASE_ROLE_PROPOSE(self, rows, edges))
    self._v879_role_parents = {
        candidate.uid: tuple(candidate.carriers)
        for candidate in result
        if getattr(candidate, "carriers", ())
    }
    return result


def _world_propose_v879(self, rows, edges=()):
    result = tuple(_BASE_WORLD_PROPOSE(self, rows, edges))
    self._v879_world_parents = {
        component.uid: tuple(component.consequences)
        for component in result
        if getattr(component, "consequences", ())
    }
    return result


def _compression_evaluate_v879(self, rows, edges=()):
    result = tuple(_BASE_COMPRESSION_EVALUATE(self, rows, edges))
    self._v879_superseded = {
        evidence.uid: tuple(evidence.superseded)
        for evidence in result
        if getattr(evidence, "superseded", ())
    }
    return result


def _submit_relation_v879(self, proposal, parent_uid, relation_type) -> None:
    extra = self._existing_proposal(
        proposal,
        parent_uid=parent_uid,
        relation_type=relation_type,
    )
    _BASE_SUBMIT(self, extra)


def _submit_v879(self, proposal) -> None:
    """Complete relations omitted by historical bounded writers."""
    _BASE_SUBMIT(self, proposal)
    parent = getattr(proposal, "parent_uid", MemoryUid.zero())

    parents = tuple(
        getattr(getattr(self, "promotion", None), "_v879_candidate_parents", {}).get(
            proposal.uid, ()
        )
    )
    if parents and parent == parents[0]:
        relation = (
            RelationType.DEPENDS_ON
            if int(proposal.level) == int(MemoryLevel.M7)
            else RelationType.EXPLAINS
        )
        for target in parents[8:]:
            _submit_relation_v879(self, proposal, target, relation)

    role_parents = tuple(
        getattr(getattr(self, "roles", None), "_v879_role_parents", {}).get(
            proposal.uid, ()
        )
    )
    if (
        role_parents
        and parent == role_parents[0]
        and int(proposal.level) == int(MemoryLevel.M3)
        and int(proposal.memory_type) == int(MemoryType.ROLE)
        and int(getattr(proposal, "support_delta", 0)) == len(role_parents)
        and abs(float(getattr(proposal, "explanatory_sum", 0.0)) - len(role_parents)) < 1e-9
    ):
        for target in role_parents[1:]:
            _submit_relation_v879(self, proposal, target, RelationType.EXPLAINS)

    world_parents = tuple(
        getattr(getattr(self, "world_model", None), "_v879_world_parents", {}).get(
            proposal.uid, ()
        )
    )
    if (
        world_parents
        and parent == world_parents[0]
        and int(proposal.level) == int(MemoryLevel.M5)
        and int(proposal.memory_type) == int(MemoryType.WORLD_MODEL)
    ):
        for target in world_parents[8:]:
            _submit_relation_v879(self, proposal, target, RelationType.EXPLAINS)

    superseded = tuple(
        getattr(getattr(self, "compression", None), "_v879_superseded", {}).get(
            proposal.uid, ()
        )
    )
    if superseded and parent.is_zero:
        for target in superseded[8:]:
            _submit_relation_v879(self, proposal, target, RelationType.SUPERSEDES)


def install_information_flow_integrity_v879() -> None:
    global _INSTALLED
    global _BASE_STRUCTURAL_TOKEN, _BASE_NORMALIZED_FAMILY_KEY
    global _BASE_PROMOTION_PROPOSE, _BASE_ROLE_PROPOSE, _BASE_WORLD_PROPOSE
    global _BASE_COMPRESSION_EVALUATE, _BASE_SUBMIT
    if _INSTALLED:
        return

    from v8 import formation_telemetry_v870 as telemetry
    from v8 import intelligence_loop_v087 as intelligence
    from v8 import normalized_memory_v086 as normalized
    from v8 import peers as peers_module
    from v8 import peers_v82
    from v8 import structural_events

    _BASE_STRUCTURAL_TOKEN = structural_events.StructuralFact.token.fget
    _BASE_NORMALIZED_FAMILY_KEY = structural_events.normalized_family_key
    structural_events.StructuralFact.token = property(_structured_fact_token_v879)
    structural_events.normalized_family_key = normalized_family_key_v879
    intelligence.normalized_family_key = normalized_family_key_v879
    telemetry.normalized_family_key = normalized_family_key_v879
    normalized.normalized_family_key = normalized_family_key_v879

    promotion_class = peers_v82.EvidenceGatedPromotionEngine
    _BASE_PROMOTION_PROPOSE = promotion_class.propose
    promotion_class.propose = _promotion_propose_v879

    role_class = peers_module.FunctionalRoleEstimator
    _BASE_ROLE_PROPOSE = role_class.propose_relational
    role_class.propose_relational = _role_propose_v879

    world_class = peers_module.WorldModelEstimator
    _BASE_WORLD_PROPOSE = world_class.propose
    world_class.propose = _world_propose_v879

    compression_class = peers_module.CompressionEstimator
    _BASE_COMPRESSION_EVALUATE = compression_class.evaluate
    compression_class.evaluate = _compression_evaluate_v879

    _BASE_SUBMIT = peers_v82.V82DevelopmentalPeerSupervisor._submit
    peers_v82.V82DevelopmentalPeerSupervisor._submit = _submit_v879

    _INSTALLED = True
