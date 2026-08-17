from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from struct import Struct

from v8 import arena as _arena
from v8 import model as _model


# v0.5.3 primary-valence semantics.  The environment supplies only a signed
# primitive valence; task-semantic interpretations remain learned.
_VALENCE_GAMMA = 0.97
_VALENCE_HORIZON = 256
_PREFERENCE_CREDIT_THRESHOLD = 0.50

_BASE_MEMORY_PROPOSAL = _model.MemoryProposal
_BASE_NODE_RECORD = _arena.NodeRecord
_BASE_NODE_STRUCT = _arena._NODE
_BASE_PROPOSAL_STRUCT = _model._PROPOSAL


@dataclass(frozen=True, slots=True)
class MemoryProposal(_BASE_MEMORY_PROPOSAL):
    primary_valence_sum: float = 0.0
    primary_valence_sq_sum: float = 0.0
    primary_valence_weight: float = 0.0
    positive_valence_count: float = 0.0
    negative_valence_count: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.primary_valence_weight < 0.0:
            raise ValueError("primary_valence_weight cannot be negative")
        if self.primary_valence_sq_sum < 0.0:
            raise ValueError("primary_valence_sq_sum cannot be negative")
        if self.positive_valence_count < 0.0 or self.negative_valence_count < 0.0:
            raise ValueError("primary valence counts cannot be negative")


@dataclass(frozen=True, slots=True)
class NodeRecord(_BASE_NODE_RECORD):
    primary_valence_sum: float = 0.0
    primary_valence_sq_sum: float = 0.0
    primary_valence_weight: float = 0.0
    positive_valence_count: float = 0.0
    negative_valence_count: float = 0.0

    @property
    def expected_primary_valence(self) -> float:
        if self.primary_valence_weight <= 0.0:
            return 0.0
        return self.primary_valence_sum / self.primary_valence_weight

    @property
    def primary_valence_variance(self) -> float:
        if self.primary_valence_weight <= 0.0:
            return 0.0
        mean = self.expected_primary_valence
        second = self.primary_valence_sq_sum / self.primary_valence_weight
        return max(0.0, second - mean * mean)

    @property
    def primary_valence_confidence(self) -> float:
        if self.primary_valence_weight <= 0.0:
            return 0.0
        return 1.0 - math.exp(-self.primary_valence_weight / 3.0)


@dataclass(frozen=True, slots=True)
class PrimaryValenceCredit:
    uid: _model.MemoryUid
    level: int
    memory_type: int
    key_parts: tuple[int, ...]
    fingerprint: int
    valence_sum: float
    valence_sq_sum: float
    weight: float
    positive_count: float
    negative_count: float


@dataclass(frozen=True, slots=True)
class PrimaryValencePreference:
    preferred: _model.MemoryUid
    other: _model.MemoryUid
    context_bucket: int
    strength: float


@dataclass(frozen=True, slots=True)
class PrimaryValenceLearningBatch:
    actor_id: int
    game_id: str
    strategy_stats: tuple[object, ...] = ()
    preference_probes: tuple[object, ...] = ()
    replanning_trials: tuple[object, ...] = ()
    replans: int = 0
    primary_valence_credits: tuple[PrimaryValenceCredit, ...] = ()
    primary_valence_preferences: tuple[PrimaryValencePreference, ...] = ()


# Five new doubles are appended before watermark/state fields.  This keeps
# canonical keys and UIDs unchanged while making valence a first-class statistic.
_NODE = Struct("<QQQBHBQQQQqdddddddddddddddQQBB")
_PROPOSAL = Struct("<QQQQQQBHBQQQQqdddddddddddddddQQHQbb")

_PENDING_NODE_VALENCE: dict[_model.MemoryUid, tuple[float, float, float, float, float]] = {}
_SCHEMA_INSTALLED = False
_RUNTIME_INSTALLED = False
_CAPTURE_ACTIVE = False
_TRAJECTORY: deque[dict[str, object]] = deque(maxlen=_VALENCE_HORIZON)
_RECENT_CONTEXTS: deque[int] = deque(maxlen=8)
_PENDING_CREDITS: dict[_model.MemoryUid, list[object]] = {}
_PENDING_VALENCE_PREFERENCES: list[PrimaryValencePreference] = []
_WINDOW_ACHIEVEMENT: dict[_model.MemoryUid, list[float]] = {}
_PRIMARY_VALENCE_BASE_ACTOR_WORKER = None


def _encode_proposal(proposal: MemoryProposal) -> bytes:
    relation_packet = _model._similarity_relation_packet(proposal)
    if relation_packet is not None:
        return relation_packet
    parts = tuple(_model.u64(value) for value in proposal.key_parts)
    padded = parts + (0,) * (4 - len(parts))
    return _PROPOSAL.pack(
        _model.u64(proposal.uid.hi), _model.u64(proposal.uid.lo),
        _model.u64(proposal.fingerprint), _model.u64(proposal.event_id.hi),
        _model.u64(proposal.event_id.lo), _model.u64(proposal.watermark),
        int(proposal.level), int(proposal.memory_type), len(parts), *padded,
        int(proposal.support_delta), float(proposal.significance_sum),
        float(proposal.prediction_error_sum), float(proposal.learning_value_sum),
        float(proposal.transfer_prior_sum), float(proposal.explanatory_sum),
        float(proposal.future_option_sum), float(proposal.score_weight),
        float(proposal.success_sum), float(proposal.cost_sum),
        float(proposal.attempt_weight), float(proposal.primary_valence_sum),
        float(proposal.primary_valence_sq_sum), float(proposal.primary_valence_weight),
        float(proposal.positive_valence_count), float(proposal.negative_valence_count),
        _model.u64(proposal.parent_uid.hi), _model.u64(proposal.parent_uid.lo),
        int(proposal.relation_type), _model.u64(proposal.source_game_hash),
        int(proposal.cognitive_state), int(proposal.validation_state),
    )


def _decode_proposal(payload: bytes) -> MemoryProposal:
    if len(payload) != _PROPOSAL.size:
        raise ValueError(f"invalid proposal packet size {len(payload)}")
    values = _PROPOSAL.unpack(payload)
    (uid_hi, uid_lo, fingerprint, event_hi, event_lo, watermark, level, memory_type,
     key_count, k0, k1, k2, k3, support_delta, significance, prediction_error,
     learning_value, transfer_prior, explanatory, future_option, score_weight,
     success_sum, cost_sum, attempt_weight, valence_sum, valence_sq_sum,
     valence_weight, positive_count, negative_count, parent_hi, parent_lo,
     relation_type, source_game_hash, cognitive_state, validation_state) = values
    keys = (k0, k1, k2, k3)[: int(key_count)]
    return MemoryProposal(
        uid=_model.MemoryUid(uid_hi, uid_lo), fingerprint=int(fingerprint),
        event_id=_model.EventId(event_hi, event_lo), watermark=int(watermark),
        level=_model.MemoryLevel(level), memory_type=_model.MemoryType(memory_type),
        key_parts=tuple(int(value) for value in keys), support_delta=int(support_delta),
        significance_sum=float(significance), prediction_error_sum=float(prediction_error),
        learning_value_sum=float(learning_value), transfer_prior_sum=float(transfer_prior),
        explanatory_sum=float(explanatory), future_option_sum=float(future_option),
        score_weight=float(score_weight), success_sum=float(success_sum),
        cost_sum=float(cost_sum), attempt_weight=float(attempt_weight),
        parent_uid=_model.MemoryUid(parent_hi, parent_lo),
        relation_type=_model.RelationType(relation_type), source_game_hash=int(source_game_hash),
        cognitive_state=int(cognitive_state), validation_state=int(validation_state),
        primary_valence_sum=float(valence_sum), primary_valence_sq_sum=float(valence_sq_sum),
        primary_valence_weight=float(valence_weight), positive_valence_count=float(positive_count),
        negative_valence_count=float(negative_count),
    )


def _node_write(self, row: int, value: NodeRecord) -> None:
    delta = _PENDING_NODE_VALENCE.pop(value.uid, None)
    if delta is not None:
        value = replace(
            value,
            primary_valence_sum=float(value.primary_valence_sum) + delta[0],
            primary_valence_sq_sum=float(value.primary_valence_sq_sum) + delta[1],
            primary_valence_weight=float(value.primary_valence_weight) + delta[2],
            positive_valence_count=float(value.positive_valence_count) + delta[3],
            negative_valence_count=float(value.negative_valence_count) + delta[4],
        )
    parts = tuple(_model.u64(v) for v in value.key_parts)
    if len(parts) > 4:
        raise ValueError("node key has more than four hot-path parts")
    padded = parts + (0,) * (4 - len(parts))
    _NODE.pack_into(
        self._shm.buf, self._offset(row), _model.u64(value.uid.hi),
        _model.u64(value.uid.lo), _model.u64(value.fingerprint), int(value.level),
        int(value.memory_type), len(parts), *padded, int(value.support_count),
        float(value.significance_sum), float(value.prediction_error_sum),
        float(value.learning_value_sum), float(value.transfer_prior_sum),
        float(value.explanatory_sum), float(value.future_option_sum),
        float(value.score_weight), float(value.success_sum), float(value.cost_sum),
        float(value.attempt_weight), float(value.primary_valence_sum),
        float(value.primary_valence_sq_sum), float(value.primary_valence_weight),
        float(value.positive_valence_count), float(value.negative_valence_count),
        _model.u64(value.updated_watermark), _model.u64(value.game_mask),
        int(value.cognitive_state) & 0xFF, int(value.validation_state) & 0xFF,
    )


def _node_read_from_buffer(self, buffer, row: int) -> NodeRecord:
    values = _NODE.unpack_from(buffer, self._offset(row))
    (hi, lo, fingerprint, level, memory_type, key_count, k0, k1, k2, k3, support,
     significance, prediction_error, learning_value, transfer_prior, explanatory,
     future_option, weight, success_sum, cost_sum, attempt_weight, valence_sum,
     valence_sq_sum, valence_weight, positive_count, negative_count, watermark,
     game_mask, cognitive_state, validation_state) = values
    return NodeRecord(
        uid=_model.MemoryUid(hi, lo), fingerprint=int(fingerprint), level=int(level),
        memory_type=int(memory_type), key_parts=tuple((k0, k1, k2, k3)[: int(key_count)]),
        support_count=int(support), significance_sum=float(significance),
        prediction_error_sum=float(prediction_error), learning_value_sum=float(learning_value),
        transfer_prior_sum=float(transfer_prior), explanatory_sum=float(explanatory),
        future_option_sum=float(future_option), score_weight=float(weight),
        updated_watermark=int(watermark), game_mask=int(game_mask),
        cognitive_state=int(cognitive_state), validation_state=int(validation_state),
        success_sum=float(success_sum), cost_sum=float(cost_sum),
        attempt_weight=float(attempt_weight), primary_valence_sum=float(valence_sum),
        primary_valence_sq_sum=float(valence_sq_sum), primary_valence_weight=float(valence_weight),
        positive_valence_count=float(positive_count), negative_valence_count=float(negative_count),
    )


def _node_read(self, row: int) -> NodeRecord:
    return _node_read_from_buffer(self, self._shm.buf, row)


def _install_snapshot_compatibility() -> None:
    from v8 import snapshot as snapshot_module
    old_current = _BASE_NODE_STRUCT
    original_old_migrate = snapshot_module._migrate_old_node

    def migrate_v3(arena, payload: bytes) -> bool:
        if len(payload) < snapshot_module._HEADER.size:
            return False
        count, _seq = snapshot_module._HEADER.unpack_from(payload, 0)
        if len(payload) != snapshot_module._HEADER.size + int(count) * old_current.size:
            return False
        arena.begin_write()
        try:
            for row in range(int(count)):
                values = old_current.unpack_from(payload, snapshot_module._HEADER.size + row * old_current.size)
                (hi, lo, fingerprint, level, memory_type, key_count, k0, k1, k2, k3,
                 support, significance, prediction_error, learning_value, transfer_prior,
                 explanatory, future_option, score_weight, success_sum, cost_sum,
                 attempt_weight, watermark, game_mask, cognitive_state, validation_state) = values
                arena.write(row, NodeRecord(
                    _model.MemoryUid(hi, lo), int(fingerprint), int(level), int(memory_type),
                    tuple((k0, k1, k2, k3)[: int(key_count)]), int(support), float(significance),
                    float(prediction_error), float(learning_value), float(transfer_prior),
                    float(explanatory), float(future_option), float(score_weight), int(watermark),
                    int(game_mask), int(cognitive_state), int(validation_state), float(success_sum),
                    float(cost_sum), float(attempt_weight),
                ))
        finally:
            arena.end_write(count=int(count))
        return True

    def load_nodes_compatible(arena, payload: bytes) -> None:
        if len(payload) < snapshot_module._HEADER.size:
            raise RuntimeError("invalid node snapshot")
        count, _seq = snapshot_module._HEADER.unpack_from(payload, 0)
        current_expected = snapshot_module._HEADER.size + int(count) * arena.record.size
        if len(payload) == current_expected:
            arena.load_snapshot(payload)
            return
        if migrate_v3(arena, payload):
            return
        if original_old_migrate(arena, payload, snapshot_module._OLD_NODE_V2, has_game_mask=True):
            return
        if original_old_migrate(arena, payload, snapshot_module._OLD_NODE_V1, has_game_mask=False):
            return
        raise RuntimeError("unsupported v8 node snapshot format")

    snapshot_module._load_nodes_compatible = load_nodes_compatible


def _install_shard_semantics() -> None:
    from v8 import shard as shard_module
    original_behavior_delta = shard_module._behavior_delta

    def shard_decode(payload: bytes):
        proposal = _decode_proposal(payload)
        if (abs(float(proposal.primary_valence_sum)) > 0.0
                or float(proposal.primary_valence_weight) > 0.0
                or float(proposal.positive_valence_count) > 0.0
                or float(proposal.negative_valence_count) > 0.0):
            _PENDING_NODE_VALENCE[proposal.uid] = (
                float(proposal.primary_valence_sum), float(proposal.primary_valence_sq_sum),
                float(proposal.primary_valence_weight), float(proposal.positive_valence_count),
                float(proposal.negative_valence_count),
            )
        return proposal

    def behavior_delta(proposal):
        value, weight = original_behavior_delta(proposal)
        valence_weight = max(0.0, float(getattr(proposal, "primary_valence_weight", 0.0)))
        if valence_weight > 0.0:
            value += 1.25 * float(proposal.primary_valence_sum)
            weight = max(float(weight), valence_weight)
        return float(value), float(weight)

    shard_module.decode_proposal = shard_decode
    shard_module._behavior_delta = behavior_delta


def _install_isf_semantics() -> None:
    from v8 import isf as isf_module
    base_score_type = isf_module.ISFScore

    @dataclass(frozen=True, slots=True)
    class PrimaryValenceISFScore(base_score_type):
        primary_valence_impact: float = 0.0

    stage_weights = {
        0: (0.60, 0.25, 0.00, 0.10, 0.00, 0.05),
        1: (0.35, 0.20, 0.20, 0.15, 0.03, 0.07),
        2: (0.25, 0.10, 0.10, 0.20, 0.15, 0.20),
        3: (0.20, 0.10, 0.10, 0.10, 0.20, 0.30),
        4: (0.25, 0.15, 0.05, 0.10, 0.15, 0.30),
        5: (0.30, 0.10, 0.05, 0.10, 0.10, 0.35),
        6: (0.35, 0.10, 0.05, 0.10, 0.10, 0.30),
    }

    def valence_impact(row: NodeRecord) -> float:
        if float(getattr(row, "primary_valence_weight", 0.0)) <= 0.0:
            return 0.0
        magnitude = min(1.0, abs(float(row.expected_primary_valence)))
        return max(0.0, min(1.0, magnitude * float(row.primary_valence_confidence)))

    def build_score(row, components, stage):
        option_impact, prediction, learning, transfer, explanatory, future = components
        primary = valence_impact(row)
        weighted = (primary, option_impact, prediction, learning, transfer, explanatory)
        total = sum(weight * component for weight, component in zip(stage_weights[stage], weighted, strict=True))
        return PrimaryValenceISFScore(float(option_impact), float(prediction), float(learning),
            float(transfer), float(explanatory), float(future), float(total), int(stage), float(primary))

    def score_memory(row, *, developmental_stage=None):
        stage = isf_module._fallback_stage(row) if developmental_stage is None else max(0, min(6, int(developmental_stage)))
        return build_score(row, isf_module.raw_components(row), stage)

    def score_memories(rows, *, developmental_stage):
        rows = tuple(rows)
        stage = max(0, min(6, int(developmental_stage)))
        grouped = defaultdict(list)
        for row in rows:
            grouped[(int(row.level), int(row.memory_type), stage)].append(row)
        result = {}
        for members in grouped.values():
            raw = {row.uid: isf_module.raw_components(row) for row in members}
            normalized = [isf_module._rank_normalize([(row.uid, raw[row.uid][i]) for row in members]) for i in range(5)]
            valence_norm = isf_module._rank_normalize([(row.uid, valence_impact(row)) for row in members])
            for row in members:
                option_impact, prediction, learning, transfer, explanatory = [normalized[i][row.uid] for i in range(5)]
                future = raw[row.uid][5]
                weighted = (valence_norm[row.uid], option_impact, prediction, learning, transfer, explanatory)
                total = sum(weight * component for weight, component in zip(stage_weights[stage], weighted, strict=True))
                result[row.uid] = PrimaryValenceISFScore(float(option_impact), float(prediction), float(learning),
                    float(transfer), float(explanatory), float(future), float(total), int(stage), float(valence_norm[row.uid]))
        return result

    isf_module.score_memory = score_memory
    isf_module.score_memories = score_memories
    isf_module.PrimaryValenceISFScore = PrimaryValenceISFScore
    isf_module.primary_valence_impact = valence_impact


def _install_observation_contract() -> None:
    from v8 import observation_contract as contract_module
    current = contract_module.ARC_GRID_CONTRACT
    forbidden = tuple(value for value in current.forbidden_semantic_fields if value not in {"reward", "win_value", "terminal_value"})
    contract_module.ARC_GRID_CONTRACT = replace(current, contract_id="arc-grid-v1-primary-valence",
        schema_version=max(2, int(current.schema_version) + 1), forbidden_semantic_fields=forbidden)


def install_primary_valence_schema() -> None:
    global _SCHEMA_INSTALLED
    if _SCHEMA_INSTALLED:
        return
    _model.MemoryProposal = MemoryProposal
    _model._PROPOSAL = _PROPOSAL
    _model.PROPOSAL_PACKET_SIZE = _PROPOSAL.size
    _model.encode_proposal = _encode_proposal
    _model.decode_proposal = _decode_proposal
    _arena.NodeRecord = NodeRecord
    _arena._NODE = _NODE
    _arena.SharedNodeArena.record = _NODE
    _arena.SharedNodeArena.write = _node_write
    _arena.SharedNodeArena._read_from_buffer = _node_read_from_buffer
    _arena.SharedNodeArena.read = _node_read
    if not hasattr(_model.ExperienceEvent, "primary_valence"):
        _model.ExperienceEvent.primary_valence = property(lambda self: int(self.terminal_polarity))
    _install_observation_contract()
    _install_snapshot_compatibility()
    _install_shard_semantics()
    _install_isf_semantics()
    _SCHEMA_INSTALLED = True


def _install_development_semantics() -> None:
    from v8 import development as development_module
    base_key_for = development_module._key_for

    def derive_proposal(level, event):
        definition = development_module.STAGES[int(level)]
        key = base_key_for(level, event)
        uid = _model.MemoryUid.from_key(level, definition.memory_type, key)
        e = event.experience
        multiplicity = max(1, int(event.multiplicity))
        structural_change = min(1.0, max(0, int(e.changed_cells)) / 32.0)
        option_magnitude = abs(math.tanh(float(e.future_option_delta)))
        primary_valence = int(e.terminal_polarity)
        significance = 0.55 * structural_change + 0.45 * option_magnitude
        learning_value = structural_change
        if int(level) <= int(_model.MemoryLevel.M1) and primary_valence != 0:
            significance = max(significance, 1.0)
        relation = _model.RelationType.LEADS_TO if level == _model.MemoryLevel.M7 else _model.RelationType.EXPLAINS
        valence_weight = float(multiplicity if primary_valence != 0 else 0)
        return MemoryProposal(
            uid=uid, fingerprint=_model.proposal_fingerprint(level, definition.memory_type, key),
            event_id=e.event_id, watermark=e.watermark, level=level, memory_type=definition.memory_type,
            key_parts=key, support_delta=multiplicity, significance_sum=significance * multiplicity,
            prediction_error_sum=float(e.prediction_error) * multiplicity if level == _model.MemoryLevel.M1 else 0.0,
            learning_value_sum=learning_value * multiplicity, future_option_sum=float(e.future_option_delta) * multiplicity,
            score_weight=float(multiplicity), parent_uid=event.parent_uid, relation_type=relation,
            source_game_hash=int(e.source_game_hash),
            cognitive_state=int(_model.CognitiveState.ACTIVE) if level <= _model.MemoryLevel.M1 else -1,
            primary_valence_sum=float(primary_valence) * valence_weight,
            primary_valence_sq_sum=float(primary_valence * primary_valence) * valence_weight,
            primary_valence_weight=valence_weight,
            positive_valence_count=valence_weight if primary_valence > 0 else 0.0,
            negative_valence_count=valence_weight if primary_valence < 0 else 0.0,
        )

    development_module.derive_proposal = derive_proposal


def _reset_actor_capture() -> None:
    _TRAJECTORY.clear(); _RECENT_CONTEXTS.clear(); _PENDING_CREDITS.clear()
    _PENDING_VALENCE_PREFERENCES.clear(); _WINDOW_ACHIEVEMENT.clear()


def _accumulate_credit(uid, *, level: int, memory_type: int, key_parts: tuple[int, ...], fingerprint: int, value: float) -> None:
    if uid.is_zero or abs(float(value)) <= 1e-12:
        return
    bucket = _PENDING_CREDITS.get(uid)
    if bucket is None:
        bucket = [int(level), int(memory_type), tuple(int(v) for v in key_parts), int(fingerprint), 0.0, 0.0, 0.0, 0.0, 0.0]
        _PENDING_CREDITS[uid] = bucket
    bucket[4] += float(value); bucket[5] += float(value) * float(value); bucket[6] += 1.0
    if value > 0.0: bucket[7] += 1.0
    else: bucket[8] += 1.0


def _credit_tuple() -> tuple[PrimaryValenceCredit, ...]:
    return tuple(PrimaryValenceCredit(uid, int(v[0]), int(v[1]), tuple(v[2]), int(v[3]), float(v[4]),
        float(v[5]), float(v[6]), float(v[7]), float(v[8])) for uid, v in sorted(_PENDING_CREDITS.items()))


def _capture_experience_factory(base_experience, behavior_module):
    def capture_experience(*args, **kwargs):
        event = base_experience(*args, **kwargs)
        if not _CAPTURE_ACTIVE:
            return event
        m0_key = (int(event.event_id.hi), int(event.event_id.lo))
        m1_key = (int(event.context_signature), int(event.action_id), int(event.outcome_signature), int(event.next_context_signature))
        view = behavior_module._CURRENT_ACTOR_VIEW
        plans = tuple(getattr(view, "_behavior_last_plans", ())) if view is not None else ()
        selected = next((plan for plan in plans if int(plan.action_id) == int(event.action_id)), None)
        alternative = next((plan for plan in plans if selected is not None and plan.outcome_uid != selected.outcome_uid), None)
        _TRAJECTORY.append({
            "event": event,
            "m0_uid": _model.MemoryUid.from_key(_model.MemoryLevel.M0, _model.MemoryType.EPISODE, m0_key),
            "m0_key": m0_key,
            "m1_uid": _model.MemoryUid.from_key(_model.MemoryLevel.M1, _model.MemoryType.CONTINGENCY, m1_key),
            "m1_key": m1_key,
            "plan": selected,
            "alternative": alternative,
            "observed": (),
        })
        return event
    return capture_experience


def _structural_cost(entry: dict[str, object]) -> float:
    event = entry["event"]
    cost = 1.0
    if int(event.changed_cells) <= 0: cost += 1.0
    if int(event.next_context_signature) == int(event.context_signature): cost += 0.5
    if int(event.next_context_signature) in set(int(v) for v in _RECENT_CONTEXTS): cost += 1.0
    return float(cost)


def _emit_terminal_credits(primary_valence: int, behavior_module) -> None:
    if primary_valence == 0:
        return
    for distance, entry in enumerate(reversed(tuple(_TRAJECTORY))):
        value = float(primary_valence) * (_VALENCE_GAMMA ** distance)
        event = entry["event"]
        if distance > 0:
            _accumulate_credit(entry["m0_uid"], level=int(_model.MemoryLevel.M0), memory_type=int(_model.MemoryType.EPISODE),
                key_parts=entry["m0_key"], fingerprint=_model.proposal_fingerprint(_model.MemoryLevel.M0, _model.MemoryType.EPISODE, entry["m0_key"]), value=value)
            _accumulate_credit(entry["m1_uid"], level=int(_model.MemoryLevel.M1), memory_type=int(_model.MemoryType.CONTINGENCY),
                key_parts=entry["m1_key"], fingerprint=_model.proposal_fingerprint(_model.MemoryLevel.M1, _model.MemoryType.CONTINGENCY, entry["m1_key"]), value=value)
        observed = tuple(uid for uid in entry.get("observed", ()) if not uid.is_zero)
        view = behavior_module._CURRENT_ACTOR_VIEW
        if observed and view is not None:
            outcome_uid = observed[0]
            outcome_row = getattr(view, "_node_by_uid", {}).get(outcome_uid)
            if outcome_row is not None:
                _accumulate_credit(outcome_uid, level=int(outcome_row.level), memory_type=int(outcome_row.memory_type),
                    key_parts=tuple(int(v) for v in outcome_row.key_parts), fingerprint=int(outcome_row.fingerprint), value=value)
        plan = entry.get("plan")
        if plan is not None and plan.outcome_uid in observed and view is not None:
            strategy_row = getattr(view, "_node_by_uid", {}).get(plan.strategy_uid)
            if strategy_row is not None:
                _accumulate_credit(plan.strategy_uid, level=int(strategy_row.level), memory_type=int(strategy_row.memory_type),
                    key_parts=tuple(int(v) for v in strategy_row.key_parts), fingerprint=int(strategy_row.fingerprint), value=value)
            alternative = entry.get("alternative")
            if value >= _PREFERENCE_CREDIT_THRESHOLD and alternative is not None and not bool(plan.preference_influenced):
                _PENDING_VALENCE_PREFERENCES.append(PrimaryValencePreference(plan.outcome_uid, alternative.outcome_uid,
                    _model.stable_u64(int(event.context_signature), person=b"v8-context"), float(value)))


def _actor_worker_with_primary_valence(*args, **kwargs):
    """Picklable process entry point for actor-local primary-valence capture."""
    global _CAPTURE_ACTIVE
    if _PRIMARY_VALENCE_BASE_ACTOR_WORKER is None:
        raise RuntimeError("primary-valence actor semantics are not installed")
    _reset_actor_capture(); _CAPTURE_ACTIVE = True
    try:
        return _PRIMARY_VALENCE_BASE_ACTOR_WORKER(*args, **kwargs)
    finally:
        _CAPTURE_ACTIVE = False; _reset_actor_capture()


def _install_actor_semantics() -> None:
    global _PRIMARY_VALENCE_BASE_ACTOR_WORKER
    from v8 import actor as actor_module
    from v8 import behavior_recovery as behavior_module
    base_experience = _model.ExperienceEvent
    behavior_observed = actor_module._observed_outcome_uids
    behavior_worker = actor_module.actor_worker
    _PRIMARY_VALENCE_BASE_ACTOR_WORKER = behavior_worker
    actor_module.ExperienceEvent = _capture_experience_factory(base_experience, behavior_module)

    def observed_with_valence(**kwargs):
        result = behavior_observed(**kwargs)
        if not _CAPTURE_ACTIVE or not _TRAJECTORY:
            return result
        entry = _TRAJECTORY[-1]
        event = entry["event"]
        if int(event.outcome_signature) != int(kwargs.get("outcome_signature", -1)):
            return result
        observed = tuple(uid for uid in result if not uid.is_zero)
        entry["observed"] = observed
        plan = entry.get("plan")
        if plan is not None:
            stat = _WINDOW_ACHIEVEMENT.setdefault(plan.strategy_uid, [0.0, 0.0, 0.0])
            stat[0] += 1.0; stat[1] += float(plan.outcome_uid in observed); stat[2] += _structural_cost(entry)
        _RECENT_CONTEXTS.append(int(event.context_signature))
        primary_valence = int(kwargs.get("terminal_polarity", 0))
        if primary_valence != 0:
            _emit_terminal_credits(primary_valence, behavior_module)
            _TRAJECTORY.clear(); _RECENT_CONTEXTS.clear()
        return result

    actor_module._observed_outcome_uids = observed_with_valence

    def learning_batch(*, job, strategy_stats, preference_probes, replanning_trials):
        del preference_probes
        stats = []
        for uid in sorted(set(strategy_stats) | set(_WINDOW_ACHIEVEMENT)):
            values = _WINDOW_ACHIEVEMENT.get(uid, strategy_stats.get(uid))
            if values is not None and values[0] > 0:
                stats.append(actor_module.StrategyRunStat(uid, int(values[0]), int(values[1]), float(values[2])))
        credits = _credit_tuple()
        preferences = tuple(_PENDING_VALENCE_PREFERENCES)
        if not stats and not replanning_trials and not credits and not preferences:
            return None
        return PrimaryValenceLearningBatch(int(job.actor_id), str(job.game_id), tuple(stats), (), tuple(replanning_trials),
            len(replanning_trials), credits, preferences)

    actor_module._learning_batch = learning_batch
    actor_module.ActorLearningBatch = PrimaryValenceLearningBatch
    original_publish = actor_module._publish_learning

    def publish_learning(*args, **kwargs):
        published = original_publish(*args, **kwargs)
        if published:
            _PENDING_CREDITS.clear(); _PENDING_VALENCE_PREFERENCES.clear(); _WINDOW_ACHIEVEMENT.clear()
        return published

    actor_module._publish_learning = publish_learning

    def merge_learning_batches(rows):
        grouped = {}
        for row in rows:
            key = (int(row.actor_id), str(row.game_id))
            bucket = grouped.setdefault(key, {"stats": {}, "trials": [], "replans": 0, "credits": {}, "preferences": []})
            for stat in row.strategy_stats:
                values = bucket["stats"].setdefault(stat.strategy_uid, [0.0, 0.0, 0.0])
                values[0] += float(stat.attempts); values[1] += float(stat.successes); values[2] += float(stat.cost)
            bucket["trials"].extend(row.replanning_trials); bucket["replans"] += int(row.replans)
            bucket["preferences"].extend(getattr(row, "primary_valence_preferences", ()))
            for credit in getattr(row, "primary_valence_credits", ()):
                target = bucket["credits"].get(credit.uid)
                if target is None:
                    bucket["credits"][credit.uid] = [credit.level, credit.memory_type, credit.key_parts, credit.fingerprint,
                        credit.valence_sum, credit.valence_sq_sum, credit.weight, credit.positive_count, credit.negative_count]
                else:
                    for i, value in enumerate((credit.valence_sum, credit.valence_sq_sum, credit.weight, credit.positive_count, credit.negative_count), start=4):
                        target[i] += value
        result = []
        for (actor_id, game_id), bucket in sorted(grouped.items()):
            stats = tuple(actor_module.StrategyRunStat(uid, int(v[0]), int(v[1]), float(v[2]))
                for uid, v in sorted(bucket["stats"].items()) if v[0] > 0)
            credits = tuple(PrimaryValenceCredit(uid, int(v[0]), int(v[1]), tuple(v[2]), int(v[3]), float(v[4]), float(v[5]),
                float(v[6]), float(v[7]), float(v[8])) for uid, v in sorted(bucket["credits"].items()))
            result.append(PrimaryValenceLearningBatch(actor_id, game_id, stats, (), tuple(bucket["trials"]),
                int(bucket["replans"]), credits, tuple(bucket["preferences"])))
        return tuple(result)

    actor_module._merge_learning_batches = merge_learning_batches

    actor_module.actor_worker = _actor_worker_with_primary_valence


def _parent_valence_stats(candidate, by_uid):
    parents = [by_uid[uid] for uid in candidate.parents if uid in by_uid]
    if not parents:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    highest = max(int(row.level) for row in parents)
    sources = [row for row in parents if int(row.level) == highest]
    return (sum(float(getattr(row, "primary_valence_sum", 0.0)) for row in sources),
        sum(float(getattr(row, "primary_valence_sq_sum", 0.0)) for row in sources),
        sum(float(getattr(row, "primary_valence_weight", 0.0)) for row in sources),
        sum(float(getattr(row, "positive_valence_count", 0.0)) for row in sources),
        sum(float(getattr(row, "negative_valence_count", 0.0)) for row in sources))


def _install_peer_formation_semantics() -> None:
    from v8 import peers_v82 as peers_v82_module

    def process_formation(self, cut, frozen):
        by_uid = {row.uid: row for row in cut.nodes}
        state = getattr(self, "_primary_valence_formation_state", None)
        if state is None:
            state = {}; self._primary_valence_formation_state = state
        for candidate in self.promotion.propose(cut.nodes, cut.edges, budget=self.candidate_budget):
            parent_watermark = max((by_uid[uid].updated_watermark for uid in candidate.parents if uid in by_uid), default=cut.watermark)
            freshness = f"v82-formation:{int(candidate.level)}:{int(candidate.memory_type)}"
            if not self._fresh(freshness, candidate.uid, parent_watermark):
                continue
            identity = self._formation_identity(candidate)
            weight = max(1.0, float(candidate.support))
            first_parent = candidate.parents[0] if candidate.parents else _model.MemoryUid.zero()
            future_option_delta = self._formation_future_option(candidate, by_uid)
            current_valence = _parent_valence_stats(candidate, by_uid)
            prior_valence = state.get(candidate.uid, (0.0, 0.0, 0.0, 0.0, 0.0))
            delta_valence = (current_valence[0] - prior_valence[0], max(0.0, current_valence[1] - prior_valence[1]),
                max(0.0, current_valence[2] - prior_valence[2]), max(0.0, current_valence[3] - prior_valence[3]),
                max(0.0, current_valence[4] - prior_valence[4]))
            state[candidate.uid] = current_valence
            proposal = MemoryProposal(uid=candidate.uid, fingerprint=identity.fingerprint, event_id=self._event_id(),
                watermark=int(cut.watermark), level=candidate.level, memory_type=candidate.memory_type,
                key_parts=candidate.key_parts, support_delta=max(1, int(candidate.support)),
                significance_sum=float(candidate.significance) * weight, learning_value_sum=float(candidate.learning_value) * weight,
                transfer_prior_sum=float(candidate.transfer_prior) * weight, explanatory_sum=float(candidate.explanatory_reach) * weight,
                future_option_sum=future_option_delta * weight, score_weight=weight, parent_uid=first_parent,
                relation_type=self._relation_for(candidate), cognitive_state=int(candidate.cognitive_state),
                validation_state=int(candidate.validation_state), primary_valence_sum=float(delta_valence[0]),
                primary_valence_sq_sum=float(delta_valence[1]), primary_valence_weight=float(delta_valence[2]),
                positive_valence_count=float(delta_valence[3]), negative_valence_count=float(delta_valence[4]))
            self._submit(proposal)
            for parent in candidate.parents[1:8]:
                self._submit(self._existing_proposal(identity, parent_uid=parent,
                    relation_type=self._relation_for(candidate, extra_parent=True)))
            provenance_games = set()
            for parent in candidate.parents: provenance_games.update(frozen.source_games(parent))
            self._append_evidence(candidate.evidence_kind, candidate, candidate.evidence_value,
                validation_state=int(candidate.validation_state), provenance_games=tuple(sorted(provenance_games)))

    peers_v82_module.V82DevelopmentalPeerSupervisor._process_formation = process_formation


def _install_peer_submit_semantics() -> None:
    from v8 import peers as peers_module
    original_submit = peers_module.DevelopmentalPeerSupervisor._submit

    def submit(self, proposal):
        if (int(proposal.level) == int(_model.MemoryLevel.M6)
                and int(proposal.memory_type) == int(_model.MemoryType.OUTCOME)
                and proposal.relation_type == _model.RelationType.SUPERSEDES
                and not proposal.parent_uid.is_zero
                and float(getattr(proposal, "primary_valence_weight", 0.0)) <= 0.0):
            parent = next((row for row in self.read_view.node_records(level=_model.MemoryLevel.M6)
                if row.uid == proposal.parent_uid), None)
            if parent is not None and float(parent.primary_valence_weight) > 0.0:
                proposal = replace(proposal, primary_valence_sum=float(parent.primary_valence_sum),
                    primary_valence_sq_sum=float(parent.primary_valence_sq_sum), primary_valence_weight=float(parent.primary_valence_weight),
                    positive_valence_count=float(parent.positive_valence_count), negative_valence_count=float(parent.negative_valence_count))
        return original_submit(self, proposal)

    peers_module.DevelopmentalPeerSupervisor._submit = submit
    from v8 import peers_v82 as peers_v82_module
    peers_v82_module.V82DevelopmentalPeerSupervisor._submit = submit


def _install_planner_semantics() -> None:
    from v8 import behavior_recovery as behavior_module
    original_score_rows = behavior_module._score_strategy_rows

    def score_rows(view, rows, **kwargs):
        plans = list(original_score_rows(view, rows, **kwargs)); adjusted = []
        by_uid = getattr(view, "_node_by_uid", {})
        for plan in plans:
            strategy = by_uid.get(plan.strategy_uid); outcome = by_uid.get(plan.outcome_uid)
            strategy_value = 0.0 if strategy is None else float(strategy.expected_primary_valence) * float(strategy.primary_valence_confidence)
            outcome_value = 0.0 if outcome is None else float(outcome.expected_primary_valence) * float(outcome.primary_valence_confidence)
            adjusted.append(replace(plan, score=float(plan.score) + 1.50 * strategy_value + 1.00 * outcome_value))
        adjusted.sort(key=lambda item: (-item.score, item.action_id, item.strategy_uid))
        return tuple(adjusted)

    behavior_module._score_strategy_rows = score_rows


def _install_runtime_learning_semantics() -> None:
    from v8 import runtime as runtime_module
    from v8.evidence import EvidenceRecord
    original_record = runtime_module.ContinuousMemoryRuntime.record_actor_results

    def record_actor_results(self, results):
        results = tuple(results); original_record(self, results)
        if not hasattr(self, "_primary_valence_sequence"): self._primary_valence_sequence = 0
        for result in results:
            game_hash = _model.stable_u64(result.game_id, person=b"v8-game")
            for credit in getattr(result, "primary_valence_credits", ()):
                self._primary_valence_sequence += 1
                proposal = MemoryProposal(uid=credit.uid, fingerprint=int(credit.fingerprint),
                    event_id=_model.EventId.from_producer(0x7FFFFFFD, self._primary_valence_sequence), watermark=int(self.watermark),
                    level=_model.MemoryLevel(int(credit.level)), memory_type=_model.MemoryType(int(credit.memory_type)),
                    key_parts=tuple(int(v) for v in credit.key_parts), support_delta=0, score_weight=0.0,
                    source_game_hash=int(game_hash), cognitive_state=-1, validation_state=-1,
                    primary_valence_sum=float(credit.valence_sum), primary_valence_sq_sum=float(credit.valence_sq_sum),
                    primary_valence_weight=float(credit.weight), positive_valence_count=float(credit.positive_count),
                    negative_valence_count=float(credit.negative_count))
                self.submit_proposal(proposal)
                if self.peers is not None:
                    self.peers.ledger.append(EvidenceRecord.for_uid(
                        f"primary-valence:{result.actor_id}:{self._primary_valence_sequence}", credit.uid,
                        evidence_kind="primary_valence_credit", watermark=self.watermark, raw_value=float(credit.valence_sum),
                        normalized_value=min(1.0, abs(float(credit.valence_sum))), developmental_stage=int(credit.level),
                        validation_state=3, source_game_hash=int(game_hash),
                        effect_direction=1 if credit.valence_sum > 0 else -1 if credit.valence_sum < 0 else 0,
                        graph_generation=self.generation))
            if self.peers is not None:
                for preference in getattr(result, "primary_valence_preferences", ()):
                    self.peers.record_preference_probe(outcome_a=preference.preferred, outcome_b=preference.other,
                        context_bucket=preference.context_bucket, chosen_outcome=preference.preferred,
                        both_reachable=True, preference_influenced=False)
                    preferred = next((row for row in self.read_view.node_records(level=_model.MemoryLevel.M6)
                        if row.uid == preference.preferred), None)
                    if preferred is not None:
                        self.peers._append_evidence("primary_valence_preference", preferred,
                            min(1.0, abs(float(preference.strength))), unique=True,
                            causal_intervention="realized_primary_valence", effect_direction=1)

    runtime_module.ContinuousMemoryRuntime.record_actor_results = record_actor_results


def _install_runtime_metadata() -> None:
    from v8 import runtime_v82 as runtime_v82_module
    runtime_v82_module.V82ContinuousMemoryRuntime.scientific_semantics_version = "v8.3-primary-valence"
    runtime_v82_module.V82ContinuousMemoryRuntime.research_paper_version = "0.5.3"


def _install_traceability() -> None:
    from v8 import scientific_traceability as trace_module
    revised = []
    for record in trace_module.TRACEABILITY:
        if record.hypothesis_id == "H15":
            revised.append(replace(record,
                paper_claim="Target-like preference is learned from realized primary-valence evidence separately from outcome equivalence.",
                candidate_evidence=("primary_valence_preference", "preference_probe"),
                required_evidence=("stable_preference_probe",)))
        else: revised.append(record)
    trace_module.TRACEABILITY = tuple(revised)


def install_primary_valence_runtime() -> None:
    global _RUNTIME_INSTALLED
    if _RUNTIME_INSTALLED:
        return
    _install_development_semantics()
    _install_actor_semantics()
    _install_peer_formation_semantics()
    _install_peer_submit_semantics()
    _install_planner_semantics()
    _install_runtime_learning_semantics()
    _install_runtime_metadata()
    _install_traceability()
    _RUNTIME_INSTALLED = True
