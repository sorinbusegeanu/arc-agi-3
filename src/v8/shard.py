from __future__ import annotations

import multiprocessing as mp
import queue
from collections import deque
from dataclasses import dataclass, replace

from v8.arena import (
    ActionRecord,
    ArenaDescriptor,
    EdgeRecord,
    NodeRecord,
    SharedActionArena,
    SharedEdgeArena,
    SharedNodeArena,
)
from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryUid,
    RELATION_PROPOSAL_PACKET_SIZE,
    RelationType,
    ValidationState,
    decode_proposal,
    decode_relation_proposal,
    signed_u64,
    u64,
)
from v8.ring import SharedRingBuffer


@dataclass(frozen=True, slots=True)
class ShardConfig:
    shard_id: int
    node_arena: ArenaDescriptor
    edge_arena: ArenaDescriptor
    action_arena: ArenaDescriptor


def _action_slot(
    arena: SharedActionArena,
    index: dict[tuple[int, int], int],
    context: int,
    action: int,
) -> tuple[int, ActionRecord | None]:
    key = (u64(context), int(action))
    existing = index.get(key)
    if existing is not None:
        occupied, value = arena.read_slot(existing)
        if not occupied:
            raise RuntimeError("action index map points at empty slot")
        return existing, value
    start = (key[0] ^ (key[1] * 0x9E3779B185EBCA87)) % arena.capacity
    for offset in range(arena.capacity):
        row = int((start + offset) % arena.capacity)
        occupied, value = arena.read_slot(row)
        if not occupied:
            index[key] = row
            return row, None
        if value.context_signature == key[0] and value.action_id == key[1]:
            index[key] = row
            return row, value
    raise MemoryError("action arena is full")


def _game_bit(source_game_hash: int) -> int:
    value = int(source_game_hash)
    return 0 if value == 0 else 1 << (value & 63)


def _write_edge(
    *,
    edges: SharedEdgeArena,
    edge_index: dict[tuple[MemoryUid, int, MemoryUid], int],
    edge_count: int,
    source_uid: MemoryUid,
    relation_type: int,
    target_uid: MemoryUid,
    support_delta: int,
    watermark: int,
    shard_id: int,
    score_sum: float = 0.0,
    score_weight: float = 0.0,
    source_version: int = 0,
    target_version: int = 0,
) -> int:
    key = (source_uid, int(relation_type), target_uid)
    row = edge_index.get(key)
    increment = max(1, int(support_delta))
    if row is None:
        if edge_count >= edges.capacity:
            raise MemoryError(f"edge arena full in shard {shard_id}")
        row = edge_count
        edge_count += 1
        edge_index[key] = row
        record = EdgeRecord(
            source_uid,
            int(relation_type),
            target_uid,
            increment,
            int(watermark),
            float(score_sum),
            max(0.0, float(score_weight)),
            int(source_version),
            int(target_version),
        )
    else:
        prior = edges.read(row)
        record = replace(
            prior,
            support_count=prior.support_count + increment,
            updated_watermark=max(prior.updated_watermark, int(watermark)),
            score_sum=prior.score_sum + float(score_sum),
            score_weight=prior.score_weight + max(0.0, float(score_weight)),
            source_version=max(prior.source_version, int(source_version)),
            target_version=max(prior.target_version, int(target_version)),
        )
    edges.write(row, record)
    return edge_count


def shard_worker(
    config: ShardConfig,
    ring_args: dict[str, object],
    stop_event: mp.synchronize.Event,
    inflight: mp.sharedctypes.Synchronized,
    watermark: mp.sharedctypes.Synchronized,
    error_queue: mp.Queue,
    batch_size: int = 256,
    global_generation: mp.sharedctypes.Synchronized | None = None,
) -> None:
    ring = SharedRingBuffer(**ring_args)
    nodes = SharedNodeArena.attach(config.node_arena)
    edges = SharedEdgeArena.attach(config.edge_arena)
    actions = SharedActionArena.attach(config.action_arena)
    try:
        node_index: dict[MemoryUid, int] = {
            record.uid: row for row, record in enumerate(nodes.records())
        }
        edge_index: dict[tuple[MemoryUid, int, MemoryUid], int] = {
            (record.source_uid, int(record.relation_type), record.target_uid): row
            for row, record in enumerate(edges.records())
        }
        action_index: dict[tuple[int, int], int] = {}
        for row in range(actions.capacity):
            occupied, value = actions.read_slot(row)
            if occupied:
                action_index[(value.context_signature, value.action_id)] = row

        dedupe_capacity = max(65_536, int(batch_size) * 1024)
        seen: set[tuple[int, ...]] = set()
        seen_order: deque[tuple[int, ...]] = deque()

        while not stop_event.is_set() or not ring.empty:
            first = ring.get(timeout=0.05)
            if first is None:
                continue
            packets = [first]
            for _ in range(max(0, int(batch_size) - 1)):
                payload = ring.get(timeout=0.0)
                if payload is None:
                    break
                packets.append(payload)

            with inflight.get_lock():
                inflight.value += len(packets)
            mutated = False
            try:
                nodes.begin_write()
                edges.begin_write()
                actions.begin_write()
                node_count = nodes.count
                edge_count = edges.count
                max_watermark = int(watermark.value)
                try:
                    for payload in packets:
                        if len(payload) == RELATION_PROPOSAL_PACKET_SIZE:
                            relation = decode_relation_proposal(payload)
                            dedupe = (
                                1,
                                int(relation.event_id.hi),
                                int(relation.event_id.lo),
                                int(relation.source_uid.hi),
                                int(relation.source_uid.lo),
                                int(relation.target_uid.hi),
                                int(relation.target_uid.lo),
                                int(relation.relation_type),
                            )
                            if dedupe in seen:
                                continue
                            seen.add(dedupe)
                            seen_order.append(dedupe)
                            if len(seen_order) > dedupe_capacity:
                                seen.discard(seen_order.popleft())

                            if node_index.get(relation.source_uid) is None:
                                continue
                            mutated = True
                            max_watermark = max(max_watermark, int(relation.watermark))
                            edge_count = _write_edge(
                                edges=edges,
                                edge_index=edge_index,
                                edge_count=edge_count,
                                source_uid=relation.source_uid,
                                relation_type=int(relation.relation_type),
                                target_uid=relation.target_uid,
                                support_delta=relation.support_delta,
                                watermark=relation.watermark,
                                shard_id=config.shard_id,
                                score_sum=relation.score_sum,
                                score_weight=relation.score_weight,
                                source_version=relation.source_version,
                                target_version=relation.target_version,
                            )
                            continue

                        proposal = decode_proposal(payload)
                        dedupe = (
                            0,
                            int(proposal.event_id.hi),
                            int(proposal.event_id.lo),
                            int(proposal.uid.hi),
                            int(proposal.uid.lo),
                        )
                        if dedupe in seen:
                            continue
                        seen.add(dedupe)
                        seen_order.append(dedupe)
                        if len(seen_order) > dedupe_capacity:
                            seen.discard(seen_order.popleft())

                        mutated = True
                        max_watermark = max(max_watermark, int(proposal.watermark))
                        game_bit = _game_bit(proposal.source_game_hash)

                        row = node_index.get(proposal.uid)
                        if row is None:
                            if node_count >= nodes.capacity:
                                raise MemoryError(f"node arena full in shard {config.shard_id}")
                            row = node_count
                            node_count += 1
                            node_index[proposal.uid] = row
                            active = (
                                int(proposal.cognitive_state)
                                if int(proposal.cognitive_state) >= 0
                                else int(CognitiveState.ACTIVE if int(proposal.level) <= int(MemoryLevel.M1) else CognitiveState.CANDIDATE)
                            )
                            validated = (
                                int(proposal.validation_state)
                                if int(proposal.validation_state) >= 0
                                else int(ValidationState.VALIDATED if int(proposal.level) <= int(MemoryLevel.M1) else ValidationState.UNTESTED)
                            )
                            record = NodeRecord(
                                uid=proposal.uid,
                                fingerprint=proposal.fingerprint,
                                level=int(proposal.level),
                                memory_type=int(proposal.memory_type),
                                key_parts=proposal.key_parts,
                                support_count=proposal.support_delta,
                                significance_sum=proposal.significance_sum,
                                prediction_error_sum=proposal.prediction_error_sum,
                                learning_value_sum=proposal.learning_value_sum,
                                transfer_prior_sum=proposal.transfer_prior_sum,
                                explanatory_sum=proposal.explanatory_sum,
                                future_option_sum=proposal.future_option_sum,
                                score_weight=proposal.score_weight,
                                updated_watermark=proposal.watermark,
                                game_mask=game_bit,
                                cognitive_state=active,
                                validation_state=validated,
                                success_sum=proposal.success_sum,
                                cost_sum=proposal.cost_sum,
                                attempt_weight=proposal.attempt_weight,
                            )
                        else:
                            current = nodes.read(row)
                            if int(current.fingerprint) != int(proposal.fingerprint):
                                raise RuntimeError(
                                    f"canonical UID collision {proposal.uid.hex()} in shard {config.shard_id}"
                                )
                            if (
                                int(current.level) != int(proposal.level)
                                or int(current.memory_type) != int(proposal.memory_type)
                                or tuple(current.key_parts) != tuple(u64(v) for v in proposal.key_parts)
                            ):
                                raise RuntimeError(f"immutable memory identity mismatch {proposal.uid.hex()}")
                            cognitive_state = (
                                int(proposal.cognitive_state)
                                if int(proposal.cognitive_state) >= 0
                                else int(current.cognitive_state)
                            )
                            validation_state = (
                                int(proposal.validation_state)
                                if int(proposal.validation_state) >= 0
                                else int(current.validation_state)
                            )
                            record = replace(
                                current,
                                support_count=current.support_count + proposal.support_delta,
                                significance_sum=current.significance_sum + proposal.significance_sum,
                                prediction_error_sum=current.prediction_error_sum + proposal.prediction_error_sum,
                                learning_value_sum=current.learning_value_sum + proposal.learning_value_sum,
                                transfer_prior_sum=current.transfer_prior_sum + proposal.transfer_prior_sum,
                                explanatory_sum=current.explanatory_sum + proposal.explanatory_sum,
                                future_option_sum=current.future_option_sum + proposal.future_option_sum,
                                score_weight=current.score_weight + proposal.score_weight,
                                success_sum=current.success_sum + proposal.success_sum,
                                cost_sum=current.cost_sum + proposal.cost_sum,
                                attempt_weight=current.attempt_weight + proposal.attempt_weight,
                                updated_watermark=max(current.updated_watermark, proposal.watermark),
                                game_mask=int(current.game_mask) | game_bit,
                                cognitive_state=cognitive_state,
                                validation_state=validation_state,
                            )
                        nodes.write(row, record)

                        if not proposal.parent_uid.is_zero:
                            edge_count = _write_edge(
                                edges=edges,
                                edge_index=edge_index,
                                edge_count=edge_count,
                                source_uid=proposal.uid,
                                relation_type=int(proposal.relation_type),
                                target_uid=proposal.parent_uid,
                                support_delta=proposal.support_delta,
                                watermark=proposal.watermark,
                                shard_id=config.shard_id,
                            )

                        if int(proposal.source_game_hash) != 0:
                            edge_count = _write_edge(
                                edges=edges,
                                edge_index=edge_index,
                                edge_count=edge_count,
                                source_uid=proposal.uid,
                                relation_type=int(RelationType.GAME_PROVENANCE),
                                target_uid=MemoryUid(0, u64(proposal.source_game_hash)),
                                support_delta=proposal.support_delta,
                                watermark=proposal.watermark,
                                shard_id=config.shard_id,
                            )

                        if proposal.level == MemoryLevel.M1 and len(proposal.key_parts) >= 2 and (
                            proposal.support_delta > 0 or abs(float(proposal.significance_sum)) > 0.0
                        ):
                            context = int(proposal.key_parts[0])
                            action = signed_u64(int(proposal.key_parts[1]))
                            action_row, prior_action = _action_slot(actions, action_index, context, action)
                            significance = float(proposal.significance_sum)
                            if prior_action is None:
                                action_record = ActionRecord(
                                    u64(context),
                                    action,
                                    proposal.support_delta,
                                    significance,
                                    max(0.0, proposal.score_weight),
                                    proposal.watermark,
                                )
                            else:
                                action_record = replace(
                                    prior_action,
                                    support_count=prior_action.support_count + proposal.support_delta,
                                    score_sum=prior_action.score_sum + significance,
                                    score_weight=prior_action.score_weight + max(0.0, proposal.score_weight),
                                    updated_watermark=max(prior_action.updated_watermark, proposal.watermark),
                                )
                            actions.write(action_row, action_record)
                    with watermark.get_lock():
                        watermark.value = max(int(watermark.value), max_watermark)
                finally:
                    nodes.end_write(count=node_count)
                    edges.end_write(count=edge_count)
                    actions.end_write(count=actions.count)
            finally:
                if mutated and global_generation is not None:
                    with global_generation.get_lock():
                        global_generation.value += 1
                with inflight.get_lock():
                    inflight.value -= len(packets)
    except BaseException as exc:
        try:
            error_queue.put((config.shard_id, type(exc).__name__, str(exc)))
        finally:
            stop_event.set()
        raise
    finally:
        ring.close()
        nodes.close()
        edges.close()
        actions.close()
