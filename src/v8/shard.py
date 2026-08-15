from __future__ import annotations

import multiprocessing as mp
import queue
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
from v8.model import CognitiveState, MemoryLevel, MemoryUid, ValidationState, decode_proposal, signed_u64, u64
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


def shard_worker(
    config: ShardConfig,
    ring_args: dict[str, object],
    stop_event: mp.synchronize.Event,
    inflight: mp.sharedctypes.Synchronized,
    watermark: mp.sharedctypes.Synchronized,
    error_queue: mp.Queue,
    batch_size: int = 256,
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

        seen: set[tuple[int, int, int, int]] = set()

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
            try:
                nodes.begin_write()
                edges.begin_write()
                actions.begin_write()
                node_count = nodes.count
                edge_count = edges.count
                max_watermark = int(watermark.value)
                try:
                    for payload in packets:
                        proposal = decode_proposal(payload)
                        dedupe = (
                            int(proposal.event_id.hi),
                            int(proposal.event_id.lo),
                            int(proposal.uid.hi),
                            int(proposal.uid.lo),
                        )
                        if dedupe in seen:
                            continue
                        seen.add(dedupe)
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
                                updated_watermark=max(current.updated_watermark, proposal.watermark),
                                game_mask=int(current.game_mask) | game_bit,
                                cognitive_state=cognitive_state,
                                validation_state=validation_state,
                            )
                        nodes.write(row, record)

                        if not proposal.parent_uid.is_zero:
                            edge_key = (
                                proposal.uid,
                                int(proposal.relation_type),
                                proposal.parent_uid,
                            )
                            edge_row = edge_index.get(edge_key)
                            if edge_row is None:
                                if edge_count >= edges.capacity:
                                    raise MemoryError(f"edge arena full in shard {config.shard_id}")
                                edge_row = edge_count
                                edge_count += 1
                                edge_index[edge_key] = edge_row
                                edge_record = EdgeRecord(
                                    proposal.uid,
                                    int(proposal.relation_type),
                                    proposal.parent_uid,
                                    max(1, int(proposal.support_delta)),
                                    proposal.watermark,
                                )
                            else:
                                prior_edge = edges.read(edge_row)
                                edge_record = replace(
                                    prior_edge,
                                    support_count=prior_edge.support_count + max(1, int(proposal.support_delta)),
                                    updated_watermark=max(prior_edge.updated_watermark, proposal.watermark),
                                )
                            edges.write(edge_row, edge_record)

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
