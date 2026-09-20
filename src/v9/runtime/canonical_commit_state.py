from __future__ import annotations

from typing import Any

from v9.cognition.developmental_stage import StageEvidence
from v9.cognition.action_selection import scoped_action_key
from v9.memory.model import MemoryLevel

from .memory_pipeline import CanonicalWrite


def ensure_fast_state(runtime: Any) -> None:
    if not hasattr(runtime, "_fast_stable_contingencies"):
        runtime._fast_stable_contingencies = sum(int(len(rows) >= 2) for rows in runtime._m1n_occurrences.values())
    if not hasattr(runtime, "_m1n_family_occurrences"):
        runtime._m1n_family_occurrences = {}
        for rows in runtime._m1n_occurrences.values():
            for row in rows:
                family = int(getattr(row, "family_signature", 0) or row.structural_signature)
                bucket = runtime._m1n_family_occurrences.setdefault(family, [])
                identity = (row.uid, row.channel.value, tuple(row.provenance.evidence))
                if all((existing.uid, existing.channel.value, tuple(existing.provenance.evidence)) != identity for existing in bucket):
                    bucket.append(row)


def stage_evidence_fast(runtime: Any) -> StageEvidence:
    ensure_fast_state(runtime)
    strategy_uids = runtime.graph.uids_at_level(MemoryLevel.M7)
    strategies = [runtime.graph.payloads[uid] for uid in strategy_uids]
    return StageEvidence(
        stable_contingencies=int(runtime._fast_stable_contingencies),
        structural_abstractions=len(runtime._m3),
        held_out_transfer_successes=sum(bool(row.validated) for row in runtime._m4.values()),
        mature_consequences=sum(bool(runtime.graph.payloads[uid].get("mature")) for uid in runtime.graph.uids_at_level(MemoryLevel.M5)),
        outcome_equivalences=runtime.graph.memory_count(MemoryLevel.M6),
        learned_preferences=sum(int(row.get("primary_valence_sum", 0)) != 0 for row in strategies),
        alternative_strategies=max(0, len(strategies) - 1),
        demonstrated_replans=runtime._replans_demonstrated,
        efficient_replans=runtime._efficient_replans,
    )


def advance_stage_fast(runtime: Any) -> Any:
    runtime._stage_interval_events += 1
    next_stage = runtime.stage_tracker.stage
    if runtime._stage_interval_events >= runtime._stage_interval_size:
        snapshot = runtime.stage_tracker.close_interval(stage_evidence_fast(runtime), evidence_watermark=runtime._watermark)
        runtime._stage_interval_events = 0
        next_stage = snapshot.next_stage
    return next_stage


def record_normalized_fast(runtime: Any, relation: Any, initial_write: CanonicalWrite, deferred_rows: list[Any], *, materialized_row: Any | None = None) -> int:
    ensure_fast_state(runtime)
    normalize = getattr(runtime, "normalize_m1n_family", None)
    if callable(normalize):
        relation, initial_write = normalize(relation, initial_write)
    accumulate = getattr(runtime, "accumulate_m1n_evidence", None)
    if callable(accumulate):
        accumulate(relation)

    signature = int(relation.structural_signature)
    occurrences = runtime._m1n_occurrences.setdefault(signature, [])
    family = int(getattr(relation, "family_signature", 0) or signature)
    family_rows = runtime._m1n_family_occurrences.setdefault(family, [])
    identity = (relation.uid, relation.channel.value, tuple(relation.provenance.evidence))
    if all((existing.uid, existing.channel.value, tuple(existing.provenance.evidence)) != identity for existing in family_rows):
        family_rows.append(relation)
        family_limit = max(4, int(runtime.config.scientific.m1n_facts_per_channel) * 4)
        if len(family_rows) > family_limit:
            del family_rows[:-family_limit]

    if getattr(relation, "channel", None) is not None and str(relation.channel.value) == "CROSS_MODAL":
        runtime._cross_modal_signatures.pop(signature, None)
        runtime._cross_modal_signatures[signature] = None
        while len(runtime._cross_modal_signatures) > 8192:
            runtime._cross_modal_signatures.pop(next(iter(runtime._cross_modal_signatures)))
    was_stable = len(occurrences) >= 2
    support = int(runtime.signature_support(signature)) + 1
    runtime._m1n_supports[signature] = support

    cache = runtime._normalized_action_cache
    if signature not in cache:
        observable = str(relation.observable_relation)
        parts = observable.split(":")
        scoped = None
        if len(parts) >= 5 and parts[0] == "ACTION":
            try:
                scoped = scoped_action_key(int(parts[3]), action_schema_id=int(parts[1]), environment_type=parts[2])
            except ValueError:
                pass
        cache[signature] = scoped
    scoped = cache[signature]
    if scoped is not None:
        runtime._actor_action_supports[scoped] = runtime._actor_action_supports.get(scoped, 0.0) + 1.0
        runtime._actor_policy_generation += 1

    if len(occurrences) < max(2, runtime.config.scientific.m1n_facts_per_channel):
        occurrences.append(relation)
    if not was_stable and len(occurrences) >= 2:
        runtime._fast_stable_contingencies += 1

    runtime._replay_pool[relation.uid] = float(support)
    if support == 1:
        deferred_rows.append(initial_write.runtime_row() if materialized_row is None else materialized_row)
    else:
        runtime._m1n_dirty.add(signature)
    return signature


def trim_replay_pool(runtime: Any) -> None:
    limit = int(runtime.config.scientific.replay_candidates)
    overflow = len(runtime._replay_pool) - limit
    if overflow > 0:
        victims = sorted(runtime._replay_pool, key=lambda uid: (runtime._replay_pool[uid], uid))[:overflow]
        for uid in victims:
            runtime._replay_pool.pop(uid, None)
