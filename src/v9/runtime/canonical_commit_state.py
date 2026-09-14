from __future__ import annotations

from typing import Any

from v9.cognition.developmental_stage import StageEvidence
from v9.memory.model import MemoryLevel

from .memory_pipeline_v2 import CanonicalWrite


def ensure_fast_state(runtime: Any) -> None:
    if not hasattr(runtime, "_fast_stable_contingencies"):
        runtime._fast_stable_contingencies = sum(
            int(int(support) >= 2) for support in runtime._m1n_supports.values()
        )


def stage_evidence_fast(runtime: Any) -> StageEvidence:
    ensure_fast_state(runtime)
    strategy_uids = runtime.graph.uids_at_level(MemoryLevel.M7)
    strategies = [runtime.graph.payloads[uid] for uid in strategy_uids]
    return StageEvidence(
        stable_contingencies=int(runtime._fast_stable_contingencies),
        structural_abstractions=len(runtime._m3),
        held_out_transfer_successes=sum(bool(row.validated) for row in runtime._m4.values()),
        mature_consequences=sum(
            bool(runtime.graph.payloads[uid].get("mature"))
            for uid in runtime.graph.uids_at_level(MemoryLevel.M5)
        ),
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
        snapshot = runtime.stage_tracker.close_interval(
            stage_evidence_fast(runtime),
            evidence_watermark=runtime._watermark,
        )
        runtime._stage_interval_events = 0
        next_stage = snapshot.next_stage
    return next_stage


def record_normalized_fast(runtime: Any, relation: Any, initial_write: CanonicalWrite, deferred_rows: list[Any]) -> int:
    ensure_fast_state(runtime)
    signature = int(relation.structural_signature)
    occurrences = runtime._m1n_occurrences.setdefault(signature, [])
    old_support = int(runtime._m1n_supports.get(signature, 0))
    support = old_support + 1
    runtime._m1n_supports[signature] = support
    if old_support < 2 <= support:
        runtime._fast_stable_contingencies += 1

    cache = runtime._normalized_action_cache
    if signature not in cache:
        observable = str(relation.observable_relation)
        prefix, separator, remainder = observable.partition(":")
        action_text, action_separator, _ = remainder.partition(":")
        action = None
        if prefix == "ACTION" and separator and action_separator:
            try:
                action = int(action_text)
            except ValueError:
                pass
        cache[signature] = action
    action = cache[signature]
    if action is not None:
        runtime._actor_action_supports[action] = runtime._actor_action_supports.get(action, 0.0) + 1.0
        runtime._actor_policy_generation += 1

    if len(occurrences) < max(2, runtime.config.scientific.m1n_facts_per_channel):
        occurrences.append(relation)
    runtime._replay_pool[relation.uid] = float(support)
    if support == 1:
        deferred_rows.append(initial_write.runtime_row())
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
