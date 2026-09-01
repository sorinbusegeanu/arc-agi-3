from __future__ import annotations

import json
import os
import queue
from dataclasses import dataclass, replace
from pathlib import Path
from random import Random

from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
    signed_u64,
    stable_u64,
)


_INSTALLED = False
_ACTOR_MODE_ENV = "ARC_AGI3_V8_ACTOR_BEHAVIOR"
_TERMINAL_EFFICIENCY_WEIGHT = 0.25
_EFFICIENCY_SEARCH_RATE = 0.05

_EPISODE_STEPS = 0
_FIRST_WIN_STEPS = 0
_BEST_WIN_STEPS = 0
_LAST_WIN_STEPS = 0
_MAX_TRANSFER_DIAGNOSTIC_ACTIONS = 256

_TRANSFER_EXECUTABLE_FIELDS = (
    "existing M3 or M4 transfer ancestor",
    "existing M7 strategy or grounded M1 action evidence or replayable source trajectory",
    "grounded evidence linked to the transfer ancestor or its source-world provenance",
    "action available in target environment",
)
_TRANSFER_LINEAGE_RELATIONS = {
    int(RelationType.PROVENANCE),
    int(RelationType.EXPLAINS),
    int(RelationType.LEADS_TO),
    int(RelationType.CONTEXT_REFINES),
}


@dataclass(frozen=True, slots=True)
class ActorProgress:
    actor_id: int
    game_id: str
    steps: int
    wins: int
    failures: int
    levels_completed: int
    replans: int = 0
    planned_steps: int = 0
    first_win_step: int = 0
    best_win_steps: int = 0
    last_win_steps: int = 0


def _provisional_concept_ready(row) -> bool:
    return bool(
        int(row.level) == int(MemoryLevel.M4)
        and int(row.memory_type) == int(MemoryType.CONCEPT)
        and int(row.cognitive_state)
        in {
            int(CognitiveState.CANDIDATE),
            int(CognitiveState.PROBATION),
            int(CognitiveState.ACTIVE),
            int(CognitiveState.VALIDATED),
            int(CognitiveState.REACTIVATED),
        }
        and int(row.validation_state)
        in {
            int(ValidationState.STRUCTURAL),
            int(ValidationState.TESTED),
            int(ValidationState.VALIDATED),
        }
        and int(row.support_count) >= 2
        and float(row.explanatory_reach) >= 1.0
        and float(row.transfer_prior) >= 0.25
    )


def _install_provisional_validation_scaffold() -> None:
    """Permit probe-only M5-M7 descendants before M4 empirical validation."""
    from v8 import behavior_recovery as behavior_module
    from v8 import intelligence_loop_v087 as loop_module
    from v8 import peers_v82
    from v8 import promotion as promotion_module

    def concept_parent_ready(candidate, by_uid) -> bool:
        if int(candidate.level) != int(MemoryLevel.M5):
            return True
        for parent_uid in candidate.parents:
            parent = by_uid.get(parent_uid)
            if parent is None:
                continue
            if int(parent.level) != int(MemoryLevel.M4) or int(parent.memory_type) != int(MemoryType.CONCEPT):
                continue
            if (
                int(parent.validation_state) == int(ValidationState.VALIDATED)
                and int(parent.cognitive_state)
                in {int(CognitiveState.VALIDATED), int(CognitiveState.REACTIVATED)}
            ):
                return True
            return _provisional_concept_ready(parent)
        return False

    loop_module._validated_concept_parent = concept_parent_ready

    current_engine = promotion_module.EvidenceGatedPromotionEngine
    base_engine = behavior_module._BasePromotionEngine

    def normalize_candidate(engine, candidate, by_uid):
        if int(candidate.level) == int(MemoryLevel.M4) and candidate.parents:
            role = by_uid.get(candidate.parents[0])
            if role is not None and int(role.memory_type) == int(MemoryType.ROLE):
                key = tuple(int(value) for value in role.key_parts)
                if key:
                    return replace(
                        candidate,
                        key_parts=key,
                        uid=MemoryUid.from_key(MemoryLevel.M4, MemoryType.CONCEPT, key),
                    )
        if int(candidate.level) == int(MemoryLevel.M6):
            canonicalize = getattr(engine, "_canonicalize_m6", None)
            if callable(canonicalize):
                return canonicalize(candidate, by_uid)
        return candidate

    def missing_higher_candidates(engine, rows, edges, by_uid, *, limit: int):
        """Recover higher layers starved by lower-level proposal churn."""
        if limit <= 0:
            return ()
        cancel = getattr(engine, "_v841_cancel_event", None)

        def cancelled() -> bool:
            return bool(cancel is not None and cancel.is_set())

        if cancelled():
            return ()
        seen: set[MemoryUid] = set(by_uid)

        def collect(candidates) -> list[object]:
            selected = []
            for raw in candidates:
                candidate = normalize_candidate(engine, raw, by_uid)
                if candidate.uid in seen:
                    continue
                if int(candidate.level) == int(MemoryLevel.M5) and not concept_parent_ready(
                    candidate, by_uid
                ):
                    continue
                seen.add(candidate.uid)
                selected.append(candidate)
            selected.sort(key=lambda item: item.uid)
            return selected

        m4_sources = []
        m5_sources = []
        m6_sources = []
        has_m6 = False
        for index, row in enumerate(rows):
            if index % 4096 == 0 and cancelled():
                return ()
            level = int(row.level)
            if level == int(MemoryLevel.M3) and int(row.memory_type) == int(MemoryType.ROLE):
                m4_sources.append(row)
            elif level == int(MemoryLevel.M4):
                m5_sources.append(row)
            elif level == int(MemoryLevel.M5):
                m6_sources.append(row)
            elif level == int(MemoryLevel.M6):
                has_m6 = True
        # Build each missing tier independently so lower-level candidate ordering
        # cannot hide it.  The base engine supplies the exact existing predicates
        # and candidate schemas for each isolated tier.
        m6_candidates = collect(
            base_engine.propose(engine, tuple(m6_sources), (), budget=limit)
        )
        if cancelled():
            return ()
        m5_candidates = collect(
            base_engine.propose(engine, tuple(m5_sources), (), budget=limit)
        )
        if cancelled():
            return ()
        m4_candidates = collect(
            base_engine.propose(engine, tuple(m4_sources), (), budget=limit)
        )

        m7_builder = getattr(engine, "_causal_strategies", None)
        m7_candidates = []
        if callable(m7_builder) and has_m6 and not cancelled():
            m7_candidates = collect(m7_builder(rows, edges, limit=limit))

        # Round-robin the tiers so sustained work at one depth cannot recreate the
        # starvation that this reservation removes at another depth.
        tiers = (m7_candidates, m6_candidates, m5_candidates, m4_candidates)
        offsets = [0, 0, 0, 0]
        result = []
        while len(result) < limit:
            progressed = False
            for index, tier in enumerate(tiers):
                offset = offsets[index]
                if offset >= len(tier):
                    continue
                result.append(tier[offset])
                offsets[index] += 1
                progressed = True
                if len(result) >= limit:
                    break
            if not progressed:
                break
        return tuple(result)

    class V088PromotionEngine(current_engine):
        def propose(self, nodes, edges, *, budget: int = 256):
            limit = max(0, int(budget))
            if limit <= 0:
                return ()
            rows = tuple(nodes)
            graph = tuple(edges)
            by_uid = {row.uid: row for row in rows}
            base = tuple(super().propose(rows, graph, budget=limit))
            reserve = min(limit, max(1, limit // 4))
            supplements = missing_higher_candidates(
                self,
                rows,
                graph,
                by_uid,
                limit=reserve,
            )
            result = []
            seen: set[MemoryUid] = set()
            base_higher = tuple(
                candidate for candidate in base if int(candidate.level) >= int(MemoryLevel.M4)
            )
            base_lower = tuple(
                candidate for candidate in base if int(candidate.level) < int(MemoryLevel.M4)
            )
            for candidate in (*supplements, *base_higher, *base_lower):
                candidate = normalize_candidate(self, candidate, by_uid)
                if candidate.uid in seen:
                    continue
                seen.add(candidate.uid)
                result.append(candidate)
                if len(result) >= limit:
                    break
            return tuple(result)

    promotion_module.EvidenceGatedPromotionEngine = V088PromotionEngine
    peers_v82.EvidenceGatedPromotionEngine = V088PromotionEngine
    behavior_module.CausalEvidenceGatedPromotionEngine = V088PromotionEngine


def _memory_free_action(actions, rng: Random) -> int:
    choices = tuple(sorted(set(int(value) for value in actions)))
    if not choices:
        raise ValueError("memory-free policy requires at least one action")
    return choices[rng.randrange(len(choices))]


def _mapped_evidence_action(
    evidence_actions: tuple[int, ...],
    available_actions: tuple[int, ...],
    cursor: int,
) -> tuple[int | None, int]:
    """Map one existing source action by exact target action identity."""
    if not evidence_actions:
        return None, int(cursor)
    available = {int(value) for value in available_actions}
    start = int(cursor) % len(evidence_actions)
    for offset in range(len(evidence_actions)):
        index = (start + offset) % len(evidence_actions)
        action = int(evidence_actions[index])
        if action in available:
            return action, (index + 1) % len(evidence_actions)
    return None, start


def _probe_policy_v088(
    *,
    read_view,
    game_id: str,
    env_root: str | None,
    seed: int,
    steps: int,
    required_ancestor: MemoryUid | None,
    execution_evidence: dict[str, object] | None = None,
    diagnostic: dict[str, object] | None = None,
) -> tuple[float, int]:
    """Matched intervention: target-memory ON versus genuinely memory-free OFF."""
    from v7.environment.arc_adapter import ArcGridEnvironment
    from v7.environment.encoding import structural_grid_signature

    env = ArcGridEnvironment(game_id=game_id, seed=seed, env_root=env_root)
    rng = Random(int(seed) ^ 0x8A11)
    wins = failures = level_gain = used = 0
    observed_contexts: set[int] = set()
    observed_actions: set[int] = set()
    executed_actions: list[int] = []
    executed_action_count = 0
    lower_actions_used: set[int] = set()
    plan_misses = 0
    memory_queries = 0
    evidence_cursor = 0
    evidence_used = 0
    evidence_unsupported_steps = 0
    planned_strategy_ids: set[str] = set()
    evidence_actions = tuple(
        int(value) for value in (execution_evidence or {}).get("action_ids", ())
    )
    initial_state_signature = None
    initial_available_actions: tuple[int, ...] = ()
    last_levels = int(env.last_levels_completed)
    for _ in range(max(1, int(steps))):
        before = env.observe()
        actions = tuple(sorted(set(int(value) for value in env.available_actions())))
        if initial_state_signature is None:
            initial_state_signature = int(structural_grid_signature(before))
            initial_available_actions = actions
        if not actions:
            env.reset()
            last_levels = int(env.last_levels_completed)
            continue
        observed_actions.update(actions)
        if required_ancestor is None:
            action = _memory_free_action(actions, rng)
        else:
            context = int(structural_grid_signature(before))
            observed_contexts.add(context)
            memory_queries += 1
            plan = read_view.planned_action(
                context,
                actions,
                required_ancestor=required_ancestor,
                ignore_preference=True,
            )
            if plan is None:
                plan_misses += 1
                mapped, evidence_cursor = _mapped_evidence_action(
                    evidence_actions, actions, evidence_cursor
                )
                if mapped is None:
                    evidence_unsupported_steps += int(bool(evidence_actions))
                    action = _memory_free_action(actions, rng)
                else:
                    action = int(mapped)
                    used += 1
                    evidence_used += 1
                    lower_actions_used.add(int(mapped))
            else:
                action = int(plan.action_id)
                used += 1
                planned_strategy_ids.add(str(plan.strategy_uid.hex()))
        env.step(action)
        executed_action_count += 1
        if len(executed_actions) < _MAX_TRANSFER_DIAGNOSTIC_ACTIONS:
            executed_actions.append(int(action))
        if env.last_outcome_polarity == "positive":
            wins += 1
        elif env.last_outcome_polarity == "negative":
            failures += 1
        current_levels = int(env.last_levels_completed)
        if current_levels > last_levels:
            level_gain += current_levels - last_levels
        last_levels = current_levels
    metric = (5.0 * wins + 2.0 * level_gain - 0.25 * failures) / max(1.0, float(steps))
    if diagnostic is not None:
        try:
            diagnostic.update(
                {
                    "probe_steps": max(1, int(steps)),
                    "plan_misses": int(plan_misses),
                    "planned_steps": int(used),
                    "m7_planned_steps": int(used - evidence_used),
                    "lower_level_evidence_steps": int(evidence_used),
                    "lower_level_evidence_unsupported_steps": int(
                        evidence_unsupported_steps
                    ),
                    "execution_evidence_kind": (execution_evidence or {}).get("kind"),
                    "execution_evidence_action_ids": list(evidence_actions[:32]),
                    "target_world": str(game_id),
                    "probe_seed": int(seed),
                    "initial_state_signature": initial_state_signature,
                    "initial_available_action_ids": list(initial_available_actions),
                    "executed_target_actions": list(executed_actions),
                    "executed_target_action_count": int(executed_action_count),
                    "executed_target_actions_truncated": bool(
                        executed_action_count > len(executed_actions)
                    ),
                    "memory_policy_enabled": required_ancestor is not None,
                    "required_ancestor_id": _uid_value(required_ancestor),
                    "target_memory_query_count": int(memory_queries),
                    "target_memory_leakage": False,
                    "outcome": {
                        "wins": int(wins),
                        "failures": int(failures),
                        "level_gain": int(level_gain),
                    },
                    "score": float(metric),
                    "grounded_source_memory_ids_actually_used": (
                        list(
                            (execution_evidence or {}).get(
                                "source_grounded_memory_ids", ()
                            )
                        )[:8]
                        if evidence_used > 0
                        else []
                    ),
                    "grounded_correspondence_memory_ids_actually_used": [
                        str(row.get("memory_id"))
                        for row in (execution_evidence or {}).get(
                            "mapped_action_evidence", ()
                        )
                        if int(row.get("action_id", 0)) in lower_actions_used
                        and row.get("memory_id") is not None
                    ][:8],
                    "observed_context_signatures": sorted(observed_contexts)[:8],
                    "observed_context_buckets": sorted(
                        stable_u64(value, person=b"v8-context")
                        for value in observed_contexts
                    )[:8],
                    "available_action_ids": sorted(observed_actions)[:32],
                    "planned_strategy_ids": sorted(planned_strategy_ids)[:8],
                }
            )
        except BaseException:
            pass
    return float(metric), used


def _held_out_games(training_games: tuple[str, ...], env_root: str | None) -> tuple[str, ...]:
    from v7.environment.arc_adapter import registered_game_ids
    from v7.game_sets import FALSIFICATION_GAMES, TRANSFER_VALIDATION_GAMES

    training = set(str(value) for value in training_games)
    available = tuple(sorted(registered_game_ids(env_root)))
    declared = tuple(dict.fromkeys((*TRANSFER_VALIDATION_GAMES, *FALSIFICATION_GAMES, *available)))
    available_set = set(available)
    return tuple(
        game_id
        for game_id in declared
        if game_id not in training and (not available_set or game_id in available_set)
    )


def _coherent_cached_transfer_cut(view):
    """Return the exact strategy-cache graph cut without rereading live arenas."""
    version = tuple(getattr(view, "_strategy_version", ()))
    node_arenas = tuple(getattr(view, "_nodes", ()))
    edge_arenas = tuple(getattr(view, "_edges", ()))
    arenas = (*node_arenas, *edge_arenas)
    if not version or len(version) != len(arenas):
        return None
    record_cache = getattr(view, "_record_cache", None)
    node_by_uid = getattr(view, "_node_by_uid", None)
    if not isinstance(record_cache, dict) or not isinstance(node_by_uid, dict):
        return None

    edge_cuts = []
    for index, arena in enumerate(arenas):
        cached = record_cache.get(id(arena))
        if cached is None or int(cached[1]) != int(version[index]):
            return None
        if index >= len(node_arenas):
            edge_cuts.append(tuple(cached[0]))
    if tuple(getattr(view, "_strategy_version", ())) != version:
        return None
    nodes = tuple(node_by_uid.values())
    edge_version = tuple(version[len(node_arenas) :])
    if tuple(getattr(view, "_v839_transfer_version", ())) == edge_version:
        edges = tuple(getattr(view, "_v839_transfer_edges", ()))
    else:
        edges = tuple(row for cut in edge_cuts for row in cut)
    return nodes, edges


def _enum_name(enum_type, value) -> str:
    try:
        return enum_type(int(value)).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{value}"


def _uid_value(value) -> str | None:
    if value is None:
        return None
    method = getattr(value, "hex", None)
    return str(method()) if callable(method) else str(value)


def _has_cached_ancestor(
    parents: dict[MemoryUid, set[MemoryUid]],
    uid: MemoryUid,
    ancestor_uid: MemoryUid,
    *,
    max_depth: int = 8,
) -> bool:
    if uid == ancestor_uid:
        return True
    frontier = {uid}
    visited = set(frontier)
    for _depth in range(max(0, int(max_depth))):
        following: set[MemoryUid] = set()
        for current in frontier:
            for parent in parents.get(current, ()):
                if parent == ancestor_uid:
                    return True
                if parent not in visited:
                    visited.add(parent)
                    following.add(parent)
        if not following:
            return False
        frontier = following
    return False


def _grounded_lineage_action_evidence(
    read_view,
    nodes,
    edges,
    candidate_uid: MemoryUid,
    *,
    max_depth: int = 8,
) -> dict[str, object]:
    """Resolve existing action-bearing M1 memories below an M3/M4 ancestor."""
    node_by_uid = getattr(read_view, "_node_by_uid", None)
    if not isinstance(node_by_uid, dict) or candidate_uid not in node_by_uid:
        node_by_uid = {row.uid: row for row in nodes}
    candidate = node_by_uid.get(candidate_uid)
    if candidate is None:
        return {
            "kind": None,
            "action_ids": [],
            "action_evidence": [],
            "grounded_memory_ids": [],
            "resolution_status": "required_ancestor_lookup_returned_none",
            "failure_reason": "required_ancestor_lookup_returned_none",
        }

    parents = getattr(read_view, "_parents", None)
    if not isinstance(parents, dict) or candidate_uid not in parents:
        parents = {}
        for edge in edges or ():
            if int(edge.relation_type) in _TRANSFER_LINEAGE_RELATIONS:
                parents.setdefault(edge.source_uid, set()).add(edge.target_uid)

    blocked_states = {
        int(CognitiveState.QUARANTINED),
        int(CognitiveState.RETIRE_PENDING),
        int(CognitiveState.RETIRED),
    }
    grounded = []
    frontier = {candidate_uid}
    visited = set(frontier)
    for _depth in range(max(0, int(max_depth))):
        following: set[MemoryUid] = set()
        for current in frontier:
            for parent in parents.get(current, ()):
                row = node_by_uid.get(parent)
                if (
                    row is not None
                    and int(row.level) == int(MemoryLevel.M1)
                    and int(row.memory_type) == int(MemoryType.CONTINGENCY)
                    and len(row.key_parts) >= 4
                    and int(row.support_count) > 0
                    and int(row.cognitive_state) not in blocked_states
                    and int(row.validation_state) != int(ValidationState.FAILED)
                ):
                    grounded.append(row)
                if parent not in visited:
                    visited.add(parent)
                    following.add(parent)
        if not following:
            break
        frontier = following

    grounded.sort(
        key=lambda row: (-int(row.support_count), -int(row.updated_watermark), row.uid)
    )
    actions = []
    action_evidence = []
    seen_actions: set[int] = set()
    for row in grounded:
        action = signed_u64(int(row.key_parts[1]))
        if action in seen_actions:
            continue
        seen_actions.add(action)
        actions.append(action)
        action_evidence.append(
            {"action_id": int(action), "memory_id": _uid_value(row.uid)}
        )
        if len(actions) >= 64:
            break
    if not actions:
        return {
            "kind": None,
            "action_ids": [],
            "action_evidence": [],
            "grounded_memory_ids": [],
            "resolution_status": "no_grounded_action_evidence_in_lineage",
            "failure_reason": "no_grounded_action_evidence_in_lineage",
        }
    return {
        "kind": "grounded_lineage",
        "action_ids": actions,
        "action_evidence": action_evidence,
        "grounded_memory_ids": [
            str(row["memory_id"]) for row in action_evidence[:8]
        ],
        "grounded_memory_count": len(grounded),
        "resolution_status": "grounded_action_evidence_resolved",
        "failure_reason": None,
    }


def _node_diagnostic(row) -> dict[str, object] | None:
    if row is None:
        return None
    return {
        "memory_id": _uid_value(row.uid),
        "memory_level": _enum_name(MemoryLevel, row.level),
        "memory_level_value": int(row.level),
        "memory_kind": _enum_name(MemoryType, row.memory_type),
        "memory_type_value": int(row.memory_type),
        "key_parts": [int(value) for value in row.key_parts],
        "support_count": int(row.support_count),
        "cognitive_state": _enum_name(CognitiveState, row.cognitive_state),
        "validation_state": _enum_name(ValidationState, row.validation_state),
        "updated_watermark": int(row.updated_watermark),
    }


def _strategy_cache_rows(read_view) -> tuple[object, ...]:
    result: list[object] = []
    seen: set[MemoryUid] = set()
    for values in getattr(read_view, "_strategy_by_context", {}).values():
        for row in values:
            if row.strategy_uid not in seen:
                seen.add(row.strategy_uid)
                result.append(row)
    for row in getattr(read_view, "_strategy_fallback", ()):
        if row.strategy_uid not in seen:
            seen.add(row.strategy_uid)
            result.append(row)
    return tuple(result)


def _candidate_execution_snapshot(
    read_view,
    nodes,
    candidate_uid: MemoryUid,
    execution_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    """Describe the already-built planning cache without making another decision."""
    node_by_uid = getattr(read_view, "_node_by_uid", None)
    if not isinstance(node_by_uid, dict):
        node_by_uid = {row.uid: row for row in nodes}
    parents = getattr(read_view, "_parents", {})
    if not isinstance(parents, dict):
        parents = {}
    candidate = node_by_uid.get(candidate_uid)
    relevant_levels = {int(MemoryLevel.M4), int(MemoryLevel.M7)}
    descendants = [
        row
        for row in nodes
        if int(row.level) in relevant_levels
        and _has_cached_ancestor(parents, row.uid, candidate_uid)
    ]
    m4_descendants = [row for row in descendants if int(row.level) == int(MemoryLevel.M4)]
    m7_descendants = [row for row in descendants if int(row.level) == int(MemoryLevel.M7)]
    cached = [
        row
        for row in _strategy_cache_rows(read_view)
        if _has_cached_ancestor(parents, row.strategy_uid, candidate_uid)
    ]
    missing_fields: list[str] = []
    if candidate is None:
        reason = "required_ancestor_lookup_returned_none"
        missing_fields.append("required transfer ancestor memory")
    elif int(candidate.level) not in {int(MemoryLevel.M3), int(MemoryLevel.M4)}:
        reason = "required_ancestor_wrong_memory_level"
        missing_fields.append("M3 or M4 required transfer ancestor")
    elif cached:
        reason = "executable_m7_descendant_available_for_target_probe"
    elif (execution_evidence or {}).get("action_ids"):
        reason = "lower_level_transfer_evidence_available_for_target_probe"
    else:
        reason = str(
            (execution_evidence or {}).get(
                "failure_reason",
                "no_grounded_action_evidence_in_lineage_and_no_replayable_source_trajectory",
            )
        )
        missing_fields.append(
            "grounded lineage action evidence or replayable source trajectory"
        )
    return {
        "lookup_result_status": (
            "not_found" if candidate is None else "found_structural_ancestor"
        ),
        "required_ancestor_memory": _node_diagnostic(candidate),
        "candidate_is_m3": bool(candidate is not None and int(candidate.level) == int(MemoryLevel.M3)),
        "candidate_is_m4": bool(candidate is not None and int(candidate.level) == int(MemoryLevel.M4)),
        "m4_descendant_count": len(m4_descendants),
        "m4_descendant_ids": [_uid_value(row.uid) for row in m4_descendants[:8]],
        "m7_descendant_count": len(m7_descendants),
        "m7_descendant_ids": [_uid_value(row.uid) for row in m7_descendants[:8]],
        "cached_executable_descendant_count": len(cached),
        "cached_executable_descendants": [
            {
                "strategy_id": _uid_value(row.strategy_uid),
                "action_id": int(row.action_id),
                "outcome_id": _uid_value(row.outcome_uid),
                "context_bucket": int(row.context_bucket),
                "support": int(row.support),
                "probationary": bool(row.probationary),
                "transferable_fallback": bool(row.transferable),
            }
            for row in cached[:8]
        ],
        "lower_level_execution_evidence": dict(execution_evidence or {}),
        "executable_predicate_failure_reason": reason,
        "required_executable_fields": list(_TRANSFER_EXECUTABLE_FIELDS),
        "missing_or_invalid_executable_fields": missing_fields,
    }


def _direct_target_memory_inventories(
    read_view,
    nodes,
    target_hashes: tuple[int, ...],
) -> dict[int, dict[str, object]]:
    direct = getattr(read_view, "_v839_direct_games", {})
    if not isinstance(direct, dict):
        direct = {}
    by_uid = getattr(read_view, "_node_by_uid", None)
    if not isinstance(by_uid, dict):
        by_uid = {row.uid: row for row in nodes}
    wanted = {int(value) for value in target_hashes}
    rows_by_hash: dict[int, list[object]] = {value: [] for value in wanted}
    seen_by_hash: dict[int, set[MemoryUid]] = {value: set() for value in wanted}
    for uid, games in direct.items():
        row = by_uid.get(uid)
        if row is None:
            continue
        for value in wanted.intersection(int(game) for game in games):
            rows_by_hash[value].append(row)
            seen_by_hash[value].add(row.uid)
    source_games = getattr(read_view, "source_games", None)
    if callable(source_games):
        for row in nodes:
            if int(row.level) not in {
                int(MemoryLevel.M3), int(MemoryLevel.M4), int(MemoryLevel.M7)
            }:
                continue
            try:
                inherited = wanted.intersection(int(game) for game in source_games(row.uid))
            except BaseException:
                continue
            for value in inherited:
                if row.uid not in seen_by_hash[value]:
                    rows_by_hash[value].append(row)
                    seen_by_hash[value].add(row.uid)
    result: dict[int, dict[str, object]] = {}
    for target_hash, rows in rows_by_hash.items():
        levels: dict[str, int] = {}
        for row in rows:
            name = _enum_name(MemoryLevel, row.level)
            levels[name] = levels.get(name, 0) + 1
        m3 = [row for row in rows if int(row.level) == int(MemoryLevel.M3)]
        m4 = [row for row in rows if int(row.level) == int(MemoryLevel.M4)]
        lower = [
            row
            for row in rows
            if int(row.level) == int(MemoryLevel.M1)
            and int(row.memory_type) == int(MemoryType.CONTINGENCY)
            and len(row.key_parts) >= 4
            and int(row.support_count) > 0
        ]
        result[target_hash] = {
            "identity_index": "GAME_PROVENANCE target UID low word plus lineage inheritance",
            "identities_actually_present_count": len(rows),
            "memory_counts_by_level": levels,
            "m3_exists": bool(m3),
            "m3_ids": [_uid_value(row.uid) for row in m3[:8]],
            "m4_exists": bool(m4),
            "m4_ids": [_uid_value(row.uid) for row in m4[:8]],
            "action_bearing_lower_level_memory_exists": bool(lower),
            "action_bearing_lower_level_memory_ids": [_uid_value(row.uid) for row in lower[:8]],
            "executable_lower_level_memory_exists": bool(lower),
            "executable_lower_level_memory_ids": [_uid_value(row.uid) for row in lower[:8]],
            "lower_level_execution_note": (
                "held-out transfer may reuse grounded M1 actions without an M7 strategy"
            ),
        }
    return result


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _trajectory_optimizer_root() -> Path | None:
    raw_root = str(os.environ.get("ARC_AGI3_V8_ROOT", "")).strip()
    trajectory_root = str(os.environ.get("ARC_AGI3_V8_TRAJECTORY_ROOT", "")).strip()
    if raw_root:
        return Path(raw_root) / "trajectory_optimizer"
    return Path(trajectory_root) if trajectory_root else None


def _trajectory_actions(row: dict[str, object]) -> tuple[int, ...]:
    actions = row.get("actions")
    if isinstance(actions, list):
        try:
            return tuple(int(value) for value in actions)
        except (TypeError, ValueError):
            return ()
    levels = row.get("levels")
    if not isinstance(levels, list):
        return ()
    result = []
    try:
        for level in levels:
            if isinstance(level, dict) and isinstance(level.get("actions"), list):
                result.extend(int(value) for value in level["actions"])
    except (TypeError, ValueError):
        return ()
    return tuple(result)


def _source_trajectory_index(games: tuple[str, ...]) -> dict[int, dict[str, object]]:
    root = _trajectory_optimizer_root()
    if root is None:
        return {}
    by_game: dict[str, list[dict[str, object]]] = {}
    for filename in ("best_successful.json", "generic_best_successful.json"):
        payload = _load_json_object(root / filename)
        for key in ("games", "environments"):
            store = payload.get(key)
            if not isinstance(store, dict):
                continue
            for game_id, raw in store.items():
                if not isinstance(raw, dict):
                    continue
                actions = _trajectory_actions(raw)
                if not actions:
                    continue
                by_game.setdefault(str(game_id), []).append(
                    {
                        "kind": "source_trajectory",
                        "action_ids": list(actions),
                        "source_game_id": str(game_id),
                        "source_world_hash": stable_u64(str(game_id), person=b"v8-game"),
                        "trajectory_id": str(
                            raw.get("trajectory_id", raw.get("variant_id", ""))
                        ) or None,
                        "trajectory_source": filename,
                    }
                )
    validated = _load_json_object(root / "validated.json").get("validated")
    if isinstance(validated, list):
        for raw in validated:
            if not isinstance(raw, dict):
                continue
            anchor = raw.get("anchor")
            game_id = str(anchor.get("source_id", "")) if isinstance(anchor, dict) else ""
            actions = _trajectory_actions(raw)
            try:
                successes = int(raw.get("successes", 0) or 0)
            except (TypeError, ValueError):
                successes = 0
            if not game_id or not actions or successes <= 0:
                continue
            by_game.setdefault(game_id, []).append(
                {
                    "kind": "source_trajectory",
                    "action_ids": list(actions),
                    "source_game_id": game_id,
                    "source_world_hash": stable_u64(game_id, person=b"v8-game"),
                    "trajectory_id": str(raw.get("variant_id", "")) or None,
                    "trajectory_source": "validated.json",
                }
            )

    result: dict[int, dict[str, object]] = {}
    ordered_games = tuple(dict.fromkeys((*games, *sorted(by_game))))
    for game_id in ordered_games:
        choices = by_game.get(str(game_id), ())
        if not choices:
            continue
        selected = min(
            choices,
            key=lambda row: (
                len(row["action_ids"]),
                str(row.get("trajectory_id") or ""),
                str(row.get("trajectory_source") or ""),
            ),
        )
        result[int(selected["source_world_hash"])] = dict(selected)
    return result


def _transfer_execution_evidence(
    read_view,
    nodes,
    edges,
    candidate,
    source_trajectories: dict[int, dict[str, object]],
) -> dict[str, object]:
    source_grounded = _grounded_lineage_action_evidence(
        read_view, nodes, edges, candidate.uid
    )
    source_trajectory = next(
        (
            source_trajectories[int(world)]
            for world in candidate.formation_games
            if int(world) in source_trajectories
        ),
        None,
    )
    source_actions = tuple(int(value) for value in source_grounded.get("action_ids", ()))
    source_kind = "grounded_source_lineage"
    if not source_actions and source_trajectory is not None:
        source_actions = tuple(
            int(value) for value in source_trajectory.get("action_ids", ())
        )
        source_kind = "replayable_source_trajectory"
    if not source_actions:
        return {
            **source_grounded,
            "kind": None,
            "action_ids": [],
            "source_action_sequence": [],
            "source_trajectory": None,
            "correspondence_conditioned_mapping": False,
            "resolution_status": "no_executable_lower_level_transfer_evidence",
            "failure_reason": (
                "no_grounded_action_evidence_in_lineage_and_no_replayable_source_trajectory"
            ),
        }

    correspondence_uid = candidate.correspondence_uid
    correspondence_grounded = _grounded_lineage_action_evidence(
        read_view, nodes, edges, correspondence_uid
    )
    correspondence_trajectory = next(
        (
            source_trajectories[int(world)]
            for world in candidate.correspondence_games
            if int(world) in source_trajectories
        ),
        None,
    )
    mapped_actions = tuple(
        int(value) for value in correspondence_grounded.get("action_ids", ())
    )
    mapped_kind = "grounded_correspondence_lineage"
    mapped_action_evidence = list(correspondence_grounded.get("action_evidence", ()))
    if not mapped_actions and correspondence_trajectory is not None:
        mapped_actions = tuple(
            int(value) for value in correspondence_trajectory.get("action_ids", ())
        )
        mapped_kind = "replayable_correspondence_trajectory"
        mapped_action_evidence = [
            {
                "action_id": int(value),
                "memory_id": None,
                "trajectory_id": correspondence_trajectory.get("trajectory_id"),
            }
            for value in mapped_actions
        ]
    if not mapped_actions:
        return {
            **source_grounded,
            "kind": None,
            "action_ids": [],
            "source_action_sequence": list(source_actions[:64]),
            "source_evidence_kind": source_kind,
            "source_grounded_memory_ids": list(
                source_grounded.get("grounded_memory_ids", ())
            )[:8],
            "source_trajectory": source_trajectory,
            "correspondence_ancestor_id": _uid_value(correspondence_uid),
            "correspondence_grounded_memory_ids": [],
            "correspondence_trajectory": None,
            "correspondence_conditioned_mapping": False,
            "resolution_status": "correspondence_action_mapping_unresolved",
            "failure_reason": (
                "no_grounded_action_evidence_in_correspondence_lineage_and_"
                "no_replayable_correspondence_trajectory"
            ),
        }

    mapping = {
        "kind": "formal_correspondence_to_grounded_counterpart_lineage",
        "source_ancestor_id": _uid_value(candidate.uid),
        "correspondence_ancestor_id": _uid_value(correspondence_uid),
        "correspondence_score": float(candidate.structural_score),
        "source_action_sequence": list(source_actions[:64]),
        "mapped_action_sequence": list(mapped_actions[:64]),
        "selection_rule": (
            "execute existing grounded actions from the formally corresponding "
            "ancestor lineage; target availability may only filter these actions"
        ),
    }
    return {
        "kind": mapped_kind,
        "action_ids": list(mapped_actions[:64]),
        "mapped_action_evidence": mapped_action_evidence[:64],
        "source_action_sequence": list(source_actions[:64]),
        "source_evidence_kind": source_kind,
        "source_grounded_memory_ids": list(
            source_grounded.get("grounded_memory_ids", ())
        )[:8],
        "source_trajectory": source_trajectory,
        "correspondence_ancestor_id": _uid_value(correspondence_uid),
        "correspondence_grounded_memory_ids": list(
            correspondence_grounded.get("grounded_memory_ids", ())
        )[:8],
        "correspondence_trajectory": correspondence_trajectory,
        "correspondence_mapping": mapping,
        "correspondence_conditioned_mapping": True,
        "generic_source_actions_reused_without_correspondence": False,
        "resolution_status": "correspondence_conditioned_transfer_evidence_resolved",
        "failure_reason": None,
    }


def _trajectory_inventory(games: tuple[str, ...]) -> dict[str, dict[str, object]]:
    root = _trajectory_optimizer_root()
    if root is None:
        return {game: {"successful_trajectory_exists": False} for game in games}
    result = {game: {"successful_trajectory_exists": False} for game in games}
    for filename in ("best_successful.json", "generic_best_successful.json"):
        payload = _load_json_object(root / filename)
        stores = []
        for key in ("games", "environments"):
            value = payload.get(key)
            if isinstance(value, dict):
                stores.append(value)
        for game in games:
            for store in stores:
                row = store.get(game)
                if not isinstance(row, dict):
                    continue
                levels = row.get("levels")
                actions = row.get("actions")
                executable = bool(
                    (isinstance(actions, list) and actions)
                    or (isinstance(levels, list) and any(
                        isinstance(level, dict) and level.get("actions") for level in levels
                    ))
                )
                result[game] = {
                    "successful_trajectory_exists": True,
                    "trajectory_id": str(row.get("trajectory_id", row.get("variant_id", ""))) or None,
                    "trajectory_source": filename,
                    "linked_memory_id": row.get("strategy_uid") or row.get("parent_strategy_uid"),
                    "memory_level": "M7" if row.get("strategy_uid") else None,
                    "executable_representation_available": executable,
                    "action_sequence_schema": "actions" if isinstance(actions, list) else "levels[].actions",
                }
                break
    validated = _load_json_object(root / "validated.json").get("validated")
    if isinstance(validated, list):
        for row in validated:
            if not isinstance(row, dict):
                continue
            anchor = row.get("anchor")
            game = str(anchor.get("source_id", "")) if isinstance(anchor, dict) else ""
            if game not in result or result[game]["successful_trajectory_exists"]:
                continue
            result[game] = {
                "successful_trajectory_exists": True,
                "trajectory_id": str(row.get("variant_id", "")) or None,
                "trajectory_source": "validated.json",
                "linked_memory_id": row.get("strategy_uid") or row.get("parent_strategy_uid"),
                "memory_level": "M7" if row.get("strategy_uid") else None,
                "executable_representation_available": bool(row.get("actions")),
                "action_sequence_schema": "actions",
            }
    return result


def _executable_reference(read_view, nodes) -> dict[str, object]:
    """Return one bounded representation already executable elsewhere in v8."""
    node_by_uid = getattr(read_view, "_node_by_uid", None)
    if not isinstance(node_by_uid, dict):
        node_by_uid = {row.uid: row for row in nodes}
    cached = _strategy_cache_rows(read_view)
    if cached:
        strategy = cached[0]
        return {
            "representation": "graph_planner_strategy_cache",
            "runtime_consumer": "LiveReadView.planned_action",
            "memory": _node_diagnostic(node_by_uid.get(strategy.strategy_uid)),
            "action_sequence_representation": {"action_id": int(strategy.action_id)},
            "trajectory_or_source_reference": None,
            "environment_or_world_reference": "context_bucket",
            "provenance": "graph lineage and GAME_PROVENANCE edges",
            "replay_or_execution_metadata": {
                "outcome_id": _uid_value(strategy.outcome_uid),
                "context_bucket": int(strategy.context_bucket),
                "transferable_fallback": bool(strategy.transferable),
            },
            "adapter_specific_fields": [],
        }
    raw_root = str(os.environ.get("ARC_AGI3_V8_ROOT", "")).strip()
    trajectory_root = str(os.environ.get("ARC_AGI3_V8_TRAJECTORY_ROOT", "")).strip()
    root = Path(raw_root) / "trajectory_optimizer" if raw_root else (
        Path(trajectory_root) if trajectory_root else None
    )
    if root is not None:
        for filename in ("best_successful.json", "generic_best_successful.json"):
            payload = _load_json_object(root / filename)
            for key in ("games", "environments"):
                store = payload.get(key)
                if not isinstance(store, dict):
                    continue
                for game, row in sorted(store.items()):
                    if not isinstance(row, dict):
                        continue
                    actions = row.get("actions")
                    levels = row.get("levels")
                    if not (
                        (isinstance(actions, list) and actions)
                        or (isinstance(levels, list) and levels)
                    ):
                        continue
                    return {
                        "representation": "durable_successful_trajectory_sidecar",
                        "runtime_consumer": "restored competence replay",
                        "memory": None,
                        "memory_level": None,
                        "payload_schema": filename,
                        "action_sequence_representation": (
                            "actions" if isinstance(actions, list) else "levels[].actions"
                        ),
                        "trajectory_or_source_reference": str(
                            row.get("trajectory_id", row.get("variant_id", ""))
                        ) or None,
                        "environment_or_world_reference": str(game),
                        "provenance": row.get("source"),
                        "replay_or_execution_metadata": {
                            "seed": row.get("seed"),
                            "terminal_state": row.get("terminal_state"),
                            "successes": row.get("successes"),
                        },
                        "adapter_specific_fields": (
                            ["levels"] if isinstance(levels, list) else ["seed"]
                        ),
                    }
    return {
        "representation": "none_found_in_current_graph_or_durable_sidecars",
        "runtime_consumer": None,
        "required_graph_schema": list(_TRANSFER_EXECUTABLE_FIELDS),
    }


def _target_resolution_event(
    *,
    read_view,
    nodes,
    candidate,
    game_id: str,
    target_hash: int,
    probe_diagnostic: dict[str, object],
    candidate_snapshot: dict[str, object],
    target_inventory: dict[str, object],
    trajectory: dict[str, object],
    executable_reference: dict[str, object],
    used: int,
) -> dict[str, object]:
    ancestor = candidate_snapshot.get("required_ancestor_memory")
    candidate_games = sorted(int(value) for value in candidate.formation_games)
    correspondence_games = sorted(int(value) for value in candidate.correspondence_games)
    linked = trajectory.get("linked_memory_id")
    requested_identity = {
        "selected_game_or_environment_id": str(game_id),
        "provenance_world_hash": int(target_hash),
        "required_ancestor_uid": _uid_value(candidate.uid),
        "context_buckets": probe_diagnostic.get("observed_context_buckets", []),
    }
    failure = candidate_snapshot["executable_predicate_failure_reason"]
    missing_fields = list(candidate_snapshot["missing_or_invalid_executable_fields"])
    if int(used) <= 0 and failure == "lower_level_transfer_evidence_available_for_target_probe":
        if not probe_diagnostic.get("available_action_ids"):
            failure = "target_environment_exposed_no_actions"
            missing_fields.append("target environment action availability")
        else:
            failure = "resolved_source_actions_unsupported_by_target_environment"
            missing_fields.append("source action supported by target environment")
    elif int(used) <= 0 and failure == "executable_m7_descendant_available_for_target_probe":
        cached = candidate_snapshot.get("cached_executable_descendants", [])
        actions = {int(value) for value in probe_diagnostic.get("available_action_ids", [])}
        contexts = {int(value) for value in probe_diagnostic.get("observed_context_buckets", [])}
        action_rows = [row for row in cached if int(row.get("action_id", -1)) in actions]
        if not action_rows:
            failure = "descendant_strategy_action_unavailable_in_target_environment"
            missing_fields.append("action available in target environment")
        elif not any(
            int(row.get("context_bucket", -1)) in contexts
            or bool(row.get("transferable_fallback"))
            for row in action_rows
        ):
            failure = "no_matching_target_context_or_transferable_fallback"
            missing_fields.append("matching target context or transferable fallback")
        else:
            failure = "composed_planner_or_execution_adapter_rejected_cached_strategy"
    if int(used) > 0:
        failure = None
    return {
        "source_world": candidate_games,
        "target_world": str(game_id),
        "transfer_candidate_id": _uid_value(candidate.uid),
        "correspondence_id": _uid_value(candidate.correspondence_uid),
        "target_memory_id": None if ancestor is None else ancestor.get("memory_id"),
        "target_memory_role": "required source-side transfer ancestor; no target-specific object is looked up",
        "target_memory_level": None if ancestor is None else ancestor.get("memory_level"),
        "target_memory_kind": None if ancestor is None else ancestor.get("memory_kind"),
        "lookup_key_used": requested_identity,
        "lookup_result_status": candidate_snapshot["lookup_result_status"],
        "target_specific_memory_lookup_performed": False,
        "requested_lookup_identity": requested_identity,
        "candidate_memory_identities_actually_present": {
            "formation_world_hashes": candidate_games,
            "correspondence_world_hashes": correspondence_games,
        },
        "provenance_world_hash": int(target_hash),
        "selected_game_or_environment_id": str(game_id),
        "adapter_family": "ARC",
        "execution_adapter": "v7.environment.arc_adapter.ArcGridEnvironment",
        "executable_predicate": (
            "probe used an existing M7 plan, grounded lineage action, or replayable "
            "source trajectory action at least once (used > 0)"
        ),
        "executable_predicate_result": bool(int(used) > 0),
        "exact_executable_predicate_failure_reason": failure,
        "required_executable_fields": candidate_snapshot["required_executable_fields"],
        "missing_or_invalid_executable_fields": missing_fields,
        "successful_trajectory": trajectory,
        "successful_trajectory_link_reached_by_transfer_lookup": bool(linked) and bool(int(used) > 0),
        "target_world_memory": target_inventory,
        "m3_exists_for_target_world": bool(target_inventory.get("m3_exists")),
        "m4_exists_for_target_world": bool(target_inventory.get("m4_exists")),
        "executable_lower_level_memory_exists_for_target_world": bool(
            target_inventory.get("executable_lower_level_memory_exists")
        ),
        "required_ancestor_resolution": candidate_snapshot,
        "probe_observations": probe_diagnostic,
        "executable_vs_rejected_comparison": {
            "executable_reference": executable_reference,
            "rejected_reference": ancestor,
            "minimal_structural_difference": (
                "rejected transfer ancestor has neither usable M7 control nor existing "
                "grounded source actions supported by the target"
            ),
        },
    }


def _transfer_trial_failure_reason(trial, threshold: float) -> str | None:
    if bool(trial.passed):
        return None
    effect = float(trial.effect)
    if effect == float(threshold):
        return "transfer_effect_equal_to_existing_threshold"
    if effect < float(threshold):
        return "transfer_effect_below_existing_threshold"
    return "transfer_trial_not_held_out"


def _matched_probe_diagnostic(
    intervention: dict[str, object],
    control: dict[str, object],
) -> dict[str, object]:
    same_target = intervention.get("target_world") == control.get("target_world")
    same_seed = intervention.get("probe_seed") == control.get("probe_seed")
    same_horizon = intervention.get("probe_steps") == control.get("probe_steps")
    same_initial_state = (
        intervention.get("initial_state_signature")
        == control.get("initial_state_signature")
    )
    same_initial_actions = (
        intervention.get("initial_available_action_ids")
        == control.get("initial_available_action_ids")
    )
    control_memory_free = (
        control.get("memory_policy_enabled") is False
        and int(control.get("target_memory_query_count", 0)) == 0
        and control.get("execution_evidence_kind") is None
    )
    return {
        "same_target_world": bool(same_target),
        "same_initial_target_state": bool(same_initial_state),
        "same_initial_available_actions": bool(same_initial_actions),
        "same_seed": bool(same_seed),
        "same_horizon": bool(same_horizon),
        "intervention_memory_policy_enabled": bool(
            intervention.get("memory_policy_enabled")
        ),
        "control_memory_policy_enabled": bool(control.get("memory_policy_enabled")),
        "control_target_memory_query_count": int(
            control.get("target_memory_query_count", 0)
        ),
        "control_target_memory_leakage": not bool(control_memory_free),
        "only_transfer_policy_differs": bool(
            same_target
            and same_initial_state
            and same_initial_actions
            and same_seed
            and same_horizon
            and control_memory_free
        ),
    }


def _run_automatic_transfer_experiments_v088(
    runtime,
    *,
    games: tuple[str, ...],
    env_root: str | None,
    seed: int,
    steps_per_trial: int = 32,
    max_trials: int = 8,
):
    from v8.experiments import ExperimentSummary
    from v8 import information_flow_diagnostics as flow

    if runtime.peers is None or max_trials <= 0 or steps_per_trial <= 0:
        reason = "peer_supervisor_unavailable" if runtime.peers is None else "transfer_trial_budget_disabled"
        flow.emit("transfer", "transfer_experiment_scheduling", input_count=0,
                  output_count=0, rejection_counts={reason: 1})
        return ExperimentSummary(0, 0, 0)

    holdouts = _held_out_games(tuple(games), env_root)
    if not holdouts:
        flow.emit("transfer", "transfer_experiment_scheduling", input_count=0,
                  output_count=0, rejection_counts={"no_held_out_worlds": 1})
        return ExperimentSummary(0, 0, 0)

    cached_cut = _coherent_cached_transfer_cut(runtime.read_view)
    if cached_cut is None:
        nodes = runtime.read_view.node_records()
        edges = None
    else:
        nodes, edges = cached_cut
    by_uid = {row.uid: row for row in nodes}
    if edges is None:
        candidates = runtime.peers.transfer.candidates(
            nodes,
            provenance=runtime.read_view.source_games,
        )
    else:
        candidates = runtime.peers.transfer.candidates(
            nodes,
            edges=edges,
            provenance=runtime.read_view.source_games,
        )
    attempted = completed = passed = 0
    eligible_target_worlds = 0
    considered_pairs = 0
    scheduling_rejections: dict[str, int] = {}
    scheduling_examples: list[dict[str, object]] = []
    eligibility_probes = 0
    eligibility_probe_limit = max(1, int(max_trials))
    game_hashes = {game: stable_u64(game, person=b"v8-game") for game in holdouts}
    target_inventories = _direct_target_memory_inventories(
        runtime.read_view, nodes, tuple(game_hashes.values())
    )
    trajectories = _trajectory_inventory(holdouts)
    source_trajectories = _source_trajectory_index(tuple(games))
    executable_reference = _executable_reference(runtime.read_view, nodes)
    candidate_execution: dict[MemoryUid, dict[str, object]] = {}
    candidate_evidence: dict[MemoryUid, dict[str, object]] = {}
    mapped_sequence_targets: dict[tuple[int, ...], set[str]] = {}

    def reject(reason: str) -> None:
        scheduling_rejections[reason] = scheduling_rejections.get(reason, 0) + 1

    def add_example(candidate, game_id, target_hash, *, eligible, decision, reason) -> None:
        if len(scheduling_examples) >= flow.MAX_EXAMPLES:
            return
        row = by_uid.get(candidate.uid)
        scheduling_examples.append(
            {"source_world": list(candidate.formation_games),
             "candidate_target_world": str(game_id),
             "candidate_target_world_hash": int(target_hash),
             "candidate_uid": flow.uid_text(candidate.uid),
             "correspondence_uid": flow.uid_text(candidate.correspondence_uid),
             "correspondence_score": float(candidate.structural_score),
             "provenance_distinct": set(candidate.formation_games) != set(candidate.correspondence_games),
             "m3_available": bool(row is not None and int(row.level) == int(MemoryLevel.M3)),
             "m4_available": bool(row is not None and int(row.level) == int(MemoryLevel.M4)),
             "held_out_eligibility": bool(eligible),
             "scheduler_decision": str(decision),
             "rejection_reason": reason}
        )

    def finish(stop_reason: str | None = None):
        failed = completed - passed
        if stop_reason is not None:
            reject(stop_reason)
        flow.add_counters(
            "transfer", eligible_target_worlds=eligible_target_worlds,
            scheduled_trials=attempted, completed_trials=completed,
            passed_trials=passed, failed_trials=failed,
        )
        flow.emit(
            "transfer", "transfer_experiment_scheduling",
            input_count=considered_pairs, output_count=attempted,
            rejection_counts=scheduling_rejections, examples=scheduling_examples,
            fields={"eligible_target_worlds": eligible_target_worlds,
                    "scheduled_trials": attempted, "completed_trials": completed,
                    "passed_trials": passed, "failed_trials": failed,
                    "eligibility_probe_limit": eligibility_probe_limit},
        )
        flow.emit(
            "transfer", "transfer_trial_completion", input_count=completed,
            output_count=passed,
            rejection_counts=(
                {"transfer_effect_not_above_existing_threshold": failed}
                if failed else {}
            ),
            examples=scheduling_examples,
            fields={"completed_trials": completed, "passed_trials": passed,
                    "failed_trials": failed},
        )
        observed = flow.counter_snapshot("transfer")
        pipeline = {
            "structural_correspondence_count": int(observed.get("structural_correspondence_count", 0)),
            "transfer_structural_count": int(observed.get("transfer_structural_count", 0)),
            "admissible_candidates": len(candidates),
            "eligible_target_worlds": eligible_target_worlds,
            "scheduled_trials": attempted,
            "completed_trials": completed,
            "passed_trials": passed,
            "failed_trials": failed,
        }
        flow.emit(
            "transfer", "pipeline_summary", input_count=pipeline["structural_correspondence_count"],
            output_count=passed, rejection_counts=scheduling_rejections,
            fields={"counters": pipeline, "counter_scope": "current_process_observed"},
        )
        return ExperimentSummary(attempted, completed, passed)

    for candidate in sorted(candidates, key=lambda row: (-row.structural_score, row.uid)):
        formation = tuple(candidate.formation_games)
        for game_id in holdouts:
            if completed >= int(max_trials):
                return finish("completed_trial_limit_reached")
            target_hash = int(game_hashes[game_id])
            considered_pairs += 1
            if target_hash in formation:
                reject("target_world_in_formation_provenance")
                add_example(candidate, game_id, target_hash, eligible=False,
                            decision="rejected", reason="target_world_in_formation_provenance")
                continue
            # max_trials bounds completed causal comparisons, but an inapplicable
            # ancestor does not become a trial. Bound that ranked eligibility scan
            # separately so a graph containing only unexecutable transfer concepts
            # cannot hold shutdown in environment probes indefinitely.
            if eligibility_probes >= eligibility_probe_limit:
                add_example(candidate, game_id, target_hash, eligible=False,
                            decision="not_scheduled", reason="eligibility_probe_limit_reached")
                return finish("eligibility_probe_limit_reached")
            eligibility_probes += 1
            trial_seed = int(seed) + (completed + 1) * 7919
            probe_diagnostic: dict[str, object] = {}
            execution_evidence = candidate_evidence.get(candidate.uid)
            if execution_evidence is None:
                execution_evidence = _transfer_execution_evidence(
                    runtime.read_view,
                    nodes,
                    edges,
                    candidate,
                    source_trajectories,
                )
                candidate_evidence[candidate.uid] = execution_evidence
            on_metric, used = _probe_policy_v088(
                read_view=runtime.read_view,
                game_id=game_id,
                env_root=env_root,
                seed=trial_seed,
                steps=steps_per_trial,
                required_ancestor=candidate.uid,
                execution_evidence=execution_evidence,
                diagnostic=probe_diagnostic,
            )
            candidate_snapshot = candidate_execution.get(candidate.uid)
            if candidate_snapshot is None:
                candidate_snapshot = _candidate_execution_snapshot(
                    runtime.read_view,
                    nodes,
                    candidate.uid,
                    execution_evidence,
                )
                candidate_execution[candidate.uid] = candidate_snapshot
            resolution = _target_resolution_event(
                read_view=runtime.read_view,
                nodes=nodes,
                candidate=candidate,
                game_id=game_id,
                target_hash=target_hash,
                probe_diagnostic=probe_diagnostic,
                candidate_snapshot=candidate_snapshot,
                target_inventory=target_inventories.get(
                    target_hash,
                    {
                        "identity_index": "GAME_PROVENANCE target UID low word",
                        "identities_actually_present_count": 0,
                        "memory_counts_by_level": {},
                        "m3_exists": False,
                        "m4_exists": False,
                        "executable_lower_level_memory_exists": False,
                    },
                ),
                trajectory=trajectories.get(
                    game_id, {"successful_trajectory_exists": False}
                ),
                executable_reference=executable_reference,
                used=used,
            )
            flow.emit_bounded(
                "transfer",
                "target_memory_resolution",
                input_count=1,
                output_count=int(used > 0),
                rejection_counts=(
                    {str(resolution["exact_executable_predicate_failure_reason"]): 1}
                    if used <= 0 else {}
                ),
                examples=(
                    {
                        "transfer_candidate_id": resolution["transfer_candidate_id"],
                        "target_world": game_id,
                        "trajectory_id": resolution["successful_trajectory"].get("trajectory_id"),
                    },
                ),
                fields=resolution,
            )
            if used <= 0:
                rejection_reason = str(
                    resolution.get("exact_executable_predicate_failure_reason")
                    or "target_memory_not_executable"
                )
                reject(rejection_reason)
                add_example(candidate, game_id, target_hash, eligible=False,
                            decision="not_scheduled", reason=rejection_reason)
                continue
            eligible_target_worlds += 1
            attempted += 1
            control_diagnostic: dict[str, object] = {}
            off_metric, _ = _probe_policy_v088(
                read_view=runtime.read_view,
                game_id=game_id,
                env_root=env_root,
                seed=trial_seed,
                steps=steps_per_trial,
                required_ancestor=None,
                diagnostic=control_diagnostic,
            )
            trial = runtime.peers.record_transfer_trial(
                candidate.uid,
                target_game_hash=target_hash,
                metric_on=on_metric,
                metric_off=off_metric,
                formation_games=formation,
                intervention="matched_arc_target_memory_vs_memory_free",
            )
            threshold = float(runtime.peers.transfer.effect_threshold)
            failure_reason = _transfer_trial_failure_reason(trial, threshold)
            mapped_sequence = tuple(
                int(value) for value in execution_evidence.get("action_ids", ())
            )
            prior_mapping_targets = mapped_sequence_targets.setdefault(
                mapped_sequence, set()
            )
            reused_mapping = bool(
                prior_mapping_targets and str(game_id) not in prior_mapping_targets
            )
            matched_control = _matched_probe_diagnostic(
                probe_diagnostic, control_diagnostic
            )
            flow.emit_bounded(
                "transfer",
                "transfer_trial_evaluation",
                input_count=1,
                output_count=int(trial.passed),
                rejection_counts=({failure_reason: 1} if failure_reason else {}),
                examples=(
                    {
                        "source_ancestor_id": _uid_value(candidate.uid),
                        "target_world": str(game_id),
                        "computed_transfer_effect": float(trial.effect),
                        "passed": bool(trial.passed),
                    },
                ),
                fields={
                    "source_ancestor_id": _uid_value(candidate.uid),
                    "source_ancestor_level": _enum_name(
                        MemoryLevel, getattr(by_uid.get(candidate.uid), "level", -1)
                    ),
                    "grounded_source_memory_ids_actually_used": list(
                        probe_diagnostic.get(
                            "grounded_source_memory_ids_actually_used", ()
                        )
                    ),
                    "grounded_correspondence_memory_ids_actually_used": list(
                        probe_diagnostic.get(
                            "grounded_correspondence_memory_ids_actually_used", ()
                        )
                    ),
                    "exact_source_action_sequence": list(
                        execution_evidence.get("source_action_sequence", ())
                    ),
                    "exact_actions_executed_on_target": list(
                        probe_diagnostic.get("executed_target_actions", ())
                    ),
                    "correspondence_ancestor_id": _uid_value(
                        candidate.correspondence_uid
                    ),
                    "correspondence_score": float(candidate.structural_score),
                    "correspondence_mapping": dict(
                        execution_evidence.get("correspondence_mapping", {})
                    ),
                    "correspondence_conditioned_mapping": bool(
                        execution_evidence.get("correspondence_conditioned_mapping")
                    ),
                    "generic_source_actions_reused_without_correspondence": bool(
                        execution_evidence.get(
                            "generic_source_actions_reused_without_correspondence"
                        )
                    ),
                    "mapped_sequence_reused_across_distinct_targets": reused_mapping,
                    "prior_targets_with_same_mapped_sequence": sorted(
                        prior_mapping_targets
                    )[:8],
                    "target_world": str(game_id),
                    "target_world_hash": int(target_hash),
                    "intervention": {
                        "outcome": dict(probe_diagnostic.get("outcome", {})),
                        "score": float(on_metric),
                        "seed": int(trial_seed),
                        "horizon": int(steps_per_trial),
                        "initial_state_signature": probe_diagnostic.get(
                            "initial_state_signature"
                        ),
                        "initial_available_action_ids": list(
                            probe_diagnostic.get("initial_available_action_ids", ())
                        ),
                        "target_memory_query_count": int(
                            probe_diagnostic.get("target_memory_query_count", 0)
                        ),
                    },
                    "matched_memory_free_control": {
                        "outcome": dict(control_diagnostic.get("outcome", {})),
                        "score": float(off_metric),
                        "seed": int(trial_seed),
                        "horizon": int(steps_per_trial),
                        "initial_state_signature": control_diagnostic.get(
                            "initial_state_signature"
                        ),
                        "initial_available_action_ids": list(
                            control_diagnostic.get("initial_available_action_ids", ())
                        ),
                        "actions": list(
                            control_diagnostic.get("executed_target_actions", ())
                        ),
                        "target_memory_query_count": int(
                            control_diagnostic.get("target_memory_query_count", 0)
                        ),
                        "target_memory_leakage": bool(
                            matched_control["control_target_memory_leakage"]
                        ),
                    },
                    "matched_control_checks": matched_control,
                    "computed_transfer_effect": float(trial.effect),
                    "existing_pass_threshold": threshold,
                    "pass_predicate": "held_out and effect > effect_threshold",
                    "passed": bool(trial.passed),
                    "exact_failure_reason": failure_reason,
                },
            )
            prior_mapping_targets.add(str(game_id))
            completed += 1
            passed += int(trial.passed)
            add_example(
                candidate, game_id, target_hash, eligible=True,
                decision="transfer_trial_pass" if trial.passed else "transfer_trial_fail",
                reason=None if trial.passed else "transfer_effect_not_above_existing_threshold",
            )
            if not trial.passed:
                row = by_uid.get(candidate.uid)
                if row is not None:
                    runtime.peers._append_evidence(
                        "transfer_trial_fail",
                        row,
                        abs(float(trial.effect)),
                        unique=True,
                        target_game_hash=target_hash,
                        provenance_games=formation,
                        causal_intervention="matched_arc_target_memory_vs_memory_free",
                        effect_direction=-1,
                    )
                    if int(row.level) == int(MemoryLevel.M4):
                        runtime.peers._append_evidence(
                            "concept_transfer_fail",
                            row,
                            abs(float(trial.effect)),
                            unique=True,
                            target_game_hash=target_hash,
                            provenance_games=formation,
                            causal_intervention="matched_arc_target_memory_vs_memory_free",
                            effect_direction=-1,
                        )
    return finish()


def _install_transfer_experiments() -> None:
    from v8 import experiments as experiments_module

    experiments_module._probe_policy = _probe_policy_v088
    experiments_module.run_automatic_transfer_experiments = _run_automatic_transfer_experiments_v088


def _install_probe_planning_and_efficiency_search() -> None:
    from v8 import behavior_recovery as behavior_module
    from v8 import learning_blockers_v055 as blocker_module
    from v8.publication import LiveReadView

    base_score_rows = behavior_module._score_strategy_rows

    def score_rows(view, rows, **kwargs):
        plans = list(base_score_rows(view, rows, **kwargs))
        by_uid = {row.strategy_uid: row for row in rows}
        adjusted = []
        for plan in plans:
            row = by_uid.get(plan.strategy_uid)
            if row is None:
                adjusted.append(plan)
                continue
            absolute = 0.10 / max(1.0, float(row.mean_cost))
            adjusted.append(replace(plan, score=float(plan.score) + absolute))
        adjusted.sort(key=lambda item: (-item.score, item.action_id, item.strategy_uid))
        return tuple(adjusted)

    behavior_module._score_strategy_rows = score_rows

    base_composites = blocker_module._composite_plans

    def composite_plans(view, context_signature, action_ids):
        plans = list(base_composites(view, context_signature, action_ids))
        by_uid = getattr(view, "_node_by_uid", {})
        adjusted = []
        for plan in plans:
            row = by_uid.get(plan.strategy_uid)
            if row is None or float(getattr(row, "attempt_weight", 0.0)) <= 0.0:
                adjusted.append(plan)
                continue
            empirical = 0.10 / max(1.0, float(getattr(row, "strategy_mean_cost", 1.0)))
            adjusted.append(replace(plan, score=float(plan.score) + empirical))
        adjusted.sort(key=lambda item: (-item.score, item.action_id, item.strategy_uid))
        return tuple(adjusted)

    blocker_module._composite_plans = composite_plans

    current_plan_candidates = LiveReadView.plan_candidates

    def plan_candidates(self, context_signature, action_ids, **kwargs):
        required_ancestor = kwargs.get("required_ancestor")
        plans = tuple(current_plan_candidates(self, context_signature, action_ids, **kwargs))

        if required_ancestor is not None:
            filtered = tuple(
                plan
                for plan in plans
                if self.strategy_has_ancestor(plan.strategy_uid, required_ancestor)
            )
            if filtered:
                return filtered

            self._refresh_strategy_cache()
            context_bucket = stable_u64(int(context_signature), person=b"v8-context")
            available = {int(value) for value in action_ids}
            exact = list(getattr(self, "_strategy_by_context", {}).get(context_bucket, ()))
            rows = [
                row
                for row in exact
                if row.action_id in available
                and self.strategy_has_ancestor(row.strategy_uid, required_ancestor)
                and behavior_module._strategy_can_probe(self, row.strategy_uid, row.outcome_uid)
            ]
            probe_plans = behavior_module._score_strategy_rows(
                self,
                rows,
                available=available,
                outcome_uid=kwargs.get("outcome_uid"),
                required_ancestor=required_ancestor,
                excluded_strategies=kwargs.get("excluded_strategies", frozenset()),
                ignore_preference=True,
                cross_context=False,
            )
            return tuple(probe_plans)

        if (
            plans
            and bool(getattr(self, "_behavior_actor_mode", False))
            and getattr(self, "_v055_active_sequence", None) is None
        ):
            best = plans[0]
            by_uid = getattr(self, "_node_by_uid", {})
            strategy = by_uid.get(best.strategy_uid)
            outcome = by_uid.get(best.outcome_uid)
            strategy_value = 0.0 if strategy is None else float(strategy.expected_primary_valence) * float(strategy.primary_valence_confidence)
            outcome_value = 0.0 if outcome is None else float(outcome.expected_primary_valence) * float(outcome.primary_valence_confidence)
            if max(strategy_value, outcome_value) > 0.05:
                rng = getattr(self, "_behavior_rng", None)
                if rng is not None and rng.random() < _EFFICIENCY_SEARCH_RATE:
                    self._refresh_strategy_cache()
                    context_bucket = stable_u64(int(context_signature), person=b"v8-context")
                    exact = list(getattr(self, "_strategy_by_context", {}).get(context_bucket, ()))
                    alternatives = [
                        row
                        for row in exact
                        if row.strategy_uid != best.strategy_uid
                        and row.outcome_uid == best.outcome_uid
                        and behavior_module._strategy_can_probe(self, row.strategy_uid, row.outcome_uid)
                    ]
                    if alternatives:
                        alternatives.sort(
                            key=lambda row: (
                                float(getattr(by_uid.get(row.strategy_uid), "attempt_weight", 0.0)),
                                float(row.mean_cost),
                                row.strategy_uid,
                            )
                        )
                        probe = behavior_module._score_strategy_rows(
                            self,
                            alternatives[:8],
                            available={int(value) for value in action_ids},
                            outcome_uid=best.outcome_uid,
                            required_ancestor=None,
                            excluded_strategies=frozenset(),
                            ignore_preference=True,
                            cross_context=False,
                        )
                        if probe:
                            self._behavior_last_plans = (probe[0],)
                            return (probe[0],)
                    self._v055_active_sequence = None
                    self._behavior_force_random = True
                    self._behavior_last_plans = ()
                    return ()
        return plans

    LiveReadView.plan_candidates = plan_candidates


def _install_terminal_efficiency_feedback() -> None:
    from v8 import runtime as runtime_module

    base_record = runtime_module.ContinuousMemoryRuntime.record_actor_results

    def record_actor_results(self, results):
        rows = tuple(results)
        base_record(self, rows)
        _record_terminal_efficiency_feedback(self, rows)

    runtime_module.ContinuousMemoryRuntime.record_actor_results = record_actor_results


def _record_terminal_efficiency_feedback(runtime, rows) -> None:
    """Record terminal efficiency from the existing live index in bounded time."""
    if runtime.peers is None:
        return

    from v8 import trajectory_efficiency_v054 as efficiency_module

    # Actor credits refer to strategies selected from this view.  Use its existing
    # coherent index instead of decoding every node after each feedback batch.  On
    # large restored graphs, a full refresh races active writers and can keep the
    # feedback worker busy past the five-minute shutdown deadline.
    by_uid = getattr(runtime.read_view, "_node_by_uid", {})
    for result in rows:
        game_hash = stable_u64(result.game_id, person=b"v8-game")
        for credit in getattr(result, "primary_valence_credits", ()):
            if (
                int(credit.level) != int(MemoryLevel.M7)
                or float(credit.valence_sum) <= 0.0
            ):
                continue
            actions = efficiency_module._actions_from_discounted_valence(credit)
            row = by_uid.get(credit.uid)
            if (
                actions is None
                or row is None
                or int(getattr(row, "level", -1)) != int(MemoryLevel.M7)
            ):
                continue
            weight = _TERMINAL_EFFICIENCY_WEIGHT
            runtime.peers._submit(
                runtime.peers._existing_proposal(
                    row,
                    success_sum=weight,
                    cost_sum=float(actions) * weight,
                    attempt_weight=weight,
                    source_game_hash=int(game_hash),
                )
            )
            runtime.peers._append_evidence(
                "terminal_strategy_efficiency",
                row,
                min(1.0, 1.0 / max(1.0, float(actions))),
                unique=True,
                provenance_games=(int(game_hash),),
                causal_intervention="positive_terminal_distance",
                effect_direction=1,
            )


def _install_solve_efficiency_reporting() -> None:
    from v7.environment import arc_adapter as adapter
    from v8 import actor as actor_module
    from v8 import diagnostics as diagnostics_module

    base_step = adapter.ArcGridEnvironment.step
    base_reset = adapter.ArcGridEnvironment.reset

    def step(self, action):
        global _EPISODE_STEPS, _FIRST_WIN_STEPS, _BEST_WIN_STEPS, _LAST_WIN_STEPS
        actor_mode = os.environ.get(_ACTOR_MODE_ENV) == "1"
        if actor_mode:
            _EPISODE_STEPS += 1
        result = base_step(self, action)
        if actor_mode:
            if bool(getattr(self, "last_step_was_reset_boundary", False)):
                _EPISODE_STEPS = 0
            else:
                state = str(getattr(self, "last_outcome_state", ""))
                if state == "WIN":
                    solved = max(1, int(_EPISODE_STEPS))
                    if _FIRST_WIN_STEPS <= 0:
                        _FIRST_WIN_STEPS = solved
                    _LAST_WIN_STEPS = solved
                    _BEST_WIN_STEPS = solved if _BEST_WIN_STEPS <= 0 else min(_BEST_WIN_STEPS, solved)
                    _EPISODE_STEPS = 0
                elif state == "GAME_OVER":
                    _EPISODE_STEPS = 0
        return result

    def reset(self, *args, **kwargs):
        global _EPISODE_STEPS
        result = base_reset(self, *args, **kwargs)
        if os.environ.get(_ACTOR_MODE_ENV) == "1":
            _EPISODE_STEPS = 0
        return result

    def publish_progress(
        progress_queue,
        reporting_queue=None,
        *,
        job,
        steps: int,
        wins: int,
        failures: int,
        levels_completed: int,
        replans: int,
        planned_steps: int,
    ) -> None:
        row = ActorProgress(
            int(job.actor_id),
            str(job.game_id),
            int(steps),
            int(wins),
            int(failures),
            int(levels_completed),
            int(replans),
            int(planned_steps),
            int(_FIRST_WIN_STEPS),
            int(_BEST_WIN_STEPS),
            int(_LAST_WIN_STEPS),
        )
        for target in (progress_queue, reporting_queue):
            if target is None:
                continue
            try:
                target.put_nowait(row)
            except queue.Full:
                pass

    def format_game_rate_line(rows) -> str:
        rows = tuple(rows)
        win_rate, level_rate, solved_games, games = diagnostics_module.game_summary(rows)
        grouped = diagnostics_module._group_games(rows)
        details = []
        for game_id, lane_rows in sorted(grouped.items()):
            solved_rows = [row for row in lane_rows if int(getattr(row, "wins", 0)) > 0]
            if not solved_rows:
                continue
            first_values = [int(getattr(row, "first_win_step", 0) or 0) for row in solved_rows]
            best_values = [int(getattr(row, "best_win_steps", 0) or 0) for row in solved_rows]
            last_values = [int(getattr(row, "last_win_steps", 0) or 0) for row in solved_rows]
            first = min(
                (value for value in first_values if value > 0),
                default=min((int(getattr(row, "steps", 0) or 0) for row in solved_rows), default=0),
            )
            explicit_best = min((value for value in best_values if value > 0), default=0)
            explicit_last = min((value for value in last_values if value > 0), default=0)
            best = explicit_best or first
            last = explicit_last or best
            if explicit_best > 0 or explicit_last > 0:
                details.append(f"{game_id}:first={first},best={best},last={last}")
            else:
                details.append(f"{game_id}:{first}")
        suffix = "" if not details else " (" + "; ".join(details) + ")"
        return (
            f"current_run_wins={win_rate:.1f}% current_run_levels_solved={level_rate:.1f}% "
            f"current_run_solved_games={solved_games}/{games}{suffix}"
        )

    adapter.ArcGridEnvironment.step = step
    adapter.ArcGridEnvironment.reset = reset
    actor_module.ActorProgress = ActorProgress
    actor_module._publish_progress = publish_progress
    diagnostics_module.format_game_rate_line = format_game_rate_line


def install_learning_fixes_v088() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_provisional_validation_scaffold()
    _install_transfer_experiments()
    _install_probe_planning_and_efficiency_search()
    _install_terminal_efficiency_feedback()
    _install_solve_efficiency_reporting()
    _INSTALLED = True
