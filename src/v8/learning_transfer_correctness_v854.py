from __future__ import annotations

"""v8.54 learning and cross-world transfer correctness.

Fixes target-scoped transfer trust, ordered M7 transfer, safe normalized grounding,
exact-provenance similarity selection, adaptive composite formation, and causal
session credit without changing the arena schema.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryProposal,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
    proposal_fingerprint,
    signed_u64,
    stable_u64,
    u64,
)
from v8.persistent_identity import world_id

_INSTALLED = False
_RELATION_SCHEMA = 1
_MAX_COMPOSITE_ACTIONS = 16
_BASE_BUILD_RESTART = None
_BASE_GROUNDED_TRANSFER = None
_BASE_CROSS_GAME = None
_BASE_PLAN_CHAIN = None
_BASE_CLEAR_ROLLOUTS = None
_BASE_OBSERVE_TRANSFER = None
_BASE_FORCED = None
_BASE_DISCOVERY = None
_BASE_RESTART_STEP = None
_BASE_RESET = None
_BASE_CREDIT = None
_BASE_VIEW_INIT = None
_BASE_RECORD_TRIAL = None

_LINEAGE = {
    int(RelationType.PROVENANCE),
    int(RelationType.EXPLAINS),
    int(RelationType.LEADS_TO),
    int(RelationType.CONTEXT_REFINES),
    int(RelationType.SUPERSEDES),
    int(RelationType.DEPENDS_ON),
    int(RelationType.ENABLES),
    int(RelationType.BLOCKS),
}


def _graph_index(nodes, edges, *, max_depth: int = 8):
    by_uid = {row.uid: row for row in nodes}
    direct: dict[MemoryUid, set[int]] = defaultdict(set)
    parents: dict[MemoryUid, set[MemoryUid]] = defaultdict(set)
    for edge in edges:
        relation = int(edge.relation_type)
        if relation == int(RelationType.GAME_PROVENANCE) and int(edge.target_uid.hi) == 0:
            direct[edge.source_uid].add(int(edge.target_uid.lo))
        elif relation in _LINEAGE:
            parents[edge.source_uid].add(edge.target_uid)
    cache: dict[MemoryUid, frozenset[int]] = {}

    def games(uid: MemoryUid) -> frozenset[int]:
        if uid in cache:
            return cache[uid]
        found = set(direct.get(uid, ()))
        frontier, visited = {uid}, {uid}
        for _ in range(max_depth):
            following = set()
            for current in frontier:
                for parent in parents.get(current, ()):
                    found.update(direct.get(parent, ()))
                    if parent not in visited:
                        visited.add(parent)
                        following.add(parent)
            if not following:
                break
            frontier = following
        cache[uid] = frozenset(found)
        return cache[uid]

    return by_uid, parents, games


# ---------------------------------------------------------------------------
# Target-scoped transfer evidence.  Transfer validity belongs to source x target,
# not to the source memory's intrinsic lifecycle state.
# ---------------------------------------------------------------------------


def _relation_key(source_uid: MemoryUid, target_game_hash: int) -> tuple[int, ...]:
    return (
        signed_u64(int(source_uid.hi)),
        signed_u64(int(source_uid.lo)),
        signed_u64(int(target_game_hash)),
        _RELATION_SCHEMA,
    )


def transfer_relation_uid(source_uid: MemoryUid, target_game_hash: int) -> MemoryUid:
    return MemoryUid.from_key(
        MemoryLevel.M4,
        MemoryType.TRANSFER_EVIDENCE,
        _relation_key(source_uid, target_game_hash),
    )


def _decode_relation(row):
    if (
        int(getattr(row, "level", -1)) != int(MemoryLevel.M4)
        or int(getattr(row, "memory_type", -1)) != int(MemoryType.TRANSFER_EVIDENCE)
        or len(getattr(row, "key_parts", ())) < 4
        or int(row.key_parts[3]) != _RELATION_SCHEMA
    ):
        return None
    return (
        MemoryUid(u64(int(row.key_parts[0])), u64(int(row.key_parts[1]))),
        u64(int(row.key_parts[2])),
    )


def _relation_adjustment(view, source_uid: MemoryUid, target_game_hash: int) -> tuple[bool, float]:
    parents: dict[MemoryUid, set[MemoryUid]] = defaultdict(set)
    for source, values in getattr(view, "_parents", {}).items():
        parents[source].update(values)
    try:
        for edge in view.edge_records():
            if int(edge.relation_type) in _LINEAGE:
                parents[edge.source_uid].add(edge.target_uid)
    except BaseException:
        pass
    for dep in getattr(view, "_behavior_strategy_dependencies", {}).get(source_uid, ()):
        parents[source_uid].add(dep)

    lineage, frontier = {source_uid}, {source_uid}
    for _ in range(8):
        following = set()
        for current in frontier:
            for parent in parents.get(current, ()):
                if parent not in lineage:
                    lineage.add(parent)
                    following.add(parent)
        if not following:
            break
        frontier = following

    attempts = 0
    weighted = 0.0
    for row in getattr(view, "_node_by_uid", {}).values():
        decoded = _decode_relation(row)
        if decoded is None:
            continue
        relation_source, target = decoded
        if relation_source not in lineage or int(target) != int(target_game_hash):
            continue
        support = max(1, int(getattr(row, "support_count", 1)))
        attempts += support
        weighted += float(getattr(row, "transfer_prior", 0.0)) * support
    effect = 0.0 if attempts <= 0 else weighted / attempts
    return bool(attempts >= 2 and effect <= 0.0), max(-1.0, min(1.0, effect))


def _record_trial_v854(
    self,
    uid: MemoryUid,
    *,
    target_game_hash: int,
    metric_on: float,
    metric_off: float,
    formation_games: tuple[int, ...] = (),
    intervention: str = "matched_memory_ablation",
):
    trial = self.transfer.record_trial(
        uid,
        target_game_hash=int(target_game_hash),
        metric_on=float(metric_on),
        metric_off=float(metric_off),
        formation_games=tuple(formation_games),
        intervention=str(intervention),
    )
    row = next((x for x in self.read_view.node_records() if x.uid == uid), None)
    if row is None:
        return trial

    key = _relation_key(uid, int(target_game_hash))
    self._submit(
        MemoryProposal(
            uid=transfer_relation_uid(uid, int(target_game_hash)),
            fingerprint=proposal_fingerprint(MemoryLevel.M4, MemoryType.TRANSFER_EVIDENCE, key),
            event_id=self._event_id(),
            watermark=int(self.current_watermark()),
            level=MemoryLevel.M4,
            memory_type=MemoryType.TRANSFER_EVIDENCE,
            key_parts=key,
            support_delta=1,
            significance_sum=min(1.0, abs(float(trial.effect))),
            learning_value_sum=min(1.0, abs(float(trial.effect))),
            transfer_prior_sum=float(trial.effect),
            score_weight=1.0,
            parent_uid=uid,
            relation_type=RelationType.DEPENDS_ON,
            source_game_hash=int(target_game_hash),
            cognitive_state=int(CognitiveState.ACTIVE),
            validation_state=int(ValidationState.TESTED),
        )
    )
    self._append_evidence(
        "transfer_relation_pass" if trial.passed else "transfer_relation_fail",
        row,
        float(trial.effect) if trial.passed else abs(float(trial.effect)),
        validation_state=int(ValidationState.TESTED),
        unique=True,
        target_game_hash=int(target_game_hash),
        provenance_games=tuple(formation_games),
        causal_intervention=str(intervention),
        effect_direction=1 if trial.passed else -1,
    )

    # A concept becomes intrinsically validated only after replicated held-out
    # success in at least two distinct target worlds. A failure never globally
    # fails/quarantines the source or its descendants.
    if int(row.level) == int(MemoryLevel.M4):
        passed_targets = {
            int(x.target_game_hash) for x in self.transfer.trials(uid) if bool(x.passed)
        }
        if len(passed_targets) >= 2:
            self._submit(
                self._existing_proposal(
                    row,
                    cognitive_state=int(CognitiveState.VALIDATED),
                    validation_state=int(ValidationState.VALIDATED),
                )
            )
            self._append_evidence(
                "concept_replicated_transfer_validation",
                row,
                min(1.0, len(passed_targets) / 3.0),
                validation_state=int(ValidationState.VALIDATED),
                unique=True,
                causal_intervention="replicated_held_out_transfer",
                effect_direction=1,
            )
    return trial


# ---------------------------------------------------------------------------
# Ordered M7 transfer. Composite procedures are grounded step-by-step through
# shared normalized M1 facts and must remain context-contiguous in the target.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrderedTransferSequence:
    score: float
    strategy_uid: MemoryUid
    outcome_uid: MemoryUid
    path: tuple[object, ...]


def _row_value(row) -> float:
    return (
        math.log1p(max(0, int(getattr(row, "support_count", 0))))
        + max(0.0, float(getattr(row, "significance", 0.0)))
        + max(0.0, float(getattr(row, "learning_value", 0.0)))
        + max(0.0, float(getattr(row, "expected_primary_valence", 0.0)))
        * max(0.0, float(getattr(row, "primary_valence_confidence", 0.0)))
    )


def _ordered_sequences(view, game_id: str) -> tuple[OrderedTransferSequence, ...]:
    from v8 import behavior_recovery as behavior
    from v8 import learning_blockers_v055 as blockers
    from v8 import normalized_memory_v086 as normalized

    view._refresh_strategy_cache()
    try:
        behavior._refresh_behavior_indexes(view)
    except BaseException:
        pass
    current_game = int(world_id(str(game_id)))
    key = (tuple(getattr(view, "_strategy_version", ())), current_game)
    if getattr(view, "_v854_ordered_key", None) == key:
        return tuple(getattr(view, "_v854_ordered", ()))

    nodes = tuple(getattr(view, "_node_by_uid", {}).values())
    edges = tuple(view.edge_records())
    by_uid, _parents, games = _graph_index(nodes, edges)
    norm_by_ground: dict[MemoryUid, set[MemoryUid]] = defaultdict(set)
    ground_by_norm: dict[MemoryUid, set[MemoryUid]] = defaultdict(set)
    for edge in edges:
        if int(edge.relation_type) != int(RelationType.EXPLAINS):
            continue
        src, dst = by_uid.get(edge.source_uid), by_uid.get(edge.target_uid)
        if src is None or dst is None:
            continue
        if normalized.is_normalized_contingency(src) and normalized.is_grounded_contingency(dst):
            norm_by_ground[dst.uid].add(src.uid)
            ground_by_norm[src.uid].add(dst.uid)

    strategies = {}
    for strategy in tuple(getattr(view, "_strategy_fallback", ())):
        strategies.setdefault(strategy.strategy_uid, strategy)

    result: list[OrderedTransferSequence] = []
    for strategy_uid, strategy in sorted(strategies.items()):
        node = by_uid.get(strategy_uid)
        if node is None or not blockers.is_composite_strategy(node):
            continue
        if not any(g != current_game for g in games(strategy_uid)):
            continue
        if not (
            behavior.strategy_can_control(view, strategy_uid, strategy.outcome_uid)
            or behavior._strategy_can_probe(view, strategy_uid, strategy.outcome_uid)
        ):
            continue
        blocked, relation_effect = _relation_adjustment(view, strategy_uid, current_game)
        if blocked:
            continue
        source_path = blockers._path_for_composite(view, node, int(node.key_parts[3]))
        if len(source_path) < 2:
            continue

        per_step = []
        for source_step in source_path:
            candidates = {}
            for norm_uid in norm_by_ground.get(source_step.uid, ()):
                for target_uid in ground_by_norm.get(norm_uid, ()):
                    target = by_uid.get(target_uid)
                    if (
                        target is not None
                        and target.uid != source_step.uid
                        and current_game in games(target.uid)
                        and normalized.is_grounded_contingency(target)
                        and len(target.key_parts) >= 4
                    ):
                        candidates[target.uid] = target
            if not candidates:
                per_step = []
                break
            per_step.append(
                tuple(sorted(candidates.values(), key=lambda r: (-_row_value(r), r.uid))[:32])
            )
        if not per_step:
            continue

        beam_width = min(256, max(32, 16 * len(per_step)))
        beam = [(_row_value(row), (row,)) for row in per_step[0]]
        beam.sort(key=lambda x: (-x[0], tuple(r.uid for r in x[1])))
        beam = beam[:beam_width]
        for candidates in per_step[1:]:
            extended = []
            for score, path in beam:
                after_context = int(path[-1].key_parts[3])
                used = {r.uid for r in path}
                for candidate in candidates:
                    if candidate.uid in used or int(candidate.key_parts[0]) != after_context:
                        continue
                    extended.append((score + _row_value(candidate), (*path, candidate)))
            if not extended:
                beam = []
                break
            extended.sort(key=lambda x: (-x[0], tuple(r.uid for r in x[1])))
            beam = extended[:beam_width]
        if not beam:
            continue

        base_score = (
            max(0.0, float(strategy.reliability))
            + 0.10 / max(1e-9, float(strategy.mean_cost))
            + 0.05 * math.log1p(max(0, int(strategy.support)))
            + 0.75 * relation_effect
            - (0.20 if bool(strategy.probationary) else 0.0)
        )
        for path_score, path in beam[:8]:
            result.append(
                OrderedTransferSequence(
                    base_score + path_score / len(path), strategy_uid, strategy.outcome_uid, path
                )
            )
    result.sort(key=lambda r: (-r.score, r.strategy_uid, tuple(x.uid for x in r.path)))
    view._v854_ordered_key = key
    view._v854_ordered = tuple(result)
    return tuple(result)


def _grounded_transfer_v854(view, game_id: str):
    from v8 import learning_blockers_v055 as blockers

    m7, m1n = _BASE_GROUNDED_TRANSFER(view, game_id)
    current_game = int(world_id(str(game_id)))
    nodes = getattr(view, "_node_by_uid", {})
    cleaned = {}
    for action, rows in m7.items():
        kept = []
        for score, strategy_uid, origin in rows:
            node = nodes.get(strategy_uid)
            if node is not None and blockers.is_composite_strategy(node):
                continue
            blocked, relation_effect = _relation_adjustment(view, strategy_uid, current_game)
            if blocked:
                continue
            # Remove v8.37's source-global transfer prior and use only target-specific trust.
            legacy = 0.0 if node is None else max(0.0, float(getattr(node, "transfer_prior", 0.0)))
            kept.append((float(score) - 0.50 * legacy + 0.75 * relation_effect, strategy_uid, origin))
        if kept:
            cleaned[int(action)] = tuple(sorted(kept, key=lambda x: (-x[0], x[1])))
    return cleaned, m1n


def _clear_active(view, game_id: str | None = None) -> None:
    if view is None:
        return
    active = getattr(view, "_v854_transfer_active", {})
    if game_id is None:
        active.clear()
    else:
        active.pop(str(game_id), None)
    view._v854_transfer_active = active


def _ordered_action(view, game_id: str, context: int, available: set[int]):
    active = getattr(view, "_v854_transfer_active", {})
    state = active.get(str(game_id))
    if state is not None:
        sequence, index = state
        if index < len(sequence.path):
            row = sequence.path[index]
            action = int(signed_u64(int(row.key_parts[1])))
            if int(row.key_parts[0]) == int(context) and action in available:
                active[str(game_id)] = (sequence, index + 1)
                view._v854_transfer_active = active
                return action, "M7_SEQUENCE_CORRESPONDENCE", sequence.strategy_uid
        active.pop(str(game_id), None)

    choices = []
    for sequence in _ordered_sequences(view, game_id):
        first = sequence.path[0]
        action = int(signed_u64(int(first.key_parts[1])))
        if int(first.key_parts[0]) == int(context) and action in available:
            choices.append((sequence.score, action, sequence))
    if not choices:
        view._v854_transfer_active = active
        return None
    _score, action, sequence = min(choices, key=lambda x: (-x[0], x[1], x[2].strategy_uid))
    active[str(game_id)] = (sequence, 1)
    view._v854_transfer_active = active
    return action, "M7_SEQUENCE_CORRESPONDENCE", sequence.strategy_uid


def _cross_game_v854(sampler, actions: tuple[int, ...]):
    from v8 import behavior_recovery as behavior

    view = getattr(behavior, "_CURRENT_ACTOR_VIEW", None)
    available = {int(x) for x in actions}
    if view is None or not available:
        return None
    context = getattr(sampler, "_v854_transfer_context", None)
    if context is not None:
        ordered = _ordered_action(view, str(sampler.game_id), int(context), available)
        if ordered is not None:
            return ordered
    return _BASE_CROSS_GAME(sampler, tuple(sorted(available)))


def _plan_chain_v854(self, context_signature, action_ids, **kwargs):
    import os
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import environment_neutrality_v837 as v837
    from v8.publication import PlannedAction

    mode = str(os.environ.get(v819._SAMPLING_MODE_ENV, v819.SamplingMode.DISCOVERY.value))
    if mode != v819.SamplingMode.TRANSFER.value:
        return _BASE_PLAN_CHAIN(self, context_signature, action_ids, **kwargs)
    game = v837._current_game_id()
    if not game:
        return ()
    ordered = _ordered_action(self, game, int(context_signature), {int(x) for x in action_ids})
    if ordered is not None:
        action, _origin, strategy_uid = ordered
        strategy = next((x for x in getattr(self, "_strategy_fallback", ()) if x.strategy_uid == strategy_uid), None)
        if strategy is not None:
            state = getattr(self, "_v854_transfer_active", {}).get(str(game))
            score = 0.0 if state is None else float(state[0].score)
            return (PlannedAction(action, strategy.outcome_uid, strategy_uid, score, False),)
    return _BASE_PLAN_CHAIN(self, context_signature, action_ids, **kwargs)


def _clear_rollouts_v854(sampler) -> None:
    _BASE_CLEAR_ROLLOUTS(sampler)
    try:
        from v8 import behavior_recovery as behavior
        _clear_active(getattr(behavior, "_CURRENT_ACTOR_VIEW", None), getattr(sampler, "game_id", None))
    except BaseException:
        pass


def _observe_transfer_v854(self, intervention, **kwargs) -> None:
    _BASE_OBSERVE_TRANSFER(self, intervention, **kwargs)
    if bool(getattr(self, "_v833_transfer_rollout", False)):
        return
    try:
        from v8 import behavior_recovery as behavior
        _clear_active(getattr(behavior, "_CURRENT_ACTOR_VIEW", None), getattr(self, "game_id", None))
    except BaseException:
        pass


def _forced_v854(self, *, level, context, actions, history):
    self._v854_transfer_context = int(context)
    return _BASE_FORCED(self, level=level, context=context, actions=actions, history=history)


def _discovery_v854(self, *, level, context, actions, history):
    self._v854_transfer_context = int(context)
    return _BASE_DISCOVERY(self, level=level, context=context, actions=actions, history=history)


# ---------------------------------------------------------------------------
# Safe normalized restart priors. A normalized fact may transfer only through a
# grounded M1 action that was actually observed in the current target world.
# ---------------------------------------------------------------------------


def _build_restart_v854(view) -> None:
    from v8 import normalized_memory_v086 as normalized
    from v8 import restart_memory_v815 as restart

    _BASE_BUILD_RESTART(view)
    if not bool(getattr(view, "_v854_runtime_view", False)):
        return
    cache_key = getattr(view, "_v815_restart_index_key", None)
    if getattr(view, "_v854_restart_key", None) == cache_key:
        return
    current_game = int(restart._current_game_hash())
    if not current_game:
        view._v815_normalized_action_priors = {}
        view._v854_restart_key = cache_key
        return

    nodes = tuple(getattr(view, "_node_by_uid", {}).values())
    edges = tuple(view.edge_records())
    by_uid, _parents, games = _graph_index(nodes, edges)
    grounded_by_norm: dict[MemoryUid, list[object]] = defaultdict(list)
    for edge in edges:
        if int(edge.relation_type) != int(RelationType.EXPLAINS):
            continue
        src, dst = by_uid.get(edge.source_uid), by_uid.get(edge.target_uid)
        if src is not None and dst is not None and normalized.is_normalized_contingency(src) and normalized.is_grounded_contingency(dst):
            grounded_by_norm[src.uid].append(dst)

    raw: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for norm_uid, grounded in grounded_by_norm.items():
        if not any(any(g != current_game for g in games(row.uid)) for row in grounded):
            continue
        source = by_uid.get(norm_uid)
        for row in grounded:
            if current_game not in games(row.uid) or len(row.key_parts) < 2:
                continue
            value = max(restart._positive_memory_value(row), restart._positive_memory_value(source) if source is not None else 0.0)
            if value <= 0.0:
                continue
            support = max(1, int(row.support_count)) + (0 if source is None else max(1, int(source.support_count)))
            action = int(signed_u64(int(row.key_parts[1])))
            bucket = raw[action]
            bucket[0] += support
            bucket[1] += value * support
            bucket[2] += support
    view._v815_normalized_action_priors = restart._finalize_prior(raw, discount=0.40)
    view._v854_restart_key = cache_key


# ---------------------------------------------------------------------------
# Exact provenance and structurally ranked similarity candidates. Candidate budget
# is spent after a cheap structural rank, never by UID order.
# ---------------------------------------------------------------------------


def _cheap_rank(source, candidate) -> float:
    score = 0.0
    score += 2.0 if source.incoming_relations == candidate.incoming_relations else 0.0
    score += 2.0 if source.outgoing_relations == candidate.outgoing_relations else 0.0
    score += 1.0 if source.neighbor_levels == candidate.neighbor_levels else 0.0
    score += 1.0 if source.neighbor_types == candidate.neighbor_types else 0.0
    score += 1.0 if source.dependency_signature and source.dependency_signature == candidate.dependency_signature else 0.0
    score += 0.5 if source.enable_block_signature and source.enable_block_signature == candidate.enable_block_signature else 0.0
    score += 1.0 / (1.0 + abs(source.future_option_bucket - candidate.future_option_bucket))
    return score


def _similarity_v854(self, nodes, edges):
    nodes, edges = tuple(nodes), tuple(edges)
    descriptors = self.descriptors(nodes, edges)
    by_uid, _parents, games = _graph_index(nodes, edges)
    index, fallback = defaultdict(list), defaultdict(list)
    for d in descriptors.values():
        index[(d.level, d.memory_type, self._relation_bucket(d), d.future_option_bucket)].append(d)
        fallback[(d.level, d.memory_type)].append(d)
    dirty = [d for d in descriptors.values() if d.descriptor_version > self._processed_versions.get(d.uid, -1)]
    dirty.sort(key=lambda d: (d.level, d.memory_type, d.descriptor_version, d.uid))
    results = {}

    def is_cross(left_uid, right_uid) -> bool:
        left, right = games(left_uid), games(right_uid)
        if left and right:
            return left != right
        # Compatibility fallback only when exact provenance is genuinely absent.
        lrow, rrow = by_uid.get(left_uid), by_uid.get(right_uid)
        lm = 0 if lrow is None else int(lrow.game_mask)
        rm = 0 if rrow is None else int(rrow.game_mask)
        return bool(lm and rm and lm != rm)

    for source in dirty:
        candidates = []
        relation_bucket = self._relation_bucket(source)
        for delta in (0, -1, 1):
            candidates.extend(index.get((source.level, source.memory_type, relation_bucket, source.future_option_bucket + delta), ()))
        if len(candidates) < self.max_candidates:
            candidates.extend(fallback.get((source.level, source.memory_type), ()))
        unique = {x.uid: x for x in candidates if x.uid != source.uid}
        ranked_candidates = sorted(
            unique.values(),
            key=lambda x: (
                0 if is_cross(source.uid, x.uid) else 1,
                -_cheap_rank(source, x),
                stable_u64(int(source.uid.hi), int(source.uid.lo), int(x.uid.hi), int(x.uid.lo), person=b"v854-sim-pair"),
            ),
        )[: self.max_candidates]
        scored = []
        for candidate in ranked_candidates:
            self.candidate_comparisons += 1
            evidence = self.score(source, candidate)
            if evidence.score >= self.threshold:
                scored.append(evidence)
        scored.sort(key=lambda x: (-x.score, x.source_uid, x.target_uid))
        cross = [x for x in scored if is_cross(x.source_uid, x.target_uid)]
        reserve = max(1, self.top_results // 2)
        chosen = cross[:reserve]
        seen = {(x.source_uid, x.target_uid) for x in chosen}
        for evidence in scored:
            pair = (evidence.source_uid, evidence.target_uid)
            if pair in seen:
                continue
            chosen.append(evidence)
            seen.add(pair)
            if len(chosen) >= self.top_results:
                break
        for evidence in chosen[: self.top_results]:
            results[(evidence.source_uid, evidence.target_uid)] = evidence
        self._processed_versions[source.uid] = source.descriptor_version
        self.processed_descriptors += 1
    return tuple(results[k] for k in sorted(results))


# ---------------------------------------------------------------------------
# Adaptive composite M7 formation: longer procedures and wider evidence beams.
# ---------------------------------------------------------------------------


def _adaptive_composites(engine, nodes, edges, *, limit: int):
    if limit <= 0:
        return ()
    from v8 import behavior_recovery as behavior
    from v8 import learning_blockers_v055 as blockers
    from v8.promotion import FormationCandidate

    cancel_event = getattr(engine, "_v841_cancel_event", None)

    def cancelled() -> bool:
        return bool(cancel_event is not None and cancel_event.is_set())

    if cancelled():
        return ()

    m1 = []
    outcomes = []
    for index, row in enumerate(nodes):
        if index % 4096 == 0 and cancelled():
            return ()
        if (
            int(row.level) == int(MemoryLevel.M1)
            and int(row.memory_type) == int(MemoryType.CONTINGENCY)
            and len(row.key_parts) >= 4
            and row.support_count >= int(getattr(engine, "min_contingency_support", 3))
            and engine._admissible(row)
        ):
            m1.append(row)
        elif (
            int(row.level) == int(MemoryLevel.M6)
            and int(row.memory_type) == int(MemoryType.OUTCOME)
            and len(row.key_parts) >= 3
            and engine._admissible(row)
        ):
            outcomes.append(row)
    incoming = defaultdict(list)
    for row in m1:
        incoming[int(row.key_parts[3])].append(row)
    for rows in incoming.values():
        rows.sort(key=lambda r: (-_row_value(r), r.uid))
    by_uid = {row.uid: row for row in m1}
    parents = behavior._parent_map(tuple(edges), cancel_event=cancel_event)
    if parents is None or cancelled():
        return ()
    graph_scale = max(1, len(m1))
    depth_limit = min(_MAX_COMPOSITE_ACTIONS, max(6, 4 + int(math.ceil(math.log2(graph_scale + 1)))))
    predecessor_cap = min(16, max(4, int(math.sqrt(max(1, limit))) + 2))
    beam_width = min(256, max(32, int(limit) * 4))
    result, seen = [], set()

    for outcome in sorted(outcomes, key=lambda r: r.uid):
        if cancelled():
            return ()
        ancestors = behavior.causal_m1_ancestors(
            outcome.uid,
            nodes=(),
            edges=(),
            max_depth=max(8, depth_limit + 2),
            cancel_event=cancel_event,
            node_by_uid=by_uid,
            parents=parents,
        )
        if cancelled():
            return ()
        for target_uid in sorted(ancestors):
            if cancelled():
                return ()
            target = by_uid.get(target_uid)
            if target is None:
                continue
            paths = [[target]]
            for _ in range(depth_limit - 1):
                if cancelled():
                    return ()
                extended = []
                for path in paths:
                    if cancelled():
                        return ()
                    before = int(path[0].key_parts[0])
                    used = {r.uid for r in path}
                    for predecessor in incoming.get(before, ())[:predecessor_cap]:
                        if predecessor.uid in used:
                            continue
                        candidate_path = [predecessor, *path]
                        extended.append(candidate_path)
                        if len(candidate_path) < 2:
                            continue
                        sequence_id = blockers._sequence_hash(candidate_path)
                        key = (int(sequence_id), int(outcome.uid.hi), int(outcome.uid.lo), int(candidate_path[0].key_parts[0]))
                        uid = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, key)
                        if uid in seen:
                            continue
                        seen.add(uid)
                        support = min([int(outcome.support_count)] + [int(r.support_count) for r in candidate_path])
                        result.append(
                            FormationCandidate(
                                uid=uid,
                                level=MemoryLevel.M7,
                                memory_type=MemoryType.STRATEGY,
                                key_parts=key,
                                parents=(outcome.uid,) + tuple(r.uid for r in candidate_path),
                                support=max(1, support),
                                significance=min(1.0, sum(float(r.significance) for r in candidate_path) / len(candidate_path)),
                                learning_value=min(1.0, sum(float(r.learning_value) for r in candidate_path) / len(candidate_path)),
                                transfer_prior=0.0,
                                explanatory_reach=1.0,
                                future_option_delta=float(candidate_path[-1].future_option_delta),
                                cognitive_state=int(CognitiveState.PROBATION),
                                validation_state=int(ValidationState.STRUCTURAL),
                                evidence_kind="multi_action_strategy",
                                evidence_value=min(1.0, support / 4.0),
                            )
                        )
                        if len(result) >= limit:
                            return tuple(result)
                if not extended:
                    break
                extended.sort(key=lambda path: (-sum(_row_value(r) for r in path), tuple(r.uid for r in path)))
                paths = extended[:beam_width]
    return tuple(result)


# ---------------------------------------------------------------------------
# Causal session credit: only productive, context-contiguous transitions receive
# positive credit. No-ops and disconnected exploratory branches are excluded.
# ---------------------------------------------------------------------------


def _base_step_v854(env, action):
    try:
        before = int(env.cognitive_context_signature())
    except BaseException:
        before = 0
    result = _BASE_RESTART_STEP(env, action)
    try:
        after = int(env.cognitive_context_signature())
    except BaseException:
        after = before
    boundary = bool(getattr(env, "level_completed_event", False)) or str(getattr(env, "last_outcome_state", "")) in {"WIN", "GAME_OVER"}
    productive = bool(before != after or boundary)
    try:
        from v8 import behavior_recovery as behavior
        view = getattr(behavior, "_CURRENT_ACTOR_VIEW", None)
    except BaseException:
        view = None
    if view is not None:
        rows = list(getattr(view, "_v854_session_transitions", ()))
        rows.append((before, int(action), after, productive))
        view._v854_session_transitions = rows[-2048:]
    return result


def _credit_v854(view, *, success: bool, failure: bool) -> None:
    from v8 import restart_memory_v815 as restart

    if not hasattr(view, "_v854_session_transitions"):
        return _BASE_CREDIT(view, success=success, failure=failure)
    transitions = list(getattr(view, "_v854_session_transitions", ()))
    if success and transitions:
        selected, needed = [], None
        for before, action, after, productive in reversed(transitions[-restart._SESSION_HORIZON:]):
            if not productive:
                continue
            if needed is not None and int(after) != int(needed):
                continue
            selected.append((int(before), int(action), int(after)))
            needed = int(before)
        priors = getattr(view, "_v815_session_action_priors", {})
        for distance, (_before, action, _after) in enumerate(selected):
            credit = restart._SESSION_GAMMA ** distance
            bucket = priors.setdefault(action, [0.0, 0.0, 0.0])
            bucket[0] += 1.0
            bucket[1] += float(credit)
            bucket[2] += 1.0
        view._v815_session_action_priors = priors
    if success or failure:
        view._v854_session_transitions = []
        view._v815_session_trajectory = []


def _reset_v854(self, *args, **kwargs):
    try:
        from v8 import behavior_recovery as behavior
        view = getattr(behavior, "_CURRENT_ACTOR_VIEW", None)
    except BaseException:
        view = None
    if view is not None:
        view._v854_session_transitions = []
        view._v815_session_trajectory = []
        _clear_active(view)
    return _BASE_RESET(self, *args, **kwargs)


def _view_init_v854(self, *args, **kwargs):
    _BASE_VIEW_INIT(self, *args, **kwargs)
    self._v854_runtime_view = True
    self._v854_session_transitions = []
    self._v854_transfer_active = {}
    self._v854_restart_key = None
    self._v854_ordered_key = None
    self._v854_ordered = ()


def install_learning_transfer_correctness_v854() -> None:
    global _INSTALLED, _BASE_BUILD_RESTART, _BASE_GROUNDED_TRANSFER, _BASE_CROSS_GAME
    global _BASE_PLAN_CHAIN, _BASE_CLEAR_ROLLOUTS, _BASE_OBSERVE_TRANSFER, _BASE_FORCED
    global _BASE_DISCOVERY, _BASE_RESTART_STEP, _BASE_RESET, _BASE_CREDIT, _BASE_VIEW_INIT
    global _BASE_RECORD_TRIAL
    if _INSTALLED:
        return

    from v7.environment.arc_adapter import ArcGridEnvironment
    from v8 import environment_neutrality_v837 as v837
    from v8 import learning_blockers_v055 as blockers
    from v8 import restart_memory_v815 as restart
    from v8 import sampling_portfolio_v831 as portfolio
    from v8 import sampling_transfer_v833 as transfer_sampling
    from v8.peers_v82 import V82DevelopmentalPeerSupervisor
    from v8.publication import LiveReadView
    from v8.similarity import BoundedNeighborhoodSimilarity

    _BASE_RECORD_TRIAL = V82DevelopmentalPeerSupervisor.record_transfer_trial
    V82DevelopmentalPeerSupervisor.record_transfer_trial = _record_trial_v854

    _BASE_VIEW_INIT = LiveReadView.__init__
    LiveReadView.__init__ = _view_init_v854

    _BASE_BUILD_RESTART = restart._build_restart_indexes
    restart._build_restart_indexes = _build_restart_v854

    _BASE_GROUNDED_TRANSFER = v837._grounded_transfer_index
    v837._grounded_transfer_index = _grounded_transfer_v854
    v837._grounded_m7_index_v837 = lambda view, game_id: _grounded_transfer_v854(view, game_id)[0]

    _BASE_CROSS_GAME = transfer_sampling._cross_game_transfer_action
    transfer_sampling._cross_game_transfer_action = _cross_game_v854
    _BASE_PLAN_CHAIN = portfolio._BASE_PLAN_CHAIN
    portfolio._BASE_PLAN_CHAIN = _plan_chain_v854

    _BASE_CLEAR_ROLLOUTS = transfer_sampling._clear_rollouts_v833
    transfer_sampling._clear_rollouts_v833 = _clear_rollouts_v854
    _BASE_OBSERVE_TRANSFER = transfer_sampling._observe_transfer_v833
    transfer_sampling._observe_transfer_v833 = _observe_transfer_v854
    _BASE_FORCED = transfer_sampling._forced_action_v833
    transfer_sampling._forced_action_v833 = _forced_v854
    _BASE_DISCOVERY = transfer_sampling._discovery_action_v833
    transfer_sampling._discovery_action_v833 = _discovery_v854

    BoundedNeighborhoodSimilarity.evaluate = _similarity_v854

    blockers._MAX_SEQUENCE_ACTIONS = _MAX_COMPOSITE_ACTIONS
    blockers._composite_candidates = _adaptive_composites

    _BASE_RESTART_STEP = restart._BASE_ENV_STEP
    restart._BASE_ENV_STEP = _base_step_v854
    _BASE_CREDIT = restart._credit_session
    restart._credit_session = _credit_v854
    _BASE_RESET = ArcGridEnvironment.reset
    ArcGridEnvironment.reset = _reset_v854

    _INSTALLED = True
