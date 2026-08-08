from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import replace
from hashlib import sha1
from queue import Full
from typing import Any


_SAMPLING_PATCHED = False
_VALIDATION_PATCHED = False
_ORIGINAL_CHOOSE_WITH_SAMPLER_PRIOR: Any = None
_ORIGINAL_MEMORY_GUIDED_CHOICE: Any = None
_ORIGINAL_LINKS_BY_SIGNATURE: Any = None
_ORIGINAL_PREDICTION_RESULT_COLUMNS: Any = None
_ORIGINAL_TRANSFER_RATE_BEFORE: Any = None
_ORIGINAL_TRANSFER_MAX_STEP_BEFORE: Any = None
_ORIGINAL_TRANSFER_MAX_ANY_STEP_BEFORE: Any = None

_IDEMPOTENT_LIVE_EVENT_TYPES = {
    "stable_contingency",
    "family_update",
    "carrier_candidate",
    "contradiction_cluster",
}


def install_v63_sampling_performance() -> None:
    """Install semantics-preserving worker hot-path optimizations."""
    global _SAMPLING_PATCHED
    global _ORIGINAL_CHOOSE_WITH_SAMPLER_PRIOR
    global _ORIGINAL_MEMORY_GUIDED_CHOICE
    if _SAMPLING_PATCHED:
        _patch_v6_system_if_loaded()
        return

    from v6.memory.v621_runtime import (
        V621MemoryController,
        V621SnapshotMemoryQueryEngine,
    )
    from v6.sampling import BaseSampler

    _ORIGINAL_CHOOSE_WITH_SAMPLER_PRIOR = (
        V621MemoryController.choose_with_sampler_prior
    )
    _ORIGINAL_MEMORY_GUIDED_CHOICE = BaseSampler._memory_guided_choice

    V621MemoryController.choose_with_sampler_prior = (
        _choose_with_sampler_prior_cached
    )
    V621MemoryController.record_prediction_outcome = (
        _record_prediction_outcome_v63_buffered
    )
    BaseSampler._memory_guided_choice = _memory_guided_choice_cached
    V621SnapshotMemoryQueryEngine.score_action = (
        _snapshot_score_action_cached
    )

    _SAMPLING_PATCHED = True
    _patch_v6_system_if_loaded()


def install_v63_validation_performance() -> None:
    """Cache immutable evidence lookups reused across H07 candidates."""
    global _VALIDATION_PATCHED
    global _ORIGINAL_LINKS_BY_SIGNATURE
    global _ORIGINAL_PREDICTION_RESULT_COLUMNS
    global _ORIGINAL_TRANSFER_RATE_BEFORE
    global _ORIGINAL_TRANSFER_MAX_STEP_BEFORE
    global _ORIGINAL_TRANSFER_MAX_ANY_STEP_BEFORE
    if _VALIDATION_PATCHED:
        return

    from v6 import higher_order_substrate as substrate

    _ORIGINAL_LINKS_BY_SIGNATURE = substrate._links_by_signature
    _ORIGINAL_PREDICTION_RESULT_COLUMNS = substrate._prediction_result_columns
    _ORIGINAL_TRANSFER_RATE_BEFORE = substrate._TransferHistoryIndex.rate_before
    _ORIGINAL_TRANSFER_MAX_STEP_BEFORE = (
        substrate._TransferHistoryIndex.max_step_before
    )
    _ORIGINAL_TRANSFER_MAX_ANY_STEP_BEFORE = (
        substrate._TransferHistoryIndex.max_any_step_before
    )

    substrate._links_by_signature = _links_by_signature_cached
    substrate._prediction_result_columns = _prediction_result_columns_cached
    substrate._TransferHistoryIndex.rate_before = _transfer_rate_before_cached
    substrate._TransferHistoryIndex.max_step_before = (
        _transfer_max_step_before_cached
    )
    substrate._TransferHistoryIndex.max_any_step_before = (
        _transfer_max_any_step_before_cached
    )
    _VALIDATION_PATCHED = True


def _patch_v6_system_if_loaded() -> None:
    module = sys.modules.get("v6.main")
    system_type = None if module is None else getattr(module, "V6System", None)
    if system_type is None:
        return
    system_type._emit_live_memory_event = _emit_live_memory_event_deduplicated
    system_type._apply_trajectory_efficiency_bonus = (
        _apply_trajectory_efficiency_bonus_v63_buffered
    )


def _emit_live_memory_event_deduplicated(
    self: Any,
    event_type: str,
    event_id: str,
    global_step: int,
    priority: float,
    payload: dict,
) -> None:
    """Avoid redundant Manager RPCs for unchanged replace-style projections."""
    if self.live_memory_queue is None:
        return
    if str(self.config.shared_live_memory_mode) not in {"write", "readwrite"}:
        return

    event_type = str(event_type)
    event_id = str(event_id)
    bounded_priority = max(0.0, min(1.0, float(priority)))
    cooked_payload = dict(payload or {})

    if event_type in _IDEMPOTENT_LIVE_EVENT_TYPES:
        fingerprints = getattr(self, "_v63_live_event_fingerprints", None)
        if fingerprints is None:
            fingerprints = {}
            self._v63_live_event_fingerprints = fingerprints
        fingerprint = (
            bounded_priority,
            json.dumps(
                cooked_payload,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
        )
        cache_key = (event_type, event_id)
        if fingerprints.get(cache_key) == fingerprint:
            self._v63_live_events_deduplicated = int(
                getattr(self, "_v63_live_events_deduplicated", 0)
            ) + 1
            return
        fingerprints[cache_key] = fingerprint

    try:
        from v6.memory.live_memory_queue import LiveMemoryEvent

        queue_started = time.perf_counter()
        self.live_memory_queue.put_nowait(
            LiveMemoryEvent(
                event_type=event_type,
                event_id=event_id,
                global_step=int(global_step),
                worker_id=str(
                    self.config.live_memory_worker_id or "unknown_worker"
                ),
                priority=bounded_priority,
                payload=cooked_payload,
            )
        )
        self.live_memory_queue_block_seconds += (
            time.perf_counter() - queue_started
        )
        self.live_memory_events_emitted += 1

        # Manager.Queue.qsize() is another synchronous proxy RPC.  It is only
        # diagnostic, so sample it instead of doing it for every event.
        if int(self.live_memory_events_emitted) % 256 == 0:
            try:
                self.live_memory_queue_peak_size = max(
                    self.live_memory_queue_peak_size,
                    int(self.live_memory_queue.qsize()),
                )
            except (AttributeError, NotImplementedError, OSError):
                pass
    except Full:
        self.live_memory_events_dropped_queue_full += 1
    except Exception:
        self.live_memory_events_dropped_error += 1


def _memory_guided_choice_cached(
    self: Any,
    system: Any,
    actions: list[int],
) -> tuple[int, object | None, bool]:
    """Reuse the sampler's ranking in the subsequent memory override gate."""
    memory_query = getattr(system, "memory_query", None)
    context_builder = getattr(system, "context_builder", None)
    depth_fn = getattr(system, "_context_depth_for_action", None)
    controller = getattr(system, "memory_controller", None)
    if controller is not None:
        controller._v63_precomputed_sampler_ranking = None

    if memory_query is None or context_builder is None or not callable(depth_fn):
        fallback = self.softmax_sample(
            actions,
            self.mixed_explorer_scores(system, actions),
        )
        self._record_memory_guided_selection(None, [])
        return int(fallback), None, True

    contexts_by_action: dict[int, dict[int, tuple]] = {}
    for action in actions:
        max_level = int(depth_fn(int(action)))
        contexts_by_action[int(action)] = context_builder.multi_scale_signatures(
            int(action),
            max_level=max_level,
        )

    ranked_actions = []
    if hasattr(memory_query, "rank_actions"):
        ranked_actions = list(
            memory_query.rank_actions(contexts_by_action, actions) or []
        )
    if controller is not None:
        controller._v63_precomputed_sampler_ranking = list(ranked_actions)

    if not ranked_actions or all(
        float(getattr(item, "score", 0.0) or 0.0) <= 0.0
        for item in ranked_actions
    ):
        fallback = self.softmax_sample(
            actions,
            self.mixed_explorer_scores(system, actions),
        )
        self._record_memory_guided_selection(None, ranked_actions)
        if hasattr(memory_query, "record_selected_action_query"):
            try:
                memory_query.record_selected_action_query(
                    context_signatures=contexts_by_action[int(fallback)],
                    action=int(fallback),
                )
            except Exception:
                pass
        return int(fallback), None, True

    ranked_by_action = {
        int(item.action): item for item in ranked_actions
    }
    chosen = self.softmax_sample(
        [int(item.action) for item in ranked_actions],
        [float(item.score) for item in ranked_actions],
    )
    selected_score = ranked_by_action.get(int(chosen))
    self._record_memory_guided_selection(selected_score, ranked_actions)
    if hasattr(memory_query, "record_selected_action_query"):
        try:
            memory_query.record_selected_action_query(
                context_signatures=contexts_by_action[int(chosen)],
                action=int(chosen),
            )
        except Exception:
            pass
    return int(chosen), selected_score, False


def _choose_with_sampler_prior_cached(
    self: Any,
    *,
    context_signatures_by_action: dict[int, dict[int, tuple]],
    available_actions: list[int],
    sampler_action: int,
    override_margin: float = 0.15,
) -> int:
    precomputed = getattr(
        self,
        "_v63_precomputed_sampler_ranking",
        None,
    )
    self._v63_precomputed_sampler_ranking = None
    if precomputed is None:
        ranked = self.choose_action_candidates(
            context_signatures_by_action,
            available_actions,
        )
    else:
        ranked = list(precomputed)
        self._last_ranked_scores = {
            int(item.action): item for item in ranked
        }

    if not ranked:
        return int(sampler_action)
    top = ranked[0]
    baseline = self._last_ranked_scores.get(int(sampler_action))
    baseline_score = float(baseline.score) if baseline is not None else 0.0
    if (
        int(top.action) != int(sampler_action)
        and float(top.score)
        >= baseline_score + max(0.0, float(override_margin))
    ):
        self._audit(
            "sampler_memory_override",
            owner_id=str(top.action),
            payload={
                "sampler_action": int(sampler_action),
                "memory_action": int(top.action),
                "memory_score": float(top.score),
                "sampler_action_memory_score": baseline_score,
                "override_margin": float(override_margin),
            },
        )
        return int(top.action)
    return int(sampler_action)


def _snapshot_score_action_cached(
    self: Any,
    context_signatures: dict[int, tuple],
    action: int,
    available_actions: list[int],
    *,
    record_query: bool = False,
) -> Any:
    """Compute role/concept matches once per action instead of 2-3 times."""
    del record_query
    from v6.memory.query_engine import MemoryPrediction, compute_memory_action_score

    best_context = self._best_context_signature(context_signatures, action)
    roles = self.find_similar_roles(best_context, action)
    concepts = self.find_concept_matches(
        best_context,
        action,
        role_matches=roles,
    )
    cache = getattr(self, "_v63_last_concept_matches_by_action", None)
    if cache is None:
        cache = {}
        self._v63_last_concept_matches_by_action = cache
    cache[int(action)] = list(concepts)

    exact = None
    exact_method = getattr(self.base, "_exact_contingency", None)
    if exact_method is not None:
        exact = exact_method(context_signatures, action)
    stable = None
    if exact is None:
        contingency_method = getattr(self.base, "_contingency", None)
        if contingency_method is not None:
            stable = contingency_method(context_signatures, action)

    if exact is not None:
        prediction = MemoryPrediction(
            predicted_family=int(
                exact.get("family", exact.get("transformation_family", 0))
            ),
            confidence=float(exact.get("confidence", 0.0) or 0.0),
            source="memory_contingency",
            evidence_node_ids=[str(exact.get("node_id", ""))],
        )
    elif stable is not None:
        prediction = MemoryPrediction(
            predicted_family=int(
                stable.get(
                    "family",
                    stable.get("transformation_family", 0),
                )
            ),
            confidence=float(stable.get("confidence", 0.0) or 0.0),
            source="contingency_learner",
            evidence_node_ids=[],
        )
    elif roles and roles[0].get("family_id") is not None:
        prediction = MemoryPrediction(
            predicted_family=int(roles[0]["family_id"]),
            confidence=float(roles[0].get("score", 0.0) or 0.0),
            source="role_match",
            evidence_node_ids=[str(roles[0]["node_id"])],
        )
    elif concepts and concepts[0].get("family_id") is not None:
        prediction = MemoryPrediction(
            predicted_family=int(concepts[0]["family_id"]),
            confidence=float(concepts[0].get("score", 0.0) or 0.0),
            source="concept_match",
            evidence_node_ids=[str(concepts[0]["node_id"])],
        )
    else:
        prediction = MemoryPrediction(None, 0.0, "none", [])

    future_failure = getattr(self.base, "_future_and_failure", None)
    if future_failure is not None:
        context_tuple = (
            tuple(json.loads(best_context))
            if str(best_context).startswith("[")
            else (best_context,)
        )
        future, failure = future_failure(context_tuple, action)
    else:
        future = {
            "expected_future_option_delta": 0.0,
            "completion_likelihood": 0.0,
            "sources": [],
        }
        failure = {
            "failure_risk": 0.0,
            "contradiction_evidence": False,
            "sources": [],
        }

    base_score = compute_memory_action_score(
        action=action,
        prediction=prediction,
        future_option_evidence=future,
        failure_evidence=failure,
        role_matches=roles,
        concept_matches=concepts,
    )
    strategy_score = self._strategy_score(best_context, action)
    world_score = _snapshot_world_model_score_from_roles(
        self,
        best_context,
        roles,
    )
    score = _clamp01(
        float(base_score.score)
        + 0.10 * strategy_score
        + 0.07 * world_score
    )
    sources = list(base_score.evidence_sources)
    if strategy_score > 0:
        sources.append("M6_strategy_memory_v621")
    if world_score > 0:
        sources.append("M5_world_model_memory_v621")
    return replace(base_score, score=score, evidence_sources=sources)


def _snapshot_world_model_score_from_roles(
    engine: Any,
    context_signature: str,
    role_matches: list[dict],
) -> float:
    role_ids = {str(item["node_id"]) for item in role_matches}
    concept_ids: set[str] = set()
    concept_map = getattr(engine.snapshot, "concept_ids_by_role", {})
    for role_id in role_ids:
        concept_ids.update(
            str(value)
            for value in concept_map.get(role_id, ())
            if engine._usable(str(value))
        )
    if not concept_ids:
        return 0.0

    best = 0.0
    for model_id, attrs in engine._world_models.items():
        if not engine._usable(model_id):
            continue
        overlap = concept_ids & {
            str(value) for value in attrs.get("concept_ids", []) or []
        }
        if not overlap:
            continue
        relations = [
            row
            for row in engine._relations_by_model.get(model_id, ())
            if row["relation_type"]
            in {"precedes", "enables", "constrains", "shared_outcome"}
        ]
        if not relations:
            continue
        strength = sum(
            float(row["confidence"])
            * min(1.0, float(row["support_count"]) / 3.0)
            for row in relations
        ) / len(relations)
        contexts = {
            str(value)
            for value in attrs.get("supported_contexts", []) or []
        }
        context_factor = (
            1.0
            if not contexts or context_signature in contexts
            else 0.5
        )
        promotion = engine._status_by_id.get(
            model_id,
            ("candidate", "active"),
        )[0]
        status_factor = 1.0 if promotion == "accepted" else 0.6
        best = max(
            best,
            context_factor * status_factor * _clamp01(strength),
        )
    return best


def _record_prediction_outcome_v63_buffered(
    self: Any,
    *,
    prediction: Any | None,
    success: bool | None,
    game: str | None,
    context_key: str,
    context_signatures: dict[int, tuple] | None = None,
    action: int,
    actual_family: str | int | None,
    global_step: int | None,
) -> None:
    """Record v6.3 transfer evidence without a per-step commit or rescan."""
    if actual_family is None:
        return

    attempts: dict[str, tuple[str | int | None, bool, str]] = {}
    if (
        prediction is not None
        and success is not None
        and prediction.source == "concept_match"
    ):
        for concept_id in prediction.evidence_node_ids:
            if str(concept_id).startswith("M4:"):
                attempts[str(concept_id)] = (
                    prediction.predicted_family,
                    bool(success),
                    "runtime_selected_concept_prediction",
                )

    matches = None
    cached = getattr(
        self.query_engine,
        "_v63_last_concept_matches_by_action",
        None,
    )
    if isinstance(cached, dict):
        matches = cached.get(int(action))

    if matches is None and context_signatures:
        best_context = json.dumps(
            list(
                context_signatures[
                    max(int(level) for level in context_signatures)
                ]
            )
        )
        finder = getattr(self.query_engine, "find_concept_matches", None)
        if finder is not None:
            try:
                matches = finder(best_context, int(action))
            except TypeError:
                matches = []
    for match in list(matches or []):
        concept_id = str(match.get("node_id") or "")
        family_id = match.get("family_id")
        if not concept_id.startswith("M4:") or family_id is None:
            continue
        attempts.setdefault(
            concept_id,
            (
                family_id,
                str(family_id) == str(actual_family),
                "runtime_counterfactual_concept_test",
            ),
        )

    if not attempts:
        return

    source_game_cache = getattr(
        self,
        "_v63_concept_source_games_cache",
        None,
    )
    if source_game_cache is None:
        source_game_cache = {}
        self._v63_concept_source_games_cache = source_game_cache
    from v6.memory.v63_transfer import (
        QUALIFIED_EVIDENCE_PREFIX,
        UNQUALIFIED_EVIDENCE_PREFIX,
        _concept_source_games,
    )

    target_game = None if game in (None, "") else str(game)
    for concept_id, (
        predicted_family,
        attempt_success,
        source,
    ) in attempts.items():
        source_games = source_game_cache.get(concept_id)
        if source_games is None:
            source_games = _concept_source_games(self.memory, concept_id)
            source_game_cache[concept_id] = set(source_games)
        qualifies = bool(
            target_game
            and source_games
            and target_game not in source_games
        )
        evidence_source = (
            QUALIFIED_EVIDENCE_PREFIX if qualifies
            else UNQUALIFIED_EVIDENCE_PREFIX
        ) + str(source)
        attempt_id = "concept_transfer:" + sha1(
            json.dumps(
                {
                    "concept_id": concept_id,
                    "game": game,
                    "context": context_key,
                    "action": int(action),
                    "global_step": global_step,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:24]
        self.memory.connection.execute(
            """
            INSERT OR REPLACE INTO concept_transfer_attempts_v621(
                attempt_id, concept_id, game, context_key,
                action, predicted_family, actual_family,
                success, evidence_source, global_step, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attempt_id,
                concept_id,
                target_game,
                str(context_key),
                int(action),
                None if predicted_family is None else str(predicted_family),
                str(actual_family),
                int(bool(attempt_success)),
                evidence_source,
                global_step,
                time.time(),
            ),
        )
    # Deliberately no commit here. V6System._commit_if_needed and close() own
    # transaction cadence, so --commit-steps is respected.


def _apply_trajectory_efficiency_bonus_v63_buffered(
    self: Any,
    *,
    trajectory_record: Any,
    interaction_ids: list[int],
) -> None:
    """v6.3 trajectory scoring without defeating --commit-steps."""
    from v6.memory.substrate import MemoryScore, trajectory_node_id
    from v6.memory.v63_policy import SCORE_POLICY_VERSION, unified_memory_fitness
    from v6.memory_lifecycle import ReplayCandidate

    efficiency_active = bool(trajectory_record.efficiency_active)
    memory_bonus = float(
        trajectory_record.efficiency_memory_bonus if efficiency_active else 0.0
    )
    replay_bonus = float(
        trajectory_record.efficiency_replay_bonus if efficiency_active else 0.0
    )
    retention_bonus = float(
        trajectory_record.efficiency_retention_bonus if efficiency_active else 0.0
    )
    promotion_bonus = float(
        trajectory_record.efficiency_promotion_bonus if efficiency_active else 0.0
    )
    useful_outcome = bool(
        str(trajectory_record.outcome_class) in {"WIN", "LEVEL_COMPLETE"}
        or float(trajectory_record.future_option_gain or 0.0) > 0.0
    )
    efficiency_score = (
        None
        if not efficiency_active
        or not useful_outcome
        or trajectory_record.efficiency_score is None
        else float(trajectory_record.efficiency_score)
    )

    for interaction_id in interaction_ids:
        row = self.connection.execute(
            """
            SELECT memory_fitness_base,
                   memory_replay_priority_base,
                   memory_replay_priority,
                   memory_status
            FROM interactions
            WHERE id = ?
            """,
            (int(interaction_id),),
        ).fetchone()
        base_fitness = float(
            (row[0] if row and row[0] is not None else 0.0) or 0.0
        )
        base_replay = float(
            (row[1] if row and row[1] is not None else 0.0) or 0.0
        )
        current_replay = float(
            (row[2] if row and row[2] is not None else base_replay)
            or base_replay
        )

        memory_fitness, components = unified_memory_fitness(
            isf_score=base_fitness,
            explanatory_reach=None,
            transfer_prior=None,
            transfer_empirical=None,
            recurrence_score=None,
            efficiency_score=efficiency_score,
        )
        if efficiency_score is None:
            memory_fitness = base_fitness
        replay_priority = max(current_replay, base_replay, memory_fitness)
        retention_score = memory_fitness

        values = (
            int(efficiency_active),
            str(trajectory_record.outcome_class),
            str(trajectory_record.comparable_outcome_group_id),
            None
            if trajectory_record.efficiency_score is None
            else float(trajectory_record.efficiency_score),
            memory_bonus,
            replay_bonus,
            retention_bonus,
            promotion_bonus,
            float(memory_fitness),
            float(replay_priority),
            float(base_fitness),
            float(retention_score),
        )
        self.connection.execute(
            """
            UPDATE interactions
            SET trajectory_efficiency_active=?, trajectory_outcome_class=?,
                comparable_outcome_group_id=?, trajectory_efficiency_score=?,
                efficiency_memory_bonus=?, efficiency_replay_bonus=?,
                efficiency_retention_bonus=?, efficiency_promotion_bonus=?,
                memory_fitness=?,
                memory_replay_priority=MAX(COALESCE(memory_replay_priority,0.0),?),
                retention_score_base=?, retention_score=?
            WHERE id=?
            """,
            (*values, int(interaction_id)),
        )
        self.connection.execute(
            """
            UPDATE prediction_results
            SET trajectory_efficiency_active=?, trajectory_outcome_class=?,
                comparable_outcome_group_id=?, trajectory_efficiency_score=?,
                efficiency_memory_bonus=?, efficiency_replay_bonus=?,
                efficiency_retention_bonus=?, efficiency_promotion_bonus=?,
                memory_fitness=?,
                memory_replay_priority=MAX(COALESCE(memory_replay_priority,0.0),?),
                retention_score_base=?, retention_score=?
            WHERE interaction_id=?
            """,
            (*values, int(interaction_id)),
        )

        candidate = self.memory_controller.replay_candidates.get(
            str(interaction_id)
        )
        if candidate is not None:
            self.memory_controller.replay_candidates[str(interaction_id)] = (
                ReplayCandidate(
                    interaction_id=candidate.interaction_id,
                    replay_priority=float(replay_priority),
                    reason=str(candidate.reason),
                    family_id=candidate.family_id,
                    context_signature=candidate.context_signature,
                    status=candidate.status,
                )
            )
            self._sync_post_factum_replay_fields(int(interaction_id))

        node_id = self._interaction_memory_node_id(interaction_id)
        self.memory.upsert_score(
            MemoryScore(
                node_id=node_id,
                replay_priority=float(replay_priority),
                memory_state="active" if efficiency_active else None,
                retention_score=float(retention_score),
                forgetting_score=float(max(0.0, 1.0 - retention_score)),
            ),
            step=int(interaction_id),
        )
        self.memory.connection.execute(
            """
            UPDATE memory_scores
            SET memory_fitness=?, efficiency_score=?,
                score_components_json=?, score_policy_version=?
            WHERE node_id=?
            """,
            (
                float(memory_fitness),
                efficiency_score,
                json.dumps(components, sort_keys=True),
                SCORE_POLICY_VERSION,
                node_id,
            ),
        )

    self.memory.update_node_support_and_attrs(
        trajectory_node_id(self.episode_id),
        {
            "trajectory_efficiency_active": bool(efficiency_active),
            "trajectory_efficiency_score": trajectory_record.efficiency_score,
            "efficiency_memory_bonus": memory_bonus,
            "efficiency_replay_bonus": replay_bonus,
            "efficiency_retention_bonus": retention_bonus,
            "efficiency_promotion_bonus": promotion_bonus,
            "memory_fitness_policy": "v63_unified_memory_fitness_v1",
        },
        support_increment=0,
    )
    # No commit: transaction cadence is controlled by V6Config.database_commit_every.


def _links_by_signature_cached(
    connection: Any,
    table: str,
    signature_column: str,
) -> Any:
    cache = getattr(connection, "_v63_links_cache", None)
    # sqlite3.Connection does not support arbitrary attributes on all Python
    # builds, so retain a process-local fallback keyed by object identity.
    if cache is None:
        cache = _LINK_CACHE.setdefault(id(connection), {})
    try:
        row = connection.execute(
            f'SELECT COUNT(*), COALESCE(MAX(rowid), 0) FROM "{table}"'
        ).fetchone()
        version = (int(row[0] or 0), int(row[1] or 0))
    except Exception:
        return _ORIGINAL_LINKS_BY_SIGNATURE(
            connection,
            table,
            signature_column,
        )
    key = (str(table), str(signature_column), version)
    if key not in cache:
        for stale in [
            item for item in cache
            if item[0] == str(table)
            and item[1] == str(signature_column)
        ]:
            cache.pop(stale, None)
        cache[key] = _ORIGINAL_LINKS_BY_SIGNATURE(
            connection,
            table,
            signature_column,
        )
    return cache[key]


_LINK_CACHE: dict[int, dict[Any, Any]] = {}
_COLUMN_CACHE: dict[tuple[int, int, int], set[str]] = {}


def _prediction_result_columns_cached(connection: Any) -> set[str]:
    try:
        row = connection.execute(
            "SELECT COUNT(*), COALESCE(MAX(rowid), 0) FROM prediction_results"
        ).fetchone()
        key = (id(connection), int(row[0] or 0), int(row[1] or 0))
    except Exception:
        return _ORIGINAL_PREDICTION_RESULT_COLUMNS(connection)
    cached = _COLUMN_CACHE.get(key)
    if cached is None:
        cached = set(_ORIGINAL_PREDICTION_RESULT_COLUMNS(connection))
        _COLUMN_CACHE.clear()
        _COLUMN_CACHE[key] = cached
    return set(cached)


def _transfer_rate_before_cached(self: Any, **kwargs: Any) -> tuple[float, int]:
    cache = getattr(self, "_v63_rate_cache", None)
    if cache is None:
        cache = {}
        object.__setattr__(self, "_v63_rate_cache", cache)
    key = (
        str(kwargs.get("role") or ""),
        int(kwargs.get("step") or 0),
        str(kwargs.get("source_game_key") or ""),
        str(kwargs.get("source_context_key") or ""),
        str(kwargs.get("target_game_key") or ""),
        str(kwargs.get("target_context_key") or ""),
    )
    if key not in cache:
        if len(cache) >= 500_000:
            cache.clear()
        cache[key] = _ORIGINAL_TRANSFER_RATE_BEFORE(self, **kwargs)
    return cache[key]


def _transfer_max_step_before_cached(
    self: Any,
    *,
    role: str,
    step: int,
) -> int | None:
    cache = getattr(self, "_v63_max_step_cache", None)
    if cache is None:
        cache = {}
        object.__setattr__(self, "_v63_max_step_cache", cache)
    key = (str(role), int(step))
    if key not in cache:
        if len(cache) >= 250_000:
            cache.clear()
        cache[key] = _ORIGINAL_TRANSFER_MAX_STEP_BEFORE(
            self,
            role=role,
            step=step,
        )
    return cache[key]


def _transfer_max_any_step_before_cached(
    self: Any,
    step: int,
) -> int | None:
    cache = getattr(self, "_v63_max_any_cache", None)
    if cache is None:
        cache = {}
        object.__setattr__(self, "_v63_max_any_cache", cache)
    key = int(step)
    if key not in cache:
        cache[key] = _ORIGINAL_TRANSFER_MAX_ANY_STEP_BEFORE(self, step)
    return cache[key]


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))
