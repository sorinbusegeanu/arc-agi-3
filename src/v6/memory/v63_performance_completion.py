from __future__ import annotations

import json
import sqlite3
import time
from collections import defaultdict
from hashlib import sha1
from queue import Empty, Full
from typing import Any


_INSTALLED = False
_ORIGINAL_UPDATE_MULTI_SCALE: Any = None
_ORIGINAL_STABLE_CONTINGENCIES: Any = None
_ORIGINAL_QUERY_PREDICT: Any = None
_ORIGINAL_CONTROLLER_PREDICT: Any = None
_ORIGINAL_SNAPSHOT_SCORE_ACTION: Any = None
_ORIGINAL_V6_RUN: Any = None

_LIVE_BATCH_SIZE = 64


def install_v63_performance_completion() -> None:
    """Install semantics-preserving v6.3 performance completions."""
    global _INSTALLED
    global _ORIGINAL_UPDATE_MULTI_SCALE
    global _ORIGINAL_STABLE_CONTINGENCIES
    global _ORIGINAL_QUERY_PREDICT
    global _ORIGINAL_CONTROLLER_PREDICT
    global _ORIGINAL_SNAPSHOT_SCORE_ACTION
    global _ORIGINAL_V6_RUN
    if _INSTALLED:
        _patch_v6_system_if_loaded()
        return

    from v6.contingency.contingency_learner import ContingencyLearner
    from v6.memory.v621_runtime import (
        V621MemoryController,
        V621MemoryQueryEngine,
        V621SnapshotMemoryQueryEngine,
    )
    from v6.memory import live_memory_queue as live
    from v6 import higher_order_substrate as substrate
    from v6 import v63_higher_order_semantics as semantics
    from v6 import v63_higher_order_compat as compat

    _ORIGINAL_UPDATE_MULTI_SCALE = ContingencyLearner.update_multi_scale
    _ORIGINAL_STABLE_CONTINGENCIES = ContingencyLearner.stable_contingencies
    ContingencyLearner.update_multi_scale = _update_multi_scale_incremental
    ContingencyLearner.stable_contingencies = _stable_contingencies_incremental

    _ORIGINAL_QUERY_PREDICT = V621MemoryQueryEngine.predict_family
    V621MemoryQueryEngine.predict_family = _query_predict_cached
    _ORIGINAL_CONTROLLER_PREDICT = V621MemoryController.predict
    V621MemoryController.predict = _controller_predict_cached
    _ORIGINAL_SNAPSHOT_SCORE_ACTION = V621SnapshotMemoryQueryEngine.score_action
    V621SnapshotMemoryQueryEngine.score_action = _snapshot_score_action_cache_prediction

    live.LiveMemoryWriter.run = _live_writer_run_batched

    # Compatibility must keep wrapping the relational builder, but its inner
    # builder is replaced with the preloaded implementation below.
    if getattr(compat, "_INSTALLED", False):
        compat._ORIGINAL_BUILD_RELATIONAL = _build_relational_world_models_optimized
        semantics._build_relational_world_models = compat._build_relational_with_legacy_diagnostics
    else:
        semantics._build_relational_world_models = _build_relational_world_models_optimized

    substrate.derive_world_model_components = _derive_world_model_components_single_pass

    _INSTALLED = True
    _patch_v6_system_if_loaded()


def _update_multi_scale_incremental(
    self: Any,
    context_signatures: dict[int, tuple],
    action: int,
    transformation_family: int,
) -> Any:
    keys = [
        (int(level), tuple(context), int(action), int(transformation_family))
        for level, context in context_signatures.items()
    ]
    before = {
        key: (
            None
            if key not in self.contingencies
            else (
                int(self.contingencies[key].support_count),
                float(self.contingencies[key].confidence),
            )
        )
        for key in keys
    }
    result = _ORIGINAL_UPDATE_MULTI_SCALE(
        self, context_signatures, action, transformation_family
    )
    dirty = []
    for key in keys:
        item = self.contingencies.get(key)
        if item is None:
            continue
        current = (int(item.support_count), float(item.confidence))
        if before.get(key) != current:
            dirty.append(item)
    self._v63_dirty_stable_contingencies = dirty
    self._v63_dirty_stable_read_pending = True
    return result


def _stable_contingencies_incremental(self: Any) -> list[Any]:
    if bool(getattr(self, "_v63_dirty_stable_read_pending", False)):
        self._v63_dirty_stable_read_pending = False
        rows = list(getattr(self, "_v63_dirty_stable_contingencies", []) or [])
        self._v63_dirty_stable_contingencies = []
        return sorted(rows, key=lambda item: (item.context_level, item.action, item.id))
    return _ORIGINAL_STABLE_CONTINGENCIES(self)


def _context_cache_key(context_signatures: dict[int, tuple], action: int) -> tuple[Any, ...]:
    return (
        int(action),
        tuple(
            (int(level), tuple(context_signatures[level]))
            for level in sorted(context_signatures)
        ),
    )


def _cache_prediction(engine: Any, context_signatures: dict[int, tuple], action: int, prediction: Any) -> None:
    engine._v63_selected_prediction_cache = (
        _context_cache_key(context_signatures, action),
        prediction,
    )


def _query_predict_cached(
    self: Any,
    context_signatures: dict[int, tuple],
    action: int,
    *,
    record_query: bool = False,
) -> Any:
    prediction = _ORIGINAL_QUERY_PREDICT(
        self,
        context_signatures,
        action,
        record_query=record_query,
    )
    if not record_query:
        _cache_prediction(self, context_signatures, action, prediction)
    return prediction


def _snapshot_score_action_cache_prediction(
    self: Any,
    context_signatures: dict[int, tuple],
    action: int,
    available_actions: list[int],
    *,
    record_query: bool = False,
) -> Any:
    # The v6.3 optimized snapshot scorer computes prediction internally.  Use
    # its ranked result to avoid a second structural traversal, while deriving
    # the exact prediction only once here and caching it for run_step().
    score = _ORIGINAL_SNAPSHOT_SCORE_ACTION(
        self,
        context_signatures,
        action,
        available_actions,
        record_query=record_query,
    )
    prediction = self.predict_family(
        context_signatures,
        action,
        record_query=False,
    )
    _cache_prediction(self, context_signatures, action, prediction)
    return score


def _controller_predict_cached(
    self: Any,
    context_signatures: dict[int, tuple],
    action: int,
    *,
    record_query: bool = False,
) -> Any:
    if not record_query:
        cached = getattr(self.query_engine, "_v63_selected_prediction_cache", None)
        if cached is not None and cached[0] == _context_cache_key(context_signatures, action):
            return cached[1]
    return _ORIGINAL_CONTROLLER_PREDICT(
        self,
        context_signatures,
        action,
        record_query=record_query,
    )


def _patch_v6_system_if_loaded() -> None:
    global _ORIGINAL_V6_RUN
    import sys

    module = sys.modules.get("v6.main")
    system_type = None if module is None else getattr(module, "V6System", None)
    if system_type is None:
        return
    system_type._emit_live_memory_event = _emit_live_memory_event_batched
    if _ORIGINAL_V6_RUN is None:
        _ORIGINAL_V6_RUN = system_type.run
        system_type.run = _v6_run_flush_live_batch


def _event_fingerprint(priority: float, payload: dict[str, Any]) -> tuple[Any, ...]:
    return (
        float(priority),
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str),
    )


def _emit_live_memory_event_batched(
    self: Any,
    event_type: str,
    event_id: str,
    global_step: int,
    priority: float,
    payload: dict,
) -> None:
    if self.live_memory_queue is None:
        return
    if str(self.config.shared_live_memory_mode) not in {"write", "readwrite"}:
        return
    from v6.memory.live_memory_queue import LiveMemoryEvent

    event_type = str(event_type)
    event_id = str(event_id)
    bounded_priority = max(0.0, min(1.0, float(priority)))
    cooked_payload = dict(payload or {})
    if event_type in {
        "stable_contingency",
        "family_update",
        "carrier_candidate",
        "contradiction_cluster",
    }:
        fingerprints = getattr(self, "_v63_live_event_fingerprints", None)
        if fingerprints is None:
            fingerprints = {}
            self._v63_live_event_fingerprints = fingerprints
        key = (event_type, event_id)
        fingerprint = _event_fingerprint(bounded_priority, cooked_payload)
        if fingerprints.get(key) == fingerprint:
            self._v63_live_events_deduplicated = int(
                getattr(self, "_v63_live_events_deduplicated", 0)
            ) + 1
            return
        fingerprints[key] = fingerprint

    batch = getattr(self, "_v63_live_memory_batch", None)
    if batch is None:
        batch = []
        self._v63_live_memory_batch = batch
    batch.append(
        LiveMemoryEvent(
            event_type=event_type,
            event_id=event_id,
            global_step=int(global_step),
            worker_id=str(self.config.live_memory_worker_id or "unknown_worker"),
            priority=bounded_priority,
            payload=cooked_payload,
        )
    )
    self.live_memory_events_emitted += 1
    if len(batch) >= _LIVE_BATCH_SIZE:
        _flush_live_memory_batch(self)


def _flush_live_memory_batch(self: Any) -> None:
    batch = list(getattr(self, "_v63_live_memory_batch", []) or [])
    if not batch or self.live_memory_queue is None:
        return
    self._v63_live_memory_batch = []
    started = time.perf_counter()
    try:
        self.live_memory_queue.put_nowait(batch)
        self.live_memory_queue_block_seconds += time.perf_counter() - started
        if int(self.live_memory_events_emitted) % 256 < _LIVE_BATCH_SIZE:
            try:
                self.live_memory_queue_peak_size = max(
                    self.live_memory_queue_peak_size,
                    int(self.live_memory_queue.qsize()),
                )
            except (AttributeError, NotImplementedError, OSError):
                pass
    except Full:
        self.live_memory_events_dropped_queue_full += len(batch)
    except Exception:
        self.live_memory_events_dropped_error += len(batch)


def _v6_run_flush_live_batch(self: Any, steps: int | None = None) -> Any:
    try:
        return _ORIGINAL_V6_RUN(self, steps=steps)
    finally:
        _flush_live_memory_batch(self)


def _live_writer_run_batched(self: Any) -> None:
    from v6.memory import live_memory_queue as live

    self.memory_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(self.sqlite_path)
    try:
        live._configure_live_memory_sqlite(connection)
        live._ensure_live_memory_schema(connection)
        self._next_sequence = int(
            connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM live_memory_events"
            ).fetchone()[0]
        )
        batch: list[dict[str, Any]] = []
        last_flush = time.time()
        while not self._stop_requested:
            timeout = max(0.05, float(self.config.flush_seconds))
            try:
                raw = self.queue.get(timeout=timeout)
            except Empty:
                raw = None
            except Exception:
                raw = None
            raw_items = list(raw) if isinstance(raw, (list, tuple)) else ([raw] if raw is not None else [])
            for raw_event in raw_items:
                self.summary["events_received"] = int(self.summary["events_received"]) + 1
                event = live._normalize_event(raw_event)
                if event is None:
                    self.summary["events_dropped_invalid"] = int(self.summary["events_dropped_invalid"]) + 1
                elif event["event_type"] == "__stop__":
                    self.summary["queue_stop_received"] = True
                    self._stop_requested = True
                    break
                elif float(event["priority"]) < float(self.config.min_priority):
                    self.summary["events_dropped_low_priority"] = int(self.summary["events_dropped_low_priority"]) + 1
                else:
                    event["sequence"] = int(self._next_sequence)
                    self._next_sequence += 1
                    batch.append(event)
                    counts = dict(self.summary.get("event_type_counts", {}) or {})
                    counts[event["event_type"]] = int(counts.get(event["event_type"], 0) or 0) + 1
                    self.summary["event_type_counts"] = counts
            now = time.time()
            if batch and (
                len(batch) >= int(self.config.batch_size)
                or self._stop_requested
                or (now - last_flush) >= float(self.config.flush_seconds)
            ):
                self._flush_batch(connection, batch)
                batch = []
                last_flush = now
        if batch:
            self._flush_batch(connection, batch)
    finally:
        connection.close()
        self._write_summary()


def _derive_world_model_components_single_pass(
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    progress_factory: Any | None = None,
    max_world_model_family_links: int = 50,
) -> dict[str, Any]:
    del graph_conn, progress_factory
    from v6 import v63_higher_order_semantics as semantics

    return semantics._build_relational_world_models(
        state_conn,
        max_world_model_family_links=max_world_model_family_links,
    )


def _pair_strengths(concept_rows: list[dict[str, Any]], concept_links: dict[str, dict[str, set[str]]]) -> dict[tuple[str, str], int]:
    strength: dict[tuple[str, str], int] = defaultdict(int)
    by_value: dict[tuple[str, str], list[str]] = defaultdict(list)
    weights = {"role": 3, "family": 2, "context": 1}
    for row in concept_rows:
        concept = str(row["concept_signature"])
        links = concept_links.get(concept, {})
        for kind in weights:
            for value in links.get(kind, set()):
                by_value[(kind, str(value))].append(concept)
    for (kind, _value), concepts in by_value.items():
        unique = sorted(set(concepts))
        weight = weights[kind]
        for index, left in enumerate(unique):
            for right in unique[index + 1 :]:
                strength[(left, right)] += weight
    return strength


def _build_relational_world_models_optimized(
    state_conn: sqlite3.Connection,
    *,
    max_world_model_family_links: int = 50,
) -> dict[str, Any]:
    from v6 import higher_order_substrate as substrate
    from v6 import v63_higher_order_semantics as semantics

    state_conn.row_factory = sqlite3.Row
    concept_rows = [
        dict(row)
        for row in state_conn.execute(
            """
            SELECT c.concept_signature, c.promotion_score,
                   c.first_seen_global_step, c.last_seen_global_step,
                   COALESCE(s.currently_promoted, c.is_promoted, 0) AS promoted
            FROM concept_candidates AS c
            LEFT JOIN concept_promotion_state AS s
              ON s.concept_signature=c.concept_signature
            WHERE COALESCE(s.currently_promoted, c.is_promoted, 0)=1
            ORDER BY COALESCE(c.promotion_score,0) DESC, c.concept_signature
            """
        ).fetchall()
    ]
    by_id = {str(row["concept_signature"]): row for row in concept_rows}
    concept_links = substrate._links_by_signature(
        state_conn, "concept_links", "concept_signature"
    )
    pair_strength = _pair_strengths(concept_rows, concept_links)
    candidates: list[tuple[tuple[int, int, float, str, str], dict[str, Any]]] = []
    for (left_id, right_id), relation_strength in pair_strength.items():
        left = by_id[left_id]
        right = by_id[right_id]
        ll = concept_links.get(left_id, {})
        rr = concept_links.get(right_id, {})
        unions = {
            kind: set(ll.get(kind, set())) | set(rr.get(kind, set()))
            for kind in ("role", "carrier", "family", "context", "game")
        }
        if len(unions["family"]) < 2 or len(unions["carrier"]) < 2:
            continue
        score = float(left.get("promotion_score") or 0.0) + float(right.get("promotion_score") or 0.0)
        candidates.append(
            (
                (
                    int(relation_strength),
                    len(unions["context"]) + len(unions["game"]),
                    score,
                    left_id,
                    right_id,
                ),
                {"left": left, "right": right, "links": unions},
            )
        )
    candidates.sort(key=lambda item: (-item[0][0], -item[0][1], -item[0][2], item[0][3], item[0][4]))
    selected = [item[1] for item in candidates[: int(semantics.MAX_RELATIONAL_COMPONENTS)]]

    role_links = substrate._links_by_signature(state_conn, "role_links", "role_signature")
    family_support = {
        str(row[0]): int(row[1] or 0)
        for row in state_conn.execute(
            "SELECT family_signature, COALESCE(SUM(support_count),0) FROM family_members GROUP BY family_signature"
        ).fetchall()
    }
    family_expr = semantics._future_event_family_expr(state_conn)
    try:
        future_counts = {
            str(row[0]): int(row[1] or 0)
            for row in state_conn.execute(
                f"SELECT {family_expr}, COUNT(*) FROM future_option_events "
                f"WHERE {family_expr} IS NOT NULL GROUP BY {family_expr}"
            ).fetchall()
        }
    except sqlite3.Error:
        future_counts = {}

    state_conn.execute("DELETE FROM world_model_family_links")
    state_conn.execute("DELETE FROM world_model_links")
    state_conn.execute("DELETE FROM world_model_components")

    current_step = semantics._current_evidence_step(state_conn)
    component_count = 0
    coherent_count = 0
    for spec in selected:
        concepts = sorted([str(spec["left"]["concept_signature"]), str(spec["right"]["concept_signature"])])
        links = spec["links"]
        signature = "wm:rel:" + sha1(json.dumps(concepts, separators=(",", ":")).encode("utf-8")).hexdigest()[:20]
        semantics._match_world_model_predictions(state_conn, signature)
        prediction = semantics._world_model_prediction_metrics(state_conn, signature)
        if current_step is not None:
            semantics._issue_world_model_prediction(
                state_conn,
                signature=signature,
                prediction_step=current_step,
                families=sorted(str(x) for x in links["family"]),
                contexts=sorted(str(x) for x in links["context"]),
                games=sorted(str(x) for x in links["game"]),
            )
            prediction = semantics._world_model_prediction_metrics(state_conn, signature)

        role_count = len(links["role"])
        family_count = len(links["family"])
        carrier_count = len(links["carrier"])
        context_count = len(links["context"])
        game_count = len(links["game"])
        node_count = 2 + role_count + family_count + carrier_count + context_count + game_count
        explanatory = float(family_count + carrier_count + role_count) / max(1, node_count)
        scope = min(1.0, float(context_count + game_count) / 5.0)
        structural = min(1.0, 0.4 + 0.1 * min(2, role_count) + 0.1 * min(2, family_count) + 0.1 * min(2, carrier_count))
        functional = max(0.0, min(1.0, 0.45 * float(prediction["accuracy"]) + 0.35 * max(0.0, float(prediction["gain"])) + 0.20 * scope))
        coherence = max(0.0, min(1.0, 0.4 * structural + 0.6 * functional))
        is_coherent = int(
            prediction["matched"] > 0
            and role_count >= 1
            and family_count >= 2
            and carrier_count >= 2
            and (context_count >= 3 or game_count >= 2)
            and prediction["gain"] > 0.0
            and coherence >= 0.55
        )
        first_values = [value for value in (spec["left"].get("first_seen_global_step"), spec["right"].get("first_seen_global_step")) if value is not None]
        last_values = [value for value in (spec["left"].get("last_seen_global_step"), spec["right"].get("last_seen_global_step")) if value is not None]
        first_seen = max(int(x) for x in first_values) if first_values else None
        last_seen = max(int(x) for x in last_values) if last_values else None
        state_conn.execute(
            """
            INSERT INTO world_model_components (
                component_signature, component_type, node_count, edge_count,
                linked_concept_count, linked_role_count, linked_family_count,
                linked_carrier_count, cross_context_count, cross_game_count,
                explanatory_coverage, prediction_support_count,
                contradiction_coverage_count, coherence_score, candidate_only,
                predicted_outcome_count, predicted_outcome_count_is_proxy,
                first_seen_global_step, last_seen_global_step, is_coherent,
                structural_prediction_support_count, observed_outcome_count,
                correct_prediction_count, prediction_error_count,
                prediction_evidence_status, baseline_prediction_score,
                component_prediction_score, heldout_prediction_gain,
                matched_prediction_event_count, unmatched_prediction_event_count,
                structural_coherence_score, functional_coherence_score,
                combined_coherence_score, candidate_family_link_count,
                retained_family_link_count, dropped_family_link_count,
                family_links_dropped_low_support, family_links_dropped_limit
            ) VALUES (?, 'relational_concept_component', ?, ?, 2, ?, ?, ?, ?, ?, ?,
                      0, 0, ?, 0, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, 0, 0, 0)
            """,
            (
                signature,
                node_count,
                role_count + family_count + carrier_count + context_count + game_count + 1,
                role_count,
                family_count,
                carrier_count,
                context_count,
                game_count,
                explanatory,
                coherence,
                prediction["matched"],
                first_seen,
                last_seen,
                is_coherent,
                family_count + role_count,
                prediction["matched"],
                prediction["correct"],
                max(0, prediction["matched"] - prediction["correct"]),
                "verified" if prediction["matched"] else ("proxy" if prediction["unmatched"] else "missing"),
                prediction["baseline"],
                prediction["component"],
                prediction["gain"],
                prediction["matched"],
                prediction["unmatched"],
                structural,
                functional,
                coherence,
                family_count,
                min(family_count, int(max_world_model_family_links)),
            ),
        )
        for concept in concepts:
            substrate._insert_link(state_conn, "world_model_links", "component_signature", signature, "concept", concept, 1, first_seen, last_seen)
        for kind in ("role", "carrier", "family", "context", "game"):
            for value in sorted(str(x) for x in links[kind]):
                substrate._insert_link(state_conn, "world_model_links", "component_signature", signature, kind, value, 1, first_seen, last_seen)
        for family in sorted(str(x) for x in links["family"])[: int(max_world_model_family_links)]:
            event_count = int(future_counts.get(family, 0))
            role_link_count = sum(
                1
                for role in links["role"]
                if family in role_links.get(str(role), {}).get("family", set())
            )
            state_conn.execute(
                """
                INSERT OR REPLACE INTO world_model_family_links (
                    component_signature, family_signature, family_link_support_count,
                    family_link_role_count, family_link_event_count,
                    family_link_prediction_gain, family_link_provenance_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signature,
                    family,
                    int(family_support.get(family, 0)),
                    int(role_link_count),
                    event_count,
                    0.0,
                    "verified" if event_count else "proxy",
                ),
            )
        state_conn.execute(
            """
            INSERT INTO world_model_component_state (
                component_signature, historically_coherent, currently_coherent,
                first_coherent_global_step, last_validated_global_step,
                consecutive_validation_failures, validation_status, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(component_signature) DO UPDATE SET
                historically_coherent=MAX(world_model_component_state.historically_coherent, excluded.historically_coherent),
                currently_coherent=excluded.currently_coherent,
                first_coherent_global_step=COALESCE(world_model_component_state.first_coherent_global_step, excluded.first_coherent_global_step),
                last_validated_global_step=excluded.last_validated_global_step,
                consecutive_validation_failures=CASE WHEN excluded.currently_coherent=1 THEN 0 ELSE world_model_component_state.consecutive_validation_failures+1 END,
                validation_status=excluded.validation_status,
                updated_at=excluded.updated_at
            """,
            (
                signature,
                is_coherent,
                is_coherent,
                first_seen if is_coherent else None,
                last_seen,
                0 if is_coherent else 1,
                "passed" if is_coherent else "awaiting_heldout_prediction" if not prediction["matched"] else "failed",
            ),
        )
        component_count += 1
        coherent_count += is_coherent

    first_component_row = state_conn.execute("SELECT MIN(first_seen_global_step) FROM world_model_components").fetchone()
    first_coherent_row = state_conn.execute("SELECT MIN(first_seen_global_step) FROM world_model_components WHERE COALESCE(is_coherent,0)=1").fetchone()
    first_component = None if first_component_row is None else first_component_row[0]
    first_coherent = None if first_coherent_row is None else first_coherent_row[0]
    substrate._write_milestone(state_conn, "first_world_model_component_step", first_component, None)
    substrate._write_milestone(state_conn, "first_coherent_world_model_step", first_coherent, None)
    state_conn.commit()
    return {
        "world_model_component_count": component_count,
        "coherent_world_model_component_count": coherent_count,
        "candidate_only_world_model_component_count": component_count - coherent_count,
        "world_model_semantics_version": "v63_relational_multiconcept_v1",
        "world_model_derivation_mode": "single_pass_preloaded_v1",
        "relational_pair_candidate_count": len(pair_strength),
    }
