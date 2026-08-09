from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from hashlib import sha1
from typing import Any


_INSTALLED = False


def install_v63_performance_compat_completion() -> None:
    """Keep legacy observables while avoiding legacy hot paths."""
    global _INSTALLED
    from v6.memory import v63_performance_completion as completion
    from v6 import v63_higher_order_compat as compat

    if not _INSTALLED:
        compat._ORIGINAL_BUILD_RELATIONAL = _build_relational_with_fast_legacy_diagnostics
        _INSTALLED = True
    _patch_v6_system()


def _patch_v6_system() -> None:
    module = sys.modules.get("v6.main")
    system_type = None if module is None else getattr(module, "V6System", None)
    if system_type is not None:
        system_type._emit_live_memory_event = _emit_live_memory_event_queue_aware


def _emit_live_memory_event_queue_aware(
    self: Any,
    event_type: str,
    event_id: str,
    global_step: int,
    priority: float,
    payload: dict,
) -> None:
    """Probe queue capacity once per local batch, not once per event."""
    if self.live_memory_queue is None:
        return
    if str(self.config.shared_live_memory_mode) not in {"write", "readwrite"}:
        return
    batch = getattr(self, "_v63_live_memory_batch", None)
    if not batch:
        full = getattr(self.live_memory_queue, "full", None)
        if callable(full):
            try:
                if bool(full()):
                    self.live_memory_events_dropped_queue_full += 1
                    return
            except (AttributeError, NotImplementedError, OSError):
                pass
    from v6.memory import v63_performance_completion as completion

    completion._emit_live_memory_event_batched(
        self,
        event_type,
        event_id,
        global_step,
        priority,
        payload,
    )


def _build_relational_with_fast_legacy_diagnostics(
    state_conn: sqlite3.Connection,
    *,
    max_world_model_family_links: int = 50,
) -> dict[str, Any]:
    from v6.memory import v63_performance_completion as completion

    summary = completion._build_relational_world_models_optimized(
        state_conn,
        max_world_model_family_links=max_world_model_family_links,
    )
    legacy_count = _insert_fast_legacy_diagnostics(
        state_conn,
        max_world_model_family_links=max_world_model_family_links,
    )
    result = dict(summary)
    result["fast_legacy_diagnostic_count"] = int(legacy_count)
    return result


def _mean(values: list[Any]) -> float | None:
    cooked = [float(value) for value in values if value is not None]
    return (sum(cooked) / len(cooked)) if cooked else None


def _insert_fast_legacy_diagnostics(
    state_conn: sqlite3.Connection,
    *,
    max_world_model_family_links: int,
) -> int:
    """Regenerate nonvalidating single-concept rows from preloaded evidence."""
    from v6 import higher_order_substrate as substrate
    from v6 import v63_higher_order_semantics as semantics

    state_conn.row_factory = sqlite3.Row
    concept_rows = [
        dict(row)
        for row in state_conn.execute(
            """
            SELECT c.concept_signature, c.promotion_score,
                   c.first_seen_global_step, c.last_seen_global_step,
                   COALESCE(s.currently_promoted, c.is_promoted, 0) AS is_promoted
            FROM concept_candidates AS c
            LEFT JOIN concept_promotion_state AS s
              ON s.concept_signature=c.concept_signature
            ORDER BY COALESCE(s.currently_promoted, c.is_promoted, 0) DESC,
                     COALESCE(c.promotion_score,0) DESC, c.concept_signature
            """
        ).fetchall()
    ]
    if not concept_rows:
        return 0
    promoted = [row for row in concept_rows if int(row.get("is_promoted", 0) or 0) == 1]
    selected = promoted if promoted else concept_rows[:20]
    concept_links = substrate._links_by_signature(
        state_conn, "concept_links", "concept_signature"
    )
    role_links = substrate._links_by_signature(
        state_conn, "role_links", "role_signature"
    )
    family_support = {
        str(row[0]): int(row[1] or 0)
        for row in state_conn.execute(
            "SELECT family_signature, COALESCE(SUM(support_count),0) "
            "FROM family_members GROUP BY family_signature"
        ).fetchall()
    }
    family_prediction = {
        str(row[0]): float(row[1] or 0.0)
        for row in state_conn.execute(
            "SELECT canonical_signature, COALESCE(AVG(prediction_lift),0.0) "
            "FROM transformation_families GROUP BY canonical_signature"
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

    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    try:
        rows = state_conn.execute(
            """
            SELECT component_signature, prediction_global_step,
                   observed_event_id, observed_global_step, prediction_correct,
                   baseline_prediction_score, component_prediction_score
            FROM world_model_prediction_events
            ORDER BY prediction_event_id
            """
        ).fetchall()
        for row in rows:
            predictions[str(row["component_signature"])].append(dict(row))
    except sqlite3.Error:
        pass

    inserted = 0
    for row in selected:
        concept = str(row["concept_signature"])
        links = concept_links.get(concept, {})
        roles = sorted(str(x) for x in links.get("role", set()))
        carriers = sorted(str(x) for x in links.get("carrier", set()))
        candidate_families = sorted(str(x) for x in links.get("family", set()))
        contexts = sorted(str(x) for x in links.get("context", set()))
        games = sorted(str(x) for x in links.get("game", set()))

        family_records = []
        for family in candidate_families:
            role_count = sum(
                1
                for role in roles
                if family in role_links.get(role, {}).get("family", set())
            )
            event_count = int(future_counts.get(family, 0))
            prediction_gain = float(family_prediction.get(family, 0.0))
            family_records.append(
                {
                    "family": family,
                    "support": int(family_support.get(family, 0)),
                    "role_count": int(role_count),
                    "event_count": event_count,
                    "prediction_gain": prediction_gain,
                    "verified": event_count > 0,
                }
            )
        eligible = [
            item
            for item in family_records
            if item["event_count"] >= 2
            or item["prediction_gain"] > 0.0
            or item["role_count"] >= 2
        ]
        eligible.sort(
            key=lambda item: (
                0 if item["verified"] else 1,
                -float(item["prediction_gain"]),
                -int(item["support"]),
                str(item["family"]),
            )
        )
        retained = eligible[: max(1, int(max_world_model_family_links))]
        families = [str(item["family"]) for item in retained]

        signature = "wm:" + sha1(concept.encode("utf-8")).hexdigest()[:20]
        prediction_rows = predictions.get(signature, [])
        matched = [
            item
            for item in prediction_rows
            if item.get("observed_event_id") not in (None, "")
            and item.get("observed_global_step") is not None
            and item.get("prediction_global_step") is not None
            and int(item["prediction_global_step"]) < int(item["observed_global_step"])
        ]
        unmatched = len(prediction_rows) - len(matched)
        correct = sum(int(item.get("prediction_correct") or 0) for item in matched)
        baseline = _mean([item.get("baseline_prediction_score") for item in matched])
        component_score = _mean([item.get("component_prediction_score") for item in matched])
        gain = (
            float(component_score) - float(baseline)
            if component_score is not None and baseline is not None
            else 0.0
        )
        observed = len(matched)
        status = "verified" if observed else "proxy" if prediction_rows else "missing"

        role_count = len(roles)
        family_count = len(families)
        carrier_count = len(carriers)
        context_count = len(contexts)
        game_count = len(games)
        node_count = 1 + role_count + family_count + carrier_count + context_count + game_count
        explanatory = float(family_count + carrier_count) / max(1, node_count)
        first_seen = None if row.get("first_seen_global_step") is None else int(row["first_seen_global_step"])
        last_seen = None if row.get("last_seen_global_step") is None else int(row["last_seen_global_step"])

        state_conn.execute(
            """
            INSERT OR REPLACE INTO world_model_components (
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
            ) VALUES (?, 'legacy_single_concept_diagnostic', ?, ?, 1, ?, ?, ?, ?, ?, ?,
                      0, 0, 0.0, 1, ?, 0, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      0.0, 0.0, 0.0, ?, ?, ?, ?, ?)
            """,
            (
                signature,
                node_count,
                role_count + family_count + carrier_count + context_count + game_count,
                role_count,
                family_count,
                carrier_count,
                context_count,
                game_count,
                explanatory,
                observed,
                first_seen,
                last_seen,
                family_count + role_count,
                observed,
                correct,
                max(0, observed - correct),
                status,
                baseline,
                component_score,
                gain,
                observed,
                unmatched,
                len(candidate_families),
                len(retained),
                max(0, len(candidate_families) - len(retained)),
                max(0, len(candidate_families) - len(eligible)),
                max(0, len(eligible) - len(retained)),
            ),
        )
        substrate._insert_link(
            state_conn,
            "world_model_links",
            "component_signature",
            signature,
            "concept",
            concept,
            1,
            first_seen,
            last_seen,
        )
        for kind, values in (
            ("role", roles),
            ("carrier", carriers),
            ("family", families),
            ("context", contexts),
            ("game", games),
        ):
            for value in values:
                substrate._insert_link(
                    state_conn,
                    "world_model_links",
                    "component_signature",
                    signature,
                    kind,
                    value,
                    1,
                    first_seen,
                    last_seen,
                )
        for item in retained:
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
                    str(item["family"]),
                    int(item["support"]),
                    int(item["role_count"]),
                    int(item["event_count"]),
                    float(item["prediction_gain"]),
                    "verified" if item["verified"] else "proxy",
                ),
            )
        state_conn.execute(
            """
            INSERT INTO world_model_component_state (
                component_signature, historically_coherent, currently_coherent,
                first_coherent_global_step, last_validated_global_step,
                consecutive_validation_failures, validation_status, updated_at
            ) VALUES (?, 0, 0, NULL, ?, 0,
                      'legacy_single_concept_noncanonical', datetime('now'))
            ON CONFLICT(component_signature) DO UPDATE SET
                currently_coherent=0,
                last_validated_global_step=excluded.last_validated_global_step,
                validation_status='legacy_single_concept_noncanonical',
                updated_at=excluded.updated_at
            """,
            (signature, last_seen),
        )
        inserted += 1
    state_conn.commit()
    return inserted
