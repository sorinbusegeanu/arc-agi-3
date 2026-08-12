from __future__ import annotations

import json
import os
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

_INSTALLED = False
_ORIGINAL_PREPARE: Any = None
_ORIGINAL_TRANSFER_LINKS: Any = None


def install_suite_runtime_repairs() -> None:
    """Install H08 ordering/lifecycle and large H11 derivation repairs."""
    global _INSTALLED, _ORIGINAL_PREPARE, _ORIGINAL_TRANSFER_LINKS
    if _INSTALLED:
        return

    from v6 import future_options as fo
    from v6 import hypothesis_suite_report as suite
    from v6 import v63_higher_order_semantics as semantics

    _ORIGINAL_PREPARE = suite.prepare_hypothesis_evidence
    _ORIGINAL_TRANSFER_LINKS = fo.derive_future_option_transfer_links

    fo.derive_future_option_transfer_links = _derive_future_option_transfer_links_bounded
    suite.prepare_hypothesis_evidence = _prepare_hypothesis_evidence_post_future_world_models
    semantics._match_world_model_predictions = _match_world_model_predictions_scope_aware
    _INSTALLED = True


def _scope_payload_is_v2(payload: Any) -> bool:
    if payload in (None, ""):
        return False
    try:
        parsed = json.loads(str(payload))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(parsed, dict) and parsed.get("scope_version") == "component_scope_v2"


def _match_world_model_predictions_scope_aware(
    conn: sqlite3.Connection,
    signature: str,
) -> None:
    """Match against component scope and never expire v2 predictions on unrelated evidence."""
    from v6 import h08_world_model_prediction_repair as repair

    family_expr = repair._future_event_family_expr(conn)
    rows = conn.execute(
        """
        SELECT prediction_event_id, prediction_global_step, predicted_family,
               game_key, context_key, predicted_outcome
        FROM world_model_prediction_events
        WHERE component_signature=? AND observed_event_id IS NULL
          AND COALESCE(provenance_status, 'prospective')='prospective'
        ORDER BY prediction_global_step ASC, prediction_event_id ASC
        """,
        (signature,),
    ).fetchall()

    current_step = repair._current_evidence_step(conn)
    for row in rows:
        prediction_step = int(row[1])
        contexts, games = repair._parse_prediction_scope(
            row[5], fallback_context=row[4], fallback_game=row[3]
        )
        where = ["last_seen_global_step > ?"]
        params: list[Any] = [prediction_step]
        if contexts:
            placeholders = ",".join("?" for _ in contexts)
            where.append(f"context_key IN ({placeholders})")
            params.extend(contexts)
        if games:
            placeholders = ",".join("?" for _ in games)
            where.append(f"game IN ({placeholders})")
            params.extend(games)
        try:
            observed = conn.execute(
                f"SELECT event_id, last_seen_global_step, {family_expr} AS family, motif_type "
                "FROM future_option_events WHERE " + " AND ".join(where)
                + " ORDER BY last_seen_global_step ASC, event_id ASC LIMIT 1",
                params,
            ).fetchone()
        except sqlite3.Error:
            observed = None

        if observed is None:
            if (
                not _scope_payload_is_v2(row[5])
                and current_step is not None
                and int(current_step) > prediction_step
            ):
                conn.execute(
                    "UPDATE world_model_prediction_events "
                    "SET provenance_status='stale' WHERE prediction_event_id=?",
                    (str(row[0]),),
                )
            continue

        observed_family = None if observed[2] is None else str(observed[2])
        correct = int(observed_family == str(row[2]))
        conn.execute(
            """
            UPDATE world_model_prediction_events
            SET observed_event_id=?, observed_global_step=?, observed_family=?,
                observed_effect=?, prediction_correct=?, component_prediction_score=?,
                provenance_status='verified'
            WHERE prediction_event_id=?
            """,
            (
                str(observed[0]),
                int(observed[1]),
                observed_family,
                None if observed[3] is None else str(observed[3]),
                correct,
                float(correct),
                str(row[0]),
            ),
        )


def _prepare_hypothesis_evidence_post_future_world_models(**kwargs: Any) -> dict[str, Any]:
    """Refresh M5 after future-option evidence so prospective H08 trials can close."""
    summary = _ORIGINAL_PREPARE(**kwargs)
    if str(kwargs.get("suite_mode") or "fast").lower() != "full":
        return summary
    memory_dir = kwargs.get("memory_dir")
    if memory_dir is None:
        return summary

    from v6 import hypothesis_suite_report as suite

    wm_started = time.perf_counter()
    refreshed = suite._call_supported(
        suite.derive_world_model_components_only,
        memory_dir=Path(memory_dir),
        run_dir=Path(kwargs["run_dir"]),
    )
    summary.setdefault("steps", {})["world_models_post_future_options"] = refreshed
    summary.setdefault("timings", {})[
        "DERIVE.world_models_post_future_options_seconds"
    ] = time.perf_counter() - wm_started

    if bool(kwargs.get("incremental_promotion_validation")):
        config = suite.IncrementalPromotionValidationConfig(
            enabled=True,
            min_incremental_coverage=float(kwargs.get("promotion_min_incremental_coverage", 0.05)),
            min_cross_context_or_game_evidence=int(kwargs.get("promotion_min_cross_context_or_game_evidence", 2)),
            min_behavioral_or_predictive_lift=float(kwargs.get("promotion_min_behavioral_or_predictive_lift", 0.01)),
            min_relevant_heldout_event_count=int(kwargs.get("promotion_min_relevant_heldout_event_count", 20)),
            promotion_population_comparability_threshold=float(kwargs.get("promotion_population_comparability_threshold", 0.80)),
            demotion_failure_limit=int(kwargs.get("promotion_demotion_failure_limit", 2)),
        )
        validation_started = time.perf_counter()
        validation = suite._call_supported(
            suite.validate_incremental_promotions_only,
            memory_dir=Path(memory_dir),
            config=config,
            validate_roles_and_concepts=False,
            validate_world_models=True,
            diagnostic_epoch_id=kwargs.get("epoch_id"),
        )
        summary["steps"]["world_model_validation_post_future_options"] = validation
        summary["timings"][
            "DERIVE.world_model_validation_post_future_options_seconds"
        ] = time.perf_counter() - validation_started
    return summary


def _transfer_scope(row: dict[str, Any]) -> str:
    sg_sur = int(row.get("source_game_is_surrogate") or 0)
    tg_sur = int(row.get("target_game_is_surrogate") or 0)
    sc_sur = int(row.get("source_context_is_surrogate") or 0)
    tc_sur = int(row.get("target_context_is_surrogate") or 0)
    if any((sg_sur, tg_sur, sc_sur, tc_sur)):
        return "surrogate_resolved"
    cross_game = (
        row.get("source_game_key") not in (None, "")
        and row.get("target_game_key") not in (None, "")
        and row.get("source_game_key") != row.get("target_game_key")
    )
    cross_context = (
        row.get("source_context_key") not in (None, "")
        and row.get("target_context_key") not in (None, "")
        and row.get("source_context_key") != row.get("target_context_key")
    )
    if cross_game and cross_context:
        return "cross_game_and_context"
    if cross_game:
        return "cross_game"
    if cross_context:
        return "cross_context"
    return "same_scope"


def _derive_future_option_transfer_links_bounded(state_conn: sqlite3.Connection) -> dict[str, Any]:
    """Avoid motif x concept x exact-scope-pair explosion on large evidence sets."""
    started = time.perf_counter()
    threshold = max(1, int(os.getenv("ARC_H11_COMPACT_TRANSFER_THRESHOLD", "5000")))
    attempt_count = int(
        state_conn.execute(
            "SELECT COUNT(*) FROM role_transfer_attempts WHERE provenance_mode='single_source'"
        ).fetchone()[0]
        or 0
    )
    motif_count = int(
        state_conn.execute("SELECT COUNT(*) FROM future_option_motifs").fetchone()[0] or 0
    )
    if attempt_count < threshold or motif_count < 10:
        result = dict(_ORIGINAL_TRANSFER_LINKS(state_conn))
        result["future_option_transfer_compaction_applied"] = False
        result["derive_future_option_transfer_links_seconds"] = time.perf_counter() - started
        return result

    from v6 import future_options as fo

    state_conn.row_factory = sqlite3.Row
    state_conn.execute("DELETE FROM future_option_transfer_links")
    motif_links = fo._links_by_signature(state_conn, "future_option_links", "motif_signature")
    motif_quality = {
        str(row["motif_signature"]): dict(row)
        for row in state_conn.execute(
            "SELECT motif_signature, is_emergent, support_count, motif_stability_score, source_interaction_ids_json "
            "FROM future_option_motifs ORDER BY motif_signature"
        ).fetchall()
    }
    concept_links = fo._links_by_signature(state_conn, "concept_links", "concept_signature")
    concept_records = fo._concept_validation_records(state_conn)
    concept_resolutions = fo._resolve_concepts_for_roles(
        state_conn, concept_links=concept_links, concept_records=concept_records
    )
    promoted = {
        signature
        for signature, record in concept_records.items()
        if str(record.get("status")) == "verified"
    }

    transfer_rows = [
        dict(row)
        for row in state_conn.execute(
            """
            SELECT source_role_signature AS role_signature, transfer_score, best_margin,
                   reuse_success, similarity_score, source_evidence_support_count,
                   candidate_role_count, source_game_key, target_game_key,
                   source_context_key, target_context_key, source_interaction_id,
                   target_interaction_id, source_game_is_surrogate,
                   target_game_is_surrogate, source_context_is_surrogate,
                   target_context_is_surrogate, source_game_resolution_source,
                   target_game_resolution_source, source_context_resolution_source,
                   target_context_resolution_source, provenance_mode, provenance_status
            FROM role_transfer_attempts
            WHERE provenance_mode='single_source'
            ORDER BY source_role_signature, attempt_id
            """
        ).fetchall()
    ]
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in transfer_rows:
        grouped[str(row.get("role_signature") or "")][_transfer_scope(row)].append(row)

    inserted = 0
    aggregate_attempts = 0
    aggregate_successes = 0
    aggregate_strong = 0
    verified_chain_count = 0
    emergent_chain_count = 0
    motifs_with_transfer: set[str] = set()
    motifs_with_strong: set[str] = set()
    motifs_with_promoted: set[str] = set()
    verified_motifs_with_transfer: set[str] = set()
    verified_motifs_with_strong: set[str] = set()
    verified_motifs_with_promoted: set[str] = set()
    roles_seen: set[str] = set()
    roles_with_attempts: set[str] = set()
    roles_with_concepts: set[str] = set()
    verified_pairs: set[tuple[str, str, str, str]] = set()
    bounds_cache: dict[tuple[str, str, str], tuple[Any, Any]] = {}

    for motif_signature in sorted(motif_links):
        quality = motif_quality.get(motif_signature, {})
        is_emergent = int(quality.get("is_emergent") or 0) == 1
        if not is_emergent and (
            int(quality.get("support_count") or 0) < 3
            or float(quality.get("motif_stability_score") or 0.0) < 0.50
        ):
            continue
        links = motif_links[motif_signature]
        roles = sorted(
            set(links.get("role", set()))
            | set(links.get("motif_associated_with_role", set()))
        )
        if not roles:
            continue
        provenance = fo._resolve_motif_transfer_provenance(
            state_conn,
            motif_signature=motif_signature,
            motif_links=links,
            direct_interaction_ids=set(
                fo._coerce_list_json(quality.get("source_interaction_ids_json"))
            ),
        )
        motif_status = str(provenance.get("status") or "missing")
        for role_signature in roles:
            roles_seen.add(role_signature)
            scope_groups = grouped.get(role_signature, {})
            if scope_groups:
                roles_with_attempts.add(role_signature)
            resolutions = concept_resolutions.get(
                role_signature,
                [{
                    "concept_signature": "__none__",
                    "mode": "missing",
                    "path": "unresolved",
                    "shared_carrier_count": 0,
                    "shared_family_count": 0,
                    "status": "missing",
                }],
            )
            if any(str(item.get("concept_signature")) != "__none__" for item in resolutions):
                roles_with_concepts.add(role_signature)
            for resolution in resolutions:
                concept_signature = str(resolution.get("concept_signature") or "__none__")
                concept_status = (
                    str(resolution.get("status") or "missing")
                    if concept_signature != "__none__"
                    else "missing"
                )
                for scope, rows in sorted(scope_groups.items()):
                    if not rows:
                        continue
                    representative = rows[0]
                    successes = sum(int(row.get("reuse_success") or 0) == 1 for row in rows)
                    strong = sum(
                        int(row.get("reuse_success") or 0) == 1
                        and int(row.get("source_evidence_support_count") or 0) >= fo.MIN_SOURCE_EVIDENCE_SUPPORT
                        and int(row.get("candidate_role_count") or 0) >= 2
                        and float(row.get("similarity_score") or 0.0) >= 0.60
                        and float(row.get("best_margin") or 0.0) >= 0.10
                        for row in rows
                    )
                    promoted_count = int(concept_signature in promoted)
                    all_verified = bool(rows) and all(
                        str(row.get("provenance_status") or "") == "verified" for row in rows
                    )
                    transfer_status = (
                        "resolved_with_surrogate"
                        if scope == "surrogate_resolved"
                        else "verified"
                        if all_verified and scope in {"cross_game", "cross_context", "cross_game_and_context"}
                        else "proxy"
                    )
                    key = (motif_signature, role_signature, concept_signature)
                    bounds = bounds_cache.get(key)
                    if bounds is None:
                        bounds = (
                            fo._safe_min(state_conn, motif_signature, role_signature, concept_signature, "first"),
                            fo._safe_min(state_conn, motif_signature, role_signature, concept_signature, "last"),
                        )
                        bounds_cache[key] = bounds
                    first_seen, last_seen = bounds
                    mean_score = fo._mean([row.get("transfer_score") for row in rows])
                    mean_margin = fo._mean(
                        [row.get("best_margin") for row in rows if row.get("best_margin") is not None]
                    )
                    state_conn.execute(
                        """
                        INSERT INTO future_option_transfer_links (
                            motif_signature, role_signature, concept_signature,
                            transfer_attempt_count, successful_transfer_count,
                            strong_transfer_success_count, promoted_concept_count,
                            mean_transfer_score, mean_best_margin, source_role_signature,
                            source_game_key, target_game_key, source_context_key,
                            target_context_key, source_interaction_id, target_interaction_id,
                            source_game_is_surrogate, target_game_is_surrogate,
                            source_context_is_surrogate, target_context_is_surrogate,
                            source_game_resolution_source, target_game_resolution_source,
                            source_context_resolution_source, target_context_resolution_source,
                            transfer_scope, provenance_mode, motif_provenance_status,
                            transfer_provenance_status, concept_validation_status,
                            motif_provenance_resolution_path, motif_resolved_interaction_count,
                            motif_resolved_family_count, motif_resolved_carrier_count,
                            motif_resolved_role_count, motif_resolved_concept_count,
                            concept_resolution_mode, concept_resolution_path,
                            shared_carrier_count, shared_family_count,
                            first_seen_global_step, last_seen_global_step
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            motif_signature, role_signature, concept_signature,
                            len(rows), successes, strong, promoted_count,
                            mean_score, mean_margin, role_signature,
                            representative.get("source_game_key"), representative.get("target_game_key"),
                            representative.get("source_context_key"), representative.get("target_context_key"),
                            representative.get("source_interaction_id"), representative.get("target_interaction_id"),
                            int(representative.get("source_game_is_surrogate") or 0),
                            int(representative.get("target_game_is_surrogate") or 0),
                            int(representative.get("source_context_is_surrogate") or 0),
                            int(representative.get("target_context_is_surrogate") or 0),
                            representative.get("source_game_resolution_source"), representative.get("target_game_resolution_source"),
                            representative.get("source_context_resolution_source"), representative.get("target_context_resolution_source"),
                            scope, "single_source", motif_status, transfer_status, concept_status,
                            provenance.get("resolution_path"), len(provenance.get("interaction_ids", ())),
                            len(provenance.get("family_ids", ())), len(provenance.get("carrier_ids", ())),
                            len(provenance.get("role_ids", ())), len(provenance.get("concept_ids", ())),
                            resolution.get("mode"), resolution.get("path"),
                            int(resolution.get("shared_carrier_count") or 0), int(resolution.get("shared_family_count") or 0),
                            first_seen, last_seen,
                        ),
                    )
                    inserted += 1
                    aggregate_attempts += len(rows)
                    aggregate_successes += successes
                    aggregate_strong += strong
                    if len(rows) > 0:
                        motifs_with_transfer.add(motif_signature)
                    if strong > 0:
                        motifs_with_strong.add(motif_signature)
                    if promoted_count > 0:
                        motifs_with_promoted.add(motif_signature)
                    fully_verified = (
                        motif_status == "verified"
                        and transfer_status == "verified"
                        and concept_status == "verified"
                    )
                    if fully_verified:
                        verified_chain_count += 1
                        if len(rows) > 0:
                            verified_motifs_with_transfer.add(motif_signature)
                        if strong > 0:
                            verified_motifs_with_strong.add(motif_signature)
                        if promoted_count > 0:
                            verified_motifs_with_promoted.add(motif_signature)
                    if is_emergent:
                        emergent_chain_count += 1
                    if transfer_status == "verified":
                        for row in rows:
                            verified_pairs.add(
                                tuple(
                                    str(row.get(name) or "")
                                    for name in (
                                        "source_game_key", "target_game_key",
                                        "source_context_key", "target_context_key",
                                    )
                                )
                            )

    elapsed = time.perf_counter() - started
    return {
        "future_option_transfer_link_count": inserted,
        "future_option_transfer_compaction_applied": True,
        "future_option_transfer_raw_attempt_count": attempt_count,
        "future_option_transfer_scope_group_count": inserted,
        "future_option_transfer_compaction_ratio": float(inserted / max(1, attempt_count * max(1, motif_count))),
        "all_motifs_with_transfer_count": len(motifs_with_transfer),
        "verified_motifs_with_transfer_count": len(verified_motifs_with_transfer),
        "all_motifs_with_strong_transfer_count": len(motifs_with_strong),
        "verified_motifs_with_strong_transfer_count": len(verified_motifs_with_strong),
        "all_motifs_with_promoted_concept_count": len(motifs_with_promoted),
        "verified_motifs_with_promoted_concept_count": len(verified_motifs_with_promoted),
        "motifs_with_transfer_count": len(motifs_with_transfer),
        "motifs_with_strong_transfer_count": len(motifs_with_strong),
        "motifs_with_promoted_concept_count": len(motifs_with_promoted),
        "motif_transfer_success_rate": aggregate_successes / aggregate_attempts if aggregate_attempts else None,
        "motif_strong_transfer_success_rate": aggregate_strong / aggregate_attempts if aggregate_attempts else None,
        "verified_concrete_transfer_link_count": verified_chain_count,
        "verified_transfer_pair_count": len(verified_pairs),
        "distinct_source_target_pair_count": len(verified_pairs),
        "emergent_future_option_transfer_link_count": emergent_chain_count,
        "emergent_motif_transfer_link_count": emergent_chain_count,
        "all_emergent_motif_transfer_link_count": emergent_chain_count,
        "roles_seen_from_motif_links": len(roles_seen),
        "roles_with_transfer_attempts": len(roles_with_attempts),
        "roles_with_concepts": len(roles_with_concepts),
        "unique_roles_seen_from_motif_links": len(roles_seen),
        "unique_roles_with_transfer_attempts": len(roles_with_attempts),
        "unique_roles_with_concepts": len(roles_with_concepts),
        "fully_verified_emergent_chain_count": verified_chain_count,
        "partially_verified_emergent_chain_count": max(0, emergent_chain_count - verified_chain_count),
        "unverified_emergent_chain_count": 0,
        "derive_future_option_transfer_links_seconds": elapsed,
    }
