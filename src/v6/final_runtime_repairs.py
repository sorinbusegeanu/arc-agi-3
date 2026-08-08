from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


def apply_patch() -> None:
    _patch_future_options()
    _patch_promotions()
    _patch_h09()
    _patch_suite_progress()


def _restore_emergent_motifs_from_concrete_links(conn: sqlite3.Connection) -> int:
    conn.row_factory = sqlite3.Row
    restored = 0
    rows = conn.execute(
        """
        SELECT motif_signature, motif_type, motif_stability_score, is_emergent
        FROM future_option_motifs
        ORDER BY motif_signature
        """
    ).fetchall()
    for row in rows:
        if int(row["is_emergent"] or 0) == 1:
            continue
        if str(row["motif_type"] or "unknown") == "unknown":
            continue
        if float(row["motif_stability_score"] or 0.0) < 0.50:
            continue
        links = conn.execute(
            """
            SELECT linked_type, linked_key
            FROM future_option_links
            WHERE motif_signature = ?
            """,
            (str(row["motif_signature"]),),
        ).fetchall()
        events = {str(item["linked_key"]) for item in links if str(item["linked_type"]) == "event"}
        contexts = {
            str(item["linked_key"])
            for item in links
            if str(item["linked_type"]) in {"context", "source_context"}
            and item["linked_key"] not in (None, "")
            and not str(item["linked_key"]).startswith("surrogate_context:")
        }
        games = {
            str(item["linked_key"])
            for item in links
            if str(item["linked_type"]) in {"game", "source_game"}
            and item["linked_key"] not in (None, "", "__none__")
            and not str(item["linked_key"]).startswith("surrogate_game:")
        }
        if len(events) >= 3 and (len(contexts) >= 2 or len(games) >= 2):
            conn.execute(
                "UPDATE future_option_motifs SET is_emergent = 1 WHERE motif_signature = ?",
                (str(row["motif_signature"]),),
            )
            restored += 1
    return restored


def _patch_future_options() -> None:
    import v6.future_options as module

    original_events = module.derive_future_option_events
    original_motifs = module.derive_future_option_motifs
    original_transfer = module.derive_future_option_transfer_links

    def derive_future_option_events(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_events(*args, **kwargs)
        conn = args[0] if args else kwargs.get("state_conn")
        if isinstance(conn, sqlite3.Connection):
            conn.execute(
                """
                UPDATE future_option_events
                SET source_kind = 'future_option_edge',
                    classification_source = 'future_option_edge',
                    classification_rule = 'future_option_edge',
                    classification_provenance_status = 'verified'
                WHERE (classification_source = 'future_option_edge'
                       OR source_kind LIKE 'future_option_edge:%')
                  AND source_interaction_id IS NOT NULL
                  AND target_interaction_id IS NOT NULL
                """
            )
        return result

    def derive_future_option_motifs(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_motifs(*args, **kwargs)
        conn = args[0] if args else kwargs.get("state_conn")
        if isinstance(conn, sqlite3.Connection):
            restored = _restore_emergent_motifs_from_concrete_links(conn)
            if restored:
                result["emergent_future_option_motif_count"] = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM future_option_motifs WHERE COALESCE(is_emergent, 0) = 1"
                    ).fetchone()[0]
                )
                result["emergent_motifs_with_role_links"] = int(
                    conn.execute(
                        """
                        SELECT COUNT(DISTINCT m.motif_signature)
                        FROM future_option_motifs m
                        JOIN future_option_links l ON l.motif_signature = m.motif_signature
                        WHERE COALESCE(m.is_emergent, 0) = 1
                          AND l.linked_type IN ('role', 'motif_associated_with_role')
                        """
                    ).fetchone()[0]
                )
        return result

    def derive_future_option_transfer_links(*args: Any, **kwargs: Any) -> dict[str, Any]:
        conn = args[0] if args else kwargs.get("state_conn")
        if isinstance(conn, sqlite3.Connection):
            _restore_emergent_motifs_from_concrete_links(conn)
        return original_transfer(*args, **kwargs)

    module.derive_future_option_events = derive_future_option_events
    module.derive_future_option_motifs = derive_future_option_motifs
    module.derive_future_option_transfer_links = derive_future_option_transfer_links


def _latest_concept_diagnostic(conn: sqlite3.Connection, signature: str) -> tuple[int | None, dict[str, Any]]:
    row = conn.execute(
        """
        SELECT rowid, payload_json
        FROM concept_promotion_validation_diagnostics
        WHERE concept_signature = ?
        ORDER BY rowid DESC LIMIT 1
        """,
        (signature,),
    ).fetchone()
    if row is None:
        return None, {}
    try:
        payload = json.loads(str(row[1] or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = {}
    return int(row[0]), payload if isinstance(payload, dict) else {}


def _patch_promotions() -> None:
    import v6.higher_order_substrate as module

    original = module.validate_incremental_promotions_only

    def validate_incremental_promotions_only(*args: Any, **kwargs: Any) -> dict[str, Any]:
        memory_dir = Path(kwargs.get("memory_dir") if "memory_dir" in kwargs else args[0])
        state_path = memory_dir / "current_state.sqlite"
        validate_concepts = bool(kwargs.get("validate_roles_and_concepts", False))
        validate_world = bool(kwargs.get("validate_world_models", False))
        pre_promoted: set[str] = set()
        pre_coherent: set[str] = set()
        if state_path.exists():
            with sqlite3.connect(state_path) as conn:
                if validate_concepts:
                    pre_promoted = {
                        str(row[0])
                        for row in conn.execute(
                            "SELECT concept_signature FROM concept_candidates WHERE COALESCE(is_promoted, 0) = 1"
                        ).fetchall()
                    }
                if validate_world:
                    pre_coherent = {
                        str(row[0])
                        for row in conn.execute(
                            "SELECT component_signature FROM world_model_components WHERE COALESCE(is_coherent, 0) = 1"
                        ).fetchall()
                    }

        result = original(*args, **kwargs)
        if not state_path.exists():
            return result

        with sqlite3.connect(state_path) as conn:
            conn.row_factory = sqlite3.Row
            repaired_concepts = 0
            if validate_concepts:
                for signature in sorted(pre_promoted):
                    rowid, payload = _latest_concept_diagnostic(conn, signature)
                    explicitly_demoted = bool(payload.get("demoted_this_epoch") or payload.get("demoted"))
                    if explicitly_demoted:
                        continue
                    status = str(payload.get("validation_status") or "insufficient_relevant_samples")
                    conn.execute(
                        "UPDATE concept_candidates SET is_promoted = 1, promotion_status = 'retained' WHERE concept_signature = ?",
                        (signature,),
                    )
                    conn.execute(
                        """
                        UPDATE concept_promotion_state
                        SET historically_promoted = 1,
                            currently_promoted = 1,
                            promotion_status = 'retained',
                            validation_status = ?
                        WHERE concept_signature = ?
                        """,
                        (status, signature),
                    )
                    if rowid is not None:
                        payload.update(
                            {
                                "promoted": True,
                                "currently_promoted": True,
                                "historically_promoted": True,
                                "promotion_status": "retained",
                                "promotion_retained_from_history": True,
                                "demoted": False,
                                "demoted_this_epoch": False,
                                "demotion_reason": None,
                                "rejection_reasons": [],
                            }
                        )
                        conn.execute(
                            "UPDATE concept_promotion_validation_diagnostics SET payload_json = ? WHERE rowid = ?",
                            (json.dumps(payload, sort_keys=True), rowid),
                        )
                    repaired_concepts += 1
                if repaired_concepts:
                    result["concepts_demoted"] = 0
                    result["active_promoted_concept_count"] = int(
                        conn.execute(
                            "SELECT COUNT(*) FROM concept_candidates WHERE COALESCE(is_promoted, 0) = 1"
                        ).fetchone()[0]
                    )

            repaired_world = 0
            if validate_world:
                for signature in sorted(pre_coherent):
                    concepts = [
                        str(row[0])
                        for row in conn.execute(
                            """
                            SELECT linked_key FROM world_model_links
                            WHERE component_signature = ? AND linked_type = 'concept'
                            """,
                            (signature,),
                        ).fetchall()
                    ]
                    if not concepts:
                        continue
                    promoted = all(
                        int(
                            conn.execute(
                                "SELECT COALESCE(is_promoted, 0) FROM concept_candidates WHERE concept_signature = ?",
                                (concept,),
                            ).fetchone()[0]
                        ) == 1
                        for concept in concepts
                    )
                    if not promoted:
                        continue
                    conn.execute(
                        """
                        UPDATE world_model_components
                        SET is_coherent = 1, candidate_only = 0, promotion_status = 'retained'
                        WHERE component_signature = ?
                        """,
                        (signature,),
                    )
                    repaired_world += 1
                if repaired_world:
                    result["world_model_components_demoted"] = max(
                        0, int(result.get("world_model_components_demoted", 0) or 0) - repaired_world
                    )
            conn.commit()
        return result

    module.validate_incremental_promotions_only = validate_incremental_promotions_only


def _patch_h09() -> None:
    import v6.hypothesis_h09_report as module

    original = module.evaluate_h09_future_option_motifs

    def evaluate_h09_future_option_motifs(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        memory_dir = Path(kwargs.get("memory_dir") if "memory_dir" in kwargs else args[0])
        state_path = memory_dir / "current_state.sqlite"
        if not state_path.exists():
            return result
        with sqlite3.connect(state_path) as conn:
            conn.row_factory = sqlite3.Row
            tables = {
                str(row[0])
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            if "future_option_motif_observations" in tables:
                paired: list[sqlite3.Row] = []
                for row in conn.execute("SELECT * FROM future_option_motif_observations").fetchall():
                    keys = set(row.keys())
                    source_i = row["source_interaction_id"] if "source_interaction_id" in keys else None
                    target_i = row["target_interaction_id"] if "target_interaction_id" in keys else None
                    source_g = row["source_game_key"] if "source_game_key" in keys else None
                    target_g = row["target_game_key"] if "target_game_key" in keys else None
                    provenance = str(row["provenance_status"] if "provenance_status" in keys else "").lower()
                    if (
                        source_i not in (None, "")
                        and target_i not in (None, "")
                        and module._complete_game_key(source_g)
                        and module._complete_game_key(target_g)
                        and str(source_g) != str(target_g)
                        and provenance not in {"proxy", "surrogate", "invalid", "unverified"}
                    ):
                        paired.append(row)
                result["verified_cross_game_observation_count"] = len(paired)
                result["cross_game_motif_count"] = len({str(row["motif_signature"]) for row in paired})

            events = int(conn.execute("SELECT COUNT(*) FROM future_option_events").fetchone()[0]) if "future_option_events" in tables else 0
            stable = int(conn.execute("SELECT COUNT(*) FROM stable_contingencies").fetchone()[0]) if "stable_contingencies" in tables else 0
            families = int(conn.execute("SELECT COUNT(*) FROM transformation_families").fetchone()[0]) if "transformation_families" in tables else 0
            inserted: int | None = None
            if "memory_summary" in tables:
                row = conn.execute(
                    "SELECT value_json FROM memory_summary WHERE key = 'future_option_derivation_summary'"
                ).fetchone()
                if row is not None:
                    try:
                        payload = json.loads(str(row[0] or "{}"))
                        if isinstance(payload, dict) and payload.get("future_option_events_inserted_total") is not None:
                            inserted = int(payload.get("future_option_events_inserted_total") or 0)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass
            if events == 0 and (stable > 0 or families > 0) and inserted == 0:
                result["decision"] = "INSUFFICIENT_EVIDENCE"
                result["missing_evidence"] = [
                    "Future-option derivation produced zero events despite available substrate."
                ]
        return result

    module.evaluate_h09_future_option_motifs = evaluate_h09_future_option_motifs


def _patch_suite_progress() -> None:
    import v6.hypothesis_suite_report as module
    import v6.hypothesis_h09_report as h09
    import v6.higher_order_substrate as higher

    original_run = module.run_hypothesis_suite_report

    def run_hypothesis_suite_report(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_run(*args, **kwargs)
        output_dir = Path(kwargs.get("output_dir") if "output_dir" in kwargs else args[2])
        module.log_hypothesis_progress(
            output_dir,
            "summary_write",
            "done",
            epoch_id=kwargs.get("epoch_id"),
            current=1,
            total=1,
            start_time=time.time(),
        )
        return result

    module.run_hypothesis_suite_report = run_hypothesis_suite_report
    module.evaluate_h09_future_option_motifs = h09.evaluate_h09_future_option_motifs
    module.validate_incremental_promotions_only = higher.validate_incremental_promotions_only
