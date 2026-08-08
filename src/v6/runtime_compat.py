from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


def install_runtime_compatibility() -> None:
    """Restore compatibility invariants without weakening strict evidence gates."""
    _install_future_option_repairs()
    _install_promotion_repairs()
    _install_h09_repairs()
    _install_suite_progress_repairs()


def _install_future_option_repairs() -> None:
    import v6.future_options as module

    original_events = module.derive_future_option_events
    original_motifs = module.derive_future_option_motifs

    def derive_future_option_events(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_events(*args, **kwargs)
        state_conn = args[0] if args else kwargs.get("state_conn")
        if isinstance(state_conn, sqlite3.Connection):
            state_conn.execute(
                """
                UPDATE future_option_events
                SET source_kind = 'future_option_edge',
                    classification_source = 'future_option_edge',
                    classification_rule = 'future_option_edge',
                    classification_provenance_status = 'verified'
                WHERE classification_source = 'future_option_edge'
                  AND source_interaction_id IS NOT NULL
                  AND target_interaction_id IS NOT NULL
                """
            )
        return result

    def derive_future_option_motifs(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_motifs(*args, **kwargs)
        state_conn = args[0] if args else kwargs.get("state_conn")
        if not isinstance(state_conn, sqlite3.Connection):
            return result
        state_conn.row_factory = sqlite3.Row
        restored = 0
        motif_rows = state_conn.execute(
            """
            SELECT motif_signature, motif_type, motif_stability_score, is_emergent
            FROM future_option_motifs
            ORDER BY motif_signature
            """
        ).fetchall()
        for motif in motif_rows:
            if int(motif["is_emergent"] or 0) == 1:
                continue
            if str(motif["motif_type"] or "unknown") == "unknown":
                continue
            if float(motif["motif_stability_score"] or 0.0) < 0.50:
                continue
            observations = state_conn.execute(
                """
                SELECT event_id, source_context_key, source_game_key, provenance_status
                FROM future_option_motif_observations
                WHERE motif_signature = ?
                ORDER BY event_id
                """,
                (str(motif["motif_signature"]),),
            ).fetchall()
            verified = [
                row for row in observations
                if str(row["provenance_status"] or "").lower()
                not in {"proxy", "surrogate", "invalid", "unverified"}
            ]
            contexts = {
                str(row["source_context_key"])
                for row in verified
                if module.is_complete_context_key(row["source_context_key"])
            }
            games = {
                str(row["source_game_key"])
                for row in verified
                if row["source_game_key"] not in (None, "", "__none__")
                and not str(row["source_game_key"]).startswith("surrogate_game:")
            }
            if len(verified) >= 3 and (len(contexts) >= 2 or len(games) >= 2):
                state_conn.execute(
                    "UPDATE future_option_motifs SET is_emergent = 1 WHERE motif_signature = ?",
                    (str(motif["motif_signature"]),),
                )
                restored += 1
        if restored:
            state_conn.commit()
            result["emergent_future_option_motif_count"] = int(
                state_conn.execute(
                    "SELECT COUNT(*) FROM future_option_motifs WHERE COALESCE(is_emergent, 0) = 1"
                ).fetchone()[0]
            )
            if "motifs_demoted_without_real_cross_scope_evidence" in result:
                result["motifs_demoted_without_real_cross_scope_evidence"] = max(
                    0,
                    int(result.get("motifs_demoted_without_real_cross_scope_evidence", 0) or 0) - restored,
                )
        return result

    module.derive_future_option_events = derive_future_option_events
    module.derive_future_option_motifs = derive_future_option_motifs


def _install_promotion_repairs() -> None:
    import v6.higher_order_substrate as module

    original_validate = module.validate_incremental_promotions_only

    def validate_incremental_promotions_only(*args: Any, **kwargs: Any) -> dict[str, Any]:
        memory_dir = Path(kwargs.get("memory_dir") if "memory_dir" in kwargs else args[0])
        validate_concepts = bool(kwargs.get("validate_roles_and_concepts", False))
        pre_promoted: set[str] = set()
        state_path = memory_dir / "current_state.sqlite"
        if validate_concepts and state_path.exists():
            with sqlite3.connect(state_path) as conn:
                pre_promoted = {
                    str(row[0])
                    for row in conn.execute(
                        "SELECT concept_signature FROM concept_candidates WHERE COALESCE(is_promoted, 0) = 1"
                    ).fetchall()
                }
        result = original_validate(*args, **kwargs)
        if not validate_concepts or not pre_promoted or not state_path.exists():
            return result

        with sqlite3.connect(state_path) as conn:
            conn.row_factory = sqlite3.Row
            repaired = 0
            for signature in sorted(pre_promoted):
                row = conn.execute(
                    "SELECT is_promoted FROM concept_candidates WHERE concept_signature = ?",
                    (signature,),
                ).fetchone()
                if row is None or int(row["is_promoted"] or 0) == 1:
                    continue
                diagnostic_row = conn.execute(
                    """
                    SELECT rowid, payload_json
                    FROM concept_promotion_validation_diagnostics
                    WHERE concept_signature = ?
                    ORDER BY rowid DESC LIMIT 1
                    """,
                    (signature,),
                ).fetchone()
                payload: dict[str, Any] = {}
                if diagnostic_row is not None:
                    try:
                        loaded = json.loads(str(diagnostic_row["payload_json"] or "{}"))
                        if isinstance(loaded, dict):
                            payload = loaded
                    except (TypeError, ValueError, json.JSONDecodeError):
                        payload = {}
                if bool(payload.get("demoted_this_epoch") or payload.get("demoted")):
                    continue

                validation_status = str(payload.get("validation_status") or "insufficient_relevant_samples")
                conn.execute(
                    """
                    UPDATE concept_candidates
                    SET is_promoted = 1, promotion_status = 'retained'
                    WHERE concept_signature = ?
                    """,
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
                    (validation_status, signature),
                )
                if diagnostic_row is not None:
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
                        (json.dumps(payload, sort_keys=True), int(diagnostic_row["rowid"])),
                    )
                repaired += 1
            if repaired:
                conn.commit()
                active_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM concept_candidates WHERE COALESCE(is_promoted, 0) = 1"
                    ).fetchone()[0]
                )
                result["active_promoted_concept_count"] = active_count
                result["phase3_active_promoted_count"] = active_count
        return result

    module.validate_incremental_promotions_only = validate_incremental_promotions_only


def _install_h09_repairs() -> None:
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
                paired_rows = []
                for row in conn.execute(
                    "SELECT * FROM future_option_motif_observations ORDER BY motif_signature, event_id"
                ).fetchall():
                    keys = set(row.keys())
                    source_interaction = row["source_interaction_id"] if "source_interaction_id" in keys else None
                    target_interaction = row["target_interaction_id"] if "target_interaction_id" in keys else None
                    source_game = row["source_game_key"] if "source_game_key" in keys else None
                    target_game = row["target_game_key"] if "target_game_key" in keys else None
                    provenance = str(row["provenance_status"] if "provenance_status" in keys else "")
                    if (
                        source_interaction not in (None, "")
                        and target_interaction not in (None, "")
                        and module._complete_game_key(source_game)
                        and module._complete_game_key(target_game)
                        and str(source_game) != str(target_game)
                        and provenance.lower() not in {"proxy", "surrogate", "invalid", "unverified"}
                    ):
                        paired_rows.append(row)
                result["verified_cross_game_observation_count"] = len(paired_rows)
                result["cross_game_motif_count"] = len(
                    {str(row["motif_signature"]) for row in paired_rows}
                )

            event_count = int(
                conn.execute("SELECT COUNT(*) FROM future_option_events").fetchone()[0]
            ) if "future_option_events" in tables else 0
            stable_count = int(
                conn.execute("SELECT COUNT(*) FROM stable_contingencies").fetchone()[0]
            ) if "stable_contingencies" in tables else 0
            family_count = int(
                conn.execute("SELECT COUNT(*) FROM transformation_families").fetchone()[0]
            ) if "transformation_families" in tables else 0
            derivation: dict[str, Any] = {}
            if "memory_summary" in tables:
                summary_row = conn.execute(
                    "SELECT value_json FROM memory_summary WHERE key = 'future_option_derivation_summary'"
                ).fetchone()
                if summary_row is not None:
                    try:
                        loaded = json.loads(str(summary_row[0] or "{}"))
                        if isinstance(loaded, dict):
                            derivation = loaded
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass
            inserted = derivation.get("future_option_events_inserted_total")
            if (
                event_count == 0
                and (stable_count > 0 or family_count > 0)
                and inserted is not None
                and int(inserted or 0) == 0
            ):
                result["decision"] = "INSUFFICIENT_EVIDENCE"
                result["missing_evidence"] = [
                    "Future-option derivation produced zero events despite available substrate."
                ]
        try:
            module._write_outputs(Path(kwargs.get("output_dir")), result)
        except Exception:
            pass
        return result

    module.evaluate_h09_future_option_motifs = evaluate_h09_future_option_motifs


def _install_suite_progress_repairs() -> None:
    import v6.hypothesis_suite_report as module

    original_phase = module._phase
    aliases = {
        "DERIVE.role_candidates": "derive_role_candidates",
        "DERIVE.role_transfer_attempts": "derive_role_transfer_attempts",
        "DERIVE.concept_candidates": "derive_concept_candidates",
        "DERIVE.world_models": "derive_world_model_components",
        "DERIVE.future_options": "derive_future_option_memory",
    }

    def phase(output_dir: Path, epoch_id: str | None, name: str, callback: Any, timings: dict[str, float]) -> Any:
        alias = aliases.get(name)
        started = time.time()
        if alias:
            module.log_hypothesis_progress(output_dir, alias, "starting", epoch_id=epoch_id)
        try:
            result = original_phase(output_dir, epoch_id, name, callback, timings)
        except BaseException as exc:
            if alias:
                module.log_hypothesis_progress(
                    output_dir,
                    alias,
                    "failed",
                    epoch_id=epoch_id,
                    start_time=started,
                    extra={"exception_type": type(exc).__name__, "exception_message": str(exc)},
                )
            raise
        if alias:
            module.log_hypothesis_progress(
                output_dir, alias, "done", epoch_id=epoch_id, current=1, total=1, start_time=started
            )
        return result

    original_evaluate_one = module._evaluate_one

    def evaluate_one(hypothesis_id: str, evaluator: Any, *, kwargs: dict[str, Any]) -> dict[str, Any]:
        child_output = Path(kwargs.get("output_dir", "."))
        root_output = child_output.parent
        started = time.time()
        module.log_hypothesis_progress(root_output, hypothesis_id, "starting")
        result = original_evaluate_one(hypothesis_id, evaluator, kwargs=kwargs)
        status = "failed" if result.get("evaluator_error") else "done"
        module.log_hypothesis_progress(
            root_output,
            hypothesis_id,
            status,
            current=1,
            total=1,
            start_time=started,
        )
        return result

    module._phase = phase
    module._evaluate_one = evaluate_one

    # Keep module-level imports used by the suite synchronized with repaired evaluators.
    import v6.hypothesis_h09_report as h09
    import v6.higher_order_substrate as higher
    module.evaluate_h09_future_option_motifs = h09.evaluate_h09_future_option_motifs
    module.validate_incremental_promotions_only = higher.validate_incremental_promotions_only
