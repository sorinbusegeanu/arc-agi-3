from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.future_options import derive_future_option_memory


def _missing_tables(connection: sqlite3.Connection, required: tuple[str, ...]) -> list[str]:
    tables = {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    return [name for name in required if name not in tables]


def evaluate_h11_future_option_transfer_concepts(
    *,
    memory_dir: Path,
    run_dir: Path | None,
    output_dir: Path,
    already_derived: bool = False,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    current_state = Path(memory_dir) / "current_state.sqlite"
    if not already_derived and current_state.exists():
        derive_future_option_memory(memory_dir=memory_dir, run_dir=run_dir)
    if not current_state.exists():
        result = {
            "hypothesis_id": "H11",
            "evidence_source": "compact_memory",
            "decision": "INSUFFICIENT_EVIDENCE",
            "missing_evidence": [f"Missing expected compact-memory file: {current_state}"],
            "core_metrics": {},
        }
        _write(output_dir, result)
        return result
    with sqlite3.connect(current_state) as conn:
        conn.row_factory = sqlite3.Row
        missing_tables = _missing_tables(
            conn,
            (
                "future_option_transfer_links",
                "future_option_motifs",
                "role_transfer_attempts",
                "concept_candidates",
            ),
        )
        if missing_tables:
            result = {
                "hypothesis_id": "H11",
                "evidence_source": "compact_memory",
                "decision": "INSUFFICIENT_EVIDENCE",
                "missing_evidence": [f"Missing expected compact-memory table(s): {', '.join(missing_tables)}"],
                "core_metrics": {},
            }
            _write(output_dir, result)
            return result
        rows = [dict(row) for row in conn.execute(
            """
            SELECT l.*, m.is_emergent, m.motif_type, m.motif_stability_score
            FROM future_option_transfer_links l
            LEFT JOIN future_option_motifs m
              ON l.motif_signature = m.motif_signature
            ORDER BY l.motif_signature ASC, l.role_signature ASC, l.concept_signature ASC
            """
        ).fetchall()]
        emergent_motifs = int(conn.execute("SELECT COUNT(*) FROM future_option_motifs WHERE COALESCE(is_emergent, 0) = 1").fetchone()[0])
        future_option_motif_count = int(conn.execute("SELECT COUNT(*) FROM future_option_motifs").fetchone()[0])
        successful_role_transfer_count = int(conn.execute(
            "SELECT COUNT(*) FROM role_transfer_attempts WHERE COALESCE(reuse_success, 0) = 1 AND provenance_mode = 'single_source'"
        ).fetchone()[0])
        promoted_concept_count = int(conn.execute("SELECT COUNT(*) FROM concept_candidates WHERE COALESCE(is_promoted, 0) = 1").fetchone()[0])
        summary_row = conn.execute(
            "SELECT value_json FROM memory_summary WHERE key = 'future_option_derivation_summary'"
        ).fetchone()
        derivation_summary = json.loads(str(summary_row[0])) if summary_row and summary_row[0] else {}
    fully_verified_rows = [
        row for row in rows
        if str(row.get("motif_provenance_status") or "missing") == "verified"
        and str(row.get("transfer_provenance_status") or "missing") == "verified"
        and str(row.get("concept_validation_status") or "missing") == "verified"
    ]
    for row in rows:
        if str(row.get("transfer_provenance_status") or "missing") == "verified":
            assert str(row.get("provenance_mode") or "") == "single_source"
            assert row.get("source_role_signature")
            assert row.get("source_game_key") and row.get("target_game_key")
            assert row.get("source_context_key") and row.get("target_context_key")
    all_motifs_with_transfer = len({str(row["motif_signature"]) for row in rows if int(row["transfer_attempt_count"] or 0) > 0})
    verified_motifs_with_transfer = len({str(row["motif_signature"]) for row in fully_verified_rows if int(row["transfer_attempt_count"] or 0) > 0})
    all_motifs_with_strong = len({str(row["motif_signature"]) for row in rows if int(row["strong_transfer_success_count"] or 0) > 0})
    verified_motifs_with_strong = len({str(row["motif_signature"]) for row in fully_verified_rows if int(row["strong_transfer_success_count"] or 0) > 0})
    all_motifs_with_promoted = len({str(row["motif_signature"]) for row in rows if int(row["promoted_concept_count"] or 0) > 0})
    verified_motifs_with_promoted = len({str(row["motif_signature"]) for row in fully_verified_rows if int(row["promoted_concept_count"] or 0) > 0})
    emergent_all_rows = [row for row in rows if int(row.get("is_emergent") or 0) == 1]
    emergent_rows = [row for row in fully_verified_rows if int(row.get("is_emergent") or 0) == 1]
    non_emergent_rows = [row for row in rows if int(row.get("is_emergent") or 0) != 1]
    emergent_motifs_with_transfer = len({str(row["motif_signature"]) for row in emergent_rows if int(row["transfer_attempt_count"] or 0) > 0})
    emergent_motifs_with_strong = len({str(row["motif_signature"]) for row in emergent_rows if int(row["strong_transfer_success_count"] or 0) > 0})
    emergent_motifs_with_promoted = len({str(row["motif_signature"]) for row in emergent_rows if int(row["promoted_concept_count"] or 0) > 0})
    non_emergent_motifs_with_strong = len({str(row["motif_signature"]) for row in non_emergent_rows if int(row["strong_transfer_success_count"] or 0) > 0})
    non_emergent_motifs_with_promoted = len({str(row["motif_signature"]) for row in non_emergent_rows if int(row["promoted_concept_count"] or 0) > 0})
    total_attempts = sum(int(row["transfer_attempt_count"] or 0) for row in fully_verified_rows)
    total_successes = sum(int(row["successful_transfer_count"] or 0) for row in fully_verified_rows)
    total_strong = sum(int(row["strong_transfer_success_count"] or 0) for row in fully_verified_rows)
    emergent_attempts = sum(int(row["transfer_attempt_count"] or 0) for row in emergent_rows)
    emergent_successes = sum(int(row["successful_transfer_count"] or 0) for row in emergent_rows)
    emergent_strong = sum(int(row["strong_transfer_success_count"] or 0) for row in emergent_rows)
    verified_cross_game_rows = [
        row for row in fully_verified_rows
        if str(row.get("provenance_mode") or "") == "single_source"
        and row.get("source_role_signature")
        and row.get("source_game_key") and row.get("target_game_key")
        and row.get("source_context_key") and row.get("target_context_key")
        and str(row["source_game_key"]) != str(row["target_game_key"])
    ]
    unverified_cross_game_rows = [
        row for row in rows
        if str(row.get("provenance_mode") or "") != "single_source"
    ]
    chain_state_counts: dict[str, int] = {
        "verified_verified_verified": 0,
        "verified_verified_proxy": 0,
        "verified_proxy_verified": 0,
        "missing_verified_verified": 0,
        "missing_proxy_verified": 0,
        "other": 0,
    }
    blocked_by_motif_provenance = 0
    blocked_by_transfer_provenance = 0
    blocked_by_concept_validation = 0
    blocked_by_missing_concept = 0
    for row in rows:
        motif_status = str(row.get("motif_provenance_status") or "missing")
        transfer_status = str(row.get("transfer_provenance_status") or "missing")
        concept_status = str(row.get("concept_validation_status") or "missing")
        key = f"{motif_status}_{transfer_status}_{concept_status}"
        chain_state_counts[key if key in chain_state_counts else "other"] += 1
        blocked_by_motif_provenance += int(motif_status != "verified")
        blocked_by_transfer_provenance += int(transfer_status != "verified")
        blocked_by_concept_validation += int(concept_status != "verified")
        blocked_by_missing_concept += int(concept_status == "missing")
    verified_cross_game_pairs = {
        (str(row["source_game_key"]), str(row["target_game_key"]), str(row["source_context_key"]), str(row["target_context_key"]))
        for row in verified_cross_game_rows
    }
    result = {
        "hypothesis_id": "H11",
        "evidence_source": "compact_memory",
        "future_option_transfer_link_count": len(rows),
        "verified_future_option_transfer_count": len(fully_verified_rows),
        "future_option_motif_count": future_option_motif_count,
        "all_motifs_with_transfer_count": all_motifs_with_transfer,
        "verified_motifs_with_transfer_count": verified_motifs_with_transfer,
        "all_motifs_with_strong_transfer_count": all_motifs_with_strong,
        "verified_motifs_with_strong_transfer_count": verified_motifs_with_strong,
        "all_motifs_with_promoted_concept_count": all_motifs_with_promoted,
        "verified_motifs_with_promoted_concept_count": verified_motifs_with_promoted,
        # Legacy H11 counters retain their previous verified-only meaning.
        "motifs_with_transfer_count": verified_motifs_with_transfer,
        "motifs_with_strong_transfer_count": verified_motifs_with_strong,
        "motifs_with_promoted_concept_count": verified_motifs_with_promoted,
        "motif_transfer_success_rate": (total_successes / total_attempts) if total_attempts else None,
        "motif_strong_transfer_success_rate": (total_strong / total_attempts) if total_attempts else None,
        "promoted_concept_motif_count": verified_motifs_with_promoted,
        "emergent_future_option_motif_count": emergent_motifs,
        "all_emergent_motif_transfer_link_count": len(emergent_all_rows),
        "emergent_motif_transfer_link_count": len(emergent_all_rows),
        "fully_verified_emergent_chain_count": len(emergent_rows),
        "partially_verified_emergent_chain_count": sum(
            1 for row in emergent_all_rows
            if row not in fully_verified_rows
            and "missing" not in {
                str(row.get("motif_provenance_status") or "missing"),
                str(row.get("transfer_provenance_status") or "missing"),
                str(row.get("concept_validation_status") or "missing"),
            }
        ),
        "unverified_emergent_chain_count": sum(
            1 for row in emergent_all_rows
            if "missing" in {
                str(row.get("motif_provenance_status") or "missing"),
                str(row.get("transfer_provenance_status") or "missing"),
                str(row.get("concept_validation_status") or "missing"),
            }
        ),
        "emergent_motifs_with_transfer_count": emergent_motifs_with_transfer,
        "emergent_motifs_with_strong_transfer_count": emergent_motifs_with_strong,
        "emergent_motifs_with_promoted_concept_count": emergent_motifs_with_promoted,
        "emergent_motif_transfer_success_rate": (emergent_successes / emergent_attempts) if emergent_attempts else None,
        "emergent_motif_strong_transfer_success_rate": (emergent_strong / emergent_attempts) if emergent_attempts else None,
        "promoted_concept_emergent_motif_count": emergent_motifs_with_promoted,
        "non_emergent_motif_transfer_link_count": len(non_emergent_rows),
        "non_emergent_motifs_with_strong_transfer_count": non_emergent_motifs_with_strong,
        "non_emergent_motifs_with_promoted_concept_count": non_emergent_motifs_with_promoted,
        "successful_role_transfer_count": successful_role_transfer_count,
        "verified_cross_game_future_option_transfer_count": len(verified_cross_game_rows),
        "verified_cross_game_motif_transfer_count": len(verified_cross_game_rows),
        "verified_cross_game_link_count": len(verified_cross_game_rows),
        "verified_cross_game_motif_count": len({str(row["motif_signature"]) for row in verified_cross_game_rows}),
        "verified_cross_game_pair_count": len(verified_cross_game_pairs),
        "unverified_cross_game_motif_transfer_count": len(unverified_cross_game_rows),
        "verified_concrete_transfer_link_count": sum(
            1 for row in rows if str(row.get("transfer_provenance_status") or "missing") == "verified"
        ),
        "verified_transfer_pair_count": len({
            (str(row.get("source_game_key") or ""), str(row.get("target_game_key") or ""),
             str(row.get("source_context_key") or ""), str(row.get("target_context_key") or ""))
            for row in rows if str(row.get("transfer_provenance_status") or "missing") == "verified"
        }),
        "distinct_source_target_pair_count": len({
            (str(row.get("source_game_key") or ""), str(row.get("target_game_key") or ""),
             str(row.get("source_context_key") or ""), str(row.get("target_context_key") or ""))
            for row in rows if str(row.get("transfer_provenance_status") or "missing") == "verified"
        }),
        "motif_transfer_chain_provenance_breakdown": chain_state_counts,
        "blocked_by_motif_provenance": blocked_by_motif_provenance,
        "blocked_by_transfer_provenance": blocked_by_transfer_provenance,
        "blocked_by_concept_validation": blocked_by_concept_validation,
        "blocked_by_missing_concept": blocked_by_missing_concept,
        "motif_transfer_chain_provenance": [
            {
                "motif_signature": str(row["motif_signature"]),
                "role_signature": row.get("source_role_signature") or row.get("role_signature"),
                "concept_signature": row.get("concept_signature"),
                "motif_provenance_status": row.get("motif_provenance_status") or "missing",
                "transfer_provenance_status": row.get("transfer_provenance_status") or "missing",
                "concept_validation_status": row.get("concept_validation_status") or "missing",
                "motif_provenance_resolution_path": row.get("motif_provenance_resolution_path") or "unresolved",
                "source_game_key": row.get("source_game_key"),
                "target_game_key": row.get("target_game_key"),
                "source_context_key": row.get("source_context_key"),
                "target_context_key": row.get("target_context_key"),
            }
            for row in rows
        ],
        "promoted_concept_count": promoted_concept_count,
        "h11_blocked_by_no_motifs": bool(future_option_motif_count == 0),
        "h11_blocked_by_no_promoted_concepts": bool(promoted_concept_count == 0),
        "events_with_owner_type_role": derivation_summary.get("events_with_owner_type_role"),
        "role_linked_event_count": derivation_summary.get("role_linked_event_count"),
        "motifs_with_role_links": derivation_summary.get("motifs_with_role_links"),
        "emergent_motifs_with_role_links": derivation_summary.get("emergent_motifs_with_role_links"),
        "motifs_with_family_provenance": derivation_summary.get("motifs_with_family_provenance"),
        "motifs_with_carrier_provenance": derivation_summary.get("motifs_with_carrier_provenance"),
        "motifs_with_role_provenance": derivation_summary.get("motifs_with_role_provenance"),
        "motifs_with_concept_provenance": derivation_summary.get("motifs_with_concept_provenance"),
        "provenance_resolution_failures": derivation_summary.get("provenance_resolution_failures"),
        "failure_occurrence_count": derivation_summary.get("failure_occurrence_count"),
        "unique_unresolved_id_count": derivation_summary.get("unique_unresolved_id_count"),
        "unresolved_family_to_carrier_count": derivation_summary.get("unresolved_family_to_carrier_count"),
        "unresolved_carrier_to_role_count": derivation_summary.get("unresolved_carrier_to_role_count"),
        "unresolved_role_to_concept_count": derivation_summary.get("unresolved_role_to_concept_count"),
        "unique_unresolved_family_to_carrier_count": derivation_summary.get("unique_unresolved_family_to_carrier_count"),
        "unique_unresolved_carrier_to_role_count": derivation_summary.get("unique_unresolved_carrier_to_role_count"),
        "unique_unresolved_role_to_concept_count": derivation_summary.get("unique_unresolved_role_to_concept_count"),
        "motifs_seen_for_transfer": derivation_summary.get("motifs_seen_for_transfer"),
        "motifs_skipped_no_role_links": derivation_summary.get("motifs_skipped_no_role_links"),
        "motifs_skipped_insufficient_support_or_stability": derivation_summary.get("motifs_skipped_insufficient_support_or_stability"),
        "motifs_with_role_links_for_transfer": derivation_summary.get("motifs_with_role_links_for_transfer"),
        "roles_seen_from_motif_links": derivation_summary.get("roles_seen_from_motif_links"),
        "roles_with_transfer_attempts": derivation_summary.get("roles_with_transfer_attempts"),
        "roles_with_concepts": derivation_summary.get("roles_with_concepts"),
        "unique_roles_seen_from_motif_links": derivation_summary.get("unique_roles_seen_from_motif_links"),
        "unique_roles_with_transfer_attempts": derivation_summary.get("unique_roles_with_transfer_attempts"),
        "unique_roles_with_concepts": derivation_summary.get("unique_roles_with_concepts"),
        "motif_role_link_count": derivation_summary.get("motif_role_link_count"),
        "motif_role_transfer_attempt_link_count": derivation_summary.get("motif_role_transfer_attempt_link_count"),
        "motif_role_concept_link_count": derivation_summary.get("motif_role_concept_link_count"),
        "missing_evidence": [],
    }
    assert int(result["fully_verified_emergent_chain_count"] or 0) <= int(
        result["all_emergent_motif_transfer_link_count"] or 0
    )
    if future_option_motif_count == 0:
        result["decision"] = "INSUFFICIENT_EVIDENCE"
        result["missing_evidence"].append("H11 blocked because future-option motifs are absent.")
    elif emergent_motifs == 0:
        result["decision"] = "INCONCLUSIVE"
    elif successful_role_transfer_count == 0 and promoted_concept_count == 0:
        result["decision"] = "INCONCLUSIVE"
    elif len(emergent_rows) == 0 and len(rows) > 0:
        result["decision"] = "PARTIALLY_VALID"
        result["missing_evidence"].append("Transfer/concept evidence exists, but it is not attached to emergent future-option motifs.")
    elif len(rows) > 0 and verified_motifs_with_strong == 0:
        result["decision"] = "PARTIALLY_VALID"
    elif (
        len(emergent_rows) >= 5
        and emergent_motifs_with_strong >= 1
        and emergent_motifs_with_promoted >= 1
        and (result["emergent_motif_strong_transfer_success_rate"] or 0.0) > 0.0
    ):
        result["decision"] = "VALID"
    elif successful_role_transfer_count > 0 and promoted_concept_count > 0 and len(rows) == 0:
        result["decision"] = "INVALID"
    else:
        result["decision"] = "PARTIALLY_VALID"
    if rows and not fully_verified_rows:
        result["decision"] = "INSUFFICIENT_EVIDENCE"
        result["missing_evidence"].append("No motif-transfer-concept chain has fully verified motif, transfer, and concept provenance.")
    if len(emergent_rows) == 0 and len(rows) > 0 and "Transfer/concept evidence exists, but it is not attached to emergent future-option motifs." not in result["missing_evidence"]:
        result["missing_evidence"].append("Transfer/concept evidence exists, but it is not attached to emergent future-option motifs.")
    elif len(emergent_rows) == 0 and "No transfer/concept links are attached to emergent future-option motifs." not in result["missing_evidence"]:
        result["missing_evidence"].append("No transfer/concept links are attached to emergent future-option motifs.")
    if promoted_concept_count == 0 and "No promoted concepts available for motif-concept linkage." not in result["missing_evidence"]:
        result["missing_evidence"].append("No promoted concepts available for motif-concept linkage.")
    if int(result.get("future_option_transfer_link_count") or 0) == 0 and int(result.get("motifs_skipped_no_role_links") or 0) > 0:
        result["missing_evidence"].append("No future-option transfer links were produced because motifs lack role links.")
    if int(result.get("roles_seen_from_motif_links") or 0) > 0 and int(result.get("roles_with_transfer_attempts") or 0) == 0:
        result["missing_evidence"].append("Future-option motifs have role links, but no matching role-transfer attempts were found.")
    if int(result.get("roles_with_transfer_attempts") or 0) > 0 and int(result.get("roles_with_concepts") or 0) == 0:
        result["missing_evidence"].append("Role-transfer evidence exists, but roles are not linked to concepts.")
    result["core_metrics"] = {
        key: result.get(key)
        for key in (
            "future_option_transfer_link_count",
            "future_option_motif_count",
            "h11_blocked_by_no_motifs",
            "h11_blocked_by_no_promoted_concepts",
            "motifs_with_transfer_count",
            "motifs_with_strong_transfer_count",
            "motifs_with_promoted_concept_count",
            "all_motifs_with_transfer_count",
            "verified_motifs_with_transfer_count",
            "all_motifs_with_strong_transfer_count",
            "verified_motifs_with_strong_transfer_count",
            "all_motifs_with_promoted_concept_count",
            "verified_motifs_with_promoted_concept_count",
            "motif_transfer_success_rate",
            "motif_strong_transfer_success_rate",
            "promoted_concept_motif_count",
            "emergent_future_option_motif_count",
            "emergent_motif_transfer_link_count",
            "all_emergent_motif_transfer_link_count",
            "fully_verified_emergent_chain_count",
            "partially_verified_emergent_chain_count",
            "unverified_emergent_chain_count",
            "emergent_motifs_with_transfer_count",
            "emergent_motifs_with_strong_transfer_count",
            "emergent_motifs_with_promoted_concept_count",
            "emergent_motif_transfer_success_rate",
            "emergent_motif_strong_transfer_success_rate",
            "promoted_concept_emergent_motif_count",
            "non_emergent_motif_transfer_link_count",
            "non_emergent_motifs_with_strong_transfer_count",
            "non_emergent_motifs_with_promoted_concept_count",
            "successful_role_transfer_count",
            "verified_concrete_transfer_link_count",
            "verified_transfer_pair_count",
            "distinct_source_target_pair_count",
            "verified_cross_game_link_count",
            "verified_cross_game_motif_count",
            "verified_cross_game_pair_count",
            "motif_transfer_chain_provenance_breakdown",
            "blocked_by_motif_provenance",
            "blocked_by_transfer_provenance",
            "blocked_by_concept_validation",
            "blocked_by_missing_concept",
            "promoted_concept_count",
            "events_with_owner_type_role",
            "role_linked_event_count",
            "motifs_with_role_links",
            "emergent_motifs_with_role_links",
            "motifs_with_family_provenance",
            "motifs_with_carrier_provenance",
            "motifs_with_role_provenance",
            "motifs_with_concept_provenance",
            "provenance_resolution_failures",
            "failure_occurrence_count",
            "unique_unresolved_id_count",
            "unresolved_family_to_carrier_count",
            "unresolved_carrier_to_role_count",
            "unresolved_role_to_concept_count",
            "unique_unresolved_family_to_carrier_count",
            "unique_unresolved_carrier_to_role_count",
            "unique_unresolved_role_to_concept_count",
            "motifs_seen_for_transfer",
            "motifs_skipped_no_role_links",
            "motifs_skipped_insufficient_support_or_stability",
            "motifs_with_role_links_for_transfer",
            "roles_seen_from_motif_links",
            "roles_with_transfer_attempts",
            "roles_with_concepts",
            "unique_roles_seen_from_motif_links",
            "unique_roles_with_transfer_attempts",
            "unique_roles_with_concepts",
            "motif_role_link_count",
            "motif_role_transfer_attempt_link_count",
            "motif_role_concept_link_count",
        )
    }
    _write(output_dir, result)
    return result


def _write(output_dir: Path, result: dict[str, object]) -> None:
    (output_dir / "h11_future_option_transfer_concepts_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = (
        f"H11 decision: {result.get('decision')}\n"
        f"future-option transfer links: {result.get('future_option_transfer_link_count')}\n"
        f"all / verified motif transfers: {result.get('all_motifs_with_transfer_count')} / {result.get('verified_motifs_with_transfer_count')}\n"
        f"all / verified strong transfers: {result.get('all_motifs_with_strong_transfer_count')} / {result.get('verified_motifs_with_strong_transfer_count')}\n"
        f"all / verified promoted-concept motifs: {result.get('all_motifs_with_promoted_concept_count')} / {result.get('verified_motifs_with_promoted_concept_count')}\n"
        f"all / fully verified emergent chains: {result.get('all_emergent_motif_transfer_link_count')} / {result.get('fully_verified_emergent_chain_count')}\n"
        f"partially / unverified emergent chains: {result.get('partially_verified_emergent_chain_count')} / {result.get('unverified_emergent_chain_count')}\n"
        f"verified cross-game links / pairs: {result.get('verified_cross_game_link_count')} / {result.get('verified_cross_game_pair_count')}\n"
        f"blocked by motif / transfer / concept: {result.get('blocked_by_motif_provenance')} / {result.get('blocked_by_transfer_provenance')} / {result.get('blocked_by_concept_validation')}\n"
        f"emergent motifs with strong transfer: {result.get('emergent_motifs_with_strong_transfer_count')}\n"
        f"emergent motifs with promoted concepts: {result.get('emergent_motifs_with_promoted_concept_count')}\n"
        f"non-emergent motif transfer links: {result.get('non_emergent_motif_transfer_link_count')}\n"
        f"motifs skipped no role links: {result.get('motifs_skipped_no_role_links')}\n"
        f"roles seen from motif links: {result.get('roles_seen_from_motif_links')}\n"
        f"roles with transfer attempts: {result.get('roles_with_transfer_attempts')}\n"
        f"roles with concepts: {result.get('roles_with_concepts')}\n"
    )
    (output_dir / "h11_future_option_transfer_concepts_report.txt").write_text(text, encoding="utf-8")
    (output_dir / "h11_future_option_transfer_concepts.md").write_text("```\n" + text + "```\n", encoding="utf-8")
