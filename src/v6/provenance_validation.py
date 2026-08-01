from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def validate_hypothesis_provenance(*, memory_dir: Path | None, output_dir: Path) -> dict[str, Any]:
    """Validate concrete identity requirements without inventing provenance.

    Missing provenance is reported separately from contradictory provenance so
    callers can downgrade only claims that actually rely on invalid rows.
    """
    report: dict[str, Any] = {
        "verified_claim_count": 0,
        "proxy_claim_count": 0,
        "legacy_claim_count": 0,
        "invalid_claim_count": 0,
        "missing_provenance_count": 0,
        "invalid_by_hypothesis": {},
        "missing_by_hypothesis": {},
        "errors": [],
    }
    if memory_dir is None:
        report["missing_provenance_count"] = 1
        report["missing_by_hypothesis"] = {"H04+": 1}
        _write(output_dir, report)
        return report
    current_state = Path(memory_dir) / "current_state.sqlite"
    if not current_state.exists():
        report["missing_provenance_count"] = 1
        report["missing_by_hypothesis"] = {"H04+": 1}
        _write(output_dir, report)
        return report
    with sqlite3.connect(current_state) as conn:
        conn.row_factory = sqlite3.Row
        tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "role_transfer_attempts" in tables:
            for row in conn.execute("SELECT * FROM role_transfer_attempts").fetchall():
                kind = str(row["transfer_kind"] or "")
                mode = str(row["provenance_mode"] or "legacy")
                source_game, target_game = row["source_game_key"], row["target_game_key"]
                source_context, target_context = row["source_context_key"], row["target_context_key"]
                if mode == "legacy":
                    _count(report, "legacy", "H06")
                elif mode == "missing_source":
                    _count(report, "missing", "H06")
                elif kind == "cross_game" and (not source_game or not target_game or source_game == target_game):
                    _count(report, "invalid", "H06", "cross-game transfer lacks two distinct concrete games")
                elif kind == "cross_context" and (not source_context or not target_context or source_context == target_context):
                    _count(report, "invalid", "H06", "cross-context transfer lacks two distinct concrete contexts")
                else:
                    _count(report, "verified" if mode == "single_source" else "proxy", "H06")
        if "future_option_motifs" in tables:
            for row in conn.execute("SELECT motif_type, classification_source, source_interaction_ids_json FROM future_option_motifs").fetchall():
                motif_type = str(row["motif_type"] or "unknown")
                source = str(row["classification_source"] or "unknown")
                has_event = bool(_json_list(row["source_interaction_ids_json"]))
                if motif_type != "unknown" and source == "unknown":
                    _count(report, "invalid", "H09", "classified motif has no permitted classification source")
                elif motif_type != "unknown" and not has_event:
                    _count(report, "missing", "H09")
                elif motif_type != "unknown":
                    _count(report, "verified", "H09")
        if "concept_candidates" in tables and "concept_links" in tables:
            promoted = conn.execute("SELECT concept_signature FROM concept_candidates WHERE COALESCE(is_promoted, 0) = 1").fetchall()
            for row in promoted:
                signature = str(row[0])
                links = int(conn.execute("SELECT COUNT(*) FROM concept_links WHERE concept_signature = ?", (signature,)).fetchone()[0])
                _count(report, "verified" if links else "missing", "H07")
        if "world_model_components" in tables and "world_model_links" in tables and "concept_candidates" in tables:
            rows = conn.execute(
                "SELECT DISTINCT c.component_signature FROM world_model_components c WHERE COALESCE(c.is_coherent, 0) = 1"
            ).fetchall()
            for row in rows:
                signature = str(row[0])
                linked_promoted = int(conn.execute(
                    """SELECT COUNT(*) FROM world_model_links l JOIN concept_candidates c ON c.concept_signature = l.linked_key
                       WHERE l.component_signature = ? AND l.linked_type = 'concept' AND COALESCE(c.is_promoted, 0) = 1""",
                    (signature,),
                ).fetchone()[0])
                _count(report, "verified" if linked_promoted else "missing", "H08")
        if "future_option_transfer_links" in tables:
            for row in conn.execute("SELECT motif_provenance_status, transfer_provenance_status, concept_validation_status FROM future_option_transfer_links").fetchall():
                statuses = {str(value or "missing") for value in row}
                if "legacy" in statuses:
                    _count(report, "legacy", "H11")
                elif "missing" in statuses:
                    _count(report, "missing", "H11")
                elif statuses == {"verified"}:
                    _count(report, "verified", "H11")
                else:
                    _count(report, "proxy", "H11")
    _write(output_dir, report)
    return report


def _count(report: dict[str, Any], status: str, hypothesis: str, error: str | None = None) -> None:
    field = {
        "verified": "verified_claim_count",
        "proxy": "proxy_claim_count",
        "legacy": "legacy_claim_count",
        "invalid": "invalid_claim_count",
        "missing": "missing_provenance_count",
    }[status]
    report[field] += 1
    if status in {"invalid", "missing"}:
        key = "invalid_by_hypothesis" if status == "invalid" else "missing_by_hypothesis"
        report[key][hypothesis] = int(report[key].get(hypothesis, 0)) + 1
    if error:
        report["errors"].append({"hypothesis": hypothesis, "error": error})


def _json_list(value: Any) -> list[Any]:
    try:
        decoded = json.loads(str(value or "[]"))
    except (TypeError, ValueError):
        return []
    return decoded if isinstance(decoded, list) else []


def _write(output_dir: Path, report: dict[str, Any]) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "provenance_validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
