from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


_HYPOTHESES = tuple(f"H{index:02d}" for index in range(1, 13))


def _blank_counts() -> dict[str, int]:
    return {
        "verified_claim_count": 0,
        "proxy_claim_count": 0,
        "legacy_claim_count": 0,
        "invalid_claim_count": 0,
        "missing_provenance_count": 0,
    }


def validate_hypothesis_provenance(
    *,
    memory_dir: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Read-only provenance validation for H01-H12."""
    report: dict[str, Any] = {
        **_blank_counts(),
        "by_hypothesis": {
            hypothesis: _blank_counts()
            for hypothesis in _HYPOTHESES
        },
        "invalid_by_hypothesis": {},
        "missing_by_hypothesis": {},
        "errors": [],
        "read_only": True,
        "schema_version": "v6.1",
    }
    if memory_dir is None:
        for hypothesis in _HYPOTHESES:
            _count(report, "missing", hypothesis)
        _write(output_dir, report)
        return report

    current_state = Path(memory_dir) / "current_state.sqlite"
    if not current_state.exists():
        for hypothesis in _HYPOTHESES:
            _count(report, "missing", hypothesis)
        _write(output_dir, report)
        return report

    uri = f"file:{current_state.resolve()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=30.0) as connection:
            connection.row_factory = sqlite3.Row
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            _validate_h01(connection, tables, report)
            _validate_h02(connection, tables, report)
            _validate_h03(connection, tables, report)
            _validate_h04(connection, tables, report)
            _validate_h05(connection, tables, report)
            _validate_h06(connection, tables, report)
            _validate_h07(connection, tables, report)
            _validate_h08(connection, tables, report)
            _validate_h09(connection, tables, report)
            _validate_h10(connection, tables, report)
            _validate_h11(connection, tables, report)
            _validate_h12(connection, tables, report)
    except sqlite3.Error as exc:
        report["errors"].append(
            {
                "hypothesis": "H01-H12",
                "error": f"provenance database read failed: {exc}",
            }
        )
        for hypothesis in _HYPOTHESES:
            _count(report, "missing", hypothesis)

    report["invalid_by_hypothesis"] = {
        hypothesis: counts["invalid_claim_count"]
        for hypothesis, counts in report["by_hypothesis"].items()
        if counts["invalid_claim_count"]
    }
    report["missing_by_hypothesis"] = {
        hypothesis: counts["missing_provenance_count"]
        for hypothesis, counts in report["by_hypothesis"].items()
        if counts["missing_provenance_count"]
    }
    _write(output_dir, report)
    return report


def _validate_h01(
    connection: sqlite3.Connection,
    tables: set[str],
    report: dict[str, Any],
) -> None:
    if "stable_contingencies" not in tables:
        _count(report, "missing", "H01")
        return
    count = int(
        connection.execute(
            "SELECT COUNT(*) FROM stable_contingencies"
        ).fetchone()[0]
    )
    _count(report, "verified" if count else "missing", "H01")


def _validate_h02(
    connection: sqlite3.Connection,
    tables: set[str],
    report: dict[str, Any],
) -> None:
    if not {"memory_scores", "memory_edges"} <= tables:
        _count(report, "missing", "H02")
        return
    row = connection.execute(
        """
        SELECT COUNT(*)
        FROM memory_edges
        WHERE edge_type='violates_prediction'
        """
    ).fetchone()
    _count(
        report,
        "verified" if int(row[0] or 0) else "missing",
        "H02",
    )


def _validate_h03(
    connection: sqlite3.Connection,
    tables: set[str],
    report: dict[str, Any],
) -> None:
    if "transformation_families" not in tables:
        _count(report, "missing", "H03")
        return
    count = int(
        connection.execute(
            "SELECT COUNT(*) FROM transformation_families"
        ).fetchone()[0]
    )
    _count(report, "verified" if count else "missing", "H03")


def _validate_h04(
    connection: sqlite3.Connection,
    tables: set[str],
    report: dict[str, Any],
) -> None:
    if "carrier_candidates" not in tables:
        _count(report, "missing", "H04")
        return
    rows = connection.execute(
        "SELECT * FROM carrier_candidates"
    ).fetchall()
    if not rows:
        _count(report, "missing", "H04")
        return
    for row in rows:
        source = str(
            _row_value(row, "carrier_source", "") or ""
        )
        if source in {"context_action_fallback", "surrogate"}:
            _count(report, "proxy", "H04")
        else:
            _count(report, "verified", "H04")


def _validate_h05(
    connection: sqlite3.Connection,
    tables: set[str],
    report: dict[str, Any],
) -> None:
    if not {"role_candidates", "role_links"} <= tables:
        _count(report, "missing", "H05")
        return
    rows = connection.execute(
        "SELECT role_signature FROM role_candidates"
    ).fetchall()
    if not rows:
        _count(report, "missing", "H05")
        return
    for row in rows:
        role = str(row[0])
        links = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM role_links
                WHERE role_signature=?
                """,
                (role,),
            ).fetchone()[0]
        )
        _count(
            report,
            "verified" if links else "missing",
            "H05",
        )


def _validate_h06(
    connection: sqlite3.Connection,
    tables: set[str],
    report: dict[str, Any],
) -> None:
    if "role_transfer_attempts" not in tables:
        _count(report, "missing", "H06")
        return
    rows = connection.execute(
        "SELECT * FROM role_transfer_attempts"
    ).fetchall()
    if not rows:
        _count(report, "missing", "H06")
        return
    for row in rows:
        kind = str(_row_value(row, "transfer_kind", "") or "")
        mode = str(
            _row_value(row, "provenance_mode", "legacy")
            or "legacy"
        )
        status = str(
            _row_value(row, "provenance_status", "missing")
            or "missing"
        )
        source_game = _row_value(row, "source_game_key")
        target_game = _row_value(row, "target_game_key")
        source_context = _row_value(row, "source_context_key")
        target_context = _row_value(row, "target_context_key")
        if status == "verified" and mode == "single_source":
            if kind == "cross_game" and (
                not source_game
                or not target_game
                or source_game == target_game
            ):
                _count(
                    report,
                    "invalid",
                    "H06",
                    "cross-game transfer lacks distinct concrete games",
                )
            elif kind == "cross_context" and (
                not source_context
                or not target_context
                or source_context == target_context
            ):
                _count(
                    report,
                    "invalid",
                    "H06",
                    "cross-context transfer lacks distinct concrete contexts",
                )
            else:
                _count(report, "verified", "H06")
        elif mode == "legacy" or status == "legacy":
            _count(report, "legacy", "H06")
        elif status in {"proxy", "resolved_with_surrogate"}:
            _count(report, "proxy", "H06")
        else:
            _count(report, "missing", "H06")


def _validate_h07(
    connection: sqlite3.Connection,
    tables: set[str],
    report: dict[str, Any],
) -> None:
    if not {
        "concept_candidates",
        "concept_links",
    } <= tables:
        _count(report, "missing", "H07")
        return
    rows = connection.execute(
        """
        SELECT concept_signature, COALESCE(is_promoted, 0)
        FROM concept_candidates
        """
    ).fetchall()
    if not rows:
        _count(report, "missing", "H07")
        return
    for signature, promoted in rows:
        links = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM concept_links
                WHERE concept_signature=?
                """,
                (str(signature),),
            ).fetchone()[0]
        )
        if not links:
            _count(report, "missing", "H07")
        elif int(promoted or 0):
            _count(report, "verified", "H07")
        else:
            _count(report, "proxy", "H07")


def _validate_h08(
    connection: sqlite3.Connection,
    tables: set[str],
    report: dict[str, Any],
) -> None:
    if not {
        "world_model_components",
        "world_model_links",
        "concept_candidates",
    } <= tables:
        _count(report, "missing", "H08")
        return
    rows = connection.execute(
        """
        SELECT component_signature,
               COALESCE(is_coherent, 0)
        FROM world_model_components
        """
    ).fetchall()
    if not rows:
        _count(report, "missing", "H08")
        return
    for signature, coherent in rows:
        linked_promoted = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM world_model_links AS link
                JOIN concept_candidates AS concept
                  ON concept.concept_signature=link.linked_key
                WHERE link.component_signature=?
                  AND link.linked_type='concept'
                  AND COALESCE(concept.is_promoted, 0)=1
                """,
                (str(signature),),
            ).fetchone()[0]
        )
        if int(coherent or 0) and linked_promoted:
            _count(report, "verified", "H08")
        elif linked_promoted:
            _count(report, "proxy", "H08")
        else:
            _count(report, "missing", "H08")


def _validate_h09(
    connection: sqlite3.Connection,
    tables: set[str],
    report: dict[str, Any],
) -> None:
    if "future_option_motifs" not in tables:
        _count(report, "missing", "H09")
        return
    rows = connection.execute(
        "SELECT * FROM future_option_motifs"
    ).fetchall()
    if not rows:
        _count(report, "missing", "H09")
        return
    for row in rows:
        motif_type = str(
            _row_value(row, "motif_type", "unknown")
            or "unknown"
        )
        status = str(
            _row_value(row, "provenance_status", "missing")
            or "missing"
        )
        source = str(
            _row_value(row, "classification_source", "unknown")
            or "unknown"
        )
        if motif_type == "unknown" or source == "unknown":
            _count(report, "invalid", "H09")
        elif status == "verified":
            _count(report, "verified", "H09")
        elif status in {"proxy", "resolved_with_surrogate"}:
            _count(report, "proxy", "H09")
        else:
            _count(report, "missing", "H09")


def _validate_h10(
    connection: sqlite3.Connection,
    tables: set[str],
    report: dict[str, Any],
) -> None:
    table = (
        "future_option_attention_links"
        if "future_option_attention_links" in tables
        else (
            "future_option_attention"
            if "future_option_attention" in tables
            else None
        )
    )
    if table is None:
        _count(report, "missing", "H10")
        return
    count = int(
        connection.execute(
            f'SELECT COUNT(*) FROM "{table}"'
        ).fetchone()[0]
    )
    _count(report, "verified" if count else "missing", "H10")


def _validate_h11(
    connection: sqlite3.Connection,
    tables: set[str],
    report: dict[str, Any],
) -> None:
    if "future_option_transfer_links" not in tables:
        _count(report, "missing", "H11")
        return
    rows = connection.execute(
        """
        SELECT motif_provenance_status,
               transfer_provenance_status,
               concept_validation_status
        FROM future_option_transfer_links
        """
    ).fetchall()
    if not rows:
        _count(report, "missing", "H11")
        return
    for row in rows:
        statuses = {
            str(value or "missing")
            for value in row
        }
        if statuses == {"verified"}:
            _count(report, "verified", "H11")
        elif "missing" in statuses:
            # A materialized H11 link with a partially resolved chain is candidate/proxy evidence,
            # not a missing required claim. Only a wholly absent provenance chain is missing.
            if statuses == {"missing"}:
                _count(report, "missing", "H11")
            else:
                _count(report, "proxy", "H11")
        elif "legacy" in statuses:
            _count(report, "legacy", "H11")
        else:
            _count(report, "proxy", "H11")


def _validate_h12(
    connection: sqlite3.Connection,
    tables: set[str],
    report: dict[str, Any],
) -> None:
    table = (
        "trajectory_efficiency"
        if "trajectory_efficiency" in tables
        else (
            "trajectory_efficiency_records"
            if "trajectory_efficiency_records" in tables
            else None
        )
    )
    if table is None:
        _count(report, "missing", "H12")
        return
    count = int(
        connection.execute(
            f'SELECT COUNT(*) FROM "{table}"'
        ).fetchone()[0]
    )
    _count(report, "verified" if count else "missing", "H12")


def _row_value(
    row: sqlite3.Row,
    key: str,
    default: Any = None,
) -> Any:
    try:
        return row[key]
    except (IndexError, KeyError):
        return default


def _count(
    report: dict[str, Any],
    status: str,
    hypothesis: str,
    error: str | None = None,
) -> None:
    field = {
        "verified": "verified_claim_count",
        "proxy": "proxy_claim_count",
        "legacy": "legacy_claim_count",
        "invalid": "invalid_claim_count",
        "missing": "missing_provenance_count",
    }[status]
    report[field] = int(report.get(field, 0)) + 1
    by_hypothesis = report["by_hypothesis"].setdefault(
        hypothesis, _blank_counts()
    )
    by_hypothesis[field] = int(
        by_hypothesis.get(field, 0)
    ) + 1
    if error:
        report["errors"].append(
            {"hypothesis": hypothesis, "error": error}
        )


def _write(
    output_dir: Path,
    report: dict[str, Any],
) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "provenance_validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
