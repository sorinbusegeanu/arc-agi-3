from __future__ import annotations

import json
import sqlite3
from collections import Counter
from hashlib import sha1
from pathlib import Path
from typing import Any

from v6.future_options import derive_future_option_memory


def _missing_tables(connection: sqlite3.Connection, required: tuple[str, ...]) -> list[str]:
    tables = {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    return [name for name in required if name not in tables]


def _complete_context_key(value: object) -> bool:
    if value in (None, ""):
        return False
    text = str(value).strip().lower()
    return bool(text) and "null" not in text and "none" not in text and text not in {"[]", "{}"}


def _context_id(value: object) -> str | None:
    if value in (None, ""):
        return None
    return "ctx:" + sha1(str(value).encode("utf-8")).hexdigest()[:20]


def evaluate_h09_future_option_motifs(
    *,
    memory_dir: Path,
    run_dir: Path | None,
    output_dir: Path,
    already_derived: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    current_state = Path(memory_dir) / "current_state.sqlite"
    if not already_derived and current_state.exists():
        derive_future_option_memory(memory_dir=memory_dir, run_dir=run_dir)
    if not current_state.exists():
        result = {
            "hypothesis_id": "H09",
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
                "future_option_events",
                "future_option_motifs",
                "higher_order_milestones",
                "stable_contingencies",
                "transformation_families",
            ),
        )
        if missing_tables:
            result = {
                "hypothesis_id": "H09",
                "evidence_source": "compact_memory",
                "decision": "INSUFFICIENT_EVIDENCE",
                "missing_evidence": [f"Missing expected compact-memory table(s): {', '.join(missing_tables)}"],
                "core_metrics": {},
            }
            _write(output_dir, result)
            return result
        events = [dict(row) for row in conn.execute("SELECT * FROM future_option_events ORDER BY event_id ASC").fetchall()]
        motifs = [dict(row) for row in conn.execute("SELECT * FROM future_option_motifs ORDER BY motif_signature ASC").fetchall()]
        observations = [dict(row) for row in conn.execute(
            "SELECT * FROM future_option_motif_observations ORDER BY motif_signature ASC, event_id ASC"
        ).fetchall()] if "future_option_motif_observations" in {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()} else []
        milestone_map = dict(conn.execute("SELECT milestone_name, first_global_step FROM higher_order_milestones").fetchall())
        summary_row = conn.execute(
            "SELECT value_json FROM memory_summary WHERE key = 'future_option_derivation_summary'"
        ).fetchone()
        derivation_summary = json.loads(str(summary_row[0])) if summary_row and summary_row[0] else {}
        stable_contingencies_count = int(conn.execute("SELECT COUNT(*) FROM stable_contingencies").fetchone()[0])
        transformation_families_count = int(conn.execute("SELECT COUNT(*) FROM transformation_families").fetchone()[0])
    motif_type_counts = Counter(str(row["motif_type"] or "unknown") for row in motifs)
    emergent_count = sum(1 for row in motifs if int(row["is_emergent"] or 0) == 1)
    verified_observations = [row for row in observations if str(row.get("provenance_status") or "missing") == "verified"]
    verified_cross_game_observations = [
        row for row in verified_observations
        if row.get("source_game_key") not in (None, "") and row.get("target_game_key") not in (None, "")
        and str(row["source_game_key"]) != str(row["target_game_key"])
        and not int(row.get("source_game_is_surrogate") or 0)
        and not int(row.get("target_game_is_surrogate") or 0)
    ]
    verified_cross_context_observations = [
        row for row in verified_observations
        if _complete_context_key(row.get("source_context_key"))
        and _complete_context_key(row.get("target_context_key"))
        and str(row["source_context_key"]) != str(row["target_context_key"])
        and not int(row.get("source_context_is_surrogate") or 0)
        and not int(row.get("target_context_is_surrogate") or 0)
    ]
    cross_context_motif_count = len({str(row["motif_signature"]) for row in verified_cross_context_observations})
    cross_game_motif_count = len({str(row["motif_signature"]) for row in verified_cross_game_observations})
    mean_abs_option_delta = _mean([abs(float(row.get("option_delta") or 0.0)) for row in events])
    max_abs_option_delta = max((abs(float(row.get("option_delta") or 0.0)) for row in events), default=None)
    mean_motif_stability_score = _mean([row.get("motif_stability_score") for row in motifs])
    result = {
        "hypothesis_id": "H09",
        "evidence_source": "compact_memory",
        "future_option_event_count": len(events),
        "future_option_motif_count": len(motifs),
        "emergent_future_option_motif_count": emergent_count,
        "motif_type_counts": dict(sorted(motif_type_counts.items())),
        "motif_type_source_counts": None,
        "unknown_motif_source_count": None,
        "unknown_motif_source_ratio": None,
        "cross_context_motif_count": cross_context_motif_count,
        "cross_game_motif_count": cross_game_motif_count,
        "mean_abs_option_delta": mean_abs_option_delta,
        "max_abs_option_delta": max_abs_option_delta,
        "mean_motif_stability_score": mean_motif_stability_score,
        "unknown_motif_count": None,
        "unknown_motif_ratio": None,
        "unknown_motif_event_count": None,
        "unknown_motif_event_ratio": None,
        "live_delta_event_count": 0,
        "structured_effect_event_count": 0,
        "text_keyword_event_count": 0,
        "future_option_edge_event_count": 0,
        "structural_effect_path_enabled": derivation_summary.get("structural_effect_path_enabled"),
        "future_option_edge_path_enabled": derivation_summary.get("future_option_edge_path_enabled"),
        "text_keyword_path_enabled": derivation_summary.get("text_keyword_path_enabled"),
        "structural_rows_seen": derivation_summary.get("structural_rows_seen"),
        "structural_rows_eligible": derivation_summary.get("structural_rows_eligible"),
        "structural_rows_inserted": derivation_summary.get("structural_rows_inserted"),
        "structural_rows_skipped_missing_effect": derivation_summary.get("structural_rows_skipped_missing_effect"),
        "structural_rows_skipped_missing_scope": derivation_summary.get("structural_rows_skipped_missing_scope"),
        "first_future_option_event_step": milestone_map.get("first_future_option_event_step"),
        "first_emergent_future_option_motif_step": milestone_map.get("first_emergent_future_option_motif_step"),
        "stable_contingencies_count": stable_contingencies_count,
        "transformation_families_count": transformation_families_count,
        "stable_contingency_rows_seen": derivation_summary.get("stable_contingency_rows_seen"),
        "stable_contingency_events_inserted": derivation_summary.get("stable_contingency_events_inserted"),
        "transformation_family_rows_seen": derivation_summary.get("transformation_family_rows_seen"),
        "transformation_family_events_inserted": derivation_summary.get("transformation_family_events_inserted"),
        "carrier_rows_seen": derivation_summary.get("carrier_rows_seen"),
        "carrier_events_inserted": derivation_summary.get("carrier_events_inserted"),
        "role_rows_seen": derivation_summary.get("role_rows_seen"),
        "role_events_inserted": derivation_summary.get("role_events_inserted"),
        "future_option_events_inserted_total": derivation_summary.get("future_option_events_inserted_total"),
        "future_option_motifs_inserted_total": derivation_summary.get("future_option_motifs_inserted_total"),
        "future_option_derivation_error": derivation_summary.get("future_option_derivation_error"),
        "future_option_stage": derivation_summary.get("future_option_stage"),
        "classified_by_structural_effect_count": derivation_summary.get("classified_by_structural_effect_count"),
        "classified_by_option_delta_count": derivation_summary.get("classified_by_option_delta_count"),
        "classified_by_graph_effect_count": derivation_summary.get("classified_by_graph_effect_count"),
        "classified_by_role_effect_count": derivation_summary.get("classified_by_role_effect_count"),
        "classified_by_concept_effect_count": derivation_summary.get("classified_by_concept_effect_count"),
        "unknown_reason_counts": derivation_summary.get("unknown_reason_counts"),
        "missing_evidence": [],
    }
    source_counts = Counter()
    unknown_event_count = 0
    for row in events:
        # The persisted classification fields are the decision evidence.  The
        # historical JSON payload is only supplemental and may describe a
        # pre-classification heuristic, so it must not override a concrete
        # classification source (in particular, a measured "unknown").
        source_counts[str(row.get("classification_source") or "unknown")] += 1
        if str(row.get("motif_type") or "unknown") == "unknown":
            unknown_event_count += 1
    unknown_motif_count = int(motif_type_counts.get("unknown", 0))
    result["motif_type_source_counts"] = dict(sorted(source_counts.items()))
    result["unknown_motif_count"] = unknown_motif_count
    result["unknown_motif_ratio"] = (unknown_motif_count / len(motifs)) if motifs else None
    result["unknown_motif_event_count"] = unknown_event_count
    result["unknown_motif_event_ratio"] = (unknown_event_count / len(events)) if events else None
    result["unknown_motif_source_count"] = int(source_counts.get("unknown", 0))
    result["unknown_motif_source_ratio"] = (result["unknown_motif_source_count"] / len(events)) if events else None
    result["live_delta_event_count"] = int(source_counts.get("live_delta", 0)) + int(source_counts.get("live_delta_rule", 0))
    result["structured_effect_event_count"] = int(source_counts.get("structural_effect", 0)) + int(source_counts.get("structured_effect", 0))
    result["text_keyword_event_count"] = int(source_counts.get("text_keyword", 0))
    result["future_option_edge_event_count"] = int(source_counts.get("future_option_edge", 0))
    allowed_sources = {"structural_effect", "structured_effect", "option_delta", "graph_effect", "role_effect", "concept_effect", "future_option_edge", "live_delta_rule"}
    verified_motifs = [row for row in motifs if str(row.get("provenance_status") or "missing") == "verified"]
    proxy_motifs = [row for row in motifs if str(row.get("provenance_status") or "missing") == "proxy"]
    legacy_motifs = [row for row in motifs if str(row.get("classification_source") or "unknown") == "unknown"]
    missing_motifs = [row for row in motifs if str(row.get("provenance_status") or "missing") not in {"verified", "proxy"}]
    result["verified_motif_count"] = len(verified_motifs)
    result["proxy_motif_count"] = len(proxy_motifs)
    result["legacy_motif_count"] = len(legacy_motifs)
    result["classified_without_provenance_count"] = len(proxy_motifs) + sum(
        1 for row in legacy_motifs if str(row.get("motif_type") or "unknown") != "unknown"
    )
    result["missing_motif_count"] = len(missing_motifs)
    result["motif_scope_summary"] = {
        "observation_count": len(observations),
        "verified_cross_game_observation_count": len(verified_cross_game_observations),
        "verified_cross_context_observation_count": len(verified_cross_context_observations),
    }
    result["motif_scope_sample"] = [
        {
            "motif_signature": row["motif_signature"], "event_id": row["event_id"],
            "source_game_key": row.get("source_game_key"), "target_game_key": row.get("target_game_key"),
            "source_context_id": _context_id(row.get("source_context_key")),
            "target_context_id": _context_id(row.get("target_context_key")),
            "provenance_status": row.get("provenance_status"),
        }
        for row in observations[:200]
    ]
    result["classification_source_counts"] = dict(sorted(Counter(str(row.get("classification_source") or "unknown") for row in events).items()))
    result["classification_provenance_status_counts"] = dict(sorted(Counter(str(row.get("classification_provenance_status") or "missing") for row in events).items()))
    result["verified_observation_count"] = len(verified_observations)
    result["proxy_observation_count"] = sum(1 for row in observations if str(row.get("provenance_status") or "missing") == "proxy")
    result["incomplete_context_observation_count"] = sum(1 for row in observations if not _complete_context_key(row.get("source_context_key")) or not _complete_context_key(row.get("target_context_key")))
    result["surrogate_context_observation_count"] = sum(1 for row in observations if int(row.get("source_context_is_surrogate") or 0) or int(row.get("target_context_is_surrogate") or 0))
    result["verified_cross_game_observation_count"] = len(verified_cross_game_observations)
    result["verified_cross_context_observation_count"] = len(verified_cross_context_observations)
    verified_motif_ids = {str(row["motif_signature"]) for row in verified_observations}
    direct_motif_ids = {
        str(row["motif_signature"])
        for row in verified_observations if row.get("source_interaction_id") not in (None, "")
    }
    family_motif_ids = {
        str(row["motif_signature"])
        for row in motifs
        if str(row.get("provenance_status") or "missing") == "verified"
        and bool(_json_list(row.get("source_family_ids_json")))
    }
    carrier_motif_ids = {
        str(row["motif_signature"])
        for row in motifs
        if str(row.get("provenance_status") or "missing") == "verified"
        and bool(_json_list(row.get("source_carrier_ids_json")))
    }
    role_motif_ids = {
        str(row["motif_signature"])
        for row in motifs
        if str(row.get("provenance_status") or "missing") == "verified"
        and bool(_json_list(row.get("source_role_ids_json")))
    }
    concept_motif_ids = {
        str(row["motif_signature"])
        for row in motifs
        if str(row.get("provenance_status") or "missing") == "verified"
        and bool(_json_list(row.get("source_concept_ids_json")))
    }
    result["motifs_verified_by_direct_interaction"] = len(direct_motif_ids)
    result["motifs_verified_by_family_path"] = len(family_motif_ids - direct_motif_ids)
    result["motifs_verified_by_carrier_path"] = len(carrier_motif_ids - direct_motif_ids - family_motif_ids)
    result["motifs_verified_by_role_path"] = len(role_motif_ids - direct_motif_ids - family_motif_ids - carrier_motif_ids)
    result["motifs_verified_by_concept_path"] = len(concept_motif_ids - direct_motif_ids - family_motif_ids - carrier_motif_ids - role_motif_ids)
    result["motifs_with_only_surrogate_scope"] = len({
        str(row["motif_signature"])
        for row in observations
        if str(row.get("motif_signature")) not in verified_motif_ids
        and (int(row.get("source_context_is_surrogate") or 0) or int(row.get("target_context_is_surrogate") or 0))
    })
    non_unknown_types = [key for key, value in motif_type_counts.items() if key != "unknown" and value > 0]
    if not events:
        if stable_contingencies_count > 0 or transformation_families_count > 0:
            result["decision"] = "INSUFFICIENT_EVIDENCE"
            result["missing_evidence"].append("Future-option derivation produced zero events despite available substrate.")
        else:
            result["decision"] = "INCONCLUSIVE"
    elif events and not motifs:
        result["decision"] = "INVALID"
    elif motifs and emergent_count == 0:
        result["decision"] = "PARTIALLY_VALID"
    elif (
        emergent_count >= 1
        and len(verified_motifs) >= 1
        and len(non_unknown_types) >= 2
        and (cross_context_motif_count >= 1 or cross_game_motif_count >= 1)
        and (mean_abs_option_delta or 0.0) > 0.0
        and ((result.get("unknown_motif_ratio") or 0.0) <= 0.20)
        and ((result.get("unknown_motif_event_ratio") or 0.0) <= 0.20)
        and ((result.get("unknown_motif_source_ratio") or 0.0) <= 0.20)
    ):
        result["decision"] = "VALID"
    else:
        result["decision"] = "PARTIALLY_VALID"
    if (result.get("unknown_motif_event_ratio") or 0.0) > 0.20:
        result["missing_evidence"].append("Future-option event classification remains mostly unknown.")
    if result["classified_without_provenance_count"]:
        result["missing_evidence"].append("Classified motifs without concrete event provenance are excluded from scientific conclusions.")
    result["evidence_diagnostics"] = {
        "stable_contingencies_count": stable_contingencies_count,
        "transformation_families_count": transformation_families_count,
        "future_option_event_count": len(events),
        "future_option_motif_count": len(motifs),
        "future_option_derivation_summary_present": bool(derivation_summary),
    }
    result["core_metrics"] = {
        key: result.get(key)
        for key in (
            "future_option_event_count",
            "stable_contingency_rows_seen",
            "stable_contingency_events_inserted",
            "transformation_family_rows_seen",
            "transformation_family_events_inserted",
            "carrier_rows_seen",
            "carrier_events_inserted",
            "role_rows_seen",
            "role_events_inserted",
            "future_option_events_inserted_total",
            "future_option_motifs_inserted_total",
            "future_option_stage",
            "classified_by_structural_effect_count",
            "classified_by_option_delta_count",
            "classified_by_graph_effect_count",
            "classified_by_role_effect_count",
            "classified_by_concept_effect_count",
            "unknown_reason_counts",
            "future_option_motif_count",
            "emergent_future_option_motif_count",
            "motif_type_counts",
            "motif_type_source_counts",
            "cross_context_motif_count",
            "cross_game_motif_count",
            "mean_abs_option_delta",
            "max_abs_option_delta",
            "mean_motif_stability_score",
            "unknown_motif_count",
            "unknown_motif_ratio",
            "unknown_motif_event_count",
            "unknown_motif_event_ratio",
            "unknown_motif_source_count",
            "unknown_motif_source_ratio",
            "live_delta_event_count",
            "structured_effect_event_count",
            "text_keyword_event_count",
            "future_option_edge_event_count",
            "verified_motif_count",
            "proxy_motif_count",
                "legacy_motif_count",
                "missing_motif_count",
                "classified_without_provenance_count",
                "classification_source_counts",
                "classification_provenance_status_counts",
                "verified_observation_count",
                "proxy_observation_count",
                "incomplete_context_observation_count",
                "surrogate_context_observation_count",
                "verified_cross_game_observation_count",
                "verified_cross_context_observation_count",
        )
    }
    _write_observations(output_dir, observations)
    result["h09_motif_observations_artifact"] = "h09_motif_observations.jsonl"
    assert result["verified_motif_count"] + result["proxy_motif_count"] + result["missing_motif_count"] == len(motifs)
    assert result["unknown_motif_source_count"] == sum(
        1 for row in events if str(row.get("classification_source") or "unknown") == "unknown"
    )
    assert result["cross_game_motif_count"] == len({str(row["motif_signature"]) for row in verified_cross_game_observations})
    assert result["cross_context_motif_count"] == len({str(row["motif_signature"]) for row in verified_cross_context_observations})
    _write(output_dir, result)
    return result


def _mean(values: list[Any]) -> float | None:
    cooked = [float(value) for value in values if value is not None]
    return (sum(cooked) / len(cooked)) if cooked else None


def _json_list(value: Any) -> list[str]:
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, ValueError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _write(output_dir: Path, result: dict[str, Any]) -> None:
    (output_dir / "h09_future_option_motifs_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = (
        f"H09 decision: {result.get('decision')}\n"
        f"future-option events: {result.get('future_option_event_count')}\n"
        f"future-option motifs: {result.get('future_option_motif_count')}\n"
        f"emergent motifs: {result.get('emergent_future_option_motif_count')}\n"
        f"development stage: {result.get('future_option_stage')}\n"
        f"motif types: {result.get('motif_type_counts')}\n"
        f"motif type sources: {result.get('motif_type_source_counts')}\n"
        f"unknown motif ratio: {result.get('unknown_motif_ratio')}\n"
    )
    (output_dir / "h09_future_option_motifs_report.txt").write_text(text, encoding="utf-8")
    (output_dir / "h09_future_option_motifs.md").write_text("```\n" + text + "```\n", encoding="utf-8")


def _write_observations(output_dir: Path, observations: list[dict[str, Any]]) -> None:
    with (output_dir / "h09_motif_observations.jsonl").open("w", encoding="utf-8") as handle:
        for row in observations:
            payload = dict(row)
            payload["source_context_id"] = _context_id(payload.pop("source_context_key", None))
            payload["target_context_id"] = _context_id(payload.pop("target_context_key", None))
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
