from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from v6.future_options import derive_future_option_memory


def _missing_tables(connection: sqlite3.Connection, required: tuple[str, ...]) -> list[str]:
    tables = {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    return [name for name in required if name not in tables]


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
        transfer_links = [dict(row) for row in conn.execute(
            "SELECT motif_signature, source_game_key, target_game_key, source_context_key, target_context_key, provenance_mode FROM future_option_transfer_links"
        ).fetchall()] if "future_option_transfer_links" in {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()} else []
        milestone_map = dict(conn.execute("SELECT milestone_name, first_global_step FROM higher_order_milestones").fetchall())
        summary_row = conn.execute(
            "SELECT value_json FROM memory_summary WHERE key = 'future_option_derivation_summary'"
        ).fetchone()
        derivation_summary = json.loads(str(summary_row[0])) if summary_row and summary_row[0] else {}
        stable_contingencies_count = int(conn.execute("SELECT COUNT(*) FROM stable_contingencies").fetchone()[0])
        transformation_families_count = int(conn.execute("SELECT COUNT(*) FROM transformation_families").fetchone()[0])
    motif_type_counts = Counter(str(row["motif_type"] or "unknown") for row in motifs)
    emergent_count = sum(1 for row in motifs if int(row["is_emergent"] or 0) == 1)
    concrete_scopes: dict[str, dict[str, set[str]]] = {}
    for link in transfer_links:
        if str(link.get("provenance_mode") or "") != "single_source":
            continue
        scope = concrete_scopes.setdefault(str(link["motif_signature"]), {"source_games": set(), "target_games": set(), "source_contexts": set(), "target_contexts": set()})
        for field, key in (("source_game_key", "source_games"), ("target_game_key", "target_games"), ("source_context_key", "source_contexts"), ("target_context_key", "target_contexts")):
            if link.get(field) not in (None, ""):
                scope[key].add(str(link[field]))
    cross_context_motif_count = sum(
        1 for scope in concrete_scopes.values()
        if scope["source_contexts"] and scope["target_contexts"]
        and any(source != target for source in scope["source_contexts"] for target in scope["target_contexts"])
    )
    cross_game_motif_count = sum(
        1 for scope in concrete_scopes.values()
        if scope["source_games"] and scope["target_games"]
        and any(source != target for source in scope["source_games"] for target in scope["target_games"])
    )
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
        evidence = json.loads(str(row.get("evidence_json") or "{}"))
        source_counts[str(evidence.get("motif_type_source") or "unknown")] += 1
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
    result["structured_effect_event_count"] = int(source_counts.get("structured_effect", 0))
    result["text_keyword_event_count"] = int(source_counts.get("text_keyword", 0))
    result["future_option_edge_event_count"] = int(source_counts.get("future_option_edge", 0))
    allowed_sources = {"structured_effect", "option_delta", "graph_effect", "role_effect", "concept_effect"}
    verified_motifs = [
        row for row in motifs
        if str(row.get("motif_type") or "unknown") != "unknown"
        and str(row.get("classification_source") or "unknown") in allowed_sources
        and str(row.get("classification_rule") or row.get("motif_classification_reason") or "").startswith("structural_")
        and bool(_json_list(row.get("source_interaction_ids_json")))
    ]
    proxy_motifs = [
        row for row in motifs
        if str(row.get("motif_type") or "unknown") != "unknown"
        and row not in verified_motifs
        and str(row.get("classification_source") or "unknown") in allowed_sources
    ]
    legacy_motifs = [row for row in motifs if str(row.get("classification_source") or "unknown") == "unknown"]
    result["verified_motif_count"] = len(verified_motifs)
    result["proxy_motif_count"] = len(proxy_motifs)
    result["legacy_motif_count"] = len(legacy_motifs)
    result["classified_without_provenance_count"] = len(proxy_motifs) + sum(
        1 for row in legacy_motifs if str(row.get("motif_type") or "unknown") != "unknown"
    )
    result["motif_scope_provenance"] = [
        {
            "motif_signature": signature,
            "source_game_keys": sorted(scope["source_games"]),
            "target_game_keys": sorted(scope["target_games"]),
            "source_context_keys": sorted(scope["source_contexts"]),
            "target_context_keys": sorted(scope["target_contexts"]),
        }
        for signature, scope in sorted(concrete_scopes.items())
    ]
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
            "classified_without_provenance_count",
        )
    }
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
