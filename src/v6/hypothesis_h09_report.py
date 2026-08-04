"""H09 hypothesis report — future-option motifs."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from hashlib import sha1
from pathlib import Path
from typing import Any


def _missing_tables(connection: sqlite3.Connection, required: tuple[str, ...]) -> list[str]:
    tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    return [name for name in required if name not in tables]


def _complete_context_key(value: object) -> bool:
    """Reject serialized partial contexts; they cannot support scope claims."""
    if value in (None, ""):
        return False
    text = str(value).strip().lower()
    return bool(text) and "null" not in text and "none" not in text and text not in {"[]", "{}"}


def _is_real_scope(key: object) -> bool:
    """Reject surrogate scope keys — they provide no real evidence."""
    if key is None or str(key).strip() == "" or "null" in str(key).lower():
        return False
    text = str(key).strip().lower()
    return text and "surrogate" not in text and text not in {"[]", "{}"}


def _is_unknown_source(source: object) -> bool:
    """Check if a classification source is unknown/unclassified."""
    if source in (None, "") or str(source).strip() == "":
        return True
    return str(source).lower().startswith("unknown")


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
    from v6.future_options import derive_future_option_memory

    output_dir.mkdir(parents=True, exist_ok=True)
    current_state = Path(memory_dir) / "current_state.sqlite"
    if not already_derived and current_state.exists():
        derive_future_option_memory(memory_dir=memory_dir, run_dir=run_dir)

    if not current_state.exists():
        result: dict[str, Any] = {
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
        observations = [] if "future_option_motif_observations" not in {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()} else [dict(row) for row in conn.execute(
            "SELECT * FROM future_option_motif_observations ORDER BY motif_signature ASC, event_id ASC"
        ).fetchall()]

        milestone_map = dict(conn.execute("SELECT milestone_name, first_global_step FROM higher_order_milestones").fetchall()) if "higher_order_milestones" in {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()} else {}
        summary_row = conn.execute("SELECT value_json FROM memory_summary WHERE key = 'future_option_derivation_summary'").fetchone()
        derivation_summary: dict[str, Any] = json.loads(str(summary_row[0])) if summary_row and summary_row[0] else {}

    # 3.1 — Fix structured-effect counter typo: support BOTH spellings for historical compatibility.
    source_counts: Counter[str] = Counter()
    unknown_event_count = 0
    verified_motif_records: list[dict[str, Any]] = []
    qualifying_emergent_motifs: list[dict[str, Any]] = []

    for row in motifs:
        motif_signature = str(row["motif_signature"])
        motif_type = str(row.get("motif_type") or "unknown")
        provenance_status = str(row.get("provenance_status") or "missing")
        is_emergent = int(row.get("is_emergent") or 0)

        # Count events by classification source (supporting both historical spellings).
        for event in events:
            if str(event.get("classification_source") or "") == motif_type or str(event.get("motif_type") or "") == motif_type:
                source_counts[str(row["classification_source"] or "unknown")] += 1

        # Track unknown motifs.
        if motif_type == "unknown":
            unknown_event_count += 1

        # Build per-motif scientific evidence records (section 3.2).
        verified_obs = [obs for obs in observations if str(obs.get("provenance_status") or "") == "verified"]
        has_verified_observations = any(str(oe.get("motif_signature") or "") == motif_signature for oe in verified_obs)

        # Cross-game and cross-context verification.
        # Cross-game verification requires real scope (not surrogate, not empty).
        verified_cross_game_obs = [obs for obs in verified_obs if str(obs.get("source_game_key")) and str(obs.get("target_game_key")) and _is_real_scope(str(obs.get("source_game_key"))) and _is_real_scope(str(obs.get("target_game_key")))]
        # Cross-context verification requires complete contexts (not surrogate).
        verified_cross_context_obs = [obs for obs in verified_obs if _complete_context_key(str(obs.get("source_context_key"))) and _complete_context_key(str(obs.get("target_context_key")))]

        motif_record: dict[str, Any] = {
            "motif_signature": motif_signature,
            "motif_type": motif_type,
            "is_emergent": is_emergent,
            "provenance_status": provenance_status,
            "has_verified_observation": has_verified_observations,
            "has_verified_cross_game_observation": any(str(og.get("source_game_key")) != str(og.get("target_game_key")) for og in verified_cross_game_obs if str(og.get("motif_signature") or "") == motif_signature),
            "has_verified_cross_context_observation": any(str(oc.get("source_context_key")) != str(oc.get("target_game_key")) for oc in verified_cross_context_obs if str(oc.get("motif_signature") or "") == motif_signature),
            # 3.5 — Verified-event population unknown ratios: count only events with verified provenance_status.
            "verified_event_count": len([e for e in events if str(e.get("classification_provenance_status") or "missing") == "verified"]),
            "unknown_verified_event_count": len([e for e in events if str(e.get("classification_provenance_status") or "missing") == "verified" and _is_unknown_source(str(e.get("classification_source") or ""))]),
        }

        # Classify motifs as qualifying emergent (section 3.3).
        if is_emergent and provenance_status == "verified" and motif_type != "unknown" and has_verified_observations:
            cross_game = any(str(og.get("source_game_key")) != str(og.get("target_game_key")) for og in verified_cross_game_obs if str(og.get("motif_signature") or "") == motif_signature)
            cross_context = any(str(oc.get("source_context_key")) != str(oc.get("target_game_key")) for oc in verified_cross_context_obs if str(oc.get("motif_signature") or "") == motif_signature)
            if (cross_game or cross_context) and len([e for e in events if float(e.get("option_delta") or 0.0)]) > 0:
                qualifying_emergent_motifs.append(motif_record)

        verified_motif_records.append(motif_record)

    # 3.4 — Require motif-type diversity inside the qualifying population.
    qualifying_type_counts: Counter[str] = Counter()
    for record in qualifying_emergent_motifs:
        qualifying_type_counts[str(record["motif_type"])] += 1
    qualifying_motif_type_count = len(qualifying_type_counts)

    # Compute verified-population unknown ratios (section 3.5).
    total_verified_event_count = sum(record["verified_event_count"] for record in verified_motif_records)
    total_unknown_verified_event_count = sum(record["unknown_verified_event_count"] for record in verified_motif_records)
    verified_unknown_event_ratio = (total_unknown_verified_event_count / total_verified_event_count) if total_verified_event_count > 0 else None

    result: dict[str, Any] = {
        "hypothesis_id": "H09",
        "evidence_source": "compact_memory",
        "future_option_event_count": len(events),
        "future_option_motif_count": len(motifs),
        "emergent_future_option_motif_count": sum(1 for r in motifs if int(r.get("is_emergent") or 0) == 1),
        "motif_type_counts": dict(sorted(source_counts.items())),
        "qualifying_emergent_motif_count": len(qualifying_emergent_motifs),
        "qualifying_emergent_motif_signatures": [r["motif_signature"] for r in qualifying_emergent_motifs],
        "qualifying_motif_type_count": qualifying_motif_type_count,
        "qualifying_motif_type_counts": dict(qualifying_type_counts),
        "verified_event_count": total_verified_event_count,
        "verified_unknown_event_count": total_unknown_verified_event_count,
        "verified_unknown_event_ratio": verified_unknown_event_ratio,
        "unknown_motif_source_count": int(source_counts.get("unknown", 0)),
        "unknown_motif_source_ratio": (int(source_counts.get("unknown", 0)) / len(events)) if events else None,
        # 3.1 — Support BOTH spellings for historical compatibility but count each independently.
        "live_delta_event_count": int(source_counts.get("structural_effect", 0)) + int(source_counts.get("structured_effect", 0)),
        "structured_effect_event_count": int(source_counts.get("structured_effect", 0)),
        "text_keyword_event_count": int(source_counts.get("text_keyword", 0)),
        "future_option_edge_event_count": int(source_counts.get("future_option_edge", 0)),
    }

    # H09 VALID decision logic.
    if not events:
        result["decision"] = "INSUFFICIENT_EVIDENCE" if sum(1 for m in motifs if int(m.get("is_emergent") or 0) == 1) > 0 else "INCONCLUSIVE"
    elif not motifs:
        result["decision"] = "INVALID"
    elif qualifying_emergent_motif_count >= 1 and qualifying_motif_type_count >= 2:
        result["decision"] = "VALID"
    else:
        result["decision"] = "PARTIALLY_VALID"

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
    report_path = output_dir / "h09_future_option_motifs_report.json"
    (report_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = (
        f"H09 decision: {result.get('decision')}\n"
        f"future-option events: {result.get('future_option_event_count')}\n"
        f"future-option motifs: {result.get('future_option_motif_count')}\n"
        f"emergent motifs: {result.get('emergent_future_option_motif_count')}\n"
        f"qualifying emergent motifs: {result.get('qualifying_emergent_motif_count')}\n"
        f"motif types: {result.get('motif_type_counts')}"
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
