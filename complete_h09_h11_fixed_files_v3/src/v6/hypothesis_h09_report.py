from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from hashlib import sha1
from pathlib import Path
from typing import Any

from v6.future_options import derive_future_option_memory


def _missing_tables(
    connection: sqlite3.Connection,
    required: tuple[str, ...],
) -> list[str]:
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    return [name for name in required if name not in tables]


def _complete_context_key(value: object) -> bool:
    if value in (None, ""):
        return False
    text = str(value).strip().lower()
    return (
        bool(text)
        and "null" not in text
        and "none" not in text
        and text not in {"[]", "{}"}
    )


def _context_id(value: object) -> str | None:
    if not _complete_context_key(value):
        return None
    return "ctx:" + sha1(
        str(value).encode("utf-8")
    ).hexdigest()[:20]


def _json_list(value: Any) -> list[str]:
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _mean(values: list[Any]) -> float | None:
    cooked = [
        float(value)
        for value in values
        if value is not None
    ]
    return sum(cooked) / len(cooked) if cooked else None


def _is_verified_event(event: dict[str, Any]) -> bool:
    return (
        str(
            event.get(
                "classification_provenance_status"
            )
            or "missing"
        )
        == "verified"
    )


def _is_unknown_event(event: dict[str, Any]) -> bool:
    motif_type = str(
        event.get("motif_type") or "unknown"
    ).strip().lower()
    source = str(
        event.get("classification_source") or "unknown"
    ).strip().lower()
    return motif_type == "unknown" or source.startswith("unknown")


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
        derive_future_option_memory(
            memory_dir=memory_dir,
            run_dir=run_dir,
        )

    if not current_state.exists():
        result = {
            "hypothesis_id": "H09",
            "evidence_source": "compact_memory",
            "decision": "INSUFFICIENT_EVIDENCE",
            "missing_evidence": [
                f"Missing expected compact-memory file: {current_state}"
            ],
            "core_metrics": {},
        }
        _write(output_dir, result)
        return result

    with sqlite3.connect(current_state) as conn:
        conn.row_factory = sqlite3.Row
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
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
                "missing_evidence": [
                    "Missing expected compact-memory table(s): "
                    + ", ".join(missing_tables)
                ],
                "core_metrics": {},
            }
            _write(output_dir, result)
            return result

        events = [
            dict(row)
            for row in conn.execute(
                """
                SELECT *
                FROM future_option_events
                ORDER BY event_id ASC
                """
            ).fetchall()
        ]
        motifs = [
            dict(row)
            for row in conn.execute(
                """
                SELECT *
                FROM future_option_motifs
                ORDER BY motif_signature ASC
                """
            ).fetchall()
        ]
        observations = (
            [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT *
                    FROM future_option_motif_observations
                    ORDER BY motif_signature ASC, event_id ASC
                    """
                ).fetchall()
            ]
            if "future_option_motif_observations" in tables
            else []
        )
        milestone_map = dict(
            conn.execute(
                """
                SELECT milestone_name, first_global_step
                FROM higher_order_milestones
                """
            ).fetchall()
        )
        summary_row = (
            conn.execute(
                """
                SELECT value_json
                FROM memory_summary
                WHERE key = 'future_option_derivation_summary'
                """
            ).fetchone()
            if "memory_summary" in tables
            else None
        )
        try:
            derivation_summary = (
                json.loads(str(summary_row[0]))
                if summary_row and summary_row[0]
                else {}
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            derivation_summary = {}

        stable_contingencies_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM stable_contingencies"
            ).fetchone()[0]
        )
        transformation_families_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM transformation_families"
            ).fetchone()[0]
        )

    events_by_event_id = {
        str(row.get("event_id")): row
        for row in events
        if row.get("event_id") is not None
    }
    observations_by_motif: dict[str, list[dict[str, Any]]] = defaultdict(list)
    verified_observations_by_motif: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for observation in observations:
        signature = str(
            observation.get("motif_signature") or ""
        )
        observations_by_motif[signature].append(observation)
        if str(
            observation.get("provenance_status") or "missing"
        ) == "verified":
            verified_observations_by_motif[signature].append(
                observation
            )

    source_counts = Counter(
        str(row.get("classification_source") or "unknown")
        for row in events
    )
    motif_type_counts = Counter(
        str(row.get("motif_type") or "unknown")
        for row in motifs
    )

    verified_events = [
        event for event in events if _is_verified_event(event)
    ]
    verified_unknown_events = [
        event for event in verified_events if _is_unknown_event(event)
    ]
    verified_unknown_event_ratio = (
        len(verified_unknown_events) / len(verified_events)
        if verified_events
        else None
    )

    verified_observations = [
        row
        for row in observations
        if str(row.get("provenance_status") or "missing")
        == "verified"
    ]
    verified_cross_game_observations = [
        row
        for row in verified_observations
        if row.get("source_game_key") not in (None, "")
        and row.get("target_game_key") not in (None, "")
        and str(row["source_game_key"])
        != str(row["target_game_key"])
        and int(row.get("source_game_is_surrogate") or 0) == 0
        and int(row.get("target_game_is_surrogate") or 0) == 0
    ]
    verified_cross_context_observations = [
        row
        for row in verified_observations
        if _complete_context_key(row.get("source_context_key"))
        and _complete_context_key(row.get("target_context_key"))
        and str(row["source_context_key"])
        != str(row["target_context_key"])
        and int(row.get("source_context_is_surrogate") or 0) == 0
        and int(row.get("target_context_is_surrogate") or 0) == 0
    ]

    motif_records: list[dict[str, Any]] = []
    qualifying_emergent_motifs: list[dict[str, Any]] = []

    for motif in motifs:
        signature = str(
            motif.get("motif_signature") or ""
        )
        motif_type = str(
            motif.get("motif_type") or "unknown"
        )
        motif_observations = observations_by_motif.get(
            signature, []
        )
        verified_motif_observations = (
            verified_observations_by_motif.get(signature, [])
        )

        motif_events: list[dict[str, Any]] = []
        seen_event_ids: set[str] = set()
        for observation in motif_observations:
            event_id = observation.get("event_id")
            if event_id is None:
                continue
            event_key = str(event_id)
            if event_key in seen_event_ids:
                continue
            event = events_by_event_id.get(event_key)
            if event is not None:
                motif_events.append(event)
                seen_event_ids.add(event_key)

        verified_motif_events = [
            event
            for event in motif_events
            if _is_verified_event(event)
            or any(
                str(obs.get("event_id"))
                == str(event.get("event_id"))
                for obs in verified_motif_observations
            )
        ]
        verified_nonzero_delta_events = [
            event
            for event in verified_motif_events
            if abs(float(event.get("option_delta") or 0.0)) > 0.0
        ]
        unknown_verified_events = [
            event
            for event in verified_motif_events
            if _is_unknown_event(event)
        ]

        has_verified_cross_game = any(
            obs.get("source_game_key") not in (None, "")
            and obs.get("target_game_key") not in (None, "")
            and str(obs["source_game_key"])
            != str(obs["target_game_key"])
            and int(
                obs.get("source_game_is_surrogate") or 0
            )
            == 0
            and int(
                obs.get("target_game_is_surrogate") or 0
            )
            == 0
            for obs in verified_motif_observations
        )
        has_verified_cross_context = any(
            _complete_context_key(
                obs.get("source_context_key")
            )
            and _complete_context_key(
                obs.get("target_context_key")
            )
            and str(obs["source_context_key"])
            != str(obs["target_context_key"])
            and int(
                obs.get("source_context_is_surrogate") or 0
            )
            == 0
            and int(
                obs.get("target_context_is_surrogate") or 0
            )
            == 0
            for obs in verified_motif_observations
        )

        record = {
            "motif_signature": signature,
            "motif_type": motif_type,
            "is_emergent":
                int(motif.get("is_emergent") or 0),
            "provenance_status":
                str(
                    motif.get("provenance_status")
                    or "missing"
                ),
            "has_verified_observation":
                bool(verified_motif_observations),
            "has_verified_cross_game_observation":
                has_verified_cross_game,
            "has_verified_cross_context_observation":
                has_verified_cross_context,
            "verified_event_count":
                len(verified_motif_events),
            "verified_nonzero_option_delta_event_count":
                len(verified_nonzero_delta_events),
            "unknown_verified_event_count":
                len(unknown_verified_events),
            "classification_sources": sorted(
                {
                    str(
                        event.get("classification_source")
                        or "unknown"
                    )
                    for event in verified_motif_events
                }
            ),
        }
        motif_records.append(record)

        if (
            record["is_emergent"] == 1
            and record["provenance_status"] == "verified"
            and motif_type != "unknown"
            and record["has_verified_observation"]
            and (
                record[
                    "has_verified_cross_game_observation"
                ]
                or record[
                    "has_verified_cross_context_observation"
                ]
            )
            and record[
                "verified_nonzero_option_delta_event_count"
            ]
            >= 1
        ):
            qualifying_emergent_motifs.append(record)

    qualifying_motif_type_counts = Counter(
        str(row["motif_type"])
        for row in qualifying_emergent_motifs
    )

    unknown_event_count = sum(
        1
        for row in events
        if str(row.get("motif_type") or "unknown")
        == "unknown"
    )
    unknown_motif_count = int(
        motif_type_counts.get("unknown", 0)
    )
    verified_motifs = [
        row
        for row in motifs
        if str(row.get("provenance_status") or "missing")
        == "verified"
    ]
    proxy_motifs = [
        row
        for row in motifs
        if str(row.get("provenance_status") or "missing")
        == "proxy"
    ]
    missing_motifs = [
        row
        for row in motifs
        if str(row.get("provenance_status") or "missing")
        not in {"verified", "proxy"}
    ]

    cross_context_motif_count = len(
        {
            str(row["motif_signature"])
            for row in verified_cross_context_observations
        }
    )
    cross_game_motif_count = len(
        {
            str(row["motif_signature"])
            for row in verified_cross_game_observations
        }
    )

    result: dict[str, Any] = {
        "hypothesis_id": "H09",
        "evidence_source": "compact_memory",
        "future_option_event_count": len(events),
        "future_option_motif_count": len(motifs),
        "emergent_future_option_motif_count": sum(
            1
            for row in motifs
            if int(row.get("is_emergent") or 0) == 1
        ),
        "motif_type_counts":
            dict(sorted(motif_type_counts.items())),
        "motif_type_source_counts":
            dict(sorted(source_counts.items())),
        "classification_source_counts":
            dict(sorted(source_counts.items())),
        "classification_provenance_status_counts": dict(
            sorted(
                Counter(
                    str(
                        row.get(
                            "classification_provenance_status"
                        )
                        or "missing"
                    )
                    for row in events
                ).items()
            )
        ),
        "unknown_motif_source_count":
            int(source_counts.get("unknown", 0)),
        "unknown_motif_source_ratio": (
            int(source_counts.get("unknown", 0))
            / len(events)
            if events
            else None
        ),
        "cross_context_motif_count":
            cross_context_motif_count,
        "cross_game_motif_count":
            cross_game_motif_count,
        "mean_abs_option_delta": _mean(
            [
                abs(float(row.get("option_delta") or 0.0))
                for row in events
            ]
        ),
        "max_abs_option_delta": max(
            (
                abs(float(row.get("option_delta") or 0.0))
                for row in events
            ),
            default=None,
        ),
        "mean_motif_stability_score": _mean(
            [
                row.get("motif_stability_score")
                for row in motifs
            ]
        ),
        "unknown_motif_count": unknown_motif_count,
        "unknown_motif_ratio": (
            unknown_motif_count / len(motifs)
            if motifs
            else None
        ),
        "unknown_motif_event_count": unknown_event_count,
        "unknown_motif_event_ratio": (
            unknown_event_count / len(events)
            if events
            else None
        ),
        "live_delta_event_count":
            int(source_counts.get("live_delta", 0))
            + int(source_counts.get("live_delta_rule", 0)),
        "structured_effect_event_count":
            int(source_counts.get("structural_effect", 0))
            + int(source_counts.get("structured_effect", 0)),
        "text_keyword_event_count":
            int(source_counts.get("text_keyword", 0)),
        "future_option_edge_event_count":
            int(source_counts.get("future_option_edge", 0)),
        "verified_event_count": len(verified_events),
        "verified_unknown_event_count":
            len(verified_unknown_events),
        "verified_unknown_event_ratio":
            verified_unknown_event_ratio,
        "verified_motif_count": len(verified_motifs),
        "proxy_motif_count": len(proxy_motifs),
        "missing_motif_count": len(missing_motifs),
        "verified_observation_count":
            len(verified_observations),
        "proxy_observation_count": sum(
            1
            for row in observations
            if str(
                row.get("provenance_status") or "missing"
            )
            == "proxy"
        ),
        "verified_cross_game_observation_count":
            len(verified_cross_game_observations),
        "verified_cross_context_observation_count":
            len(verified_cross_context_observations),
        "incomplete_context_observation_count": sum(
            1
            for row in observations
            if not _complete_context_key(
                row.get("source_context_key")
            )
            or not _complete_context_key(
                row.get("target_context_key")
            )
        ),
        "surrogate_context_observation_count": sum(
            1
            for row in observations
            if int(
                row.get("source_context_is_surrogate") or 0
            )
            or int(
                row.get("target_context_is_surrogate") or 0
            )
        ),
        "qualifying_emergent_motif_count":
            len(qualifying_emergent_motifs),
        "qualifying_emergent_motif_signatures": [
            str(row["motif_signature"])
            for row in qualifying_emergent_motifs
        ],
        "qualifying_motif_type_count":
            len(qualifying_motif_type_counts),
        "qualifying_motif_type_counts":
            dict(sorted(qualifying_motif_type_counts.items())),
        "motif_scientific_evidence_records":
            motif_records[:200],
        "motif_scope_summary": {
            "observation_count": len(observations),
            "verified_cross_game_observation_count":
                len(verified_cross_game_observations),
            "verified_cross_context_observation_count":
                len(verified_cross_context_observations),
        },
        "motif_scope_sample": [
            {
                "motif_signature":
                    row.get("motif_signature"),
                "event_id": row.get("event_id"),
                "source_game_key":
                    row.get("source_game_key"),
                "target_game_key":
                    row.get("target_game_key"),
                "source_context_id":
                    _context_id(
                        row.get("source_context_key")
                    ),
                "target_context_id":
                    _context_id(
                        row.get("target_context_key")
                    ),
                "provenance_status":
                    row.get("provenance_status"),
            }
            for row in observations[:200]
        ],
        "first_future_option_event_step":
            milestone_map.get(
                "first_future_option_event_step"
            ),
        "first_emergent_future_option_motif_step":
            milestone_map.get(
                "first_emergent_future_option_motif_step"
            ),
        "stable_contingencies_count":
            stable_contingencies_count,
        "transformation_families_count":
            transformation_families_count,
        "missing_evidence": [],
    }

    for key, value in derivation_summary.items():
        result.setdefault(key, value)

    if not events:
        substrate_available = (
            stable_contingencies_count > 0
            or transformation_families_count > 0
        )
        result["decision"] = (
            "INSUFFICIENT_EVIDENCE"
            if substrate_available
            or result["emergent_future_option_motif_count"] > 0
            else "INCONCLUSIVE"
        )
        result["missing_evidence"] = [
            (
                'Future-option derivation produced zero events despite available substrate.'
                if substrate_available
                else "No future-option events available."
            )
        ]
    elif not motifs:
        result["decision"] = "INVALID"
        result["missing_evidence"] = [
            "Future-option events exist but no motifs were derived."
        ]
    elif (
        len(qualifying_emergent_motifs) >= 1
        and len(qualifying_motif_type_counts) >= 2
        and (
            verified_unknown_event_ratio is None
            or verified_unknown_event_ratio <= 0.20
        )
    ):
        result["decision"] = "VALID"
    elif result["emergent_future_option_motif_count"] <= 0:
        result["decision"] = "PARTIALLY_VALID"
        result["missing_evidence"] = [
            "No emergent future-option motif available."
        ]
    else:
        result["decision"] = "PARTIALLY_VALID"
        result["missing_evidence"] = [
            "No motif population satisfies the complete "
            "verified H09 evidence chain."
        ]

    result["core_metrics"] = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "core_metrics",
            "motif_scope_sample",
            "motif_scientific_evidence_records",
        }
    }

    _write_observations(output_dir, observations)
    _write(output_dir, result)
    return result


def _write(
    output_dir: Path,
    result: dict[str, Any],
) -> None:
    (
        output_dir / "h09_future_option_motifs_report.json"
    ).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    text = (
        f"H09 decision: {result.get('decision')}\n"
        "future-option events: "
        f"{result.get('future_option_event_count')}\n"
        "future-option motifs: "
        f"{result.get('future_option_motif_count')}\n"
        "emergent motifs: "
        f"{result.get('emergent_future_option_motif_count')}\n"
        "qualifying emergent motifs: "
        f"{result.get('qualifying_emergent_motif_count')}\n"
        f"motif types: {result.get('motif_type_counts')}\n"
    )
    (
        output_dir / "h09_future_option_motifs_report.txt"
    ).write_text(text, encoding="utf-8")
    (
        output_dir / "h09_future_option_motifs.md"
    ).write_text(
        "```\n" + text + "```\n",
        encoding="utf-8",
    )


def _write_observations(
    output_dir: Path,
    observations: list[dict[str, Any]],
) -> None:
    with (
        output_dir / "h09_motif_observations.jsonl"
    ).open("w", encoding="utf-8") as handle:
        for row in observations:
            payload = dict(row)
            payload["source_context_id"] = _context_id(
                payload.pop("source_context_key", None)
            )
            payload["target_context_id"] = _context_id(
                payload.pop("target_context_key", None)
            )
            handle.write(
                json.dumps(payload, sort_keys=True) + "\n"
            )
