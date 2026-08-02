from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from hashlib import sha1
from pathlib import Path
from typing import Any

from v6.future_options import derive_future_option_memory


DEFAULT_PROVENANCE_SAMPLE_LIMIT = 200
DEFAULT_MAX_MAIN_REPORT_BYTES = 5_000_000


def _missing_tables(connection: sqlite3.Connection, required: tuple[str, ...]) -> list[str]:
    tables = {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    return [name for name in required if name not in tables]


def _context_id(context_key: object) -> str | None:
    if context_key in (None, ""):
        return None
    return "ctx:" + sha1(str(context_key).encode("utf-8")).hexdigest()[:20]


def _transfer_pair_id(
    source_game_key: object,
    target_game_key: object,
    source_context_id: str | None,
    target_context_id: str | None,
) -> str:
    payload = {
        "source_game_key": source_game_key,
        "target_game_key": target_game_key,
        "source_context_id": source_context_id,
        "target_context_id": target_context_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "h11pair:" + sha1(encoded.encode("utf-8")).hexdigest()[:20]


def _is_fully_verified(row: dict[str, object]) -> bool:
    return (
        str(row.get("motif_provenance_status") or "missing") == "verified"
        and str(row.get("transfer_provenance_status") or "missing") == "verified"
        and str(row.get("concept_validation_status") or "missing") == "verified"
    )


def _is_missing_chain(row: dict[str, object]) -> bool:
    return "missing" in {
        str(row.get("motif_provenance_status") or "missing"),
        str(row.get("transfer_provenance_status") or "missing"),
        str(row.get("concept_validation_status") or "missing"),
    }


def _int(value: object) -> int:
    return int(value or 0)


def _float(value: object) -> float | None:
    return None if value is None else float(value)


def _chain_output_row(row: dict[str, object]) -> tuple[dict[str, object], tuple[str, str, str, str], tuple[str, str, str, str]]:
    source_context_id = _context_id(row.get("source_context_key"))
    target_context_id = _context_id(row.get("target_context_key"))
    source_game_key = row.get("source_game_key")
    target_game_key = row.get("target_game_key")
    output = {
        "motif_signature": row.get("motif_signature"),
        "role_signature": row.get("role_signature"),
        "concept_signature": row.get("concept_signature"),
        "source_role_signature": row.get("source_role_signature") or row.get("role_signature"),
        "source_game_key": source_game_key,
        "target_game_key": target_game_key,
        "source_interaction_id": row.get("source_interaction_id"),
        "target_interaction_id": row.get("target_interaction_id"),
        "source_context_id": source_context_id,
        "target_context_id": target_context_id,
        "transfer_pair_id": _transfer_pair_id(source_game_key, target_game_key, source_context_id, target_context_id),
        "provenance_mode": row.get("provenance_mode"),
        "transfer_scope": row.get("transfer_scope") or "same_scope",
        "source_game_is_surrogate": _int(row.get("source_game_is_surrogate")),
        "target_game_is_surrogate": _int(row.get("target_game_is_surrogate")),
        "source_context_is_surrogate": _int(row.get("source_context_is_surrogate")),
        "target_context_is_surrogate": _int(row.get("target_context_is_surrogate")),
        "source_game_resolution_source": row.get("source_game_resolution_source"),
        "target_game_resolution_source": row.get("target_game_resolution_source"),
        "source_context_resolution_source": row.get("source_context_resolution_source"),
        "target_context_resolution_source": row.get("target_context_resolution_source"),
        "motif_provenance_status": row.get("motif_provenance_status") or "missing",
        "transfer_provenance_status": row.get("transfer_provenance_status") or "missing",
        "concept_validation_status": row.get("concept_validation_status") or "missing",
        "concept_resolution_mode": row.get("concept_resolution_mode") or "missing",
        "concept_resolution_path": row.get("concept_resolution_path") or "unresolved",
        "shared_carrier_count": _int(row.get("shared_carrier_count")),
        "shared_family_count": _int(row.get("shared_family_count")),
        "transfer_attempt_count": _int(row.get("transfer_attempt_count")),
        "successful_transfer_count": _int(row.get("successful_transfer_count")),
        "strong_transfer_success_count": _int(row.get("strong_transfer_success_count")),
        "promoted_concept_count": _int(row.get("promoted_concept_count")),
        "mean_transfer_score": _float(row.get("mean_transfer_score")),
        "mean_best_margin": _float(row.get("mean_best_margin")),
        "motif_provenance_resolution_path": row.get("motif_provenance_resolution_path") or "unresolved",
        "first_seen_global_step": row.get("first_seen_global_step"),
        "last_seen_global_step": row.get("last_seen_global_step"),
    }
    game_pair = (str(source_game_key or ""), str(target_game_key or ""), "", "")
    context_pair = (source_context_id or "", target_context_id or "", "", "")
    return output, game_pair, context_pair


def _pair_accumulator() -> dict[str, object]:
    return {
        "link_count": 0,
        "motifs": set(),
        "roles": set(),
        "concepts": set(),
        "transfer_attempt_count": 0,
        "successful_transfer_count": 0,
        "strong_transfer_success_count": 0,
        "fully_verified_chain_count": 0,
    }


def _add_pair_row(bucket: dict[str, object], output: dict[str, object], fully_verified: bool) -> None:
    bucket["link_count"] = _int(bucket["link_count"]) + 1
    for set_key, row_key in (("motifs", "motif_signature"), ("roles", "role_signature"), ("concepts", "concept_signature")):
        value = output.get(row_key)
        if value not in (None, "", "__none__"):
            bucket[set_key].add(str(value))  # type: ignore[union-attr]
    for key in ("transfer_attempt_count", "successful_transfer_count", "strong_transfer_success_count"):
        bucket[key] = _int(bucket[key]) + _int(output.get(key))
    bucket["fully_verified_chain_count"] = _int(bucket["fully_verified_chain_count"]) + int(fully_verified)


def _write_pair_artifact(
    path: Path,
    pairs: dict[tuple[str, str, str, str], dict[str, object]],
    *,
    context_pairs: bool,
) -> int:
    with path.open("w", encoding="utf-8") as handle:
        for key in sorted(pairs):
            values = pairs[key]
            attempt_count = _int(values["transfer_attempt_count"])
            payload = {
                ("source_context_id" if context_pairs else "source_game_key"): key[0],
                ("target_context_id" if context_pairs else "target_game_key"): key[1],
                "link_count": _int(values["link_count"]),
                "motif_count": len(values["motifs"]),
                "role_count": len(values["roles"]),
                "concept_count": len(values["concepts"]),
                "transfer_attempt_count": attempt_count,
                "successful_transfer_count": _int(values["successful_transfer_count"]),
                "strong_transfer_success_count": _int(values["strong_transfer_success_count"]),
                "success_rate": (_int(values["successful_transfer_count"]) / attempt_count) if attempt_count else None,
                "fully_verified_chain_count": _int(values["fully_verified_chain_count"]),
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return path.stat().st_size


def _scalar(connection: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()) -> object:
    row = connection.execute(sql, params).fetchone()
    return None if row is None else row[0]


def evaluate_h11_future_option_transfer_concepts(
    *,
    memory_dir: Path,
    run_dir: Path | None,
    output_dir: Path,
    already_derived: bool = False,
    provenance_sample_limit: int = DEFAULT_PROVENANCE_SAMPLE_LIMIT,
    write_full_provenance_jsonl: bool = True,
    max_main_report_bytes: int = DEFAULT_MAX_MAIN_REPORT_BYTES,
) -> dict[str, object]:
    if provenance_sample_limit < 0:
        raise ValueError("h11 provenance sample limit must be non-negative")
    if max_main_report_bytes <= 0:
        raise ValueError("max_h11_main_report_bytes must be positive")
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
        _write(output_dir, result, max_main_report_bytes=max_main_report_bytes)
        return result

    full_path = output_dir / "h11_transfer_chain_provenance.jsonl"
    game_pair_path = output_dir / "h11_transfer_by_game_pair.jsonl"
    context_pair_path = output_dir / "h11_transfer_by_context_pair.jsonl"
    context_lookup_path = output_dir / "h11_context_lookup.jsonl"
    with sqlite3.connect(current_state) as conn:
        conn.row_factory = sqlite3.Row
        missing_tables = _missing_tables(
            conn,
            ("future_option_transfer_links", "future_option_motifs", "role_transfer_attempts", "concept_candidates"),
        )
        if missing_tables:
            result = {
                "hypothesis_id": "H11",
                "evidence_source": "compact_memory",
                "decision": "INSUFFICIENT_EVIDENCE",
                "missing_evidence": [f"Missing expected compact-memory table(s): {', '.join(missing_tables)}"],
                "core_metrics": {},
            }
            _write(output_dir, result, max_main_report_bytes=max_main_report_bytes)
            return result

        full_condition = """
            COALESCE(l.motif_provenance_status, 'missing') = 'verified'
            AND COALESCE(l.transfer_provenance_status, 'missing') = 'verified'
            AND COALESCE(l.concept_validation_status, 'missing') = 'verified'
        """
        cross_game_condition = f"""
            {full_condition}
            AND COALESCE(l.provenance_mode, '') = 'single_source'
            AND l.source_role_signature IS NOT NULL
            AND l.source_game_key IS NOT NULL AND l.target_game_key IS NOT NULL
            AND l.source_context_key IS NOT NULL AND l.target_context_key IS NOT NULL
            AND COALESCE(l.source_game_is_surrogate, 0) = 0
            AND COALESCE(l.target_game_is_surrogate, 0) = 0
            AND l.source_game_key != l.target_game_key
        """
        aggregate = conn.execute(
            f"""
            SELECT
                COUNT(*) AS total_links,
                SUM(CASE WHEN {full_condition} THEN 1 ELSE 0 END) AS fully_verified_links,
                COUNT(DISTINCT CASE WHEN l.transfer_attempt_count > 0 THEN l.motif_signature END) AS all_motifs_with_transfer,
                COUNT(DISTINCT CASE WHEN {full_condition} AND l.transfer_attempt_count > 0 THEN l.motif_signature END) AS verified_motifs_with_transfer,
                COUNT(DISTINCT CASE WHEN l.strong_transfer_success_count > 0 THEN l.motif_signature END) AS all_motifs_with_strong,
                COUNT(DISTINCT CASE WHEN {full_condition} AND l.strong_transfer_success_count > 0 THEN l.motif_signature END) AS verified_motifs_with_strong,
                COUNT(DISTINCT CASE WHEN l.promoted_concept_count > 0 THEN l.motif_signature END) AS all_motifs_with_promoted,
                COUNT(DISTINCT CASE WHEN {full_condition} AND l.promoted_concept_count > 0 THEN l.motif_signature END) AS verified_motifs_with_promoted,
                SUM(CASE WHEN {full_condition} THEN COALESCE(l.transfer_attempt_count, 0) ELSE 0 END) AS total_attempts,
                SUM(CASE WHEN {full_condition} THEN COALESCE(l.successful_transfer_count, 0) ELSE 0 END) AS total_successes,
                SUM(CASE WHEN {full_condition} THEN COALESCE(l.strong_transfer_success_count, 0) ELSE 0 END) AS total_strong,
                SUM(CASE WHEN COALESCE(m.is_emergent, 0) = 1 THEN 1 ELSE 0 END) AS all_emergent_links,
                SUM(CASE WHEN COALESCE(m.is_emergent, 0) = 1 AND {full_condition} THEN 1 ELSE 0 END) AS fully_verified_emergent_links,
                COUNT(DISTINCT CASE WHEN COALESCE(m.is_emergent, 0) = 1 AND {full_condition} AND l.transfer_attempt_count > 0 THEN l.motif_signature END) AS emergent_motifs_with_transfer,
                COUNT(DISTINCT CASE WHEN COALESCE(m.is_emergent, 0) = 1 AND {full_condition} AND l.strong_transfer_success_count > 0 THEN l.motif_signature END) AS emergent_motifs_with_strong,
                COUNT(DISTINCT CASE WHEN COALESCE(m.is_emergent, 0) = 1 AND {full_condition} AND l.promoted_concept_count > 0 THEN l.motif_signature END) AS emergent_motifs_with_promoted,
                SUM(CASE WHEN COALESCE(m.is_emergent, 0) = 1 AND {full_condition} THEN COALESCE(l.transfer_attempt_count, 0) ELSE 0 END) AS emergent_attempts,
                SUM(CASE WHEN COALESCE(m.is_emergent, 0) = 1 AND {full_condition} THEN COALESCE(l.successful_transfer_count, 0) ELSE 0 END) AS emergent_successes,
                SUM(CASE WHEN COALESCE(m.is_emergent, 0) = 1 AND {full_condition} THEN COALESCE(l.strong_transfer_success_count, 0) ELSE 0 END) AS emergent_strong,
                SUM(CASE WHEN COALESCE(m.is_emergent, 0) != 1 THEN 1 ELSE 0 END) AS non_emergent_links,
                COUNT(DISTINCT CASE WHEN COALESCE(m.is_emergent, 0) != 1 AND l.strong_transfer_success_count > 0 THEN l.motif_signature END) AS non_emergent_motifs_with_strong,
                COUNT(DISTINCT CASE WHEN COALESCE(m.is_emergent, 0) != 1 AND l.promoted_concept_count > 0 THEN l.motif_signature END) AS non_emergent_motifs_with_promoted,
                SUM(CASE WHEN {cross_game_condition} THEN 1 ELSE 0 END) AS verified_cross_game_links,
                COUNT(DISTINCT CASE WHEN {cross_game_condition} THEN l.motif_signature END) AS verified_cross_game_motifs,
                COUNT(DISTINCT CASE WHEN {cross_game_condition} THEN l.source_game_key || char(31) || l.target_game_key || char(31) || l.source_context_key || char(31) || l.target_context_key END) AS verified_cross_game_pairs,
                SUM(CASE WHEN COALESCE(l.provenance_mode, '') != 'single_source' THEN 1 ELSE 0 END) AS unverified_cross_game_links,
                SUM(CASE WHEN COALESCE(l.transfer_provenance_status, 'missing') = 'verified' THEN 1 ELSE 0 END) AS verified_concrete_links,
                SUM(CASE WHEN COALESCE(l.transfer_provenance_status, 'missing') = 'resolved_with_surrogate' THEN 1 ELSE 0 END) AS surrogate_resolved_links,
                SUM(CASE WHEN COALESCE(l.transfer_provenance_status, 'missing') = 'proxy' THEN 1 ELSE 0 END) AS proxy_links,
                SUM(CASE WHEN {full_condition} AND COALESCE(l.source_game_is_surrogate, 0) = 0 AND COALESCE(l.target_game_is_surrogate, 0) = 0 AND l.source_game_key != l.target_game_key THEN 1 ELSE 0 END) AS real_cross_game_links,
                SUM(CASE WHEN {full_condition} AND COALESCE(l.source_context_is_surrogate, 0) = 0 AND COALESCE(l.target_context_is_surrogate, 0) = 0 AND l.source_context_key != l.target_context_key THEN 1 ELSE 0 END) AS real_cross_context_links,
                SUM(CASE WHEN COALESCE(l.source_game_is_surrogate, 0) = 1 OR COALESCE(l.target_game_is_surrogate, 0) = 1 THEN 1 ELSE 0 END) AS surrogate_game_links,
                SUM(CASE WHEN COALESCE(l.source_context_is_surrogate, 0) = 1 OR COALESCE(l.target_context_is_surrogate, 0) = 1 THEN 1 ELSE 0 END) AS surrogate_context_links,
                COUNT(DISTINCT CASE WHEN COALESCE(l.transfer_provenance_status, 'missing') = 'verified' AND l.source_game_key IS NOT NULL AND l.target_game_key IS NOT NULL AND l.source_context_key IS NOT NULL AND l.target_context_key IS NOT NULL THEN l.source_game_key || char(31) || l.target_game_key || char(31) || l.source_context_key || char(31) || l.target_context_key END) AS verified_transfer_pairs
            FROM future_option_transfer_links l
            LEFT JOIN future_option_motifs m ON l.motif_signature = m.motif_signature
            """
        ).fetchone()
        aggregate = dict(aggregate) if aggregate is not None else {}
        emergent_motifs = _int(_scalar(conn, "SELECT COUNT(*) FROM future_option_motifs WHERE COALESCE(is_emergent, 0) = 1"))
        future_option_motif_count = _int(_scalar(conn, "SELECT COUNT(*) FROM future_option_motifs"))
        successful_role_transfer_count = _int(_scalar(conn, "SELECT COUNT(*) FROM role_transfer_attempts WHERE COALESCE(reuse_success, 0) = 1 AND provenance_mode = 'single_source'"))
        promoted_concept_count = _int(_scalar(
            conn,
            """
            SELECT COUNT(*) FROM concept_candidates AS candidate
            LEFT JOIN concept_promotion_state AS persistent
              ON persistent.concept_signature = candidate.concept_signature
            WHERE COALESCE(persistent.currently_promoted, candidate.is_promoted, 0) = 1
            """,
        ))
        summary_row = conn.execute("SELECT value_json FROM memory_summary WHERE key = 'future_option_derivation_summary'").fetchone()
        derivation_summary = json.loads(str(summary_row[0])) if summary_row and summary_row[0] else {}

        samples: list[dict[str, object]] = []
        context_lookup: dict[str, str] = {}
        provenance_context_ids: set[str] = set()
        game_pairs: dict[tuple[str, str, str, str], dict[str, object]] = {}
        context_pairs: dict[tuple[str, str, str, str], dict[str, object]] = {}
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
        partially_verified_emergent = 0
        unverified_emergent = 0
        streamed_count = 0
        cursor = conn.execute(
            """
            SELECT l.*, COALESCE(m.is_emergent, 0) AS is_emergent
            FROM future_option_transfer_links l
            LEFT JOIN future_option_motifs m ON l.motif_signature = m.motif_signature
            ORDER BY l.motif_signature ASC, l.role_signature ASC, l.concept_signature ASC,
                     l.source_game_key ASC, l.target_game_key ASC,
                     l.source_context_key ASC, l.target_context_key ASC
            """
        )
        if not write_full_provenance_jsonl and full_path.exists():
            full_path.unlink()
        with (full_path.open("w", encoding="utf-8") if write_full_provenance_jsonl else _NullWriter()) as full_handle:
            for sqlite_row in cursor:
                row = dict(sqlite_row)
                output, game_pair, context_pair = _chain_output_row(row)
                streamed_count += 1
                fully_verified = _is_fully_verified(row)
                if str(row.get("transfer_provenance_status") or "missing") == "verified":
                    assert str(row.get("provenance_mode") or "") == "single_source"
                    assert row.get("source_role_signature")
                    assert row.get("source_game_key") and row.get("target_game_key")
                    assert row.get("source_context_key") and row.get("target_context_key")
                    real_cross_game = (
                        not _int(row.get("source_game_is_surrogate"))
                        and not _int(row.get("target_game_is_surrogate"))
                        and str(row["source_game_key"]) != str(row["target_game_key"])
                    )
                    real_cross_context = (
                        not _int(row.get("source_context_is_surrogate"))
                        and not _int(row.get("target_context_is_surrogate"))
                        and str(row["source_context_key"]) != str(row["target_context_key"])
                    )
                    assert real_cross_game or real_cross_context
                motif_status = str(output["motif_provenance_status"])
                transfer_status = str(output["transfer_provenance_status"])
                concept_status = str(output["concept_validation_status"])
                state_key = f"{motif_status}_{transfer_status}_{concept_status}"
                chain_state_counts[state_key if state_key in chain_state_counts else "other"] += 1
                blocked_by_motif_provenance += int(motif_status != "verified")
                blocked_by_transfer_provenance += int(transfer_status != "verified")
                blocked_by_concept_validation += int(concept_status != "verified")
                blocked_by_missing_concept += int(concept_status == "missing")
                if _int(row.get("is_emergent")) == 1 and not fully_verified:
                    if _is_missing_chain(row):
                        unverified_emergent += 1
                    else:
                        partially_verified_emergent += 1
                source_context_key = row.get("source_context_key")
                target_context_key = row.get("target_context_key")
                for context_id, context_key in ((_context_id(source_context_key), source_context_key), (_context_id(target_context_key), target_context_key)):
                    if context_id is not None and context_key not in (None, ""):
                        previous = context_lookup.setdefault(context_id, str(context_key))
                        assert previous == str(context_key)
                        if write_full_provenance_jsonl:
                            provenance_context_ids.add(context_id)
                if game_pair[0] and game_pair[1]:
                    _add_pair_row(game_pairs.setdefault(game_pair, _pair_accumulator()), output, fully_verified)
                if context_pair[0] and context_pair[1]:
                    _add_pair_row(context_pairs.setdefault(context_pair, _pair_accumulator()), output, fully_verified)
                if len(samples) < provenance_sample_limit:
                    samples.append(output)
                if write_full_provenance_jsonl:
                    full_handle.write(json.dumps(output, sort_keys=True) + "\n")

    with context_lookup_path.open("w", encoding="utf-8") as handle:
        for context_id in sorted(context_lookup):
            handle.write(json.dumps({"context_id": context_id, "context_key": context_lookup[context_id]}, sort_keys=True) + "\n")
    provenance_bytes = full_path.stat().st_size if write_full_provenance_jsonl else 0
    game_pair_bytes = _write_pair_artifact(game_pair_path, game_pairs, context_pairs=False)
    context_pair_bytes = _write_pair_artifact(context_pair_path, context_pairs, context_pairs=True)
    context_lookup_bytes = context_lookup_path.stat().st_size
    total_links = _int(aggregate.get("total_links"))
    assert streamed_count == total_links
    assert len(samples) <= provenance_sample_limit
    assert all(
        context_id in context_lookup
        for sample in samples
        for context_id in (sample.get("source_context_id"), sample.get("target_context_id"))
        if context_id is not None
    )
    assert provenance_context_ids <= set(context_lookup)

    fully_verified_emergent = _int(aggregate.get("fully_verified_emergent_links"))
    all_emergent = _int(aggregate.get("all_emergent_links"))
    result: dict[str, object] = {
        "hypothesis_id": "H11",
        "evidence_source": "compact_memory",
        "future_option_transfer_link_count": total_links,
        "verified_future_option_transfer_count": _int(aggregate.get("fully_verified_links")),
        "future_option_motif_count": future_option_motif_count,
        "all_motifs_with_transfer_count": _int(aggregate.get("all_motifs_with_transfer")),
        "verified_motifs_with_transfer_count": _int(aggregate.get("verified_motifs_with_transfer")),
        "all_motifs_with_strong_transfer_count": _int(aggregate.get("all_motifs_with_strong")),
        "verified_motifs_with_strong_transfer_count": _int(aggregate.get("verified_motifs_with_strong")),
        "all_motifs_with_promoted_concept_count": _int(aggregate.get("all_motifs_with_promoted")),
        "verified_motifs_with_promoted_concept_count": _int(aggregate.get("verified_motifs_with_promoted")),
        "motif_transfer_success_rate": (_int(aggregate.get("total_successes")) / _int(aggregate.get("total_attempts"))) if _int(aggregate.get("total_attempts")) else None,
        "motif_strong_transfer_success_rate": (_int(aggregate.get("total_strong")) / _int(aggregate.get("total_attempts"))) if _int(aggregate.get("total_attempts")) else None,
        "promoted_concept_motif_count": _int(aggregate.get("verified_motifs_with_promoted")),
        "emergent_future_option_motif_count": emergent_motifs,
        "all_emergent_motif_transfer_link_count": all_emergent,
        "emergent_motif_transfer_link_count": all_emergent,
        "fully_verified_emergent_chain_count": fully_verified_emergent,
        "partially_verified_emergent_chain_count": partially_verified_emergent,
        "unverified_emergent_chain_count": unverified_emergent,
        "emergent_motifs_with_transfer_count": _int(aggregate.get("emergent_motifs_with_transfer")),
        "emergent_motifs_with_strong_transfer_count": _int(aggregate.get("emergent_motifs_with_strong")),
        "emergent_motifs_with_promoted_concept_count": _int(aggregate.get("emergent_motifs_with_promoted")),
        "emergent_motif_transfer_success_rate": (_int(aggregate.get("emergent_successes")) / _int(aggregate.get("emergent_attempts"))) if _int(aggregate.get("emergent_attempts")) else None,
        "emergent_motif_strong_transfer_success_rate": (_int(aggregate.get("emergent_strong")) / _int(aggregate.get("emergent_attempts"))) if _int(aggregate.get("emergent_attempts")) else None,
        "promoted_concept_emergent_motif_count": _int(aggregate.get("emergent_motifs_with_promoted")),
        "non_emergent_motif_transfer_link_count": _int(aggregate.get("non_emergent_links")),
        "non_emergent_motifs_with_strong_transfer_count": _int(aggregate.get("non_emergent_motifs_with_strong")),
        "non_emergent_motifs_with_promoted_concept_count": _int(aggregate.get("non_emergent_motifs_with_promoted")),
        "successful_role_transfer_count": successful_role_transfer_count,
        "verified_cross_game_future_option_transfer_count": _int(aggregate.get("verified_cross_game_links")),
        "verified_cross_game_motif_transfer_count": _int(aggregate.get("verified_cross_game_links")),
        "verified_cross_game_link_count": _int(aggregate.get("verified_cross_game_links")),
        "verified_cross_game_motif_count": _int(aggregate.get("verified_cross_game_motifs")),
        "verified_cross_game_pair_count": _int(aggregate.get("verified_cross_game_pairs")),
        "unverified_cross_game_motif_transfer_count": _int(aggregate.get("unverified_cross_game_links")),
        "verified_concrete_transfer_link_count": _int(aggregate.get("verified_concrete_links")),
        "verified_real_transfer_link_count": _int(aggregate.get("verified_concrete_links")),
        "surrogate_resolved_transfer_link_count": _int(aggregate.get("surrogate_resolved_links")),
        "proxy_transfer_link_count": _int(aggregate.get("proxy_links")),
        "real_cross_game_link_count": _int(aggregate.get("real_cross_game_links")),
        "real_cross_context_link_count": _int(aggregate.get("real_cross_context_links")),
        "surrogate_game_link_count": _int(aggregate.get("surrogate_game_links")),
        "surrogate_context_link_count": _int(aggregate.get("surrogate_context_links")),
        "verified_transfer_pair_count": _int(aggregate.get("verified_transfer_pairs")),
        "distinct_source_target_pair_count": _int(aggregate.get("verified_transfer_pairs")),
        "roles_with_direct_concept_links": _int(derivation_summary.get("roles_with_direct_concept_links")),
        "roles_resolved_via_shared_carrier": _int(derivation_summary.get("roles_resolved_via_shared_carrier")),
        "roles_resolved_via_shared_family": _int(derivation_summary.get("roles_resolved_via_shared_family")),
        "roles_resolved_via_carrier_and_family": _int(derivation_summary.get("roles_resolved_via_carrier_and_family")),
        "roles_still_without_concept": _int(derivation_summary.get("roles_still_without_concept")),
        "h11_links_using_direct_concept_resolution": _int(derivation_summary.get("h11_links_using_direct_concept_resolution")),
        "h11_links_using_indirect_concept_resolution": _int(derivation_summary.get("h11_links_using_indirect_concept_resolution")),
        "indirect_verified_chain_count": _int(derivation_summary.get("indirect_verified_chain_count")),
        "indirect_proxy_chain_count": _int(derivation_summary.get("indirect_proxy_chain_count")),
        "motif_transfer_chain_provenance_breakdown": chain_state_counts,
        "blocked_by_motif_provenance": blocked_by_motif_provenance,
        "blocked_by_transfer_provenance": blocked_by_transfer_provenance,
        "blocked_by_concept_validation": blocked_by_concept_validation,
        "blocked_by_missing_concept": blocked_by_missing_concept,
        "motif_transfer_chain_provenance_sample": samples,
        "motif_transfer_chain_provenance": samples,
        "motif_transfer_chain_provenance_is_sample": True,
        "motif_transfer_chain_provenance_sample_count": len(samples),
        "motif_transfer_chain_provenance_total_count": total_links,
        "motif_transfer_chain_provenance_truncated": len(samples) < total_links,
        "h11_transfer_chain_provenance_artifact": full_path.name if write_full_provenance_jsonl else None,
        "h11_transfer_by_game_pair_artifact": game_pair_path.name,
        "h11_transfer_by_context_pair_artifact": context_pair_path.name,
        "h11_context_lookup_artifact": context_lookup_path.name,
        "report_detail_mode": "sampled",
        "provenance_sample_limit": provenance_sample_limit,
        "h11_provenance_jsonl_bytes": provenance_bytes,
        "h11_game_pair_jsonl_bytes": game_pair_bytes,
        "h11_context_pair_jsonl_bytes": context_pair_bytes,
        "h11_context_lookup_jsonl_bytes": context_lookup_bytes,
        "unique_context_count": len(context_lookup),
        "unique_game_pair_count": len(game_pairs),
        "unique_context_pair_count": len(context_pairs),
        "promoted_concept_count": promoted_concept_count,
        "h11_blocked_by_no_motifs": future_option_motif_count == 0,
        "h11_blocked_by_no_promoted_concepts": promoted_concept_count == 0,
        "missing_evidence": [],
    }
    for name in (
        "events_with_owner_type_role", "role_linked_event_count", "motifs_with_role_links", "emergent_motifs_with_role_links",
        "motifs_with_family_provenance", "motifs_with_carrier_provenance", "motifs_with_role_provenance", "motifs_with_concept_provenance",
        "provenance_resolution_failures", "failure_occurrence_count", "unique_unresolved_id_count",
        "unresolved_family_to_carrier_count", "unresolved_carrier_to_role_count", "unresolved_role_to_concept_count",
        "unique_unresolved_family_to_carrier_count", "unique_unresolved_carrier_to_role_count", "unique_unresolved_role_to_concept_count",
        "motifs_seen_for_transfer", "motifs_skipped_no_role_links", "motifs_skipped_insufficient_support_or_stability",
        "motifs_with_role_links_for_transfer", "roles_seen_from_motif_links", "roles_with_transfer_attempts", "roles_with_concepts",
        "unique_roles_seen_from_motif_links", "unique_roles_with_transfer_attempts", "unique_roles_with_concepts",
        "motif_role_link_count", "motif_role_transfer_attempt_link_count", "motif_role_concept_link_count",
        "source_game_missing_before_resolution_count", "target_game_missing_before_resolution_count",
        "source_context_missing_before_resolution_count", "target_context_missing_before_resolution_count",
        "source_game_surrogate_count", "target_game_surrogate_count",
        "source_context_surrogate_count", "target_context_surrogate_count", "scope_resolution_source_counts",
    ):
        result[name] = derivation_summary.get(name)

    # Preserve legacy aliases and the existing H11 decision sequence.
    result["motifs_with_transfer_count"] = result["verified_motifs_with_transfer_count"]
    result["motifs_with_strong_transfer_count"] = result["verified_motifs_with_strong_transfer_count"]
    result["motifs_with_promoted_concept_count"] = result["verified_motifs_with_promoted_concept_count"]
    if future_option_motif_count == 0:
        result["decision"] = "INSUFFICIENT_EVIDENCE"
        result["missing_evidence"].append("H11 blocked because future-option motifs are absent.")  # type: ignore[union-attr]
    elif emergent_motifs == 0:
        result["decision"] = "INCONCLUSIVE"
    elif successful_role_transfer_count == 0 and promoted_concept_count == 0:
        result["decision"] = "INCONCLUSIVE"
    elif fully_verified_emergent == 0 and total_links > 0:
        result["decision"] = "PARTIALLY_VALID"
        result["missing_evidence"].append("Transfer/concept evidence exists, but it is not attached to emergent future-option motifs.")  # type: ignore[union-attr]
    elif total_links > 0 and _int(result["verified_motifs_with_strong_transfer_count"]) == 0:
        result["decision"] = "PARTIALLY_VALID"
    elif (
        fully_verified_emergent >= 5
        and _int(result["emergent_motifs_with_strong_transfer_count"]) >= 1
        and _int(result["emergent_motifs_with_promoted_concept_count"]) >= 1
        and (_float(result["emergent_motif_strong_transfer_success_rate"]) or 0.0) > 0.0
    ):
        result["decision"] = "VALID"
    elif successful_role_transfer_count > 0 and promoted_concept_count > 0 and total_links == 0:
        result["decision"] = "INVALID"
    else:
        result["decision"] = "PARTIALLY_VALID"
    if total_links and not _int(result["verified_future_option_transfer_count"]):
        result["decision"] = "INSUFFICIENT_EVIDENCE"
        result["missing_evidence"].append("No motif-transfer-concept chain has fully verified motif, transfer, and concept provenance.")  # type: ignore[union-attr]
    if fully_verified_emergent == 0 and total_links > 0 and "Transfer/concept evidence exists, but it is not attached to emergent future-option motifs." not in result["missing_evidence"]:  # type: ignore[operator]
        result["missing_evidence"].append("Transfer/concept evidence exists, but it is not attached to emergent future-option motifs.")  # type: ignore[union-attr]
    elif fully_verified_emergent == 0 and "No transfer/concept links are attached to emergent future-option motifs." not in result["missing_evidence"]:  # type: ignore[operator]
        result["missing_evidence"].append("No transfer/concept links are attached to emergent future-option motifs.")  # type: ignore[union-attr]
    if promoted_concept_count == 0 and "No promoted concepts available for motif-concept linkage." not in result["missing_evidence"]:  # type: ignore[operator]
        result["missing_evidence"].append("No promoted concepts available for motif-concept linkage.")  # type: ignore[union-attr]
    if total_links == 0 and _int(result.get("motifs_skipped_no_role_links")) > 0:
        result["missing_evidence"].append("No future-option transfer links were produced because motifs lack role links.")  # type: ignore[union-attr]
    if _int(result.get("roles_seen_from_motif_links")) > 0 and _int(result.get("roles_with_transfer_attempts")) == 0:
        result["missing_evidence"].append("Future-option motifs have role links, but no matching role-transfer attempts were found.")  # type: ignore[union-attr]
    if _int(result.get("roles_with_transfer_attempts")) > 0 and _int(result.get("roles_with_concepts")) == 0:
        result["missing_evidence"].append("Role-transfer evidence exists, but roles are not linked to concepts.")  # type: ignore[union-attr]
    assert _int(result["motif_transfer_chain_provenance_sample_count"]) <= provenance_sample_limit
    assert _int(result["motif_transfer_chain_provenance_total_count"]) == total_links
    assert fully_verified_emergent <= all_emergent
    result["core_metrics"] = {key: value for key, value in result.items() if key not in {"missing_evidence", "decision", "core_metrics", "motif_transfer_chain_provenance_sample", "motif_transfer_chain_provenance"}}
    _write(output_dir, result, max_main_report_bytes=max_main_report_bytes)
    return result


class _NullWriter:
    def __enter__(self) -> "_NullWriter":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        return None

    def write(self, _value: str) -> int:
        return 0


def _write(output_dir: Path, result: dict[str, object], *, max_main_report_bytes: int) -> None:
    report_path = output_dir / "h11_future_option_transfer_concepts_report.json"
    result.setdefault("h11_main_report_bytes", 0)
    while True:
        serialized = json.dumps(result, indent=2, sort_keys=True)
        byte_count = len(serialized.encode("utf-8"))
        if result["h11_main_report_bytes"] == byte_count:
            break
        result["h11_main_report_bytes"] = byte_count
    if byte_count > max_main_report_bytes:
        raise ValueError(
            f"H11 main report is {byte_count} bytes, exceeding max_h11_main_report_bytes={max_main_report_bytes}"
        )
    report_path.write_text(serialized + "\n", encoding="utf-8")
    text = (
        f"H11 decision: {result.get('decision')}\n"
        f"future-option transfer links: {result.get('future_option_transfer_link_count')}\n"
        f"all / verified motif transfers: {result.get('all_motifs_with_transfer_count')} / {result.get('verified_motifs_with_transfer_count')}\n"
        f"all / fully verified emergent chains: {result.get('all_emergent_motif_transfer_link_count')} / {result.get('fully_verified_emergent_chain_count')}\n"
        f"verified cross-game links / pairs: {result.get('verified_cross_game_link_count')} / {result.get('verified_cross_game_pair_count')}\n"
        f"provenance sample: {result.get('motif_transfer_chain_provenance_sample_count')} / {result.get('motif_transfer_chain_provenance_total_count')}\n"
    )
    (output_dir / "h11_future_option_transfer_concepts_report.txt").write_text(text, encoding="utf-8")
    (output_dir / "h11_future_option_transfer_concepts.md").write_text("```\n" + text + "```\n", encoding="utf-8")
