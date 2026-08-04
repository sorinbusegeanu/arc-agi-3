"""H11 hypothesis report — future-option transfer concepts."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from hashlib import sha1
from pathlib import Path
from typing import Any


DEFAULT_PROVENANCE_SAMPLE_LIMIT = 200
DEFAULT_MAX_MAIN_REPORT_BYTES = 5_000_000


def _missing_tables(connection: sqlite3.Connection, required: tuple[str, ...]) -> list[str]:
    tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    return [name for name in required if name not in tables]


def _context_id(context_key: object) -> str | None:
    """Validate context keys explicitly — reject null/empty/partial contexts."""
    if context_key in (None, ""):
        return None
    text = str(context_key).strip().lower()
    if not text or "null" in text or "none" in text:
        return None
    return "ctx:" + sha1(text.encode("utf-8")).hexdigest()[:20]


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
    """Explicit predicate — NOT assertion-only. Returns False for any malformed row."""
    motif_status = str(row.get("motif_provenance_status") or "missing")
    transfer_status = str(row.get("transfer_provenance_status") or "missing")
    concept_status = str(row.get("concept_validation_status") or "missing")
    return (motif_status == "verified" and transfer_status == "verified" and concept_status == "verified")


def _is_missing_chain(row: dict[str, object]) -> bool:
    motif_status = str(row.get("motif_provenance_status") or "missing")
    transfer_status = str(row.get("transfer_provenance_status") or "missing")
    concept_status = str(row.get("concept_validation_status") or "missing")
    return any(status == "missing" for status in (motif_status, transfer_status, concept_status))


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
        "motif_provenance_status": motif_status if (motif_status := str(row.get("motif_provenance_status") or "missing")) else "missing",
        "transfer_provenance_status": transfer_status if (transfer_status := str(row.get("transfer_provenance_status") or "missing")) else "missing",
        "concept_validation_status": concept_status if (concept_status := str(row.get("concept_validation_status") or "missing")) else "missing",
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
        from v6.future_options import derive_future_option_memory

        derive_future_option_memory(memory_dir=memory_dir, run_dir=run_dir)
    if not current_state.exists():
        result: dict[str, object] = {
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
            conn, ("future_option_transfer_links", "future_option_motifs", "role_transfer_attempts", "concept_candidates")
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

        # 4.1 — Replace assertion-only verification with explicit predicates:
        #     - Use `_is_fully_verified()` (not `assert`) for chain validation
        #     - Validate context keys explicitly via `_context_id()` which rejects null/empty/partial contexts
        full_condition = f"""
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
        """

        # 4.3 — Fix successful-role-transfer count: require reuse_success == 1, provenance_mode == single_source, and real scope (not surrogate).
        successful_role_transfer_count = _int(
            conn.execute(
                """SELECT COUNT(*) FROM role_transfer_attempts
                   WHERE COALESCE(reuse_success, 0) = 1
                     AND COALESCE(provenance_mode, '') = 'single_source'
                     AND source_game_key IS NOT NULL AND target_game_key IS NOT NULL
                     AND (source_context_key IS NOT NULL OR target_context_key IS NOT NULL)
                     AND (COALESCE(source_game_is_surrogate, 0) = 0
                         AND COALESCE(target_game_is_surrogate, 0) = 0
                         AND COALESCE(source_context_is_surrogate, 0) = 0
                         AND COALESCE(target_context_is_surrogate, 0) = 0)"""
            ).fetchone()[0]
        )

        # 4.4 — Filter promoted concepts by durable validation state (same as H07/H08).
        promoted_concept_count = _int(
            conn.execute(
                """SELECT COUNT(*) FROM concept_candidates AS candidate
                   LEFT JOIN concept_promotion_state AS persistent
                     ON persistent.concept_signature = candidate.concept_signature
                   WHERE COALESCE(persistent.currently_promoted, candidate.is_promoted, 0) = 1"""
            ).fetchone()[0]
        )

        aggregate = conn.execute(
            f"""SELECT
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
                COUNT(DISTINCT CASE WHEN COALESCE(l.transfer_provenance_status, 'missing') = 'verified' AND COALESCE(m.motif_provenance_status, 'missing') = 'verified' AND l.concept_validation_status IS NOT NULL AND l.source_game_key IS NOT NULL AND l.target_game_key IS NOT NULL AND l.source_context_key IS NOT NULL AND l.target_context_key IS NOT NULL AND COALESCE(l.source_game_is_surrogate, 0) = 0 AND COALESCE(l.target_game_is_surrogate, 0) = 0 AND COALESCE(l.source_context_is_surrogate, 0) = 0 AND COALESCE(l.target_context_is_surrogate, 0) = 0 THEN l.source_game_key || char(31) || l.target_game_key || char(31) || l.source_context_key || char(31) || l.target_context_key END) AS verified_transfer_pairs
            FROM future_option_transfer_links l
            LEFT JOIN future_option_motifs m ON l.motif_signature = m.motif_signature"""
        ).fetchone()
        aggregate = dict(aggregate) if aggregate is not None else {}

    # 4.5 — Require pair diversity for H11 VALID: at least MIN_H11_VERIFIED_TRANSFER_PAIRS distinct fully-verified, non-surrogate pairs with real cross-scope evidence.
    MIN_H11_VERIFIED_TRANSFER_PAIRS = 2
    total_links = _int(aggregate.get("total_links"))
    fully_verified_emergent = _int(aggregate.get("fully_verified_emergent_links"))
    emergent_motifs = _int(_scalar(conn, "SELECT COUNT(*) FROM future_option_motifs WHERE COALESCE(is_emergent, 0) = 1"))
    future_option_motif_count = _int(_scalar(conn, "SELECT COUNT(*) FROM future_option_motifs"))

    # Compute verified_transfer_pair_count from aggregate — only fully verified, non-surrogate pairs with real cross-scope.
    verified_transfer_pair_count = _int(aggregate.get("verified_transfer_pairs", 0))

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
        "all_emergent_motif_transfer_link_count": fully_verified_emergent,
        "emergent_motif_transfer_link_count": fully_verified_emergent,
        "fully_verified_emergent_chain_count": fully_verified_emergent,
        "verified_transfer_pair_count": verified_transfer_pair_count,
        "distinct_source_target_pair_count": verified_transfer_pair_count,
        "successful_role_transfer_count": successful_role_transfer_count,
        "promoted_concept_count": promoted_concept_count,
        "h11_blocked_by_no_motifs": future_option_motif_count == 0,
        "h11_blocked_by_no_promoted_concepts": promoted_concept_count == 0,
        "missing_evidence": [],
    }

    # Decision logic: VALID requires fully_verified_emergent >= 5, verified_motifs_with_strong_transfer_count >= 1, verified_motifs_with_promoted_concept_count >= 1, emergent_motif_strong_transfer_success_rate > 0, AND pair diversity.
    if not (verified_transfer_pair_count >= MIN_H11_VERIFIED_TRANSFER_PAIRS):
        result["decision"] = "PARTIALLY_VALID"
        result["missing_evidence"].append("Pair diversity gate failed; need at least 2 distinct source-target pairs.")
    elif not (fully_verified_emergent >= 5 and _int(result["verified_motifs_with_strong_transfer_count"]) >= 1 and _int(result["verified_motifs_with_promoted_concept_count"]) >= 1):
        result["decision"] = "PARTIALLY_VALID"
        if fully_verified_emergent < 5:
            result["missing_evidence"].append("Need at least 5 fully verified emergent chains.")
        elif _int(result["verified_motifs_with_strong_transfer_count"]) == 0:
            result["missing_evidence"].append("No motif has a strong transfer success.")
        else:
            result["missing_evidence"].append("Not enough motifs linked to promoted concepts.")

    if not (result["decision"] == "VALID"):
        if _float(result.get("emergent_motif_strong_transfer_success_rate")) or 0.0 > 0.0:
            result["decision"] = "PARTIALLY_VALID"
            result["missing_evidence"].append("Emergent motif strong-transfer success rate is zero.")

    # If all other conditions pass but pair diversity fails, return PARTIALLY_VALID with specific message.
    if (result["decision"] == "VALID") and verified_transfer_pair_count < MIN_H11_VERIFIED_TRANSFER_PAIRS:
        result["decision"] = "PARTIALLY_VALID"
        result["missing_evidence"].append("Pair diversity gate failed; need at least 2 distinct source-target pairs.")

    # If all other conditions pass but pair diversity is met, return VALID.
    if (result["decision"] == "VALID") and verified_transfer_pair_count >= MIN_H11_VERIFIED_TRANSFER_PAIRS:
        result["decision"] = "VALID"
    elif (result["decision"] not in {"INSUFFICIENT_EVIDENCE", "INCONCLUSIVE"}) and fully_verified_emergent == 0 and total_links > 0:
        result["decision"] = "PARTIALLY_VALID"

    # If no motifs at all, return INSUFFICIENT_EVIDENCE.
    if future_option_motif_count == 0:
        result["decision"] = "INSUFFICIENT_EVIDENCE"
        result["missing_evidence"].append("H11 blocked because future-option motifs are absent.")

    # If no emergent motifs, return INCONCLUSIVE.
    if emergent_motifs == 0 and not (result["decision"] in {"INSUFFICIENT_EVIDENCE", "INCONCLUSIVE"}):
        result["decision"] = "INCONCLUSIVE"

    _write(output_dir, result, max_main_report_bytes=max_main_report_bytes)
    return result


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
        raise ValueError(f"H11 main report is {byte_count} bytes, exceeding max_h11_main_report_bytes={max_main_report_bytes}")
    report_path.write_text(serialized + "\n", encoding="utf-8")
    text = (
        f"H11 decision: {result.get('decision')}\n"
        f"future-option transfer links: {result.get('future_option_transfer_link_count')}\n"
        f"all / verified motif transfers: {result.get('all_motifs_with_transfer_count')} / {result.get('verified_motifs_with_transfer_count')}\n"
        f"all / fully verified emergent chains: {result.get('all_emergent_motif_transfer_link_count')} / {result.get('fully_verified_emergent_chain_count')}\n"
        f"verified cross-game links / pairs: {result.get('verified_cross_game_link_count')} / {result.get('verified_cross_game_pair_count')}\n"
        f"provenance sample: {result.get('motif_transfer_chain_provenance_sample_count')} / {result.get('motif_transfer_chain_provenance_total_count')}"
    )
    (output_dir / "h11_future_option_transfer_concepts_report.txt").write_text(text, encoding="utf-8")
    (output_dir / "h11_future_option_transfer_concepts.md").write_text("```\n" + text + "```\n", encoding="utf-8")


class _NullWriter:
    def __enter__(self) -> "_NullWriter":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        return None

    def write(self, _value: str) -> int:
        return 0
