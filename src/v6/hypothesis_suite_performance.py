from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


LARGE_H11_LINK_THRESHOLD = 100_000
_PATCHED = False


@dataclass
class SuiteEvidenceCache:
    memory_dir: Path | None
    tables: frozenset[str] = frozenset()
    table_counts: dict[str, int] = field(default_factory=dict)
    summary_values: dict[str, Any] = field(default_factory=dict)
    rows_indexed: int = 0
    build_seconds: float = 0.0
    _query_cache: dict[tuple[str, tuple[Any, ...]], tuple[dict[str, Any], ...]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def current_state(self) -> Path | None:
        if self.memory_dir is None:
            return None
        return Path(self.memory_dir) / "current_state.sqlite"

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> tuple[dict[str, Any], ...]:
        key = (str(sql), tuple(params))
        with self._lock:
            cached = self._query_cache.get(key)
        if cached is not None:
            return cached
        current_state = self.current_state
        if current_state is None or not current_state.exists():
            return ()
        with sqlite3.connect(current_state) as connection:
            connection.row_factory = sqlite3.Row
            rows = tuple(dict(row) for row in connection.execute(sql, params).fetchall())
        with self._lock:
            self._query_cache.setdefault(key, rows)
            return self._query_cache[key]


def build_suite_evidence_cache(memory_dir: Path | None) -> SuiteEvidenceCache:
    started = time.perf_counter()
    cache = SuiteEvidenceCache(memory_dir=None if memory_dir is None else Path(memory_dir))
    current_state = cache.current_state
    if current_state is None or not current_state.exists():
        cache.build_seconds = time.perf_counter() - started
        return cache

    with sqlite3.connect(current_state) as connection:
        connection.row_factory = sqlite3.Row
        tables = frozenset(
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        )
        cache.tables = tables
        common_tables = (
            "stable_contingencies",
            "transformation_families",
            "role_candidates",
            "role_links",
            "role_transfer_attempts",
            "concept_candidates",
            "concept_links",
            "world_model_components",
            "future_option_events",
            "future_option_motifs",
            "future_option_transfer_links",
        )
        for table in common_tables:
            if table not in tables:
                continue
            count = int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] or 0)
            cache.table_counts[table] = count
            cache.rows_indexed += count
        if "memory_summary" in tables:
            for row in connection.execute("SELECT key, value_json FROM memory_summary").fetchall():
                try:
                    cache.summary_values[str(row["key"])] = json.loads(str(row["value_json"]))
                except (TypeError, ValueError, json.JSONDecodeError):
                    cache.summary_values[str(row["key"])] = row["value_json"]
    cache.build_seconds = time.perf_counter() - started
    return cache


def _directory_size(path: Path) -> int:
    started = time.perf_counter()
    total = 0
    if path.exists():
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except OSError:
                    pass
    return total, time.perf_counter() - started


def _profiled_evaluate(
    suite: Any,
    hypothesis_id: str,
    evaluator: Callable[..., dict[str, Any]],
    kwargs: dict[str, Any],
    cache: SuiteEvidenceCache,
) -> dict[str, Any]:
    output_dir = Path(kwargs["output_dir"])
    before_bytes, pre_scan = _directory_size(output_dir)
    started = time.perf_counter()
    result = suite._evaluate_one(hypothesis_id, evaluator, kwargs=kwargs)
    evaluation_seconds = time.perf_counter() - started
    after_bytes, post_scan = _directory_size(output_dir)
    profile = {
        "evaluation_seconds": evaluation_seconds,
        "output_scan_seconds": pre_scan + post_scan,
        "output_bytes": max(0, after_bytes - before_bytes),
        "shared_cache_build_seconds": cache.build_seconds,
        "shared_cache_rows_indexed": cache.rows_indexed,
        "shared_cache_table_count": len(cache.tables),
        "rows_available": sum(cache.table_counts.values()),
    }
    result["evaluator_seconds"] = evaluation_seconds + pre_scan + post_scan
    result["performance_profile"] = profile
    return result


def _evaluate_hypotheses_threaded(
    *,
    run_dir: Path,
    evidence_memory_dir: Path | None,
    output_dir: Path,
    suite_mode: str,
    max_db_files: int,
    max_rows: int,
    scan_all_dbs: bool,
    incremental_promotion_validation: bool,
    promotion_min_incremental_coverage: float,
    promotion_min_cross_context_or_game_evidence: int,
    promotion_min_behavioral_or_predictive_lift: float,
    promotion_min_relevant_heldout_event_count: int,
    promotion_population_comparability_threshold: float,
    promotion_demotion_failure_limit: int,
    h11_provenance_sample_limit: int,
    h11_write_full_provenance_jsonl: bool,
    max_h11_main_report_bytes: int,
) -> dict[str, dict[str, Any]]:
    from v6 import hypothesis_suite_report as suite

    dirs = {f"H{index:02d}": Path(output_dir) / f"h{index:02d}" for index in range(1, 13)}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    validation_config = suite.IncrementalPromotionValidationConfig(
        enabled=bool(incremental_promotion_validation),
        min_incremental_coverage=float(promotion_min_incremental_coverage),
        min_cross_context_or_game_evidence=int(promotion_min_cross_context_or_game_evidence),
        min_behavioral_or_predictive_lift=float(promotion_min_behavioral_or_predictive_lift),
        min_relevant_heldout_event_count=int(promotion_min_relevant_heldout_event_count),
        promotion_population_comparability_threshold=float(promotion_population_comparability_threshold),
        demotion_failure_limit=int(promotion_demotion_failure_limit),
    )
    cache = build_suite_evidence_cache(evidence_memory_dir)
    common = {
        "run_dir": Path(run_dir),
        "memory_dir": evidence_memory_dir,
        "suite_evidence_cache": cache,
    }
    tasks: list[tuple[str, Callable[..., dict[str, Any]], dict[str, Any]]] = [
        ("H01", suite.evaluate_h01_contingency_emergence, {**common, "output_dir": dirs["H01"]}),
        ("H02", suite.evaluate_h02_prediction_violation_attention, {**common, "output_dir": dirs["H02"], "max_rows": int(max_rows), "max_db_files": int(max_db_files), "scan_all_dbs": bool(scan_all_dbs)}),
        ("H03", suite.evaluate_h03_transformation_family_formation, {**common, "output_dir": dirs["H03"], "max_rows": int(max_rows), "max_db_files": int(max_db_files), "scan_all_dbs": bool(scan_all_dbs)}),
        ("H04", suite.evaluate_h04_carrier_emergence, {**common, "output_dir": dirs["H04"]}),
        ("H05", suite.evaluate_h05_role_emergence, {**common, "output_dir": dirs["H05"], "already_derived": True}),
        ("H12", suite.evaluate_h12_efficiency_emergence, {**common, "output_dir": dirs["H12"]}),
    ]
    results: dict[str, dict[str, Any]] = {}
    if str(suite_mode or "fast").lower() != "full":
        for hypothesis_id in ("H06", "H07", "H08", "H09", "H10", "H11"):
            results[hypothesis_id] = suite._skipped_result(hypothesis_id)
    else:
        tasks.extend(
            [
                ("H06", suite.evaluate_h06_role_transfer, {**common, "output_dir": dirs["H06"], "already_derived": True}),
                ("H07", suite.evaluate_h07_concept_emergence, {**common, "output_dir": dirs["H07"], "already_derived": True, "incremental_promotion_validation": validation_config}),
                ("H08", suite.evaluate_h08_world_model_coherence, {**common, "output_dir": dirs["H08"], "already_derived": True}),
                ("H09", suite.evaluate_h09_future_option_motifs, {**common, "output_dir": dirs["H09"], "already_derived": True}),
                ("H10", suite.evaluate_h10_future_option_attention, {**common, "output_dir": dirs["H10"], "already_derived": True}),
                ("H11", suite.evaluate_h11_future_option_transfer_concepts, {**common, "output_dir": dirs["H11"], "already_derived": True, "provenance_sample_limit": int(h11_provenance_sample_limit), "write_full_provenance_jsonl": bool(h11_write_full_provenance_jsonl), "max_main_report_bytes": int(max_h11_main_report_bytes)}),
            ]
        )

    configured = int(getattr(suite, "_V64_EVALUATOR_WORKERS", 4) or 4)
    configured = int(os.getenv("ARC_HYPOTHESIS_EVALUATOR_WORKERS", str(configured)))
    worker_count = max(1, min(configured, len(tasks)))
    if worker_count == 1:
        for hypothesis_id, evaluator, kwargs in tasks:
            results[hypothesis_id] = _profiled_evaluate(suite, hypothesis_id, evaluator, kwargs, cache)
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="hypothesis") as executor:
            future_map = {
                executor.submit(_profiled_evaluate, suite, hypothesis_id, evaluator, kwargs, cache): hypothesis_id
                for hypothesis_id, evaluator, kwargs in tasks
            }
            for future in as_completed(future_map):
                hypothesis_id = future_map[future]
                try:
                    results[hypothesis_id] = future.result()
                except Exception as exc:
                    results[hypothesis_id] = suite._failed_evaluator_result(hypothesis_id, exc)

    profile = {
        hypothesis_id: result.get("performance_profile", {})
        for hypothesis_id, result in sorted(results.items())
        if result.get("performance_profile")
    }
    (Path(output_dir) / "hypothesis_evaluator_profile.json").write_text(
        json.dumps(
            {
                "evaluator_workers": worker_count,
                "shared_cache_build_seconds": cache.build_seconds,
                "shared_cache_rows_indexed": cache.rows_indexed,
                "shared_cache_table_counts": cache.table_counts,
                "evaluators": profile,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return results


def _h11_streaming_large(
    *,
    memory_dir: Path,
    run_dir: Path | None,
    output_dir: Path,
    already_derived: bool = False,
    provenance_sample_limit: int = 200,
    write_full_provenance_jsonl: bool = False,
    max_main_report_bytes: int = 5_000_000,
    suite_evidence_cache: SuiteEvidenceCache | None = None,
) -> dict[str, object]:
    from v6 import hypothesis_h11_report as h11

    cache = suite_evidence_cache or build_suite_evidence_cache(memory_dir)
    current_state = Path(memory_dir) / "current_state.sqlite"
    if not current_state.exists():
        return _ORIGINAL_H11(
            memory_dir=memory_dir,
            run_dir=run_dir,
            output_dir=output_dir,
            already_derived=already_derived,
            provenance_sample_limit=provenance_sample_limit,
            write_full_provenance_jsonl=write_full_provenance_jsonl,
            max_main_report_bytes=max_main_report_bytes,
        )
    link_count = int(cache.table_counts.get("future_option_transfer_links", -1))
    if link_count < 0:
        with sqlite3.connect(current_state) as connection:
            try:
                link_count = int(connection.execute("SELECT COUNT(*) FROM future_option_transfer_links").fetchone()[0] or 0)
            except sqlite3.Error:
                link_count = 0
    if write_full_provenance_jsonl or link_count <= LARGE_H11_LINK_THRESHOLD:
        return _ORIGINAL_H11(
            memory_dir=memory_dir,
            run_dir=run_dir,
            output_dir=output_dir,
            already_derived=already_derived,
            provenance_sample_limit=provenance_sample_limit,
            write_full_provenance_jsonl=write_full_provenance_jsonl,
            max_main_report_bytes=max_main_report_bytes,
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "h11_transfer_chain_provenance.jsonl"
    game_pair_path = output_dir / "h11_transfer_by_game_pair.jsonl"
    context_pair_path = output_dir / "h11_transfer_by_context_pair.jsonl"
    context_lookup_path = output_dir / "h11_context_lookup.jsonl"

    link_total = 0
    fully_verified_count = 0
    emergent_link_count = 0
    fully_verified_emergent_count = 0
    verified_status_but_invalid_scope_count = 0
    verified_status_but_missing_identity_count = 0
    verified_status_but_surrogate_scope_count = 0
    missing_chain_count = 0
    verified_attempts = verified_successes = verified_strong = 0
    emergent_attempts = emergent_successes = emergent_strong = 0
    motifs: set[str] = set()
    emergent_motifs: set[str] = set()
    verified_motifs_with_transfer: set[str] = set()
    verified_motifs_with_strong: set[str] = set()
    verified_motifs_with_promoted: set[str] = set()
    emergent_motifs_with_transfer: set[str] = set()
    emergent_motifs_with_strong: set[str] = set()
    emergent_motifs_with_promoted: set[str] = set()
    verified_pair_ids: set[str] = set()
    verified_cross_game_pair_ids: set[str] = set()
    provenance_sample: list[dict[str, object]] = []
    game_pairs: dict[Any, dict[str, object]] = defaultdict(h11._pair_accumulator)
    context_pairs: dict[Any, dict[str, object]] = defaultdict(h11._pair_accumulator)
    context_lookup: dict[str, str] = {}
    motif_fallback: dict[str, dict[str, bool]] = defaultdict(lambda: {"role": False, "transfer": False, "concept": False})

    with sqlite3.connect(current_state) as connection:
        connection.row_factory = sqlite3.Row
        cursor = connection.execute(
            """
            SELECT l.*, m.is_emergent
            FROM future_option_transfer_links AS l
            LEFT JOIN future_option_motifs AS m ON m.motif_signature = l.motif_signature
            ORDER BY l.motif_signature ASC, l.role_signature ASC, l.concept_signature ASC
            """
        )
        for sqlite_row in cursor:
            row = dict(sqlite_row)
            link_total += 1
            motif = str(row.get("motif_signature") or "")
            if motif:
                motifs.add(motif)
                state = motif_fallback[motif]
                state["role"] = state["role"] or row.get("role_signature") not in (None, "") or row.get("source_role_signature") not in (None, "")
                state["transfer"] = state["transfer"] or h11._int(row.get("transfer_attempt_count")) > 0
                state["concept"] = state["concept"] or row.get("concept_signature") not in (None, "", "__none__")
            is_emergent = int(row.get("is_emergent") or 0) == 1
            if is_emergent:
                emergent_link_count += 1
                if motif:
                    emergent_motifs.add(motif)
            if h11._is_missing_chain(row):
                missing_chain_count += 1
            status_verified = h11._status_chain_verified(row)
            if status_verified and h11._has_complete_scope(row) and not h11._has_real_scope_difference(row):
                verified_status_but_invalid_scope_count += 1
            if status_verified and not h11._has_required_identity(row):
                verified_status_but_missing_identity_count += 1
            if status_verified and h11._has_surrogate_scope(row):
                verified_status_but_surrogate_scope_count += 1
            fully_verified = h11._is_fully_verified(row)
            if len(provenance_sample) < provenance_sample_limit:
                provenance_sample.append(h11._chain_output_row(row)[0])
            if not fully_verified:
                continue
            fully_verified_count += 1
            output, game_pair, context_pair = h11._chain_output_row(row)
            h11._add_pair_row(game_pairs[game_pair], output, True)
            h11._add_pair_row(context_pairs[context_pair], output, True)
            source_context_id = output.get("source_context_id")
            target_context_id = output.get("target_context_id")
            if source_context_id:
                context_lookup[str(source_context_id)] = str(row.get("source_context_key") or "")
            if target_context_id:
                context_lookup[str(target_context_id)] = str(row.get("target_context_key") or "")
            pair_id = str(output["transfer_pair_id"])
            verified_pair_ids.add(pair_id)
            if str(row.get("source_game_key")) != str(row.get("target_game_key")):
                verified_cross_game_pair_ids.add(pair_id)
            attempts = h11._int(row.get("transfer_attempt_count"))
            successes = h11._int(row.get("successful_transfer_count"))
            strong = h11._int(row.get("strong_transfer_success_count"))
            verified_attempts += attempts
            verified_successes += successes
            verified_strong += strong
            if motif and attempts > 0:
                verified_motifs_with_transfer.add(motif)
            if motif and strong > 0:
                verified_motifs_with_strong.add(motif)
            if motif and h11._int(row.get("promoted_concept_count")) > 0:
                verified_motifs_with_promoted.add(motif)
            if is_emergent:
                fully_verified_emergent_count += 1
                emergent_attempts += attempts
                emergent_successes += successes
                emergent_strong += strong
                if motif and attempts > 0:
                    emergent_motifs_with_transfer.add(motif)
                if motif and strong > 0:
                    emergent_motifs_with_strong.add(motif)
                if motif and h11._int(row.get("promoted_concept_count")) > 0:
                    emergent_motifs_with_promoted.add(motif)

        tables = cache.tables
        derivation_summary = cache.summary_values.get("future_option_derivation_summary")
        if not isinstance(derivation_summary, dict):
            derivation_summary = {}

        transfer_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(role_transfer_attempts)").fetchall()
        }
        provenance_column = next((candidate for candidate in ("provenance_status", "transfer_provenance_status") if candidate in transfer_columns), None)
        successful_role_transfer_count = 0
        unverified_successful_role_transfer_count = 0
        roles_with_transfer_attempts: set[str] = set()
        roles_with_successful_transfers: set[str] = set()
        for sqlite_row in connection.execute("SELECT * FROM role_transfer_attempts"):
            row = dict(sqlite_row)
            source_role = row.get("source_role_signature") or row.get("role_signature")
            if source_role not in (None, "") and h11._int(row.get("reuse_success")) in {0, 1}:
                roles_with_transfer_attempts.add(str(source_role))
            if h11._int(row.get("reuse_success")) == 1 and source_role not in (None, ""):
                roles_with_successful_transfers.add(str(source_role))
            if h11._int(row.get("reuse_success")) != 1 or str(row.get("provenance_mode") or "") != "single_source":
                continue
            if source_role in (None, "") or h11._has_surrogate_scope(row) or not h11._has_real_scope_difference(row):
                continue
            if provenance_column is None or str(row.get(provenance_column) or "missing") != "verified":
                unverified_successful_role_transfer_count += 1
            else:
                successful_role_transfer_count += 1

        if "concept_promotion_state" in tables:
            concept_query = """
                SELECT candidate.concept_signature, candidate.is_promoted AS candidate_is_promoted,
                       persistent.currently_promoted AS persistent_currently_promoted,
                       persistent.promotion_status AS persistent_promotion_status,
                       persistent.validation_status AS persistent_validation_status
                FROM concept_candidates AS candidate
                LEFT JOIN concept_promotion_state AS persistent
                  ON persistent.concept_signature = candidate.concept_signature
            """
        else:
            concept_query = """
                SELECT concept_signature, is_promoted AS candidate_is_promoted,
                       NULL AS persistent_currently_promoted,
                       NULL AS persistent_promotion_status,
                       NULL AS persistent_validation_status
                FROM concept_candidates
            """
        promoted_concept_count = sum(
            1 for row in connection.execute(concept_query)
            if h11._effective_concept_is_promoted(dict(row))
        )

    game_pair_bytes = h11._write_pair_artifact(game_pair_path, game_pairs, context_pairs=False)
    context_pair_bytes = h11._write_pair_artifact(context_pair_path, context_pairs, context_pairs=True)
    with context_lookup_path.open("w", encoding="utf-8") as handle:
        for context_id in sorted(context_lookup):
            handle.write(json.dumps({"context_id": context_id, "context_key": context_lookup[context_id]}, sort_keys=True) + "\n")

    verified_transfer_pair_count = len(verified_pair_ids)
    verified_cross_game_pair_count = len(verified_cross_game_pair_ids)
    pair_diversity_passed = verified_transfer_pair_count >= h11.MIN_H11_VERIFIED_TRANSFER_PAIRS
    emergent_success_rate = emergent_successes / emergent_attempts if emergent_attempts else None
    emergent_strong_success_rate = emergent_strong / emergent_attempts if emergent_attempts else None
    base_valid_conditions = (
        fully_verified_emergent_count >= 5
        and bool(emergent_motifs_with_strong)
        and bool(emergent_motifs_with_promoted)
        and emergent_strong_success_rate is not None
        and emergent_strong_success_rate > 0.0
    )
    missing_evidence: list[str] = []
    if link_total == 0:
        decision = "INSUFFICIENT_EVIDENCE"
        missing_evidence = ["No future-option transfer links available."]
    elif base_valid_conditions and pair_diversity_passed:
        decision = "VALID"
    elif base_valid_conditions:
        decision = "PARTIALLY_VALID"
        missing_evidence = ["Insufficient diversity of fully verified transfer pairs."]
    elif fully_verified_count:
        decision = "PARTIALLY_VALID"
        if fully_verified_emergent_count < 5:
            missing_evidence.append("Fewer than five fully verified emergent transfer chains.")
        if not emergent_motifs_with_strong:
            missing_evidence.append("No fully verified emergent motif with strong transfer success.")
        if not emergent_motifs_with_promoted:
            missing_evidence.append("No fully verified emergent motif with a promoted concept.")
    else:
        decision = "INSUFFICIENT_EVIDENCE"
        missing_evidence = ["No fully verified transfer chains available."]
    if emergent_link_count == 0:
        missing_evidence.append("No emergent future-option motifs with transfer evidence.")

    motifs_skipped_no_role_links = int(derivation_summary.get("motifs_skipped_no_role_links") or 0)
    motifs_skipped_no_transfer_attempts = int(derivation_summary.get("motifs_skipped_no_transfer_attempts") or 0)
    motifs_skipped_no_concepts = int(derivation_summary.get("motifs_skipped_no_concepts") or 0)
    if "motifs_skipped_no_role_links" not in derivation_summary:
        motifs_skipped_no_role_links = sum(1 for state in motif_fallback.values() if not state["role"])
    if "motifs_skipped_no_transfer_attempts" not in derivation_summary:
        motifs_skipped_no_transfer_attempts = sum(1 for state in motif_fallback.values() if not state["transfer"])
    if "motifs_skipped_no_concepts" not in derivation_summary:
        motifs_skipped_no_concepts = sum(1 for state in motif_fallback.values() if not state["concept"])

    provenance_report_sample = [
        {
            "motif_signature": row.get("motif_signature"),
            "role_signature": row.get("role_signature"),
            "concept_signature": row.get("concept_signature"),
            "transfer_pair_id": row.get("transfer_pair_id"),
            "fully_verified": row.get("fully_verified"),
        }
        for row in provenance_sample
    ]
    metrics: dict[str, object] = {
        "future_option_transfer_link_count": link_total,
        "fully_verified_link_count": fully_verified_count,
        "verified_future_option_transfer_count": fully_verified_count,
        "motif_transfer_chain_provenance_total_count": link_total,
        "emergent_transfer_link_count": emergent_link_count,
        "emergent_motif_transfer_link_count": emergent_link_count,
        "non_emergent_motif_transfer_link_count": link_total - emergent_link_count,
        "fully_verified_emergent_chain_count": fully_verified_emergent_count,
        "verified_motifs_with_transfer_count": len(verified_motifs_with_transfer),
        "verified_motifs_with_strong_transfer_count": len(verified_motifs_with_strong),
        "motifs_with_strong_transfer_count": len(verified_motifs_with_strong),
        "verified_motifs_with_promoted_concept_count": len(verified_motifs_with_promoted),
        "motifs_with_promoted_concept_count": len(verified_motifs_with_promoted),
        "verified_emergent_motifs_with_transfer_count": len(emergent_motifs_with_transfer),
        "verified_emergent_motifs_with_strong_transfer_count": len(emergent_motifs_with_strong),
        "emergent_motifs_with_strong_transfer_count": len(emergent_motifs_with_strong),
        "verified_emergent_motifs_with_promoted_concept_count": len(emergent_motifs_with_promoted),
        "emergent_motifs_with_promoted_concept_count": len(emergent_motifs_with_promoted),
        "verified_transfer_attempt_count": verified_attempts,
        "verified_successful_transfer_count": verified_successes,
        "verified_strong_transfer_success_count": verified_strong,
        "emergent_motif_transfer_attempt_count": emergent_attempts,
        "emergent_motif_successful_transfer_count": emergent_successes,
        "emergent_motif_strong_transfer_success_count": emergent_strong,
        "emergent_motif_transfer_success_rate": emergent_success_rate,
        "emergent_motif_strong_transfer_success_rate": emergent_strong_success_rate,
        "successful_role_transfer_count": successful_role_transfer_count,
        "unverified_successful_role_transfer_count": unverified_successful_role_transfer_count,
        "promoted_concept_count": promoted_concept_count,
        "h11_blocked_by_no_motifs": len(motifs) == 0,
        "h11_blocked_by_no_promoted_concepts": promoted_concept_count == 0,
        "motifs_skipped_no_role_links": motifs_skipped_no_role_links,
        "motifs_skipped_no_transfer_attempts": motifs_skipped_no_transfer_attempts,
        "motifs_skipped_no_concepts": motifs_skipped_no_concepts,
        "roles_with_transfer_attempts": int(derivation_summary.get("roles_with_transfer_attempts") or derivation_summary.get("unique_roles_with_transfer_attempts") or len(roles_with_transfer_attempts)),
        "roles_with_successful_transfers": len(roles_with_successful_transfers),
        "motif_count": len(motifs),
        "emergent_motif_count": len(emergent_motifs),
        "verified_transfer_pair_count": verified_transfer_pair_count,
        "verified_cross_game_pair_count": verified_cross_game_pair_count,
        "motif_transfer_chain_provenance": provenance_report_sample,
        "motif_transfer_chain_provenance_sample": provenance_report_sample,
        "motif_transfer_chain_provenance_sample_count": len(provenance_sample),
        "motif_transfer_chain_provenance_truncated": len(provenance_sample) < link_total,
        "motif_transfer_chain_provenance_is_sample": len(provenance_sample) < link_total,
        "minimum_verified_transfer_pairs_required": h11.MIN_H11_VERIFIED_TRANSFER_PAIRS,
        "verified_transfer_pair_diversity_gate_passed": pair_diversity_passed,
        "verified_status_but_invalid_scope_count": verified_status_but_invalid_scope_count,
        "verified_status_but_missing_identity_count": verified_status_but_missing_identity_count,
        "verified_status_but_surrogate_scope_count": verified_status_but_surrogate_scope_count,
        "missing_provenance_chain_count": missing_chain_count,
        "game_pair_count": len(game_pairs),
        "context_pair_count": len(context_pairs),
        "provenance_sample": [],
        "full_provenance_jsonl_written": False,
        "full_provenance_jsonl_path": str(full_path),
        "game_pair_artifact_path": str(game_pair_path),
        "context_pair_artifact_path": str(context_pair_path),
        "context_lookup_artifact_path": str(context_lookup_path),
        "game_pair_artifact_bytes": game_pair_bytes,
        "context_pair_artifact_bytes": context_pair_bytes,
        "streaming_evaluation": True,
    }
    result = {
        "hypothesis_id": "H11",
        "evidence_source": "compact_memory",
        "decision": decision,
        "missing_evidence": missing_evidence,
        **metrics,
        "core_metrics": dict(metrics),
    }
    h11._write(output_dir, result, max_main_report_bytes=max_main_report_bytes)
    return result


def _run_suite_with_worker_binding(original: Callable[..., Any], suite: Any, *args: Any, **kwargs: Any) -> Any:
    previous = getattr(suite, "_V64_EVALUATOR_WORKERS", 4)
    suite._V64_EVALUATOR_WORKERS = max(1, int(kwargs.get("higher_order_workers", previous) or previous))
    try:
        return original(*args, **kwargs)
    finally:
        suite._V64_EVALUATOR_WORKERS = previous


_ORIGINAL_H11: Callable[..., dict[str, object]]


def install_hypothesis_suite_performance_policy() -> None:
    global _PATCHED, _ORIGINAL_H11
    if _PATCHED:
        return
    from v6 import hypothesis_h11_report as h11
    from v6 import hypothesis_suite_report as suite

    _ORIGINAL_H11 = h11.evaluate_h11_future_option_transfer_concepts
    h11.evaluate_h11_future_option_transfer_concepts = _h11_streaming_large
    suite.evaluate_h11_future_option_transfer_concepts = _h11_streaming_large
    suite.evaluate_hypotheses_read_only = _evaluate_hypotheses_threaded
    original_run = suite.run_hypothesis_suite_report

    def run_hypothesis_suite_report(*args: Any, **kwargs: Any) -> Any:
        return _run_suite_with_worker_binding(original_run, suite, *args, **kwargs)

    suite.run_hypothesis_suite_report = run_hypothesis_suite_report
    _PATCHED = True
