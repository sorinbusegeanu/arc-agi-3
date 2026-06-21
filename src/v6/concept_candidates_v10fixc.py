from __future__ import annotations

import gc
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from v6.concept_candidates_v10fixb import (
    VALID_CONCEPT_LABELS,
    _load_optional_json,
    annotate_projection_outcomes,
    apply_target_metrics,
    average_role_fingerprint_similarity,
    bin_name,
    build_collision_rows,
    build_concept_by_family_rows,
    build_graph_edges,
    build_label_rows,
    build_role_composition_rows,
    build_source_role_map,
    build_surface_comparison_rows,
    build_target_projection_mode_rows,
    canonical_role_fingerprint,
    concepts_are_fuzzy_compatible,
    detect_available_memory_bytes,
    format_report as format_fixb_report,
    fuzzy_group_candidates,
    generate_subcomposition_candidates,
    get_game_families,
    get_games,
    is_stable_candidate,
    is_transferable_candidate,
    load_source_manifest_family_map,
    mean_metric,
    merge_exact_candidates,
    remap_concept_ids,
    resolve_manifest_families_for_record,
    role_graph_sort_key,
    score_concept_against_target_family,
    strict_label_candidate,
)
from v6.m4_failure_diagnostics import count_by_reason, ensure_failure_buckets, merge_reason_counts
from v6.role_transfer_v09 import _write_parquet
from v6.role_transfer_v09b import FamilyContext
from v6.role_transfer_v09c import RoleTransferV09cConfig, prepare_family_context_stream


EVIDENCE_LANES = {
    "eligible_stable": 0,
    "diagnostic_fallback": 0,
    "diagnostic_mixed": 0,
    "diagnostic_unknown_manifest": 0,
    "diagnostic_unresolved_manifest": 0,
    "diagnostic_unclassified": 0,
}


@dataclass(frozen=True)
class ConceptCandidatesV10FixCConfig:
    m3_input_dir: str = "runs/v6/v08d_cd2_extended32_sourceclean"
    transfer_input_dir: str = "runs/v6/v09c_transfer_hardened_extended32"
    m2_input_dir: str = "runs/v6/v07_cd2_extended32_expanded"
    m1_input_dir: str = "runs/v6/v06_cd2_extended32"
    previous_v09b_dir: str = "runs/v6/v09b_role_transfer_refined_sourceclean_extended32"
    output_dir: str = "runs/v6/v10_m4_concepts_fixc_extended32"
    game_set_manifest: str | None = None
    game_set_name: str | None = None
    workers: int = 1
    min_games: int = 3
    min_manifest_families: int = 2
    min_role_count: int = 2
    max_role_count: int = 5
    grouping_mode: str = "fuzzy_structural"
    role_fingerprint_similarity_threshold: float = 0.75
    concept_fingerprint_similarity_threshold: float = 0.70
    streaming: bool = True
    max_workers_for_context_build: int = 1
    memory_safe: bool = True
    write_shards: bool = True
    resume_from_shards: bool = False


def classify_candidate_evidence_lane(candidate: dict) -> str:
    source = str(candidate.get("candidate_source", "")).lower()
    if source == "stable_role":
        source = "stable"
    elif source in {"fallback_neighborhood", "fallback_diagnostic_only"}:
        source = "fallback"

    manifest_status = str(candidate.get("manifest_resolution_status", "")).lower()
    manifest_family_ids = (
        candidate.get("manifest_family_ids")
        or candidate.get("source_manifest_families")
        or candidate.get("source_manifest_families_present")
        or []
    )
    if isinstance(manifest_family_ids, str):
        manifest_family_ids = [manifest_family_ids]
    has_unknown_manifest = any(str(item) == "unknown_manifest_family" for item in manifest_family_ids)

    member_sources = candidate.get("member_candidate_sources") or candidate.get("member_sources") or []
    if isinstance(member_sources, str):
        member_sources = [member_sources]
    normalized_member_sources = []
    for item in member_sources:
        value = str(item).lower()
        if value in {"fallback_neighborhood", "fallback_diagnostic_only"}:
            value = "fallback"
        elif value == "stable_role":
            value = "stable"
        normalized_member_sources.append(value)
    has_fallback_member = any(item == "fallback" for item in normalized_member_sources)

    if has_unknown_manifest:
        return "diagnostic_unknown_manifest"
    if manifest_status and manifest_status not in {"resolved", "ok"}:
        return "diagnostic_unresolved_manifest"
    if source == "fallback" or has_fallback_member:
        return "diagnostic_fallback"
    if source == "mixed":
        return "diagnostic_mixed"
    if source == "unknown_manifest":
        return "diagnostic_unknown_manifest"
    if source == "stable":
        return "eligible_stable"
    return "diagnostic_unclassified"


def is_candidate_concept_eligible(candidate: dict) -> bool:
    return classify_candidate_evidence_lane(candidate) == "eligible_stable"


def _append_concept_rejection_reason(candidate: dict, reason: str) -> None:
    reasons = candidate.get("concept_rejection_reasons")
    if reasons is None:
        existing = candidate.get("concept_rejection_reason")
        reasons = [] if not existing else ([existing] if not isinstance(existing, list) else list(existing))
    elif not isinstance(reasons, list):
        reasons = [reasons]
    if reason not in reasons:
        reasons.append(reason)
    candidate["concept_rejection_reasons"] = reasons
    candidate["concept_rejection_reason"] = reasons[0] if reasons else ""


def apply_candidate_evidence_policy(
    candidates: list[dict[str, Any]],
    *,
    stable_predicate,
    transferable_predicate,
) -> dict[str, Any]:
    lane_counts = dict(EVIDENCE_LANES)
    for candidate in candidates:
        lane = classify_candidate_evidence_lane(candidate)
        candidate["candidate_evidence_lane"] = lane
        candidate["diagnostic_only"] = lane != "eligible_stable"
        lane_counts[lane] = lane_counts.get(lane, 0) + 1
        if candidate["diagnostic_only"]:
            _append_concept_rejection_reason(candidate, "diagnostic_only_candidate_source")

    eligible_candidates = [candidate for candidate in candidates if is_candidate_concept_eligible(candidate)]
    diagnostic_only_candidates = [candidate for candidate in candidates if candidate.get("diagnostic_only")]
    stable_concepts = [candidate for candidate in eligible_candidates if stable_predicate(candidate)]
    transferable_concepts = [candidate for candidate in eligible_candidates if transferable_predicate(candidate)]
    return {
        "lane_counts": lane_counts,
        "eligible_candidates": eligible_candidates,
        "diagnostic_only_candidates": diagnostic_only_candidates,
        "stable_concepts": stable_concepts,
        "transferable_concepts": transferable_concepts,
    }


def run_concept_candidates_v10fixc(config: ConceptCandidatesV10FixCConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    original_v10fixb = _load_optional_json(Path("runs/v6/v10_m4_concepts_fixb_extended32") / "v10fixb_report.json")
    transfer_report = json.loads((Path(config.transfer_input_dir) / "v09c_report.json").read_text(encoding="utf-8"))
    transfer_rows = pd.read_parquet(Path(config.transfer_input_dir) / "v09c_hardened_assignments.parquet").to_dict(orient="records")
    transfer_by_heldout = defaultdict(list)
    for row in transfer_rows:
        transfer_by_heldout[str(row["heldout_family"])].append(row)

    source_manifest_family_map = load_source_manifest_family_map(Path(config.m3_input_dir))
    game_to_manifest_family = load_game_to_manifest_family(config)

    by_family_rows = []
    attrition_rows = []
    role_diag_rows = []
    manifest_diag_rows = []
    memory_rows = []

    context_config = RoleTransferV09cConfig(
        m2_input_dir=config.m2_input_dir,
        m1_input_dir=config.m1_input_dir,
        previous_v09b_dir=config.previous_v09b_dir,
        output_dir=config.output_dir,
        game_set_manifest=config.game_set_manifest,
        game_set_name=config.game_set_name,
        workers=config.max_workers_for_context_build,
    )
    contexts = prepare_family_context_stream(context_config) if config.streaming else prepare_family_context_stream(context_config)
    for context in contexts:
        shard_paths = family_shard_paths(shards_dir, context.heldout_family)
        if config.resume_from_shards and shard_paths["source_diag"].exists():
            by_family_rows.extend(load_records(shard_paths["summary"]))
            attrition_rows.extend(load_records(shard_paths["attrition"]))
            role_diag_rows.extend(load_records(shard_paths["source_diag"]))
            manifest_diag_rows.extend(load_records(shard_paths["manifest_diag"]))
            memory_rows.extend(load_records(shard_paths["memory_diag"]))
            continue

        family_result = evaluate_family_fixc(
            context=context,
            target_rows=transfer_by_heldout.get(context.heldout_family, []),
            config=config,
            source_manifest_family_map=source_manifest_family_map,
            game_to_manifest_family=game_to_manifest_family,
        )
        if config.write_shards:
            write_family_shards(shard_paths, family_result)
        by_family_rows.append(family_result["summary"])
        attrition_rows.append(family_result["attrition"])
        role_diag_rows.append(family_result["source_role_map_diag"])
        manifest_diag_rows.extend(family_result["manifest_diag_rows"])
        memory_rows.append(family_result["memory_diag"])
        if config.memory_safe:
            del family_result
            gc.collect()

    role_diag_rows = role_diag_rows + [aggregate_source_role_map_diagnostics(role_diag_rows)]
    memory_rows = memory_rows + [aggregate_memory_diagnostics(memory_rows)]

    raw_candidate_rows = load_shard_group(shards_dir, "raw_concept_candidates_premerge__*.parquet")
    transfer_score_rows = load_shard_group(shards_dir, "concept_transfer_scores__*.parquet")
    target_family_score_rows = load_shard_group(shards_dir, "concept_target_family_scores__*.parquet")
    membership_rows = load_shard_group(shards_dir, "concept_membership__*.parquet")
    failure_rows = load_shard_group(shards_dir, "concept_failure_cases__*.parquet")

    collision_rows = build_collision_rows(raw_candidate_rows)
    exact_merged_rows, exact_filter_diag = merge_exact_candidates(
        raw_candidate_rows,
        min_games=config.min_games,
        min_manifest_families=config.min_manifest_families,
        min_role_count=config.min_role_count,
    )
    exact_candidate_count = len(exact_merged_rows)
    fuzzy_rows, fuzzy_diag_rows, exact_to_final = fuzzy_group_candidates(
        exact_merged_rows,
        role_threshold=config.role_fingerprint_similarity_threshold,
        concept_threshold=config.concept_fingerprint_similarity_threshold,
    )
    final_concept_rows = fuzzy_rows if config.grouping_mode == "fuzzy_structural" else exact_merged_rows
    if config.grouping_mode != "fuzzy_structural":
        exact_to_final = {row["concept_id"]: row["concept_id"] for row in exact_merged_rows}

    mapped_transfer_rows = remap_concept_ids(transfer_score_rows, exact_to_final)
    mapped_target_family_rows = remap_concept_ids(target_family_score_rows, exact_to_final)
    mapped_membership_rows = remap_concept_ids(membership_rows, exact_to_final)
    mapped_failure_rows = remap_concept_ids(failure_rows, exact_to_final)

    concept_rows = apply_target_metrics(final_concept_rows, mapped_transfer_rows)
    concept_rows = annotate_projection_outcomes(concept_rows, mapped_transfer_rows)
    policy = apply_candidate_evidence_policy(
        concept_rows,
        stable_predicate=is_stable_candidate,
        transferable_predicate=is_transferable_candidate,
    )
    lane_counts = policy["lane_counts"]
    eligible_candidates = policy["eligible_candidates"]
    diagnostic_only_candidates = policy["diagnostic_only_candidates"]
    stable_concepts = policy["stable_concepts"]
    transferable_concepts = policy["transferable_concepts"]

    family_counts = Counter()
    for row in raw_candidate_rows:
        for family in row["source_manifest_families_present"]:
            family_counts[family] += 1
    collision_pass = all(not row["collision_detected"] for row in collision_rows)

    candidate_attrition_rows = build_attrition_rows_fixc(
        attrition_rows=attrition_rows,
        role_diag_rows=role_diag_rows,
        raw_candidate_rows=raw_candidate_rows,
        exact_candidate_count=exact_candidate_count,
        fuzzy_candidate_count=len(fuzzy_rows),
        stable_candidate_count=len(stable_concepts),
        transferable_candidate_count=len(transferable_concepts),
        exact_filter_diag=exact_filter_diag,
        mapped_transfer_rows=mapped_transfer_rows,
    )
    label_rows = build_label_rows(concept_rows)
    composition_rows = build_role_composition_rows(concept_rows)
    graph_edges = build_graph_edges(concept_rows)
    surface_rows = build_surface_comparison_rows(mapped_transfer_rows)
    target_projection_mode_rows = build_target_projection_mode_rows(mapped_transfer_rows)
    concept_by_family_rows = build_concept_by_family_rows(concept_rows, mapped_transfer_rows)
    failure_diagnostics = build_fixc_failure_diagnostics(
        by_family_rows=by_family_rows,
        source_role_map_diag_rows=role_diag_rows,
        attrition_rows=candidate_attrition_rows,
        raw_candidate_rows=raw_candidate_rows,
        stable_concepts=stable_concepts,
        transferable_concepts=transferable_concepts,
        mapped_transfer_rows=mapped_transfer_rows,
    )

    payload = build_report_payload_fixc(
        config=config,
        original_v10fixb=original_v10fixb,
        transfer_report=transfer_report,
        concept_rows=concept_rows,
        stable_concepts=stable_concepts,
        transferable_concepts=transferable_concepts,
        by_family_rows=by_family_rows,
        collision_rows=collision_rows,
        label_rows=label_rows,
        attrition_rows=candidate_attrition_rows,
        fuzzy_diag_rows=fuzzy_diag_rows,
        exact_candidate_count=exact_candidate_count,
        fuzzy_candidate_count=len(fuzzy_rows),
        family_counts=family_counts,
        collision_pass=collision_pass,
        source_role_map_diag_rows=role_diag_rows,
        memory_rows=memory_rows,
        failure_diagnostics=failure_diagnostics,
        lane_counts=lane_counts,
        eligible_candidate_count=len(eligible_candidates),
        diagnostic_only_candidate_count=len(diagnostic_only_candidates),
        accepted_candidate_count=len(transferable_concepts),
    )

    _write_parquet(output_dir / "raw_concept_candidates_premerge_fixc.parquet", raw_candidate_rows)
    _write_parquet(output_dir / "m4_concept_candidates_fixc.parquet", concept_rows)
    _write_parquet(output_dir / "eligible_concept_candidates.parquet", eligible_candidates)
    _write_parquet(output_dir / "diagnostic_only_candidates.parquet", diagnostic_only_candidates)
    _write_parquet(output_dir / "concept_candidates_accepted.parquet", transferable_concepts)
    _write_parquet(output_dir / "concept_membership_fixc.parquet", mapped_membership_rows)
    _write_parquet(output_dir / "concept_transfer_scores_fixc.parquet", mapped_transfer_rows)
    _write_parquet(output_dir / "concept_target_family_scores_fixc.parquet", mapped_target_family_rows)
    _write_parquet(output_dir / "concept_by_family_fixc.parquet", concept_by_family_rows)
    _write_parquet(output_dir / "concept_by_role_composition_fixc.parquet", composition_rows)
    _write_parquet(output_dir / "concept_failure_cases_fixc.parquet", mapped_failure_rows)
    _write_parquet(output_dir / "concept_graph_edges_fixc.parquet", graph_edges)
    _write_parquet(output_dir / "concept_id_collision_diagnostics.parquet", collision_rows)
    _write_parquet(output_dir / "concept_label_diagnostics.parquet", label_rows)
    _write_parquet(output_dir / "surface_baseline_comparison.parquet", surface_rows)
    _write_parquet(output_dir / "candidate_attrition_diagnostics.parquet", candidate_attrition_rows)
    _write_parquet(output_dir / "fuzzy_grouping_diagnostics.parquet", fuzzy_diag_rows)
    _write_parquet(output_dir / "target_projection_mode_comparison.parquet", target_projection_mode_rows)
    _write_parquet(output_dir / "source_role_map_diagnostics.parquet", role_diag_rows)
    _write_parquet(output_dir / "source_manifest_resolution_diagnostics.parquet", manifest_diag_rows)
    _write_parquet(output_dir / "memory_diagnostics.parquet", memory_rows)
    (output_dir / "m4_concept_candidates_fixc.json").write_text(json.dumps(concept_rows, indent=2), encoding="utf-8")
    (output_dir / "m4_failure_diagnostics.json").write_text(json.dumps({"m4_failure_diagnostics": failure_diagnostics}, indent=2), encoding="utf-8")
    (output_dir / "v10fixc_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "v10fixc_report.txt").write_text(format_report_fixc(payload), encoding="utf-8")
    return payload


def validate_completed_fixc_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    shards_dir = run_path / "shards"
    required_files = [
        run_path / "v10fixc_report.json",
        run_path / "v10fixc_report.txt",
        run_path / "concept_target_family_scores_fixc.parquet",
        run_path / "concept_transfer_scores_fixc.parquet",
        run_path / "source_role_map_diagnostics.parquet",
        run_path / "source_manifest_resolution_diagnostics.parquet",
        run_path / "memory_diagnostics.parquet",
        run_path / "candidate_attrition_diagnostics.parquet",
    ]
    missing_files = [str(path) for path in required_files if not path.exists()]
    shard_file_count = len(list(shards_dir.glob("*.parquet"))) if shards_dir.exists() else 0
    shard_outputs_exist = shard_file_count > 0

    transfer_rows = pd.read_parquet(run_path / "concept_transfer_scores_fixc.parquet") if (run_path / "concept_transfer_scores_fixc.parquet").exists() else pd.DataFrame()
    source_diag = pd.read_parquet(run_path / "source_role_map_diagnostics.parquet") if (run_path / "source_role_map_diagnostics.parquet").exists() else pd.DataFrame()
    attrition = pd.read_parquet(run_path / "candidate_attrition_diagnostics.parquet") if (run_path / "candidate_attrition_diagnostics.parquet").exists() else pd.DataFrame()
    report_json = json.loads((run_path / "v10fixc_report.json").read_text(encoding="utf-8")) if (run_path / "v10fixc_report.json").exists() else {}
    report = report_json.get("report", {})

    nested_payload_columns = [col for col in transfer_rows.columns if "target_family_rows" in col or "nested" in col]
    zero_candidate_reason_present = True
    raw_candidate_count = int(report.get("raw_candidate_count_premerge", 0) or 0)
    if raw_candidate_count == 0:
        failure_mode = str(report.get("failure_mode_if_not_established", "") or "")
        diag_modes = set(source_diag.get("failure_mode", [])) if not source_diag.empty and "failure_mode" in source_diag.columns else set()
        zero_candidate_reason_present = bool(failure_mode) and (failure_mode in diag_modes or failure_mode == "unexpected_empty_pipeline")

    checks = {
        "report_files_exist": (run_path / "v10fixc_report.json").exists() and (run_path / "v10fixc_report.txt").exists(),
        "shard_outputs_exist": shard_outputs_exist,
        "concept_target_family_scores_exists": (run_path / "concept_target_family_scores_fixc.parquet").exists(),
        "transfer_rows_no_nested_target_family_payloads": not nested_payload_columns,
        "source_role_map_diagnostics_exist": (run_path / "source_role_map_diagnostics.parquet").exists(),
        "manifest_resolution_diagnostics_exist": (run_path / "source_manifest_resolution_diagnostics.parquet").exists(),
        "memory_diagnostics_exist": (run_path / "memory_diagnostics.parquet").exists(),
        "zero_candidate_reason_present": zero_candidate_reason_present,
    }
    return {
        "run_dir": str(run_path),
        "valid": all(checks.values()) and not missing_files,
        "checks": checks,
        "missing_files": missing_files,
        "shard_file_count": shard_file_count,
        "nested_payload_columns": nested_payload_columns,
        "raw_candidate_count_premerge": raw_candidate_count,
        "failure_mode_if_not_established": report.get("failure_mode_if_not_established", ""),
        "candidate_attrition_rows": len(attrition),
    }


def evaluate_family_fixc(
    *,
    context: FamilyContext,
    target_rows: list[dict[str, Any]],
    config: ConceptCandidatesV10FixCConfig,
    source_manifest_family_map: dict[str, tuple[str, ...]],
    game_to_manifest_family: dict[str, str],
) -> dict[str, Any]:
    start_time = time.time()
    rss_before = current_rss_mb()
    source_role_map = build_source_role_map(context.source_roles)
    rss_after_context = current_rss_mb()
    raw_candidate_rows, attrition, source_diag, manifest_diag_rows = discover_source_only_candidates_fixc(
        context=context,
        source_role_map=source_role_map,
        source_manifest_family_map=source_manifest_family_map,
        game_to_manifest_family=game_to_manifest_family,
        config=config,
    )
    rss_after_candidate_generation = current_rss_mb()
    local_exact_rows, _ = merge_exact_candidates(
        raw_candidate_rows,
        min_games=1,
        min_manifest_families=1,
        min_role_count=config.min_role_count,
    )

    transfer_rows = []
    target_family_score_rows = []
    membership_rows = []
    failure_rows = []
    for concept in local_exact_rows:
        projection, detail_rows = evaluate_target_projection_by_family_fixc(concept, context, target_rows)
        transfer_rows.append({"heldout_family": context.heldout_family, **projection})
        target_family_score_rows.extend({"heldout_family": context.heldout_family, **row} for row in detail_rows)
        if projection["projection_used"]:
            for fingerprint in concept["canonical_role_fingerprint_hashes"]:
                membership_rows.append(
                    {
                        "heldout_family": context.heldout_family,
                        "concept_id": concept["concept_id"],
                        "source_fold": concept["source_fold"],
                        "heldout_source_family": concept["heldout_family"],
                        "canonical_role_fingerprint_hash": fingerprint,
                        "fold_local_role_ids": list(concept["fold_local_role_ids"]),
                    }
                )
        else:
            failure_rows.append(
                {
                    "heldout_family": context.heldout_family,
                    "concept_id": concept["concept_id"],
                    "failure_reason": projection["failure_reason"],
                }
            )
    rss_after_projection = current_rss_mb()

    summary = {
        "heldout_family": context.heldout_family,
        "source_only_concept_discovery": True,
        "target_role_overlap_used_in_main_score": False,
        "concept_candidates": len(local_exact_rows),
        "positive_concept_lift": int(any(row["target_mean_concept_lift_vs_role_raw"] > 0 and row["target_mean_concept_lift_vs_surface_raw"] > 0 for row in transfer_rows)),
        "target_mean_concept_lift_vs_role_raw": mean_metric(transfer_rows, "target_mean_concept_lift_vs_role_raw"),
        "target_mean_concept_lift_vs_m2": mean_metric(transfer_rows, "target_mean_concept_lift_vs_m2"),
        "target_mean_concept_lift_vs_surface_raw": mean_metric(transfer_rows, "target_mean_concept_lift_vs_surface_raw"),
        "target_mean_concept_lift_vs_surface_hardened": mean_metric(transfer_rows, "target_mean_concept_lift_vs_surface_hardened"),
    }
    memory_diag = {
        "heldout_family": context.heldout_family,
        "rss_mb_before": rss_before,
        "rss_mb_after_context": rss_after_context,
        "rss_mb_after_candidate_generation": rss_after_candidate_generation,
        "rss_mb_after_projection": rss_after_projection,
        "rss_mb_after_shard_write": rss_after_projection,
        "raw_candidate_count": len(raw_candidate_rows),
        "transfer_row_count": len(transfer_rows),
        "elapsed_seconds": round(time.time() - start_time, 3),
    }
    return {
        "heldout_family": context.heldout_family,
        "raw_candidate_rows": raw_candidate_rows,
        "transfer_rows": transfer_rows,
        "target_family_score_rows": target_family_score_rows,
        "membership_rows": membership_rows,
        "failure_rows": failure_rows,
        "summary": summary,
        "attrition": attrition,
        "source_role_map_diag": source_diag,
        "manifest_diag_rows": manifest_diag_rows,
        "memory_diag": memory_diag,
    }


def discover_source_only_candidates_fixc(
    *,
    context: FamilyContext,
    source_role_map: dict[str, dict[str, Any]],
    source_manifest_family_map: dict[str, tuple[str, ...]],
    game_to_manifest_family: dict[str, str],
    config: ConceptCandidatesV10FixCConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    source_ids = set(context.source_neighborhoods)
    role_map_ids = set(source_role_map)
    overlap_ids = source_ids & role_map_ids
    overlap_ratio = len(overlap_ids) / max(1, len(source_ids))
    global_fallback = len(source_role_map) == 0 or overlap_ratio < 0.25

    manifest_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    manifest_diag_rows: list[dict[str, Any]] = []
    missing_role_info = 0
    unresolved_manifest_count = 0
    stable_items_used = 0
    fallback_items_used = 0

    per_family_rows: dict[str, dict[str, Any]] = {}
    for family_id, record in sorted(context.source_neighborhoods.items()):
        role_info = source_role_map.get(family_id)
        manifest_from_config = resolve_manifest_from_games(record, game_to_manifest_family)
        manifest_from_map = list(source_manifest_family_map.get(family_id, ()))
        manifest_from_record = list(get_game_families(record))
        final_manifest_resolution = (
            manifest_from_config
            or manifest_from_map
            or manifest_from_record
            or ["unknown_manifest_family"]
        )
        if final_manifest_resolution == ["unknown_manifest_family"]:
            unresolved_manifest_count += 1
        if role_info is None:
            missing_role_info += 1
        stable_item = None
        if role_info is not None:
            stable_item = build_role_item_fixc(context.heldout_family, family_id, record, role_info, "stable_role")
            stable_item["source_manifest_families"] = list(final_manifest_resolution)
        fallback_item = build_fallback_item(context.heldout_family, family_id, record, "fallback_neighborhood")
        fallback_item["source_manifest_families"] = list(final_manifest_resolution)
        per_family_rows[family_id] = {
            "stable": stable_item,
            "fallback": fallback_item,
            "manifest": final_manifest_resolution[0],
        }
        manifest_diag_rows.append(
            {
                "heldout_family": context.heldout_family,
                "family_id": family_id,
                "role_info_exists": role_info is not None,
                "manifest_from_config": manifest_from_config,
                "manifest_from_full_data_map": manifest_from_map,
                "manifest_from_record_fallback": manifest_from_record,
                "final_manifest_resolution": list(final_manifest_resolution),
            }
        )

    for family_id, row in per_family_rows.items():
        item = row["fallback"] if global_fallback else (row["stable"] or row["fallback"])
        if item["candidate_source"] == "stable_role":
            stable_items_used += 1
        else:
            fallback_items_used += 1
        manifest_groups[row["manifest"]].append(item)

    remapped_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for manifest_family, items in sorted(manifest_groups.items()):
        stable_count = sum(1 for item in items if item["candidate_source"] == "stable_role")
        if stable_count < 2:
            items = [per_family_rows[item["family_id"]]["fallback"] for item in items]
        for item in items:
            remapped_groups[manifest_family].append(item)
    manifest_groups = remapped_groups

    raw_rows = []
    stable_role_raw_candidate_count = 0
    fallback_raw_candidate_count = 0
    mixed_candidates_generated = 0
    fallback_manifest_group_count = 0
    unknown_manifest_group_candidate_count = 0
    for manifest_family, items in sorted(manifest_groups.items()):
        ordered = sorted(items, key=role_graph_sort_key)
        candidates = generate_subcomposition_candidates(
            source_fold=context.heldout_family,
            heldout_family=context.heldout_family,
            manifest_family=manifest_family,
            items=ordered,
            max_role_count=config.max_role_count,
        )
        for candidate in candidates:
            sources = {item["candidate_source"] for item in ordered if item["canonical_role_fingerprint_hash"] in candidate["canonical_role_fingerprint_hashes"]}
            candidate["member_candidate_sources"] = sorted(sources)
            candidate["candidate_source"] = summarize_candidate_source(sources)
            candidate["manifest_family_ids"] = list(candidate.get("source_manifest_families_present", [manifest_family]))
            candidate["manifest_resolution_status"] = "resolved" if manifest_family != "unknown_manifest_family" else "unknown_manifest"
            if candidate["candidate_source"] == "stable_role":
                stable_role_raw_candidate_count += 1
            elif candidate["candidate_source"] == "fallback_neighborhood":
                fallback_raw_candidate_count += 1
            else:
                mixed_candidates_generated += 1
            raw_rows.append(candidate)
        if any(item["candidate_source"] == "fallback_neighborhood" for item in items):
            fallback_manifest_group_count += 1
        if manifest_family == "unknown_manifest_family":
            unknown_manifest_group_candidate_count += len(candidates)

    failure_mode = "none"
    if len(context.source_neighborhoods) == 0:
        failure_mode = "no_source_neighborhoods"
    elif len(context.source_neighborhoods) > 0 and len(source_role_map) == 0:
        failure_mode = "no_source_roles"
    elif len(source_role_map) > 0 and len(overlap_ids) == 0:
        failure_mode = "source_role_map_family_id_mismatch"
    elif len(overlap_ids) > 0 and len(manifest_groups) == 0:
        failure_mode = "manifest_resolution_failure"
    elif len(manifest_groups) > 0 and len(raw_rows) == 0:
        failure_mode = "subcomposition_generation_failure"

    source_diag = {
        "heldout_family": context.heldout_family,
        "source_neighborhood_count": len(context.source_neighborhoods),
        "source_role_count": len(context.source_roles),
        "source_role_member_family_count": sum(len(role["member_family_ids"]) for role in context.source_roles.values()),
        "source_role_map_size": len(source_role_map),
        "source_neighborhood_family_ids_count": len(source_ids),
        "source_role_map_family_ids_count": len(role_map_ids),
        "source_role_map_overlap_count": len(overlap_ids),
        "source_role_map_overlap_ratio": overlap_ratio,
        "families_skipped_missing_role_info": missing_role_info,
        "families_skipped_missing_manifest": 0,
        "families_used_for_manifest_groups": sum(len(items) for items in manifest_groups.values()),
        "manifest_groups_created": len(manifest_groups),
        "source_manifest_structures_total": len(manifest_groups),
        "generated_subcomposition_candidates_total": len(raw_rows),
        "failure_mode": failure_mode,
        "unresolved_manifest_count": unresolved_manifest_count,
        "stable_role_candidate_count": stable_items_used,
        "fallback_candidate_count": fallback_items_used,
        "fallback_manifest_group_count": fallback_manifest_group_count,
        "fallback_raw_candidate_count": fallback_raw_candidate_count,
        "stable_role_raw_candidate_count": stable_role_raw_candidate_count,
        "mixed_candidates_generated": mixed_candidates_generated,
        "unknown_manifest_group_candidate_count": unknown_manifest_group_candidate_count,
    }
    attrition = {
        "heldout_family": context.heldout_family,
        "source_manifest_structures_total": len(manifest_groups),
        "generated_subcomposition_candidates_total": len(raw_rows),
        "stable_role_candidate_count": stable_items_used,
        "fallback_candidate_count": fallback_items_used,
        "fallback_manifest_group_count": fallback_manifest_group_count,
        "fallback_raw_candidate_count": fallback_raw_candidate_count,
        "stable_role_raw_candidate_count": stable_role_raw_candidate_count,
        "mixed_candidates_generated": mixed_candidates_generated,
        "raw_candidate_count_premerge": len(raw_rows),
        "failure_mode": failure_mode,
    }
    return sorted(raw_rows, key=lambda row: (row["concept_id"], row["local_candidate_id"])), attrition, source_diag, manifest_diag_rows


def build_role_item_fixc(source_fold: str, family_id: str, record: Any, role_info: dict[str, Any], candidate_source: str) -> dict[str, Any]:
    fingerprint = canonical_role_fingerprint(role_info["role_label_candidate"], record)
    return {
        "source_fold": source_fold,
        "family_id": family_id,
        "record": record,
        "role_id": role_info["role_id"],
        "role_label": role_info["role_label_candidate"],
        "canonical_role_fingerprint_hash": fingerprint["canonical_role_fingerprint_hash"],
        "canonical_role_signature_json": fingerprint["canonical_role_signature_json"],
        "canonical_role_label_or_family": fingerprint["canonical_role_label_or_family"],
        "canonical_role_similarity_vector": fingerprint["canonical_role_similarity_vector"],
        "unknown_role_flag": fingerprint["unknown_role_flag"],
        "candidate_source": candidate_source,
    }


def build_fallback_item(source_fold: str, family_id: str, record: Any, candidate_source: str) -> dict[str, Any]:
    fingerprint = canonical_role_fingerprint("unknown_role_candidate", record)
    return {
        "source_fold": source_fold,
        "family_id": family_id,
        "record": record,
        "role_id": family_id,
        "role_label": "unknown_role_candidate",
        "canonical_role_fingerprint_hash": fingerprint["canonical_role_fingerprint_hash"],
        "canonical_role_signature_json": fingerprint["canonical_role_signature_json"],
        "canonical_role_label_or_family": fingerprint["canonical_role_label_or_family"],
        "canonical_role_similarity_vector": fingerprint["canonical_role_similarity_vector"],
        "unknown_role_flag": True,
        "candidate_source": candidate_source,
    }


def evaluate_target_projection_by_family_fixc(
    concept: dict[str, Any],
    context: FamilyContext,
    target_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    targets = []
    target_rows_by_family = defaultdict(list)
    for row in target_rows:
        target_rows_by_family[str(row.get("target_family_id", ""))].append(row)
    for family in sorted(context.target_families, key=lambda item: item.family_id):
        target_record = context.full_neighborhoods.get(family.family_id)
        if target_record is None:
            continue
        family_target_rows = target_rows_by_family.get(family.family_id, [])
        row = score_concept_against_target_family(concept, context, family.family_id, target_record, family_target_rows)
        row["concept_id"] = concept["concept_id"]
        targets.append(row)
    if not targets:
        return (
            {
                "concept_id": concept["concept_id"],
                "projection_used": False,
                "failure_reason": "missing_target_rows",
                "target_family_count": 0,
                "target_concept_prediction_score": 0.0,
                "target_best_match_score": 0.0,
                "target_top3_mean_score": 0.0,
                "target_full_mean_score": 0.0,
                "target_projection_coverage": 0.0,
                "target_mean_concept_lift_vs_role_raw": 0.0,
                "target_mean_concept_lift_vs_role_discounted": 0.0,
                "target_mean_concept_lift_vs_m2": 0.0,
                "target_mean_concept_lift_vs_surface_raw": 0.0,
                "target_mean_concept_lift_vs_surface_hardened": 0.0,
                "score_mode_best": 0.0,
                "score_mode_top3": 0.0,
                "score_mode_mean": 0.0,
                "role_id_overlap_diagnostic": 0.0,
                "role_sequence_similarity_diagnostic": 0.0,
            },
            [],
        )
    target_scores = sorted((row["target_family_score"] for row in targets), reverse=True)
    top3 = target_scores[:3]
    best = target_scores[0]
    mean_score = float(sum(target_scores) / len(target_scores))
    top3_mean = float(sum(top3) / len(top3))
    projection = {
        "concept_id": concept["concept_id"],
        "projection_used": True,
        "failure_reason": "",
        "target_family_count": len(targets),
        "target_concept_prediction_score": top3_mean,
        "target_best_match_score": best,
        "target_top3_mean_score": top3_mean,
        "target_full_mean_score": mean_score,
        "target_projection_coverage": len(targets) / max(1, len(context.target_families)),
        "target_mean_concept_lift_vs_role_raw": mean_metric(targets, "target_family_score") - mean_metric(targets, "best_individual_role_baseline_raw"),
        "target_mean_concept_lift_vs_role_discounted": mean_metric(targets, "target_family_score") - mean_metric(targets, "best_individual_role_baseline_discounted"),
        "target_mean_concept_lift_vs_m2": mean_metric(targets, "target_family_score") - mean_metric(targets, "best_raw_m2_baseline"),
        "target_mean_concept_lift_vs_surface_raw": mean_metric(targets, "target_family_score") - mean_metric(targets, "best_surface_raw_baseline"),
        "target_mean_concept_lift_vs_surface_hardened": mean_metric(targets, "target_family_score") - mean_metric(targets, "surface_hardened_baseline"),
        "score_mode_best": best,
        "score_mode_top3": top3_mean,
        "score_mode_mean": mean_score,
        "best_individual_role_baseline_raw": mean_metric(targets, "best_individual_role_baseline_raw"),
        "best_individual_role_baseline_discounted": mean_metric(targets, "best_individual_role_baseline_discounted"),
        "best_raw_m2_baseline": mean_metric(targets, "best_raw_m2_baseline"),
        "best_surface_raw_baseline": mean_metric(targets, "best_surface_raw_baseline"),
        "surface_hardened_baseline": mean_metric(targets, "surface_hardened_baseline"),
        "role_id_overlap_diagnostic": mean_metric(targets, "role_id_overlap_diagnostic"),
        "role_sequence_similarity_diagnostic": mean_metric(targets, "role_sequence_similarity_diagnostic"),
    }
    return projection, targets


def build_attrition_rows_fixc(
    *,
    attrition_rows: list[dict[str, Any]],
    role_diag_rows: list[dict[str, Any]],
    raw_candidate_rows: list[dict[str, Any]],
    exact_candidate_count: int,
    fuzzy_candidate_count: int,
    stable_candidate_count: int,
    transferable_candidate_count: int,
    exact_filter_diag: dict[str, int],
    mapped_transfer_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    projection_failures = sum(1 for row in mapped_transfer_rows if not row["projection_used"])
    no_positive_lift = sum(1 for row in mapped_transfer_rows if row["projection_used"] and (row["target_mean_concept_lift_vs_role_raw"] <= 0 or row["target_mean_concept_lift_vs_surface_raw"] <= 0))
    summary = {
        "heldout_family": "__all__",
        "source_manifest_structures_total": sum(row["source_manifest_structures_total"] for row in attrition_rows),
        "generated_subcomposition_candidates_total": sum(row["generated_subcomposition_candidates_total"] for row in attrition_rows),
        "raw_candidate_count_premerge": len(raw_candidate_rows),
        **exact_filter_diag,
        "rejected_due_to_projection_failure": projection_failures,
        "rejected_due_to_no_positive_lift": no_positive_lift,
        "stable_candidate_count": stable_candidate_count,
        "transferable_candidate_count": transferable_candidate_count,
        "exact_candidate_count": exact_candidate_count,
        "fuzzy_candidate_count": fuzzy_candidate_count,
        "exact_vs_fuzzy_delta": fuzzy_candidate_count - exact_candidate_count,
        "stable_role_candidate_count": sum(row.get("stable_role_candidate_count", 0) for row in attrition_rows),
        "fallback_candidate_count": sum(row.get("fallback_candidate_count", 0) for row in attrition_rows),
        "fallback_manifest_group_count": sum(row.get("fallback_manifest_group_count", 0) for row in attrition_rows),
        "fallback_raw_candidate_count": sum(row.get("fallback_raw_candidate_count", 0) for row in attrition_rows),
        "stable_role_raw_candidate_count": sum(row.get("stable_role_raw_candidate_count", 0) for row in attrition_rows),
        "mixed_candidates_generated": sum(row.get("mixed_candidates_generated", 0) for row in attrition_rows),
        "failure_mode": summarize_failure_modes(role_diag_rows),
    }
    return sorted(attrition_rows + [summary], key=lambda row: row["heldout_family"])


def build_fixc_failure_diagnostics(
    *,
    by_family_rows: list[dict[str, Any]],
    source_role_map_diag_rows: list[dict[str, Any]],
    attrition_rows: list[dict[str, Any]],
    raw_candidate_rows: list[dict[str, Any]],
    stable_concepts: list[dict[str, Any]],
    transferable_concepts: list[dict[str, Any]],
    mapped_transfer_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def _clean(counts: dict[str, int]) -> dict[str, int]:
        return {key: value for key, value in counts.items() if key not in {"", "none", "unexpected_empty_pipeline"}}

    family_ids = sorted(
        {
            str(row.get("heldout_family", ""))
            for row in by_family_rows + source_role_map_diag_rows + attrition_rows + raw_candidate_rows + stable_concepts + transferable_concepts + mapped_transfer_rows
            if row.get("heldout_family") and row.get("heldout_family") != "__all__"
        }
    )
    by_family = {str(row["heldout_family"]): row for row in by_family_rows if row.get("heldout_family") != "__all__"}
    source_diag = {str(row["heldout_family"]): row for row in source_role_map_diag_rows if row.get("heldout_family") != "__all__"}
    attrition = {str(row["heldout_family"]): row for row in attrition_rows if row.get("heldout_family") != "__all__"}
    raw_by_family = defaultdict(list)
    for row in raw_candidate_rows:
        raw_by_family[str(row.get("heldout_family", ""))].append(row)
    stable_by_family = defaultdict(list)
    for row in stable_concepts:
        stable_by_family[str(row.get("heldout_family", ""))].append(row)
    transferable_by_family = defaultdict(list)
    for row in transferable_concepts:
        transferable_by_family[str(row.get("heldout_family", ""))].append(row)
    transfer_by_family = defaultdict(list)
    for row in mapped_transfer_rows:
        transfer_by_family[str(row.get("heldout_family", ""))].append(row)

    per_family = []
    for family_id in family_ids:
        diag = source_diag.get(family_id, {})
        attr = attrition.get(family_id, {})
        summary = by_family.get(family_id, {})
        raw_rows = raw_by_family.get(family_id, [])
        stable_rows = stable_by_family.get(family_id, [])
        transferable_rows = transferable_by_family.get(family_id, [])
        transfer_rows = transfer_by_family.get(family_id, [])
        reason_counts = merge_reason_counts(
            _clean(count_by_reason([diag], "failure_mode")),
            _clean(count_by_reason([summary], "zero_candidate_reason")),
        )
        projection_failures = [row for row in transfer_rows if not row.get("projection_used")]
        if projection_failures:
            reason_counts = merge_reason_counts(reason_counts, {"no_target_projection": len(projection_failures)})
        role_failures = sum(1 for row in transfer_rows if row.get("projection_used") and float(row.get("target_mean_concept_lift_vs_role_raw", 0.0)) <= 0.0)
        surface_failures = sum(1 for row in transfer_rows if row.get("projection_used") and float(row.get("target_mean_concept_lift_vs_surface_raw", 0.0)) <= 0.0)
        if role_failures:
            reason_counts = merge_reason_counts(reason_counts, {"no_lift_vs_best_individual_role": role_failures})
        if surface_failures:
            reason_counts = merge_reason_counts(reason_counts, {"no_lift_vs_surface_effect_raw": surface_failures})
        rejected_candidate_ids = {
            str(row.get("concept_id", ""))
            for row in transfer_rows
            if (not row.get("projection_used"))
            or float(row.get("target_mean_concept_lift_vs_role_raw", 0.0)) <= 0.0
            or float(row.get("target_mean_concept_lift_vs_surface_raw", 0.0)) <= 0.0
        }
        family_row = {
            "heldout_family_id": family_id,
            "source_neighborhoods_available": bool(diag.get("source_neighborhood_count", 0)),
            "source_roles_available": bool(diag.get("source_role_count", 0)),
            "source_role_overlap_ok": bool(diag.get("source_role_map_overlap_count", 0)),
            "stable_role_items_count": int(diag.get("stable_role_candidate_count", 0)),
            "raw_candidates_count": int(attr.get("raw_candidate_count_premerge", len(raw_rows))),
            "fallback_candidate_count": sum(1 for row in raw_rows if row.get("candidate_source") in {"fallback", "fallback_neighborhood"}),
            "mixed_candidate_count": sum(1 for row in raw_rows if row.get("candidate_source") == "mixed"),
            "unknown_manifest_candidate_count": sum(
                1
                for row in raw_rows
                if "unknown_manifest_family" in row.get("source_manifest_families_present", [])
            ),
            "rejected_candidate_count": len({concept_id for concept_id in rejected_candidate_ids if concept_id}),
            "stable_candidate_count": len(stable_rows),
            "transferable_candidate_count": len(transferable_rows),
            "failure_reason_counts": ensure_failure_buckets(reason_counts),
        }
        per_family.append(family_row)

    aggregate_attrition = next((row for row in attrition_rows if row.get("heldout_family") == "__all__"), {})
    total_reason_counts = merge_reason_counts(*(row["failure_reason_counts"] for row in per_family))
    total_reason_counts = merge_reason_counts(
        total_reason_counts,
        {
            "insufficient_games": int(aggregate_attrition.get("rejected_due_to_min_games", 0)),
            "insufficient_manifest_families": int(aggregate_attrition.get("rejected_due_to_min_families", 0)),
        },
    )
    attrition_totals = {
        "families_loaded": len(per_family),
        "families_with_source_neighborhoods": sum(1 for row in per_family if row["source_neighborhoods_available"]),
        "families_with_source_roles": sum(1 for row in per_family if row["source_roles_available"]),
        "families_with_source_role_overlap": sum(1 for row in per_family if row["source_role_overlap_ok"]),
        "stable_role_items_total": sum(row["stable_role_items_count"] for row in per_family),
        "raw_candidates_total": sum(row["raw_candidates_count"] for row in per_family),
        "projected_target_families_total": sum(int(row.get("target_family_count", 0)) for row in mapped_transfer_rows if row.get("projection_used")),
        "projection_used_total": sum(1 for row in mapped_transfer_rows if row.get("projection_used")),
        "rejected_candidate_total": sum(row["rejected_candidate_count"] for row in per_family),
        "stable_candidate_total": sum(row["stable_candidate_count"] for row in per_family),
        "transferable_candidate_total": sum(row["transferable_candidate_count"] for row in per_family),
        "fallback_candidate_total": sum(row["fallback_candidate_count"] for row in per_family),
        "mixed_candidate_total": sum(row["mixed_candidate_count"] for row in per_family),
        "unknown_manifest_candidate_total": sum(row["unknown_manifest_candidate_count"] for row in per_family),
    }
    return {
        "per_family": per_family,
        "total_failure_reason_counts": ensure_failure_buckets(total_reason_counts),
        "attrition_totals": attrition_totals,
    }


def build_report_payload_fixc(
    *,
    config: ConceptCandidatesV10FixCConfig,
    original_v10fixb: dict[str, Any] | None,
    transfer_report: dict[str, Any],
    concept_rows: list[dict[str, Any]],
    stable_concepts: list[dict[str, Any]],
    transferable_concepts: list[dict[str, Any]],
    by_family_rows: list[dict[str, Any]],
    collision_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    attrition_rows: list[dict[str, Any]],
    fuzzy_diag_rows: list[dict[str, Any]],
    exact_candidate_count: int,
    fuzzy_candidate_count: int,
    family_counts: Counter,
    collision_pass: bool,
    source_role_map_diag_rows: list[dict[str, Any]],
    memory_rows: list[dict[str, Any]],
    failure_diagnostics: dict[str, Any],
    lane_counts: dict[str, int],
    eligible_candidate_count: int,
    diagnostic_only_candidate_count: int,
    accepted_candidate_count: int,
) -> dict[str, Any]:
    metric = lambda rows, key: mean_metric(rows, key)
    positive_families = sum(1 for row in by_family_rows if row["positive_concept_lift"])
    families_spanned = sorted({family for row in concept_rows for family in row["source_manifest_families_present"]})
    dominant = max(label_rows, key=lambda row: row["percent"]) if label_rows else {"concept_label_candidate": "", "percent": 0.0}
    contribution_denominator = sum(max(0, row["positive_lift_family_count"]) for row in transferable_concepts) or 1
    max_contribution_share = max((row["positive_lift_family_count"] / contribution_denominator for row in transferable_concepts), default=0.0)
    aggregate_attrition = next((row for row in attrition_rows if row["heldout_family"] == "__all__"), {})
    aggregate_source_diag = next((row for row in source_role_map_diag_rows if row["heldout_family"] == "__all__"), None)
    peak_rss_mb = max((row["rss_mb_after_shard_write"] for row in memory_rows), default=0.0)
    max_family_rss_mb = max((row["rss_mb_after_projection"] for row in memory_rows), default=0.0)

    conclusion = "m4_concepts_fixc_not_established"
    if (
        len(stable_concepts) >= 8
        and len(transferable_concepts) >= 5
        and metric(transferable_concepts, "target_mean_concept_lift_vs_role_raw") >= 0.10
        and metric(transferable_concepts, "target_mean_concept_lift_vs_surface_raw") >= 0.10
        and positive_families >= 12
        and max_contribution_share <= 0.40
        and dominant["percent"] <= 0.60
        and collision_pass
    ):
        conclusion = "m4_concepts_fixc_very_strong"
    elif (
        len(stable_concepts) >= 5
        and len(transferable_concepts) >= 3
        and metric(transferable_concepts, "target_mean_concept_lift_vs_role_raw") >= 0.05
        and metric(transferable_concepts, "target_mean_concept_lift_vs_surface_raw") >= 0.05
        and positive_families >= 8
        and len(families_spanned) >= 8
        and collision_pass
    ):
        conclusion = "m4_concepts_fixc_strong"
    elif (
        len(stable_concepts) >= 3
        and len(transferable_concepts) >= 2
        and metric(transferable_concepts, "target_mean_concept_lift_vs_role_raw") > 0
        and metric(transferable_concepts, "target_mean_concept_lift_vs_surface_raw") > 0
        and positive_families >= 6
        and collision_pass
        and aggregate_attrition.get("source_manifest_structures_total", 0) > 0
    ):
        conclusion = "m4_concepts_fixc_weak"

    report = {
        "original_v10fixb_summary": (original_v10fixb or {}).get("report", {}),
        "transfer_report_summary": transfer_report.get("report", {}),
        "source_only_concept_discovery": True,
        "target_role_id_overlap_removed_from_main_score": True,
        "target_role_overlap_diagnostic_only": True,
        "concept_id_collision_check_passed": collision_pass,
        "family_context_count": len(by_family_rows),
        "raw_candidate_count_premerge": aggregate_attrition.get("raw_candidate_count_premerge", 0),
        "source_manifest_structures_total": aggregate_attrition.get("source_manifest_structures_total", 0),
        "generated_subcomposition_candidates_total": aggregate_attrition.get("generated_subcomposition_candidates_total", 0),
        "concept_id_collision_count": sum(1 for row in collision_rows if row["collision_detected"]),
        "corrected_concept_candidate_count": len(concept_rows),
        "corrected_stable_concepts": len(stable_concepts),
        "corrected_transferable_concepts": len(transferable_concepts),
        "candidate_evidence_lanes": lane_counts,
        "eligible_candidate_count": eligible_candidate_count,
        "diagnostic_only_candidate_count": diagnostic_only_candidate_count,
        "accepted_candidate_count": accepted_candidate_count,
        "diagnostic_only_policy": {
            "fallback_candidates_promoted": False,
            "mixed_candidates_promoted": False,
            "unknown_manifest_candidates_promoted": False,
            "unresolved_manifest_candidates_promoted": False,
        },
        "target_mean_concept_lift_vs_role_raw": metric(transferable_concepts, "target_mean_concept_lift_vs_role_raw"),
        "target_mean_concept_lift_vs_m2": metric(transferable_concepts, "target_mean_concept_lift_vs_m2"),
        "target_mean_concept_lift_vs_surface_raw": metric(transferable_concepts, "target_mean_concept_lift_vs_surface_raw"),
        "target_mean_concept_lift_vs_surface_hardened": metric(transferable_concepts, "target_mean_concept_lift_vs_surface_hardened"),
        "positive_concept_lift_families": positive_families,
        "concepts_span_manifest_families": len(families_spanned),
        "concept_label_distribution": {row["concept_label_candidate"]: row["count"] for row in label_rows},
        "dominant_concept_label": dominant["concept_label_candidate"],
        "dominant_label_percent": dominant["percent"],
        "exact_candidate_count": exact_candidate_count,
        "fuzzy_candidate_count": fuzzy_candidate_count,
        "exact_vs_fuzzy_delta": fuzzy_candidate_count - exact_candidate_count,
        "target_projection_mode_best_mean": metric(transferable_concepts or concept_rows, "score_mode_best"),
        "target_projection_mode_top3_mean": metric(transferable_concepts or concept_rows, "score_mode_top3"),
        "target_projection_mode_full_mean": metric(transferable_concepts or concept_rows, "score_mode_mean"),
        "heldout_families_where_concepts_transfer": sorted([row["heldout_family"] for row in by_family_rows if row["positive_concept_lift"]]),
        "heldout_families_where_concepts_fail": sorted([row["heldout_family"] for row in by_family_rows if not row["positive_concept_lift"]]),
        "scientific_conclusion": conclusion,
        "v10a_can_proceed": conclusion != "m4_concepts_fixc_not_established",
        "failure_mode_if_not_established": summarize_failure_modes(source_role_map_diag_rows),
        "grouping_mode": config.grouping_mode,
        "family_occurrence_counts": dict(sorted(family_counts.items())),
        "attrition": aggregate_attrition,
        "source_role_map_diag_summary": aggregate_source_diag or {},
        "peak_rss_mb": peak_rss_mb,
        "max_family_rss_mb": max_family_rss_mb,
        "memory_safe_streaming_used": bool(config.streaming and config.memory_safe),
    }
    return {
        "config": {
            "output_dir": config.output_dir,
            "workers": config.workers,
            "previous_v09b_dir": config.previous_v09b_dir,
            "streaming": config.streaming,
            "memory_safe": config.memory_safe,
            "write_shards": config.write_shards,
            "resume_from_shards": config.resume_from_shards,
        },
        "report": report,
        "m4_failure_diagnostics": failure_diagnostics,
        "validation": {
            "diagnostic_success": bool(concept_rows),
            "scientific_conclusion": conclusion,
            "proceed_to_v10a": report["v10a_can_proceed"],
        },
    }


def format_report_fixc(payload: dict[str, Any]) -> str:
    report = payload["report"]
    attrition = report["attrition"]
    diag = report["source_role_map_diag_summary"]
    return "\n".join(
        [
            "ARC-AGI3 v0.10fix-c: source-role-map repair, manifest diagnostics, and memory-safe streaming execution",
            "",
            "1. v10fix-b summary",
            f"v10fixb_scientific_conclusion={report['original_v10fixb_summary'].get('scientific_conclusion', '')}",
            "",
            "2. Source-role-map diagnostics",
            f"source_neighborhood_count={diag.get('source_neighborhood_count', 0)}",
            f"source_role_map_size={diag.get('source_role_map_size', 0)}",
            f"source_role_map_overlap_count={diag.get('source_role_map_overlap_count', 0)}",
            f"source_role_map_overlap_ratio={diag.get('source_role_map_overlap_ratio', 0.0):.6f}",
            f"failure_mode={diag.get('failure_mode', 'unexpected_empty_pipeline')}",
            "",
            "3. Manifest resolution diagnostics",
            f"manifest_groups_created={diag.get('manifest_groups_created', 0)}",
            f"unresolved_manifest_count={diag.get('unresolved_manifest_count', 0)}",
            f"unknown_manifest_group_candidate_count={diag.get('unknown_manifest_group_candidate_count', 0)}",
            "",
            "4. Candidate generation funnel",
            f"source_manifest_structures_total={report['source_manifest_structures_total']}",
            f"generated_subcomposition_candidates_total={report['generated_subcomposition_candidates_total']}",
            f"raw_candidate_count_premerge={report['raw_candidate_count_premerge']}",
            "",
            "5. Stable-role vs fallback-neighborhood candidate counts",
            f"stable_role_candidate_count={attrition.get('stable_role_candidate_count', 0)}",
            f"fallback_candidate_count={attrition.get('fallback_candidate_count', 0)}",
            f"stable_role_raw_candidate_count={attrition.get('stable_role_raw_candidate_count', 0)}",
            f"fallback_raw_candidate_count={attrition.get('fallback_raw_candidate_count', 0)}",
            f"mixed_candidates_generated={attrition.get('mixed_candidates_generated', 0)}",
            "",
            "6. Candidate attrition diagnostics",
            f"candidate_groups_before_support_filter={attrition.get('candidate_groups_before_support_filter', 0)}",
            f"candidate_groups_after_support_filter={attrition.get('candidate_groups_after_support_filter', 0)}",
            f"rejected_due_to_projection_failure={attrition.get('rejected_due_to_projection_failure', 0)}",
            f"rejected_due_to_no_positive_lift={attrition.get('rejected_due_to_no_positive_lift', 0)}",
            "",
            "7. Exact vs fuzzy grouping comparison",
            f"exact_candidate_count={report['exact_candidate_count']}",
            f"fuzzy_candidate_count={report['fuzzy_candidate_count']}",
            f"exact_vs_fuzzy_delta={report['exact_vs_fuzzy_delta']}",
            "",
            "8. Concept ID collision check",
            f"concept_id_collision_check_passed={report['concept_id_collision_check_passed']}",
            f"concept_id_collision_count={report['concept_id_collision_count']}",
            "",
            "9. Target projection mode comparison",
            f"score_mode_best={report['target_projection_mode_best_mean']:.6f}",
            f"score_mode_top3={report['target_projection_mode_top3_mean']:.6f}",
            f"score_mode_mean={report['target_projection_mode_full_mean']:.6f}",
            "",
            "10. Raw role / raw M2 / raw surface baseline comparison",
            f"target_mean_concept_lift_vs_role_raw={report['target_mean_concept_lift_vs_role_raw']:.6f}",
            f"target_mean_concept_lift_vs_m2={report['target_mean_concept_lift_vs_m2']:.6f}",
            f"target_mean_concept_lift_vs_surface_raw={report['target_mean_concept_lift_vs_surface_raw']:.6f}",
            "",
            "11. Memory diagnostics",
            f"peak_rss_mb={report['peak_rss_mb']:.2f}",
            f"max_family_rss_mb={report['max_family_rss_mb']:.2f}",
            f"memory_safe_streaming_used={report['memory_safe_streaming_used']}",
            "",
            "12. Corrected stable and transferable concepts",
            f"corrected_stable_concepts={report['corrected_stable_concepts']}",
            f"corrected_transferable_concepts={report['corrected_transferable_concepts']}",
            "",
            "13. Held-out families where concepts transfer",
            ",".join(report["heldout_families_where_concepts_transfer"]) or "none",
            "",
            "14. Held-out families where concepts fail",
            ",".join(report["heldout_families_where_concepts_fail"]) or "none",
            "",
            "15. Failure mode if not established",
            f"failure_mode_if_not_established={report['failure_mode_if_not_established']}",
            "",
            "16. Corrected scientific conclusion",
            f"scientific_conclusion={report['scientific_conclusion']}",
            "",
            "17. Whether v0.10a can proceed",
            f"v10a_can_proceed={report['v10a_can_proceed']}",
        ]
    )


def family_shard_paths(shards_dir: Path, heldout_family: str) -> dict[str, Path]:
    return {
        "raw": shards_dir / f"raw_concept_candidates_premerge__{heldout_family}.parquet",
        "transfer": shards_dir / f"concept_transfer_scores__{heldout_family}.parquet",
        "target_family": shards_dir / f"concept_target_family_scores__{heldout_family}.parquet",
        "membership": shards_dir / f"concept_membership__{heldout_family}.parquet",
        "failure": shards_dir / f"concept_failure_cases__{heldout_family}.parquet",
        "source_diag": shards_dir / f"source_role_map_diagnostics__{heldout_family}.parquet",
        "manifest_diag": shards_dir / f"source_manifest_resolution_diagnostics__{heldout_family}.parquet",
        "attrition": shards_dir / f"candidate_attrition__{heldout_family}.parquet",
        "summary": shards_dir / f"family_summary__{heldout_family}.parquet",
        "memory_diag": shards_dir / f"memory_diagnostics__{heldout_family}.parquet",
    }


def write_family_shards(paths: dict[str, Path], family_result: dict[str, Any]) -> None:
    _write_parquet(paths["raw"], family_result["raw_candidate_rows"])
    _write_parquet(paths["transfer"], family_result["transfer_rows"])
    _write_parquet(paths["target_family"], family_result["target_family_score_rows"])
    _write_parquet(paths["membership"], family_result["membership_rows"])
    _write_parquet(paths["failure"], family_result["failure_rows"])
    _write_parquet(paths["source_diag"], [family_result["source_role_map_diag"]])
    _write_parquet(paths["manifest_diag"], family_result["manifest_diag_rows"])
    _write_parquet(paths["attrition"], [family_result["attrition"]])
    _write_parquet(paths["summary"], [family_result["summary"]])
    _write_parquet(paths["memory_diag"], [family_result["memory_diag"]])


def load_shard_group(shards_dir: Path, pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(shards_dir.glob(pattern)):
        rows.extend(normalize_records(pd.read_parquet(path).to_dict(orient="records")))
    return rows


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return normalize_records(pd.read_parquet(path).to_dict(orient="records"))


def summarize_failure_modes(rows: list[dict[str, Any]]) -> str:
    counts = Counter(row.get("failure_mode", "unexpected_empty_pipeline") for row in rows)
    if not counts:
        return "unexpected_empty_pipeline"
    return counts.most_common(1)[0][0]


def aggregate_source_role_map_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"heldout_family": "__all__", "failure_mode": "unexpected_empty_pipeline"}
    total_neighborhoods = sum(row.get("source_neighborhood_count", 0) for row in rows)
    total_overlap = sum(row.get("source_role_map_overlap_count", 0) for row in rows)
    return {
        "heldout_family": "__all__",
        "source_neighborhood_count": total_neighborhoods,
        "source_role_count": sum(row.get("source_role_count", 0) for row in rows),
        "source_role_member_family_count": sum(row.get("source_role_member_family_count", 0) for row in rows),
        "source_role_map_size": sum(row.get("source_role_map_size", 0) for row in rows),
        "source_neighborhood_family_ids_count": total_neighborhoods,
        "source_role_map_family_ids_count": sum(row.get("source_role_map_family_ids_count", 0) for row in rows),
        "source_role_map_overlap_count": total_overlap,
        "source_role_map_overlap_ratio": total_overlap / max(1, total_neighborhoods),
        "families_skipped_missing_role_info": sum(row.get("families_skipped_missing_role_info", 0) for row in rows),
        "families_skipped_missing_manifest": sum(row.get("families_skipped_missing_manifest", 0) for row in rows),
        "families_used_for_manifest_groups": sum(row.get("families_used_for_manifest_groups", 0) for row in rows),
        "manifest_groups_created": sum(row.get("manifest_groups_created", 0) for row in rows),
        "source_manifest_structures_total": sum(row.get("source_manifest_structures_total", 0) for row in rows),
        "generated_subcomposition_candidates_total": sum(row.get("generated_subcomposition_candidates_total", 0) for row in rows),
        "failure_mode": summarize_failure_modes(rows),
        "unresolved_manifest_count": sum(row.get("unresolved_manifest_count", 0) for row in rows),
        "stable_role_candidate_count": sum(row.get("stable_role_candidate_count", 0) for row in rows),
        "fallback_candidate_count": sum(row.get("fallback_candidate_count", 0) for row in rows),
        "fallback_manifest_group_count": sum(row.get("fallback_manifest_group_count", 0) for row in rows),
        "fallback_raw_candidate_count": sum(row.get("fallback_raw_candidate_count", 0) for row in rows),
        "stable_role_raw_candidate_count": sum(row.get("stable_role_raw_candidate_count", 0) for row in rows),
        "mixed_candidates_generated": sum(row.get("mixed_candidates_generated", 0) for row in rows),
        "unknown_manifest_group_candidate_count": sum(row.get("unknown_manifest_group_candidate_count", 0) for row in rows),
    }


def aggregate_memory_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"heldout_family": "__all__"}
    return {
        "heldout_family": "__all__",
        "rss_mb_before": max(row.get("rss_mb_before", 0.0) for row in rows),
        "rss_mb_after_context": max(row.get("rss_mb_after_context", 0.0) for row in rows),
        "rss_mb_after_candidate_generation": max(row.get("rss_mb_after_candidate_generation", 0.0) for row in rows),
        "rss_mb_after_projection": max(row.get("rss_mb_after_projection", 0.0) for row in rows),
        "rss_mb_after_shard_write": max(row.get("rss_mb_after_shard_write", 0.0) for row in rows),
        "raw_candidate_count": sum(row.get("raw_candidate_count", 0) for row in rows),
        "transfer_row_count": sum(row.get("transfer_row_count", 0) for row in rows),
        "elapsed_seconds": sum(row.get("elapsed_seconds", 0.0) for row in rows),
    }


def summarize_candidate_source(sources: set[str]) -> str:
    if sources == {"stable_role"}:
        return "stable_role"
    if sources == {"fallback_neighborhood"}:
        return "fallback_neighborhood"
    return "mixed"


def load_game_to_manifest_family(config: ConceptCandidatesV10FixCConfig) -> dict[str, str]:
    if not config.game_set_manifest and not config.game_set_name:
        return {}
    payload = json.loads(Path(config.game_set_manifest).read_text(encoding="utf-8"))
    mapping = {}
    for family, games in payload.get("families", {}).items():
        for game in games:
            mapping[str(game)] = str(family)
    return mapping


def resolve_manifest_from_games(record: Any, mapping: dict[str, str]) -> list[str]:
    families = sorted({mapping[game] for game in get_games(record) if game in mapping})
    return families


def current_rss_mb() -> float:
    try:
        import psutil  # type: ignore

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024 * 1024)
    except Exception:
        status = Path("/proc/self/status")
        if status.exists():
            for line in status.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    return 0.0


def normalize_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        normalized.append({key: decode_jsonish(key, value) for key, value in row.items()})
    return normalized


def decode_jsonish(key: str, value: Any) -> Any:
    if key.endswith("_json") or key.endswith("_jsons"):
        return value
    if isinstance(value, str) and value and value[0] in "[{":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value
