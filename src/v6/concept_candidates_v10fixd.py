from __future__ import annotations

import gc
import json
import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from v6.concept_candidates_v10fixb import (
    _load_optional_json,
    annotate_projection_outcomes,
    apply_target_metrics,
    build_collision_rows,
    build_concept_by_family_rows,
    build_graph_edges,
    build_label_rows,
    build_role_composition_rows,
    build_source_role_map,
    build_surface_comparison_rows,
    build_target_projection_mode_rows,
    canonical_role_fingerprint,
    detect_available_memory_bytes,
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
    role_graph_sort_key,
    score_concept_against_target_family,
)
from v6.concept_candidates_v10fixc import (
    apply_candidate_evidence_policy,
)
from v6.m4_failure_diagnostics import count_by_reason, ensure_failure_buckets, merge_reason_counts
from v6.role_transfer_v09 import _write_parquet
from v6.role_transfer_v09c import (
    RoleTransferV09cConfig,
    SingleFamilyContext,
    build_single_family_context,
    list_heldout_families,
)


@dataclass(frozen=True)
class ConceptCandidatesV10FixDConfig:
    m3_input_dir: str = "runs/v6/v08d_cd2_extended32_sourceclean"
    transfer_input_dir: str = "runs/v6/v09c_transfer_hardened_extended32"
    m2_input_dir: str = "runs/v6/v07_cd2_extended32_expanded"
    m1_input_dir: str = "runs/v6/v06_cd2_extended32"
    previous_v09b_dir: str = "runs/v6/v09b_role_transfer_refined_sourceclean_extended32"
    output_dir: str = "runs/v6/v10_m4_concepts_fixd_extended32"
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
    memory_safe: bool = True
    write_shards: bool = True
    resume_from_shards: bool = False
    max_items_per_manifest_group: int = 64
    max_candidates_per_manifest_group: int = 50000
    candidate_chunk_size: int = 5000
    max_total_candidates_per_heldout: int = 250000


def run_concept_candidates_v10fixd(config: ConceptCandidatesV10FixDConfig) -> dict[str, Any]:
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
    game_to_manifest_family = load_game_to_manifest_family(config.game_set_manifest)
    heldout_families = list_heldout_families(
        RoleTransferV09cConfig(
            m2_input_dir=config.m2_input_dir,
            m1_input_dir=config.m1_input_dir,
            previous_v09b_dir=config.previous_v09b_dir,
            output_dir=config.output_dir,
            game_set_manifest=config.game_set_manifest,
            game_set_name=config.game_set_name,
            workers=1,
        )
    )

    family_summaries: list[dict[str, Any]] = []
    source_diag_rows: list[dict[str, Any]] = []
    manifest_diag_rows: list[dict[str, Any]] = []
    attrition_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    pending_families: list[str] = []
    for heldout_family in heldout_families:
        paths = family_fixd_paths(shards_dir, heldout_family)
        if config.resume_from_shards and paths["complete"].exists():
            family_summaries.extend(load_records(paths["summary"]))
            source_diag_rows.extend(load_records(paths["source_diag"]))
            manifest_diag_rows.extend(load_records(paths["manifest_diag"]))
            attrition_rows.extend(load_records(paths["attrition"]))
            memory_rows.extend(load_records(paths["memory_all"]))
            continue
        clear_family_partial_shards(paths)
        pending_families.append(heldout_family)

    effective_workers = choose_fixd_worker_count(config.workers, len(pending_families))
    if effective_workers <= 1:
        for heldout_family in pending_families:
            family_result = run_single_family_fixd(
                heldout_family=heldout_family,
                config=config,
                source_manifest_family_map=source_manifest_family_map,
                transfer_rows=transfer_by_heldout.get(heldout_family, []),
                game_to_manifest_family=game_to_manifest_family,
                shards_dir=shards_dir,
            )
            family_summaries.append(family_result["summary"])
            source_diag_rows.append(family_result["source_diag"])
            manifest_diag_rows.extend(family_result["manifest_diag_rows"])
            attrition_rows.append(family_result["attrition"])
            memory_rows.extend(family_result["memory_rows"])
            if config.memory_safe:
                del family_result
                gc.collect()
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(
                    run_single_family_fixd,
                    heldout_family=heldout_family,
                    config=config,
                    source_manifest_family_map=source_manifest_family_map,
                    transfer_rows=transfer_by_heldout.get(heldout_family, []),
                    game_to_manifest_family=game_to_manifest_family,
                    shards_dir=shards_dir,
                ): heldout_family
                for heldout_family in pending_families
            }
            for future in as_completed(futures):
                family_result = future.result()
                family_summaries.append(family_result["summary"])
                source_diag_rows.append(family_result["source_diag"])
                manifest_diag_rows.extend(family_result["manifest_diag_rows"])
                attrition_rows.append(family_result["attrition"])
                memory_rows.extend(family_result["memory_rows"])
                if config.memory_safe:
                    del family_result
                    gc.collect()

    source_diag_rows.append(aggregate_source_role_map_diagnostics(source_diag_rows))
    attrition_rows.append(aggregate_attrition_rows(attrition_rows))
    memory_rows.append(aggregate_memory_rows(memory_rows))

    try:
        payload = finalize_fixd_run(
            config=config,
            output_dir=output_dir,
            shards_dir=shards_dir,
            original_v10fixb=original_v10fixb,
            transfer_report=transfer_report,
            family_summaries=family_summaries,
            source_diag_rows=source_diag_rows,
            manifest_diag_rows=manifest_diag_rows,
            attrition_rows=attrition_rows,
            memory_rows=memory_rows,
        )
    except Exception as exc:
        payload = build_operational_incomplete_payload(
            config=config,
            original_v10fixb=original_v10fixb,
            source_diag_rows=source_diag_rows,
            attrition_rows=attrition_rows,
            memory_rows=memory_rows,
            error=str(exc),
        )
        (output_dir / "m4_failure_diagnostics.json").write_text(
            json.dumps({"m4_failure_diagnostics": payload.get("m4_failure_diagnostics", {})}, indent=2),
            encoding="utf-8",
        )
        (output_dir / "v10fixd_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (output_dir / "v10fixd_report.txt").write_text(format_report_fixd(payload), encoding="utf-8")
        _write_parquet(output_dir / "source_role_map_diagnostics.parquet", source_diag_rows)
        _write_parquet(output_dir / "source_manifest_resolution_diagnostics.parquet", manifest_diag_rows)
        _write_parquet(output_dir / "candidate_attrition_diagnostics.parquet", attrition_rows)
        _write_parquet(output_dir / "memory_diagnostics.parquet", memory_rows)
        return payload

    (output_dir / "m4_failure_diagnostics.json").write_text(
        json.dumps({"m4_failure_diagnostics": payload.get("m4_failure_diagnostics", {})}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "v10fixd_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "v10fixd_report.txt").write_text(format_report_fixd(payload), encoding="utf-8")
    return payload


def run_single_family_fixd(
    *,
    heldout_family: str,
    config: ConceptCandidatesV10FixDConfig,
    source_manifest_family_map: dict[str, tuple[str, ...]],
    transfer_rows: list[dict[str, Any]],
    game_to_manifest_family: dict[str, str],
    shards_dir: Path,
) -> dict[str, Any]:
    memory_rows: list[dict[str, Any]] = []
    start_time = time.time()
    append_memory_row(memory_rows, heldout_family, "start", start_time, {})

    context = build_single_family_context(
        RoleTransferV09cConfig(
            m2_input_dir=config.m2_input_dir,
            m1_input_dir=config.m1_input_dir,
            previous_v09b_dir=config.previous_v09b_dir,
            output_dir=config.output_dir,
            game_set_manifest=config.game_set_manifest,
            game_set_name=config.game_set_name,
            workers=1,
        ),
        heldout_family,
    )
    append_memory_row(
        memory_rows,
        heldout_family,
        "after_load_single_family_context",
        start_time,
        {"source_neighborhood_count": len(context.source_neighborhoods), "source_role_count": len(context.source_roles)},
    )

    source_role_map = build_source_role_map(context.source_roles)
    append_memory_row(
        memory_rows,
        heldout_family,
        "after_source_role_map",
        start_time,
        {"source_neighborhood_count": len(context.source_neighborhoods), "source_role_count": len(context.source_roles)},
    )

    source_items, source_diag, manifest_diag_rows, manifest_groups = collect_source_items_and_manifest_groups(
        context=context,
        source_role_map=source_role_map,
        source_manifest_family_map=source_manifest_family_map,
        game_to_manifest_family=game_to_manifest_family,
    )
    append_memory_row(
        memory_rows,
        heldout_family,
        "after_manifest_groups",
        start_time,
        {
            "source_neighborhood_count": len(context.source_neighborhoods),
            "source_role_count": len(context.source_roles),
            "manifest_group_count": len(manifest_groups),
        },
    )

    paths = family_fixd_paths(shards_dir, heldout_family)
    _write_parquet(paths["source_diag"], [source_diag])
    _write_parquet(paths["manifest_diag"], manifest_diag_rows)
    _write_parquet(paths["memory_stage0"], [memory_rows[-1]])
    append_memory_row(
        memory_rows,
        heldout_family,
        "after_diagnostics_write",
        start_time,
        {
            "source_neighborhood_count": len(context.source_neighborhoods),
            "source_role_count": len(context.source_roles),
            "manifest_group_count": len(manifest_groups),
        },
    )

    raw_parts = 0
    transfer_parts = 0
    total_raw_written = 0
    total_transfer_written = 0
    stable_count = 0
    fallback_count = 0
    mixed_count = 0
    for manifest_family, items in sorted(manifest_groups.items()):
        group_raw_total = 0
        for chunk in generate_candidate_chunks_for_group(
            source_fold=heldout_family,
            heldout_family=heldout_family,
            manifest_family=manifest_family,
            items=items,
            config=config,
        ):
            if not chunk:
                continue
            total_raw_written += len(chunk)
            group_raw_total += len(chunk)
            raw_parts += 1
            _write_parquet(paths["raw_part"](raw_parts), chunk)
            append_memory_row(
                memory_rows,
                heldout_family,
                "after_candidate_chunk",
                start_time,
                {
                    "manifest_group_count": len(manifest_groups),
                    "current_chunk_rows": len(chunk),
                    "total_raw_candidates_written": total_raw_written,
                    "total_transfer_rows_written": total_transfer_written,
                },
            )

            transfer_chunk, detail_chunk, failure_chunk, membership_chunk = score_candidate_chunk(chunk, context, transfer_rows)
            total_transfer_written += len(transfer_chunk)
            transfer_parts += 1
            _write_parquet(paths["transfer_part"](transfer_parts), transfer_chunk)
            _write_parquet(paths["target_part"](transfer_parts), detail_chunk)
            _write_parquet(paths["failure_part"](transfer_parts), failure_chunk)
            _write_parquet(paths["membership_part"](transfer_parts), membership_chunk)
            append_memory_row(
                memory_rows,
                heldout_family,
                "after_projection_chunk",
                start_time,
                {
                    "manifest_group_count": len(manifest_groups),
                    "current_chunk_rows": len(chunk),
                    "total_raw_candidates_written": total_raw_written,
                    "total_transfer_rows_written": total_transfer_written,
                },
            )
            stable_count += sum(1 for row in chunk if row["candidate_source"] == "stable_role")
            fallback_count += sum(1 for row in chunk if row["candidate_source"] == "fallback_neighborhood")
            mixed_count += sum(1 for row in chunk if row["candidate_source"] == "mixed")

            del chunk, transfer_chunk, detail_chunk, failure_chunk, membership_chunk
            gc.collect()
            append_memory_row(
                memory_rows,
                heldout_family,
                "after_gc",
                start_time,
                {
                    "manifest_group_count": len(manifest_groups),
                    "current_chunk_rows": 0,
                    "total_raw_candidates_written": total_raw_written,
                    "total_transfer_rows_written": total_transfer_written,
                },
            )
        if group_raw_total == 0 and manifest_family == "unknown_manifest_family":
            source_diag["unknown_manifest_group_candidate_count"] += 0

    attrition = {
        "heldout_family": heldout_family,
        "source_manifest_structures_total": len(manifest_groups),
        "generated_subcomposition_candidates_total": total_raw_written,
        "raw_candidate_count_premerge": total_raw_written,
        "stable_role_candidate_count": source_diag["stable_items_available"],
        "fallback_candidate_count": source_diag["fallback_items_available"],
        "fallback_manifest_group_count": sum(1 for items in manifest_groups.values() if any(item["candidate_source"] != "stable_role" for item in items)),
        "fallback_raw_candidate_count": fallback_count,
        "stable_role_raw_candidate_count": stable_count,
        "mixed_candidates_generated": mixed_count,
        "failure_mode": source_diag["failure_mode_if_zero_structures"],
    }
    summary = {
        "heldout_family": heldout_family,
        "source_only_concept_discovery": True,
        "target_role_overlap_used_in_main_score": False,
        "concept_candidates": total_raw_written,
        "positive_concept_lift": int(total_transfer_written > 0),
        "target_mean_concept_lift_vs_role_raw": 0.0,
        "target_mean_concept_lift_vs_m2": 0.0,
        "target_mean_concept_lift_vs_surface_raw": 0.0,
        "target_mean_concept_lift_vs_surface_hardened": 0.0,
    }
    _write_parquet(paths["attrition"], [attrition])
    _write_parquet(paths["summary"], [summary])
    append_memory_row(
        memory_rows,
        heldout_family,
        "end",
        start_time,
        {
            "manifest_group_count": len(manifest_groups),
            "total_raw_candidates_written": total_raw_written,
            "total_transfer_rows_written": total_transfer_written,
        },
    )
    _write_parquet(paths["memory_all"], memory_rows)
    paths["complete"].write_text(json.dumps({"heldout_family": heldout_family}), encoding="utf-8")
    return {
        "summary": summary,
        "source_diag": source_diag,
        "manifest_diag_rows": manifest_diag_rows,
        "attrition": attrition,
        "memory_rows": memory_rows,
    }


def collect_source_items_and_manifest_groups(
    *,
    context: SingleFamilyContext,
    source_role_map: dict[str, dict[str, Any]],
    source_manifest_family_map: dict[str, tuple[str, ...]],
    game_to_manifest_family: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    source_ids = set(context.source_neighborhoods)
    role_map_ids = set(source_role_map)
    overlap_ids = source_ids & role_map_ids
    overlap_ratio = len(overlap_ids) / max(1, len(source_ids))

    source_items: list[dict[str, Any]] = []
    manifest_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    manifest_diag_rows: list[dict[str, Any]] = []
    missing_role_info = 0
    missing_manifest = 0
    stable_items_available = 0
    fallback_items_available = 0

    for family_id, record in sorted(context.source_neighborhoods.items()):
        role_info = source_role_map.get(family_id)
        manifest_from_config = resolve_manifest_from_games(record, game_to_manifest_family)
        manifest_from_map = list(source_manifest_family_map.get(family_id, ()))
        manifest_from_record = list(get_game_families(record))
        final_manifest_resolution = manifest_from_config or manifest_from_map or manifest_from_record or ["unknown_manifest_family"]
        if not (manifest_from_config or manifest_from_map or manifest_from_record):
            missing_manifest += 1
        if role_info is None:
            missing_role_info += 1
            item = build_fallback_item(heldout_family=context.heldout_family, family_id=family_id, record=record)
            fallback_items_available += 1
        else:
            item = build_stable_item(heldout_family=context.heldout_family, family_id=family_id, record=record, role_info=role_info)
            stable_items_available += 1
        item["source_manifest_families"] = list(final_manifest_resolution)
        source_items.append(item)
        manifest_groups[final_manifest_resolution[0]].append(item)
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

    failure_mode = "unexpected_empty_pipeline"
    if len(context.source_neighborhoods) == 0:
        failure_mode = "no_source_neighborhoods"
    elif len(source_role_map) == 0:
        failure_mode = "no_source_roles"
    elif len(overlap_ids) == 0:
        failure_mode = "source_role_map_family_id_mismatch"
    elif len(manifest_groups) == 0:
        failure_mode = "manifest_resolution_failure"
    else:
        failure_mode = "none"

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
        "stable_items_available": stable_items_available,
        "fallback_items_available": fallback_items_available,
        "families_skipped_missing_role_info": missing_role_info,
        "families_skipped_missing_manifest": missing_manifest,
        "families_used_for_manifest_groups": sum(len(items) for items in manifest_groups.values()),
        "manifest_groups_created": len(manifest_groups),
        "source_manifest_structures_total": len(manifest_groups),
        "failure_mode_if_zero_structures": failure_mode,
    }
    return source_items, source_diag, manifest_diag_rows, manifest_groups


def build_stable_item(*, heldout_family: str, family_id: str, record: Any, role_info: dict[str, Any]) -> dict[str, Any]:
    fingerprint = canonical_role_fingerprint(role_info["role_label_candidate"], record)
    return {
        "source_fold": heldout_family,
        "family_id": family_id,
        "record": record,
        "role_id": role_info["role_id"],
        "role_label": role_info["role_label_candidate"],
        "canonical_role_fingerprint_hash": fingerprint["canonical_role_fingerprint_hash"],
        "canonical_role_signature_json": fingerprint["canonical_role_signature_json"],
        "canonical_role_label_or_family": fingerprint["canonical_role_label_or_family"],
        "canonical_role_similarity_vector": fingerprint["canonical_role_similarity_vector"],
        "unknown_role_flag": fingerprint["unknown_role_flag"],
        "candidate_source": "stable_role",
    }


def build_fallback_item(*, heldout_family: str, family_id: str, record: Any) -> dict[str, Any]:
    fingerprint = canonical_role_fingerprint("unknown_role_candidate", record)
    return {
        "source_fold": heldout_family,
        "family_id": family_id,
        "record": record,
        "role_id": family_id,
        "role_label": "unknown_role_candidate",
        "canonical_role_fingerprint_hash": fingerprint["canonical_role_fingerprint_hash"],
        "canonical_role_signature_json": fingerprint["canonical_role_signature_json"],
        "canonical_role_label_or_family": fingerprint["canonical_role_label_or_family"],
        "canonical_role_similarity_vector": fingerprint["canonical_role_similarity_vector"],
        "unknown_role_flag": True,
        "candidate_source": "fallback_neighborhood",
    }


def generate_candidate_chunks_for_group(
    *,
    source_fold: str,
    heldout_family: str,
    manifest_family: str,
    items: list[dict[str, Any]],
    config: ConceptCandidatesV10FixDConfig,
) -> Iterable[list[dict[str, Any]]]:
    usable_items = sorted(items, key=role_graph_sort_key)[: config.max_items_per_manifest_group]
    if len(usable_items) < 2:
        return
    source_by_role_id = {str(item["role_id"]): str(item["candidate_source"]) for item in usable_items}
    candidates: list[dict[str, Any]]
    if len(usable_items) <= 12:
        candidates = generate_subcomposition_candidates(
            source_fold=source_fold,
            heldout_family=heldout_family,
            manifest_family=manifest_family,
            items=usable_items,
            max_role_count=config.max_role_count,
        )[: config.max_candidates_per_manifest_group]
    else:
        candidates = generate_large_group_candidates(
            source_fold=source_fold,
            heldout_family=heldout_family,
            manifest_family=manifest_family,
            items=usable_items,
            config=config,
        )
    for candidate in candidates:
        fold_local_role_ids = [str(role_id) for role_id in candidate.get("fold_local_role_ids", ())]
        candidate_sources = {source_by_role_id.get(role_id, "fallback_neighborhood") for role_id in fold_local_role_ids}
        candidate["member_candidate_sources"] = sorted(candidate_sources)
        if candidate_sources == {"stable_role"}:
            candidate["candidate_source"] = "stable_role"
        elif candidate_sources == {"fallback_neighborhood"}:
            candidate["candidate_source"] = "fallback_neighborhood"
        else:
            candidate["candidate_source"] = "mixed"
        candidate["manifest_family_ids"] = list(candidate.get("source_manifest_families_present", [manifest_family]))
        candidate["manifest_resolution_status"] = "resolved" if manifest_family != "unknown_manifest_family" else "unknown_manifest"
    capped = candidates[: min(config.max_candidates_per_manifest_group, config.max_total_candidates_per_heldout)]
    for start in range(0, len(capped), config.candidate_chunk_size):
        yield capped[start : start + config.candidate_chunk_size]


def generate_large_group_candidates(
    *,
    source_fold: str,
    heldout_family: str,
    manifest_family: str,
    items: list[dict[str, Any]],
    config: ConceptCandidatesV10FixDConfig,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()

    def add_window(group: list[dict[str, Any]]) -> None:
        if len(group) < 2:
            return
        key = tuple(item["canonical_role_fingerprint_hash"] for item in group)
        if key in seen:
            return
        seen.add(key)
        selected.extend(
            generate_subcomposition_candidates(
                source_fold=source_fold,
                heldout_family=heldout_family,
                manifest_family=manifest_family,
                items=group,
                max_role_count=min(config.max_role_count, len(group)),
            )[:64]
        )

    for index in range(len(items) - 1):
        add_window(items[index : index + 2])
    for index in range(len(items) - 2):
        add_window(items[index : index + 3])
    for width in range(2, 6):
        for index in range(0, len(items) - width + 1):
            add_window(items[index : index + width])
    for combo in top_k_pairs(items, 16):
        add_window(list(combo))
    for combo in top_k_triples(items, 16):
        add_window(list(combo))
    return selected[: config.max_candidates_per_manifest_group]


def top_k_pairs(items: list[dict[str, Any]], k: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    scored = []
    for left, right in combinations(items, 2):
        score = abs(float(left["canonical_role_similarity_vector"].get("future:reachable_delta_mean", 0.0)) - float(right["canonical_role_similarity_vector"].get("future:reachable_delta_mean", 0.0)))
        scored.append((score, (left, right)))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [pair for _, pair in scored[:k]]


def top_k_triples(items: list[dict[str, Any]], k: int) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    scored = []
    for triple in combinations(items, 3):
        score = sum(float(item["canonical_role_similarity_vector"].get("graph:directional_asymmetry_score", 0.0)) for item in triple)
        scored.append((abs(score), triple))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [triple for _, triple in scored[:k]]


def score_candidate_chunk(
    raw_chunk: list[dict[str, Any]],
    context: SingleFamilyContext,
    target_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    transfer_rows = []
    target_detail_rows = []
    failure_rows = []
    membership_rows = []
    target_rows_by_family = defaultdict(list)
    for row in target_rows:
        target_rows_by_family[str(row.get("target_family_id", ""))].append(row)
    for candidate in raw_chunk:
        per_target = []
        for family in context.target_families:
            target_record = context.target_neighborhoods.get(family.family_id)
            if target_record is None:
                continue
            row = score_concept_against_target_family(candidate, context_like_fixb(context, target_rows_by_family), family.family_id, target_record, target_rows_by_family.get(family.family_id, []))
            row["concept_id"] = candidate["concept_id"]
            per_target.append(row)
            target_detail_rows.append({"heldout_family": context.heldout_family, **row})
        if not per_target:
            transfer = {
                "heldout_family": context.heldout_family,
                "concept_id": candidate["concept_id"],
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
            }
            failure_rows.append(
                {"heldout_family": context.heldout_family, "concept_id": candidate["concept_id"], "failure_reason": "missing_target_rows"}
            )
        else:
            scores = sorted((row["target_family_score"] for row in per_target), reverse=True)
            top3 = scores[:3]
            best = scores[0]
            mean_score = float(sum(scores) / len(scores))
            top3_mean = float(sum(top3) / len(top3))
            transfer = {
                "heldout_family": context.heldout_family,
                "concept_id": candidate["concept_id"],
                "projection_used": True,
                "failure_reason": "",
                "target_family_count": len(per_target),
                "target_concept_prediction_score": top3_mean,
                "target_best_match_score": best,
                "target_top3_mean_score": top3_mean,
                "target_full_mean_score": mean_score,
                "target_projection_coverage": len(per_target) / max(1, len(context.target_families)),
                "target_mean_concept_lift_vs_role_raw": mean_metric(per_target, "target_family_score") - mean_metric(per_target, "best_individual_role_baseline_raw"),
                "target_mean_concept_lift_vs_role_discounted": mean_metric(per_target, "target_family_score") - mean_metric(per_target, "best_individual_role_baseline_discounted"),
                "target_mean_concept_lift_vs_m2": mean_metric(per_target, "target_family_score") - mean_metric(per_target, "best_raw_m2_baseline"),
                "target_mean_concept_lift_vs_surface_raw": mean_metric(per_target, "target_family_score") - mean_metric(per_target, "best_surface_raw_baseline"),
                "target_mean_concept_lift_vs_surface_hardened": mean_metric(per_target, "target_family_score") - mean_metric(per_target, "surface_hardened_baseline"),
                "score_mode_best": best,
                "score_mode_top3": top3_mean,
                "score_mode_mean": mean_score,
                "best_individual_role_baseline_raw": mean_metric(per_target, "best_individual_role_baseline_raw"),
                "best_individual_role_baseline_discounted": mean_metric(per_target, "best_individual_role_baseline_discounted"),
                "best_raw_m2_baseline": mean_metric(per_target, "best_raw_m2_baseline"),
                "best_surface_raw_baseline": mean_metric(per_target, "best_surface_raw_baseline"),
                "surface_hardened_baseline": mean_metric(per_target, "surface_hardened_baseline"),
                "role_id_overlap_diagnostic": mean_metric(per_target, "role_id_overlap_diagnostic"),
                "role_sequence_similarity_diagnostic": mean_metric(per_target, "role_sequence_similarity_diagnostic"),
            }
            for fingerprint in candidate["canonical_role_fingerprint_hashes"]:
                membership_rows.append(
                    {
                        "heldout_family": context.heldout_family,
                        "concept_id": candidate["concept_id"],
                        "source_fold": candidate["source_fold"],
                        "heldout_source_family": candidate["heldout_family"],
                        "canonical_role_fingerprint_hash": fingerprint,
                        "fold_local_role_ids": list(candidate["fold_local_role_ids"]),
                    }
                )
        transfer_rows.append(transfer)
    return transfer_rows, target_detail_rows, failure_rows, membership_rows


def finalize_fixd_run(
    *,
    config: ConceptCandidatesV10FixDConfig,
    output_dir: Path,
    shards_dir: Path,
    original_v10fixb: dict[str, Any] | None,
    transfer_report: dict[str, Any],
    family_summaries: list[dict[str, Any]],
    source_diag_rows: list[dict[str, Any]],
    manifest_diag_rows: list[dict[str, Any]],
    attrition_rows: list[dict[str, Any]],
    memory_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_candidate_rows = load_shard_group(shards_dir, "raw_concept_candidates_premerge__*.parquet")
    transfer_rows = load_shard_group(shards_dir, "concept_transfer_scores__*.parquet")
    target_rows = load_shard_group(shards_dir, "concept_target_family_scores__*.parquet")
    membership_rows = load_shard_group(shards_dir, "concept_membership__*.parquet")
    failure_rows = load_shard_group(shards_dir, "concept_failure_cases__*.parquet")

    _write_parquet(output_dir / "source_role_map_diagnostics.parquet", source_diag_rows)
    _write_parquet(output_dir / "source_manifest_resolution_diagnostics.parquet", manifest_diag_rows)
    _write_parquet(output_dir / "candidate_attrition_diagnostics.parquet", attrition_rows)
    _write_parquet(output_dir / "memory_diagnostics.parquet", memory_rows)
    if raw_candidate_rows:
        _write_parquet(output_dir / "raw_concept_candidates_premerge_fixd.parquet", raw_candidate_rows)
    if transfer_rows:
        _write_parquet(output_dir / "concept_transfer_scores_fixd.parquet", transfer_rows)
    if target_rows:
        _write_parquet(output_dir / "concept_target_family_scores_fixd.parquet", target_rows)

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

    mapped_transfer_rows = remap_concept_ids(transfer_rows, exact_to_final)
    mapped_target_rows = remap_concept_ids(target_rows, exact_to_final)
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

    label_rows = build_label_rows(concept_rows)
    composition_rows = build_role_composition_rows(concept_rows)
    graph_edges = build_graph_edges(concept_rows)
    surface_rows = build_surface_comparison_rows(mapped_transfer_rows)
    target_projection_mode_rows = build_target_projection_mode_rows(mapped_transfer_rows)
    concept_by_family_rows = build_concept_by_family_rows(concept_rows, mapped_transfer_rows)
    _write_parquet(output_dir / "concept_id_collision_diagnostics.parquet", collision_rows)
    _write_parquet(output_dir / "concept_label_diagnostics.parquet", label_rows)
    _write_parquet(output_dir / "surface_baseline_comparison.parquet", surface_rows)
    _write_parquet(output_dir / "fuzzy_grouping_diagnostics.parquet", fuzzy_diag_rows)
    _write_parquet(output_dir / "target_projection_mode_comparison.parquet", target_projection_mode_rows)
    _write_parquet(output_dir / "concept_membership_fixd.parquet", mapped_membership_rows)
    _write_parquet(output_dir / "concept_failure_cases_fixd.parquet", mapped_failure_rows)
    _write_parquet(output_dir / "concept_by_family_fixd.parquet", concept_by_family_rows)
    _write_parquet(output_dir / "concept_by_role_composition_fixd.parquet", composition_rows)
    _write_parquet(output_dir / "concept_graph_edges_fixd.parquet", graph_edges)
    _write_parquet(output_dir / "eligible_concept_candidates.parquet", eligible_candidates)
    _write_parquet(output_dir / "diagnostic_only_candidates.parquet", diagnostic_only_candidates)
    _write_parquet(output_dir / "concept_candidates_accepted.parquet", transferable_concepts)
    if concept_rows:
        _write_parquet(output_dir / "m4_concept_candidates_fixd.parquet", concept_rows)
        (output_dir / "m4_concept_candidates_fixd.json").write_text(json.dumps(concept_rows, indent=2), encoding="utf-8")

    payload = build_report_payload_fixd(
        config=config,
        original_v10fixb=original_v10fixb,
        transfer_report=transfer_report,
        concept_rows=concept_rows,
        stable_concepts=stable_concepts,
        transferable_concepts=transferable_concepts,
        family_summaries=family_summaries,
        source_diag_rows=source_diag_rows,
        attrition_rows=attrition_rows,
        collision_rows=collision_rows,
        label_rows=label_rows,
        fuzzy_diag_rows=fuzzy_diag_rows,
        exact_candidate_count=exact_candidate_count,
        fuzzy_candidate_count=len(fuzzy_rows),
        family_counts=family_counts,
        collision_pass=collision_pass,
        memory_rows=memory_rows,
        exact_filter_diag=exact_filter_diag,
        target_detail_rows=mapped_target_rows,
        mapped_transfer_rows=mapped_transfer_rows,
        lane_counts=lane_counts,
        eligible_candidate_count=len(eligible_candidates),
        diagnostic_only_candidate_count=len(diagnostic_only_candidates),
        accepted_candidate_count=len(transferable_concepts),
    )
    return payload


def build_fixd_failure_diagnostics(
    *,
    family_summaries: list[dict[str, Any]],
    source_diag_rows: list[dict[str, Any]],
    attrition_rows: list[dict[str, Any]],
    stable_concepts: list[dict[str, Any]],
    transferable_concepts: list[dict[str, Any]],
    mapped_transfer_rows: list[dict[str, Any]],
    exact_filter_diag: dict[str, int],
) -> dict[str, Any]:
    def _clean(counts: dict[str, int]) -> dict[str, int]:
        return {key: value for key, value in counts.items() if key not in {"", "none", "unexpected_empty_pipeline"}}

    family_ids = sorted(
        {
            str(row.get("heldout_family", ""))
            for row in family_summaries + source_diag_rows + attrition_rows + stable_concepts + transferable_concepts + mapped_transfer_rows
            if row.get("heldout_family") and row.get("heldout_family") != "__all__"
        }
    )
    summaries = {str(row["heldout_family"]): row for row in family_summaries if row.get("heldout_family") != "__all__"}
    source_diag = {str(row["heldout_family"]): row for row in source_diag_rows if row.get("heldout_family") != "__all__"}
    attrition = {str(row["heldout_family"]): row for row in attrition_rows if row.get("heldout_family") != "__all__"}
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
        summary = summaries.get(family_id, {})
        transfer_rows = transfer_by_family.get(family_id, [])
        stable_rows = stable_by_family.get(family_id, [])
        transferable_rows = transferable_by_family.get(family_id, [])
        reason_counts = merge_reason_counts(
            _clean(count_by_reason([diag], "failure_mode_if_zero_structures")),
            _clean(count_by_reason([summary], "zero_candidate_reason")),
        )
        projection_failures = sum(1 for row in transfer_rows if not row.get("projection_used"))
        if projection_failures:
            reason_counts = merge_reason_counts(reason_counts, {"no_target_projection": projection_failures})
        family_row = {
            "heldout_family_id": family_id,
            "source_neighborhoods_available": bool(diag.get("source_neighborhood_count", 0)),
            "source_roles_available": bool(diag.get("source_role_count", 0)),
            "source_role_overlap_ok": bool(diag.get("source_role_map_overlap_count", 0)),
            "stable_role_items_count": int(diag.get("stable_items_available", 0)),
            "raw_candidates_count": int(attr.get("raw_candidate_count_premerge", 0)),
            "fallback_candidate_count": int(attr.get("fallback_raw_candidate_count", 0)),
            "mixed_candidate_count": int(attr.get("mixed_candidates_generated", 0)),
            "unknown_manifest_candidate_count": 0,
            "rejected_candidate_count": max(0, int(attr.get("raw_candidate_count_premerge", 0)) - len(stable_rows)),
            "stable_candidate_count": len(stable_rows),
            "transferable_candidate_count": len(transferable_rows),
            "failure_reason_counts": ensure_failure_buckets(reason_counts),
        }
        per_family.append(family_row)

    total_reason_counts = merge_reason_counts(*(row["failure_reason_counts"] for row in per_family))
    total_reason_counts = merge_reason_counts(
        total_reason_counts,
        {
            "insufficient_games": int(exact_filter_diag.get("rejected_due_to_min_games", 0)),
            "insufficient_manifest_families": int(exact_filter_diag.get("rejected_due_to_min_families", 0)),
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


def build_report_payload_fixd(
    *,
    config: ConceptCandidatesV10FixDConfig,
    original_v10fixb: dict[str, Any] | None,
    transfer_report: dict[str, Any],
    concept_rows: list[dict[str, Any]],
    stable_concepts: list[dict[str, Any]],
    transferable_concepts: list[dict[str, Any]],
    family_summaries: list[dict[str, Any]],
    source_diag_rows: list[dict[str, Any]],
    attrition_rows: list[dict[str, Any]],
    collision_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    fuzzy_diag_rows: list[dict[str, Any]],
    exact_candidate_count: int,
    fuzzy_candidate_count: int,
    family_counts: Counter,
    collision_pass: bool,
    memory_rows: list[dict[str, Any]],
    exact_filter_diag: dict[str, int],
    target_detail_rows: list[dict[str, Any]],
    mapped_transfer_rows: list[dict[str, Any]],
    lane_counts: dict[str, int],
    eligible_candidate_count: int,
    diagnostic_only_candidate_count: int,
    accepted_candidate_count: int,
) -> dict[str, Any]:
    aggregate_source = next((row for row in source_diag_rows if row["heldout_family"] == "__all__"), aggregate_source_role_map_diagnostics(source_diag_rows))
    aggregate_attrition = next((row for row in attrition_rows if row["heldout_family"] == "__all__"), aggregate_attrition_rows(attrition_rows))
    peak_rss = max((row.get("rss_mb", 0.0) for row in memory_rows), default=0.0)
    positive_families = sum(1 for row in family_summaries if row["positive_concept_lift"])
    failure_diagnostics = build_fixd_failure_diagnostics(
        family_summaries=family_summaries,
        source_diag_rows=source_diag_rows,
        attrition_rows=attrition_rows,
        stable_concepts=stable_concepts,
        transferable_concepts=transferable_concepts,
        mapped_transfer_rows=mapped_transfer_rows,
        exact_filter_diag=exact_filter_diag,
    )

    if aggregate_attrition.get("source_manifest_structures_total", 0) == 0:
        conclusion = "m4_concepts_fixd_pipeline_not_diagnostic"
    elif not concept_rows and target_detail_rows:
        conclusion = "m4_concepts_fixd_not_established"
    elif not concept_rows and not target_detail_rows:
        conclusion = "m4_concepts_fixd_not_established"
    elif len(stable_concepts) >= 8 and len(transferable_concepts) >= 5 and mean_metric(transferable_concepts, "target_mean_concept_lift_vs_role_raw") >= 0.10 and mean_metric(transferable_concepts, "target_mean_concept_lift_vs_surface_raw") >= 0.10 and positive_families >= 12 and collision_pass:
        conclusion = "m4_concepts_fixd_very_strong"
    elif len(stable_concepts) >= 5 and len(transferable_concepts) >= 3 and mean_metric(transferable_concepts, "target_mean_concept_lift_vs_role_raw") >= 0.05 and mean_metric(transferable_concepts, "target_mean_concept_lift_vs_surface_raw") >= 0.05 and positive_families >= 8 and collision_pass:
        conclusion = "m4_concepts_fixd_strong"
    elif len(stable_concepts) >= 3 and len(transferable_concepts) >= 2 and mean_metric(transferable_concepts, "target_mean_concept_lift_vs_role_raw") > 0 and mean_metric(transferable_concepts, "target_mean_concept_lift_vs_surface_raw") > 0 and positive_families >= 6 and collision_pass:
        conclusion = "m4_concepts_fixd_weak"
    else:
        conclusion = "m4_concepts_fixd_not_established"

    report = {
        "original_v10fixb_summary": (original_v10fixb or {}).get("report", {}),
        "transfer_report_summary": transfer_report.get("report", {}),
        "source_only_concept_discovery": True,
        "target_role_id_overlap_removed_from_main_score": True,
        "target_role_overlap_diagnostic_only": True,
        "source_manifest_structures_total": aggregate_attrition.get("source_manifest_structures_total", 0),
        "generated_subcomposition_candidates_total": aggregate_attrition.get("generated_subcomposition_candidates_total", 0),
        "raw_candidate_count_premerge": aggregate_attrition.get("raw_candidate_count_premerge", 0),
        "exact_candidate_count": exact_candidate_count,
        "fuzzy_candidate_count": fuzzy_candidate_count,
        "exact_vs_fuzzy_delta": fuzzy_candidate_count - exact_candidate_count,
        "concept_id_collision_check_passed": collision_pass,
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
        "target_mean_concept_lift_vs_role_raw": mean_metric(transferable_concepts, "target_mean_concept_lift_vs_role_raw"),
        "target_mean_concept_lift_vs_m2": mean_metric(transferable_concepts, "target_mean_concept_lift_vs_m2"),
        "target_mean_concept_lift_vs_surface_raw": mean_metric(transferable_concepts, "target_mean_concept_lift_vs_surface_raw"),
        "positive_concept_lift_families": positive_families,
        "peak_rss_mb": peak_rss,
        "memory_safe_streaming_used": bool(config.streaming and config.memory_safe),
        "heldout_families_where_concepts_transfer": sorted([row["heldout_family"] for row in family_summaries if row["positive_concept_lift"]]),
        "heldout_families_where_concepts_fail": sorted([row["heldout_family"] for row in family_summaries if not row["positive_concept_lift"]]),
        "failure_mode_if_not_established": aggregate_source.get("failure_mode_if_zero_structures", "unexpected_empty_pipeline"),
        "scientific_conclusion": conclusion,
        "v10a_can_proceed": conclusion in {"m4_concepts_fixd_weak", "m4_concepts_fixd_strong", "m4_concepts_fixd_very_strong"},
        "family_occurrence_counts": dict(sorted(family_counts.items())),
        "source_role_map_diag_summary": aggregate_source,
        "attrition": aggregate_attrition,
        "target_family_score_count": len(target_detail_rows),
    }
    return {
        "config": {
            "output_dir": config.output_dir,
            "workers": config.workers,
            "previous_v09b_dir": config.previous_v09b_dir,
            "streaming": config.streaming,
            "memory_safe": config.memory_safe,
            "resume_from_shards": config.resume_from_shards,
        },
        "report": report,
        "m4_failure_diagnostics": failure_diagnostics,
        "validation": {
            "diagnostic_success": bool(source_diag_rows),
            "scientific_conclusion": conclusion,
            "proceed_to_v10a": report["v10a_can_proceed"],
        },
    }


def build_operational_incomplete_payload(
    *,
    config: ConceptCandidatesV10FixDConfig,
    original_v10fixb: dict[str, Any] | None,
    source_diag_rows: list[dict[str, Any]],
    attrition_rows: list[dict[str, Any]],
    memory_rows: list[dict[str, Any]],
    error: str,
) -> dict[str, Any]:
    aggregate_source = next((row for row in source_diag_rows if row["heldout_family"] == "__all__"), aggregate_source_role_map_diagnostics(source_diag_rows))
    aggregate_attrition = next((row for row in attrition_rows if row["heldout_family"] == "__all__"), aggregate_attrition_rows(attrition_rows))
    failure_diagnostics = build_fixd_failure_diagnostics(
        family_summaries=[],
        source_diag_rows=source_diag_rows,
        attrition_rows=attrition_rows,
        stable_concepts=[],
        transferable_concepts=[],
        mapped_transfer_rows=[],
        exact_filter_diag={},
    )
    return {
        "config": {
            "output_dir": config.output_dir,
            "workers": config.workers,
            "previous_v09b_dir": config.previous_v09b_dir,
        },
        "report": {
            "original_v10fixb_summary": (original_v10fixb or {}).get("report", {}),
            "source_role_map_diag_summary": aggregate_source,
            "attrition": aggregate_attrition,
            "peak_rss_mb": max((row.get("rss_mb", 0.0) for row in memory_rows), default=0.0),
            "failure_mode_if_not_established": aggregate_source.get("failure_mode_if_zero_structures", "unexpected_empty_pipeline"),
            "scientific_conclusion": "m4_concepts_fixd_operational_incomplete",
            "v10a_can_proceed": False,
            "operational_error": error,
        },
        "m4_failure_diagnostics": failure_diagnostics,
        "validation": {
            "diagnostic_success": bool(source_diag_rows),
            "scientific_conclusion": "m4_concepts_fixd_operational_incomplete",
            "proceed_to_v10a": False,
        },
    }


def format_report_fixd(payload: dict[str, Any]) -> str:
    report = payload["report"]
    attr = report.get("attrition", {})
    diag = report.get("source_role_map_diag_summary", {})
    return "\n".join(
        [
            "ARC-AGI3 v0.10fix-d: diagnostics-first, memory-safe M4 pipeline",
            "",
            "1. v10fix-b summary",
            f"v10fixb_scientific_conclusion={report.get('original_v10fixb_summary', {}).get('scientific_conclusion', '')}",
            "",
            "2. source-role-map diagnostics",
            f"source_neighborhood_count={diag.get('source_neighborhood_count', 0)}",
            f"source_role_map_size={diag.get('source_role_map_size', 0)}",
            f"source_role_map_overlap_ratio={diag.get('source_role_map_overlap_ratio', 0.0)}",
            "",
            "3. manifest-resolution diagnostics",
            f"manifest_groups_created={diag.get('manifest_groups_created', 0)}",
            f"source_manifest_structures_total={diag.get('source_manifest_structures_total', 0)}",
            "",
            "4. candidate generation funnel",
            f"generated_subcomposition_candidates_total={report.get('generated_subcomposition_candidates_total', 0)}",
            f"raw_candidate_count_premerge={report.get('raw_candidate_count_premerge', 0)}",
            "",
            "5. stable vs fallback candidate counts",
            f"stable_items_available={diag.get('stable_items_available', 0)}",
            f"fallback_items_available={diag.get('fallback_items_available', 0)}",
            f"stable_role_raw_candidate_count={attr.get('stable_role_raw_candidate_count', 0)}",
            f"fallback_raw_candidate_count={attr.get('fallback_raw_candidate_count', 0)}",
            f"mixed_candidates_generated={attr.get('mixed_candidates_generated', 0)}",
            "",
            "6. candidate attrition diagnostics",
            f"candidate_groups_before_support_filter={attr.get('candidate_groups_before_support_filter', 0)}",
            f"candidate_groups_after_support_filter={attr.get('candidate_groups_after_support_filter', 0)}",
            "",
            "7. exact vs fuzzy grouping comparison",
            f"exact_candidate_count={report.get('exact_candidate_count', 0)}",
            f"fuzzy_candidate_count={report.get('fuzzy_candidate_count', 0)}",
            "",
            "8. concept ID collision check",
            f"concept_id_collision_check_passed={report.get('concept_id_collision_check_passed', False)}",
            f"concept_id_collision_count={report.get('concept_id_collision_count', 0)}",
            "",
            "9. target projection mode comparison",
            f"target_family_score_count={report.get('target_family_score_count', 0)}",
            "",
            "10. raw role / raw M2 / raw surface baseline comparison",
            f"target_mean_concept_lift_vs_role_raw={report.get('target_mean_concept_lift_vs_role_raw', 0.0)}",
            f"target_mean_concept_lift_vs_m2={report.get('target_mean_concept_lift_vs_m2', 0.0)}",
            f"target_mean_concept_lift_vs_surface_raw={report.get('target_mean_concept_lift_vs_surface_raw', 0.0)}",
            "",
            "11. memory diagnostics",
            f"peak_rss_mb={report.get('peak_rss_mb', 0.0)}",
            f"memory_safe_streaming_used={report.get('memory_safe_streaming_used', False)}",
            "",
            "12. stable and transferable concepts",
            f"corrected_stable_concepts={report.get('corrected_stable_concepts', 0)}",
            f"corrected_transferable_concepts={report.get('corrected_transferable_concepts', 0)}",
            "",
            "13. held-out families where concepts transfer",
            ",".join(report.get("heldout_families_where_concepts_transfer", [])) or "none",
            "",
            "14. held-out families where concepts fail",
            ",".join(report.get("heldout_families_where_concepts_fail", [])) or "none",
            "",
            "15. failure mode if not established",
            f"failure_mode_if_not_established={report.get('failure_mode_if_not_established', '')}",
            "",
            "16. scientific conclusion",
            f"scientific_conclusion={report.get('scientific_conclusion', '')}",
            "",
            "17. whether v0.10a can proceed",
            f"v10a_can_proceed={report.get('v10a_can_proceed', False)}",
        ]
    )


def family_fixd_paths(shards_dir: Path, heldout_family: str) -> dict[str, Any]:
    return {
        "source_diag": shards_dir / f"source_role_map_diagnostics__{heldout_family}.parquet",
        "manifest_diag": shards_dir / f"source_manifest_resolution_diagnostics__{heldout_family}.parquet",
        "memory_stage0": shards_dir / f"memory_diagnostics__{heldout_family}__stage0.parquet",
        "memory_all": shards_dir / f"memory_diagnostics__{heldout_family}.parquet",
        "raw_part": lambda n: shards_dir / f"raw_concept_candidates_premerge__{heldout_family}__part-{n:04d}.parquet",
        "transfer_part": lambda n: shards_dir / f"concept_transfer_scores__{heldout_family}__part-{n:04d}.parquet",
        "target_part": lambda n: shards_dir / f"concept_target_family_scores__{heldout_family}__part-{n:04d}.parquet",
        "failure_part": lambda n: shards_dir / f"concept_failure_cases__{heldout_family}__part-{n:04d}.parquet",
        "membership_part": lambda n: shards_dir / f"concept_membership__{heldout_family}__part-{n:04d}.parquet",
        "attrition": shards_dir / f"candidate_attrition__{heldout_family}.parquet",
        "summary": shards_dir / f"family_summary__{heldout_family}.parquet",
        "complete": shards_dir / f"complete__{heldout_family}.json",
    }


def clear_family_partial_shards(paths: dict[str, Any]) -> None:
    paths["complete"].unlink(missing_ok=True)
    for key in ("source_diag", "manifest_diag", "memory_stage0", "memory_all", "attrition", "summary"):
        value = paths[key]
        if isinstance(value, Path):
            value.unlink(missing_ok=True)
    for prefix in ("raw_part", "transfer_part", "target_part", "failure_part", "membership_part"):
        template = paths[prefix](0).name
        stem = template.replace("0000", "*")
        for shard in paths[prefix](0).parent.glob(stem):
            shard.unlink(missing_ok=True)


def choose_fixd_worker_count(requested_workers: int, task_count: int) -> int:
    if requested_workers <= 1 or task_count <= 1:
        return 1
    cpu_limit = os.cpu_count() or 1
    return max(1, min(requested_workers, task_count, cpu_limit))


def aggregate_source_role_map_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"heldout_family": "__all__", "failure_mode_if_zero_structures": "unexpected_empty_pipeline"}
    total_neighborhoods = sum(row.get("source_neighborhood_count", 0) for row in rows)
    total_overlap = sum(row.get("source_role_map_overlap_count", 0) for row in rows)
    modes = Counter(row.get("failure_mode_if_zero_structures", "unexpected_empty_pipeline") for row in rows)
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
        "stable_items_available": sum(row.get("stable_items_available", 0) for row in rows),
        "fallback_items_available": sum(row.get("fallback_items_available", 0) for row in rows),
        "families_skipped_missing_role_info": sum(row.get("families_skipped_missing_role_info", 0) for row in rows),
        "families_skipped_missing_manifest": sum(row.get("families_skipped_missing_manifest", 0) for row in rows),
        "families_used_for_manifest_groups": sum(row.get("families_used_for_manifest_groups", 0) for row in rows),
        "manifest_groups_created": sum(row.get("manifest_groups_created", 0) for row in rows),
        "source_manifest_structures_total": sum(row.get("source_manifest_structures_total", 0) for row in rows),
        "failure_mode_if_zero_structures": modes.most_common(1)[0][0],
    }


def aggregate_attrition_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"heldout_family": "__all__"}
    return {
        "heldout_family": "__all__",
        "source_manifest_structures_total": sum(row.get("source_manifest_structures_total", 0) for row in rows),
        "generated_subcomposition_candidates_total": sum(row.get("generated_subcomposition_candidates_total", 0) for row in rows),
        "raw_candidate_count_premerge": sum(row.get("raw_candidate_count_premerge", 0) for row in rows),
        "stable_role_candidate_count": sum(row.get("stable_role_candidate_count", 0) for row in rows),
        "fallback_candidate_count": sum(row.get("fallback_candidate_count", 0) for row in rows),
        "fallback_manifest_group_count": sum(row.get("fallback_manifest_group_count", 0) for row in rows),
        "fallback_raw_candidate_count": sum(row.get("fallback_raw_candidate_count", 0) for row in rows),
        "stable_role_raw_candidate_count": sum(row.get("stable_role_raw_candidate_count", 0) for row in rows),
        "mixed_candidates_generated": sum(row.get("mixed_candidates_generated", 0) for row in rows),
        "failure_mode": Counter(row.get("failure_mode", "unexpected_empty_pipeline") for row in rows).most_common(1)[0][0],
    }


def aggregate_memory_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"heldout_family": "__all__", "stage": "aggregate", "rss_mb": 0.0}
    return {
        "heldout_family": "__all__",
        "stage": "aggregate",
        "rss_mb": max(row.get("rss_mb", 0.0) for row in rows),
        "elapsed_seconds": sum(row.get("elapsed_seconds", 0.0) for row in rows),
        "source_neighborhood_count": sum(row.get("source_neighborhood_count", 0) for row in rows),
        "source_role_count": sum(row.get("source_role_count", 0) for row in rows),
        "manifest_group_count": sum(row.get("manifest_group_count", 0) for row in rows),
        "current_chunk_rows": 0,
        "total_raw_candidates_written": sum(row.get("total_raw_candidates_written", 0) for row in rows),
        "total_transfer_rows_written": sum(row.get("total_transfer_rows_written", 0) for row in rows),
    }


def append_memory_row(rows: list[dict[str, Any]], heldout_family: str, stage: str, start_time: float, extra: dict[str, Any]) -> None:
    rows.append(
        {
            "heldout_family": heldout_family,
            "stage": stage,
            "rss_mb": current_rss_mb(),
            "elapsed_seconds": round(time.time() - start_time, 3),
            "source_neighborhood_count": int(extra.get("source_neighborhood_count", 0)),
            "source_role_count": int(extra.get("source_role_count", 0)),
            "manifest_group_count": int(extra.get("manifest_group_count", 0)),
            "current_chunk_rows": int(extra.get("current_chunk_rows", 0)),
            "total_raw_candidates_written": int(extra.get("total_raw_candidates_written", 0)),
            "total_transfer_rows_written": int(extra.get("total_transfer_rows_written", 0)),
        }
    )


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


def load_game_to_manifest_family(manifest_path: str | None) -> dict[str, str]:
    if not manifest_path:
        return {}
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    mapping = {}
    for family, games in payload.get("families", {}).items():
        for game in games:
            mapping[str(game)] = str(family)
    return mapping


def resolve_manifest_from_games(record: Any, mapping: dict[str, str]) -> list[str]:
    return sorted({mapping[game] for game in get_games(record) if game in mapping})


def context_like_fixb(context: SingleFamilyContext, target_rows_by_family: dict[str, list[dict[str, Any]]]) -> Any:
    class _Context:
        pass

    proxy = _Context()
    proxy.source_roles = context.source_roles
    proxy.source_neighborhoods = context.source_neighborhoods
    proxy.target_families = context.target_families
    proxy.full_neighborhoods = context.target_neighborhoods
    return proxy


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return normalize_records(pd.read_parquet(path).to_dict(orient="records"))


def load_shard_group(shards_dir: Path, pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(shards_dir.glob(pattern)):
        rows.extend(load_records(path))
    return rows


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
