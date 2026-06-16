from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v6.concept_candidates_v10 import effect_temporal_profile, extract_role_graph_motif
from v6.role_transfer_v09 import _write_parquet, appearance_features, cosine_similarity, mean_vector
from v6.role_transfer_v09b import FamilyContext
from v6.role_transfer_v09c import (
    RoleTransferV09cConfig,
    future_option_behavior_features,
    graph_position_features,
    local_graph_motif_features,
    prepare_family_contexts,
)


RESIDUAL_PROFILE_KEYS = (
    "effect_residual_mean",
    "effect_residual_abs_mean",
    "effect_complexity",
    "reversible_effect_rate",
    "temporal_variance_proxy",
)

VALID_CONCEPT_LABELS = {
    "access_control_concept",
    "movement_constraint_concept",
    "coverage_expansion_concept",
    "sequence_dependency_concept",
    "reversible_system_concept",
    "transport_network_concept",
    "resource_unlock_concept",
    "delayed_trigger_concept",
    "state_preservation_concept",
    "unknown_concept_candidate",
}

MOTIF_FAMILY = {
    "source_to_sink": "flow",
    "chain": "flow",
    "bridge": "flow",
    "bottleneck": "constraint",
    "fork": "branching",
    "join": "branching",
    "loop": "cyclic",
    "reversible_pair": "cyclic",
    "delayed_dependency": "temporal",
}

_WORKER_SOURCE_MANIFEST_FAMILY_MAP: dict[str, tuple[str, ...]] | None = None


@dataclass(frozen=True)
class ConceptCandidatesV10FixBConfig:
    m3_input_dir: str = "runs/v6/v08d_cd2_extended32_sourceclean"
    transfer_input_dir: str = "runs/v6/v09c_transfer_hardened_extended32"
    m2_input_dir: str = "runs/v6/v07_cd2_extended32_expanded"
    m1_input_dir: str = "runs/v6/v06_cd2_extended32"
    output_dir: str = "runs/v6/v10_m4_concepts_fixb_extended32"
    game_set_manifest: str | None = None
    game_set_name: str | None = None
    workers: int = 25
    min_games: int = 3
    min_manifest_families: int = 2
    min_role_count: int = 2
    max_role_count: int = 5
    grouping_mode: str = "fuzzy_structural"
    role_fingerprint_similarity_threshold: float = 0.75
    concept_fingerprint_similarity_threshold: float = 0.70


def run_concept_candidates_v10fixb(config: ConceptCandidatesV10FixBConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_v10 = _load_optional_json(Path("runs/v6/v10_m4_concepts_extended32") / "v10_report.json")
    original_v10fix = _load_optional_json(Path("runs/v6/v10_m4_concepts_methodology_fixed_extended32") / "v10fix_report.json")
    transfer_report = json.loads((Path(config.transfer_input_dir) / "v09c_report.json").read_text(encoding="utf-8"))
    transfer_rows = pd.read_parquet(Path(config.transfer_input_dir) / "v09c_hardened_assignments.parquet").to_dict(orient="records")
    transfer_by_heldout = defaultdict(list)
    for row in transfer_rows:
        transfer_by_heldout[str(row["heldout_family"])].append(row)
    source_manifest_family_map = load_source_manifest_family_map(Path(config.m3_input_dir))

    family_contexts = prepare_family_contexts(
        RoleTransferV09cConfig(
            m2_input_dir=config.m2_input_dir,
            m1_input_dir=config.m1_input_dir,
            previous_v09b_dir="runs/v6/v09b_role_transfer_refined_sourceclean_extended32",
            output_dir=config.output_dir,
            game_set_manifest=config.game_set_manifest,
            game_set_name=config.game_set_name,
            workers=config.workers,
        )
    )
    tasks = [(context, transfer_by_heldout.get(context.heldout_family, []), config) for context in family_contexts]
    effective_workers = choose_effective_worker_count(
        tasks,
        requested_workers=config.workers,
        shared_state_bytes=estimate_pickle_size_bytes(source_manifest_family_map),
    )
    if effective_workers <= 1 or len(tasks) <= 1:
        family_results = [_evaluate_family(*task, source_manifest_family_map=source_manifest_family_map) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            initializer=_init_family_worker,
            initargs=(source_manifest_family_map,),
        ) as executor:
            family_results = list(executor.map(_evaluate_family_task, tasks, chunksize=1))
    family_results = sorted(family_results, key=lambda item: item["heldout_family"])

    raw_candidate_rows = [row for item in family_results for row in item["raw_candidate_rows"]]
    local_exact_rows = [row for item in family_results for row in item["local_exact_rows"]]
    transfer_score_rows = [row for item in family_results for row in item["transfer_rows"]]
    membership_rows = [row for item in family_results for row in item["membership_rows"]]
    failure_rows = [row for item in family_results for row in item["failure_rows"]]
    by_family_rows = [item["summary"] for item in family_results]
    attrition_rows = [item["attrition"] for item in family_results]

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
    mapped_membership_rows = remap_concept_ids(membership_rows, exact_to_final)
    mapped_failure_rows = remap_concept_ids(failure_rows, exact_to_final)
    concept_rows = apply_target_metrics(final_concept_rows, mapped_transfer_rows)
    concept_rows = annotate_projection_outcomes(concept_rows, mapped_transfer_rows)

    stable_concepts = [row for row in concept_rows if is_stable_candidate(row)]
    transferable_concepts = [row for row in stable_concepts if is_transferable_candidate(row)]

    family_counts = Counter()
    for row in raw_candidate_rows:
        for family in row["source_manifest_families_present"]:
            family_counts[family] += 1
    collision_pass = all(not row["collision_detected"] for row in collision_rows)

    candidate_attrition_rows = build_attrition_rows(
        attrition_rows=attrition_rows,
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
    payload = build_report_payload(
        config=config,
        effective_workers=effective_workers,
        original_v10=original_v10,
        original_v10fix=original_v10fix,
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
        local_exact_rows=local_exact_rows,
        family_counts=family_counts,
        collision_pass=collision_pass,
    )

    _write_parquet(output_dir / "raw_concept_candidates_premerge.parquet", raw_candidate_rows)
    _write_parquet(output_dir / "m4_concept_candidates_fixb.parquet", concept_rows)
    _write_parquet(output_dir / "concept_membership_fixb.parquet", mapped_membership_rows)
    _write_parquet(output_dir / "concept_transfer_scores_fixb.parquet", mapped_transfer_rows)
    _write_parquet(output_dir / "concept_by_family_fixb.parquet", concept_by_family_rows)
    _write_parquet(output_dir / "concept_by_role_composition_fixb.parquet", composition_rows)
    _write_parquet(output_dir / "concept_failure_cases_fixb.parquet", mapped_failure_rows)
    _write_parquet(output_dir / "concept_graph_edges_fixb.parquet", graph_edges)
    _write_parquet(output_dir / "concept_id_collision_diagnostics.parquet", collision_rows)
    _write_parquet(output_dir / "concept_label_diagnostics.parquet", label_rows)
    _write_parquet(output_dir / "surface_baseline_comparison.parquet", surface_rows)
    _write_parquet(output_dir / "candidate_attrition_diagnostics.parquet", candidate_attrition_rows)
    _write_parquet(output_dir / "fuzzy_grouping_diagnostics.parquet", fuzzy_diag_rows)
    _write_parquet(output_dir / "target_projection_mode_comparison.parquet", target_projection_mode_rows)
    (output_dir / "m4_concept_candidates_fixb.json").write_text(json.dumps(concept_rows, indent=2), encoding="utf-8")
    (output_dir / "v10fixb_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "v10fixb_report.txt").write_text(format_report(payload), encoding="utf-8")
    return payload


def _init_family_worker(source_manifest_family_map: dict[str, tuple[str, ...]]) -> None:
    global _WORKER_SOURCE_MANIFEST_FAMILY_MAP
    _WORKER_SOURCE_MANIFEST_FAMILY_MAP = source_manifest_family_map


def _evaluate_family_task(task: tuple[FamilyContext, list[dict[str, Any]], ConceptCandidatesV10FixBConfig]) -> dict[str, Any]:
    if _WORKER_SOURCE_MANIFEST_FAMILY_MAP is None:
        raise RuntimeError("source manifest family map was not initialized in worker")
    return _evaluate_family(*task, source_manifest_family_map=_WORKER_SOURCE_MANIFEST_FAMILY_MAP)


def choose_effective_worker_count(
    tasks: list[tuple[FamilyContext, list[dict[str, Any]], ConceptCandidatesV10FixBConfig]],
    *,
    requested_workers: int,
    shared_state_bytes: int = 0,
) -> int:
    if requested_workers <= 1 or len(tasks) <= 1:
        return 1

    cpu_limit = os.cpu_count() or 1
    worker_cap = max(1, min(requested_workers, len(tasks), cpu_limit))
    available_memory = detect_available_memory_bytes()
    if available_memory is None:
        return worker_cap

    task_sizes = [estimate_task_payload_bytes(task) for task in tasks]
    if not task_sizes:
        return worker_cap

    largest_task_bytes = max(task_sizes)
    baseline_worker_bytes = 16 * 1024 * 1024 * 1024
    # Process workers keep the full deserialized task, shared state, imports,
    # and transient pandas/numpy allocations during family evaluation.
    estimated_per_worker = max(
        largest_task_bytes * 6 + shared_state_bytes * 2 + 256 * 1024 * 1024,
        baseline_worker_bytes,
    )
    safe_budget = int(available_memory * 0.15)
    if estimated_per_worker <= 0 or safe_budget <= 0:
        return worker_cap
    memory_cap = max(1, safe_budget // estimated_per_worker)
    return max(1, min(worker_cap, memory_cap))


def estimate_task_payload_bytes(task: tuple[FamilyContext, list[dict[str, Any]], ConceptCandidatesV10FixBConfig]) -> int:
    return estimate_pickle_size_bytes(task)


def estimate_pickle_size_bytes(payload: Any) -> int:
    try:
        return len(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return 128 * 1024 * 1024


def detect_available_memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
        except OSError:
            return None
    return None


def _evaluate_family(
    context: FamilyContext,
    target_rows: list[dict[str, Any]],
    config: ConceptCandidatesV10FixBConfig,
    source_manifest_family_map: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    source_role_map = build_source_role_map(context.source_roles)
    raw_candidate_rows, family_attrition = discover_source_only_candidates(context, source_role_map, source_manifest_family_map, config)
    local_exact_rows, _ = merge_exact_candidates(raw_candidate_rows, min_games=1, min_manifest_families=1, min_role_count=config.min_role_count)

    transfer_rows = []
    membership_rows = []
    failure_rows = []
    for concept in local_exact_rows:
        projection = evaluate_target_projection_by_family(concept, context, target_rows)
        transfer_rows.append({"heldout_family": context.heldout_family, **projection})
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
    return {
        "heldout_family": context.heldout_family,
        "raw_candidate_rows": raw_candidate_rows,
        "local_exact_rows": local_exact_rows,
        "transfer_rows": transfer_rows,
        "membership_rows": membership_rows,
        "failure_rows": failure_rows,
        "summary": summary,
        "attrition": family_attrition,
    }


def discover_source_only_candidates(
    context: FamilyContext,
    source_role_map: dict[str, dict[str, Any]],
    source_manifest_family_map: dict[str, tuple[str, ...]],
    config: ConceptCandidatesV10FixBConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_groups = defaultdict(list)
    for family_id, record in sorted(context.source_neighborhoods.items()):
        role_info = source_role_map.get(family_id)
        if role_info is None:
            continue
        manifest_families = resolve_manifest_families_for_record(
            family_id=family_id,
            record=record,
            source_manifest_family_map=source_manifest_family_map,
        )
        if not manifest_families:
            continue
        item = build_role_item(context.heldout_family, family_id, record, role_info)
        item["source_manifest_families"] = list(manifest_families)
        manifest_groups[manifest_families[0]].append(item)

    raw_rows = []
    generated_total = 0
    for manifest_family, items in sorted(manifest_groups.items()):
        ordered = sorted(items, key=role_graph_sort_key)
        candidates = generate_subcomposition_candidates(
            source_fold=context.heldout_family,
            heldout_family=context.heldout_family,
            manifest_family=manifest_family,
            items=ordered,
            max_role_count=config.max_role_count,
        )
        generated_total += len(candidates)
        raw_rows.extend(candidates)

    attrition = {
        "heldout_family": context.heldout_family,
        "source_manifest_structures_total": len(manifest_groups),
        "generated_subcomposition_candidates_total": generated_total,
    }
    return sorted(raw_rows, key=lambda row: (row["concept_id"], row["local_candidate_id"])), attrition


def build_role_item(
    source_fold: str,
    family_id: str,
    record: Any,
    role_info: dict[str, Any],
) -> dict[str, Any]:
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
    }


def generate_subcomposition_candidates(
    *,
    source_fold: str,
    heldout_family: str,
    manifest_family: str,
    items: list[dict[str, Any]],
    max_role_count: int,
) -> list[dict[str, Any]]:
    if len(items) < 2:
        return []
    candidates = []
    seen = set()
    local_index = 0

    def add_candidate(generator_type: str, selected: list[dict[str, Any]], motif_override: str | None = None, future_override: str | None = None) -> None:
        nonlocal local_index
        role_ids = tuple(item["role_id"] for item in selected)
        key = (
            generator_type,
            tuple(item["canonical_role_fingerprint_hash"] for item in selected),
            motif_override or "",
            future_override or "",
        )
        if key in seen:
            return
        seen.add(key)
        local_index += 1
        candidates.append(
            build_raw_candidate_row(
                source_fold=source_fold,
                heldout_family=heldout_family,
                manifest_family=manifest_family,
                local_candidate_id=f"{manifest_family}-{local_index:04d}",
                generator_type=generator_type,
                items=selected,
                motif_override=motif_override,
                future_override=future_override,
            )
        )

    count = min(len(items), max_role_count)
    for combo_size in range(2, count + 1):
        for combo in combinations(items, combo_size):
            add_candidate("role_bundles", list(combo))

    for left_index in range(len(items) - 1):
        add_candidate("role_pairs", items[left_index : left_index + 2])

    for combo in combinations(items, 2):
        add_candidate("role_pairs", list(combo))

    for width in range(2, count + 1):
        for start in range(0, len(items) - width + 1):
            add_candidate("subchains", items[start : start + width])

    for start in range(0, len(items) - 2):
        add_candidate("role_triples", items[start : start + 3])

    if len(items) >= 3:
        for combo in combinations(items, 3):
            if role_group_coherence(list(combo)) >= 0.75:
                add_candidate("role_triples", list(combo))

    motif_candidates = build_graph_motif_candidates(items)
    for motif_type, selected in motif_candidates:
        add_candidate("graph_motifs", selected, motif_override=motif_type)

    for selected in candidate_future_compositions(items):
        add_candidate("future_option_compositions", selected)

    return candidates


def build_graph_motif_candidates(items: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    candidates = []
    ordered = items
    if len(ordered) >= 2:
        candidates.append(("source_to_sink", [ordered[0], ordered[-1]]))
        candidates.append(("reversible_pair", [ordered[0], ordered[1]]))
    if len(ordered) >= 3:
        candidates.append(("chain", ordered[:3]))
        candidates.append(("bridge", [ordered[0], ordered[1], ordered[-1]]))
        candidates.append(("fork", ordered[:3]))
        candidates.append(("join", ordered[-3:]))
        candidates.append(("bottleneck", [ordered[0], ordered[len(ordered) // 2], ordered[-1]]))
        candidates.append(("delayed_dependency", ordered[: min(len(ordered), 4)]))
    if len(ordered) >= 4:
        candidates.append(("loop", ordered[:4]))
    return candidates


def candidate_future_compositions(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    output = []
    for width in range(2, min(len(items), 5) + 1):
        for start in range(0, len(items) - width + 1):
            window = items[start : start + width]
            output.append(window)
    return output


def build_raw_candidate_row(
    *,
    source_fold: str,
    heldout_family: str,
    manifest_family: str,
    local_candidate_id: str,
    generator_type: str,
    items: list[dict[str, Any]],
    motif_override: str | None = None,
    future_override: str | None = None,
) -> dict[str, Any]:
    ordered = sorted(items, key=role_graph_sort_key)
    role_labels = tuple(item["role_label"] for item in ordered)
    graph_pattern = tuple(item["canonical_role_fingerprint_hash"] for item in ordered)
    motif_type = motif_override or extract_role_graph_motif(
        [
            {
                "role_id": item["role_id"],
                "predecessor_count": float(item["record"].directional_features.get("predecessor_count", 0.0)),
                "successor_count": float(item["record"].directional_features.get("successor_count", 0.0)),
                "future_delta": float(item["record"].future_option_features.get("reachable_delta_mean", 0.0)),
            }
            for item in ordered
        ]
    )
    future_composition_type = future_override or infer_future_composition_type(ordered)
    source_games_present = tuple(sorted({game for item in ordered for game in get_games(item["record"])}))
    source_manifest_families_present = tuple(
        sorted({family for item in ordered for family in item.get("source_manifest_families", [manifest_family])})
    )
    future_profile = mean_vector([future_option_behavior_features(item["record"]) for item in ordered])
    graph_profile = mean_vector([graph_position_features(item["record"]) for item in ordered])
    motif_profile = mean_vector([local_graph_motif_features(item["record"]) for item in ordered])
    predsucc_profile = mean_vector([predecessor_successor_profile(item["record"]) for item in ordered])
    temporal_profile = mean_vector([effect_temporal_profile(item["record"]) for item in ordered])
    residual_profile = mean_vector([build_effect_residual_profile_from_record(item["record"]) for item in ordered])
    role_vectors = [item["canonical_role_similarity_vector"] for item in ordered]
    concept_fingerprint = build_concept_structural_fingerprint(
        motif_type=motif_type,
        future_composition_type=future_composition_type,
        role_count=len(ordered),
        future_profile=future_profile,
        graph_profile=graph_profile,
        motif_profile=motif_profile,
        predsucc_profile=predsucc_profile,
        temporal_profile=temporal_profile,
        residual_profile=residual_profile,
    )
    structural_bins = coarse_structural_profile_bins(len(ordered), graph_profile, future_profile, residual_profile)
    pattern_type = graph_pattern_type(ordered)
    signature = canonical_json(
        {
            "canonical_role_fingerprint_hashes": sorted(item["canonical_role_fingerprint_hash"] for item in ordered),
            "motif_type": motif_type,
            "future_composition_type": future_composition_type,
            "coarse_structural_profile_bins": structural_bins,
            "graph_ordered_pattern_type": pattern_type,
        }
    )
    concept_id = "m4-" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
    return {
        "source_fold": source_fold,
        "heldout_family": heldout_family,
        "generator_type": generator_type,
        "local_candidate_id": local_candidate_id,
        "role_ids_fold_local": [item["role_id"] for item in ordered],
        "fold_local_role_ids": [item["role_id"] for item in ordered],
        "role_labels": list(role_labels),
        "canonical_role_fingerprint_hashes": [item["canonical_role_fingerprint_hash"] for item in ordered],
        "canonical_role_signatures_json": [item["canonical_role_signature_json"] for item in ordered],
        "canonical_role_label_or_family": [item["canonical_role_label_or_family"] for item in ordered],
        "graph_ordered_role_pattern": list(graph_pattern),
        "graph_ordered_pattern_type": pattern_type,
        "motif_type": motif_type,
        "future_composition_type": future_composition_type,
        "source_games_present": list(source_games_present),
        "source_manifest_families_present": list(source_manifest_families_present),
        "source_support_count": len(source_games_present),
        "future_option_delta_profile": future_profile,
        "graph_position_profile": graph_profile,
        "local_motif_profile": motif_profile,
        "predecessor_successor_profile": predsucc_profile,
        "temporal_profile": temporal_profile,
        "effect_residual_profile": residual_profile,
        "role_similarity_vectors": role_vectors,
        "concept_structural_fingerprint": concept_fingerprint,
        "coarse_structural_profile_bins": structural_bins,
        "concept_signature_json": signature,
        "concept_id": concept_id,
        "source_manifest_family_support_signature": list(source_manifest_families_present),
        "heldout_families_seen": [heldout_family],
        "graph_order_fallback_used": True,
        "episode_order_available": False,
        "episode_ordered_role_sequence": [],
        "motif_family": MOTIF_FAMILY.get(motif_type, motif_type),
        "concept_label_candidate": strict_label_candidate(
            {
                "role_labels": tuple(role_labels),
                "future_option_delta_profile": future_profile,
                "motif_type": motif_type,
                "source_concept_quality_score": source_quality_score(ordered, motif_type),
                "manifest_family_count": 1,
                "role_ids": tuple(item["role_id"] for item in ordered),
                "effect_residual_profile": residual_profile,
                "temporal_profile": temporal_profile,
                "graph_position_profile": graph_profile,
            }
        )[0],
    }


def canonical_role_fingerprint(role_label_candidate: str, record: Any) -> dict[str, Any]:
    future_profile = future_option_behavior_features(record)
    graph_profile = graph_position_features(record)
    motif_profile = local_graph_motif_features(record)
    predsucc_profile = predecessor_successor_profile(record)
    temporal_profile = effect_temporal_profile(record)
    residual_profile = build_effect_residual_profile_from_record(record)
    unknown_role_flag = "unknown" in role_label_candidate
    signature = canonical_json(
        {
            "role_label_candidate": role_label_candidate,
            "future_option_features": rounded_dict(future_profile),
            "graph_position_features": rounded_dict(graph_profile),
            "local_motif_features": rounded_dict(motif_profile),
            "predecessor_successor_profile": rounded_dict(predsucc_profile),
            "effect_residual_temporal_profile": rounded_dict({**temporal_profile, **residual_profile}),
            "unknown_role_flag": unknown_role_flag,
        }
    )
    similarity_vector = {
        **prefix_dict("future", future_profile),
        **prefix_dict("graph", graph_profile),
        **prefix_dict("motif", motif_profile),
        **prefix_dict("predsucc", predsucc_profile),
        **prefix_dict("temp", temporal_profile),
        **prefix_dict("residual", residual_profile),
        "unknown_role_flag": 1.0 if unknown_role_flag else 0.0,
    }
    return {
        "canonical_role_fingerprint_hash": hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16],
        "canonical_role_signature_json": signature,
        "canonical_role_label_or_family": role_label_candidate,
        "canonical_role_similarity_vector": similarity_vector,
        "unknown_role_flag": unknown_role_flag,
    }


def predecessor_successor_profile(record: Any) -> dict[str, float]:
    return {
        "predecessor_count": float(record.directional_features.get("predecessor_count", 0.0)),
        "successor_count": float(record.directional_features.get("successor_count", 0.0)),
        "directional_asymmetry_score": float(record.directional_features.get("directional_asymmetry_score", 0.0)),
    }


def build_effect_residual_profile_from_record(record: Any) -> dict[str, float]:
    temporal = getattr(record, "temporal_effect_features", {})
    values = [float(value) for value in temporal.values()] if temporal else [0.0]
    reversible = float(temporal.get("reversible_effect_rate", 0.0))
    return {
        "effect_residual_mean": float(np.mean(values)),
        "effect_residual_abs_mean": float(np.mean([abs(value) for value in values])),
        "effect_complexity": float(np.std(values)),
        "reversible_effect_rate": reversible,
        "temporal_variance_proxy": float(np.var(values)),
    }


def build_effect_residual_profile_from_target_rows(target_rows: list[dict[str, Any]]) -> dict[str, float]:
    values = [float(row.get("effect_residual_score", 0.0)) for row in target_rows] or [0.0]
    abs_values = [abs(value) for value in values]
    return {
        "effect_residual_mean": float(np.mean(values)),
        "effect_residual_abs_mean": float(np.mean(abs_values)),
        "effect_complexity": float(np.std(values)),
        "reversible_effect_rate": float(np.mean([float(row.get("reversible_effect_rate", 0.0)) for row in target_rows])) if target_rows else 0.0,
        "temporal_variance_proxy": float(np.var(values)),
    }


def merge_exact_candidates(
    raw_rows: list[dict[str, Any]],
    *,
    min_games: int,
    min_manifest_families: int,
    min_role_count: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped = defaultdict(list)
    for row in raw_rows:
        grouped[row["concept_id"]].append(row)

    before_support = len(grouped)
    output = []
    diag = {
        "candidate_groups_before_support_filter": before_support,
        "candidate_groups_after_support_filter": 0,
        "rejected_due_to_min_games": 0,
        "rejected_due_to_min_families": 0,
        "rejected_due_to_min_role_count": 0,
        "rejected_due_to_exact_role_id_mismatch": 0,
    }
    for concept_id in sorted(grouped):
        rows = grouped[concept_id]
        merged = merge_candidate_group(rows)
        if len(merged["canonical_role_fingerprint_hashes"]) < min_role_count:
            diag["rejected_due_to_min_role_count"] += 1
            continue
        if merged["game_count"] < min_games:
            diag["rejected_due_to_min_games"] += 1
            continue
        if merged["manifest_family_count"] < min_manifest_families:
            diag["rejected_due_to_min_families"] += 1
            continue
        output.append(merged)
    diag["candidate_groups_after_support_filter"] = len(output)
    return output, diag


def merge_candidate_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = dict(rows[0])
    first["source_games_present"] = sorted({game for row in rows for game in row["source_games_present"]})
    first["source_manifest_families_present"] = sorted({family for row in rows for family in row["source_manifest_families_present"]})
    first["source_manifest_family_support_signature"] = sorted(first["source_manifest_families_present"])
    first["heldout_families_seen"] = sorted({family for row in rows for family in row["heldout_families_seen"]})
    first["fold_local_role_ids"] = sorted({role_id for row in rows for role_id in row["fold_local_role_ids"]})
    first["source_support_count"] = len(first["source_games_present"])
    first["future_option_delta_profile"] = mean_vector([row["future_option_delta_profile"] for row in rows])
    first["graph_position_profile"] = mean_vector([row["graph_position_profile"] for row in rows])
    first["local_motif_profile"] = mean_vector([row["local_motif_profile"] for row in rows])
    first["predecessor_successor_profile"] = mean_vector([row["predecessor_successor_profile"] for row in rows])
    first["temporal_profile"] = mean_vector([row["temporal_profile"] for row in rows])
    first["effect_residual_profile"] = mean_vector([row["effect_residual_profile"] for row in rows])
    first["concept_structural_fingerprint"] = mean_vector([row["concept_structural_fingerprint"] for row in rows])
    first["manifest_family_count"] = len(first["source_manifest_families_present"])
    first["game_count"] = len(first["source_games_present"])
    first["source_concept_quality_score"] = float(np.mean([source_quality_score_from_row(row) for row in rows]))
    first["concept_label_candidate"] = dominant_label(rows)
    return first


def evaluate_target_projection_by_family(
    concept: dict[str, Any],
    context: FamilyContext,
    target_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    targets = []
    target_rows_by_family = defaultdict(list)
    for row in target_rows:
        target_rows_by_family[str(row.get("target_family_id", ""))].append(row)

    for family in sorted(context.target_families, key=lambda item: item.family_id):
        target_record = context.full_neighborhoods.get(family.family_id)
        if target_record is None:
            continue
        family_target_rows = target_rows_by_family.get(family.family_id, [])
        targets.append(score_concept_against_target_family(concept, context, family.family_id, target_record, family_target_rows))

    if not targets:
        return {
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
        }

    target_scores = sorted((row["target_family_score"] for row in targets), reverse=True)
    top3 = target_scores[:3]
    best = target_scores[0]
    mean_score = float(np.mean(target_scores))
    top3_mean = float(np.mean(top3))
    return {
        "concept_id": concept["concept_id"],
        "projection_used": True,
        "failure_reason": "",
        "target_family_count": len(targets),
        "target_family_rows": targets,
        "target_concept_prediction_score": top3_mean,
        "target_best_match_score": best,
        "target_top3_mean_score": top3_mean,
        "target_full_mean_score": mean_score,
        "target_projection_coverage": len(targets) / max(1, len(context.target_families)),
        "target_mean_concept_lift_vs_role_raw": float(np.mean([row["target_family_score"] - row["best_individual_role_baseline_raw"] for row in targets])),
        "target_mean_concept_lift_vs_role_discounted": float(np.mean([row["target_family_score"] - row["best_individual_role_baseline_discounted"] for row in targets])),
        "target_mean_concept_lift_vs_m2": float(np.mean([row["target_family_score"] - row["best_raw_m2_baseline"] for row in targets])),
        "target_mean_concept_lift_vs_surface_raw": float(np.mean([row["target_family_score"] - row["best_surface_raw_baseline"] for row in targets])),
        "target_mean_concept_lift_vs_surface_hardened": float(np.mean([row["target_family_score"] - row["surface_hardened_baseline"] for row in targets])),
        "score_mode_best": best,
        "score_mode_top3": top3_mean,
        "score_mode_mean": mean_score,
        "best_individual_role_baseline_raw": float(np.mean([row["best_individual_role_baseline_raw"] for row in targets])),
        "best_individual_role_baseline_discounted": float(np.mean([row["best_individual_role_baseline_discounted"] for row in targets])),
        "best_raw_m2_baseline": float(np.mean([row["best_raw_m2_baseline"] for row in targets])),
        "best_surface_raw_baseline": float(np.mean([row["best_surface_raw_baseline"] for row in targets])),
        "surface_hardened_baseline": float(np.mean([row["surface_hardened_baseline"] for row in targets])),
        "role_id_overlap_diagnostic": float(np.mean([row["role_id_overlap_diagnostic"] for row in targets])),
        "role_sequence_similarity_diagnostic": float(np.mean([row["role_sequence_similarity_diagnostic"] for row in targets])),
    }


def score_concept_against_target_family(
    concept: dict[str, Any],
    context: FamilyContext,
    target_family_id: str,
    target_record: Any,
    target_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    target_future = future_option_behavior_features(target_record)
    target_graph = graph_position_features(target_record)
    target_motif = local_graph_motif_features(target_record)
    target_predsucc = predecessor_successor_profile(target_record)
    target_temporal = effect_temporal_profile(target_record)
    target_residual = build_effect_residual_profile_from_target_rows(target_rows)
    _ensure_residual_overlap(concept["effect_residual_profile"], target_residual)

    future_match = cosine_similarity(concept["future_option_delta_profile"], target_future)
    graph_match = cosine_similarity(concept["graph_position_profile"], target_graph)
    motif_match = cosine_similarity(concept["local_motif_profile"], target_motif)
    predsucc_match = cosine_similarity(concept["predecessor_successor_profile"], target_predsucc)
    temporal_match = cosine_similarity(concept["temporal_profile"], target_temporal)
    residual_match = cosine_similarity(concept["effect_residual_profile"], target_residual)
    target_score = float(0.28 * future_match + 0.24 * graph_match + 0.16 * motif_match + 0.14 * predsucc_match + 0.10 * temporal_match + 0.08 * residual_match)

    role_baseline_raw = best_individual_role_score_raw(context, target_record)
    role_baseline_discounted = role_baseline_raw * 0.75
    raw_m2_baseline = best_raw_m2_score(context, target_record)
    surface_raw_baseline = best_surface_raw_score(context, target_record)
    surface_hardened_baseline = float(np.mean([float(row.get("surface_hardened_score", 0.0)) for row in target_rows])) if target_rows else 0.0

    target_role_ids = sorted(dict.fromkeys(str(row.get("assigned_role_id", "")) for row in target_rows if row.get("assigned_role_id")))
    role_overlap = len(set(concept["fold_local_role_ids"]) & set(target_role_ids)) / max(1, len(set(concept["fold_local_role_ids"])))
    role_sequence_similarity = sequence_similarity(tuple(concept["graph_ordered_role_pattern"]), tuple(target_role_ids))

    return {
        "target_family_id": target_family_id,
        "target_family_score": target_score,
        "best_individual_role_baseline_raw": role_baseline_raw,
        "best_individual_role_baseline_discounted": role_baseline_discounted,
        "best_raw_m2_baseline": raw_m2_baseline,
        "best_surface_raw_baseline": surface_raw_baseline,
        "surface_hardened_baseline": surface_hardened_baseline,
        "role_id_overlap_diagnostic": role_overlap,
        "role_sequence_similarity_diagnostic": role_sequence_similarity,
    }


def best_individual_role_score_raw(context: FamilyContext, target_record: Any) -> float:
    target_future = future_option_behavior_features(target_record)
    target_graph = graph_position_features(target_record)
    best = 0.0
    for role in context.source_roles.values():
        future = subset_prefixed(role["all_features"], "future:")
        graph = subset_prefixed(role["all_features"], "directional:")
        score = 0.6 * cosine_similarity(target_future, future) + 0.4 * cosine_similarity(target_graph, graph)
        best = max(best, float(score))
    return best


def best_raw_m2_score(context: FamilyContext, target_record: Any) -> float:
    target_future = future_option_behavior_features(target_record)
    target_graph = graph_position_features(target_record)
    best = 0.0
    for record in context.source_neighborhoods.values():
        score = 0.6 * cosine_similarity(target_future, future_option_behavior_features(record)) + 0.4 * cosine_similarity(target_graph, graph_position_features(record))
        best = max(best, float(score))
    return best


def best_surface_raw_score(context: FamilyContext, target_record: Any) -> float:
    target_surface = appearance_features(target_record)
    best = 0.0
    for record in context.source_neighborhoods.values():
        best = max(best, cosine_similarity(target_surface, appearance_features(record)))
    return float(best)


def fuzzy_group_candidates(
    exact_rows: list[dict[str, Any]],
    *,
    role_threshold: float,
    concept_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    if not exact_rows:
        return [], [], {}
    ordered = sorted(exact_rows, key=lambda row: row["concept_id"])
    clusters: list[list[dict[str, Any]]] = []
    mapping: dict[str, str] = {}

    for row in ordered:
        placed = False
        for cluster in clusters:
            representative = cluster[0]
            if concepts_are_fuzzy_compatible(row, representative, role_threshold=role_threshold, concept_threshold=concept_threshold):
                cluster.append(row)
                placed = True
                break
        if not placed:
            clusters.append([row])

    fuzzy_rows = []
    diag_rows = []
    for cluster_index, cluster in enumerate(clusters, start=1):
        cluster = sorted(cluster, key=lambda row: row["concept_id"])
        representative = cluster[0]
        final_id = representative["concept_id"]
        for row in cluster:
            mapping[row["concept_id"]] = final_id
        merged = merge_candidate_group(cluster)
        merged["exact_member_concept_ids"] = [row["concept_id"] for row in cluster]
        merged["exact_member_count"] = len(cluster)
        fuzzy_rows.append(merged)
        diag_rows.append(
            {
                "fuzzy_group_index": cluster_index,
                "final_concept_id": final_id,
                "exact_member_count": len(cluster),
                "exact_member_concept_ids": [row["concept_id"] for row in cluster],
                "motif_type": merged["motif_type"],
                "future_composition_type": merged["future_composition_type"],
            }
        )
    return sorted(fuzzy_rows, key=lambda row: row["concept_id"]), diag_rows, mapping


def concepts_are_fuzzy_compatible(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    role_threshold: float,
    concept_threshold: float,
) -> bool:
    same_motif_family = MOTIF_FAMILY.get(left["motif_type"], left["motif_type"]) == MOTIF_FAMILY.get(right["motif_type"], right["motif_type"])
    same_future_comp = left["future_composition_type"] == right["future_composition_type"]
    role_sim = average_role_fingerprint_similarity(left["role_similarity_vectors"], right["role_similarity_vectors"])
    concept_sim = cosine_similarity(left["concept_structural_fingerprint"], right["concept_structural_fingerprint"])
    return same_motif_family and same_future_comp and role_sim >= role_threshold and concept_sim >= concept_threshold


def apply_target_metrics(concept_rows: list[dict[str, Any]], transfer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in transfer_rows:
        if row["projection_used"]:
            grouped[row["concept_id"]].append(row)
    output = []
    for row in concept_rows:
        projections = grouped.get(row["concept_id"], [])
        updated = dict(row)
        updated["target_mean_concept_prediction_score"] = mean_metric(projections, "target_concept_prediction_score")
        updated["target_best_match_score"] = mean_metric(projections, "target_best_match_score")
        updated["target_top3_mean_score"] = mean_metric(projections, "target_top3_mean_score")
        updated["target_full_mean_score"] = mean_metric(projections, "target_full_mean_score")
        updated["target_projection_coverage"] = mean_metric(projections, "target_projection_coverage")
        updated["target_mean_concept_lift_vs_role_raw"] = mean_metric(projections, "target_mean_concept_lift_vs_role_raw")
        updated["target_mean_concept_lift_vs_role_discounted"] = mean_metric(projections, "target_mean_concept_lift_vs_role_discounted")
        updated["target_mean_concept_lift_vs_m2"] = mean_metric(projections, "target_mean_concept_lift_vs_m2")
        updated["target_mean_concept_lift_vs_surface_raw"] = mean_metric(projections, "target_mean_concept_lift_vs_surface_raw")
        updated["target_mean_concept_lift_vs_surface_hardened"] = mean_metric(projections, "target_mean_concept_lift_vs_surface_hardened")
        updated["score_mode_best"] = mean_metric(projections, "score_mode_best")
        updated["score_mode_top3"] = mean_metric(projections, "score_mode_top3")
        updated["score_mode_mean"] = mean_metric(projections, "score_mode_mean")
        updated["target_concept_prediction_score"] = updated["target_top3_mean_score"]
        updated["transfer_stability_score"] = max(0.0, updated["target_mean_concept_lift_vs_surface_raw"])
        updated["concept_stability_score"] = float(min(1.0, 0.55 * source_stability(updated) + 0.45 * updated["target_projection_coverage"]))
        output.append(updated)
    return output


def annotate_projection_outcomes(concept_rows: list[dict[str, Any]], transfer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in transfer_rows:
        grouped[row["concept_id"]].append(row)
    output = []
    for row in concept_rows:
        updated = dict(row)
        projections = grouped.get(row["concept_id"], [])
        updated["projection_failure_count"] = sum(1 for item in projections if not item["projection_used"])
        updated["positive_lift_family_count"] = sum(1 for item in projections if item["target_mean_concept_lift_vs_role_raw"] > 0 and item["target_mean_concept_lift_vs_surface_raw"] > 0)
        output.append(updated)
    return output


def build_attrition_rows(
    *,
    attrition_rows: list[dict[str, Any]],
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
    }
    return sorted(attrition_rows + [summary], key=lambda row: row["heldout_family"])


def build_report_payload(
    *,
    config: ConceptCandidatesV10FixBConfig,
    effective_workers: int,
    original_v10: dict[str, Any] | None,
    original_v10fix: dict[str, Any] | None,
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
    local_exact_rows: list[dict[str, Any]],
    family_counts: Counter,
    collision_pass: bool,
) -> dict[str, Any]:
    metric = lambda rows, key: mean_metric(rows, key)
    positive_families = sum(1 for row in by_family_rows if row["positive_concept_lift"])
    families_spanned = sorted({family for row in concept_rows for family in row["source_manifest_families_present"]})
    dominant = max(label_rows, key=lambda row: row["percent"]) if label_rows else {"concept_label_candidate": "", "percent": 0.0}
    contribution_denominator = sum(max(0, row["positive_lift_family_count"]) for row in transferable_concepts) or 1
    max_contribution_share = max((row["positive_lift_family_count"] / contribution_denominator for row in transferable_concepts), default=0.0)

    conclusion = "m4_concepts_fixb_not_established"
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
        conclusion = "m4_concepts_fixb_very_strong"
    elif (
        len(stable_concepts) >= 5
        and len(transferable_concepts) >= 3
        and metric(transferable_concepts, "target_mean_concept_lift_vs_role_raw") >= 0.05
        and metric(transferable_concepts, "target_mean_concept_lift_vs_surface_raw") >= 0.05
        and positive_families >= 8
        and len(families_spanned) >= 8
        and collision_pass
    ):
        conclusion = "m4_concepts_fixb_strong"
    elif (
        len(stable_concepts) >= 3
        and len(transferable_concepts) >= 2
        and metric(transferable_concepts, "target_mean_concept_lift_vs_role_raw") > 0
        and metric(transferable_concepts, "target_mean_concept_lift_vs_surface_raw") > 0
        and positive_families >= 6
        and collision_pass
    ):
        conclusion = "m4_concepts_fixb_weak"

    aggregate_attrition = next((row for row in attrition_rows if row["heldout_family"] == "__all__"), {})
    failure_reason = "none"
    if fuzzy_candidate_count == 0:
        if aggregate_attrition.get("generated_subcomposition_candidates_total", 0) == 0:
            failure_reason = "lack_of_generated_subcompositions"
        elif aggregate_attrition.get("candidate_groups_after_support_filter", 0) == 0:
            failure_reason = "lack_of_cross_family_recurrence"
        elif aggregate_attrition.get("rejected_due_to_projection_failure", 0) > 0:
            failure_reason = "projection_failure"
        else:
            failure_reason = "no_lift_over_raw_baselines"

    projection_mode_rows = transferable_concepts or concept_rows
    report = {
        "original_v10_summary": original_v10["report"] if original_v10 else {},
        "original_v10fix_summary": original_v10fix["report"] if original_v10fix else {},
        "transfer_report_summary": transfer_report.get("report", {}),
        "source_only_concept_discovery": True,
        "target_role_id_overlap_removed_from_main_score": True,
        "target_role_overlap_diagnostic_only": True,
        "cross_fold_assignment_reuse": False,
        "target_family_excluded_from_source": True,
        "concept_id_collision_check_passed": collision_pass,
        "family_context_count": len(by_family_rows),
        "raw_candidate_count_premerge": aggregate_attrition.get("raw_candidate_count_premerge", 0),
        "concept_id_collision_count": sum(1 for row in collision_rows if row["collision_detected"]),
        "corrected_concept_candidate_count": len(concept_rows),
        "corrected_stable_concepts": len(stable_concepts),
        "corrected_transferable_concepts": len(transferable_concepts),
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
        "target_projection_mode_best_mean": metric(projection_mode_rows, "score_mode_best"),
        "target_projection_mode_top3_mean": metric(projection_mode_rows, "score_mode_top3"),
        "target_projection_mode_full_mean": metric(projection_mode_rows, "score_mode_mean"),
        "episode_order_available": any(row.get("episode_order_available", False) for row in concept_rows),
        "episode_order_coverage": float(np.mean([1.0 if row.get("episode_order_available", False) else 0.0 for row in concept_rows])) if concept_rows else 0.0,
        "graph_order_fallback_used": any(row.get("graph_order_fallback_used", False) for row in concept_rows),
        "heldout_families_where_concepts_transfer": sorted([row["heldout_family"] for row in by_family_rows if row["positive_concept_lift"]]),
        "heldout_families_where_concepts_fail": sorted([row["heldout_family"] for row in by_family_rows if not row["positive_concept_lift"]]),
        "scientific_conclusion": conclusion,
        "v10a_can_proceed": conclusion != "m4_concepts_fixb_not_established",
        "failure_mode_if_not_established": failure_reason,
        "grouping_mode": config.grouping_mode,
        "fuzzy_group_count_gt1": sum(1 for row in fuzzy_diag_rows if row["exact_member_count"] > 1),
        "family_occurrence_counts": dict(sorted(family_counts.items())),
        "attrition": aggregate_attrition,
    }
    return {
        "config": {
            "m3_input_dir": config.m3_input_dir,
            "transfer_input_dir": config.transfer_input_dir,
            "m2_input_dir": config.m2_input_dir,
            "m1_input_dir": config.m1_input_dir,
            "output_dir": config.output_dir,
            "game_set_manifest": config.game_set_manifest,
            "game_set_name": config.game_set_name,
            "workers": config.workers,
            "effective_workers": effective_workers,
            "grouping_mode": config.grouping_mode,
        },
        "report": report,
        "validation": {
            "diagnostic_success": bool(concept_rows),
            "scientific_conclusion": conclusion,
            "proceed_to_v10a": report["v10a_can_proceed"],
        },
    }


def format_report(payload: dict[str, Any]) -> str:
    report = payload["report"]
    attrition = report["attrition"]
    return "\n".join(
        [
            "ARC-AGI3 v0.10fix-b: clean M4 concept discovery and validation repair",
            "",
            "1. v0.10 and v0.10fix summaries",
            f"v0.10_scientific_conclusion={report['original_v10_summary'].get('scientific_conclusion', '')}",
            f"v0.10fix_scientific_conclusion={report['original_v10fix_summary'].get('scientific_conclusion', '')}",
            "",
            "2. Candidate generation funnel",
            f"source_manifest_structures_total={attrition.get('source_manifest_structures_total', 0)}",
            f"generated_subcomposition_candidates_total={attrition.get('generated_subcomposition_candidates_total', 0)}",
            f"raw_candidate_count_premerge={attrition.get('raw_candidate_count_premerge', 0)}",
            "",
            "3. Candidate attrition diagnostics",
            f"candidate_groups_before_support_filter={attrition.get('candidate_groups_before_support_filter', 0)}",
            f"candidate_groups_after_support_filter={attrition.get('candidate_groups_after_support_filter', 0)}",
            f"rejected_due_to_min_games={attrition.get('rejected_due_to_min_games', 0)}",
            f"rejected_due_to_min_families={attrition.get('rejected_due_to_min_families', 0)}",
            f"rejected_due_to_min_role_count={attrition.get('rejected_due_to_min_role_count', 0)}",
            f"rejected_due_to_exact_role_id_mismatch={attrition.get('rejected_due_to_exact_role_id_mismatch', 0)}",
            f"rejected_due_to_projection_failure={attrition.get('rejected_due_to_projection_failure', 0)}",
            f"rejected_due_to_no_positive_lift={attrition.get('rejected_due_to_no_positive_lift', 0)}",
            "",
            "4. Exact vs fuzzy grouping comparison",
            f"exact_candidate_count={report['exact_candidate_count']}",
            f"fuzzy_candidate_count={report['fuzzy_candidate_count']}",
            f"exact_vs_fuzzy_delta={report['exact_vs_fuzzy_delta']}",
            "",
            "5. Concept ID collision check",
            f"concept_id_collision_check_passed={report['concept_id_collision_check_passed']}",
            f"concept_id_collision_count={report['concept_id_collision_count']}",
            "",
            "6. Source-clean validation check",
            f"source_only_concept_discovery={report['source_only_concept_discovery']}",
            f"cross_fold_assignment_reuse={report['cross_fold_assignment_reuse']}",
            f"target_family_excluded_from_source={report['target_family_excluded_from_source']}",
            "",
            "7. Confirmation that target role-ID overlap is diagnostic-only",
            f"target_role_id_overlap_removed_from_main_score={report['target_role_id_overlap_removed_from_main_score']}",
            f"target_role_overlap_diagnostic_only={report['target_role_overlap_diagnostic_only']}",
            "",
            "8. Target projection mode comparison: best / top3 / mean",
            f"score_mode_best={report['target_projection_mode_best_mean']:.6f}",
            f"score_mode_top3={report['target_projection_mode_top3_mean']:.6f}",
            f"score_mode_mean={report['target_projection_mode_full_mean']:.6f}",
            "",
            "9. Raw vs hardened surface baseline comparison",
            f"target_mean_concept_lift_vs_surface_raw={report['target_mean_concept_lift_vs_surface_raw']:.6f}",
            f"target_mean_concept_lift_vs_surface_hardened={report['target_mean_concept_lift_vs_surface_hardened']:.6f}",
            "",
            "10. Corrected stable and transferable concepts",
            f"corrected_stable_concepts={report['corrected_stable_concepts']}",
            f"corrected_transferable_concepts={report['corrected_transferable_concepts']}",
            "",
            "11. Lift vs raw individual M3 role baseline",
            f"target_mean_concept_lift_vs_role_raw={report['target_mean_concept_lift_vs_role_raw']:.6f}",
            "",
            "12. Lift vs raw M2 baseline",
            f"target_mean_concept_lift_vs_m2={report['target_mean_concept_lift_vs_m2']:.6f}",
            "",
            "13. Lift vs raw surface baseline",
            f"target_mean_concept_lift_vs_surface_raw={report['target_mean_concept_lift_vs_surface_raw']:.6f}",
            "",
            "14. Concept label distribution",
            json.dumps(report["concept_label_distribution"], separators=(",", ":")),
            "",
            "15. Held-out families where concepts transfer",
            ",".join(report["heldout_families_where_concepts_transfer"]) or "none",
            "",
            "16. Held-out families where concepts fail",
            ",".join(report["heldout_families_where_concepts_fail"]) or "none",
            "",
            "17. Corrected scientific conclusion",
            f"scientific_conclusion={report['scientific_conclusion']}",
            "",
            "18. Whether v0.10a can proceed",
            f"v10a_can_proceed={report['v10a_can_proceed']}",
        ]
    )


def strict_label_candidate(concept: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    role_labels = set(concept["role_labels"])
    reachability = concept["future_option_delta_profile"]
    motif_type = concept["motif_type"]
    residual = concept.get("effect_residual_profile", {})
    graph_profile = concept.get("graph_position_profile", {})
    transfer_evidence = concept["source_concept_quality_score"] >= 0.45 and concept["manifest_family_count"] >= 1
    evidence = {
        "required_roles_present": sorted(role_labels),
        "reachability_gating_evidence": abs(float(reachability.get("reachable_delta_mean", 0.0))) > 0.2 or abs(float(reachability.get("enable_score", 0.0))) > 0.2 or abs(float(reachability.get("block_score", 0.0))) > 0.2,
        "motif_evidence": motif_type,
        "transfer_evidence": transfer_evidence,
        "rejected_alternative_labels": [],
    }
    if {"blocker_candidate", "connector_candidate"} <= role_labels:
        return "access_control_concept", evidence
    if {"blocker_candidate", "movement_controller_candidate"} <= role_labels:
        evidence["rejected_alternative_labels"].append("access_control_concept")
        return "movement_constraint_concept", evidence
    if "coverage_expander_candidate" in role_labels:
        return "coverage_expansion_concept", evidence
    if motif_type == "chain" and len(concept["role_ids"]) >= 3:
        return "sequence_dependency_concept", evidence
    if motif_type in {"loop", "reversible_pair"} or float(residual.get("reversible_effect_rate", 0.0)) > 0.5:
        return "reversible_system_concept", evidence
    if motif_type in {"bridge", "source_to_sink"}:
        return "transport_network_concept", evidence
    if float(reachability.get("enable_score", 0.0)) > 0.4:
        return "resource_unlock_concept", evidence
    if motif_type == "delayed_dependency":
        return "delayed_trigger_concept", evidence
    if float(graph_profile.get("sink_like_score", 0.0)) < 0.2 and float(residual.get("effect_residual_abs_mean", 0.0)) < 0.2:
        return "state_preservation_concept", evidence
    return "unknown_concept_candidate", evidence


def build_source_role_map(source_roles: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for role_id, role in sorted(source_roles.items()):
        for family_id in role["member_family_ids"]:
            output[family_id] = {
                "role_id": role_id,
                "role_label_candidate": role["role_label_candidate"],
                "all_features": role["all_features"],
            }
    return output


def build_role_composition_rows(concept_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "concept_id": row["concept_id"],
            "role_count": len(row["canonical_role_fingerprint_hashes"]),
            "motif_type": row["motif_type"],
            "future_composition_type": row["future_composition_type"],
            "canonical_role_fingerprint_hashes": row["canonical_role_fingerprint_hashes"],
        }
        for row in concept_rows
    ]


def build_graph_edges(concept_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges = []
    for row in concept_rows:
        pattern = row["graph_ordered_role_pattern"]
        for left, right in zip(pattern, pattern[1:]):
            edges.append(
                {
                    "concept_id": row["concept_id"],
                    "source_role_fingerprint": left,
                    "target_role_fingerprint": right,
                    "graph_order_fallback_used": row["graph_order_fallback_used"],
                }
            )
    return edges


def build_collision_rows(raw_candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(set)
    for row in raw_candidate_rows:
        grouped[row["concept_id"]].add(row["concept_signature_json"])
    return [
        {
            "concept_id": concept_id,
            "distinct_signature_count": len(signatures),
            "collision_detected": len(signatures) > 1,
        }
        for concept_id, signatures in sorted(grouped.items())
    ]


def build_label_rows(concept_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(row["concept_label_candidate"] for row in concept_rows)
    total = float(sum(counts.values())) or 1.0
    return [{"concept_label_candidate": label, "count": count, "percent": count / total} for label, count in sorted(counts.items())]


def build_surface_comparison_rows(transfer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "heldout_family": row["heldout_family"],
            "concept_id": row["concept_id"],
            "best_surface_raw_baseline": row.get("best_surface_raw_baseline", 0.0),
            "surface_hardened_baseline": row.get("surface_hardened_baseline", 0.0),
            "target_concept_prediction_score": row.get("target_concept_prediction_score", 0.0),
            "target_mean_concept_lift_vs_surface_raw": row.get("target_mean_concept_lift_vs_surface_raw", 0.0),
            "target_mean_concept_lift_vs_surface_hardened": row.get("target_mean_concept_lift_vs_surface_hardened", 0.0),
        }
        for row in transfer_rows
    ]


def build_target_projection_mode_rows(transfer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "heldout_family": row["heldout_family"],
            "concept_id": row["concept_id"],
            "score_mode_best": row.get("score_mode_best", 0.0),
            "score_mode_top3": row.get("score_mode_top3", 0.0),
            "score_mode_mean": row.get("score_mode_mean", 0.0),
            "target_projection_coverage": row.get("target_projection_coverage", 0.0),
        }
        for row in transfer_rows
    ]


def build_concept_by_family_rows(concept_rows: list[dict[str, Any]], transfer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in transfer_rows:
        grouped[(row["concept_id"], row["heldout_family"])].append(row)
    output = []
    for concept in concept_rows:
        for heldout_family in sorted({row["heldout_family"] for row in transfer_rows}):
            rows = grouped.get((concept["concept_id"], heldout_family), [])
            output.append(
                {
                    "concept_id": concept["concept_id"],
                    "heldout_family": heldout_family,
                    "projection_count": len(rows),
                    "target_concept_prediction_score": mean_metric(rows, "target_concept_prediction_score"),
                    "target_mean_concept_lift_vs_role_raw": mean_metric(rows, "target_mean_concept_lift_vs_role_raw"),
                    "target_mean_concept_lift_vs_surface_raw": mean_metric(rows, "target_mean_concept_lift_vs_surface_raw"),
                }
            )
    return output


def remap_concept_ids(rows: list[dict[str, Any]], mapping: dict[str, str]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        updated = dict(row)
        updated["concept_id"] = mapping.get(row["concept_id"], row["concept_id"])
        output.append(updated)
    return output


def source_quality_score(items: list[dict[str, Any]], motif_type: str) -> float:
    future_mean = float(np.mean([abs(item["record"].future_option_features.get("reachable_delta_mean", 0.0)) for item in items])) if items else 0.0
    graph_mean = float(np.mean([abs(item["record"].directional_features.get("directional_asymmetry_score", 0.0)) for item in items])) if items else 0.0
    motif_bonus = {"chain": 0.08, "fork": 0.10, "join": 0.10, "loop": 0.12, "source_to_sink": 0.11, "bridge": 0.09, "reversible_pair": 0.12}.get(motif_type, 0.07)
    return float(min(1.0, 0.35 + 0.25 * future_mean + 0.20 * graph_mean + motif_bonus))


def source_quality_score_from_row(row: dict[str, Any]) -> float:
    return source_quality_score(
        [{"record": _dict_to_record_proxy(row["future_option_delta_profile"], row["graph_position_profile"])}],
        row["motif_type"],
    )


def source_stability(row: dict[str, Any]) -> float:
    return float(min(1.0, 0.35 * min(row["manifest_family_count"] / 4.0, 1.0) + 0.35 * min(row["game_count"] / 5.0, 1.0) + 0.30 * row["source_concept_quality_score"]))


def role_graph_sort_key(item: dict[str, Any]) -> tuple[float, float, str]:
    record = item["record"]
    return (
        float(record.directional_features.get("predecessor_count", 0.0)),
        -float(record.directional_features.get("successor_count", 0.0)),
        item["canonical_role_fingerprint_hash"],
    )


def graph_pattern_type(items: list[dict[str, Any]]) -> str:
    if len(items) == 2:
        return "pair"
    if len(items) == 3:
        return "triple"
    return "chain_window"


def coarse_structural_profile_bins(
    role_count: int,
    graph_profile: dict[str, float],
    future_profile: dict[str, float],
    residual_profile: dict[str, float],
) -> dict[str, str]:
    return {
        "role_count_bin": str(role_count),
        "graph_flow_bin": bin_name(abs(float(graph_profile.get("bridge_like_score", 0.0))) + abs(float(graph_profile.get("source_like_score", 0.0)))),
        "future_delta_bin": bin_name(abs(float(future_profile.get("reachable_delta_mean", 0.0)))),
        "residual_bin": bin_name(abs(float(residual_profile.get("effect_residual_abs_mean", 0.0)))),
    }


def infer_future_composition_type(items: list[dict[str, Any]]) -> str:
    enable = np.mean([float(item["record"].future_option_features.get("enable_score", 0.0)) for item in items])
    block = np.mean([float(item["record"].future_option_features.get("block_score", 0.0)) for item in items])
    preserve = np.mean([float(item["record"].future_option_features.get("preserve_score", 0.0)) for item in items])
    terminate = np.mean([float(item["record"].future_option_features.get("terminate_score", 0.0)) for item in items])
    reach = np.mean([float(item["record"].future_option_features.get("reachable_delta_mean", 0.0)) for item in items])
    if enable > 0.4 and reach > 0.2:
        return "enable_then_expand"
    if block > 0.4 and reach < 0.0:
        return "block_then_redirect"
    if preserve > 0.3 and enable > 0.2:
        return "preserve_then_trigger"
    if enable > 0.3 and np.mean([float(item["record"].directional_features.get("bridge_like_score", 0.0)) for item in items]) > 0.2:
        return "transport_then_unlock"
    if block > 0.3 and terminate > 0.1:
        return "constrain_then_terminate"
    if reach > 0.4:
        return "expand_then_cover"
    return "unknown_future_composition"


def build_concept_structural_fingerprint(
    *,
    motif_type: str,
    future_composition_type: str,
    role_count: int,
    future_profile: dict[str, float],
    graph_profile: dict[str, float],
    motif_profile: dict[str, float],
    predsucc_profile: dict[str, float],
    temporal_profile: dict[str, float],
    residual_profile: dict[str, float],
) -> dict[str, float]:
    return {
        **prefix_dict("future", future_profile),
        **prefix_dict("graph", graph_profile),
        **prefix_dict("motif", motif_profile),
        **prefix_dict("predsucc", predsucc_profile),
        **prefix_dict("temp", temporal_profile),
        **prefix_dict("residual", residual_profile),
        f"motif_family::{MOTIF_FAMILY.get(motif_type, motif_type)}": 1.0,
        f"future_comp::{future_composition_type}": 1.0,
        "role_count": float(role_count),
    }


def role_group_coherence(items: list[dict[str, Any]]) -> float:
    if len(items) < 2:
        return 0.0
    scores = []
    for left, right in combinations(items, 2):
        scores.append(role_fingerprint_similarity(left, right))
    return float(np.mean(scores)) if scores else 0.0


def role_fingerprint_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    base = cosine_similarity(left["canonical_role_similarity_vector"], right["canonical_role_similarity_vector"])
    label_bonus = 0.1 if left["role_label"] == right["role_label"] else 0.0
    return float(min(1.0, base + label_bonus))


def average_role_fingerprint_similarity(left_vectors: list[dict[str, float]], right_vectors: list[dict[str, float]]) -> float:
    if not left_vectors or not right_vectors:
        return 0.0
    scores = []
    for left in left_vectors:
        scores.append(max(cosine_similarity(left, right) for right in right_vectors))
    for right in right_vectors:
        scores.append(max(cosine_similarity(right, left) for left in left_vectors))
    return float(np.mean(scores)) if scores else 0.0


def dominant_label(rows: list[dict[str, Any]]) -> str:
    counts = Counter(row["concept_label_candidate"] for row in rows if row["concept_label_candidate"] in VALID_CONCEPT_LABELS)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0] if counts else "unknown_concept_candidate"


def is_stable_candidate(row: dict[str, Any]) -> bool:
    return row["manifest_family_count"] >= 2 and row["game_count"] >= 3 and row["concept_stability_score"] >= 0.45


def is_transferable_candidate(row: dict[str, Any]) -> bool:
    return row["target_projection_coverage"] > 0 and row["target_mean_concept_lift_vs_role_raw"] > 0 and row["target_mean_concept_lift_vs_surface_raw"] > 0 and row["transfer_stability_score"] >= 0.0


def subset_prefixed(vector: dict[str, float], prefix: str) -> dict[str, float]:
    return {key[len(prefix):]: float(value) for key, value in vector.items() if key.startswith(prefix)}


def sequence_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(set(left) & set(right))
    order_bonus = sum(1 for index, role_id in enumerate(left[: min(len(left), len(right))]) if right[index] == role_id)
    return float(0.6 * overlap / max(1, len(set(left))) + 0.4 * order_bonus / max(len(left), len(right)))


def get_games(record: Any) -> tuple[str, ...]:
    return tuple(sorted(getattr(record, "games_present", getattr(record, "game_ids", ()))))


def get_game_families(record: Any) -> tuple[str, ...]:
    return tuple(sorted(getattr(record, "game_families_present", getattr(record, "game_family_ids", ()))))


def rounded_dict(values: dict[str, float], places: int = 4) -> dict[str, float]:
    return {key: round(float(value), places) for key, value in sorted(values.items())}


def prefix_dict(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}:{key}": float(value) for key, value in values.items()}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def bin_name(value: float) -> str:
    if value < 0.2:
        return "low"
    if value < 0.5:
        return "mid"
    return "high"


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row.get(key, 0.0)) for row in rows])) if rows else 0.0


def _ensure_residual_overlap(source_profile: dict[str, float], target_profile: dict[str, float]) -> None:
    if set(source_profile) != set(target_profile):
        raise ValueError("source and target residual profile keys must overlap exactly")


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_source_manifest_family_map(m3_input_dir: Path) -> dict[str, tuple[str, ...]]:
    neighborhoods_path = m3_input_dir / "role_neighborhoods.parquet"
    if not neighborhoods_path.exists():
        return {}
    rows = pd.read_parquet(neighborhoods_path).to_dict(orient="records")
    output: dict[str, tuple[str, ...]] = {}
    for row in rows:
        family_id = str(row.get("family_id", ""))
        if not family_id:
            continue
        raw_families = row.get("game_families_present", [])
        if isinstance(raw_families, str):
            try:
                raw_families = json.loads(raw_families)
            except json.JSONDecodeError:
                raw_families = [raw_families]
        families = tuple(sorted(str(item) for item in raw_families if str(item)))
        if families:
            output[family_id] = families
    return output


def resolve_manifest_families_for_record(
    *,
    family_id: str,
    record: Any,
    source_manifest_family_map: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    mapped = tuple(sorted(source_manifest_family_map.get(family_id, ())))
    if mapped:
        return mapped
    return tuple(sorted(get_game_families(record)))


def _dict_to_record_proxy(future_profile: dict[str, float], graph_profile: dict[str, float]) -> Any:
    class Proxy:
        future_option_features = future_profile
        directional_features = graph_profile

    return Proxy()
