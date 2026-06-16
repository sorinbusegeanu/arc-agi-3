from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v6.concept_candidates_v10 import effect_temporal_profile
from v6.concept_candidates_v10fixb import (
    FamilyContext,
    _load_optional_json,
    apply_target_metrics,
    annotate_projection_outcomes,
    best_individual_role_score_raw,
    best_raw_m2_score,
    best_surface_raw_score,
    build_graph_edges,
    build_label_rows,
    build_raw_candidate_row,
    build_role_composition_rows,
    build_source_role_map,
    build_surface_comparison_rows,
    build_target_projection_mode_rows,
    canonical_role_fingerprint,
    future_option_behavior_features,
    generate_subcomposition_candidates,
    get_games,
    graph_position_features,
    local_graph_motif_features,
    mean_metric,
    merge_exact_candidates,
    predecessor_successor_profile,
    remap_concept_ids,
    role_graph_sort_key,
    score_concept_against_target_family,
    choose_effective_worker_count,
    detect_available_memory_bytes,
    estimate_pickle_size_bytes,
)
from v6.role_transfer_v09 import _write_parquet, cosine_similarity, mean_vector
from v6.role_transfer_v09c import RoleTransferV09cConfig, prepare_family_contexts


@dataclass(frozen=True)
class M4RoleConceptsV10eConfig:
    m3_input_dir: str = "runs/v6/v08d_cd2_extended32_sourceclean"
    transfer_input_dir: str = "runs/v6/v09c_transfer_hardened_extended32"
    m2_input_dir: str = "runs/v6/v07_cd2_extended32_expanded"
    m1_input_dir: str = "runs/v6/v06_cd2_extended32"
    output_dir: str = "runs/v6/v10e_role_based_m4_extended32"
    game_set_manifest: str | None = None
    game_set_name: str | None = None
    workers: int = 1
    min_games: int = 2
    min_manifest_families: int = 1
    min_role_count: int = 2
    max_role_count: int = 3


def run_m4_role_concepts_v10e(config: M4RoleConceptsV10eConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transfer_report = json.loads((Path(config.transfer_input_dir) / "v09c_report.json").read_text(encoding="utf-8"))
    transfer_rows = pd.read_parquet(Path(config.transfer_input_dir) / "v09c_hardened_assignments.parquet").to_dict(orient="records")
    transfer_by_heldout = defaultdict(list)
    for row in transfer_rows:
        transfer_by_heldout[str(row["heldout_family"])].append(row)

    contexts = prepare_family_contexts(
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
    tasks = [(context, transfer_by_heldout.get(context.heldout_family, []), config) for context in contexts]
    effective_workers = choose_effective_worker_count(tasks, requested_workers=config.workers, shared_state_bytes=0)
    if effective_workers <= 1 or len(tasks) <= 1:
        family_results = [_evaluate_role_based_family(*task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            family_results = list(executor.map(_evaluate_role_based_family_task, tasks, chunksize=1))
    family_results = sorted(family_results, key=lambda item: item["heldout_family"])

    raw_candidate_rows = [row for item in family_results for row in item["raw_candidate_rows"]]
    fallback_rows = [row for item in family_results for row in item["fallback_rows"]]
    transfer_score_rows = [row for item in family_results for row in item["transfer_rows"]]
    by_family_rows = [item["summary"] for item in family_results]

    merged_concepts, merge_diag = merge_exact_candidates(
        raw_candidate_rows,
        min_games=config.min_games,
        min_manifest_families=config.min_manifest_families,
        min_role_count=config.min_role_count,
    )
    concept_rows = apply_target_metrics(merged_concepts, transfer_score_rows)
    concept_rows = annotate_projection_outcomes(concept_rows, transfer_score_rows)
    concept_rows = apply_role_based_gates(concept_rows, transfer_score_rows)
    transferable_concepts = [row for row in concept_rows if is_transferable_role_based_concept(row)]

    payload = build_v10e_report(
        config=config,
        transfer_report=transfer_report,
        concept_rows=concept_rows,
        transfer_rows=transfer_score_rows,
        fallback_rows=fallback_rows,
        by_family_rows=by_family_rows,
        merge_diag=merge_diag,
        effective_workers=effective_workers,
    )

    _write_parquet(output_dir / "role_based_m4_concepts.parquet", concept_rows)
    _write_parquet(output_dir / "role_based_concept_transfer_scores.parquet", transfer_score_rows)
    _write_parquet(output_dir / "fallback_diagnostic_candidates.parquet", fallback_rows)
    (output_dir / "v10e_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "v10e_report.txt").write_text(format_v10e_report(payload), encoding="utf-8")
    return payload


def _evaluate_role_based_family_task(task: tuple[FamilyContext, list[dict[str, Any]], M4RoleConceptsV10eConfig]) -> dict[str, Any]:
    return _evaluate_role_based_family(*task)


def _evaluate_role_based_family(
    context: FamilyContext,
    target_rows: list[dict[str, Any]],
    config: M4RoleConceptsV10eConfig,
) -> dict[str, Any]:
    source_role_map = build_source_role_map(context.source_roles)
    stable_items = build_stable_role_items(context)
    fallback_rows = build_fallback_diagnostic_rows(context, source_role_map)
    raw_candidate_rows = generate_role_based_candidates(context, stable_items, max_role_count=config.max_role_count)

    local_exact_rows, _ = merge_exact_candidates(
        raw_candidate_rows,
        min_games=1,
        min_manifest_families=1,
        min_role_count=config.min_role_count,
    )
    transfer_rows = []
    for concept in local_exact_rows:
        projection = evaluate_role_based_projection_by_family(concept, context, target_rows)
        transfer_rows.append({"heldout_family": context.heldout_family, **projection})

    summary = {
        "heldout_family": context.heldout_family,
        "source_only_concept_discovery": True,
        "role_based_candidate_count": len(local_exact_rows),
        "fallback_candidate_count": len(fallback_rows),
        "positive_concept_lift": int(any(row["projection_used"] and row["target_mean_concept_lift_vs_role_raw"] > 0 and row["target_mean_concept_lift_vs_role_bag"] > 0 for row in transfer_rows)),
    }
    return {
        "heldout_family": context.heldout_family,
        "raw_candidate_rows": raw_candidate_rows,
        "fallback_rows": fallback_rows,
        "transfer_rows": transfer_rows,
        "summary": summary,
    }


def build_stable_role_items(context: FamilyContext) -> list[dict[str, Any]]:
    items = []
    for role_id, role_info in sorted(context.source_roles.items()):
        records = [
            context.source_neighborhoods[family_id]
            for family_id in role_info.get("member_family_ids", ())
            if family_id in context.source_neighborhoods
        ]
        if not records:
            continue
        record = sorted(records, key=lambda item: str(item.family_id))[0]
        fingerprint = canonical_role_fingerprint(role_info["role_label_candidate"], record)
        items.append(
            {
                "source_fold": context.heldout_family,
                "family_id": str(record.family_id),
                "record": record,
                "role_id": role_id,
                "role_label": role_info["role_label_candidate"],
                "canonical_role_fingerprint_hash": fingerprint["canonical_role_fingerprint_hash"],
                "canonical_role_signature_json": fingerprint["canonical_role_signature_json"],
                "canonical_role_label_or_family": fingerprint["canonical_role_label_or_family"],
                "canonical_role_similarity_vector": fingerprint["canonical_role_similarity_vector"],
                "unknown_role_flag": False,
                "source_manifest_families": sorted({family for member in records for family in getattr(member, "game_family_ids", ())}) or ["source_clean_roles"],
            }
        )
    return sorted(items, key=role_graph_sort_key)


def build_fallback_diagnostic_rows(context: FamilyContext, source_role_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for family_id, record in sorted(context.source_neighborhoods.items()):
        if family_id in source_role_map:
            continue
        fingerprint = canonical_role_fingerprint("unknown_role_candidate", record)
        rows.append(
            {
                "heldout_family": context.heldout_family,
                "family_id": family_id,
                "games_present": list(get_games(record)),
                "role_label_candidate": "unknown_role_candidate",
                "canonical_role_fingerprint_hash": fingerprint["canonical_role_fingerprint_hash"],
                "candidate_source": "fallback_diagnostic_only",
            }
        )
    return rows


def generate_role_based_candidates(context: FamilyContext, stable_items: list[dict[str, Any]], *, max_role_count: int) -> list[dict[str, Any]]:
    if len(stable_items) < 2:
        return []
    output = []
    for size in range(2, min(len(stable_items), max_role_count) + 1):
        for index, combo in enumerate(combinations(stable_items, size), start=1):
            row = build_role_based_candidate_row(
                source_fold=context.heldout_family,
                heldout_family=context.heldout_family,
                local_candidate_id=f"roles-{size}-{index:04d}",
                items=list(combo),
            )
            output.append(row)
    return sorted(output, key=lambda row: (row["concept_id"], row["local_candidate_id"]))


def build_role_based_candidate_row(
    *,
    source_fold: str,
    heldout_family: str,
    local_candidate_id: str,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    row = build_raw_candidate_row(
        source_fold=source_fold,
        heldout_family=heldout_family,
        manifest_family="role_based",
        local_candidate_id=local_candidate_id,
        generator_type="stable_role_composition",
        items=items,
    )
    row["candidate_source"] = "stable_role"
    row["source_manifest_families_present"] = sorted(
        {family for item in items for family in item.get("source_manifest_families", [])}
    )
    row["compression_gain"] = compression_gain(row)
    return row


def evaluate_role_based_projection_by_family(
    concept: dict[str, Any],
    context: FamilyContext,
    target_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    per_target = []
    target_rows_by_family = defaultdict(list)
    for row in target_rows:
        target_rows_by_family[str(row.get("target_family_id", ""))].append(row)
    for family in sorted(context.target_families, key=lambda item: item.family_id):
        target_record = context.full_neighborhoods.get(family.family_id)
        if target_record is None:
            continue
        family_rows = target_rows_by_family.get(family.family_id, [])
        per_target.append(score_role_based_concept_against_target_family(concept, context, family.family_id, target_record, family_rows))
    if not per_target:
        return {
            "concept_id": concept["concept_id"],
            "projection_used": False,
            "failure_reason": "missing_target_rows",
            "target_family_count": 0,
            "target_concept_prediction_score": 0.0,
            "target_mean_concept_lift_vs_role_raw": 0.0,
            "target_mean_concept_lift_vs_role_bag": 0.0,
            "target_mean_concept_lift_vs_m2": 0.0,
            "target_mean_concept_lift_vs_surface_raw": 0.0,
            "target_mean_concept_lift_vs_graph_no_label": 0.0,
            "target_mean_future_option_prediction_lift": 0.0,
            "mean_compression_gain": float(concept.get("compression_gain", 0.0)),
        }
    target_scores = [row["target_family_score"] for row in per_target]
    return {
        "concept_id": concept["concept_id"],
        "projection_used": True,
        "failure_reason": "",
        "target_family_count": len(per_target),
        "target_concept_prediction_score": float(np.mean(target_scores)),
        "target_mean_concept_lift_vs_role_raw": float(np.mean([row["target_family_score"] - row["best_individual_role_baseline_raw"] for row in per_target])),
        "target_mean_concept_lift_vs_role_bag": float(np.mean([row["target_family_score"] - row["unordered_role_bag_baseline"] for row in per_target])),
        "target_mean_concept_lift_vs_m2": float(np.mean([row["target_family_score"] - row["best_raw_m2_baseline"] for row in per_target])),
        "target_mean_concept_lift_vs_surface_raw": float(np.mean([row["target_family_score"] - row["best_surface_raw_baseline"] for row in per_target])),
        "target_mean_concept_lift_vs_graph_no_label": float(np.mean([row["target_family_score"] - row["graph_no_label_baseline"] for row in per_target])),
        "target_mean_future_option_prediction_lift": float(np.mean([row["future_option_prediction_lift"] for row in per_target])),
        "mean_compression_gain": float(concept.get("compression_gain", 0.0)),
        "best_individual_role_baseline_raw": float(np.mean([row["best_individual_role_baseline_raw"] for row in per_target])),
        "unordered_role_bag_baseline": float(np.mean([row["unordered_role_bag_baseline"] for row in per_target])),
        "best_raw_m2_baseline": float(np.mean([row["best_raw_m2_baseline"] for row in per_target])),
        "best_surface_raw_baseline": float(np.mean([row["best_surface_raw_baseline"] for row in per_target])),
        "graph_no_label_baseline": float(np.mean([row["graph_no_label_baseline"] for row in per_target])),
    }


def score_role_based_concept_against_target_family(
    concept: dict[str, Any],
    context: FamilyContext,
    target_family_id: str,
    target_record: Any,
    target_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    base = score_concept_against_target_family(concept, context, target_family_id, target_record, target_rows)
    target_future = future_option_behavior_features(target_record)
    target_graph = graph_position_features(target_record)
    unordered_role_bag_baseline = unordered_role_bag_score(concept, target_record)
    graph_no_label_baseline = best_graph_no_label_score(context, target_record)
    best_future_role = best_future_option_role_score(context, target_record)
    future_option_prediction_lift = cosine_similarity(concept["future_option_delta_profile"], target_future) - best_future_role
    return {
        **base,
        "unordered_role_bag_baseline": unordered_role_bag_baseline,
        "graph_no_label_baseline": graph_no_label_baseline,
        "future_option_prediction_lift": float(future_option_prediction_lift),
        "target_future_similarity": cosine_similarity(concept["future_option_delta_profile"], target_future),
        "target_graph_similarity": cosine_similarity(concept["graph_position_profile"], target_graph),
    }


def unordered_role_bag_score(concept: dict[str, Any], target_record: Any) -> float:
    target_future = future_option_behavior_features(target_record)
    target_graph = graph_position_features(target_record)
    return float(
        0.5 * cosine_similarity(concept["future_option_delta_profile"], target_future)
        + 0.5 * cosine_similarity(concept["graph_position_profile"], target_graph)
    )


def best_graph_no_label_score(context: FamilyContext, target_record: Any) -> float:
    target_graph = graph_position_features(target_record)
    best = 0.0
    for record in context.full_no_label_neighborhoods.values():
        best = max(best, float(cosine_similarity(target_graph, graph_position_features(record))))
    return best


def best_future_option_role_score(context: FamilyContext, target_record: Any) -> float:
    target_future = future_option_behavior_features(target_record)
    best = 0.0
    for role in context.source_roles.values():
        future = {key.removeprefix("future:"): float(value) for key, value in role["all_features"].items() if str(key).startswith("future:")}
        best = max(best, float(cosine_similarity(target_future, future)))
    return best


def compression_gain(concept: dict[str, Any]) -> float:
    role_count = max(1, len(concept.get("canonical_role_fingerprint_hashes", [])))
    return float(concept.get("source_support_count", 0) / role_count - 1.0)


def apply_role_based_gates(concept_rows: list[dict[str, Any]], transfer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in transfer_rows:
        if row.get("projection_used"):
            grouped[row["concept_id"]].append(row)
    output = []
    for row in concept_rows:
        projections = grouped.get(row["concept_id"], [])
        updated = dict(row)
        updated["target_mean_concept_lift_vs_role_bag"] = mean_metric(projections, "target_mean_concept_lift_vs_role_bag")
        updated["target_mean_concept_lift_vs_graph_no_label"] = mean_metric(projections, "target_mean_concept_lift_vs_graph_no_label")
        updated["target_mean_future_option_prediction_lift"] = mean_metric(projections, "target_mean_future_option_prediction_lift")
        updated["mean_compression_gain"] = mean_metric(projections, "mean_compression_gain") or compression_gain(updated)
        updated["passes_role_based_gate"] = is_transferable_role_based_concept(updated)
        output.append(updated)
    return output


def is_transferable_role_based_concept(row: dict[str, Any]) -> bool:
    return (
        len(row.get("canonical_role_fingerprint_hashes", [])) >= 2
        and float(row.get("target_mean_concept_lift_vs_role_raw", 0.0)) > 0.0
        and float(row.get("target_mean_concept_lift_vs_role_bag", 0.0)) > 0.0
        and float(row.get("mean_compression_gain", 0.0)) > 0.0
        and float(row.get("target_mean_future_option_prediction_lift", 0.0)) > 0.0
    )


def build_v10e_report(
    *,
    config: M4RoleConceptsV10eConfig,
    transfer_report: dict[str, Any],
    concept_rows: list[dict[str, Any]],
    transfer_rows: list[dict[str, Any]],
    fallback_rows: list[dict[str, Any]],
    by_family_rows: list[dict[str, Any]],
    merge_diag: dict[str, int],
    effective_workers: int,
) -> dict[str, Any]:
    transferable = [row for row in concept_rows if is_transferable_role_based_concept(row)]
    positive_families = sum(1 for row in by_family_rows if row["positive_concept_lift"])
    role_lift = mean_metric(transferable, "target_mean_concept_lift_vs_role_raw")
    bag_lift = mean_metric(transferable, "target_mean_concept_lift_vs_role_bag")
    conclusion = "m4_role_based_not_established"
    if len(transferable) >= 3 and role_lift >= 0.10 and bag_lift >= 0.05:
        conclusion = "m4_role_based_very_strong"
    elif len(transferable) >= 2 and role_lift >= 0.05 and bag_lift >= 0.03:
        conclusion = "m4_role_based_strong"
    elif len(transferable) >= 1 and role_lift > 0.0 and bag_lift > 0.0:
        conclusion = "m4_role_based_weak"
    report = {
        "transfer_report_summary": transfer_report.get("report", {}),
        "role_based_candidate_count": len(concept_rows),
        "transferable_role_based_concepts": len(transferable),
        "fallback_diagnostic_candidate_count": len(fallback_rows),
        "target_mean_concept_lift_vs_role_raw": role_lift,
        "target_mean_concept_lift_vs_role_bag": bag_lift,
        "target_mean_concept_lift_vs_m2": mean_metric(transferable, "target_mean_concept_lift_vs_m2"),
        "target_mean_concept_lift_vs_surface_raw": mean_metric(transferable, "target_mean_concept_lift_vs_surface_raw"),
        "target_mean_concept_lift_vs_graph_no_label": mean_metric(transferable, "target_mean_concept_lift_vs_graph_no_label"),
        "target_mean_future_option_prediction_lift": mean_metric(transferable, "target_mean_future_option_prediction_lift"),
        "mean_compression_gain": mean_metric(transferable, "mean_compression_gain"),
        "positive_lift_families": positive_families,
        "scientific_conclusion": conclusion,
        "v10a_can_proceed": conclusion in {"m4_role_based_weak", "m4_role_based_strong", "m4_role_based_very_strong"},
        "merge_diag": merge_diag,
        "workers_used": effective_workers,
    }
    return {
        "config": {
            "output_dir": config.output_dir,
            "workers": config.workers,
        },
        "report": report,
        "validation": {
            "scientific_conclusion": conclusion,
            "proceed_to_v10a": report["v10a_can_proceed"],
        },
    }


def format_v10e_report(payload: dict[str, Any]) -> str:
    report = payload["report"]
    return "\n".join(
        [
            "ARC-AGI3 v0.10e: role-based M4 validation",
            "",
            f"scientific_conclusion={report['scientific_conclusion']}",
            f"role_based_candidate_count={report['role_based_candidate_count']}",
            f"transferable_role_based_concepts={report['transferable_role_based_concepts']}",
            f"fallback_diagnostic_candidate_count={report['fallback_diagnostic_candidate_count']}",
            f"target_mean_concept_lift_vs_role_raw={report['target_mean_concept_lift_vs_role_raw']}",
            f"target_mean_concept_lift_vs_role_bag={report['target_mean_concept_lift_vs_role_bag']}",
            f"target_mean_concept_lift_vs_m2={report['target_mean_concept_lift_vs_m2']}",
            f"target_mean_concept_lift_vs_surface_raw={report['target_mean_concept_lift_vs_surface_raw']}",
            f"target_mean_concept_lift_vs_graph_no_label={report['target_mean_concept_lift_vs_graph_no_label']}",
            f"target_mean_future_option_prediction_lift={report['target_mean_future_option_prediction_lift']}",
            f"mean_compression_gain={report['mean_compression_gain']}",
            f"positive_lift_families={report['positive_lift_families']}",
            f"v10a_can_proceed={report['v10a_can_proceed']}",
        ]
    )
