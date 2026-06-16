from __future__ import annotations

import gc
import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v6.concept_candidates_v10 import effect_temporal_profile
from v6.concept_candidates_v10fixb import (
    _load_optional_json,
    annotate_projection_outcomes,
    apply_target_metrics,
    best_individual_role_score_raw,
    best_raw_m2_score,
    best_surface_raw_score,
    build_raw_candidate_row,
    build_source_role_map,
    canonical_role_fingerprint,
    future_option_behavior_features,
    get_games,
    graph_position_features,
    local_graph_motif_features,
    mean_metric,
    merge_exact_candidates,
    predecessor_successor_profile,
    score_concept_against_target_family,
)
from v6.role_transfer_v09 import _write_parquet, cosine_similarity, mean_vector
from v6.role_transfer_v09c import (
    RoleTransferV09cConfig,
    SingleFamilyContext,
    build_single_family_context,
    list_heldout_families,
)


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
        transfer_by_heldout[str(row.get("heldout_family", ""))].append(row)

    role_config = RoleTransferV09cConfig(
        m2_input_dir=config.m2_input_dir,
        m1_input_dir=config.m1_input_dir,
        previous_v09b_dir="runs/v6/v09b_role_transfer_refined_sourceclean_extended32",
        output_dir=config.output_dir,
        game_set_manifest=config.game_set_manifest,
        game_set_name=config.game_set_name,
        workers=1,
    )
    heldout_families = list_heldout_families(role_config)

    family_results = []
    for heldout_family in heldout_families:
        context = build_single_family_context(role_config, heldout_family)
        family_results.append(
            _evaluate_role_based_family(
                context,
                transfer_by_heldout.get(heldout_family, []),
                config,
            )
        )
        del context
        gc.collect()

    family_results = sorted(family_results, key=lambda item: item["heldout_family"])
    raw_candidate_rows = [row for item in family_results for row in item["raw_candidate_rows"]]
    fallback_rows = [row for item in family_results for row in item["fallback_rows"]]
    transfer_score_rows = [row for item in family_results for row in item["transfer_rows"]]
    target_family_rows = [row for item in family_results for row in item["target_family_rows"]]
    fallback_score_rows = [row for item in family_results for row in item["fallback_score_rows"]]
    by_family_rows = [item["summary"] for item in family_results]

    exact_rows, merge_diag = merge_exact_candidates(
        raw_candidate_rows,
        min_games=config.min_games,
        min_manifest_families=config.min_manifest_families,
        min_role_count=config.min_role_count,
    )
    concept_rows = apply_target_metrics(exact_rows, transfer_score_rows)
    concept_rows = annotate_projection_outcomes(concept_rows, transfer_score_rows)
    concept_rows = apply_role_based_gates(concept_rows, transfer_score_rows)
    concept_rows = sorted(concept_rows, key=lambda row: row["concept_id"])

    fallback_only_signal_detected = any(
        float(row.get("target_family_score", 0.0)) > float(row.get("best_individual_role_baseline_raw", 0.0))
        and float(row.get("target_family_score", 0.0)) > float(row.get("best_surface_raw_baseline", 0.0))
        for row in fallback_score_rows
    )
    transferable = [
        row
        for row in concept_rows
        if row.get("candidate_source") == "stable_role" and is_transferable_role_based_concept(row)
    ]
    rejected_rows = build_rejected_candidate_rows(concept_rows)
    identity_rows = build_concept_identity_rows(raw_candidate_rows)
    compression_rows = build_compression_rows(concept_rows)
    future_rows = build_future_option_rows(target_family_rows)
    baseline_rows = build_baseline_rows(target_family_rows)

    payload = build_v10e_report(
        config=config,
        transfer_report=transfer_report,
        concept_rows=concept_rows,
        transferable_rows=transferable,
        transfer_rows=transfer_score_rows,
        fallback_rows=fallback_rows,
        fallback_score_rows=fallback_score_rows,
        by_family_rows=by_family_rows,
        merge_diag=merge_diag,
        fallback_only_signal_detected=fallback_only_signal_detected,
    )

    _write_parquet(output_dir / "role_based_m4_concepts.parquet", concept_rows)
    (output_dir / "role_based_m4_concepts.json").write_text(json.dumps(concept_rows, indent=2), encoding="utf-8")
    _write_parquet(output_dir / "role_based_concept_transfer_scores.parquet", transfer_score_rows)
    _write_parquet(output_dir / "role_based_concept_by_family.parquet", by_family_rows)
    _write_parquet(output_dir / "role_based_concept_compression_scores.parquet", compression_rows)
    _write_parquet(output_dir / "role_based_future_option_prediction_scores.parquet", future_rows)
    _write_parquet(output_dir / "role_based_baseline_comparison.parquet", baseline_rows)
    _write_parquet(output_dir / "fallback_diagnostic_candidates.parquet", fallback_rows)
    _write_parquet(output_dir / "fallback_diagnostic_scores.parquet", fallback_score_rows)
    _write_parquet(output_dir / "rejected_candidate_diagnostics.parquet", rejected_rows)
    _write_parquet(output_dir / "concept_identity_diagnostics.parquet", identity_rows)
    (output_dir / "v10e_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "v10e_report.txt").write_text(format_v10e_report(payload), encoding="utf-8")
    return payload


def _evaluate_role_based_family(
    context: SingleFamilyContext,
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
    transfer_rows: list[dict[str, Any]] = []
    target_family_rows: list[dict[str, Any]] = []
    for concept in local_exact_rows:
        projection, per_target_rows = evaluate_role_based_projection_by_family(concept, context, target_rows)
        transfer_rows.append({"heldout_family": context.heldout_family, "candidate_source": "stable_role", **projection})
        target_family_rows.extend(
            {
                "heldout_family": context.heldout_family,
                "candidate_source": "stable_role",
                "concept_id": concept["concept_id"],
                **row,
            }
            for row in per_target_rows
        )

    fallback_score_rows = score_fallback_rows(context, fallback_rows, target_rows)
    summary = {
        "heldout_family": context.heldout_family,
        "source_only_concept_discovery": True,
        "role_based_candidate_count": len(local_exact_rows),
        "fallback_candidate_count": len(fallback_rows),
        "positive_lift_families": sum(
            1
            for row in transfer_rows
            if row.get("projection_used")
            and row.get("passes_role_based_gate")
            and float(row.get("target_mean_concept_lift_vs_role_raw", 0.0)) > 0.0
            and float(row.get("target_mean_concept_lift_vs_role_bag", 0.0)) > 0.0
            and float(row.get("target_mean_concept_lift_vs_surface_raw", 0.0)) > 0.0
        ),
        "fallback_only_signal_detected": any(
            float(row.get("target_family_score", 0.0)) > float(row.get("best_individual_role_baseline_raw", 0.0))
            and float(row.get("target_family_score", 0.0)) > float(row.get("best_surface_raw_baseline", 0.0))
            for row in fallback_score_rows
        ),
    }
    return {
        "heldout_family": context.heldout_family,
        "raw_candidate_rows": raw_candidate_rows,
        "fallback_rows": fallback_rows,
        "transfer_rows": transfer_rows,
        "target_family_rows": target_family_rows,
        "fallback_score_rows": fallback_score_rows,
        "summary": summary,
    }


def build_stable_role_items(context: Any) -> list[dict[str, Any]]:
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
                "source_manifest_families": sorted({family for member in records for family in getattr(member, "game_family_ids", ())})
                or ["source_clean_roles"],
            }
        )
    return sorted(items, key=lambda item: (item["role_id"], item["family_id"]))


def build_fallback_diagnostic_rows(context: Any, source_role_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for family_id, record in sorted(context.source_neighborhoods.items()):
        if family_id in source_role_map:
            continue
        fingerprint = canonical_role_fingerprint("unknown_role_candidate", record)
        row = build_raw_candidate_row(
            source_fold=context.heldout_family,
            heldout_family=context.heldout_family,
            manifest_family="fallback_diagnostic",
            local_candidate_id=f"fallback-{family_id}",
            generator_type="fallback_diagnostic_only",
            items=[
                {
                    "source_fold": context.heldout_family,
                    "family_id": family_id,
                    "record": record,
                    "role_id": family_id,
                    "role_label": "unknown_role_candidate",
                    "canonical_role_fingerprint_hash": fingerprint["canonical_role_fingerprint_hash"],
                    "canonical_role_signature_json": fingerprint["canonical_role_signature_json"],
                    "canonical_role_label_or_family": fingerprint["canonical_role_label_or_family"],
                    "canonical_role_similarity_vector": fingerprint["canonical_role_similarity_vector"],
                    "source_manifest_families": sorted(getattr(record, "game_family_ids", ()) or ("unknown_manifest_family",)),
                }
            ],
        )
        row["candidate_source"] = "fallback_diagnostic_only"
        rows.append(row)
    return rows


def generate_role_based_candidates(context: Any, stable_items: list[dict[str, Any]], *, max_role_count: int) -> list[dict[str, Any]]:
    if len(stable_items) < 2:
        return []
    output = []
    for size in range(2, min(len(stable_items), max_role_count) + 1):
        for index, combo in enumerate(combinations(stable_items, size), start=1):
            output.append(
                build_role_based_candidate_row(
                    source_fold=context.heldout_family,
                    heldout_family=context.heldout_family,
                    local_candidate_id=f"roles-{size}-{index:04d}",
                    items=list(combo),
                )
            )
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
    row["source_manifest_families_present"] = sorted({family for item in items for family in item.get("source_manifest_families", [])})
    return row


def evaluate_role_based_projection_by_family(
    concept: dict[str, Any],
    context: Any,
    target_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_target_rows = []
    target_rows_by_family = defaultdict(list)
    for row in target_rows:
        target_rows_by_family[str(row.get("target_family_id", ""))].append(row)

    target_records = _target_records(context)
    for family in sorted(context.target_families, key=lambda item: item.family_id):
        target_record = target_records.get(family.family_id)
        if target_record is None:
            continue
        family_rows = target_rows_by_family.get(family.family_id, [])
        per_target_rows.append(
            score_role_based_concept_against_target_family(concept, context, family.family_id, target_record, family_rows)
        )

    if not per_target_rows:
        return (
            {
                "concept_id": concept["concept_id"],
                "projection_used": False,
                "failure_reason": "missing_target_rows",
                "target_family_count": 0,
                "target_concept_prediction_score": 0.0,
                "target_mean_concept_lift_vs_role_raw": 0.0,
                "target_mean_concept_lift_vs_role_bag": 0.0,
                "target_mean_concept_lift_vs_surface_raw": 0.0,
                "target_mean_concept_lift_vs_graph_no_label": 0.0,
                "target_mean_concept_lift_vs_m2": 0.0,
                "target_mean_future_option_prediction_lift": 0.0,
                "mean_compression_gain": 0.0,
                "explained_m2_family_count": 0,
                "positive_lift_family_count": 0,
                "passes_role_based_gate": False,
            },
            [],
        )

    target_scores = [row["target_family_score"] for row in per_target_rows]
    mean_score = float(np.mean(target_scores))
    projection = {
        "concept_id": concept["concept_id"],
        "projection_used": True,
        "failure_reason": "",
        "target_family_count": len(per_target_rows),
        "target_concept_prediction_score": mean_score,
        "target_mean_concept_lift_vs_role_raw": float(np.mean([row["target_family_score"] - row["best_individual_role_baseline_raw"] for row in per_target_rows])),
        "target_mean_concept_lift_vs_role_bag": float(np.mean([row["target_family_score"] - row["unordered_role_bag_baseline"] for row in per_target_rows])),
        "target_mean_concept_lift_vs_surface_raw": float(np.mean([row["target_family_score"] - row["best_surface_raw_baseline"] for row in per_target_rows])),
        "target_mean_concept_lift_vs_graph_no_label": float(np.mean([row["target_family_score"] - row["graph_no_label_baseline"] for row in per_target_rows])),
        "target_mean_concept_lift_vs_m2": float(np.mean([row["target_family_score"] - row["best_raw_m2_baseline"] for row in per_target_rows])),
        "target_mean_future_option_prediction_lift": float(np.mean([row["future_option_prediction_lift"] for row in per_target_rows])),
        "mean_compression_gain": float(np.mean([row["compression_gain"] for row in per_target_rows])),
        "explained_m2_family_count": sum(1 for row in per_target_rows if row["compression_gain"] > 0.0 and row["target_family_score"] > row["best_raw_m2_baseline"]),
        "positive_lift_family_count": sum(
            1
            for row in per_target_rows
            if row["target_family_score"] > row["best_individual_role_baseline_raw"]
            and row["target_family_score"] > row["unordered_role_bag_baseline"]
            and row["target_family_score"] > row["best_surface_raw_baseline"]
            and row["compression_gain"] > 0.0
            and row["future_option_prediction_lift"] > 0.0
        ),
        "passes_role_based_gate": False,
    }
    projection["passes_role_based_gate"] = is_transferable_role_based_concept({**concept, **projection, "candidate_source": concept.get("candidate_source", "stable_role")})
    return projection, per_target_rows


def score_role_based_concept_against_target_family(
    concept: dict[str, Any],
    context: Any,
    target_family_id: str,
    target_record: Any,
    target_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    base_context = _base_scoring_context(context)
    base = score_concept_against_target_family(concept, base_context, target_family_id, target_record, target_rows)
    target_future = future_option_behavior_features(target_record)
    target_graph = graph_position_features(target_record)
    concept_future_similarity = cosine_similarity(concept["future_option_delta_profile"], target_future)
    future_role_score = best_future_option_role_score(context, target_record)
    unordered_role_bag_baseline = unordered_role_bag_score(concept, context, target_record)
    graph_no_label_baseline = best_graph_no_label_score(context, target_record)
    separate_role_score = separate_role_explanation_score(concept, context, target_record)
    compression = float(base["target_family_score"] - separate_role_score)
    return {
        **base,
        "unordered_role_bag_baseline": unordered_role_bag_baseline,
        "graph_no_label_baseline": graph_no_label_baseline,
        "future_option_prediction_lift": float(concept_future_similarity - future_role_score),
        "best_future_option_role_score": float(future_role_score),
        "target_future_similarity": float(concept_future_similarity),
        "target_graph_similarity": float(cosine_similarity(concept["graph_position_profile"], target_graph)),
        "separate_role_explanation_score": float(separate_role_score),
        "composed_concept_explanation_score": float(base["target_family_score"]),
        "compression_gain": compression,
        "target_role_overlap_diagnostic": float(
            len(set(concept.get("fold_local_role_ids", ())) & {str(row.get("assigned_role_id", "")) for row in target_rows if row.get("assigned_role_id")})
            / max(1, len(set(concept.get("fold_local_role_ids", ()))))
        ),
    }


def unordered_role_bag_score(concept: dict[str, Any], context: Any, target_record: Any) -> float:
    target_future = future_option_behavior_features(target_record)
    target_graph = graph_position_features(target_record)
    role_futures = []
    role_graphs = []
    for role_id in concept.get("fold_local_role_ids", ()):
        role = context.source_roles.get(role_id)
        if role is None:
            continue
        role_futures.append(_subset_prefixed(role.get("all_features", {}), "future:"))
        role_graphs.append(_subset_prefixed(role.get("all_features", {}), "directional:"))
    if not role_futures:
        return 0.0
    future_bag = mean_vector(role_futures)
    graph_bag = mean_vector(role_graphs) if role_graphs else {}
    return float(0.6 * cosine_similarity(target_future, future_bag) + 0.4 * cosine_similarity(target_graph, graph_bag))


def best_graph_no_label_score(context: Any, target_record: Any) -> float:
    target_graph = graph_position_features(target_record)
    best = 0.0
    for record in context.source_neighborhoods.values():
        best = max(best, float(cosine_similarity(target_graph, graph_position_features(record))))
    return best


def best_future_option_role_score(context: Any, target_record: Any) -> float:
    target_future = future_option_behavior_features(target_record)
    best = 0.0
    for role in context.source_roles.values():
        future = _subset_prefixed(role.get("all_features", {}), "future:")
        best = max(best, float(cosine_similarity(target_future, future)))
    return best


def separate_role_explanation_score(concept: dict[str, Any], context: Any, target_record: Any) -> float:
    target_future = future_option_behavior_features(target_record)
    target_graph = graph_position_features(target_record)
    scores = []
    for role_id in concept.get("fold_local_role_ids", ()):
        role = context.source_roles.get(role_id)
        if role is None:
            continue
        role_future = _subset_prefixed(role.get("all_features", {}), "future:")
        role_graph = _subset_prefixed(role.get("all_features", {}), "directional:")
        scores.append(0.6 * cosine_similarity(target_future, role_future) + 0.4 * cosine_similarity(target_graph, role_graph))
    return float(np.mean(scores)) if scores else 0.0


def apply_role_based_gates(concept_rows: list[dict[str, Any]], transfer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in transfer_rows:
        if row.get("projection_used"):
            grouped[row["concept_id"]].append(row)
    output = []
    for row in concept_rows:
        projections = grouped.get(row["concept_id"], [])
        updated = dict(row)
        updated["candidate_source"] = updated.get("candidate_source", "stable_role")
        updated["target_mean_concept_lift_vs_role_bag"] = mean_metric(projections, "target_mean_concept_lift_vs_role_bag")
        updated["target_mean_concept_lift_vs_graph_no_label"] = mean_metric(projections, "target_mean_concept_lift_vs_graph_no_label")
        updated["target_mean_future_option_prediction_lift"] = mean_metric(projections, "target_mean_future_option_prediction_lift")
        updated["mean_compression_gain"] = mean_metric(projections, "mean_compression_gain")
        updated["explained_m2_family_count"] = int(sum(int(row.get("explained_m2_family_count", 0)) for row in projections))
        updated["passes_role_based_gate"] = is_transferable_role_based_concept(updated)
        output.append(updated)
    return output


def is_transferable_role_based_concept(row: dict[str, Any]) -> bool:
    if row.get("candidate_source", "stable_role") != "stable_role":
        return False
    return (
        len(row.get("canonical_role_fingerprint_hashes", [])) >= 2
        and float(row.get("target_mean_concept_lift_vs_role_raw", 0.0)) > 0.0
        and float(row.get("target_mean_concept_lift_vs_role_bag", 0.0)) > 0.0
        and float(row.get("target_mean_concept_lift_vs_surface_raw", 0.0)) > 0.0
        and float(row.get("mean_compression_gain", 0.0)) > 0.0
        and float(row.get("target_mean_future_option_prediction_lift", 0.0)) > 0.0
        and int(row.get("positive_lift_family_count", 0)) >= 1
        and int(row.get("explained_m2_family_count", 0)) >= 2
    )


def score_fallback_rows(
    context: Any,
    fallback_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_rows_by_family = defaultdict(list)
    for row in target_rows:
        target_rows_by_family[str(row.get("target_family_id", ""))].append(row)
    rows = []
    target_records = _target_records(context)
    for candidate in fallback_rows:
        for family in sorted(context.target_families, key=lambda item: item.family_id):
            target_record = target_records.get(family.family_id)
            if target_record is None:
                continue
            scored = score_role_based_concept_against_target_family(
                candidate,
                context,
                family.family_id,
                target_record,
                target_rows_by_family.get(family.family_id, []),
            )
            rows.append(
                {
                    "heldout_family": context.heldout_family,
                    "candidate_source": "fallback_diagnostic_only",
                    "concept_id": candidate["concept_id"],
                    **scored,
                }
            )
    return rows


def build_rejected_candidate_rows(concept_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in concept_rows:
        reasons = []
        if row.get("candidate_source") != "stable_role":
            reasons.append("diagnostic_only_candidate_source")
        if float(row.get("target_mean_concept_lift_vs_role_raw", 0.0)) <= 0.0:
            reasons.append("no_lift_vs_best_individual_role")
        if float(row.get("target_mean_concept_lift_vs_role_bag", 0.0)) <= 0.0:
            reasons.append("no_lift_vs_unordered_role_bag")
        if float(row.get("target_mean_concept_lift_vs_surface_raw", 0.0)) <= 0.0:
            reasons.append("no_lift_vs_surface_effect_raw")
        if float(row.get("mean_compression_gain", 0.0)) <= 0.0:
            reasons.append("no_positive_compression_gain")
        if float(row.get("target_mean_future_option_prediction_lift", 0.0)) <= 0.0:
            reasons.append("no_future_option_prediction_lift")
        if int(row.get("explained_m2_family_count", 0)) < 2:
            reasons.append("insufficient_explained_m2_families")
        output.append(
            {
                "concept_id": row["concept_id"],
                "candidate_source": row.get("candidate_source", "stable_role"),
                "passes_role_based_gate": bool(row.get("passes_role_based_gate")),
                "rejection_reasons": reasons,
            }
        )
    return output


def build_concept_identity_rows(raw_candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "concept_id": row["concept_id"],
            "concept_signature_json": row["concept_signature_json"],
            "local_candidate_id": row["local_candidate_id"],
            "candidate_source": row.get("candidate_source", ""),
            "source_fold": row["source_fold"],
            "heldout_family": row["heldout_family"],
            "canonical_role_fingerprint_hashes": row["canonical_role_fingerprint_hashes"],
            "source_manifest_families_present": row["source_manifest_families_present"],
        }
        for row in raw_candidate_rows
    ]


def build_compression_rows(concept_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "concept_id": row["concept_id"],
            "candidate_source": row.get("candidate_source", "stable_role"),
            "mean_compression_gain": float(row.get("mean_compression_gain", 0.0)),
            "explained_m2_family_count": int(row.get("explained_m2_family_count", 0)),
            "positive_lift_family_count": int(row.get("positive_lift_family_count", 0)),
        }
        for row in concept_rows
    ]


def build_future_option_rows(target_family_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "heldout_family": row["heldout_family"],
            "concept_id": row["concept_id"],
            "candidate_source": row.get("candidate_source", "stable_role"),
            "target_family_id": row["target_family_id"],
            "target_future_similarity": float(row.get("target_future_similarity", 0.0)),
            "best_future_option_role_score": float(row.get("best_future_option_role_score", 0.0)),
            "future_option_prediction_lift": float(row.get("future_option_prediction_lift", 0.0)),
        }
        for row in target_family_rows
    ]


def build_baseline_rows(target_family_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "heldout_family": row["heldout_family"],
            "concept_id": row["concept_id"],
            "candidate_source": row.get("candidate_source", "stable_role"),
            "target_family_id": row["target_family_id"],
            "target_family_score": float(row.get("target_family_score", 0.0)),
            "best_individual_role_baseline_raw": float(row.get("best_individual_role_baseline_raw", 0.0)),
            "unordered_role_bag_baseline": float(row.get("unordered_role_bag_baseline", 0.0)),
            "best_raw_m2_baseline": float(row.get("best_raw_m2_baseline", 0.0)),
            "best_surface_raw_baseline": float(row.get("best_surface_raw_baseline", 0.0)),
            "graph_no_label_baseline": float(row.get("graph_no_label_baseline", 0.0)),
            "compression_gain": float(row.get("compression_gain", 0.0)),
        }
        for row in target_family_rows
    ]


def build_v10e_report(
    *,
    config: M4RoleConceptsV10eConfig,
    transfer_report: dict[str, Any],
    concept_rows: list[dict[str, Any]],
    transferable_rows: list[dict[str, Any]],
    transfer_rows: list[dict[str, Any]],
    fallback_rows: list[dict[str, Any]],
    fallback_score_rows: list[dict[str, Any]],
    by_family_rows: list[dict[str, Any]],
    merge_diag: dict[str, int],
    fallback_only_signal_detected: bool,
) -> dict[str, Any]:
    stable_rows = [row for row in concept_rows if row.get("candidate_source") == "stable_role"]
    role_based_transferable = [row for row in transferable_rows if row.get("candidate_source") == "stable_role"]
    mean_lift_vs_role = mean_metric(role_based_transferable, "target_mean_concept_lift_vs_role_raw")
    mean_lift_vs_bag = mean_metric(role_based_transferable, "target_mean_concept_lift_vs_role_bag")
    mean_lift_vs_surface = mean_metric(role_based_transferable, "target_mean_concept_lift_vs_surface_raw")
    mean_lift_vs_graph = mean_metric(role_based_transferable, "target_mean_concept_lift_vs_graph_no_label")
    mean_lift_vs_m2 = mean_metric(role_based_transferable, "target_mean_concept_lift_vs_m2")
    mean_compression = mean_metric(role_based_transferable, "mean_compression_gain")
    future_lift = mean_metric(role_based_transferable, "target_mean_future_option_prediction_lift")
    positive_families = sum(1 for row in by_family_rows if int(row.get("positive_lift_families", 0)) > 0)
    target_family_score_count = sum(1 for row in transfer_rows if row.get("projection_used"))
    dominant_transfer_share = _dominant_transfer_share(role_based_transferable)

    conclusion = "m4_role_based_not_established"
    if not stable_rows and fallback_only_signal_detected:
        conclusion = "m4_fallback_signal_only_m3_bottleneck"
    elif not stable_rows:
        conclusion = "m4_role_based_pipeline_not_diagnostic"
    elif not role_based_transferable and fallback_only_signal_detected:
        conclusion = "m4_fallback_signal_only_m3_bottleneck"
    elif (
        len(role_based_transferable) >= 5
        and mean_lift_vs_role >= 0.10
        and mean_lift_vs_bag >= 0.10
        and mean_lift_vs_surface >= 0.10
        and mean_compression >= 0.10
        and future_lift > 0.0
        and positive_families >= 12
        and dominant_transfer_share <= 0.40
    ):
        conclusion = "m4_role_based_very_strong"
    elif (
        len(role_based_transferable) >= 3
        and mean_lift_vs_role >= 0.05
        and mean_lift_vs_bag >= 0.05
        and mean_lift_vs_surface >= 0.05
        and mean_compression >= 0.05
        and future_lift > 0.0
        and positive_families >= 8
    ):
        conclusion = "m4_role_based_strong"
    elif (
        len(role_based_transferable) >= 2
        and mean_lift_vs_role > 0.0
        and mean_lift_vs_bag > 0.0
        and mean_lift_vs_surface > 0.0
        and mean_compression > 0.0
        and future_lift > 0.0
        and positive_families >= 6
    ):
        conclusion = "m4_role_based_weak"

    report = {
        "transfer_report_summary": transfer_report.get("report", {}),
        "role_based_stable_concepts": len(stable_rows),
        "role_based_transferable_concepts": len(role_based_transferable),
        "transferable_role_based_concepts": len(role_based_transferable),
        "role_based_candidate_count": len(concept_rows),
        "fallback_diagnostic_candidate_count": len(fallback_rows),
        "fallback_diagnostic_score_count": len(fallback_score_rows),
        "target_family_score_count": target_family_score_count,
        "mean_lift_vs_best_individual_m3_role": mean_lift_vs_role,
        "mean_lift_vs_unordered_role_bag": mean_lift_vs_bag,
        "mean_lift_vs_surface_effect_raw": mean_lift_vs_surface,
        "mean_lift_vs_graph_no_label": mean_lift_vs_graph,
        "mean_lift_vs_raw_m2": mean_lift_vs_m2,
        "mean_compression_gain": mean_compression,
        "future_option_prediction_lift_vs_best_role": future_lift,
        "positive_lift_families": positive_families,
        "fallback_only_signal_detected": fallback_only_signal_detected,
        "scientific_conclusion": conclusion,
        "v10a_can_proceed": conclusion in {"m4_role_based_weak", "m4_role_based_strong", "m4_role_based_very_strong"},
        "source_clean_context_mode": "single_family_streaming",
        "target_role_overlap_diagnostic_only": True,
        "manifest_family_used_for_grouping": False,
        "graph_no_label_source_clean": True,
        "merge_diag": merge_diag,
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
            "ARC-AGI3 v0.10e-a: strict role-based M4 validation",
            "",
            f"scientific_conclusion={report['scientific_conclusion']}",
            f"role_based_stable_concepts={report['role_based_stable_concepts']}",
            f"role_based_transferable_concepts={report['role_based_transferable_concepts']}",
            f"fallback_diagnostic_candidate_count={report['fallback_diagnostic_candidate_count']}",
            f"fallback_diagnostic_score_count={report['fallback_diagnostic_score_count']}",
            f"target_family_score_count={report['target_family_score_count']}",
            f"mean_lift_vs_best_individual_m3_role={report['mean_lift_vs_best_individual_m3_role']}",
            f"mean_lift_vs_unordered_role_bag={report['mean_lift_vs_unordered_role_bag']}",
            f"mean_lift_vs_surface_effect_raw={report['mean_lift_vs_surface_effect_raw']}",
            f"mean_lift_vs_graph_no_label={report['mean_lift_vs_graph_no_label']}",
            f"mean_lift_vs_raw_m2={report['mean_lift_vs_raw_m2']}",
            f"mean_compression_gain={report['mean_compression_gain']}",
            f"future_option_prediction_lift_vs_best_role={report['future_option_prediction_lift_vs_best_role']}",
            f"positive_lift_families={report['positive_lift_families']}",
            f"fallback_only_signal_detected={report['fallback_only_signal_detected']}",
            f"v10a_can_proceed={report['v10a_can_proceed']}",
        ]
    )


def _base_scoring_context(context: Any) -> Any:
    if hasattr(context, "full_neighborhoods"):
        return context

    class _Proxy:
        pass

    proxy = _Proxy()
    proxy.source_neighborhoods = context.source_neighborhoods
    proxy.source_roles = context.source_roles
    proxy.target_families = context.target_families
    proxy.full_neighborhoods = _target_records(context)
    proxy.full_no_label_neighborhoods = dict(context.source_neighborhoods)
    return proxy


def _target_records(context: Any) -> dict[str, Any]:
    if hasattr(context, "target_neighborhoods"):
        return context.target_neighborhoods
    return getattr(context, "full_neighborhoods", {})


def _subset_prefixed(features: dict[str, Any], prefix: str) -> dict[str, float]:
    return {str(key).removeprefix(prefix): float(value) for key, value in features.items() if str(key).startswith(prefix)}


def _dominant_transfer_share(rows: list[dict[str, Any]]) -> float:
    denominator = sum(max(0, int(row.get("positive_lift_family_count", 0))) for row in rows) or 1
    return max((int(row.get("positive_lift_family_count", 0)) / denominator for row in rows), default=0.0)
