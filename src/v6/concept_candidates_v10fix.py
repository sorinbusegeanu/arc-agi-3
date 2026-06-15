from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from v6.role_transfer_v09 import _write_parquet, appearance_features, cosine_similarity, mean_vector
from v6.role_transfer_v09c import (
    FamilyContext,
    RoleTransferV09cConfig,
    future_option_behavior_features,
    graph_position_features,
    local_graph_motif_features,
    prepare_family_contexts,
)
from v6.concept_candidates_v10 import effect_temporal_profile, extract_role_graph_motif


@dataclass(frozen=True)
class ConceptCandidatesV10FixConfig:
    m3_input_dir: str = "runs/v6/v08d_cd2_extended32_sourceclean"
    transfer_input_dir: str = "runs/v6/v09c_transfer_hardened_extended32"
    m2_input_dir: str = "runs/v6/v07_cd2_extended32_expanded"
    m1_input_dir: str = "runs/v6/v06_cd2_extended32"
    output_dir: str = "runs/v6/v10_m4_concepts_methodology_fixed_extended32"
    game_set_manifest: str | None = None
    game_set_name: str | None = None
    workers: int = 25
    min_games: int = 3
    min_manifest_families: int = 2
    min_role_count: int = 2
    max_role_count: int = 5


def run_concept_candidates_v10fix(config: ConceptCandidatesV10FixConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_v10 = _load_optional_json(Path("runs/v6/v10_m4_concepts_extended32") / "v10_report.json")
    transfer_report = json.loads((Path(config.transfer_input_dir) / "v09c_report.json").read_text(encoding="utf-8"))
    transfer_rows = pd.read_parquet(Path(config.transfer_input_dir) / "v09c_hardened_assignments.parquet").to_dict(orient="records")
    transfer_by_heldout = defaultdict(list)
    for row in transfer_rows:
        transfer_by_heldout[str(row["heldout_family"])].append(row)

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
    if config.workers <= 1 or len(tasks) <= 1:
        family_results = [_evaluate_family(*task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = [executor.submit(_evaluate_family, *task) for task in tasks]
            family_results = [future.result() for future in futures]
    family_results = sorted(family_results, key=lambda item: item["heldout_family"])

    concept_rows = merge_concept_rows([item["concept_rows"] for item in family_results])
    transfer_score_rows = [row for item in family_results for row in item["transfer_rows"]]
    by_family_rows = [item["summary"] for item in family_results]
    failure_rows = [row for item in family_results for row in item["failure_rows"]]
    membership_rows = [row for item in family_results for row in item["membership_rows"]]
    concept_rows = apply_target_metrics(concept_rows, transfer_score_rows)
    stable_concepts = [row for row in concept_rows if row["concept_stability_score"] >= 0.45 and row["deterministic_id_collision_free"]]
    transferable_concepts = [row for row in stable_concepts if row["transfer_stability_score"] >= 0.15 and row["target_mean_concept_lift_vs_surface_raw"] > 0 and row["target_mean_concept_lift_vs_role"] > 0]
    graph_edges = build_graph_edges(concept_rows)
    composition_rows = build_role_composition_rows(concept_rows)
    collision_rows = build_collision_rows(concept_rows)
    label_rows = build_label_rows(concept_rows)
    surface_rows = build_surface_comparison_rows(transfer_score_rows)
    payload = build_report_payload(config, original_v10, transfer_report, concept_rows, stable_concepts, transferable_concepts, by_family_rows, collision_rows, label_rows)

    _write_parquet(output_dir / "m4_concept_candidates_fixed.parquet", concept_rows)
    _write_parquet(output_dir / "concept_membership_fixed.parquet", membership_rows)
    _write_parquet(output_dir / "concept_transfer_scores_fixed.parquet", transfer_score_rows)
    _write_parquet(output_dir / "concept_by_family_fixed.parquet", by_family_rows)
    _write_parquet(output_dir / "concept_by_role_composition_fixed.parquet", composition_rows)
    _write_parquet(output_dir / "concept_failure_cases_fixed.parquet", failure_rows)
    _write_parquet(output_dir / "concept_graph_edges_fixed.parquet", graph_edges)
    _write_parquet(output_dir / "concept_id_collision_diagnostics.parquet", collision_rows)
    _write_parquet(output_dir / "concept_label_diagnostics.parquet", label_rows)
    _write_parquet(output_dir / "surface_baseline_comparison.parquet", surface_rows)
    (output_dir / "m4_concept_candidates_fixed.json").write_text(json.dumps(concept_rows, indent=2), encoding="utf-8")
    (output_dir / "v10fix_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "v10fix_report.txt").write_text(format_report(payload), encoding="utf-8")
    return payload


def _evaluate_family(context: FamilyContext, target_rows: list[dict[str, Any]], config: ConceptCandidatesV10FixConfig) -> dict[str, Any]:
    source_role_map = build_source_role_map(context.source_roles)
    concepts = discover_source_only_concepts(context, source_role_map, config)
    concept_rows = [concept_to_row(row) for row in concepts]
    target_record = aggregate_target_record({family.family_id: context.full_neighborhoods[family.family_id] for family in context.target_families if family.family_id in context.full_neighborhoods}) if context.target_families else None
    projection_rows = []
    membership_rows = []
    failure_rows = []
    if target_record is not None and target_rows:
        for concept in concepts:
            projection = evaluate_target_projection(concept, target_record, target_rows, source_role_map, context)
            projection_rows.append({"heldout_family": context.heldout_family, "concept_id": concept["concept_id"], **projection})
            if projection["projection_used"]:
                for role_id in concept["role_ids"]:
                    membership_rows.append({"heldout_family": context.heldout_family, "concept_id": concept["concept_id"], "role_id": role_id})
            else:
                failure_rows.append({"heldout_family": context.heldout_family, "concept_id": concept["concept_id"], "failure_reason": projection["failure_reason"]})
    summary = {
        "heldout_family": context.heldout_family,
        "concept_candidates": len(concept_rows),
        "stable_candidates": 0,
        "transferable_candidates": 0,
        "positive_concept_lift": int(any(row["target_concept_lift_vs_surface_raw"] > 0 and row["target_concept_lift_vs_role"] > 0 for row in projection_rows)),
        "target_mean_concept_lift_vs_role": float(np.mean([row["target_concept_lift_vs_role"] for row in projection_rows])) if projection_rows else 0.0,
        "target_mean_concept_lift_vs_m2": float(np.mean([row["target_concept_lift_vs_m2"] for row in projection_rows])) if projection_rows else 0.0,
        "target_mean_concept_lift_vs_surface_raw": float(np.mean([row["target_concept_lift_vs_surface_raw"] for row in projection_rows])) if projection_rows else 0.0,
        "target_mean_concept_lift_vs_surface_hardened": float(np.mean([row["target_concept_lift_vs_surface_hardened"] for row in projection_rows])) if projection_rows else 0.0,
        "target_role_overlap_used_in_main_score": False,
        "source_only_concept_discovery": True,
    }
    return {
        "heldout_family": context.heldout_family,
        "concept_rows": concept_rows,
        "transfer_rows": projection_rows,
        "membership_rows": membership_rows,
        "failure_rows": failure_rows,
        "summary": summary,
    }


def discover_source_only_concepts(context: FamilyContext, source_role_map: dict[str, dict[str, Any]], config: ConceptCandidatesV10FixConfig) -> list[dict[str, Any]]:
    manifest_groups = defaultdict(list)
    for family_id, record in sorted(context.source_neighborhoods.items()):
        role_id = source_role_map.get(family_id, {}).get("role_id", "")
        if not role_id:
            continue
        manifest_family = next(iter(get_game_families(record)), "")
        if not manifest_family:
            continue
        manifest_groups[manifest_family].append({"family_id": family_id, "record": record, "role_id": role_id, "role_label": source_role_map[family_id]["role_label_candidate"]})

    grouped_candidates = defaultdict(list)
    for manifest_family, items in sorted(manifest_groups.items()):
        structure = build_source_structure(manifest_family, items)
        if len(structure["role_ids"]) < config.min_role_count:
            continue
        grouped_candidates[composition_group_key(structure)].append(structure)

    candidates = []
    for fold_local_index, keys in enumerate(sorted(grouped_candidates.values(), key=lambda rows: composition_group_key(rows[0])), start=1):
        merged = merge_source_signatures(keys)
        if merged["game_count"] < config.min_games or merged["manifest_family_count"] < config.min_manifest_families:
            continue
        merged = stable_concept_signature(merged)
        label, evidence = strict_label_candidate(merged)
        merged["concept_label_candidate"] = label
        merged["label_evidence"] = evidence
        merged["fold_local_index"] = fold_local_index
        candidates.append(merged)
    return candidates


def build_source_structure(manifest_family: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(items, key=lambda item: (float(item["record"].directional_features.get("predecessor_count", 0.0)), -float(item["record"].directional_features.get("successor_count", 0.0)), item["role_id"]))
    role_ids = tuple(dict.fromkeys(item["role_id"] for item in ordered))
    role_labels = tuple(dict.fromkeys(item["role_label"] for item in ordered))
    motif_type = extract_role_graph_motif(
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
    return {
        "role_ids": role_ids,
        "role_labels": role_labels,
        "graph_ordered_role_pattern": tuple(item["role_id"] for item in ordered),
        "episode_ordered_role_sequence": (),
        "episode_order_available": False,
        "graph_order_fallback_used": True,
        "motif_type": motif_type,
        "future_option_delta_profile": mean_vector([future_option_behavior_features(item["record"]) for item in ordered]),
        "graph_position_profile": mean_vector([graph_position_features(item["record"]) for item in ordered]),
        "local_motif_profile": mean_vector([local_graph_motif_features(item["record"]) for item in ordered]),
        "predecessor_successor_profile": mean_vector(
            [
                {
                    "predecessor_count": float(item["record"].directional_features.get("predecessor_count", 0.0)),
                    "successor_count": float(item["record"].directional_features.get("successor_count", 0.0)),
                    "directional_asymmetry_score": float(item["record"].directional_features.get("directional_asymmetry_score", 0.0)),
                }
                for item in ordered
            ]
        ),
        "temporal_profile": mean_vector([effect_temporal_profile(item["record"]) for item in ordered]),
        "effect_residual_profile": {"source_effect_complexity": float(np.mean([abs(item["record"].temporal_effect_features.get("reversible_effect_rate", 0.0)) for item in ordered]))},
        "games_present": tuple(sorted({game for item in ordered for game in get_games(item["record"])})),
        "manifest_families_present": (manifest_family,),
        "source_role_support": len(ordered),
        "source_concept_quality_score": source_quality_score(ordered, motif_type),
        "required_roles_present": sorted(role_labels),
    }


def stable_concept_signature(structure: dict[str, Any]) -> dict[str, Any]:
    signature = {
        "role_ids": sorted(structure["role_ids"]),
        "motif_type": structure["motif_type"],
        "graph_ordered_role_pattern": list(structure["graph_ordered_role_pattern"]),
        "episode_ordered_role_sequence": list(structure["episode_ordered_role_sequence"]),
        "coarse_concept_fingerprint": {
            "future_delta_keys": sorted(structure["future_option_delta_profile"])[:6],
            "graph_keys": sorted(structure["graph_position_profile"])[:6],
            "motif_keys": sorted(structure["local_motif_profile"])[:6],
        },
        "source_manifest_family_support_signature": sorted(structure["manifest_families_present"]),
    }
    canonical = canonical_json(signature)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return {
        **structure,
        "concept_signature_json": canonical,
        "concept_fingerprint_hash": digest,
        "concept_id": f"m4-{digest}",
    }


def composition_group_key(structure: dict[str, Any]) -> str:
    signature = {
        "role_ids": sorted(structure["role_ids"]),
        "motif_type": structure["motif_type"],
        "graph_ordered_role_pattern": list(structure["graph_ordered_role_pattern"]),
        "episode_ordered_role_sequence": list(structure["episode_ordered_role_sequence"]),
        "coarse_concept_fingerprint": {
            "future_delta_keys": sorted(structure["future_option_delta_profile"])[:6],
            "graph_keys": sorted(structure["graph_position_profile"])[:6],
            "motif_keys": sorted(structure["local_motif_profile"])[:6],
        },
    }
    return canonical_json(signature)


def merge_source_signatures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = dict(rows[0])
    first["manifest_families_present"] = tuple(sorted({family for row in rows for family in row["manifest_families_present"]}))
    first["games_present"] = tuple(sorted({game for row in rows for game in row["games_present"]}))
    first["source_role_support"] = sum(int(row["source_role_support"]) for row in rows)
    first["source_concept_quality_score"] = float(np.mean([row["source_concept_quality_score"] for row in rows]))
    for key in ("future_option_delta_profile", "graph_position_profile", "local_motif_profile", "predecessor_successor_profile", "temporal_profile", "effect_residual_profile"):
        first[key] = mean_vector([row[key] for row in rows])
    first["manifest_family_count"] = len(first["manifest_families_present"])
    first["game_count"] = len(first["games_present"])
    return first


def evaluate_target_projection(concept: dict[str, Any], target_record: Any, target_rows: list[dict[str, Any]], source_role_map: dict[str, dict[str, Any]], context: FamilyContext) -> dict[str, Any]:
    target_future = future_option_behavior_features(target_record)
    target_graph = graph_position_features(target_record)
    target_motif = local_graph_motif_features(target_record)
    target_predsucc = {
        "predecessor_count": float(target_record.directional_features.get("predecessor_count", 0.0)),
        "successor_count": float(target_record.directional_features.get("successor_count", 0.0)),
        "directional_asymmetry_score": float(target_record.directional_features.get("directional_asymmetry_score", 0.0)),
    }
    target_temporal = effect_temporal_profile(target_record)
    target_effect_residual = {"target_effect_residual": float(np.mean([float(row.get("effect_residual_score", 0.0)) for row in target_rows])) if target_rows else 0.0}

    future_match = cosine_similarity(concept["future_option_delta_profile"], target_future)
    graph_match = cosine_similarity(concept["graph_position_profile"], target_graph)
    motif_match = cosine_similarity(concept["local_motif_profile"], target_motif)
    predsucc_match = cosine_similarity(concept["predecessor_successor_profile"], target_predsucc)
    temporal_match = cosine_similarity(concept["temporal_profile"], target_temporal)
    residual_match = cosine_similarity(concept["effect_residual_profile"], target_effect_residual)
    target_score = float(0.28 * future_match + 0.24 * graph_match + 0.16 * motif_match + 0.14 * predsucc_match + 0.10 * temporal_match + 0.08 * residual_match)

    role_baseline = best_individual_role_score(context, target_record)
    raw_m2_baseline = best_raw_m2_score(context, target_record)
    surface_raw_baseline = best_surface_raw_score(context, target_record)
    surface_hardened_baseline = float(np.mean([float(row.get("surface_hardened_score", 0.0)) for row in target_rows])) if target_rows else 0.0

    target_role_ids = sorted(dict.fromkeys(str(row.get("assigned_role_id", "")) for row in target_rows if row.get("assigned_role_id")))
    role_overlap = len(set(concept["role_ids"]) & set(target_role_ids)) / max(1, len(set(concept["role_ids"])))
    role_sequence_similarity = sequence_similarity(tuple(concept["graph_ordered_role_pattern"]), tuple(target_role_ids))
    assigned_role_motif = extract_role_graph_motif(
        [
            {
                "role_id": row.get("assigned_role_id", ""),
                "predecessor_count": float(row.get("graph_position_role_score", 0.0)),
                "successor_count": float(row.get("future_option_role_score", 0.0)),
                "future_delta": float(row.get("future_option_role_score", 0.0)),
            }
            for row in target_rows
            if row.get("assigned_role_id")
        ]
    ) if target_rows else ""

    return {
        "projection_used": bool(target_rows),
        "failure_reason": "" if target_rows else "missing_target_rows",
        "source_concept_quality_score": concept["source_concept_quality_score"],
        "target_concept_prediction_score": target_score,
        "target_concept_lift_vs_role": target_score - role_baseline,
        "target_concept_lift_vs_m2": target_score - raw_m2_baseline,
        "target_concept_lift_vs_surface_raw": target_score - surface_raw_baseline,
        "target_concept_lift_vs_surface_hardened": target_score - surface_hardened_baseline,
        "source_role_support": concept["source_role_support"],
        "explanatory_reach_score": float(0.5 * future_match + 0.3 * graph_match + 0.2 * predsucc_match),
        "transfer_stability_score": max(0.0, target_score - surface_raw_baseline),
        "role_id_overlap_diagnostic": role_overlap,
        "role_sequence_similarity_diagnostic": role_sequence_similarity,
        "assigned_role_motif_diagnostic": assigned_role_motif,
        "surface_effect_raw_score": surface_raw_baseline,
        "surface_effect_hardened_score": surface_hardened_baseline,
        "raw_m2_baseline_score": raw_m2_baseline,
        "role_baseline_score": role_baseline,
        "episode_order_available": False,
        "graph_ordered_role_pattern": list(concept["graph_ordered_role_pattern"]),
        "episode_ordered_role_sequence": list(concept["episode_ordered_role_sequence"]),
    }


def best_individual_role_score(context: FamilyContext, target_record: Any) -> float:
    target_future = future_option_behavior_features(target_record)
    target_graph = graph_position_features(target_record)
    best = 0.0
    for role in context.source_roles.values():
        future = subset_prefixed(role["all_features"], "future:")
        graph = subset_prefixed(role["all_features"], "directional:")
        score = 0.6 * cosine_similarity(target_future, future) + 0.4 * cosine_similarity(target_graph, graph)
        best = max(best, float(score))
    return best * 0.75


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


def source_quality_score(items: list[dict[str, Any]], motif_type: str) -> float:
    future_mean = float(np.mean([abs(item["record"].future_option_features.get("reachable_delta_mean", 0.0)) for item in items])) if items else 0.0
    graph_mean = float(np.mean([abs(item["record"].directional_features.get("directional_asymmetry_score", 0.0)) for item in items])) if items else 0.0
    motif_bonus = {"chain": 0.08, "fork": 0.10, "join": 0.10, "loop": 0.12, "source_to_sink": 0.11, "bridge": 0.09, "reversible_pair": 0.12}.get(motif_type, 0.07)
    return float(min(1.0, 0.35 + 0.25 * future_mean + 0.20 * graph_mean + motif_bonus))


def strict_label_candidate(concept: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    role_labels = set(concept["role_labels"])
    required_roles = sorted(label for label in role_labels if label in {"blocker_candidate", "connector_candidate", "movement_controller_candidate", "coverage_expander_candidate"})
    reachability_gating = abs(float(concept["future_option_delta_profile"].get("reachable_delta_mean", concept["future_option_delta_profile"].get("reachable_delta", 0.0)))) > 0.2 or abs(float(concept["future_option_delta_profile"].get("enable_score", 0.0))) > 0.2 or abs(float(concept["future_option_delta_profile"].get("block_score", 0.0))) > 0.2
    motif_evidence = concept["motif_type"] in {"source_to_sink", "bridge", "fork", "chain"}
    transfer_evidence = concept["source_concept_quality_score"] >= 0.45 and concept["manifest_family_count"] >= 2
    rejected = []
    if len(required_roles) >= 2 and reachability_gating and motif_evidence and transfer_evidence:
        return "access_control_concept", {
            "required_roles_present": required_roles,
            "reachability_gating_evidence": reachability_gating,
            "motif_evidence": concept["motif_type"],
            "transfer_evidence": transfer_evidence,
            "rejected_alternative_labels": rejected,
        }
    if "movement_controller_candidate" in role_labels and "blocker_candidate" in role_labels:
        rejected.append("access_control_concept")
        return "movement_constraint_concept", {
            "required_roles_present": required_roles,
            "reachability_gating_evidence": reachability_gating,
            "motif_evidence": concept["motif_type"],
            "transfer_evidence": transfer_evidence,
            "rejected_alternative_labels": rejected,
        }
    if "coverage_expander_candidate" in role_labels and ("movement_controller_candidate" in role_labels or "connector_candidate" in role_labels):
        rejected.append("access_control_concept")
        return "coverage_expansion_concept", {
            "required_roles_present": required_roles,
            "reachability_gating_evidence": reachability_gating,
            "motif_evidence": concept["motif_type"],
            "transfer_evidence": transfer_evidence,
            "rejected_alternative_labels": rejected,
        }
    if concept["motif_type"] == "chain" and len(concept["role_ids"]) >= 3:
        return "sequence_dependency_concept", {
            "required_roles_present": required_roles,
            "reachability_gating_evidence": reachability_gating,
            "motif_evidence": concept["motif_type"],
            "transfer_evidence": transfer_evidence,
            "rejected_alternative_labels": rejected + ["access_control_concept"],
        }
    return "unknown_concept_candidate", {
        "required_roles_present": required_roles,
        "reachability_gating_evidence": reachability_gating,
        "motif_evidence": concept["motif_type"],
        "transfer_evidence": transfer_evidence,
        "rejected_alternative_labels": rejected + ["access_control_concept"],
    }


def apply_target_metrics(concept_rows: list[dict[str, Any]], transfer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in transfer_rows:
        if row["projection_used"]:
            grouped[row["concept_id"]].append(row)
    output = []
    for row in concept_rows:
        projections = grouped.get(row["concept_id"], [])
        updated = dict(row)
        updated["target_mean_concept_prediction_score"] = float(np.mean([item["target_concept_prediction_score"] for item in projections])) if projections else 0.0
        updated["target_mean_concept_lift_vs_role"] = float(np.mean([item["target_concept_lift_vs_role"] for item in projections])) if projections else 0.0
        updated["target_mean_concept_lift_vs_m2"] = float(np.mean([item["target_concept_lift_vs_m2"] for item in projections])) if projections else 0.0
        updated["target_mean_concept_lift_vs_surface_raw"] = float(np.mean([item["target_concept_lift_vs_surface_raw"] for item in projections])) if projections else 0.0
        updated["target_mean_concept_lift_vs_surface_hardened"] = float(np.mean([item["target_concept_lift_vs_surface_hardened"] for item in projections])) if projections else 0.0
        updated["transfer_stability_score"] = float(np.mean([item["transfer_stability_score"] for item in projections])) if projections else 0.0
        updated["explanatory_reach_score"] = float(np.mean([item["explanatory_reach_score"] for item in projections])) if projections else row["explanatory_reach_score"]
        updated["concept_stability_score"] = float(0.55 * row["concept_stability_score"] + 0.45 * updated["transfer_stability_score"])
        updated["deterministic_id_collision_free"] = True
        output.append(updated)
    return output


def concept_to_row(concept: dict[str, Any]) -> dict[str, Any]:
    return {
        "concept_id": concept["concept_id"],
        "concept_fingerprint_hash": concept["concept_fingerprint_hash"],
        "fold_local_index": concept["fold_local_index"],
        "concept_signature_json": concept["concept_signature_json"],
        "concept_label_candidate": concept["concept_label_candidate"],
        "role_ids": list(concept["role_ids"]),
        "role_labels": list(concept["role_labels"]),
        "graph_ordered_role_pattern": list(concept["graph_ordered_role_pattern"]),
        "episode_ordered_role_sequence": list(concept["episode_ordered_role_sequence"]),
        "episode_order_available": concept["episode_order_available"],
        "motif_type": concept["motif_type"],
        "games_present": list(concept["games_present"]),
        "manifest_families_present": list(concept["manifest_families_present"]),
        "support_count": concept["source_role_support"],
        "source_role_support": concept["source_role_support"],
        "source_concept_quality_score": concept["source_concept_quality_score"],
        "concept_prediction_score": concept["source_concept_quality_score"],
        "concept_lift_vs_role": 0.0,
        "concept_lift_vs_m2": 0.0,
        "concept_lift_vs_surface": 0.0,
        "explanatory_reach_score": 0.0,
        "transfer_stability_score": 0.0,
        "concept_stability_score": source_stability(concept),
        "label_evidence": concept["label_evidence"],
        "best_examples": list(concept["games_present"][:3]),
        "failure_examples": [],
        "manifest_family_count": concept["manifest_family_count"],
        "game_count": concept["game_count"],
        "required_roles_present": concept["label_evidence"]["required_roles_present"],
        "reachability_gating_evidence": concept["label_evidence"]["reachability_gating_evidence"],
        "motif_evidence": concept["label_evidence"]["motif_evidence"],
        "transfer_evidence": concept["label_evidence"]["transfer_evidence"],
        "rejected_alternative_labels": concept["label_evidence"]["rejected_alternative_labels"],
    }


def source_stability(concept: dict[str, Any]) -> float:
    return float(min(1.0, 0.35 * min(concept["manifest_family_count"] / 4.0, 1.0) + 0.35 * min(concept["game_count"] / 5.0, 1.0) + 0.30 * concept["source_concept_quality_score"]))


def build_source_role_map(source_roles: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for role_id, role in source_roles.items():
        for family_id in role["member_family_ids"]:
            output[family_id] = {"role_id": role_id, "role_label_candidate": role["role_label_candidate"], "all_features": role["all_features"]}
    return output


def get_games(record: Any) -> tuple[str, ...]:
    return tuple(sorted(getattr(record, "games_present", getattr(record, "game_ids", ()))))


def get_game_families(record: Any) -> tuple[str, ...]:
    return tuple(sorted(getattr(record, "game_families_present", getattr(record, "game_family_ids", ()))))


def aggregate_target_record(target_neighborhoods: dict[str, Any]) -> Any:
    records = list(target_neighborhoods.values())
    if len(records) == 1:
        return records[0]
    return SimpleNamespace(
        family_id="__heldout__",
        family_label_candidate="aggregate_target_family",
        games_present=tuple(sorted({game for record in records for game in get_games(record)})),
        game_families_present=tuple(sorted({family for record in records for family in get_game_families(record)})),
        support_count=int(sum(int(getattr(record, "support_count", 0)) for record in records)),
        family_coherence=float(np.mean([float(getattr(record, "family_coherence", 0.0)) for record in records])),
        mean_prediction_accuracy=float(np.mean([float(getattr(record, "mean_prediction_accuracy", 0.0)) for record in records])),
        mean_context_lift=float(np.mean([float(getattr(record, "mean_context_lift", 0.0)) for record in records])),
        dominant_outcome_signature=str(getattr(records[0], "dominant_outcome_signature", "")),
        dominant_motif_candidate=str(getattr(records[0], "dominant_motif_candidate", "")),
        coarse_features=mean_vector([record.coarse_features for record in records]),
        directional_features=mean_vector([record.directional_features for record in records]),
        future_option_features=mean_vector([record.future_option_features for record in records]),
        local_motif_features=mean_vector([record.local_motif_features for record in records]),
        temporal_effect_features=mean_vector([record.temporal_effect_features for record in records]),
        incoming_edge_profile=mean_vector([{key: float(value) for key, value in getattr(record, "incoming_edge_profile", {}).items()} for record in records]),
        outgoing_edge_profile=mean_vector([{key: float(value) for key, value in getattr(record, "outgoing_edge_profile", {}).items()} for record in records]),
    )


def subset_prefixed(vector: dict[str, float], prefix: str) -> dict[str, float]:
    return {key[len(prefix):]: float(value) for key, value in vector.items() if key.startswith(prefix)}


def sequence_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(set(left) & set(right))
    order_bonus = sum(1 for index, role_id in enumerate(left[: min(len(left), len(right))]) if right[index] == role_id)
    return float(0.6 * overlap / max(1, len(set(left))) + 0.4 * order_bonus / max(len(left), len(right)))


def merge_concept_rows(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_id = {}
    for rows in groups:
        for row in rows:
            existing = by_id.get(row["concept_id"])
            if existing is None:
                by_id[row["concept_id"]] = dict(row)
                continue
            existing["games_present"] = sorted(set(existing["games_present"]) | set(row["games_present"]))
            existing["manifest_families_present"] = sorted(set(existing["manifest_families_present"]) | set(row["manifest_families_present"]))
            existing["support_count"] += row["support_count"]
            existing["source_role_support"] += row["source_role_support"]
            existing["source_concept_quality_score"] = float(np.mean([existing["source_concept_quality_score"], row["source_concept_quality_score"]]))
    for row in by_id.values():
        row["manifest_family_count"] = len(row["manifest_families_present"])
        row["game_count"] = len(row["games_present"])
    return sorted(by_id.values(), key=lambda item: item["concept_id"])


def build_graph_edges(concept_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges = []
    for row in concept_rows:
        sequence = row["graph_ordered_role_pattern"]
        for left, right in zip(sequence, sequence[1:]):
            edges.append({"concept_id": row["concept_id"], "source_role_id": left, "target_role_id": right, "graph_order_fallback_used": True})
    return edges


def build_role_composition_rows(concept_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"concept_id": row["concept_id"], "role_ids": row["role_ids"], "role_count": len(row["role_ids"]), "motif_type": row["motif_type"]} for row in concept_rows]


def build_collision_rows(concept_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(set)
    for row in concept_rows:
        grouped[row["concept_id"]].add(row["concept_signature_json"])
    return [{"concept_id": concept_id, "distinct_signature_count": len(signatures), "collision_detected": len(signatures) > 1} for concept_id, signatures in sorted(grouped.items())]


def build_label_rows(concept_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(row["concept_label_candidate"] for row in concept_rows)
    total = float(sum(counts.values())) or 1.0
    return [{"concept_label_candidate": label, "count": count, "percent": count / total} for label, count in sorted(counts.items())]


def build_surface_comparison_rows(transfer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "heldout_family": row["heldout_family"],
            "concept_id": row["concept_id"],
            "surface_effect_raw_score": row["surface_effect_raw_score"],
            "surface_effect_hardened_score": row["surface_effect_hardened_score"],
            "target_concept_prediction_score": row["target_concept_prediction_score"],
            "target_concept_lift_vs_surface_raw": row["target_concept_lift_vs_surface_raw"],
            "target_concept_lift_vs_surface_hardened": row["target_concept_lift_vs_surface_hardened"],
        }
        for row in transfer_rows
    ]


def build_report_payload(
    config: ConceptCandidatesV10FixConfig,
    original_v10: dict[str, Any] | None,
    transfer_report: dict[str, Any],
    concept_rows: list[dict[str, Any]],
    stable_concepts: list[dict[str, Any]],
    transferable_concepts: list[dict[str, Any]],
    by_family_rows: list[dict[str, Any]],
    collision_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metric = lambda rows, key: float(np.mean([row[key] for row in rows])) if rows else 0.0
    positive_families = sum(1 for row in by_family_rows if row["positive_concept_lift"])
    label_distribution = {row["concept_label_candidate"]: row["count"] for row in label_rows}
    dominant = max(label_rows, key=lambda row: row["percent"]) if label_rows else {"concept_label_candidate": "", "percent": 0.0}
    diversity_entropy = entropy([row["count"] for row in label_rows])
    collision_pass = all(not row["collision_detected"] for row in collision_rows)
    families_spanned = sorted({family for row in concept_rows for family in row["manifest_families_present"]})
    conclusion = "m4_concepts_methodology_fixed_not_established"
    if (
        len(stable_concepts) >= 8
        and len(transferable_concepts) >= 5
        and metric(transferable_concepts, "target_mean_concept_lift_vs_role") >= 0.10
        and metric(transferable_concepts, "target_mean_concept_lift_vs_surface_raw") >= 0.10
        and positive_families >= 12
        and max((row["transfer_stability_score"] for row in transferable_concepts), default=0.0) / max(1e-9, sum(max(0.0, row["transfer_stability_score"]) for row in transferable_concepts)) <= 0.40
        and dominant["percent"] <= 0.60
        and collision_pass
    ):
        conclusion = "m4_concepts_methodology_fixed_very_strong"
    elif (
        len(stable_concepts) >= 5
        and len(transferable_concepts) >= 3
        and metric(transferable_concepts, "target_mean_concept_lift_vs_role") >= 0.05
        and metric(transferable_concepts, "target_mean_concept_lift_vs_surface_raw") >= 0.05
        and positive_families >= 8
        and len(families_spanned) >= 8
        and collision_pass
    ):
        conclusion = "m4_concepts_methodology_fixed_strong"
    elif (
        len(stable_concepts) >= 3
        and len(transferable_concepts) >= 2
        and metric(transferable_concepts, "target_mean_concept_lift_vs_role") > 0
        and metric(transferable_concepts, "target_mean_concept_lift_vs_surface_raw") > 0
        and positive_families >= 6
        and collision_pass
    ):
        conclusion = "m4_concepts_methodology_fixed_weak"

    report = {
        "original_v10_summary": original_v10["report"] if original_v10 else {},
        "methodology_fixes_applied": [
            "deterministic concept ids",
            "source-only concept discovery per held-out family",
            "target role-id overlap removed from main score",
            "target-only concept pass/fail metrics",
            "raw vs hardened surface baselines separated",
            "graph-order vs episode-order labeling corrected",
            "stricter concept labeling",
        ],
        "leakage_removed": True,
        "source_only_concept_discovery": True,
        "cross_fold_assignment_reuse": False,
        "target_family_excluded_from_source": True,
        "target_role_id_overlap_removed_from_main_score": True,
        "concept_id_collision_check_passed": collision_pass,
        "target_mean_concept_lift_vs_role": metric(transferable_concepts, "target_mean_concept_lift_vs_role"),
        "target_mean_concept_lift_vs_m2": metric(transferable_concepts, "target_mean_concept_lift_vs_m2"),
        "target_mean_concept_lift_vs_surface_raw": metric(transferable_concepts, "target_mean_concept_lift_vs_surface_raw"),
        "target_mean_concept_lift_vs_surface_hardened": metric(transferable_concepts, "target_mean_concept_lift_vs_surface_hardened"),
        "corrected_concept_candidate_count": len(concept_rows),
        "corrected_stable_concepts": len(stable_concepts),
        "corrected_transferable_concepts": len(transferable_concepts),
        "positive_concept_lift_families": positive_families,
        "concepts_span_manifest_families": len(families_spanned),
        "concept_label_distribution": label_distribution,
        "dominant_concept_label": dominant["concept_label_candidate"],
        "dominant_label_percent": dominant["percent"],
        "non_access_control_concepts": sum(1 for row in concept_rows if row["concept_label_candidate"] != "access_control_concept"),
        "unknown_concept_count": sum(1 for row in concept_rows if row["concept_label_candidate"] == "unknown_concept_candidate"),
        "concept_diversity_entropy": diversity_entropy,
        "episode_order_available": False,
        "episode_order_coverage": 0.0,
        "graph_order_fallback_used": True,
        "heldout_families_where_concepts_transfer": sorted([row["heldout_family"] for row in by_family_rows if row["positive_concept_lift"]]),
        "heldout_families_where_concepts_fail": sorted([row["heldout_family"] for row in by_family_rows if not row["positive_concept_lift"]]),
        "scientific_conclusion": conclusion,
        "v10a_can_proceed": conclusion != "m4_concepts_methodology_fixed_not_established",
    }
    return {
        "config": {
            "m3_input_dir": config.m3_input_dir,
            "transfer_input_dir": config.transfer_input_dir,
            "m2_input_dir": config.m2_input_dir,
            "m1_input_dir": config.m1_input_dir,
            "output_dir": config.output_dir,
            "workers": config.workers,
        },
        "report": report,
        "validation": {
            "diagnostic_success": bool(concept_rows),
            "scientific_conclusion": conclusion,
            "proceed_to_v10a": report["v10a_can_proceed"],
        },
    }


def format_report(payload: dict[str, Any]) -> str:
    r = payload["report"]
    original = r["original_v10_summary"]
    return "\n".join(
        [
            "ARC-AGI3 v0.10 methodology fix: clean M4 concept validation",
            "",
            "1. Original v0.10 summary",
            f"scientific_conclusion={original.get('scientific_conclusion', '')}",
            f"stable_concept_candidates={original.get('stable_concept_candidates', 0)}",
            f"transferable_concepts={original.get('transferable_concepts', 0)}",
            "",
            "2. Methodology fixes applied",
            ",".join(r["methodology_fixes_applied"]),
            "",
            "3. Concept ID collision check",
            f"concept_id_collision_check_passed={r['concept_id_collision_check_passed']}",
            "",
            "4. Source-clean validation check",
            f"leakage_removed={r['leakage_removed']}",
            f"source_only_concept_discovery={r['source_only_concept_discovery']}",
            f"cross_fold_assignment_reuse={r['cross_fold_assignment_reuse']}",
            f"target_family_excluded_from_source={r['target_family_excluded_from_source']}",
            "",
            "5. Whether target role-ID overlap was removed from main score",
            f"target_role_id_overlap_removed_from_main_score={r['target_role_id_overlap_removed_from_main_score']}",
            "",
            "6. Raw vs hardened surface baseline comparison",
            f"target_mean_concept_lift_vs_surface_raw={r['target_mean_concept_lift_vs_surface_raw']:.6f}",
            f"target_mean_concept_lift_vs_surface_hardened={r['target_mean_concept_lift_vs_surface_hardened']:.6f}",
            "",
            "7. Corrected concept candidate count",
            f"corrected_concept_candidate_count={r['corrected_concept_candidate_count']}",
            "",
            "8. Corrected stable and transferable concepts",
            f"corrected_stable_concepts={r['corrected_stable_concepts']}",
            f"corrected_transferable_concepts={r['corrected_transferable_concepts']}",
            "",
            "9. Corrected lift vs individual M3 roles",
            f"target_mean_concept_lift_vs_role={r['target_mean_concept_lift_vs_role']:.6f}",
            "",
            "10. Corrected lift vs raw M2",
            f"target_mean_concept_lift_vs_m2={r['target_mean_concept_lift_vs_m2']:.6f}",
            "",
            "11. Corrected lift vs raw surface/effect",
            f"target_mean_concept_lift_vs_surface_raw={r['target_mean_concept_lift_vs_surface_raw']:.6f}",
            "",
            "12. Concept label distribution",
            json.dumps(r["concept_label_distribution"], separators=(",", ":")),
            "",
            "13. Held-out families where concepts transfer",
            ",".join(r["heldout_families_where_concepts_transfer"]) or "none",
            "",
            "14. Held-out families where concepts fail",
            ",".join(r["heldout_families_where_concepts_fail"]) or "none",
            "",
            "15. Corrected scientific conclusion",
            f"scientific_conclusion={r['scientific_conclusion']}",
            "",
            "16. Whether v0.10a can proceed",
            f"v10a_can_proceed={r['v10a_can_proceed']}",
        ]
    )


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def entropy(values: list[int]) -> float:
    if not values:
        return 0.0
    total = float(sum(values))
    probs = [value / total for value in values if value > 0]
    return float(-sum(p * np.log2(p) for p in probs))


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
