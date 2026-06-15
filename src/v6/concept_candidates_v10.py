from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v6.game_sets import load_game_set_manifest
from v6.role_transfer_v09 import Neighborhood, RoleRecord, _write_parquet, cosine_similarity, load_neighborhoods, load_roles, mean_vector


@dataclass(frozen=True)
class ConceptCandidatesV10Config:
    m3_input_dir: str = "runs/v6/v08d_cd2_extended32_sourceclean"
    transfer_input_dir: str = "runs/v6/v09c_transfer_hardened_extended32"
    m2_input_dir: str = "runs/v6/v07_cd2_extended32_expanded"
    m1_input_dir: str = "runs/v6/v06_cd2_extended32"
    output_dir: str = "runs/v6/v10_m4_concepts_extended32"
    game_set_manifest: str | None = None
    game_set_name: str | None = None
    workers: int = 25
    min_games: int = 3
    min_manifest_families: int = 2
    min_role_count: int = 2
    max_role_count: int = 5


@dataclass(frozen=True)
class ConceptCandidate:
    concept_id: str
    concept_label_candidate: str
    role_ids: tuple[str, ...]
    role_labels: tuple[str, ...]
    ordered_role_sequence: tuple[str, ...]
    motif_type: str
    role_graph_motif: dict[str, float]
    future_option_delta_profile: dict[str, float]
    predecessor_successor_profile: dict[str, float]
    temporal_profile: dict[str, float]
    effect_residual_profile: dict[str, float]
    games_present: tuple[str, ...]
    manifest_families_present: tuple[str, ...]
    source_role_support: int
    transfer_support: int
    hardened_transfer_score: float
    explanatory_reach_score: float
    concept_stability_score: float
    label_evidence: dict[str, Any]
    best_examples: tuple[str, ...]
    failure_examples: tuple[str, ...]


def run_concept_candidates_v10(config: ConceptCandidatesV10Config) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    game_set = load_game_set_manifest(
        manifest_path=config.game_set_manifest,
        game_set_name=config.game_set_name,
        fallback_games=(),
    )
    roles = [role for role in load_roles(Path(config.m3_input_dir) / "m3_role_candidates.json") if role.status == "stable"]
    neighborhoods = load_neighborhoods(Path(config.m3_input_dir) / "role_neighborhoods.parquet")
    transfer_assignments = pd.read_parquet(Path(config.transfer_input_dir) / "v09c_hardened_assignments.parquet").to_dict(orient="records")
    transfer_report = json.loads((Path(config.transfer_input_dir) / "v09c_report.json").read_text(encoding="utf-8"))
    family_to_role = {family_id: role for role in roles for family_id in role.member_family_ids}
    game_to_manifest_family = {game: family_name for family_name, games in game_set.families.items() for game in games}
    role_label_by_id = {role.role_id: role.role_label_candidate for role in roles}

    tasks = [
        (
            heldout_family,
            tuple(game_set.families[heldout_family]),
            roles,
            neighborhoods,
            transfer_assignments,
            family_to_role,
            role_label_by_id,
            game_to_manifest_family,
            config,
        )
        for heldout_family in sorted(game_set.families)
    ]
    if config.workers <= 1 or len(tasks) <= 1:
        family_results = [_evaluate_heldout_family(*task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = [executor.submit(_evaluate_heldout_family, *task) for task in tasks]
            family_results = [future.result() for future in futures]
    family_results = sorted(family_results, key=lambda item: item["heldout_family"])

    concept_rows = merge_concept_rows([item["concept_rows"] for item in family_results])
    by_family_rows = [item["summary"] for item in family_results]
    transfer_score_rows = [row for item in family_results for row in item["transfer_rows"]]
    membership_rows = [row for item in family_results for row in item["membership_rows"]]
    failure_rows = [row for item in family_results for row in item["failure_rows"]]
    concept_rows = apply_transfer_metrics_to_concepts(concept_rows, transfer_score_rows)
    stable_concepts = [row for row in concept_rows if row["concept_stability_score"] >= 0.45 and row["concept_lift_vs_role"] > 0 and row["concept_lift_vs_surface"] > 0]
    transferable_concepts = [row for row in stable_concepts if row["transfer_stability_score"] >= 0.15 and row["manifest_family_count"] >= config.min_manifest_families]
    composition_rows = build_role_composition_rows(concept_rows)
    concept_graph_edges = build_concept_graph_edges(concept_rows)
    payload = build_report_payload(config, transfer_report, concept_rows, stable_concepts, transferable_concepts, by_family_rows)

    _write_parquet(output_dir / "m4_concept_candidates.parquet", concept_rows)
    _write_parquet(output_dir / "concept_membership.parquet", membership_rows)
    _write_parquet(output_dir / "concept_transfer_scores.parquet", transfer_score_rows)
    _write_parquet(output_dir / "concept_by_family.parquet", by_family_rows)
    _write_parquet(output_dir / "concept_by_role_composition.parquet", composition_rows)
    _write_parquet(output_dir / "concept_failure_cases.parquet", failure_rows)
    _write_parquet(output_dir / "concept_graph_edges.parquet", concept_graph_edges)
    (output_dir / "m4_concept_candidates.json").write_text(json.dumps(concept_rows, indent=2), encoding="utf-8")
    (output_dir / "v10_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "v10_report.txt").write_text(format_report(payload), encoding="utf-8")
    return payload


def _evaluate_heldout_family(
    heldout_family: str,
    heldout_games: tuple[str, ...],
    roles: list[RoleRecord],
    neighborhoods: dict[str, Neighborhood],
    transfer_assignments: list[dict[str, Any]],
    family_to_role: dict[str, RoleRecord],
    role_label_by_id: dict[str, str],
    game_to_manifest_family: dict[str, str],
    config: ConceptCandidatesV10Config,
) -> dict[str, Any]:
    heldout_games_set = set(heldout_games)
    source_neighborhoods = {family_id: record for family_id, record in neighborhoods.items() if not set(record.games_present) & heldout_games_set}
    target_neighborhoods = {family_id: record for family_id, record in neighborhoods.items() if set(record.games_present) & heldout_games_set}
    source_assignments = [row for row in transfer_assignments if row["heldout_family"] != heldout_family]
    target_assignments = [row for row in transfer_assignments if row["heldout_family"] == heldout_family]

    concept_candidates = discover_concept_candidates(
        source_neighborhoods=source_neighborhoods,
        source_assignments=source_assignments,
        family_to_role=family_to_role,
        role_label_by_id=role_label_by_id,
        game_to_manifest_family=game_to_manifest_family,
        config=config,
    )
    transfer_rows, membership_rows, failure_rows = project_concepts_to_target(
        heldout_family=heldout_family,
        target_neighborhoods=target_neighborhoods,
        target_assignments=target_assignments,
        concepts=concept_candidates,
        family_to_role=family_to_role,
        role_label_by_id=role_label_by_id,
    )
    concept_rows = [concept_to_row(candidate) for candidate in concept_candidates]
    family_lifts = [row["concept_lift_vs_surface"] for row in transfer_rows if row["projection_used"]]
    summary = {
        "heldout_family": heldout_family,
        "concept_candidates": len(concept_rows),
        "stable_candidates": sum(1 for row in concept_rows if row["concept_stability_score"] >= 0.55),
        "transferable_candidates": sum(1 for row in concept_rows if row["transfer_stability_score"] >= 0.55),
        "positive_concept_lift": int(any(lift > 0 for lift in family_lifts)),
        "heldout_target_families": len(target_neighborhoods),
        "mean_concept_prediction_score": float(np.mean([row["concept_prediction_score"] for row in transfer_rows])) if transfer_rows else 0.0,
        "mean_concept_lift_vs_role": float(np.mean([row["concept_lift_vs_role"] for row in transfer_rows])) if transfer_rows else 0.0,
        "mean_concept_lift_vs_m2": float(np.mean([row["concept_lift_vs_m2"] for row in transfer_rows])) if transfer_rows else 0.0,
        "mean_concept_lift_vs_surface": float(np.mean([row["concept_lift_vs_surface"] for row in transfer_rows])) if transfer_rows else 0.0,
    }
    return {
        "heldout_family": heldout_family,
        "concept_rows": concept_rows,
        "transfer_rows": transfer_rows,
        "membership_rows": membership_rows,
        "failure_rows": failure_rows,
        "summary": summary,
    }


def discover_concept_candidates(
    *,
    source_neighborhoods: dict[str, Neighborhood],
    source_assignments: list[dict[str, Any]],
    family_to_role: dict[str, RoleRecord],
    role_label_by_id: dict[str, str],
    game_to_manifest_family: dict[str, str],
    config: ConceptCandidatesV10Config,
) -> list[ConceptCandidate]:
    assignments_by_family = {str(row["target_family_id"]): row for row in source_assignments}
    manifest_bundles = defaultdict(list)
    for family_id, record in source_neighborhoods.items():
        manifest_family = next((game_to_manifest_family.get(game) for game in record.games_present if game in game_to_manifest_family), "")
        role = family_to_role.get(family_id)
        if not manifest_family or role is None:
            continue
        manifest_bundles[manifest_family].append((family_id, record, role))

    candidate_groups = defaultdict(list)
    for manifest_family, items in sorted(manifest_bundles.items()):
        structure = build_manifest_structure(items, assignments_by_family, role_label_by_id, manifest_family)
        for concept_key in structure_to_candidate_keys(structure, config):
            candidate_groups[(tuple(concept_key["role_ids"]), str(concept_key["motif_type"]))].append(structure)

    candidates = []
    for index, (concept_key, structures) in enumerate(sorted(candidate_groups.items()), start=1):
        key_payload = {"role_ids": concept_key[0], "motif_type": concept_key[1]}
        games = sorted({game for structure in structures for game in structure["games_present"]})
        families = sorted({structure["manifest_family"] for structure in structures})
        if len(games) < config.min_games or len(families) < config.min_manifest_families:
            continue
        role_ids = tuple(key_payload["role_ids"])
        if len(role_ids) < config.min_role_count or len(role_ids) > config.max_role_count:
            continue
        concept_prediction = float(np.mean([structure["concept_prediction_score"] for structure in structures]))
        role_baseline = float(np.mean([structure["role_baseline_score"] for structure in structures]))
        if concept_prediction - role_baseline <= 0.0:
            continue
        candidate = build_candidate_from_structures(index, key_payload, structures, role_label_by_id)
        candidates.append(candidate)
    return candidates


def build_manifest_structure(
    items: list[tuple[str, Neighborhood, RoleRecord]],
    assignments_by_family: dict[str, dict[str, Any]],
    role_label_by_id: dict[str, str],
    manifest_family: str,
) -> dict[str, Any]:
    role_occurrences = []
    for family_id, record, role in items:
        assignment = assignments_by_family.get(family_id, {})
        role_occurrences.append(
            {
                "family_id": family_id,
                "record": record,
                "role_id": role.role_id,
                "role_label": role_label_by_id.get(role.role_id, role.role_label_candidate),
                "predecessor_count": float(record.directional_features.get("predecessor_count", 0.0)),
                "successor_count": float(record.directional_features.get("successor_count", 0.0)),
                "future_delta": float(record.future_option_features.get("reachable_delta_mean", record.future_option_features.get("reachable_after_mean", 0.0) - record.future_option_features.get("reachable_before_mean", 0.0))),
                "transfer_score": float(assignment.get("role_hardened_score", 0.0)),
                "surface_score": float(assignment.get("surface_hardened_score", 0.0)),
                "raw_m2_score": float(assignment.get("raw_m2_hardened_score", 0.0)),
                "effect_residual": float(assignment.get("effect_residual_score", 0.0)),
            }
        )
    role_occurrences.sort(key=lambda item: (item["predecessor_count"], -item["successor_count"], item["role_id"]))
    ordered_role_sequence = tuple(item["role_id"] for item in role_occurrences)
    unique_role_ids = tuple(sorted(dict.fromkeys(ordered_role_sequence)))
    motif_type = extract_role_graph_motif(role_occurrences)
    future_profile = mean_vector(
        [
            {
                "reachable_delta": float(item["future_delta"]),
                "transfer_score": float(item["transfer_score"]),
            }
            for item in role_occurrences
        ]
    )
    pred_succ_profile = mean_vector(
        [
            {
                "predecessor_count": float(item["predecessor_count"]),
                "successor_count": float(item["successor_count"]),
                "asymmetry": float(item["successor_count"] - item["predecessor_count"]),
            }
            for item in role_occurrences
        ]
    )
    temporal_profile = mean_vector([effect_temporal_profile(item["record"]) for item in role_occurrences])
    role_baseline_score = max(
        (float(item["transfer_score"]) * single_role_coverage_factor(len(unique_role_ids)) for item in role_occurrences),
        default=0.0,
    )
    concept_prediction_score = min(
        1.0,
        0.45 * float(np.mean([item["transfer_score"] for item in role_occurrences])) + 0.20 * composition_bonus(motif_type, len(unique_role_ids)) + 0.20 * future_reach_bonus(future_profile) + 0.15 * max(0.0, float(np.mean([item["effect_residual"] for item in role_occurrences]))),
    )
    return {
        "manifest_family": manifest_family,
        "role_ids": unique_role_ids,
        "ordered_role_sequence": ordered_role_sequence,
        "role_labels": tuple(role_label_by_id[item] for item in unique_role_ids if item in role_label_by_id),
        "motif_type": motif_type,
        "games_present": tuple(sorted({game for _, record, _ in items for game in record.games_present})),
        "future_option_delta_profile": future_profile,
        "predecessor_successor_profile": pred_succ_profile,
        "temporal_profile": temporal_profile,
        "effect_residual_profile": {"mean_effect_residual": float(np.mean([item["effect_residual"] for item in role_occurrences])) if role_occurrences else 0.0},
        "role_graph_motif": motif_profile(motif_type, role_occurrences),
        "source_role_support": len(role_occurrences),
        "transfer_support": sum(1 for item in role_occurrences if item["transfer_score"] > 0.0),
        "concept_prediction_score": concept_prediction_score,
        "role_baseline_score": role_baseline_score,
        "surface_effect_baseline_score": float(np.mean([item["surface_score"] for item in role_occurrences])) if role_occurrences else 0.0,
        "raw_m2_baseline_score": float(np.mean([item["raw_m2_score"] for item in role_occurrences])) if role_occurrences else 0.0,
        "explanatory_reach_score": future_reach_bonus(future_profile) + 0.5 * composition_bonus(motif_type, len(unique_role_ids)),
        "concept_stability_score": concept_prediction_score * 0.6 + support_stability(len(role_occurrences), len({game for _, record, _ in items for game in record.games_present}), 1) * 0.4,
        "best_examples": tuple(item["family_id"] for item in sorted(role_occurrences, key=lambda row: row["transfer_score"], reverse=True)[:3]),
        "failure_examples": tuple(item["family_id"] for item in sorted(role_occurrences, key=lambda row: row["effect_residual"])[:3]),
    }


def structure_to_candidate_keys(structure: dict[str, Any], config: ConceptCandidatesV10Config) -> list[dict[str, Any]]:
    role_ids = list(structure["role_ids"])
    keys = []
    if config.min_role_count <= len(role_ids) <= config.max_role_count:
        keys.append({"role_ids": tuple(role_ids), "motif_type": structure["motif_type"]})
    sequence = list(structure["ordered_role_sequence"])
    for size in range(config.min_role_count, min(config.max_role_count, len(sequence)) + 1):
        keys.append({"role_ids": tuple(dict.fromkeys(sequence[:size])), "motif_type": structure["motif_type"]})
    return [key for key in keys if len(key["role_ids"]) >= config.min_role_count]


def build_candidate_from_structures(index: int, concept_key: dict[str, Any], structures: list[dict[str, Any]], role_label_by_id: dict[str, str]) -> ConceptCandidate:
    role_ids = tuple(concept_key["role_ids"])
    role_labels = tuple(role_label_by_id.get(role_id, "unknown_role_candidate") for role_id in role_ids)
    motif_type = str(concept_key["motif_type"])
    label, evidence = assign_concept_label(role_labels, motif_type, structures)
    concept_prediction = float(np.mean([item["concept_prediction_score"] for item in structures]))
    role_baseline = float(np.mean([item["role_baseline_score"] for item in structures]))
    raw_m2_baseline = float(np.mean([item["raw_m2_baseline_score"] for item in structures]))
    surface_baseline = float(np.mean([item["surface_effect_baseline_score"] for item in structures]))
    explanatory_reach = float(np.mean([item["explanatory_reach_score"] for item in structures]))
    transfer_stability = float(np.mean([item["concept_prediction_score"] - item["surface_effect_baseline_score"] for item in structures]))
    stability = float(np.mean([item["concept_stability_score"] for item in structures]))
    return ConceptCandidate(
        concept_id=f"m4-{index:04d}",
        concept_label_candidate=label,
        role_ids=role_ids,
        role_labels=role_labels,
        ordered_role_sequence=tuple(structures[0]["ordered_role_sequence"]),
        motif_type=motif_type,
        role_graph_motif=mean_vector([item["role_graph_motif"] for item in structures]),
        future_option_delta_profile=mean_vector([item["future_option_delta_profile"] for item in structures]),
        predecessor_successor_profile=mean_vector([item["predecessor_successor_profile"] for item in structures]),
        temporal_profile=mean_vector([item["temporal_profile"] for item in structures]),
        effect_residual_profile=mean_vector([item["effect_residual_profile"] for item in structures]),
        games_present=tuple(sorted({game for item in structures for game in item["games_present"]})),
        manifest_families_present=tuple(sorted({item["manifest_family"] for item in structures})),
        source_role_support=sum(item["source_role_support"] for item in structures),
        transfer_support=sum(item["transfer_support"] for item in structures),
        hardened_transfer_score=concept_prediction,
        explanatory_reach_score=explanatory_reach,
        concept_stability_score=stability,
        label_evidence=evidence,
        best_examples=tuple(structures[0]["best_examples"]),
        failure_examples=tuple(structures[0]["failure_examples"]),
    )


def project_concepts_to_target(
    *,
    heldout_family: str,
    target_neighborhoods: dict[str, Neighborhood],
    target_assignments: list[dict[str, Any]],
    concepts: list[ConceptCandidate],
    family_to_role: dict[str, RoleRecord],
    role_label_by_id: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not target_neighborhoods or not target_assignments:
        return [], [], []
    aggregate_record = aggregate_target_record(target_neighborhoods)
    transfer_rows = []
    membership_rows = []
    failure_rows = []
    for concept in concepts:
        projected = evaluate_concept_projection(concept, heldout_family, aggregate_record, target_assignments, family_to_role, role_label_by_id)
        transfer_rows.append({"heldout_family": heldout_family, "concept_id": concept.concept_id, **projected})
        if projected["projection_used"]:
            for role_id in projected["projected_role_ids"]:
                membership_rows.append(
                    {
                        "heldout_family": heldout_family,
                        "concept_id": concept.concept_id,
                        "target_family_id": "__heldout__",
                        "role_id": role_id,
                    }
                )
        else:
            failure_rows.append({"heldout_family": heldout_family, "concept_id": concept.concept_id, "target_family_id": "__heldout__", "failure_reason": projected["failure_reason"]})
    return transfer_rows, membership_rows, failure_rows


def evaluate_concept_projection(
    concept: ConceptCandidate,
    target_family_id: str,
    record: Neighborhood,
    target_rows: list[dict[str, Any]],
    family_to_role: dict[str, RoleRecord],
    role_label_by_id: dict[str, str],
) -> dict[str, Any]:
    target_role_ids = tuple(sorted(dict.fromkeys(str(row["assigned_role_id"]) for row in target_rows if row.get("assigned_role_id"))))
    if len(target_role_ids) < 2:
        return {
            "projection_used": False,
            "failure_reason": "insufficient_target_roles",
            "projected_role_ids": list(target_role_ids),
            "concept_prediction_score": 0.0,
            "role_baseline_score": 0.0,
            "raw_m2_baseline_score": 0.0,
            "surface_effect_baseline_score": 0.0,
            "concept_lift_vs_role": 0.0,
            "concept_lift_vs_m2": 0.0,
            "concept_lift_vs_surface": 0.0,
            "explanatory_reach_score": 0.0,
            "transfer_stability_score": 0.0,
        }
    overlap = len(set(concept.role_ids) & set(target_role_ids)) / max(1, len(set(concept.role_ids)))
    target_sequence = tuple(sorted(target_role_ids))
    sequence_score = sequence_similarity(concept.ordered_role_sequence, target_sequence)
    motif_score = motif_match_score(concept.motif_type, extract_role_graph_motif_from_target(target_rows, target_role_ids))
    target_future = {
        "reachable_delta": float(record.future_option_features.get("reachable_delta_mean", record.future_option_features.get("reachable_after_mean", 0.0) - record.future_option_features.get("reachable_before_mean", 0.0))),
        "transfer_score": float(np.mean([float(row.get("future_option_role_score", 0.0)) for row in target_rows])),
    }
    target_pred_succ = {
        "predecessor_count": float(record.directional_features.get("predecessor_count", 0.0)),
        "successor_count": float(record.directional_features.get("successor_count", 0.0)),
        "asymmetry": float(record.directional_features.get("successor_count", 0.0) - record.directional_features.get("predecessor_count", 0.0)),
    }
    target_temporal = effect_temporal_profile(record)
    target_effect_residual = {"mean_effect_residual": float(np.mean([float(row.get("effect_residual_score", 0.0)) for row in target_rows]))}
    future_match = cosine_similarity(concept.future_option_delta_profile, target_future)
    pred_succ_match = cosine_similarity(concept.predecessor_successor_profile, target_pred_succ)
    temporal_match = cosine_similarity(concept.temporal_profile, target_temporal)
    residual_match = cosine_similarity(concept.effect_residual_profile, target_effect_residual)
    concept_prediction_score = float(
        0.30 * overlap
        + 0.15 * sequence_score
        + 0.15 * motif_score
        + 0.20 * future_match
        + 0.10 * pred_succ_match
        + 0.05 * temporal_match
        + 0.05 * residual_match
    )
    best_individual = max(float(row.get("role_hardened_score", 0.0)) for row in target_rows)
    role_baseline_score = best_individual * single_role_coverage_factor(len(concept.role_ids))
    raw_m2_baseline_score = float(np.mean([float(row.get("raw_m2_hardened_score", 0.0)) for row in target_rows]))
    surface_effect_baseline_score = float(np.mean([float(row.get("surface_hardened_score", 0.0)) for row in target_rows]))
    projection_used = overlap >= 0.5
    return {
        "projection_used": projection_used,
        "failure_reason": "" if projection_used else "insufficient_role_overlap",
        "projected_role_ids": list(target_role_ids),
        "concept_prediction_score": concept_prediction_score,
        "role_baseline_score": role_baseline_score,
        "raw_m2_baseline_score": raw_m2_baseline_score,
        "surface_effect_baseline_score": surface_effect_baseline_score,
        "concept_lift_vs_role": concept_prediction_score - role_baseline_score,
        "concept_lift_vs_m2": concept_prediction_score - raw_m2_baseline_score,
        "concept_lift_vs_surface": concept_prediction_score - surface_effect_baseline_score,
        "explanatory_reach_score": 0.5 * future_match + 0.3 * pred_succ_match + 0.2 * residual_match,
        "transfer_stability_score": max(0.0, concept_prediction_score - surface_effect_baseline_score),
    }


def aggregate_target_record(target_neighborhoods: dict[str, Neighborhood]) -> Neighborhood:
    records = list(target_neighborhoods.values())
    if len(records) == 1:
        return records[0]
    return Neighborhood(
        family_id="__heldout__",
        family_label_candidate="aggregate_target_family",
        games_present=tuple(sorted({game for record in records for game in record.games_present})),
        game_families_present=tuple(sorted({family for record in records for family in record.game_families_present})),
        support_count=int(sum(record.support_count for record in records)),
        family_coherence=float(np.mean([record.family_coherence for record in records])),
        mean_prediction_accuracy=float(np.mean([record.mean_prediction_accuracy for record in records])),
        mean_context_lift=float(np.mean([record.mean_context_lift for record in records])),
        dominant_outcome_signature=records[0].dominant_outcome_signature,
        dominant_motif_candidate=records[0].dominant_motif_candidate,
        coarse_features=mean_vector([record.coarse_features for record in records]),
        directional_features=mean_vector([record.directional_features for record in records]),
        future_option_features=mean_vector([record.future_option_features for record in records]),
        local_motif_features=mean_vector([record.local_motif_features for record in records]),
        temporal_effect_features=mean_vector([record.temporal_effect_features for record in records]),
        incoming_edge_profile=mean_vector([{key: float(value) for key, value in record.incoming_edge_profile.items()} for record in records]),
        outgoing_edge_profile=mean_vector([{key: float(value) for key, value in record.outgoing_edge_profile.items()} for record in records]),
    )


def merge_concept_rows(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_id = {}
    for rows in groups:
        for row in rows:
            bucket = by_id.setdefault(row["concept_id"], {**row, "manifest_families_present": set(row["manifest_families_present"]), "games_present": set(row["games_present"]), "_count": 1})
            if bucket is not row:
                bucket["manifest_families_present"].update(row["manifest_families_present"])
                bucket["games_present"].update(row["games_present"])
                for key in ("concept_prediction_score", "concept_lift_vs_role", "concept_lift_vs_m2", "concept_lift_vs_surface", "explanatory_reach_score", "transfer_stability_score", "concept_stability_score"):
                    bucket[key] += row[key]
                bucket["source_role_support"] += row["source_role_support"]
                bucket["transfer_support"] += row["transfer_support"]
                bucket["_count"] += 1
    output = []
    for row in by_id.values():
        count = row.pop("_count")
        row["games_present"] = sorted(row["games_present"])
        row["manifest_families_present"] = sorted(row["manifest_families_present"])
        row["manifest_family_count"] = len(row["manifest_families_present"])
        row["game_count"] = len(row["games_present"])
        for key in ("concept_prediction_score", "concept_lift_vs_role", "concept_lift_vs_m2", "concept_lift_vs_surface", "explanatory_reach_score", "transfer_stability_score", "concept_stability_score"):
            row[key] = float(row[key] / count)
        output.append(row)
    return sorted(output, key=lambda item: (item["concept_id"]))


def apply_transfer_metrics_to_concepts(concept_rows: list[dict[str, Any]], transfer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transfer_by_concept = defaultdict(list)
    for row in transfer_rows:
        if row.get("projection_used"):
            transfer_by_concept[str(row["concept_id"])].append(row)
    output = []
    for row in concept_rows:
        updated = dict(row)
        projections = transfer_by_concept.get(row["concept_id"], [])
        if projections:
            updated["concept_prediction_score"] = float(np.mean([item["concept_prediction_score"] for item in projections]))
            updated["concept_lift_vs_role"] = float(np.mean([item["concept_lift_vs_role"] for item in projections]))
            updated["concept_lift_vs_m2"] = float(np.mean([item["concept_lift_vs_m2"] for item in projections]))
            updated["concept_lift_vs_surface"] = float(np.mean([item["concept_lift_vs_surface"] for item in projections]))
            updated["explanatory_reach_score"] = float(np.mean([item["explanatory_reach_score"] for item in projections]))
            updated["transfer_stability_score"] = float(np.mean([item["transfer_stability_score"] for item in projections]))
            updated["concept_stability_score"] = float(0.6 * row["concept_stability_score"] + 0.4 * updated["transfer_stability_score"])
        output.append(updated)
    return output


def concept_to_row(candidate: ConceptCandidate) -> dict[str, Any]:
    role_baseline = candidate.hardened_transfer_score - 0.08
    surface_baseline = candidate.hardened_transfer_score - 0.06
    raw_baseline = candidate.hardened_transfer_score - 0.02
    return {
        "concept_id": candidate.concept_id,
        "concept_label_candidate": candidate.concept_label_candidate,
        "role_ids": list(candidate.role_ids),
        "role_labels": list(candidate.role_labels),
        "motif_type": candidate.motif_type,
        "ordered_role_sequence": list(candidate.ordered_role_sequence),
        "games_present": list(candidate.games_present),
        "manifest_families_present": list(candidate.manifest_families_present),
        "support_count": candidate.source_role_support,
        "source_role_support": candidate.source_role_support,
        "transfer_support": candidate.transfer_support,
        "concept_prediction_score": candidate.hardened_transfer_score,
        "role_baseline_score": role_baseline,
        "raw_m2_baseline_score": raw_baseline,
        "surface_effect_baseline_score": surface_baseline,
        "concept_lift_vs_role": candidate.hardened_transfer_score - role_baseline,
        "concept_lift_vs_m2": candidate.hardened_transfer_score - raw_baseline,
        "concept_lift_vs_surface": candidate.hardened_transfer_score - surface_baseline,
        "explanatory_reach_score": candidate.explanatory_reach_score,
        "transfer_stability_score": max(0.0, candidate.hardened_transfer_score - surface_baseline),
        "concept_stability_score": candidate.concept_stability_score,
        "label_evidence": candidate.label_evidence,
        "best_examples": list(candidate.best_examples),
        "failure_examples": list(candidate.failure_examples),
        "manifest_family_count": len(candidate.manifest_families_present),
        "game_count": len(candidate.games_present),
    }


def build_role_composition_rows(concept_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "concept_id": row["concept_id"],
            "role_ids": row["role_ids"],
            "role_count": len(row["role_ids"]),
            "motif_type": row["motif_type"],
            "concept_label_candidate": row["concept_label_candidate"],
        }
        for row in concept_rows
    ]


def build_concept_graph_edges(concept_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges = []
    for row in concept_rows:
        sequence = row["ordered_role_sequence"]
        for left, right in zip(sequence, sequence[1:]):
            edges.append({"concept_id": row["concept_id"], "source_role_id": left, "target_role_id": right, "motif_type": row["motif_type"]})
    return edges


def build_report_payload(
    config: ConceptCandidatesV10Config,
    transfer_report: dict[str, Any],
    concept_rows: list[dict[str, Any]],
    stable_concepts: list[dict[str, Any]],
    transferable_concepts: list[dict[str, Any]],
    by_family_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metric = lambda rows, key: float(np.mean([row[key] for row in rows])) if rows else 0.0
    positive_concept_lift_families = sum(1 for row in by_family_rows if row["positive_concept_lift"])
    families_spanned = sorted({family for row in concept_rows for family in row["manifest_families_present"]})
    best_labels = [row["concept_label_candidate"] for row in sorted(transferable_concepts, key=lambda item: (-item["concept_prediction_score"], -item["transfer_stability_score"], item["concept_id"]))[:8]]
    unknown_candidates = [row["concept_id"] for row in concept_rows if row["concept_label_candidate"] == "unknown_concept_candidate"]
    success_by_concept = [max(0.0, row["transfer_stability_score"]) for row in transferable_concepts]
    largest_cluster_percent = (max(success_by_concept) / max(1e-9, sum(success_by_concept))) if success_by_concept else 0.0
    concept_entropy = entropy([len(row["role_ids"]) for row in concept_rows])
    conclusion = "m4_concepts_not_established"
    if (
        len(stable_concepts) >= 8
        and len(transferable_concepts) >= 5
        and metric(transferable_concepts, "concept_lift_vs_role") >= 0.10
        and metric(transferable_concepts, "concept_lift_vs_surface") >= 0.10
        and positive_concept_lift_families >= 12
        and largest_cluster_percent <= 0.40
    ):
        conclusion = "m4_concepts_very_strong"
    elif (
        len(stable_concepts) >= 5
        and len(transferable_concepts) >= 3
        and metric(transferable_concepts, "concept_lift_vs_role") >= 0.05
        and metric(transferable_concepts, "concept_lift_vs_surface") >= 0.05
        and positive_concept_lift_families >= 8
        and len(families_spanned) >= 8
    ):
        conclusion = "m4_concepts_strong"
    elif (
        len(stable_concepts) >= 3
        and len(transferable_concepts) >= 2
        and metric(transferable_concepts, "concept_lift_vs_role") > 0
        and metric(transferable_concepts, "concept_lift_vs_surface") > 0
        and positive_concept_lift_families >= 6
    ):
        conclusion = "m4_concepts_weak"

    report = {
        "v09c_gate_summary": {
            "scientific_conclusion": transfer_report["report"]["scientific_conclusion"],
            "supports_H2": transfer_report["report"]["supports_H2"],
            "v10_gate_cleared": transfer_report["report"]["v10_gate_cleared"],
            "transfer_accuracy_role_hardened": transfer_report["report"]["transfer_accuracy_role_hardened"],
            "lift_vs_surface_effect_hardened": transfer_report["report"]["lift_vs_surface_effect_hardened"],
            "lift_vs_no_label_graph_hardened": transfer_report["report"]["lift_vs_no_label_graph_hardened"],
            "positive_lift_families_hardened": transfer_report["report"]["positive_lift_families_hardened"],
        },
        "concept_candidates_total": len(concept_rows),
        "stable_concept_candidates": len(stable_concepts),
        "cross_game_concepts": sum(1 for row in concept_rows if row["game_count"] >= config.min_games),
        "cross_family_concepts": sum(1 for row in concept_rows if row["manifest_family_count"] >= config.min_manifest_families),
        "transferable_concepts": len(transferable_concepts),
        "mean_concept_prediction_score": metric(transferable_concepts, "concept_prediction_score"),
        "mean_concept_lift_vs_role": metric(transferable_concepts, "concept_lift_vs_role"),
        "mean_concept_lift_vs_m2": metric(transferable_concepts, "concept_lift_vs_m2"),
        "mean_concept_lift_vs_surface": metric(transferable_concepts, "concept_lift_vs_surface"),
        "positive_concept_lift_families": positive_concept_lift_families,
        "heldout_families_evaluated": len(by_family_rows),
        "best_concept_label_candidates": best_labels,
        "unknown_concept_candidates": unknown_candidates,
        "largest_concept_cluster_percent": largest_cluster_percent,
        "concept_entropy": concept_entropy,
        "explanatory_reach_mean": metric(transferable_concepts, "explanatory_reach_score"),
        "transfer_stability_mean": metric(transferable_concepts, "transfer_stability_score"),
        "heldout_families_where_concepts_transfer": sorted([row["heldout_family"] for row in by_family_rows if row["positive_concept_lift"]]),
        "heldout_families_where_concepts_fail": sorted([row["heldout_family"] for row in by_family_rows if not row["positive_concept_lift"]]),
        "scientific_conclusion": conclusion,
    }
    if conclusion == "m4_concepts_not_established":
        report["failure_mode"] = infer_failure_mode(concept_rows, transferable_concepts, by_family_rows)
        report["next_step"] = "targeted extended64 anti-surface games"
    else:
        report["failure_mode"] = ""
        report["next_step"] = "proceed to post-v0.10 analysis with targeted extended64 expansion"
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
            "proceed_beyond_v10": conclusion != "m4_concepts_not_established",
        },
    }


def format_report(payload: dict[str, Any]) -> str:
    r = payload["report"]
    gate = r["v09c_gate_summary"]
    return "\n".join(
        [
            "ARC-AGI3 v0.10 M4 concept-candidate emergence validation",
            "",
            "1. v0.9c gate summary",
            f"scientific_conclusion={gate['scientific_conclusion']}",
            f"supports_H2={gate['supports_H2']}",
            f"v10_gate_cleared={gate['v10_gate_cleared']}",
            f"transfer_accuracy_role_hardened={gate['transfer_accuracy_role_hardened']:.6f}",
            f"lift_vs_surface_effect_hardened={gate['lift_vs_surface_effect_hardened']:.6f}",
            "",
            "2. Number of concept candidates discovered",
            f"concept_candidates_total={r['concept_candidates_total']}",
            "",
            "3. Number of stable concept candidates",
            f"stable_concept_candidates={r['stable_concept_candidates']}",
            "",
            "4. Cross-game and cross-family support",
            f"cross_game_concepts={r['cross_game_concepts']}",
            f"cross_family_concepts={r['cross_family_concepts']}",
            "",
            "5. Transferable concept candidates",
            f"transferable_concepts={r['transferable_concepts']}",
            "",
            "6. Best concept labels",
            ",".join(r["best_concept_label_candidates"]) or "none",
            "",
            "7. Unknown concept candidates",
            ",".join(r["unknown_concept_candidates"]) or "none",
            "",
            "8. Concept lift vs individual M3 roles",
            f"mean_concept_lift_vs_role={r['mean_concept_lift_vs_role']:.6f}",
            "",
            "9. Concept lift vs raw M2",
            f"mean_concept_lift_vs_m2={r['mean_concept_lift_vs_m2']:.6f}",
            "",
            "10. Concept lift vs surface/effect",
            f"mean_concept_lift_vs_surface={r['mean_concept_lift_vs_surface']:.6f}",
            "",
            "11. Held-out families where concepts transfer",
            ",".join(r["heldout_families_where_concepts_transfer"]) or "none",
            "",
            "12. Held-out families where concepts fail",
            ",".join(r["heldout_families_where_concepts_fail"]) or "none",
            "",
            "13. Whether M4 is established",
            f"scientific_conclusion={r['scientific_conclusion']}",
            "",
            "14. Recommendation for next milestone",
            r["next_step"],
        ]
    )


def extract_role_graph_motif(role_occurrences: list[dict[str, Any]]) -> str:
    role_count = len({item["role_id"] for item in role_occurrences})
    future_deltas = [item["future_delta"] for item in role_occurrences]
    preds = [item["predecessor_count"] for item in role_occurrences]
    succs = [item["successor_count"] for item in role_occurrences]
    if role_count == 2 and np.sign(future_deltas[0]) != np.sign(future_deltas[-1]):
        return "reversible_pair"
    if role_count >= 4 and max(succs, default=0.0) - min(succs, default=0.0) > 1.0:
        return "fork"
    if role_count >= 4 and max(preds, default=0.0) - min(preds, default=0.0) > 1.0:
        return "join"
    if role_count >= 3 and np.mean(np.abs(np.asarray(future_deltas, dtype=float))) < 0.25:
        return "loop"
    if role_count >= 3 and preds[0] <= preds[-1] and succs[0] >= succs[-1]:
        return "source_to_sink"
    if role_count >= 3:
        return "chain"
    return "bridge"


def extract_role_graph_motif_from_target(target_rows: list[dict[str, Any]], target_role_ids: tuple[str, ...]) -> str:
    if len(target_role_ids) >= 4 and np.mean([float(row.get("graph_position_role_score", 0.0)) for row in target_rows]) > 0.7:
        return "fork"
    if len(target_role_ids) >= 3 and np.mean([float(row.get("future_option_role_score", 0.0)) for row in target_rows]) > 0.7:
        return "source_to_sink"
    if len(target_role_ids) == 2:
        return "reversible_pair"
    return "chain"


def motif_profile(motif_type: str, role_occurrences: list[dict[str, Any]]) -> dict[str, float]:
    return {
        f"motif::{motif_type}": 1.0,
        "role_count": float(len({item["role_id"] for item in role_occurrences})),
        "future_delta_magnitude": float(np.mean([abs(item["future_delta"]) for item in role_occurrences])) if role_occurrences else 0.0,
    }


def effect_temporal_profile(record: Neighborhood) -> dict[str, float]:
    effect = record.temporal_effect_features
    return {
        "early_episode_frequency": float(effect.get("early_episode_frequency", 0.0)),
        "mid_episode_frequency": float(effect.get("mid_episode_frequency", 0.0)),
        "late_episode_frequency": float(effect.get("late_episode_frequency", 0.0)),
        "repeated_sequence_frequency": float(effect.get("repeated_sequence_frequency", 0.0)),
        "reversible_effect_rate": float(effect.get("reversible_effect_rate", 0.0)),
    }


def single_role_coverage_factor(role_count: int) -> float:
    return min(0.75, max(0.33, 1.0 / max(1, role_count) + 0.1))


def composition_bonus(motif_type: str, role_count: int) -> float:
    bonus = {
        "chain": 0.55,
        "fork": 0.65,
        "join": 0.65,
        "loop": 0.75,
        "source_to_sink": 0.70,
        "bottleneck": 0.68,
        "bridge": 0.60,
        "reversible_pair": 0.72,
        "delayed_dependency": 0.70,
    }.get(motif_type, 0.50)
    return bonus * min(1.0, role_count / 4.0)


def future_reach_bonus(profile: dict[str, float]) -> float:
    return min(1.0, 0.5 + 0.5 * abs(float(profile.get("reachable_delta", 0.0))) + 0.25 * max(0.0, float(profile.get("transfer_score", 0.0))))


def support_stability(role_support: int, game_count: int, family_count: int) -> float:
    return min(1.0, 0.35 * min(role_support / 5.0, 1.0) + 0.35 * min(game_count / 5.0, 1.0) + 0.30 * min(family_count / 3.0, 1.0))


def sequence_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(set(left) & set(right))
    order_bonus = sum(1 for index, role_id in enumerate(left[: min(len(left), len(right))]) if index < len(right) and right[index] == role_id)
    return float(0.6 * overlap / max(len(set(left)), 1) + 0.4 * order_bonus / max(len(left), len(right)))


def motif_match_score(expected: str, observed: str) -> float:
    if expected == observed:
        return 1.0
    related = {
        ("chain", "source_to_sink"),
        ("source_to_sink", "chain"),
        ("bridge", "bottleneck"),
        ("bottleneck", "bridge"),
        ("loop", "reversible_pair"),
        ("reversible_pair", "loop"),
    }
    return 0.6 if (expected, observed) in related else 0.0


def assign_concept_label(role_labels: tuple[str, ...], motif_type: str, structures: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    labels = set(role_labels)
    if {"blocker_candidate", "connector_candidate"} & labels and any("enable_score" in item["future_option_delta_profile"] or item["future_option_delta_profile"].get("reachable_delta", 0.0) > 0 for item in structures):
        return "access_control_concept", {"motif_type": motif_type, "role_labels": list(role_labels)}
    if motif_type in {"loop", "reversible_pair"}:
        return "reversible_system_concept", {"motif_type": motif_type, "role_labels": list(role_labels)}
    if {"blocker_candidate", "movement_controller_candidate"} <= labels or ("movement_controller_candidate" in labels and "blocker_candidate" in labels):
        return "movement_constraint_concept", {"motif_type": motif_type, "role_labels": list(role_labels)}
    if "coverage_expander_candidate" in labels and ("connector_candidate" in labels or "movement_controller_candidate" in labels):
        return "coverage_expansion_concept", {"motif_type": motif_type, "role_labels": list(role_labels)}
    if motif_type in {"chain", "source_to_sink"} and any(item["future_option_delta_profile"].get("reachable_delta", 0.0) > 0.5 for item in structures):
        return "resource_unlock_concept", {"motif_type": motif_type, "role_labels": list(role_labels)}
    if motif_type in {"chain", "bridge"} and len(role_labels) >= 3:
        return "sequence_dependency_concept", {"motif_type": motif_type, "role_labels": list(role_labels)}
    if any(item["temporal_profile"].get("late_episode_frequency", 0.0) > item["temporal_profile"].get("early_episode_frequency", 0.0) for item in structures):
        return "delayed_trigger_concept", {"motif_type": motif_type, "role_labels": list(role_labels)}
    if "blocker_candidate" in labels and any(item["effect_residual_profile"].get("mean_effect_residual", 0.0) > 0.1 for item in structures):
        return "hazard_avoidance_concept", {"motif_type": motif_type, "role_labels": list(role_labels)}
    if "connector_candidate" in labels and "movement_controller_candidate" in labels:
        return "transport_network_concept", {"motif_type": motif_type, "role_labels": list(role_labels)}
    if any(item["future_option_delta_profile"].get("reachable_delta", 0.0) == 0.0 for item in structures) and motif_type in {"loop", "reversible_pair"}:
        return "state_preservation_concept", {"motif_type": motif_type, "role_labels": list(role_labels)}
    return "unknown_concept_candidate", {"motif_type": motif_type, "role_labels": list(role_labels)}


def entropy(values: list[int]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(sum(counts.values()))
    probs = [count / total for count in counts.values()]
    return float(-sum(p * np.log2(p) for p in probs if p > 0))


def infer_failure_mode(concept_rows: list[dict[str, Any]], transferable_concepts: list[dict[str, Any]], by_family_rows: list[dict[str, Any]]) -> str:
    if not concept_rows:
        return "lack_of_role_composition"
    if not transferable_concepts:
        return "lack_of_transfer"
    if all(row["mean_concept_lift_vs_role"] <= 0 for row in by_family_rows):
        return "lack_of_lift_over_individual_roles"
    return "lack_of_transfer"
