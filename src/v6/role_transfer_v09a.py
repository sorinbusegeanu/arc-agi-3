from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from v6.game_sets import load_game_set_manifest
from v6.role_candidates_v08d import (
    DiscNeighborhood,
    GraphBuildDiagnostics,
    RoleCandidatesV08dConfig,
    SimilarityWeights,
    build_discriminative_neighborhoods,
    build_game_family_map,
    build_role_candidates,
    build_similarity_adjacency,
    cluster_role_candidates,
    evaluate_pairwise_similarity,
    load_episode_summaries,
    load_m1_support,
    load_m2_families,
    load_m2_graph_edges,
    mean_similarity,
    role_status,
)
from v6.role_transfer_v09 import (
    appearance_features,
    cosine_similarity,
    deterministic_random_role_id,
    mean_vector,
    raw_m2_features,
)


@dataclass(frozen=True)
class RoleTransferV09aConfig:
    m2_input_dir: str = "runs/v6/v07_cd2_extended32_expanded"
    m1_input_dir: str = "runs/v6/v06_cd2_extended32"
    output_dir: str = "runs/v6/v09a_role_transfer_sourceclean_extended32"
    game_set_manifest: str | None = None
    game_set_name: str | None = None
    split_mode: str = "leave_family_out"
    workers: int = 25
    structural_success_threshold: float = 0.70
    graph_source: str = "hybrid"


def run_role_transfer_v09a(config: RoleTransferV09aConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    game_set = load_game_set_manifest(
        manifest_path=config.game_set_manifest,
        game_set_name=config.game_set_name,
        fallback_games=(),
    )
    m2_families = load_m2_families(Path(config.m2_input_dir))
    m1_support = load_m1_support(Path(config.m1_input_dir))
    episode_summaries = load_episode_summaries(Path(config.m1_input_dir))
    m2_graph_edges = load_m2_graph_edges(Path(config.m2_input_dir))
    selected_games = set(game_set.games) if game_set.games else {game for family in m2_families for game in family.games_present}
    m2_families = [family for family in m2_families if selected_games.intersection(family.games_present)]
    m1_support = {key: value for key, value in m1_support.items() if value.game_id in selected_games}
    game_family_map = build_game_family_map(game_set, tuple(sorted(selected_games)))
    normal_neighborhoods, _ = build_discriminative_neighborhoods(
        m2_families,
        m1_support,
        game_family_map,
        graph_source=config.graph_source,
        ablation="none",
        m2_graph_edges=m2_graph_edges,
        episode_summaries=episode_summaries,
    )
    no_label_neighborhoods, _ = build_discriminative_neighborhoods(
        m2_families,
        m1_support,
        game_family_map,
        graph_source=config.graph_source,
        ablation="no_m2_labels",
        m2_graph_edges=m2_graph_edges,
        episode_summaries=episode_summaries,
    )
    tasks = [
        (
            family_name,
            tuple(game_set.families[family_name]),
            m2_families,
            m1_support,
            game_family_map,
            m2_graph_edges,
            episode_summaries,
            config.graph_source,
            config.structural_success_threshold,
            normal_neighborhoods,
            no_label_neighborhoods,
        )
        for family_name in sorted(game_set.families)
    ]
    if config.workers <= 1 or len(tasks) <= 1:
        family_results = [_evaluate_heldout_family(*task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = [executor.submit(_evaluate_heldout_family, *task) for task in tasks]
            family_results = [future.result() for future in futures]
    family_results = sorted(family_results, key=lambda item: item["heldout_family"])

    assignment_rows = [row for result in family_results for row in result["assignments"]]
    failure_rows = [row for result in family_results for row in result["failures"]]
    by_family_rows = [result["summary"] for result in family_results]
    payload = build_payload(config, family_results, assignment_rows)

    from v6.role_transfer_v09 import _write_parquet

    _write_parquet(output_dir / "role_transfer_assignments.parquet", assignment_rows)
    _write_parquet(output_dir / "role_transfer_by_family.parquet", by_family_rows)
    _write_parquet(output_dir / "role_transfer_failures.parquet", failure_rows)
    (output_dir / "v09a_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "v09a_report.txt").write_text(format_report(payload), encoding="utf-8")
    return payload


def _evaluate_heldout_family(
    heldout_family: str,
    heldout_games: tuple[str, ...],
    m2_families: list[Any],
    m1_support: dict[str, Any],
    game_family_map: dict[str, str],
    m2_graph_edges: list[dict[str, Any]],
    episode_summaries: list[dict[str, Any]],
    graph_source: str,
    threshold: float,
    full_neighborhoods: dict[str, DiscNeighborhood],
    full_no_label_neighborhoods: dict[str, DiscNeighborhood],
) -> dict[str, Any]:
    heldout_games_set = set(heldout_games)
    source_families = [family for family in m2_families if not heldout_games_set.intersection(family.games_present)]
    target_families = [family for family in m2_families if heldout_games_set.intersection(family.games_present)]
    source_game_set = {game for family in source_families for game in family.games_present}
    source_support = {key: value for key, value in m1_support.items() if value.game_id in source_game_set}
    source_neighborhoods, graph_diag = build_discriminative_neighborhoods(
        source_families,
        source_support,
        game_family_map,
        graph_source=graph_source,
        ablation="none",
        m2_graph_edges=m2_graph_edges,
        episode_summaries=episode_summaries,
    )
    source_roles = build_source_only_roles(source_families, source_neighborhoods)
    source_no_label_neighborhoods, _ = build_discriminative_neighborhoods(
        source_families,
        source_support,
        game_family_map,
        graph_source=graph_source,
        ablation="no_m2_labels",
        m2_graph_edges=m2_graph_edges,
        episode_summaries=episode_summaries,
    )
    source_no_label_roles = build_source_only_roles(source_families, source_no_label_neighborhoods)

    assignments = []
    failures = []
    for family in sorted(target_families, key=lambda item: item.family_id):
        target = full_neighborhoods.get(family.family_id)
        target_no_label = full_no_label_neighborhoods.get(family.family_id)
        if target is None or target_no_label is None:
            failures.append({"heldout_family": heldout_family, "target_family_id": family.family_id, "failure_reason": "missing_target_neighborhood"})
            continue
        if family.support_count < 3:
            failures.append({"heldout_family": heldout_family, "target_family_id": family.family_id, "failure_reason": "insufficient_target_support"})
            continue
        if not source_roles:
            failures.append({"heldout_family": heldout_family, "target_family_id": family.family_id, "failure_reason": "no_source_roles"})
            continue

        assigned_role_id, assigned_scores = nearest_role_prototype(target, source_roles)
        random_role_id = deterministic_random_role_id({key: None for key in sorted(source_roles)}, heldout_family, family.family_id)
        majority_role_id = max(sorted(source_roles), key=lambda key: len(source_roles[key]["member_family_ids"]))
        coarse_role_id = nearest_role_by_vector(target.coarse_features, source_roles, "coarse_features")
        surface_role_id = nearest_role_by_vector(appearance_features(target), source_roles, "appearance_features")
        raw_family_id = nearest_source_family(target, source_neighborhoods, mode="raw_m2")
        m2_label_role_id = m2_label_baseline_role(target, source_neighborhoods, source_roles)
        no_label_role_id, no_label_scores = nearest_role_prototype(target_no_label, source_no_label_roles)

        row = {
            "heldout_family": heldout_family,
            "target_family_id": family.family_id,
            "target_games": list(family.games_present),
            "target_game_families": [game_family_map.get(game, "unknown") for game in family.games_present],
            "assigned_role_id": assigned_role_id,
            "assigned_structural_prediction_score": assigned_scores["structural_prediction_score"],
            "future_option_prediction_score": assigned_scores["future_option_prediction_score"],
            "directional_prediction_score": assigned_scores["directional_prediction_score"],
            "motif_prediction_score": assigned_scores["motif_prediction_score"],
            "effect_prediction_score": assigned_scores["effect_prediction_score"],
            "role_success": int(assigned_scores["structural_prediction_score"] >= threshold),
            "random_role_id": random_role_id,
            "random_success": int(structural_score(target, source_roles[random_role_id])["structural_prediction_score"] >= threshold),
            "majority_role_id": majority_role_id,
            "majority_success": int(structural_score(target, source_roles[majority_role_id])["structural_prediction_score"] >= threshold),
            "coarse_role_id": coarse_role_id,
            "coarse_success": int(structural_score(target, source_roles[coarse_role_id])["structural_prediction_score"] >= threshold),
            "surface_role_id": surface_role_id,
            "surface_success": int(structural_score(target, source_roles[surface_role_id])["structural_prediction_score"] >= threshold),
            "raw_m2_family_id": raw_family_id,
            "raw_m2_success": int(structural_score_to_family(target, source_neighborhoods[raw_family_id])["structural_prediction_score"] >= threshold) if raw_family_id else 0,
            "m2_label_role_id": m2_label_role_id,
            "m2_label_success": int(structural_score(target, source_roles[m2_label_role_id])["structural_prediction_score"] >= threshold) if m2_label_role_id else 0,
            "graph_role_no_label_role_id": no_label_role_id,
            "graph_role_no_label_success": int(no_label_scores["structural_prediction_score"] >= threshold),
            "diagnostic_full_data_role_label": target.family_label_candidate,
        }
        assignments.append(row)

    metrics = {}
    if assignments:
        for prefix in ("role", "random", "majority", "raw_m2", "coarse", "surface", "m2_label", "graph_role_no_label"):
            metrics[prefix] = float(np.mean([row[f"{prefix}_success"] for row in assignments]))
    else:
        metrics = {prefix: 0.0 for prefix in ("role", "random", "majority", "raw_m2", "coarse", "surface", "m2_label", "graph_role_no_label")}
    best_baseline = max(metrics["random"], metrics["majority"], metrics["raw_m2"], metrics["coarse"], metrics["surface"], metrics["m2_label"], metrics["graph_role_no_label"])
    summary = {
        "heldout_family": heldout_family,
        "graph_source_used": graph_diag.graph_source_used,
        "graph_edge_coverage": graph_diag.graph_edge_coverage,
        "source_only_roles": len(source_roles),
        "target_m2_families": len(assignments),
        "transfer_accuracy_structural_role": metrics["role"],
        "transfer_accuracy_random": metrics["random"],
        "transfer_accuracy_majority": metrics["majority"],
        "transfer_accuracy_raw_m2": metrics["raw_m2"],
        "transfer_accuracy_coarse": metrics["coarse"],
        "transfer_accuracy_surface_effect": metrics["surface"],
        "transfer_accuracy_m2_label": metrics["m2_label"],
        "transfer_accuracy_graph_role_no_label": metrics["graph_role_no_label"],
        "role_lift_over_best_baseline": metrics["role"] - best_baseline,
        "mean_structural_score": float(np.mean([row["assigned_structural_prediction_score"] for row in assignments])) if assignments else 0.0,
    }
    return {"heldout_family": heldout_family, "assignments": assignments, "failures": failures, "summary": summary}


def build_source_only_roles(source_families: list[Any], source_neighborhoods: dict[str, DiscNeighborhood]) -> dict[str, dict[str, Any]]:
    weights = SimilarityWeights(0.25, 0.20, 0.25, 0.20, 0.10)
    pair_results = evaluate_pairwise_similarity(source_neighborhoods, weights=weights, threshold=0.70, workers=1)
    adjacency = build_similarity_adjacency(pair_results, 0.70)
    clusters, rejected = cluster_role_candidates(adjacency, pair_results, source_neighborhoods, 0.70)
    roles = build_role_candidates(
        clusters=clusters,
        neighborhoods=source_neighborhoods,
        min_role_support=3,
        role_similarity_threshold=0.70,
        pair_results=pair_results,
    )
    output = {}
    for role in roles:
        if role.status != "stable":
            continue
        members = [source_neighborhoods[item] for item in role.member_family_ids if item in source_neighborhoods]
        if not members:
            continue
        output[role.role_id] = {
            "role_id": role.role_id,
            "role_label_candidate": role.role_label_candidate,
            "member_family_ids": tuple(role.member_family_ids),
            "all_features": mean_vector([all_feature_groups(record) for record in members]),
            "coarse_features": mean_vector([record.coarse_features for record in members]),
            "appearance_features": mean_vector([appearance_features(record) for record in members]),
        }
    return output


def all_feature_groups(record: DiscNeighborhood) -> dict[str, float]:
    output = {}
    for prefix, group in (
        ("coarse", record.coarse_features),
        ("directional", record.directional_features),
        ("future", record.future_option_features),
        ("motif", record.local_motif_features),
        ("effect", record.temporal_effect_features),
    ):
        for key, value in group.items():
            output[f"{prefix}:{key}"] = float(value)
    return output


def structural_score(target: DiscNeighborhood, role_entry: dict[str, Any]) -> dict[str, float]:
    return {
        "future_option_prediction_score": cosine_similarity(target.future_option_features, subset_prefixed(role_entry["all_features"], "future:")),
        "directional_prediction_score": cosine_similarity(target.directional_features, subset_prefixed(role_entry["all_features"], "directional:")),
        "motif_prediction_score": cosine_similarity(target.local_motif_features, subset_prefixed(role_entry["all_features"], "motif:")),
        "effect_prediction_score": cosine_similarity(target.temporal_effect_features, subset_prefixed(role_entry["all_features"], "effect:")),
        "structural_prediction_score": cosine_similarity(all_feature_groups(target), role_entry["all_features"]),
    }


def structural_score_to_family(target: DiscNeighborhood, source: DiscNeighborhood) -> dict[str, float]:
    return {
        "structural_prediction_score": cosine_similarity(all_feature_groups(target), all_feature_groups(source))
    }


def subset_prefixed(vector: dict[str, float], prefix: str) -> dict[str, float]:
    return {key[len(prefix):]: value for key, value in vector.items() if key.startswith(prefix)}


def nearest_role_prototype(target: DiscNeighborhood, roles: dict[str, dict[str, Any]]) -> tuple[str, dict[str, float]]:
    best_role = ""
    best_scores = {"structural_prediction_score": 0.0, "future_option_prediction_score": 0.0, "directional_prediction_score": 0.0, "motif_prediction_score": 0.0, "effect_prediction_score": 0.0}
    for role_id, entry in sorted(roles.items()):
        scores = structural_score(target, entry)
        if scores["structural_prediction_score"] > best_scores["structural_prediction_score"]:
            best_role = role_id
            best_scores = scores
    return best_role, best_scores


def nearest_role_by_vector(target_vector: dict[str, float], roles: dict[str, dict[str, Any]], field_name: str) -> str:
    best_role = ""
    best_score = -1.0
    for role_id, entry in sorted(roles.items()):
        score = cosine_similarity(target_vector, entry[field_name])
        if score > best_score:
            best_role = role_id
            best_score = score
    return best_role


def nearest_source_family(target: DiscNeighborhood, source_neighborhoods: dict[str, DiscNeighborhood], *, mode: str) -> str:
    best_family = ""
    best_score = -1.0
    target_vector = raw_m2_features(target) if mode == "raw_m2" else appearance_features(target)
    for family_id, record in sorted(source_neighborhoods.items()):
        source_vector = raw_m2_features(record) if mode == "raw_m2" else appearance_features(record)
        score = cosine_similarity(target_vector, source_vector)
        if score > best_score:
            best_family = family_id
            best_score = score
    return best_family


def m2_label_baseline_role(target: DiscNeighborhood, source_neighborhoods: dict[str, DiscNeighborhood], source_roles: dict[str, dict[str, Any]]) -> str:
    matching_source_families = [family_id for family_id, record in source_neighborhoods.items() if record.family_label_candidate == target.family_label_candidate]
    counts = {}
    for role_id, entry in source_roles.items():
        counts[role_id] = sum(1 for family_id in matching_source_families if family_id in entry["member_family_ids"])
    if counts and max(counts.values()) > 0:
        return max(sorted(counts), key=lambda key: counts[key])
    return max(sorted(source_roles), key=lambda key: len(source_roles[key]["member_family_ids"])) if source_roles else ""


def build_payload(config: RoleTransferV09aConfig, family_results: list[dict[str, Any]], assignment_rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [item["summary"] for item in family_results if item["summary"]["target_m2_families"] > 0]
    metric = lambda key: float(np.mean([row[key] for row in evaluable])) if evaluable else 0.0
    role_acc = metric("transfer_accuracy_structural_role")
    baselines = {
        "random": metric("transfer_accuracy_random"),
        "majority": metric("transfer_accuracy_majority"),
        "raw_m2": metric("transfer_accuracy_raw_m2"),
        "coarse": metric("transfer_accuracy_coarse"),
        "surface_effect": metric("transfer_accuracy_surface_effect"),
        "m2_label": metric("transfer_accuracy_m2_label"),
        "graph_role_no_label": metric("transfer_accuracy_graph_role_no_label"),
    }
    positive_families = sum(1 for row in evaluable if row["role_lift_over_best_baseline"] > 0)
    mean_lift_best = float(np.mean([row["role_lift_over_best_baseline"] for row in evaluable])) if evaluable else 0.0
    beats_avg = role_acc > baselines["surface_effect"] and role_acc > baselines["raw_m2"]
    if not assignment_rows:
        conclusion = "transfer_methodology_invalid"
    elif role_acc > max(baselines.values()) and positive_families >= 8 and mean_lift_best >= 0.08:
        conclusion = "role_transfer_sourceclean_strong"
    elif role_acc > baselines["surface_effect"] and role_acc > baselines["raw_m2"] and positive_families >= 3:
        conclusion = "role_transfer_sourceclean_weak"
    elif beats_avg:
        conclusion = "role_transfer_sourceclean_partial"
    else:
        conclusion = "role_transfer_sourceclean_not_established"
    return {
        "config": {
            "m2_input_dir": config.m2_input_dir,
            "m1_input_dir": config.m1_input_dir,
            "output_dir": config.output_dir,
            "graph_source": config.graph_source,
            "workers": config.workers,
        },
        "report": {
            "leakage_removed": True,
            "graph_source_mode": config.graph_source,
            "graph_edge_coverage": metric("graph_edge_coverage"),
            "source_only_roles_per_heldout_family": {row["heldout_family"]: row["source_only_roles"] for row in evaluable},
            "target_families_evaluated": {row["heldout_family"]: row["target_m2_families"] for row in evaluable},
            "transfer_accuracy_structural_role": role_acc,
            "transfer_accuracy_random": baselines["random"],
            "transfer_accuracy_majority": baselines["majority"],
            "transfer_accuracy_raw_m2": baselines["raw_m2"],
            "transfer_accuracy_coarse": baselines["coarse"],
            "transfer_accuracy_surface_effect": baselines["surface_effect"],
            "transfer_accuracy_m2_label": baselines["m2_label"],
            "transfer_accuracy_graph_role_no_label": baselines["graph_role_no_label"],
            "lift_vs_surface_effect": role_acc - baselines["surface_effect"],
            "lift_vs_raw_m2": role_acc - baselines["raw_m2"],
            "lift_vs_m2_label": role_acc - baselines["m2_label"],
            "lift_vs_no_label_graph": role_acc - baselines["graph_role_no_label"],
            "effect_of_removing_m2_label_features": role_acc - baselines["graph_role_no_label"],
            "positive_lift_families": positive_families,
            "evaluable_heldout_families": len(evaluable),
            "mean_structural_score": float(np.mean([row["assigned_structural_prediction_score"] for row in assignment_rows])) if assignment_rows else 0.0,
            "mean_role_lift_over_best_baseline": mean_lift_best,
            "supports_H2": beats_avg,
            "scientific_conclusion": conclusion,
            "v09b_prototype_tuning_justified": beats_avg and positive_families >= 3,
            "families_with_positive_lift": sorted([row["heldout_family"] for row in evaluable if row["role_lift_over_best_baseline"] > 0]),
            "families_without_positive_lift": sorted([row["heldout_family"] for row in evaluable if row["role_lift_over_best_baseline"] <= 0]),
        },
        "validation": {
            "diagnostic_success": bool(assignment_rows),
            "scientific_conclusion": conclusion,
            "proceed_to_v09b": beats_avg and positive_families >= 3,
        },
    }


def format_report(payload: dict[str, Any]) -> str:
    r = payload["report"]
    return "\n".join(
        [
            "ARC-AGI3 v0.9a-role-transfer-sourceclean",
            f"scientific_conclusion={payload['validation']['scientific_conclusion']}",
            f"leakage_removed={r['leakage_removed']}",
            f"graph_source_mode={r['graph_source_mode']}",
            f"graph_edge_coverage={r['graph_edge_coverage']:.6f}",
            f"transfer_accuracy_structural_role={r['transfer_accuracy_structural_role']:.6f}",
            f"transfer_accuracy_random={r['transfer_accuracy_random']:.6f}",
            f"transfer_accuracy_majority={r['transfer_accuracy_majority']:.6f}",
            f"transfer_accuracy_raw_m2={r['transfer_accuracy_raw_m2']:.6f}",
            f"transfer_accuracy_coarse={r['transfer_accuracy_coarse']:.6f}",
            f"transfer_accuracy_surface_effect={r['transfer_accuracy_surface_effect']:.6f}",
            f"transfer_accuracy_m2_label={r['transfer_accuracy_m2_label']:.6f}",
            f"transfer_accuracy_graph_role_no_label={r['transfer_accuracy_graph_role_no_label']:.6f}",
            f"lift_vs_surface_effect={r['lift_vs_surface_effect']:.6f}",
            f"lift_vs_raw_m2={r['lift_vs_raw_m2']:.6f}",
            f"lift_vs_m2_label={r['lift_vs_m2_label']:.6f}",
            f"lift_vs_no_label_graph={r['lift_vs_no_label_graph']:.6f}",
            f"effect_of_removing_m2_label_features={r['effect_of_removing_m2_label_features']:.6f}",
            f"positive_lift_families={r['positive_lift_families']}",
            f"evaluable_heldout_families={r['evaluable_heldout_families']}",
            f"mean_structural_score={r['mean_structural_score']:.6f}",
            f"mean_role_lift_over_best_baseline={r['mean_role_lift_over_best_baseline']:.6f}",
            f"supports_H2={r['supports_H2']}",
            f"v09b_prototype_tuning_justified={r['v09b_prototype_tuning_justified']}",
            f"families_with_positive_lift={','.join(r['families_with_positive_lift'])}",
            f"families_without_positive_lift={','.join(r['families_without_positive_lift'])}",
        ]
    )
