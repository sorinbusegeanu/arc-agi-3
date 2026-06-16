from __future__ import annotations

import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v6.game_sets import load_game_set_manifest
from v6.role_candidates_v08d import (
    DiscNeighborhood,
    build_discriminative_neighborhoods,
    build_game_family_map,
    load_episode_summaries,
    load_m1_support,
    load_m2_families,
    load_m2_graph_edges,
)
from v6.role_transfer_v09 import _write_parquet, appearance_features, cosine_similarity
from v6.role_transfer_v09a import build_source_only_roles, m2_label_baseline_role, nearest_source_family
from v6.role_transfer_v09b import (
    FamilyContext,
    PrototypeEntry,
    StrategySpec,
    WEIGHT_PROFILES,
    WeightProfile,
    build_strategy_prototypes,
    feature_groups,
    nearest_no_label_role,
    rank_roles_for_target,
    score_source_role,
)


@dataclass(frozen=True)
class RoleTransferV09cConfig:
    m2_input_dir: str = "runs/v6/v07_cd2_extended32_expanded"
    m1_input_dir: str = "runs/v6/v06_cd2_extended32"
    previous_v09b_dir: str = "runs/v6/v09b_role_transfer_refined_sourceclean_extended32"
    output_dir: str = "runs/v6/v09c_transfer_hardened_extended32"
    game_set_manifest: str | None = None
    game_set_name: str | None = None
    split_mode: str = "leave_family_out"
    workers: int = 25
    graph_source: str = "hybrid"
    hardened_success_threshold: float = 0.67
    graph_position_threshold: float = 0.62
    future_option_threshold: float = 0.62


DEFAULT_BEST_STRATEGY = StrategySpec(
    name="top_k_neighbors__k1__weights_default__include_unknown_roles_but_downweight__no_gating",
    prototype_mode="top_k_neighbors",
    weight_profile=WEIGHT_PROFILES[0],
    unknown_mode="include_unknown_roles_but_downweight",
    confidence_mode="no_gating",
    similarity_threshold=0.0,
    margin=0.0,
    top_k=1,
)


def detect_available_memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def choose_parallel_worker_count(requested_workers: int, task_count: int) -> int:
    if requested_workers <= 1 or task_count <= 1:
        return 1

    cpu_limit = os.cpu_count() or 1
    worker_cap = max(1, min(requested_workers, task_count, cpu_limit))
    available_memory = detect_available_memory_bytes()
    if available_memory is None:
        return worker_cap

    # Empirically, extended32 family preparation/evaluation can transiently hold
    # much larger pandas-backed state than the serialized task suggests.
    baseline_worker_bytes = 16 * 1024 * 1024 * 1024
    safe_budget = int(available_memory * 0.15)
    if safe_budget <= 0:
        return 1
    memory_cap = max(1, safe_budget // baseline_worker_bytes)
    return max(1, min(worker_cap, memory_cap))


def run_role_transfer_v09c(config: RoleTransferV09cConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_strategy = load_best_v09b_strategy(Path(config.previous_v09b_dir))
    family_contexts = prepare_family_contexts(config)
    effective_workers = choose_parallel_worker_count(config.workers, len(family_contexts))
    if effective_workers <= 1 or len(family_contexts) <= 1:
        family_results = [_evaluate_hardened_family(context, best_strategy, config) for context in family_contexts]
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = [executor.submit(_evaluate_hardened_family, context, best_strategy, config) for context in family_contexts]
            family_results = [future.result() for future in futures]
    family_results = sorted(family_results, key=lambda item: item["heldout_family"])

    assignment_rows = [row for result in family_results for row in result["assignments"]]
    by_family_rows = [result["summary"] for result in family_results]
    surface_bin_rows = build_surface_bin_rows(assignment_rows)
    challenge_rows = build_challenge_rows(assignment_rows)
    effect_residual_rows = build_effect_residual_rows(assignment_rows)
    failure_rows = [row for row in assignment_rows if int(row["role_hardened_success"]) == 0 or float(row["effect_residual_score"]) <= 0.0]
    previous_v09b = json.loads((Path(config.previous_v09b_dir) / "v09b_report.json").read_text(encoding="utf-8"))
    payload = build_report_payload(config, previous_v09b, best_strategy, assignment_rows, by_family_rows, surface_bin_rows, challenge_rows, effect_residual_rows)
    write_outputs(output_dir, payload, assignment_rows, surface_bin_rows, challenge_rows, by_family_rows, effect_residual_rows, failure_rows)
    return payload


def prepare_family_contexts(config: RoleTransferV09cConfig) -> list[FamilyContext]:
    m2_families = load_m2_families(Path(config.m2_input_dir))
    game_set = load_game_set_manifest(
        manifest_path=config.game_set_manifest,
        game_set_name=config.game_set_name,
        fallback_games=tuple(sorted({game for family in m2_families for game in family.games_present})),
    )
    m1_support = load_m1_support(Path(config.m1_input_dir))
    episode_summaries = load_episode_summaries(Path(config.m1_input_dir))
    m2_graph_edges = load_m2_graph_edges(Path(config.m2_input_dir))
    selected_games = set(game_set.games) if game_set.games else {game for family in m2_families for game in family.games_present}
    m2_families = [family for family in m2_families if selected_games.intersection(family.games_present)]
    m1_support = {key: value for key, value in m1_support.items() if value.game_id in selected_games}
    game_family_map = build_game_family_map(game_set, tuple(sorted(selected_games)))
    full_neighborhoods, _ = build_discriminative_neighborhoods(
        m2_families,
        m1_support,
        game_family_map,
        graph_source=config.graph_source,
        ablation="none",
        m2_graph_edges=m2_graph_edges,
        episode_summaries=episode_summaries,
    )
    full_no_label_neighborhoods, _ = build_discriminative_neighborhoods(
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
            full_neighborhoods,
            full_no_label_neighborhoods,
        )
        for family_name in sorted(game_set.families)
    ]
    effective_workers = choose_parallel_worker_count(config.workers, len(tasks))
    if effective_workers <= 1 or len(tasks) <= 1:
        return [_prepare_family_context(*task) for task in tasks]
    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        futures = [executor.submit(_prepare_family_context, *task) for task in tasks]
        return sorted([future.result() for future in futures], key=lambda item: item.heldout_family)


def _prepare_family_context(
    heldout_family: str,
    heldout_games: tuple[str, ...],
    m2_families: list[Any],
    m1_support: dict[str, Any],
    game_family_map: dict[str, str],
    m2_graph_edges: list[dict[str, Any]],
    episode_summaries: list[dict[str, Any]],
    graph_source: str,
    full_neighborhoods: dict[str, DiscNeighborhood],
    full_no_label_neighborhoods: dict[str, DiscNeighborhood],
) -> FamilyContext:
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
    source_no_label_neighborhoods, _ = build_discriminative_neighborhoods(
        source_families,
        source_support,
        game_family_map,
        graph_source=graph_source,
        ablation="no_m2_labels",
        m2_graph_edges=m2_graph_edges,
        episode_summaries=episode_summaries,
    )
    return FamilyContext(
        heldout_family=heldout_family,
        heldout_games=heldout_games,
        source_neighborhoods=source_neighborhoods,
        source_roles=build_source_only_roles(source_families, source_neighborhoods),
        source_no_label_roles=build_source_only_roles(source_families, source_no_label_neighborhoods),
        target_families=tuple(sorted(target_families, key=lambda item: item.family_id)),
        full_neighborhoods=full_neighborhoods,
        full_no_label_neighborhoods=full_no_label_neighborhoods,
        graph_source_used=graph_diag.graph_source_used,
        graph_edge_coverage=float(graph_diag.graph_edge_coverage),
    )


def load_best_v09b_strategy(previous_v09b_dir: Path) -> StrategySpec:
    path = previous_v09b_dir / "v09b_report.json"
    if not path.exists():
        return DEFAULT_BEST_STRATEGY
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload["report"]["best_strategy"]
    weight_profile = next((item for item in WEIGHT_PROFILES if item.name == row.get("weight_profile")), WEIGHT_PROFILES[0])
    return StrategySpec(
        name=str(row.get("strategy_name", DEFAULT_BEST_STRATEGY.name)),
        prototype_mode=str(row.get("prototype_mode", DEFAULT_BEST_STRATEGY.prototype_mode)),
        weight_profile=weight_profile,
        unknown_mode=str(row.get("unknown_mode", DEFAULT_BEST_STRATEGY.unknown_mode)),
        confidence_mode=str(row.get("confidence_mode", DEFAULT_BEST_STRATEGY.confidence_mode)),
        similarity_threshold=float(row.get("similarity_threshold", 0.0)),
        margin=float(row.get("margin", 0.0)),
        top_k=int(row.get("top_k", 1)),
    )


def _evaluate_hardened_family(context: FamilyContext, strategy: StrategySpec, config: RoleTransferV09cConfig) -> dict[str, Any]:
    source_prototypes = build_strategy_prototypes(strategy, context.source_roles, context.source_neighborhoods)
    thresholds = calibrated_thresholds(context, source_prototypes, strategy, config)
    assignments = []
    for family in context.target_families:
        target = context.full_neighborhoods.get(family.family_id)
        target_no_label = context.full_no_label_neighborhoods.get(family.family_id)
        if target is None or target_no_label is None or family.support_count < 3 or not source_prototypes:
            continue

        ranked = rank_roles_for_target(
            target,
            source_prototypes,
            mode="weighted_structural",
            top_k=strategy.top_k,
            weights=strategy.weight_profile.as_dict(),
            strategy=strategy,
        )
        best_role_id = ranked[0][0] if ranked else ""
        surface_role_id = nearest_surface_prototype_id(target, source_prototypes)
        raw_family_id = nearest_source_family(target, context.source_neighborhoods, mode="raw_m2")
        no_label_role_id, _ = nearest_no_label_role(target_no_label, context.source_no_label_roles, strategy.weight_profile.as_dict())
        m2_label_role_id = m2_label_baseline_role(target, context.source_neighborhoods, context.source_roles)
        if not best_role_id or not surface_role_id or not raw_family_id or not no_label_role_id:
            continue

        role_hardened = hardened_scores_for_prototype(target, source_prototypes[best_role_id], top_k=strategy.top_k)
        surface_hardened = hardened_scores_for_prototype(target, source_prototypes[surface_role_id], top_k=0)
        raw_hardened = hardened_scores_for_record(target, context.source_neighborhoods[raw_family_id])
        no_label_hardened = hardened_scores_for_role(target, context.source_no_label_roles[no_label_role_id])
        m2_label_hardened = hardened_scores_for_role(target, context.source_roles[m2_label_role_id]) if m2_label_role_id else zero_hardened_scores()

        role_surface_similarity = cosine_similarity(appearance_features(target), source_prototypes[best_role_id].appearance_vectors[0])
        baseline_surface_similarity = cosine_similarity(appearance_features(target), source_prototypes[surface_role_id].appearance_vectors[0])
        raw_surface_similarity = cosine_similarity(appearance_features(target), appearance_features(context.source_neighborhoods[raw_family_id]))
        bin_name = surface_similarity_bin(baseline_surface_similarity)
        dissimilar_surface = int(max_surface_similarity_to_any_source(target, source_prototypes) < 0.85)

        row = {
            "heldout_family": context.heldout_family,
            "target_family_id": family.family_id,
            "target_games": list(get_games(family)),
            "target_game_families": list(get_game_families(target)),
            "assigned_role_id": best_role_id,
            "surface_role_id": surface_role_id,
            "raw_m2_family_id": raw_family_id,
            "graph_role_no_label_role_id": no_label_role_id,
            "m2_label_role_id": m2_label_role_id or "",
            "surface_similarity_bin": bin_name,
            "role_surface_similarity": float(role_surface_similarity),
            "surface_baseline_similarity": float(baseline_surface_similarity),
            "raw_surface_similarity": float(raw_surface_similarity),
            "role_hardened_score": role_hardened["hardened_score"],
            "surface_hardened_score": surface_hardened["hardened_score"],
            "raw_m2_hardened_score": raw_hardened["hardened_score"],
            "graph_role_no_label_hardened_score": no_label_hardened["hardened_score"],
            "m2_label_hardened_score": m2_label_hardened["hardened_score"],
            "role_hardened_success": int(role_hardened["hardened_score"] >= thresholds["hardened"]),
            "surface_hardened_success": int(surface_hardened["hardened_score"] >= thresholds["hardened"]),
            "raw_m2_hardened_success": int(raw_hardened["hardened_score"] >= thresholds["hardened"]),
            "graph_role_no_label_hardened_success": int(no_label_hardened["hardened_score"] >= thresholds["hardened"]),
            "graph_position_role_score": role_hardened["graph_position_score"],
            "graph_position_surface_score": surface_hardened["graph_position_score"],
            "graph_position_prediction_success": int(role_hardened["graph_position_score"] >= thresholds["graph_position"]),
            "graph_position_surface_success": int(surface_hardened["graph_position_score"] >= thresholds["graph_position"]),
            "future_option_role_score": role_hardened["future_option_score"],
            "future_option_surface_score": surface_hardened["future_option_score"],
            "future_option_prediction_success": int(role_hardened["future_option_score"] >= thresholds["future_option"]),
            "future_option_surface_success": int(surface_hardened["future_option_score"] >= thresholds["future_option"]),
            "effect_residual_score": float(role_hardened["hardened_score"] - surface_hardened["hardened_score"]),
            "same_effect_different_role_case": int(select_same_effect_different_role_case(role_hardened, surface_hardened, baseline_surface_similarity)),
            "same_role_different_effect_case": int(select_same_role_different_effect_case(role_hardened, role_surface_similarity, thresholds["hardened"])),
            "dissimilar_family_transfer_case": int(dissimilar_surface),
            "same_effect_different_role_success": int(role_hardened["challenge_core_score"] >= thresholds["challenge_core"]),
            "same_effect_different_role_surface_success": int(surface_hardened["challenge_core_score"] >= thresholds["challenge_core"]),
            "same_role_different_effect_success": int(role_hardened["hardened_score"] >= thresholds["hardened"]),
            "same_role_different_effect_surface_success": int(surface_hardened["hardened_score"] >= thresholds["hardened"]),
            "graph_source_used": context.graph_source_used,
            "graph_edge_coverage": context.graph_edge_coverage,
            "hardened_threshold": thresholds["hardened"],
            "graph_position_threshold": thresholds["graph_position"],
            "future_option_threshold": thresholds["future_option"],
        }
        assignments.append(row)

    summary = build_family_summary(context, assignments)
    return {"heldout_family": context.heldout_family, "assignments": assignments, "summary": summary}


def hardened_scores_for_role(target: DiscNeighborhood, role_entry: dict[str, Any]) -> dict[str, float]:
    source = {
        "graph_position": subset_prefixed(role_entry["all_features"], "directional:"),
        "future_option": subset_prefixed(role_entry["all_features"], "future:"),
        "local_motif": subset_prefixed(role_entry["all_features"], "motif:"),
        "coarse": subset_prefixed(role_entry["all_features"], "coarse:"),
        "effect": subset_prefixed(role_entry["all_features"], "effect:"),
    }
    return hardened_scores(target, source)


def hardened_scores_for_prototype(target: DiscNeighborhood, prototype: PrototypeEntry, top_k: int) -> dict[str, float]:
    score_rows = []
    for group_vector in prototype.group_vectors:
        source = {
            "graph_position": graph_position_features_from_groups(group_vector),
            "future_option": group_vector["future_option"],
            "local_motif": group_vector["local_motif"],
            "coarse": group_vector["coarse"],
            "effect": group_vector["effect"],
        }
        score_rows.append(hardened_scores(target, source))
    if not score_rows:
        return zero_hardened_scores()
    if prototype.prototype_mode == "top_k_neighbors":
        k = max(1, min(top_k or len(score_rows), len(score_rows)))
        score_rows = sorted(score_rows, key=lambda row: row["hardened_score"], reverse=True)[:k]
    else:
        score_rows = [max(score_rows, key=lambda row: row["hardened_score"])]
    return {key: float(np.mean([row[key] for row in score_rows])) for key in score_rows[0]}


def hardened_scores_for_record(target: DiscNeighborhood, source: DiscNeighborhood) -> dict[str, float]:
    source_groups = {
        "graph_position": graph_position_features(source),
        "future_option": future_option_behavior_features(source),
        "local_motif": local_graph_motif_features(source),
        "coarse": coarse_transfer_features(source),
        "effect": effect_only_features(source),
    }
    return hardened_scores(target, source_groups)


def hardened_scores(target: DiscNeighborhood, source_groups: dict[str, dict[str, float]]) -> dict[str, float]:
    target_groups = {
        "graph_position": graph_position_features(target),
        "future_option": future_option_behavior_features(target),
        "local_motif": local_graph_motif_features(target),
        "coarse": coarse_transfer_features(target),
        "effect": effect_only_features(target),
    }
    graph_score = cosine_similarity(target_groups["graph_position"], source_groups["graph_position"])
    future_score = cosine_similarity(target_groups["future_option"], source_groups["future_option"])
    motif_score = cosine_similarity(target_groups["local_motif"], source_groups["local_motif"])
    coarse_score = cosine_similarity(target_groups["coarse"], source_groups["coarse"])
    effect_score = cosine_similarity(target_groups["effect"], source_groups["effect"])
    challenge_core = 0.45 * graph_score + 0.40 * future_score + 0.15 * motif_score
    hardened_score = 0.40 * future_score + 0.35 * graph_score + 0.15 * motif_score + 0.10 * coarse_score - 0.10 * max(0.0, effect_score - challenge_core)
    return {
        "hardened_score": float(hardened_score),
        "graph_position_score": float(graph_score),
        "future_option_score": float(future_score),
        "local_motif_score": float(motif_score),
        "coarse_score": float(coarse_score),
        "effect_score": float(effect_score),
        "challenge_core_score": float(challenge_core),
    }


def zero_hardened_scores() -> dict[str, float]:
    return {
        "hardened_score": 0.0,
        "graph_position_score": 0.0,
        "future_option_score": 0.0,
        "local_motif_score": 0.0,
        "coarse_score": 0.0,
        "effect_score": 0.0,
        "challenge_core_score": 0.0,
    }


def calibrated_thresholds(
    context: FamilyContext,
    source_prototypes: dict[str, PrototypeEntry],
    strategy: StrategySpec,
    config: RoleTransferV09cConfig,
) -> dict[str, float]:
    hardened_scores = []
    graph_scores = []
    future_scores = []
    challenge_scores = []
    for role_id, prototype in source_prototypes.items():
        for family_id in prototype.member_family_ids:
            record = context.source_neighborhoods.get(family_id)
            if record is None:
                continue
            scores = hardened_scores_for_prototype(record, prototype, top_k=strategy.top_k)
            hardened_scores.append(scores["hardened_score"])
            graph_scores.append(scores["graph_position_score"])
            future_scores.append(scores["future_option_score"])
            challenge_scores.append(scores["challenge_core_score"])

    def q(values: list[float], fallback: float) -> float:
        if not values:
            return fallback
        return float(np.quantile(np.asarray(values, dtype=float), 0.25))

    return {
        "hardened": q(hardened_scores, config.hardened_success_threshold),
        "graph_position": q(graph_scores, config.graph_position_threshold),
        "future_option": q(future_scores, config.future_option_threshold),
        "challenge_core": q(challenge_scores, config.hardened_success_threshold),
    }


def graph_position_features(record: DiscNeighborhood) -> dict[str, float]:
    directional = record.directional_features
    incoming = record.incoming_edge_profile
    outgoing = record.outgoing_edge_profile
    incoming_mass = float(sum(float(value or 0.0) for value in incoming.values()))
    outgoing_mass = float(sum(float(value or 0.0) for value in outgoing.values()))
    return {
        "predecessor_count": float(directional.get("predecessor_count", 0.0)),
        "successor_count": float(directional.get("successor_count", 0.0)),
        "source_like_score": float(directional.get("source_like_score", 0.0)),
        "sink_like_score": float(directional.get("sink_like_score", 0.0)),
        "bridge_like_score": float(directional.get("bridge_like_score", 0.0)),
        "bottleneck_like_score": float(directional.get("bottleneck_like_score", 0.0)),
        "branch_in_score": float(directional.get("branch_in_score", 0.0)),
        "branch_out_score": float(directional.get("branch_out_score", 0.0)),
        "loop_score": float(directional.get("loop_score", 0.0)),
        "directional_asymmetry_score": float(directional.get("directional_asymmetry_score", 0.0)),
        "incoming_mass": incoming_mass,
        "outgoing_mass": outgoing_mass,
        "profile_asymmetry": float(outgoing_mass - incoming_mass),
    }


def graph_position_features_from_groups(groups: dict[str, dict[str, float]]) -> dict[str, float]:
    directional = groups["directional"]
    return {
        "predecessor_count": float(directional.get("predecessor_count", 0.0)),
        "successor_count": float(directional.get("successor_count", 0.0)),
        "source_like_score": float(directional.get("source_like_score", 0.0)),
        "sink_like_score": float(directional.get("sink_like_score", 0.0)),
        "bridge_like_score": float(directional.get("bridge_like_score", 0.0)),
        "bottleneck_like_score": float(directional.get("bottleneck_like_score", 0.0)),
        "branch_in_score": float(directional.get("branch_in_score", 0.0)),
        "branch_out_score": float(directional.get("branch_out_score", 0.0)),
        "loop_score": float(directional.get("loop_score", 0.0)),
        "directional_asymmetry_score": float(directional.get("directional_asymmetry_score", 0.0)),
        "incoming_mass": 0.0,
        "outgoing_mass": 0.0,
        "profile_asymmetry": 0.0,
    }


def future_option_behavior_features(record: DiscNeighborhood) -> dict[str, float]:
    future = record.future_option_features
    return {
        "reachable_before_rate": float(future.get("reachable_before_rate", 0.0)),
        "reachable_after_rate": float(future.get("reachable_after_rate", 0.0)),
        "enable_score": float(future.get("enable_score", 0.0)),
        "block_score": float(future.get("block_score", 0.0)),
        "preserve_score": float(future.get("preserve_score", 0.0)),
        "terminate_score": float(future.get("terminate_score", 0.0)),
        "reversibility_score": float(future.get("reversibility_score", 0.0)),
        "reachable_delta": float(future.get("reachable_after_rate", 0.0) - future.get("reachable_before_rate", 0.0)),
    }


def local_graph_motif_features(record: DiscNeighborhood) -> dict[str, float]:
    motif = record.local_motif_features
    return {
        "cross_game_family_presence": float(motif.get("cross_game_family_presence", 0.0)),
        "motif_entropy": float(motif.get("motif_entropy", 0.0)),
        "local_branching_score": float(motif.get("local_branching_score", 0.0)),
        "local_loop_score": float(motif.get("local_loop_score", 0.0)),
        f"motif::{record.dominant_motif_candidate}": 1.0,
    }


def coarse_transfer_features(record: DiscNeighborhood) -> dict[str, float]:
    coarse = record.coarse_features
    return {
        "family_coherence": float(record.family_coherence),
        "support_count": float(record.support_count),
        "mean_prediction_accuracy": float(record.mean_prediction_accuracy),
        "mean_context_lift": float(record.mean_context_lift),
        **{str(key): float(value) for key, value in coarse.items()},
    }


def effect_only_features(record: DiscNeighborhood) -> dict[str, float]:
    effect = record.temporal_effect_features
    return {
        "no_change_rate": float(effect.get("no_change_rate", 0.0)),
        "position_change_rate": float(effect.get("position_change_rate", 0.0)),
        "discontinuous_position_change_rate": float(effect.get("discontinuous_position_change_rate", 0.0)),
        "terminal_rate": float(effect.get("terminal_rate", 0.0)),
        "multi_cell_change_rate": float(effect.get("multi_cell_change_rate", 0.0)),
        "coverage_change_rate_if_derivable": float(effect.get("coverage_change_rate_if_derivable", 0.0)),
        "repeated_toggle_like_rate": float(effect.get("repeated_toggle_like_rate", 0.0)),
    }


def subset_prefixed(vector: dict[str, float], prefix: str) -> dict[str, float]:
    return {key[len(prefix):]: float(value) for key, value in vector.items() if key.startswith(prefix)}


def nearest_surface_role_id(target: DiscNeighborhood, source_roles: dict[str, dict[str, Any]]) -> str:
    best_role = ""
    best_score = -1.0
    target_vector = appearance_features(target)
    for role_id, role_entry in sorted(source_roles.items()):
        score = cosine_similarity(target_vector, role_entry["appearance_features"])
        if score > best_score:
            best_role = role_id
            best_score = score
    return best_role


def nearest_surface_prototype_id(target: DiscNeighborhood, source_prototypes: dict[str, PrototypeEntry]) -> str:
    best_role = ""
    best_score = -1.0
    target_vector = appearance_features(target)
    for role_id, prototype in sorted(source_prototypes.items()):
        score = max(cosine_similarity(target_vector, vector) for vector in prototype.appearance_vectors) if prototype.appearance_vectors else -1.0
        if score > best_score:
            best_role = role_id
            best_score = score
    return best_role


def max_surface_similarity_to_any_source(target: DiscNeighborhood, source_prototypes: dict[str, PrototypeEntry]) -> float:
    if not source_prototypes:
        return 0.0
    target_vector = appearance_features(target)
    return float(max(cosine_similarity(target_vector, vector) for prototype in source_prototypes.values() for vector in prototype.appearance_vectors))


def surface_similarity_bin(similarity: float) -> str:
    if similarity < 0.60:
        return "low_surface_similarity"
    if similarity < 0.85:
        return "medium_surface_similarity"
    return "high_surface_similarity"


def select_same_effect_different_role_case(role_scores: dict[str, float], surface_scores: dict[str, float], surface_similarity: float) -> bool:
    return bool(surface_similarity >= 0.80 and abs(role_scores["challenge_core_score"] - surface_scores["challenge_core_score"]) >= 0.04 and surface_scores["effect_score"] >= 0.80)


def select_same_role_different_effect_case(role_scores: dict[str, float], role_surface_similarity: float, hardened_threshold: float) -> bool:
    return bool(role_scores["challenge_core_score"] >= hardened_threshold and role_surface_similarity < 0.90 and role_scores["effect_score"] < role_scores["challenge_core_score"])


def get_games(record: Any) -> tuple[str, ...]:
    return tuple(sorted(getattr(record, "games_present", getattr(record, "game_ids", ()))))


def get_game_families(record: Any) -> tuple[str, ...]:
    return tuple(sorted(getattr(record, "game_families_present", getattr(record, "game_family_ids", ()))))


def build_family_summary(context: FamilyContext, assignments: list[dict[str, Any]]) -> dict[str, Any]:
    if not assignments:
        return {
            "heldout_family": context.heldout_family,
            "target_m2_families": 0,
            "transfer_accuracy_role_hardened": 0.0,
            "transfer_accuracy_surface_effect_hardened": 0.0,
            "transfer_accuracy_raw_m2_hardened": 0.0,
            "transfer_accuracy_graph_role_no_label_hardened": 0.0,
            "lift_vs_surface_effect_hardened": 0.0,
            "lift_vs_raw_m2_hardened": 0.0,
            "lift_vs_no_label_graph_hardened": 0.0,
            "mean_effect_residual_score": 0.0,
        }
    role_acc = float(np.mean([row["role_hardened_success"] for row in assignments]))
    surface_acc = float(np.mean([row["surface_hardened_success"] for row in assignments]))
    raw_acc = float(np.mean([row["raw_m2_hardened_success"] for row in assignments]))
    no_label_acc = float(np.mean([row["graph_role_no_label_hardened_success"] for row in assignments]))
    return {
        "heldout_family": context.heldout_family,
        "target_m2_families": len(assignments),
        "transfer_accuracy_role_hardened": role_acc,
        "transfer_accuracy_surface_effect_hardened": surface_acc,
        "transfer_accuracy_raw_m2_hardened": raw_acc,
        "transfer_accuracy_graph_role_no_label_hardened": no_label_acc,
        "lift_vs_surface_effect_hardened": role_acc - surface_acc,
        "lift_vs_raw_m2_hardened": role_acc - raw_acc,
        "lift_vs_no_label_graph_hardened": role_acc - no_label_acc,
        "positive_lift_family_hardened": int(role_acc > surface_acc),
        "graph_position_prediction_accuracy": float(np.mean([row["graph_position_prediction_success"] for row in assignments])),
        "future_option_prediction_accuracy": float(np.mean([row["future_option_prediction_success"] for row in assignments])),
        "mean_effect_residual_score": float(np.mean([row["effect_residual_score"] for row in assignments])),
    }


def build_surface_bin_rows(assignment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for bin_name in ("low_surface_similarity", "medium_surface_similarity", "high_surface_similarity"):
        subset = [row for row in assignment_rows if row["surface_similarity_bin"] == bin_name]
        rows.append(
            {
                "surface_similarity_bin": bin_name,
                "count": len(subset),
                "role_accuracy": float(np.mean([row["role_hardened_success"] for row in subset])) if subset else 0.0,
                "surface_accuracy": float(np.mean([row["surface_hardened_success"] for row in subset])) if subset else 0.0,
                "lift_vs_surface": float(np.mean([row["role_hardened_success"] - row["surface_hardened_success"] for row in subset])) if subset else 0.0,
            }
        )
    return rows


def build_challenge_rows(assignment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = (
        ("same_effect_different_role", "same_effect_different_role_case", "same_effect_different_role_success", "same_effect_different_role_surface_success"),
        ("same_role_different_effect", "same_role_different_effect_case", "same_role_different_effect_success", "same_role_different_effect_surface_success"),
        ("dissimilar_family_transfer", "dissimilar_family_transfer_case", "role_hardened_success", "surface_hardened_success"),
        ("graph_position_prediction", None, "graph_position_prediction_success", "graph_position_surface_success"),
        ("future_option_prediction", None, "future_option_prediction_success", "future_option_surface_success"),
    )
    rows = []
    for name, flag, role_key, surface_key in specs:
        subset = assignment_rows if flag is None else [row for row in assignment_rows if row[flag]]
        rows.append(
            {
                "challenge_mode": name,
                "count": len(subset),
                "role_accuracy": float(np.mean([row[role_key] for row in subset])) if subset else 0.0,
                "surface_accuracy": float(np.mean([row[surface_key] for row in subset])) if subset else 0.0,
                "lift_vs_surface": float(np.mean([row[role_key] - row[surface_key] for row in subset])) if subset else 0.0,
            }
        )
    return rows


def build_effect_residual_rows(assignment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "heldout_family": row["heldout_family"],
            "target_family_id": row["target_family_id"],
            "effect_residual_score": row["effect_residual_score"],
            "role_hardened_score": row["role_hardened_score"],
            "surface_hardened_score": row["surface_hardened_score"],
        }
        for row in assignment_rows
    ]


def build_report_payload(
    config: RoleTransferV09cConfig,
    previous_v09b: dict[str, Any],
    best_strategy: StrategySpec,
    assignment_rows: list[dict[str, Any]],
    by_family_rows: list[dict[str, Any]],
    surface_bin_rows: list[dict[str, Any]],
    challenge_rows: list[dict[str, Any]],
    effect_residual_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluable = [row for row in by_family_rows if row["target_m2_families"] > 0]
    metric = lambda key: float(np.mean([row[key] for row in evaluable])) if evaluable else 0.0
    challenge_lookup = {row["challenge_mode"]: row for row in challenge_rows}
    surface_bins = {row["surface_similarity_bin"]: row for row in surface_bin_rows}
    positive_lift_families = sum(int(row["positive_lift_family_hardened"]) for row in evaluable)
    mean_effect_residual = float(np.mean([row["effect_residual_score"] for row in effect_residual_rows])) if effect_residual_rows else 0.0
    hard_modes_beating_surface = sum(1 for row in challenge_rows if row["lift_vs_surface"] > 0)

    conclusion = "hardened_transfer_not_established"
    if (
        (metric("transfer_accuracy_role_hardened") - metric("transfer_accuracy_surface_effect_hardened")) >= 0.10
        and positive_lift_families >= 10
        and hard_modes_beating_surface >= 5
        and surface_bins["low_surface_similarity"]["lift_vs_surface"] > 0
        and surface_bins["medium_surface_similarity"]["lift_vs_surface"] > 0
        and challenge_lookup["graph_position_prediction"]["lift_vs_surface"] >= 0.10
        and challenge_lookup["future_option_prediction"]["lift_vs_surface"] >= 0.10
    ):
        conclusion = "hardened_transfer_very_strong"
    elif (
        (metric("transfer_accuracy_role_hardened") - metric("transfer_accuracy_surface_effect_hardened")) >= 0.05
        and positive_lift_families >= 8
        and hard_modes_beating_surface >= 3
        and surface_bins["low_surface_similarity"]["role_accuracy"] > surface_bins["low_surface_similarity"]["surface_accuracy"]
        and mean_effect_residual > 0
    ):
        conclusion = "hardened_transfer_strong"
    elif (
        (metric("transfer_accuracy_role_hardened") - metric("transfer_accuracy_surface_effect_hardened")) > 0
        and positive_lift_families >= 6
        and hard_modes_beating_surface >= 1
    ):
        conclusion = "hardened_transfer_weak"

    report = {
        "v09b_baseline_summary": previous_v09b["report"]["best_strategy"],
        "surface_effect_controlled": True,
        "best_v09b_strategy_used": {
            "strategy_name": best_strategy.name,
            "prototype_mode": best_strategy.prototype_mode,
            "top_k": best_strategy.top_k,
            "weight_profile": best_strategy.weight_profile.name,
            "unknown_mode": best_strategy.unknown_mode,
            "confidence_mode": best_strategy.confidence_mode,
        },
        "transfer_accuracy_role_hardened": metric("transfer_accuracy_role_hardened"),
        "transfer_accuracy_surface_effect_hardened": metric("transfer_accuracy_surface_effect_hardened"),
        "transfer_accuracy_raw_m2_hardened": metric("transfer_accuracy_raw_m2_hardened"),
        "transfer_accuracy_graph_role_no_label_hardened": metric("transfer_accuracy_graph_role_no_label_hardened"),
        "lift_vs_surface_effect_hardened": metric("transfer_accuracy_role_hardened") - metric("transfer_accuracy_surface_effect_hardened"),
        "lift_vs_raw_m2_hardened": metric("transfer_accuracy_role_hardened") - metric("transfer_accuracy_raw_m2_hardened"),
        "lift_vs_no_label_graph_hardened": metric("transfer_accuracy_role_hardened") - metric("transfer_accuracy_graph_role_no_label_hardened"),
        "positive_lift_families_hardened": positive_lift_families,
        "low_surface_bin_role_accuracy": surface_bins["low_surface_similarity"]["role_accuracy"],
        "low_surface_bin_surface_accuracy": surface_bins["low_surface_similarity"]["surface_accuracy"],
        "medium_surface_bin_role_accuracy": surface_bins["medium_surface_similarity"]["role_accuracy"],
        "medium_surface_bin_surface_accuracy": surface_bins["medium_surface_similarity"]["surface_accuracy"],
        "high_surface_bin_role_accuracy": surface_bins["high_surface_similarity"]["role_accuracy"],
        "high_surface_bin_surface_accuracy": surface_bins["high_surface_similarity"]["surface_accuracy"],
        "same_effect_different_role_accuracy": challenge_lookup["same_effect_different_role"]["role_accuracy"],
        "same_effect_different_role_surface_accuracy": challenge_lookup["same_effect_different_role"]["surface_accuracy"],
        "same_role_different_effect_accuracy": challenge_lookup["same_role_different_effect"]["role_accuracy"],
        "same_role_different_effect_surface_accuracy": challenge_lookup["same_role_different_effect"]["surface_accuracy"],
        "dissimilar_family_transfer_accuracy": challenge_lookup["dissimilar_family_transfer"]["role_accuracy"],
        "dissimilar_family_transfer_surface_accuracy": challenge_lookup["dissimilar_family_transfer"]["surface_accuracy"],
        "graph_position_prediction_accuracy": challenge_lookup["graph_position_prediction"]["role_accuracy"],
        "graph_position_prediction_surface_accuracy": challenge_lookup["graph_position_prediction"]["surface_accuracy"],
        "future_option_prediction_accuracy": challenge_lookup["future_option_prediction"]["role_accuracy"],
        "future_option_prediction_surface_accuracy": challenge_lookup["future_option_prediction"]["surface_accuracy"],
        "mean_effect_residual_score": mean_effect_residual,
        "hard_challenge_modes_beating_surface": hard_modes_beating_surface,
        "scientific_conclusion": conclusion,
        "supports_H2": conclusion != "hardened_transfer_not_established",
        "v10_gate_cleared": conclusion in {"hardened_transfer_strong", "hardened_transfer_very_strong"},
        "recommend_extended64_targeted_games": conclusion != "hardened_transfer_very_strong",
    }
    return {
        "config": {
            "m2_input_dir": config.m2_input_dir,
            "m1_input_dir": config.m1_input_dir,
            "previous_v09b_dir": config.previous_v09b_dir,
            "output_dir": config.output_dir,
            "split_mode": config.split_mode,
            "workers": config.workers,
            "graph_source": config.graph_source,
        },
        "report": report,
        "validation": {
            "diagnostic_success": bool(assignment_rows),
            "scientific_conclusion": conclusion,
            "proceed_to_v10": report["v10_gate_cleared"],
        },
    }


def write_outputs(
    output_dir: Path,
    payload: dict[str, Any],
    assignment_rows: list[dict[str, Any]],
    surface_bin_rows: list[dict[str, Any]],
    challenge_rows: list[dict[str, Any]],
    by_family_rows: list[dict[str, Any]],
    effect_residual_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> None:
    _write_parquet(output_dir / "v09c_hardened_assignments.parquet", assignment_rows)
    _write_parquet(output_dir / "v09c_surface_bins.parquet", surface_bin_rows)
    _write_parquet(output_dir / "v09c_challenge_results.parquet", challenge_rows)
    _write_parquet(output_dir / "v09c_transfer_by_family.parquet", by_family_rows)
    _write_parquet(output_dir / "v09c_effect_residuals.parquet", effect_residual_rows)
    _write_parquet(output_dir / "v09c_failure_cases.parquet", failure_rows)
    (output_dir / "v09c_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "v09c_report.txt").write_text(format_report(payload, surface_bin_rows, challenge_rows), encoding="utf-8")


def format_report(payload: dict[str, Any], surface_bin_rows: list[dict[str, Any]], challenge_rows: list[dict[str, Any]]) -> str:
    report = payload["report"]
    base = report["v09b_baseline_summary"]
    bin_lookup = {row["surface_similarity_bin"]: row for row in surface_bin_rows}
    challenge_lookup = {row["challenge_mode"]: row for row in challenge_rows}
    return "\n".join(
        [
            "ARC-AGI3 v0.9c transfer target hardening / surface-effect control",
            "",
            "1. v0.9b baseline summary",
            f"strategy_name={base['strategy_name']}",
            f"scientific_conclusion={base['scientific_conclusion']}",
            f"transfer_accuracy_structural_role={base['transfer_accuracy_structural_role']:.6f}",
            f"transfer_accuracy_surface_effect={base['transfer_accuracy_surface_effect']:.6f}",
            f"lift_vs_surface_effect={base['lift_vs_surface_effect']:.6f}",
            "",
            "2. Whether surface/effect baseline was controlled",
            f"surface_effect_controlled={report['surface_effect_controlled']}",
            f"transfer_accuracy_role_hardened={report['transfer_accuracy_role_hardened']:.6f}",
            f"transfer_accuracy_surface_effect_hardened={report['transfer_accuracy_surface_effect_hardened']:.6f}",
            f"lift_vs_surface_effect_hardened={report['lift_vs_surface_effect_hardened']:.6f}",
            "",
            "3. Results by surface-similarity bin",
            f"low_surface_similarity=role:{bin_lookup['low_surface_similarity']['role_accuracy']:.6f},surface:{bin_lookup['low_surface_similarity']['surface_accuracy']:.6f},lift:{bin_lookup['low_surface_similarity']['lift_vs_surface']:.6f}",
            f"medium_surface_similarity=role:{bin_lookup['medium_surface_similarity']['role_accuracy']:.6f},surface:{bin_lookup['medium_surface_similarity']['surface_accuracy']:.6f},lift:{bin_lookup['medium_surface_similarity']['lift_vs_surface']:.6f}",
            f"high_surface_similarity=role:{bin_lookup['high_surface_similarity']['role_accuracy']:.6f},surface:{bin_lookup['high_surface_similarity']['surface_accuracy']:.6f},lift:{bin_lookup['high_surface_similarity']['lift_vs_surface']:.6f}",
            "",
            "4. Results for same-effect/different-role challenge",
            f"role_accuracy={challenge_lookup['same_effect_different_role']['role_accuracy']:.6f}",
            f"surface_accuracy={challenge_lookup['same_effect_different_role']['surface_accuracy']:.6f}",
            f"lift_vs_surface={challenge_lookup['same_effect_different_role']['lift_vs_surface']:.6f}",
            "",
            "5. Results for same-role/different-effect challenge",
            f"role_accuracy={challenge_lookup['same_role_different_effect']['role_accuracy']:.6f}",
            f"surface_accuracy={challenge_lookup['same_role_different_effect']['surface_accuracy']:.6f}",
            f"lift_vs_surface={challenge_lookup['same_role_different_effect']['lift_vs_surface']:.6f}",
            "",
            "6. Results for dissimilar-family transfer",
            f"role_accuracy={challenge_lookup['dissimilar_family_transfer']['role_accuracy']:.6f}",
            f"surface_accuracy={challenge_lookup['dissimilar_family_transfer']['surface_accuracy']:.6f}",
            f"lift_vs_surface={challenge_lookup['dissimilar_family_transfer']['lift_vs_surface']:.6f}",
            "",
            "7. Graph-position prediction result",
            f"role_accuracy={challenge_lookup['graph_position_prediction']['role_accuracy']:.6f}",
            f"surface_accuracy={challenge_lookup['graph_position_prediction']['surface_accuracy']:.6f}",
            f"lift_vs_surface={challenge_lookup['graph_position_prediction']['lift_vs_surface']:.6f}",
            "",
            "8. Future-option prediction result",
            f"role_accuracy={challenge_lookup['future_option_prediction']['role_accuracy']:.6f}",
            f"surface_accuracy={challenge_lookup['future_option_prediction']['surface_accuracy']:.6f}",
            f"lift_vs_surface={challenge_lookup['future_option_prediction']['lift_vs_surface']:.6f}",
            "",
            "9. Effect-residual result",
            f"mean_effect_residual_score={report['mean_effect_residual_score']:.6f}",
            "",
            "10. Whether H2 remains supported after hardening",
            f"supports_H2={report['supports_H2']}",
            f"scientific_conclusion={report['scientific_conclusion']}",
            "",
            "11. Whether v0.10/M4 gate is cleared",
            f"v10_gate_cleared={report['v10_gate_cleared']}",
            "",
            "12. Recommendation on adding extended64 targeted games",
            f"recommend_extended64_targeted_games={report['recommend_extended64_targeted_games']}",
        ]
    )
