from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
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
from v6.role_transfer_v09 import (
    _write_parquet,
    appearance_features,
    cosine_similarity,
    deterministic_random_role_id,
    mean_vector,
    raw_m2_features,
)
from v6.role_transfer_v09a import (
    build_source_only_roles,
    m2_label_baseline_role,
    nearest_source_family,
    structural_score_to_family,
)


@dataclass(frozen=True)
class RoleTransferV09bConfig:
    m3_input_dir: str | None = None
    m2_input_dir: str = "runs/v6/v07_cd2_extended32_expanded"
    m1_input_dir: str = "runs/v6/v06_cd2_extended32"
    previous_v09_dir: str | None = None
    previous_v09a_dir: str = "runs/v6/v09a_role_transfer_sourceclean_extended32"
    output_dir: str = "runs/v6/v09b_role_transfer_refined_sourceclean_extended32"
    game_set_manifest: str | None = None
    game_set_name: str | None = None
    split_mode: str = "leave_family_out"
    workers: int = 25
    structural_success_threshold: float = 0.70
    graph_source: str = "hybrid"


@dataclass(frozen=True)
class WeightProfile:
    name: str
    future_option: float
    directional: float
    local_motif: float
    effect: float
    coarse: float

    def as_dict(self) -> dict[str, float]:
        return {
            "future_option": self.future_option,
            "directional": self.directional,
            "local_motif": self.local_motif,
            "effect": self.effect,
            "coarse": self.coarse,
        }


@dataclass(frozen=True)
class StrategySpec:
    name: str
    prototype_mode: str
    weight_profile: WeightProfile = field(default_factory=lambda: WEIGHT_PROFILES[0])
    unknown_mode: str = "include_unknown_roles"
    confidence_mode: str = "no_gating"
    similarity_threshold: float = 0.0
    margin: float = 0.0
    top_k: int = 0


@dataclass(frozen=True)
class PrototypeEntry:
    role_id: str
    role_label_candidate: str
    prototype_mode: str
    vectors: tuple[dict[str, float], ...]
    coarse_vectors: tuple[dict[str, float], ...]
    appearance_vectors: tuple[dict[str, float], ...]
    group_vectors: tuple[dict[str, dict[str, float]], ...]
    member_family_ids: tuple[str, ...]
    source_games: tuple[str, ...]
    source_families: tuple[str, ...]
    subtype_count: int
    unknown_role: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FamilyContext:
    heldout_family: str
    heldout_games: tuple[str, ...]
    source_neighborhoods: dict[str, DiscNeighborhood]
    source_roles: dict[str, dict[str, Any]]
    source_no_label_roles: dict[str, dict[str, Any]]
    target_families: tuple[Any, ...]
    full_neighborhoods: dict[str, DiscNeighborhood]
    full_no_label_neighborhoods: dict[str, DiscNeighborhood]
    graph_source_used: str
    graph_edge_coverage: float


WEIGHT_PROFILES = (
    WeightProfile("default", future_option=0.30, directional=0.25, local_motif=0.20, effect=0.20, coarse=0.05),
    WeightProfile("alternative_a", future_option=0.35, directional=0.20, local_motif=0.20, effect=0.20, coarse=0.05),
    WeightProfile("alternative_b", future_option=0.25, directional=0.30, local_motif=0.20, effect=0.20, coarse=0.05),
    WeightProfile("alternative_c", future_option=0.25, directional=0.20, local_motif=0.25, effect=0.25, coarse=0.05),
)

UNKNOWN_MODES = (
    "include_unknown_roles",
    "exclude_unknown_roles_from_assignment",
    "include_unknown_roles_but_downweight",
)

CONFIDENCE_SPECS = (
    ("no_gating", 0.0, 0.0),
    *[("similarity_threshold_gating", threshold, 0.0) for threshold in (0.55, 0.60, 0.65, 0.70)],
    *[("margin_gating", 0.0, margin) for margin in (0.00, 0.03, 0.05, 0.10)],
    *[
        ("threshold_margin_gating", threshold, margin)
        for threshold in (0.55, 0.60, 0.65, 0.70)
        for margin in (0.00, 0.03, 0.05, 0.10)
    ],
)


def run_role_transfer_v09b(config: RoleTransferV09bConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    if config.workers <= 1 or len(tasks) <= 1:
        family_contexts = [_prepare_family_context(*task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = [executor.submit(_prepare_family_context, *task) for task in tasks]
            family_contexts = [future.result() for future in futures]
    family_contexts = sorted(family_contexts, key=lambda item: item.heldout_family)

    strategies = build_strategy_specs()
    if config.workers <= 1 or len(strategies) <= 1:
        strategy_payloads = [evaluate_strategy(strategy, family_contexts, config.structural_success_threshold) for strategy in strategies]
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = [executor.submit(evaluate_strategy, strategy, family_contexts, config.structural_success_threshold) for strategy in strategies]
            strategy_payloads = [future.result() for future in futures]
    best_payload = select_best_strategy_payload(strategy_payloads)

    previous_dir = Path(config.previous_v09a_dir if Path(config.previous_v09a_dir).exists() else (config.previous_v09_dir or config.previous_v09a_dir))
    previous_report = json.loads((previous_dir / ("v09a_report.json" if (previous_dir / "v09a_report.json").exists() else "v09_report.json")).read_text(encoding="utf-8"))
    previous_by_family = _load_previous_by_family(previous_dir / ("role_transfer_by_family.parquet"))
    regression_rows = build_regression_rows(best_payload["by_family_rows"], previous_by_family)

    payload = build_report_payload(
        config=config,
        previous_payload=previous_report,
        strategy_payloads=strategy_payloads,
        best_payload=best_payload,
        regression_rows=regression_rows,
    )
    write_v09b_outputs(output_dir, payload, strategy_payloads, best_payload, regression_rows)
    return payload


def build_strategy_specs() -> list[StrategySpec]:
    base_modes = (
        ("centroid", 0),
        ("medoid", 0),
        ("top_k_neighbors", 1),
        ("top_k_neighbors", 3),
        ("top_k_neighbors", 5),
        ("family_balanced_centroid", 0),
        ("subtype_aware", 0),
        ("ensemble_strategy", 0),
    )
    specs: list[StrategySpec] = []
    for weight_profile in WEIGHT_PROFILES:
        for unknown_mode in UNKNOWN_MODES:
            for confidence_mode, threshold, margin in CONFIDENCE_SPECS:
                for prototype_mode, top_k in base_modes:
                    name_parts = [
                        prototype_mode,
                        f"weights_{weight_profile.name}",
                        unknown_mode,
                        confidence_mode,
                    ]
                    if top_k:
                        name_parts.insert(1, f"k{top_k}")
                    if threshold > 0:
                        name_parts.append(f"t{int(round(threshold * 100)):02d}")
                    if margin > 0 or confidence_mode != "no_gating":
                        name_parts.append(f"m{int(round(margin * 100)):02d}")
                    specs.append(
                        StrategySpec(
                            name="__".join(name_parts),
                            prototype_mode=prototype_mode,
                            weight_profile=weight_profile,
                            unknown_mode=unknown_mode,
                            confidence_mode=confidence_mode,
                            similarity_threshold=float(threshold),
                            margin=float(margin),
                            top_k=int(top_k),
                        )
                    )
    return specs


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


def evaluate_strategy(strategy: StrategySpec, family_contexts: list[FamilyContext], threshold: float) -> dict[str, Any]:
    family_results = [_evaluate_strategy_family(strategy, context, threshold) for context in family_contexts]
    assignment_rows = [row for result in family_results for row in result["assignments"]]
    failure_rows = [row for result in family_results for row in result["failures"]]
    by_family_rows = [result["summary"] for result in family_results]
    by_role_rows = build_by_role_rows(strategy.name, assignment_rows, family_contexts)
    confidence_rows = build_confidence_rows(strategy.name, assignment_rows)
    subtype_rows = build_subtype_rows(strategy.name, family_results)
    metrics = build_strategy_metrics(strategy, by_family_rows, by_role_rows, assignment_rows)
    return {
        "strategy": strategy,
        "assignments": assignment_rows,
        "failures": failure_rows,
        "by_family_rows": by_family_rows,
        "by_role_rows": by_role_rows,
        "confidence_rows": confidence_rows,
        "subtype_rows": subtype_rows,
        "strategy_metrics": metrics,
    }


def _evaluate_strategy_family(strategy: StrategySpec, context: FamilyContext, threshold: float) -> dict[str, Any]:
    source_prototypes = build_strategy_prototypes(strategy, context.source_roles, context.source_neighborhoods)
    no_label_roles = context.source_no_label_roles
    assignments: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for family in context.target_families:
        target = context.full_neighborhoods.get(family.family_id)
        target_no_label = context.full_no_label_neighborhoods.get(family.family_id)
        if target is None or target_no_label is None:
            failures.append({"strategy_name": strategy.name, "heldout_family": context.heldout_family, "target_family_id": family.family_id, "failure_reason": "missing_target_neighborhood"})
            continue
        if family.support_count < 3:
            failures.append({"strategy_name": strategy.name, "heldout_family": context.heldout_family, "target_family_id": family.family_id, "failure_reason": "insufficient_target_support"})
            continue
        if not source_prototypes:
            failures.append({"strategy_name": strategy.name, "heldout_family": context.heldout_family, "target_family_id": family.family_id, "failure_reason": "no_source_roles"})
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
        best_scores = ranked[0][2] if ranked else empty_structural_scores()
        best_similarity = float(best_scores["structural_prediction_score"])
        second_similarity = float(ranked[1][2]["structural_prediction_score"]) if len(ranked) > 1 else 0.0
        low_confidence = int(not passes_confidence_gate(strategy, best_similarity, second_similarity))

        random_role_id = deterministic_random_role_id({key: None for key in sorted(source_prototypes)}, context.heldout_family, family.family_id)
        majority_role_id = max(sorted(source_prototypes), key=lambda key: len(source_prototypes[key].member_family_ids))
        coarse_role_id, coarse_scores = nearest_role_by_vector(target.coarse_features, context.source_roles, "coarse_features")
        surface_role_id, surface_scores = nearest_role_by_vector(appearance_features(target), context.source_roles, "appearance_features")
        raw_family_id = nearest_source_family(target, context.source_neighborhoods, mode="raw_m2")
        m2_label_role_id = m2_label_baseline_role(target, context.source_neighborhoods, context.source_roles)
        no_label_role_id, no_label_scores = nearest_no_label_role(target_no_label, no_label_roles, strategy.weight_profile.as_dict())

        row = {
            "strategy_name": strategy.name,
            "heldout_family": context.heldout_family,
            "target_family_id": family.family_id,
            "target_games": list(get_games(family)),
            "target_game_families": list(get_game_families(target)),
            "assigned_role_id": best_role_id,
            "assigned_role_label": source_prototypes[best_role_id].role_label_candidate if best_role_id else "",
            "assigned_structural_prediction_score": best_similarity,
            "assigned_similarity": best_similarity,
            "second_best_similarity": second_similarity,
            "future_option_prediction_score": best_scores["future_option_prediction_score"],
            "directional_prediction_score": best_scores["directional_prediction_score"],
            "motif_prediction_score": best_scores["motif_prediction_score"],
            "effect_prediction_score": best_scores["effect_prediction_score"],
            "coarse_prediction_score": best_scores["coarse_prediction_score"],
            "role_success": int(best_similarity >= threshold),
            "low_confidence_assignment": low_confidence,
            "unknown_role_assignment": int(bool(best_role_id) and source_prototypes[best_role_id].unknown_role),
            "random_role_id": random_role_id,
            "random_success": int(score_source_role(target, context.source_roles[random_role_id], strategy.weight_profile.as_dict())["structural_prediction_score"] >= threshold),
            "majority_role_id": majority_role_id,
            "majority_success": int(score_source_role(target, context.source_roles[majority_role_id], strategy.weight_profile.as_dict())["structural_prediction_score"] >= threshold),
            "coarse_role_id": coarse_role_id,
            "coarse_success": int(coarse_scores["structural_prediction_score"] >= threshold),
            "surface_role_id": surface_role_id,
            "surface_success": int(surface_scores["structural_prediction_score"] >= threshold),
            "raw_m2_family_id": raw_family_id,
            "raw_m2_success": int(structural_score_to_family(target, context.source_neighborhoods[raw_family_id])["structural_prediction_score"] >= threshold) if raw_family_id else 0,
            "m2_label_role_id": m2_label_role_id,
            "m2_label_success": int(score_source_role(target, context.source_roles[m2_label_role_id], strategy.weight_profile.as_dict())["structural_prediction_score"] >= threshold) if m2_label_role_id else 0,
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
        "strategy_name": strategy.name,
        "heldout_family": context.heldout_family,
        "graph_source_used": context.graph_source_used,
        "graph_edge_coverage": context.graph_edge_coverage,
        "source_only_roles": len(context.source_roles),
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
        "coverage_rate": float(np.mean([1 - row["low_confidence_assignment"] for row in assignments])) if assignments else 0.0,
        "low_confidence_assignment_rate": float(np.mean([row["low_confidence_assignment"] for row in assignments])) if assignments else 0.0,
    }
    return {"summary": summary, "assignments": assignments, "failures": failures, "prototypes": source_prototypes}


def build_strategy_prototypes(
    strategy: StrategySpec,
    source_roles: dict[str, dict[str, Any]],
    source_neighborhoods: dict[str, DiscNeighborhood],
) -> dict[str, PrototypeEntry]:
    output = {}
    for role_id, role_entry in sorted(source_roles.items()):
        members = [source_neighborhoods[item] for item in role_entry["member_family_ids"] if item in source_neighborhoods]
        if len(members) < 3:
            continue
        role_obj = _RoleView(role_id=role_id, role_label_candidate=role_entry["role_label_candidate"])
        entry = build_prototype_entry(strategy, role_obj, members)
        if entry is not None:
            output[role_id] = entry
    return output


@dataclass(frozen=True)
class _RoleView:
    role_id: str
    role_label_candidate: str


def build_prototype_entry(strategy: StrategySpec, role: Any, members: list[Any]) -> PrototypeEntry | None:
    normalized_mode = {
        "family_balanced": "family_balanced_centroid",
        "topk": "top_k_neighbors",
        "subtype": "subtype_aware",
    }.get(strategy.prototype_mode, strategy.prototype_mode)
    role_id = str(getattr(role, "role_id"))
    role_label_candidate = str(getattr(role, "role_label_candidate"))
    grouped = [feature_groups(member) for member in members]
    vectors_all = [flatten_feature_groups(item) for item in grouped]
    vectors_coarse = [item["coarse"] for item in grouped]
    vectors_appearance = [appearance_features(member) for member in members]

    if normalized_mode == "centroid":
        group_vectors = (mean_group_vectors(grouped),)
    elif normalized_mode == "medoid":
        index = medoid_index(vectors_all)
        group_vectors = (grouped[index],)
    elif normalized_mode == "family_balanced_centroid":
        family_groups = []
        for _, group_members in _group_members_by_manifest_family(members).items():
            family_groups.append(mean_group_vectors([feature_groups(item) for item in group_members]))
        group_vectors = (mean_group_vectors(family_groups),)
    elif normalized_mode == "top_k_neighbors":
        group_vectors = tuple(grouped)
    elif normalized_mode == "subtype_aware":
        subtype_groups = _group_members_by_subtype(members)
        group_vectors = tuple(mean_group_vectors([feature_groups(item) for item in group_members]) for _, group_members in sorted(subtype_groups.items()))
    elif normalized_mode == "ensemble_strategy":
        component_specs = (
            StrategySpec("fb", "family_balanced_centroid", strategy.weight_profile, top_k=0),
            StrategySpec("medoid", "medoid", strategy.weight_profile, top_k=0),
            StrategySpec("topk3", "top_k_neighbors", strategy.weight_profile, top_k=3),
            StrategySpec("subtype", "subtype_aware", strategy.weight_profile, top_k=0),
        )
        metadata = {
            "ensemble_components": {
                spec.prototype_mode if spec.prototype_mode != "top_k_neighbors" else f"top_k_neighbors_{spec.top_k}": build_prototype_entry(spec, role, members)
                for spec in component_specs
            }
        }
        component = metadata["ensemble_components"]["family_balanced_centroid"]
        if component is None:
            return None
        return PrototypeEntry(
            role_id=role_id,
            role_label_candidate=role_label_candidate,
            prototype_mode=normalized_mode,
            vectors=component.vectors,
            coarse_vectors=component.coarse_vectors,
            appearance_vectors=component.appearance_vectors,
            group_vectors=component.group_vectors,
            member_family_ids=tuple(sorted(getattr(item, "family_id") for item in members)),
            source_games=tuple(sorted({game for item in members for game in get_games(item)})),
            source_families=tuple(sorted({family for item in members for family in get_game_families(item)})),
            subtype_count=sum(entry.subtype_count for entry in metadata["ensemble_components"].values() if entry is not None),
            unknown_role=role_label_candidate == "unknown_role_candidate",
            metadata=metadata,
        )
    else:
        return None

    vectors = tuple(flatten_feature_groups(item) for item in group_vectors)
    return PrototypeEntry(
        role_id=role_id,
        role_label_candidate=role_label_candidate,
        prototype_mode=normalized_mode,
        vectors=vectors,
        coarse_vectors=tuple(item["coarse"] for item in group_vectors),
        appearance_vectors=tuple(_appearance_from_groups(item, role_label_candidate) for item in group_vectors),
        group_vectors=tuple(group_vectors),
        member_family_ids=tuple(sorted(getattr(item, "family_id") for item in members)),
        source_games=tuple(sorted({game for item in members for game in get_games(item)})),
        source_families=tuple(sorted({family for item in members for family in get_game_families(item)})),
        subtype_count=len(group_vectors),
        unknown_role=role_label_candidate == "unknown_role_candidate",
    )


def medoid_index(vectors: list[dict[str, float]]) -> int:
    scores = []
    for index, left in enumerate(vectors):
        sims = [cosine_similarity(left, right) for other_index, right in enumerate(vectors) if other_index != index]
        scores.append((float(np.mean(sims)) if sims else 0.0, -index))
    best = max(scores)
    return -best[1]


def rank_roles_for_target(
    target: Any,
    entries: dict[str, PrototypeEntry],
    *,
    mode: str,
    top_k: int,
    weights: dict[str, float] | None = None,
    strategy: StrategySpec | None = None,
) -> list[tuple[str, float, dict[str, float]]]:
    target_groups = feature_groups(target)
    weights = weights or WEIGHT_PROFILES[0].as_dict()
    ranked = []
    for role_id, entry in sorted(entries.items()):
        score = score_prototype_entry(target_groups, entry, weights, top_k=top_k or strategy.top_k if strategy else top_k)
        adjusted = float(score["structural_prediction_score"])
        if strategy is not None:
            if strategy.unknown_mode == "exclude_unknown_roles_from_assignment" and entry.unknown_role:
                continue
            if strategy.unknown_mode == "include_unknown_roles_but_downweight" and entry.unknown_role:
                adjusted *= 0.97
        ranked.append((role_id, adjusted, {**score, "structural_prediction_score": adjusted}))
    ranked.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return ranked


def score_prototype_entry(
    target_groups: dict[str, dict[str, float]],
    entry: PrototypeEntry,
    weights: dict[str, float],
    *,
    top_k: int,
) -> dict[str, float]:
    if entry.prototype_mode == "ensemble_strategy":
        component_weights = {
            "family_balanced_centroid": 0.30,
            "medoid": 0.20,
            "top_k_neighbors_3": 0.25,
            "subtype_aware": 0.25,
        }
        combined = defaultdict(float)
        for name, component in sorted(entry.metadata.get("ensemble_components", {}).items()):
            if component is None:
                continue
            weight = float(component_weights.get(name, 0.0))
            scores = score_prototype_entry(target_groups, component, weights, top_k=3 if name == "top_k_neighbors_3" else 0)
            for key, value in scores.items():
                combined[key] += weight * float(value)
        return {key: float(value) for key, value in combined.items()} if combined else empty_structural_scores()

    score_rows = [weighted_structural_score(target_groups, group_vector, weights) for group_vector in entry.group_vectors]
    if not score_rows:
        return empty_structural_scores()
    if entry.prototype_mode == "top_k_neighbors":
        k = max(1, min(top_k or len(score_rows), len(score_rows)))
        score_rows = sorted(score_rows, key=lambda row: row["structural_prediction_score"], reverse=True)[:k]
    else:
        score_rows = [max(score_rows, key=lambda row: row["structural_prediction_score"])]
    return {key: float(np.mean([row[key] for row in score_rows])) for key in score_rows[0]}


def weighted_structural_score(
    target_groups: dict[str, dict[str, float]],
    source_groups: dict[str, dict[str, float]],
    weights: dict[str, float],
) -> dict[str, float]:
    scores = {
        "future_option_prediction_score": cosine_similarity(target_groups["future_option"], source_groups["future_option"]),
        "directional_prediction_score": cosine_similarity(target_groups["directional"], source_groups["directional"]),
        "motif_prediction_score": cosine_similarity(target_groups["local_motif"], source_groups["local_motif"]),
        "effect_prediction_score": cosine_similarity(target_groups["effect"], source_groups["effect"]),
        "coarse_prediction_score": cosine_similarity(target_groups["coarse"], source_groups["coarse"]),
    }
    scores["structural_prediction_score"] = float(
        weights["future_option"] * scores["future_option_prediction_score"]
        + weights["directional"] * scores["directional_prediction_score"]
        + weights["local_motif"] * scores["motif_prediction_score"]
        + weights["effect"] * scores["effect_prediction_score"]
        + weights["coarse"] * scores["coarse_prediction_score"]
    )
    return scores


def empty_structural_scores() -> dict[str, float]:
    return {
        "future_option_prediction_score": 0.0,
        "directional_prediction_score": 0.0,
        "motif_prediction_score": 0.0,
        "effect_prediction_score": 0.0,
        "coarse_prediction_score": 0.0,
        "structural_prediction_score": 0.0,
    }


def passes_confidence_gate(strategy: StrategySpec, best_similarity: float, second_similarity: float) -> bool:
    if strategy.confidence_mode == "no_gating":
        return True
    if best_similarity < strategy.similarity_threshold:
        return False
    if (best_similarity - second_similarity) < strategy.margin:
        return False
    return True


def nearest_role_by_vector(target_vector: dict[str, float], source_roles: dict[str, dict[str, Any]], field_name: str) -> tuple[str, dict[str, float]]:
    best_role = ""
    best_score = -1.0
    for role_id, entry in sorted(source_roles.items()):
        score = cosine_similarity(target_vector, entry[field_name])
        if score > best_score:
            best_role = role_id
            best_score = score
    return best_role, {"structural_prediction_score": float(best_score)}


def nearest_no_label_role(
    target: DiscNeighborhood,
    source_roles: dict[str, dict[str, Any]],
    weights: dict[str, float],
) -> tuple[str, dict[str, float]]:
    best_role = ""
    best_scores = empty_structural_scores()
    for role_id, entry in sorted(source_roles.items()):
        scores = score_source_role(target, entry, weights)
        if scores["structural_prediction_score"] > best_scores["structural_prediction_score"]:
            best_role = role_id
            best_scores = scores
    return best_role, best_scores


def score_source_role(target: DiscNeighborhood, role_entry: dict[str, Any], weights: dict[str, float]) -> dict[str, float]:
    source_groups = {
        "coarse": subset_prefixed(role_entry["all_features"], "coarse:"),
        "directional": subset_prefixed(role_entry["all_features"], "directional:"),
        "future_option": subset_prefixed(role_entry["all_features"], "future:"),
        "local_motif": subset_prefixed(role_entry["all_features"], "motif:"),
        "effect": subset_prefixed(role_entry["all_features"], "effect:"),
    }
    return weighted_structural_score(feature_groups(target), source_groups, weights)


def subset_prefixed(vector: dict[str, float], prefix: str) -> dict[str, float]:
    return {key[len(prefix):]: float(value) for key, value in vector.items() if key.startswith(prefix)}


def feature_groups(record: Any) -> dict[str, dict[str, float]]:
    return {
        "coarse": {str(key): float(value) for key, value in getattr(record, "coarse_features").items()},
        "directional": {str(key): float(value) for key, value in getattr(record, "directional_features").items()},
        "future_option": {str(key): float(value) for key, value in getattr(record, "future_option_features").items()},
        "local_motif": {str(key): float(value) for key, value in getattr(record, "local_motif_features").items()},
        "effect": {str(key): float(value) for key, value in getattr(record, "temporal_effect_features").items()},
    }


def flatten_feature_groups(groups: dict[str, dict[str, float]]) -> dict[str, float]:
    output = {}
    prefixes = {
        "coarse": "coarse",
        "directional": "directional",
        "future_option": "future",
        "local_motif": "motif",
        "effect": "effect",
    }
    for group_name, prefix in prefixes.items():
        for key, value in groups[group_name].items():
            output[f"{prefix}:{key}"] = float(value)
    return output


def mean_group_vectors(group_vectors: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    return {
        group_name: mean_vector([item[group_name] for item in group_vectors])
        for group_name in ("coarse", "directional", "future_option", "local_motif", "effect")
    }


def _appearance_from_groups(groups: dict[str, dict[str, float]], role_label_candidate: str) -> dict[str, float]:
    effect = groups["effect"]
    return {
        f"label::{role_label_candidate}": 1.0,
        "no_change_rate": float(effect.get("no_change_rate", 0.0)),
        "position_change_rate": float(effect.get("position_change_rate", 0.0)),
        "discontinuous_position_change_rate": float(effect.get("discontinuous_position_change_rate", 0.0)),
        "terminal_rate": float(effect.get("terminal_rate", 0.0)),
        "multi_cell_change_rate": float(effect.get("multi_cell_change_rate", 0.0)),
        "coverage_change_rate": float(effect.get("coverage_change_rate_if_derivable", 0.0)),
        "repeated_toggle_like_rate": float(effect.get("repeated_toggle_like_rate", 0.0)),
    }


def _group_members_by_manifest_family(members: list[Any]) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = defaultdict(list)
    for member in members:
        key = "|".join(sorted(get_game_families(member))) or "unknown_family"
        groups[key].append(member)
    return dict(sorted(groups.items()))


def _group_members_by_subtype(members: list[Any]) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = defaultdict(list)
    for member in members:
        future = getattr(member, "future_option_features")
        directional = getattr(member, "directional_features")
        motif = getattr(member, "local_motif_features")
        effect = getattr(member, "temporal_effect_features")
        key = json.dumps(
            [
                getattr(member, "dominant_outcome_signature"),
                getattr(member, "dominant_motif_candidate"),
                _band(float(future.get("enable_score", 0.0) - future.get("block_score", 0.0))),
                _band(float(future.get("terminate_score", 0.0))),
                _band(float(directional.get("directional_asymmetry_score", 0.0))),
                _band(float(motif.get("cross_game_family_presence", 0.0))),
                _band(float(effect.get("position_change_rate", 0.0))),
                _band(float(effect.get("terminal_rate", 0.0))),
            ],
            separators=(",", ":"),
        )
        groups[key].append(member)
    return dict(sorted(groups.items()))


def _band(value: float) -> str:
    if value >= 0.66:
        return "high"
    if value >= 0.33:
        return "mid"
    if value <= -0.66:
        return "neg_high"
    if value <= -0.33:
        return "neg_mid"
    return "low"


def get_games(record: Any) -> tuple[str, ...]:
    return tuple(sorted(getattr(record, "games_present", getattr(record, "game_ids", ()))))


def get_game_families(record: Any) -> tuple[str, ...]:
    return tuple(sorted(getattr(record, "game_families_present", getattr(record, "game_family_ids", ()))))


def build_by_role_rows(strategy_name: str, assignment_rows: list[dict[str, Any]], family_contexts: list[FamilyContext]) -> list[dict[str, Any]]:
    observed = defaultdict(lambda: {"success": 0, "total": 0, "families": set(), "games": set(), "label": "", "unknown": False})
    all_roles = {}
    all_source_neighborhoods = {}
    for context in family_contexts:
        all_source_neighborhoods.update(context.source_neighborhoods)
        for role_id, role_entry in context.source_roles.items():
            if role_id not in all_roles:
                all_roles[role_id] = role_entry
    for row in assignment_rows:
        role_id = row["assigned_role_id"]
        if not role_id:
            continue
        bucket = observed[role_id]
        bucket["success"] += int(row["role_success"])
        bucket["total"] += 1
        bucket["families"].add(row["heldout_family"])
        bucket["games"].update(row["target_games"])
        bucket["label"] = row["assigned_role_label"]
        bucket["unknown"] = bool(row["unknown_role_assignment"])
    rows = []
    for role_id, role_entry in sorted(all_roles.items()):
        bucket = observed.get(role_id, {"success": 0, "total": 0, "families": set(), "games": set(), "label": role_entry["role_label_candidate"], "unknown": role_entry["role_label_candidate"] == "unknown_role_candidate"})
        rows.append(
            {
                "strategy_name": strategy_name,
                "role_id": role_id,
                "role_label_candidate": bucket["label"] or role_entry["role_label_candidate"],
                "source_games": list(sorted({game for family_id in role_entry["member_family_ids"] if family_id in all_source_neighborhoods for game in get_games(all_source_neighborhoods[family_id])})),
                "target_games_matched": sorted(bucket["games"]),
                "target_families_matched": sorted(bucket["families"]),
                "transfer_success_count": int(bucket["success"]),
                "transfer_failure_count": int(max(0, bucket["total"] - bucket["success"])),
                "transfer_accuracy": float(bucket["success"] / max(1, bucket["total"])),
                "unknown_role_candidate": bool(bucket["unknown"]),
            }
        )
    return rows


def build_confidence_rows(strategy_name: str, assignment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "strategy_name": strategy_name,
            "heldout_family": row["heldout_family"],
            "target_family_id": row["target_family_id"],
            "assigned_similarity": row["assigned_similarity"],
            "second_best_similarity": row["second_best_similarity"],
            "low_confidence_assignment": row["low_confidence_assignment"],
            "role_success": row["role_success"],
        }
        for row in assignment_rows
    ]


def build_subtype_rows(strategy_name: str, family_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for result in family_results:
        prototypes = result["prototypes"]
        heldout_family = result["summary"]["heldout_family"]
        for role_id, entry in sorted(prototypes.items()):
            key = (heldout_family, role_id)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "strategy_name": strategy_name,
                    "heldout_family": heldout_family,
                    "role_id": role_id,
                    "role_label_candidate": entry.role_label_candidate,
                    "prototype_mode": entry.prototype_mode,
                    "subtype_count": entry.subtype_count,
                    "member_family_count": len(entry.member_family_ids),
                    "source_games": list(entry.source_games),
                    "source_families": list(entry.source_families),
                    "unknown_role_candidate": entry.unknown_role,
                }
            )
    return rows


def build_strategy_metrics(
    strategy: StrategySpec,
    by_family_rows: list[dict[str, Any]],
    by_role_rows: list[dict[str, Any]],
    assignment_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluable_rows = [row for row in by_family_rows if row["target_m2_families"] > 0]
    metric = lambda key: float(np.mean([row[key] for row in evaluable_rows])) if evaluable_rows else 0.0
    role_acc = metric("transfer_accuracy_structural_role")
    raw_m2 = metric("transfer_accuracy_raw_m2")
    surface = metric("transfer_accuracy_surface_effect")
    no_label = metric("transfer_accuracy_graph_role_no_label")
    mean_lift_best = metric("role_lift_over_best_baseline")
    positive_families = sum(1 for row in evaluable_rows if row["role_lift_over_best_baseline"] > 0)
    coverage_rate = float(np.mean([1 - row["low_confidence_assignment"] for row in assignment_rows])) if assignment_rows else 0.0
    low_conf_rate = float(np.mean([row["low_confidence_assignment"] for row in assignment_rows])) if assignment_rows else 0.0
    confident_rows = [row for row in assignment_rows if not row["low_confidence_assignment"]]
    confident_precision = float(np.mean([row["role_success"] for row in confident_rows])) if confident_rows else 0.0
    successful_role_candidates = sum(1 for row in by_role_rows if row["transfer_success_count"] > row["transfer_failure_count"])
    successful_assignments = [row for row in assignment_rows if row["role_success"]]
    success_by_role = defaultdict(int)
    for row in successful_assignments:
        success_by_role[row["assigned_role_id"]] += 1
    dominant_successful_role_share = float(max(success_by_role.values()) / max(1, len(successful_assignments))) if success_by_role else 0.0

    conclusion = "role_transfer_refined_not_established"
    if (
        positive_families >= 12
        and (role_acc - raw_m2) >= 0.05
        and (role_acc - surface) >= 0.12
        and (role_acc - no_label) >= 0.03
        and coverage_rate >= 0.85
        and dominant_successful_role_share <= 0.40
    ):
        conclusion = "role_transfer_refined_very_strong"
    elif (
        positive_families >= 10
        and (role_acc - raw_m2) >= 0.03
        and (role_acc - surface) >= 0.10
        and (role_acc - no_label) > 0.0
        and coverage_rate >= 0.80
    ):
        conclusion = "role_transfer_refined_strong"
    elif (
        positive_families >= 8
        and (role_acc - raw_m2) > 0.0
        and (role_acc - surface) > 0.0
        and (role_acc - no_label) > 0.0
        and mean_lift_best > 0.0
    ):
        conclusion = "role_transfer_refined_weak"

    return {
        "strategy_name": strategy.name,
        "prototype_mode": strategy.prototype_mode,
        "weight_profile": strategy.weight_profile.name,
        "weight_profile_values": strategy.weight_profile.as_dict(),
        "top_k": strategy.top_k,
        "confidence_mode": strategy.confidence_mode,
        "similarity_threshold": strategy.similarity_threshold,
        "margin": strategy.margin,
        "unknown_mode": strategy.unknown_mode,
        "transfer_accuracy_structural_role": role_acc,
        "transfer_accuracy_raw_m2": raw_m2,
        "transfer_accuracy_surface_effect": surface,
        "transfer_accuracy_graph_role_no_label": no_label,
        "lift_vs_raw_m2": role_acc - raw_m2,
        "lift_vs_surface_effect": role_acc - surface,
        "lift_vs_no_label_graph": role_acc - no_label,
        "positive_lift_families": positive_families,
        "evaluable_heldout_families": len(evaluable_rows),
        "mean_structural_score": float(np.mean([row["assigned_structural_prediction_score"] for row in assignment_rows])) if assignment_rows else 0.0,
        "mean_role_lift_over_best_baseline": mean_lift_best,
        "coverage_rate": coverage_rate,
        "low_confidence_assignment_rate": low_conf_rate,
        "confident_assignment_precision": confident_precision,
        "successful_role_candidates": successful_role_candidates,
        "dominant_successful_role_share": dominant_successful_role_share,
        "scientific_conclusion": conclusion,
        "supports_H2": conclusion != "role_transfer_refined_not_established",
    }


def select_best_strategy_payload(strategy_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    def key(payload: dict[str, Any]) -> tuple[Any, ...]:
        row = payload["strategy_metrics"]
        return (
            row.get("positive_lift_families", row.get("positive_role_lift_families", 0)),
            row.get("lift_vs_raw_m2", row.get("role_transfer_lift_vs_raw_m2", 0.0)),
            row["mean_role_lift_over_best_baseline"],
            row.get("transfer_accuracy_structural_role", row.get("transfer_accuracy_role", 0.0)),
            row["coverage_rate"],
            row.get("confident_assignment_precision", row.get("precision_on_confident_assignments", 0.0)),
            -row["low_confidence_assignment_rate"],
            getattr(payload.get("strategy"), "name", row["strategy_name"]),
        )

    return max(strategy_payloads, key=key)


def build_regression_rows(by_family_rows: list[dict[str, Any]], previous_by_family: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in by_family_rows:
        prev = previous_by_family.get(row["heldout_family"], {})
        previous_acc = float(prev.get("transfer_accuracy_structural_role", 0.0))
        previous_lift = float(prev.get("role_lift_over_best_baseline", 0.0))
        rows.append(
            {
                "heldout_family": row["heldout_family"],
                "v09a_transfer_accuracy_structural_role": previous_acc,
                "v09b_transfer_accuracy_structural_role": row["transfer_accuracy_structural_role"],
                "delta_transfer_accuracy_structural_role": row["transfer_accuracy_structural_role"] - previous_acc,
                "v09a_role_lift_over_best_baseline": previous_lift,
                "v09b_role_lift_over_best_baseline": row["role_lift_over_best_baseline"],
                "delta_role_lift_over_best_baseline": row["role_lift_over_best_baseline"] - previous_lift,
                "improved_vs_v09a": bool(row["role_lift_over_best_baseline"] > previous_lift),
                "regressed_vs_v09a": bool(row["role_lift_over_best_baseline"] < previous_lift),
            }
        )
    return rows


def build_report_payload(
    *,
    config: RoleTransferV09bConfig,
    previous_payload: dict[str, Any],
    strategy_payloads: list[dict[str, Any]],
    best_payload: dict[str, Any],
    regression_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    previous = previous_payload["report"]
    best = dict(best_payload["strategy_metrics"])
    improved = sorted(row["heldout_family"] for row in regression_rows if row["improved_vs_v09a"])
    regressed = sorted(row["heldout_family"] for row in regression_rows if row["regressed_vs_v09a"])
    by_role_rows = best_payload["by_role_rows"]
    successful_roles = sorted(
        [row for row in by_role_rows if row["transfer_success_count"] > row["transfer_failure_count"]],
        key=lambda row: (-row["transfer_accuracy"], -row["transfer_success_count"], row["role_id"]),
    )
    failing_roles = sorted(
        [row for row in by_role_rows if row["transfer_success_count"] <= row["transfer_failure_count"]],
        key=lambda row: (row["transfer_accuracy"], row["role_id"]),
    )
    include_unknown_rows = [item["strategy_metrics"] for item in strategy_payloads if item["strategy_metrics"]["unknown_mode"] == "include_unknown_roles"]
    exclude_unknown_rows = [item["strategy_metrics"] for item in strategy_payloads if item["strategy_metrics"]["unknown_mode"] == "exclude_unknown_roles_from_assignment"]
    downweight_unknown_rows = [item["strategy_metrics"] for item in strategy_payloads if item["strategy_metrics"]["unknown_mode"] == "include_unknown_roles_but_downweight"]
    selector = lambda row: (row["positive_lift_families"], row["lift_vs_raw_m2"], row["mean_role_lift_over_best_baseline"], row["transfer_accuracy_structural_role"])
    best_include = max(include_unknown_rows, key=selector) if include_unknown_rows else best
    best_exclude = max(exclude_unknown_rows, key=selector) if exclude_unknown_rows else best
    best_downweight = max(downweight_unknown_rows, key=selector) if downweight_unknown_rows else best
    best["families_improved_vs_v09a"] = improved
    best["families_regressed_vs_v09a"] = regressed

    proceed_to_v10 = best["scientific_conclusion"] in {"role_transfer_refined_strong", "role_transfer_refined_very_strong"}
    report = {
        "v09a_baseline_summary": {
            "scientific_conclusion": previous["scientific_conclusion"],
            "supports_H2": previous["supports_H2"],
            "transfer_accuracy_structural_role": previous["transfer_accuracy_structural_role"],
            "transfer_accuracy_raw_m2": previous["transfer_accuracy_raw_m2"],
            "transfer_accuracy_surface_effect": previous["transfer_accuracy_surface_effect"],
            "transfer_accuracy_graph_role_no_label": previous["transfer_accuracy_graph_role_no_label"],
            "lift_vs_raw_m2": previous["lift_vs_raw_m2"],
            "lift_vs_surface_effect": previous["lift_vs_surface_effect"],
            "lift_vs_no_label_graph": previous["lift_vs_no_label_graph"],
            "positive_lift_families": previous["positive_lift_families"],
        },
        "best_strategy": best,
        "strategy_count": len(strategy_payloads),
        "families_improved_vs_v09a": improved,
        "families_regressed_vs_v09a": regressed,
        "role_candidates_that_transfer_well": successful_roles[:12],
        "role_candidates_that_fail_transfer": failing_roles[:12],
        "unknown_role_handling_effect": {
            "best_include_unknown": best_include,
            "best_exclude_unknown": best_exclude,
            "best_downweight_unknown": best_downweight,
        },
        "confidence_coverage_tradeoff": {
            "coverage_rate": best["coverage_rate"],
            "low_confidence_assignment_rate": best["low_confidence_assignment_rate"],
            "confident_assignment_precision": best["confident_assignment_precision"],
        },
        "supports_H2": True if best["scientific_conclusion"] != "role_transfer_refined_not_established" else previous["supports_H2"],
        "h2_support_strengthened": bool(
            best["positive_lift_families"] > previous["positive_lift_families"]
            and best["lift_vs_raw_m2"] > previous["lift_vs_raw_m2"]
            and best["transfer_accuracy_structural_role"] > previous["transfer_accuracy_structural_role"]
        ),
        "v10_gate_cleared": proceed_to_v10,
        "scientific_conclusion": best["scientific_conclusion"],
    }
    return {
        "config": {
            "m2_input_dir": config.m2_input_dir,
            "m1_input_dir": config.m1_input_dir,
            "previous_v09a_dir": config.previous_v09a_dir,
            "output_dir": config.output_dir,
            "split_mode": config.split_mode,
            "workers": config.workers,
            "graph_source": config.graph_source,
        },
        "report": report,
        "validation": {
            "diagnostic_success": True,
            "scientific_conclusion": best["scientific_conclusion"],
            "proceed_to_v10": proceed_to_v10,
        },
    }


def write_v09b_outputs(
    output_dir: Path,
    payload: dict[str, Any],
    strategy_payloads: list[dict[str, Any]],
    best_payload: dict[str, Any],
    regression_rows: list[dict[str, Any]],
) -> None:
    comparison_rows = [item["strategy_metrics"] for item in strategy_payloads]
    _write_parquet(output_dir / "v09b_strategy_comparison.parquet", comparison_rows)
    _write_parquet(output_dir / "v09b_best_strategy_assignments.parquet", best_payload["assignments"])
    _write_parquet(output_dir / "v09b_transfer_by_family.parquet", best_payload["by_family_rows"])
    _write_parquet(output_dir / "v09b_transfer_by_role.parquet", best_payload["by_role_rows"])
    _write_parquet(output_dir / "v09b_confidence_analysis.parquet", best_payload["confidence_rows"])
    _write_parquet(output_dir / "v09b_subtype_diagnostics.parquet", best_payload["subtype_rows"])
    _write_parquet(output_dir / "v09b_regression_vs_v09a.parquet", regression_rows)
    (output_dir / "v09b_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "v09b_report.txt").write_text(format_v09b_report(payload, comparison_rows), encoding="utf-8")


def format_v09b_report(payload: dict[str, Any], comparison_rows: list[dict[str, Any]]) -> str:
    report = payload["report"]
    base = report["v09a_baseline_summary"]
    best = report["best_strategy"]
    strategy_lines = []
    ranked = sorted(
        comparison_rows,
        key=lambda row: (
            -row["positive_lift_families"],
            -row["lift_vs_raw_m2"],
            -row["mean_role_lift_over_best_baseline"],
            -row["transfer_accuracy_structural_role"],
            -row["coverage_rate"],
            -row["confident_assignment_precision"],
            row["strategy_name"],
        ),
    )[:12]
    for row in ranked:
        strategy_lines.append(
            f"{row['strategy_name']} | pos={row['positive_lift_families']} | acc={row['transfer_accuracy_structural_role']:.4f} | lift_raw={row['lift_vs_raw_m2']:.4f} | lift_surface={row['lift_vs_surface_effect']:.4f} | lift_graph={row['lift_vs_no_label_graph']:.4f} | coverage={row['coverage_rate']:.4f} | conf_prec={row['confident_assignment_precision']:.4f}"
        )

    def _fmt_roles(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "none"
        return "; ".join(
            f"{row['role_id']}:{row['role_label_candidate']}:acc={row['transfer_accuracy']:.3f}:succ={row['transfer_success_count']}:fail={row['transfer_failure_count']}"
            for row in rows
        )

    unknown = report["unknown_role_handling_effect"]
    return "\n".join(
        [
            "ARC-AGI3 v0.9b source-clean role-transfer prototype refinement",
            "",
            "1. v0.9a baseline summary",
            f"scientific_conclusion={base['scientific_conclusion']}",
            f"supports_H2={base['supports_H2']}",
            f"transfer_accuracy_structural_role={base['transfer_accuracy_structural_role']:.6f}",
            f"transfer_accuracy_raw_m2={base['transfer_accuracy_raw_m2']:.6f}",
            f"transfer_accuracy_surface_effect={base['transfer_accuracy_surface_effect']:.6f}",
            f"transfer_accuracy_graph_role_no_label={base['transfer_accuracy_graph_role_no_label']:.6f}",
            f"lift_vs_raw_m2={base['lift_vs_raw_m2']:.6f}",
            f"lift_vs_surface_effect={base['lift_vs_surface_effect']:.6f}",
            f"lift_vs_no_label_graph={base['lift_vs_no_label_graph']:.6f}",
            f"positive_lift_families={base['positive_lift_families']}",
            "",
            "2. best strategy",
            f"strategy_name={best['strategy_name']}",
            f"prototype_mode={best['prototype_mode']}",
            f"weight_profile={best['weight_profile']}",
            f"unknown_mode={best['unknown_mode']}",
            f"confidence_mode={best['confidence_mode']}",
            f"transfer_accuracy_structural_role={best['transfer_accuracy_structural_role']:.6f}",
            f"transfer_accuracy_raw_m2={best['transfer_accuracy_raw_m2']:.6f}",
            f"transfer_accuracy_surface_effect={best['transfer_accuracy_surface_effect']:.6f}",
            f"transfer_accuracy_graph_role_no_label={best['transfer_accuracy_graph_role_no_label']:.6f}",
            f"lift_vs_raw_m2={best['lift_vs_raw_m2']:.6f}",
            f"lift_vs_surface_effect={best['lift_vs_surface_effect']:.6f}",
            f"lift_vs_no_label_graph={best['lift_vs_no_label_graph']:.6f}",
            f"positive_lift_families={best['positive_lift_families']}",
            f"coverage_rate={best['coverage_rate']:.6f}",
            f"low_confidence_assignment_rate={best['low_confidence_assignment_rate']:.6f}",
            f"confident_assignment_precision={best['confident_assignment_precision']:.6f}",
            f"scientific_conclusion={best['scientific_conclusion']}",
            "",
            "3. strategy comparison",
            *strategy_lines,
            "",
            "4. families improved vs v0.9a",
            ",".join(report["families_improved_vs_v09a"]) or "none",
            "",
            "5. families regressed vs v0.9a",
            ",".join(report["families_regressed_vs_v09a"]) or "none",
            "",
            "6. role candidates that transfer well",
            _fmt_roles(report["role_candidates_that_transfer_well"]),
            "",
            "7. role candidates that fail transfer",
            _fmt_roles(report["role_candidates_that_fail_transfer"]),
            "",
            "8. effect of unknown-role handling",
            f"best_include_unknown={unknown['best_include_unknown']['strategy_name']}:pos={unknown['best_include_unknown']['positive_lift_families']}:lift_raw={unknown['best_include_unknown']['lift_vs_raw_m2']:.6f}",
            f"best_exclude_unknown={unknown['best_exclude_unknown']['strategy_name']}:pos={unknown['best_exclude_unknown']['positive_lift_families']}:lift_raw={unknown['best_exclude_unknown']['lift_vs_raw_m2']:.6f}",
            f"best_downweight_unknown={unknown['best_downweight_unknown']['strategy_name']}:pos={unknown['best_downweight_unknown']['positive_lift_families']}:lift_raw={unknown['best_downweight_unknown']['lift_vs_raw_m2']:.6f}",
            "",
            "9. confidence/coverage tradeoff",
            json.dumps(report["confidence_coverage_tradeoff"], separators=(",", ":")),
            "",
            "10. whether H2 support strengthened",
            f"supports_H2={report['supports_H2']}",
            f"h2_support_strengthened={report['h2_support_strengthened']}",
            "",
            "11. whether v0.10 gate is cleared",
            f"v10_gate_cleared={report['v10_gate_cleared']}",
        ]
    )


def _load_previous_by_family(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = pd.read_parquet(path).to_dict(orient="records")
    return {str(row["heldout_family"]): row for row in rows}
