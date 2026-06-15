from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v6.game_sets import GameSetManifest, load_game_set_manifest
from v6.role_transfer_v09 import (
    Neighborhood,
    RoleRecord,
    all_features,
    appearance_features,
    cosine_similarity,
    deterministic_random_role_id,
    load_family_to_role,
    load_neighborhoods,
    load_roles,
    majority_role,
    mean_vector,
    raw_m2_features,
    _write_parquet,
)


@dataclass(frozen=True)
class RoleTransferV09bConfig:
    m3_input_dir: str = "runs/v6/v08_cd2_extended32_discriminative"
    m2_input_dir: str = "runs/v6/v07_cd2_extended32_expanded"
    m1_input_dir: str = "runs/v6/v06_cd2_extended32"
    previous_v09_dir: str = "runs/v6/v09_role_transfer_extended32"
    output_dir: str = "runs/v6/v09b_role_transfer_refined_extended32"
    game_set_manifest: str | None = None
    game_set_name: str | None = None
    split_mode: str = "leave_family_out"
    workers: int = 25


@dataclass(frozen=True)
class StrategySpec:
    name: str
    prototype_mode: str
    top_k: int = 0
    similarity_threshold: float = 0.0
    margin: float = 0.0
    exclude_unknown_roles: bool = False


@dataclass(frozen=True)
class PrototypeEntry:
    role_id: str
    role_label_candidate: str
    vectors: tuple[dict[str, float], ...]
    coarse_vectors: tuple[dict[str, float], ...]
    appearance_vectors: tuple[dict[str, float], ...]
    member_family_ids: tuple[str, ...]
    source_games: tuple[str, ...]
    source_families: tuple[str, ...]
    subtype_count: int


def run_role_transfer_v09b(config: RoleTransferV09bConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    game_set = load_game_set_manifest(
        manifest_path=config.game_set_manifest,
        game_set_name=config.game_set_name,
        fallback_games=(),
    )
    neighborhoods = load_neighborhoods(Path(config.m3_input_dir) / "role_neighborhoods.parquet")
    roles = load_roles(Path(config.m3_input_dir) / "m3_role_candidates.json")
    family_to_role = load_family_to_role(roles)
    previous_payload = json.loads((Path(config.previous_v09_dir) / "v09_report.json").read_text(encoding="utf-8"))
    strategies = build_strategy_specs()

    strategy_payloads: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    subtype_rows: list[dict[str, Any]] = []
    for strategy in strategies:
        payload = evaluate_strategy(
            strategy=strategy,
            game_set=game_set,
            neighborhoods=neighborhoods,
            roles=roles,
            family_to_role=family_to_role,
            workers=config.workers,
        )
        strategy_payloads.append(payload)
        comparison_rows.append(payload["strategy_metrics"])
        subtype_rows.extend(payload["subtype_rows"])

    best_payload = select_best_strategy_payload(strategy_payloads)
    report_payload = build_report_payload(
        config=config,
        game_set=game_set,
        previous_payload=previous_payload,
        strategy_payloads=strategy_payloads,
        best_payload=best_payload,
    )
    write_v09b_outputs(
        output_dir=output_dir,
        payload=report_payload,
        strategy_payloads=strategy_payloads,
        best_payload=best_payload,
        comparison_rows=comparison_rows,
        subtype_rows=subtype_rows,
    )
    return report_payload


def build_strategy_specs() -> list[StrategySpec]:
    strategies = [
        StrategySpec("centroid_include_unknown", "centroid"),
        StrategySpec("centroid_exclude_unknown", "centroid", exclude_unknown_roles=True),
        StrategySpec("medoid_include_unknown", "medoid"),
        StrategySpec("medoid_exclude_unknown", "medoid", exclude_unknown_roles=True),
        StrategySpec("topk1_include_unknown", "topk", top_k=1),
        StrategySpec("topk3_include_unknown", "topk", top_k=3),
        StrategySpec("topk5_include_unknown", "topk", top_k=5),
        StrategySpec("family_balanced_include_unknown", "family_balanced"),
        StrategySpec("family_balanced_exclude_unknown", "family_balanced", exclude_unknown_roles=True),
        StrategySpec("subtype_include_unknown", "subtype"),
        StrategySpec("subtype_exclude_unknown", "subtype", exclude_unknown_roles=True),
    ]
    for mode in ("family_balanced", "subtype"):
        for exclude_unknown in (False, True):
            label = "exclude_unknown" if exclude_unknown else "include_unknown"
            for threshold in (0.60, 0.65, 0.70):
                for margin in (0.00, 0.05, 0.10):
                    strategies.append(
                        StrategySpec(
                            name=f"{mode}_gate_t{int(threshold*100):02d}_m{int(margin*100):02d}_{label}",
                            prototype_mode=mode,
                            similarity_threshold=threshold,
                            margin=margin,
                            exclude_unknown_roles=exclude_unknown,
                        )
                    )
    return strategies


def evaluate_strategy(
    *,
    strategy: StrategySpec,
    game_set: GameSetManifest,
    neighborhoods: dict[str, Neighborhood],
    roles: list[RoleRecord],
    family_to_role: dict[str, RoleRecord],
    workers: int,
) -> dict[str, Any]:
    heldout_items = sorted(game_set.families.items())
    tasks = [
        (
            strategy,
            heldout_family,
            tuple(games),
            neighborhoods,
            roles,
            family_to_role,
        )
        for heldout_family, games in heldout_items
    ]
    if workers <= 1 or len(tasks) <= 1:
        family_results = [_evaluate_strategy_family(*task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_evaluate_strategy_family, *task) for task in tasks]
            family_results = [future.result() for future in futures]
    family_results = sorted(family_results, key=lambda item: item["heldout_family"])
    assignment_rows = [row for item in family_results for row in item["assignments"]]
    failure_rows = [row for item in family_results for row in item["failures"]]
    subtype_rows = [row for item in family_results for row in item["subtype_rows"]]
    by_family_rows = build_by_family_rows(strategy.name, family_results)
    by_role_rows = build_by_role_rows(strategy.name, assignment_rows, roles)
    confidence_rows = build_confidence_rows(strategy.name, assignment_rows)
    strategy_metrics = build_strategy_metrics(strategy, by_family_rows, by_role_rows, assignment_rows, roles)
    return {
        "strategy": strategy,
        "family_results": family_results,
        "assignments": assignment_rows,
        "failures": failure_rows,
        "by_family_rows": by_family_rows,
        "by_role_rows": by_role_rows,
        "confidence_rows": confidence_rows,
        "subtype_rows": subtype_rows,
        "strategy_metrics": strategy_metrics,
    }


def _evaluate_strategy_family(
    strategy: StrategySpec,
    heldout_family: str,
    heldout_games: tuple[str, ...],
    neighborhoods: dict[str, Neighborhood],
    roles: list[RoleRecord],
    family_to_role: dict[str, RoleRecord],
) -> dict[str, Any]:
    target_family_ids = [
        family_id
        for family_id, record in neighborhoods.items()
        if set(record.games_present) & set(heldout_games)
    ]
    source_family_ids = [family_id for family_id in neighborhoods if family_id not in target_family_ids]
    source_neighborhoods = {family_id: neighborhoods[family_id] for family_id in source_family_ids}
    source_roles = build_strategy_prototypes(strategy, roles, source_neighborhoods)
    baseline_roles = build_strategy_prototypes(
        StrategySpec(
            name="baseline_centroid",
            prototype_mode="centroid",
            exclude_unknown_roles=strategy.exclude_unknown_roles,
        ),
        roles,
        source_neighborhoods,
    )
    majority_role_id = majority_role_id_from_entries(baseline_roles)
    assignment_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    subtype_rows = build_subtype_rows(strategy.name, source_roles)
    per_metric = defaultdict(list)

    for family_id in sorted(target_family_ids):
        target = neighborhoods[family_id]
        target_role = family_to_role.get(family_id)
        if target.support_count < 3:
            failure_rows.append(
                {"strategy_name": strategy.name, "heldout_family": heldout_family, "target_family_id": family_id, "failure_reason": "insufficient_target_support"}
            )
            continue
        if target_role is None or target_role.status != "stable":
            failure_rows.append(
                {"strategy_name": strategy.name, "heldout_family": heldout_family, "target_family_id": family_id, "failure_reason": "no_stable_ground_truth_role"}
            )
            continue
        if strategy.exclude_unknown_roles and target_role.role_label_candidate == "unknown_role_candidate":
            failure_rows.append(
                {
                    "strategy_name": strategy.name,
                    "heldout_family": heldout_family,
                    "target_family_id": family_id,
                    "failure_reason": "excluded_unknown_target_role",
                    "ground_truth_role_id": target_role.role_id,
                }
            )
            continue
        if target_role.role_id not in source_roles:
            failure_rows.append(
                {
                    "strategy_name": strategy.name,
                    "heldout_family": heldout_family,
                    "target_family_id": family_id,
                    "failure_reason": "no_transferable_source_role",
                    "ground_truth_role_id": target_role.role_id,
                }
            )
            continue

        ranked = rank_roles_for_target(target, source_roles, mode="all", top_k=strategy.top_k)
        coarse_ranked = rank_roles_for_target(target, baseline_roles, mode="coarse", top_k=0)
        appearance_ranked = rank_roles_for_target(target, baseline_roles, mode="appearance", top_k=0)
        raw_role_id, raw_similarity = nearest_raw_m2_role(target, source_neighborhoods, family_to_role, strategy.exclude_unknown_roles)
        best_role_id, best_similarity = ranked[0] if ranked else ("", 0.0)
        second_similarity = ranked[1][1] if len(ranked) > 1 else 0.0
        confident = bool(best_role_id) and best_similarity >= strategy.similarity_threshold and (best_similarity - second_similarity) >= strategy.margin
        assigned_role_id = best_role_id if confident or (strategy.similarity_threshold <= 0.0 and strategy.margin <= 0.0) else ""
        assigned_similarity = best_similarity if assigned_role_id else 0.0

        coarse_role_id = coarse_ranked[0][0] if coarse_ranked else ""
        coarse_similarity = coarse_ranked[0][1] if coarse_ranked else 0.0
        appearance_role_id = appearance_ranked[0][0] if appearance_ranked else ""
        appearance_similarity = appearance_ranked[0][1] if appearance_ranked else 0.0
        random_role_id = deterministic_random_role_id({key: None for key in sorted(baseline_roles)}, heldout_family, family_id)
        assigned_entry = source_roles.get(assigned_role_id)
        assignment_rows.append(
            {
                "strategy_name": strategy.name,
                "heldout_family": heldout_family,
                "target_family_id": family_id,
                "target_games": list(target.games_present),
                "target_game_families": list(target.game_families_present),
                "ground_truth_role_id": target_role.role_id,
                "ground_truth_role_label": target_role.role_label_candidate,
                "assigned_role_id": assigned_role_id,
                "assigned_role_label": assigned_entry.role_label_candidate if assigned_entry else "",
                "assigned_similarity": assigned_similarity,
                "second_best_similarity": second_similarity,
                "low_confidence_assignment": int(not confident),
                "coarse_role_id": coarse_role_id,
                "coarse_similarity": coarse_similarity,
                "appearance_role_id": appearance_role_id,
                "appearance_similarity": appearance_similarity,
                "raw_m2_role_id": raw_role_id,
                "raw_m2_similarity": raw_similarity,
                "majority_role_id": majority_role_id,
                "random_role_id": random_role_id,
                "role_correct": int(assigned_role_id == target_role.role_id),
                "coarse_correct": int(coarse_role_id == target_role.role_id),
                "appearance_correct": int(appearance_role_id == target_role.role_id),
                "raw_m2_correct": int(raw_role_id == target_role.role_id),
                "majority_correct": int(majority_role_id == target_role.role_id),
                "random_correct": int(random_role_id == target_role.role_id),
                "cross_family_transfer_score": cosine_similarity(all_features(target), assigned_entry.vectors[0]) if assigned_entry else 0.0,
            }
        )
        last = assignment_rows[-1]
        for key in ("role", "random", "majority", "raw_m2", "coarse", "appearance"):
            per_metric[key].append(int(last[f"{key}_correct"]))

    role_accuracy = float(np.mean(per_metric["role"])) if per_metric["role"] else 0.0
    baseline_scores = {key: float(np.mean(values)) if values else 0.0 for key, values in per_metric.items() if key != "role"}
    best_baseline_accuracy = max(baseline_scores.values(), default=0.0)
    return {
        "strategy_name": strategy.name,
        "heldout_family": heldout_family,
        "target_games": sorted({game for row in assignment_rows for game in row["target_games"]}),
        "target_m2_families": len(assignment_rows),
        "role_transfer_accuracy": role_accuracy,
        "baseline_scores": baseline_scores,
        "best_baseline_accuracy": best_baseline_accuracy,
        "role_lift_over_best_baseline": role_accuracy - best_baseline_accuracy,
        "assignments": assignment_rows,
        "failures": failure_rows,
        "subtype_rows": subtype_rows,
    }


def build_strategy_prototypes(
    strategy: StrategySpec,
    roles: list[RoleRecord],
    source_neighborhoods: dict[str, Neighborhood],
) -> dict[str, PrototypeEntry]:
    output = {}
    for role in roles:
        if role.status != "stable":
            continue
        if strategy.exclude_unknown_roles and role.role_label_candidate == "unknown_role_candidate":
            continue
        members = [source_neighborhoods[item] for item in role.member_family_ids if item in source_neighborhoods]
        if len(members) < 3:
            continue
        entry = build_prototype_entry(strategy, role, members)
        if entry is not None:
            output[role.role_id] = entry
    return output


def build_prototype_entry(strategy: StrategySpec, role: RoleRecord, members: list[Neighborhood]) -> PrototypeEntry | None:
    vectors_all = [all_features(member) for member in members]
    vectors_coarse = [member.coarse_features for member in members]
    vectors_appearance = [appearance_features(member) for member in members]

    if strategy.prototype_mode == "centroid":
        out_all = (mean_vector(vectors_all),)
        out_coarse = (mean_vector(vectors_coarse),)
        out_appearance = (mean_vector(vectors_appearance),)
        subtype_count = 1
    elif strategy.prototype_mode == "medoid":
        index = medoid_index(vectors_all)
        out_all = (vectors_all[index],)
        out_coarse = (vectors_coarse[index],)
        out_appearance = (vectors_appearance[index],)
        subtype_count = 1
    elif strategy.prototype_mode == "family_balanced":
        group_all = []
        group_coarse = []
        group_appearance = []
        for _, group in _group_members_by_manifest_family(members).items():
            group_all.append(mean_vector([all_features(item) for item in group]))
            group_coarse.append(mean_vector([item.coarse_features for item in group]))
            group_appearance.append(mean_vector([appearance_features(item) for item in group]))
        out_all = (mean_vector(group_all),)
        out_coarse = (mean_vector(group_coarse),)
        out_appearance = (mean_vector(group_appearance),)
        subtype_count = len(group_all)
    elif strategy.prototype_mode == "topk":
        out_all = tuple(vectors_all)
        out_coarse = tuple(vectors_coarse)
        out_appearance = tuple(vectors_appearance)
        subtype_count = len(out_all)
    elif strategy.prototype_mode == "subtype":
        subgroups = _group_members_by_subtype(members)
        out_all = tuple(mean_vector([all_features(item) for item in group]) for group in subgroups.values())
        out_coarse = tuple(mean_vector([item.coarse_features for item in group]) for group in subgroups.values())
        out_appearance = tuple(mean_vector([appearance_features(item) for item in group]) for group in subgroups.values())
        subtype_count = len(out_all)
    else:
        return None

    return PrototypeEntry(
        role_id=role.role_id,
        role_label_candidate=role.role_label_candidate,
        vectors=out_all,
        coarse_vectors=out_coarse,
        appearance_vectors=out_appearance,
        member_family_ids=tuple(sorted(member.family_id for member in members)),
        source_games=tuple(sorted({game for member in members for game in member.games_present})),
        source_families=tuple(sorted({family for member in members for family in member.game_families_present})),
        subtype_count=subtype_count,
    )


def medoid_index(vectors: list[dict[str, float]]) -> int:
    scores = []
    for i, left in enumerate(vectors):
        sims = [cosine_similarity(left, right) for j, right in enumerate(vectors) if j != i]
        scores.append((float(np.mean(sims)) if sims else 0.0, i))
    scores.sort(reverse=True)
    return scores[0][1]


def _group_members_by_manifest_family(members: list[Neighborhood]) -> dict[str, list[Neighborhood]]:
    groups: dict[str, list[Neighborhood]] = defaultdict(list)
    for member in members:
        key = "|".join(sorted(member.game_families_present)) or "unknown_family"
        groups[key].append(member)
    return dict(sorted(groups.items()))


def _group_members_by_subtype(members: list[Neighborhood]) -> dict[str, list[Neighborhood]]:
    groups: dict[str, list[Neighborhood]] = defaultdict(list)
    for member in members:
        future = member.future_option_features
        directional = member.directional_features
        effect = member.temporal_effect_features
        key = json.dumps(
            [
                member.dominant_motif_candidate,
                member.dominant_outcome_signature,
                _band((future.get("enable_score", 0.0) - future.get("block_score", 0.0))),
                _band(future.get("terminate_score", 0.0)),
                _band(effect.get("position_change_rate", 0.0)),
                _band(effect.get("terminal_rate", 0.0)),
                _band(abs(directional.get("directional_asymmetry_score", 0.0))),
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


def rank_roles_for_target(target: Neighborhood, entries: dict[str, PrototypeEntry], *, mode: str, top_k: int) -> list[tuple[str, float]]:
    target_vector = {
        "all": all_features(target),
        "coarse": target.coarse_features,
        "appearance": appearance_features(target),
    }[mode]
    ranked = []
    for role_id, entry in sorted(entries.items()):
        candidate_vectors = {
            "all": entry.vectors,
            "coarse": entry.coarse_vectors,
            "appearance": entry.appearance_vectors,
        }[mode]
        sims = sorted((cosine_similarity(target_vector, vector) for vector in candidate_vectors), reverse=True)
        if top_k > 0:
            score = float(np.mean(sims[: min(top_k, len(sims))])) if sims else 0.0
        else:
            score = max(sims) if sims else 0.0
        ranked.append((role_id, score))
    ranked.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return ranked


def nearest_raw_m2_role(
    target: Neighborhood,
    source_neighborhoods: dict[str, Neighborhood],
    family_to_role: dict[str, RoleRecord],
    exclude_unknown_roles: bool,
) -> tuple[str, float]:
    best = None
    target_vector = raw_m2_features(target)
    for family_id, source in sorted(source_neighborhoods.items()):
        source_role = family_to_role.get(family_id)
        if source_role is None or source_role.status != "stable":
            continue
        if exclude_unknown_roles and source_role.role_label_candidate == "unknown_role_candidate":
            continue
        similarity = cosine_similarity(target_vector, raw_m2_features(source))
        candidate = (similarity, source_role.role_id)
        if best is None or candidate > (best[0], best[1]):
            best = candidate
    if best is None:
        return ("", 0.0)
    return best[1], float(best[0])


def majority_role_id_from_entries(entries: dict[str, PrototypeEntry]) -> str:
    if not entries:
        return ""
    return max(sorted(entries), key=lambda role_id: len(entries[role_id].member_family_ids))


def build_by_family_rows(strategy_name: str, family_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in family_results:
        rows.append(
            {
                "strategy_name": strategy_name,
                "heldout_family": result["heldout_family"],
                "target_games": result["target_games"],
                "target_m2_families": result["target_m2_families"],
                "evaluable": bool(result["target_m2_families"] > 0),
                "role_transfer_accuracy": result["role_transfer_accuracy"],
                "best_baseline_accuracy": result["best_baseline_accuracy"],
                "role_lift_over_best_baseline": result["role_lift_over_best_baseline"],
            }
        )
    return rows


def build_by_role_rows(strategy_name: str, assignment_rows: list[dict[str, Any]], roles: list[RoleRecord]) -> list[dict[str, Any]]:
    success_rows = defaultdict(list)
    fail_rows = defaultdict(list)
    for row in assignment_rows:
        if row["role_correct"]:
            success_rows[row["ground_truth_role_id"]].append(row)
        else:
            fail_rows[row["ground_truth_role_id"]].append(row)
    rows = []
    for role in roles:
        if role.status != "stable":
            continue
        succ = success_rows.get(role.role_id, [])
        fail = fail_rows.get(role.role_id, [])
        rows.append(
            {
                "strategy_name": strategy_name,
                "role_id": role.role_id,
                "role_label_candidate": role.role_label_candidate,
                "source_games": list(role.games_present),
                "source_families": list(role.game_families_present),
                "target_games_matched": sorted({game for row in succ for game in row["target_games"]}),
                "target_families_matched": sorted({row["heldout_family"] for row in succ}),
                "transfer_success_count": len(succ),
                "transfer_failure_count": len(fail),
                "transfer_accuracy": len(succ) / max(1, len(succ) + len(fail)),
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
            "role_correct": row["role_correct"],
        }
        for row in assignment_rows
    ]


def build_subtype_rows(strategy_name: str, entries: dict[str, PrototypeEntry]) -> list[dict[str, Any]]:
    rows = []
    for role_id, entry in sorted(entries.items()):
        rows.append(
            {
                "strategy_name": strategy_name,
                "role_id": role_id,
                "role_label_candidate": entry.role_label_candidate,
                "subtype_count": entry.subtype_count,
                "member_family_count": len(entry.member_family_ids),
                "source_games": list(entry.source_games),
                "source_families": list(entry.source_families),
            }
        )
    return rows


def build_strategy_metrics(
    strategy: StrategySpec,
    by_family_rows: list[dict[str, Any]],
    by_role_rows: list[dict[str, Any]],
    assignment_rows: list[dict[str, Any]],
    roles: list[RoleRecord],
) -> dict[str, Any]:
    evaluable_rows = [row for row in by_family_rows if row["target_m2_families"] > 0]
    role_acc = float(np.mean([row["role_transfer_accuracy"] for row in evaluable_rows])) if evaluable_rows else 0.0
    best_baseline_acc = float(np.mean([row["best_baseline_accuracy"] for row in evaluable_rows])) if evaluable_rows else 0.0
    positive_families = sum(1 for row in evaluable_rows if row["role_lift_over_best_baseline"] > 0)
    successful_roles = sum(1 for row in by_role_rows if row["transfer_success_count"] > row["transfer_failure_count"])
    mean_lift_best = float(np.mean([row["role_lift_over_best_baseline"] for row in evaluable_rows])) if evaluable_rows else 0.0
    if assignment_rows:
        low_conf_rate = float(np.mean([row["low_confidence_assignment"] for row in assignment_rows]))
        coverage_rate = float(np.mean([1 - row["low_confidence_assignment"] for row in assignment_rows]))
        confident_rows = [row for row in assignment_rows if not row["low_confidence_assignment"]]
        precision_conf = float(np.mean([row["role_correct"] for row in confident_rows])) if confident_rows else 0.0
        random_acc = float(np.mean([row["random_correct"] for row in assignment_rows]))
        majority_acc = float(np.mean([row["majority_correct"] for row in assignment_rows]))
        raw_m2_acc = float(np.mean([row["raw_m2_correct"] for row in assignment_rows]))
        coarse_acc = float(np.mean([row["coarse_correct"] for row in assignment_rows]))
        appearance_acc = float(np.mean([row["appearance_correct"] for row in assignment_rows]))
    else:
        low_conf_rate = 0.0
        coverage_rate = 0.0
        precision_conf = 0.0
        random_acc = 0.0
        majority_acc = 0.0
        raw_m2_acc = 0.0
        coarse_acc = 0.0
        appearance_acc = 0.0
    transfer_roles = [row for row in by_role_rows if row["transfer_success_count"] + row["transfer_failure_count"] > 0]
    success_counts = [row["transfer_success_count"] for row in transfer_roles]
    dominant_share = (max(success_counts) / max(1, sum(success_counts))) if success_counts else 1.0
    beats_all = role_acc > random_acc and role_acc > majority_acc and role_acc > raw_m2_acc and role_acc > coarse_acc and role_acc > appearance_acc
    if beats_all and positive_families >= 12 and successful_roles >= 7 and mean_lift_best >= 0.10 and dominant_share <= 0.40:
        conclusion = "role_transfer_refined_very_strong"
    elif beats_all and positive_families >= 8 and successful_roles >= 6 and mean_lift_best >= 0.08:
        conclusion = "role_transfer_refined_strong"
    elif beats_all and positive_families >= 6 and successful_roles >= 5 and mean_lift_best >= 0.05:
        conclusion = "role_transfer_refined_weak"
    else:
        conclusion = "role_transfer_refined_not_established"
    return {
        "strategy_name": strategy.name,
        "prototype_mode": strategy.prototype_mode,
        "top_k": strategy.top_k,
        "similarity_threshold": strategy.similarity_threshold,
        "margin": strategy.margin,
        "exclude_unknown_roles": strategy.exclude_unknown_roles,
        "transfer_accuracy_role": role_acc,
        "transfer_accuracy_random": random_acc,
        "transfer_accuracy_majority": majority_acc,
        "transfer_accuracy_raw_m2": raw_m2_acc,
        "transfer_accuracy_coarse": coarse_acc,
        "transfer_accuracy_appearance_proxy": appearance_acc,
        "role_transfer_lift_vs_random": role_acc - random_acc,
        "role_transfer_lift_vs_majority": role_acc - majority_acc,
        "role_transfer_lift_vs_raw_m2": role_acc - raw_m2_acc,
        "role_transfer_lift_vs_coarse": role_acc - coarse_acc,
        "role_transfer_lift_vs_appearance": role_acc - appearance_acc,
        "positive_role_lift_families": positive_families,
        "evaluable_heldout_families": len(evaluable_rows),
        "successful_role_candidates": successful_roles,
        "mean_role_lift_over_best_baseline": mean_lift_best,
        "low_confidence_assignment_rate": low_conf_rate,
        "coverage_rate": coverage_rate,
        "precision_on_confident_assignments": precision_conf,
        "scientific_conclusion": conclusion,
        "oracle_same_family_upper_bound": 1.0 if evaluable_rows else 0.0,
        "best_possible_role_label_upper_bound": 1.0 if evaluable_rows else 0.0,
        "families_where_transfer_works": sorted([row["heldout_family"] for row in evaluable_rows if row["role_lift_over_best_baseline"] > 0]),
        "families_where_transfer_fails": sorted([row["heldout_family"] for row in evaluable_rows if row["role_lift_over_best_baseline"] <= 0]),
    }


def select_best_strategy_payload(strategy_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    def key(payload: dict[str, Any]) -> tuple[Any, ...]:
        row = payload["strategy_metrics"]
        return (
            row["positive_role_lift_families"],
            row["mean_role_lift_over_best_baseline"],
            row["transfer_accuracy_role"],
            row["successful_role_candidates"],
            int(row["coverage_rate"] >= 0.50),
            -row["low_confidence_assignment_rate"],
            row["coverage_rate"],
            strategy_specificity_rank(row["prototype_mode"]),
            row["strategy_name"],
        )

    return max(strategy_payloads, key=key)


def strategy_specificity_rank(mode: str) -> int:
    order = {"family_balanced": 5, "subtype": 4, "medoid": 3, "centroid": 2, "topk": 1}
    return order.get(mode, 0)


def build_report_payload(
    *,
    config: RoleTransferV09bConfig,
    game_set: GameSetManifest,
    previous_payload: dict[str, Any],
    strategy_payloads: list[dict[str, Any]],
    best_payload: dict[str, Any],
) -> dict[str, Any]:
    previous_report = previous_payload["report"]
    best = best_payload["strategy_metrics"]
    previous_positive = int(previous_report["positive_role_lift_family_count"])
    previous_successful = int(previous_report["successful_role_candidates_count"])
    previous_h2 = bool(previous_report["supports_H2"])
    all_comparison_rows = [item["strategy_metrics"] for item in strategy_payloads]
    unknown_rows = [row for row in all_comparison_rows if row["exclude_unknown_roles"]]
    include_rows = [row for row in all_comparison_rows if not row["exclude_unknown_roles"]]
    best_unknown = max(unknown_rows, key=lambda row: (row["positive_role_lift_families"], row["mean_role_lift_over_best_baseline"], row["transfer_accuracy_role"]))
    best_include = max(include_rows, key=lambda row: (row["positive_role_lift_families"], row["mean_role_lift_over_best_baseline"], row["transfer_accuracy_role"]))
    newly_improved = sorted(set(best["families_where_transfer_works"]) - set(previous_report["families_where_transfer_works"]))
    still_failing = sorted(set(best["families_where_transfer_fails"]))
    by_role_rows = best_payload["by_role_rows"]
    successful_role_rows = sorted([row for row in by_role_rows if row["transfer_success_count"] > row["transfer_failure_count"]], key=lambda row: (-row["transfer_accuracy"], -row["transfer_success_count"], row["role_id"]))
    failing_role_rows = sorted([row for row in by_role_rows if row["transfer_success_count"] <= row["transfer_failure_count"]], key=lambda row: (row["transfer_accuracy"], row["role_id"]))
    gate_cleared = best["scientific_conclusion"] in {
        "role_transfer_refined_weak",
        "role_transfer_refined_strong",
        "role_transfer_refined_very_strong",
    }
    report = {
        "previous_v09_summary": {
            "scientific_conclusion": previous_report["scientific_conclusion"],
            "supports_H2": previous_h2,
            "transfer_accuracy_role": previous_report["transfer_accuracy_role"],
            "transfer_accuracy_raw_m2": previous_report["transfer_accuracy_raw_m2"],
            "transfer_accuracy_appearance_proxy": previous_report["transfer_accuracy_appearance_proxy"],
            "positive_role_lift_family_count": previous_positive,
            "successful_role_candidates_count": previous_successful,
            "mean_role_lift_over_best_baseline": previous_report["mean_role_lift_over_best_baseline"],
        },
        "best_strategy": best,
        "strategy_count": len(strategy_payloads),
        "h2_support_improved": bool(best["role_transfer_lift_vs_raw_m2"] > previous_report["role_transfer_lift_vs_raw_m2"] and best["role_transfer_lift_vs_appearance"] > previous_report["role_transfer_lift_vs_appearance"]),
        "breadth_improved": bool(best["positive_role_lift_families"] > previous_positive),
        "families_newly_improved_vs_v09": newly_improved,
        "families_still_failing": still_failing,
        "roles_transferring_successfully": successful_role_rows[:8],
        "roles_failing_transfer": failing_role_rows[:8],
        "exclude_unknown_effect": {
            "best_include_unknown_strategy": best_include["strategy_name"],
            "best_include_unknown_positive_families": best_include["positive_role_lift_families"],
            "best_exclude_unknown_strategy": best_unknown["strategy_name"],
            "best_exclude_unknown_positive_families": best_unknown["positive_role_lift_families"],
            "exclude_unknown_helped": best_unknown["positive_role_lift_families"] > best_include["positive_role_lift_families"],
        },
        "confidence_coverage_tradeoff": {
            "strategy_name": best["strategy_name"],
            "coverage_rate": best["coverage_rate"],
            "low_confidence_assignment_rate": best["low_confidence_assignment_rate"],
            "precision_on_confident_assignments": best["precision_on_confident_assignments"],
        },
        "v10_gate_cleared": gate_cleared,
        "scientific_conclusion": best["scientific_conclusion"],
    }
    validation = {
        "diagnostic_success": True,
        "scientific_conclusion": best["scientific_conclusion"],
        "proceed_to_v10": gate_cleared,
    }
    return {
        "config": {
            "m3_input_dir": config.m3_input_dir,
            "m2_input_dir": config.m2_input_dir,
            "m1_input_dir": config.m1_input_dir,
            "previous_v09_dir": config.previous_v09_dir,
            "output_dir": config.output_dir,
            "game_set_name": game_set.name,
            "split_mode": config.split_mode,
            "workers": int(config.workers),
        },
        "report": report,
        "validation": validation,
    }


def write_v09b_outputs(
    *,
    output_dir: Path,
    payload: dict[str, Any],
    strategy_payloads: list[dict[str, Any]],
    best_payload: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    subtype_rows: list[dict[str, Any]],
) -> None:
    best_assignments = best_payload["assignments"]
    _write_parquet(output_dir / "v09b_strategy_comparison.parquet", comparison_rows)
    _write_parquet(output_dir / "v09b_best_strategy_assignments.parquet", best_assignments)
    _write_parquet(output_dir / "v09b_transfer_by_family.parquet", best_payload["by_family_rows"])
    _write_parquet(output_dir / "v09b_transfer_by_role.parquet", best_payload["by_role_rows"])
    _write_parquet(output_dir / "v09b_failure_analysis.parquet", best_payload["failures"])
    _write_parquet(output_dir / "v09b_confidence_analysis.parquet", best_payload["confidence_rows"])
    _write_parquet(output_dir / "v09b_subtype_diagnostics.parquet", subtype_rows)
    (output_dir / "v09b_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "v09b_report.txt").write_text(format_v09b_report(payload, comparison_rows), encoding="utf-8")


def format_v09b_report(payload: dict[str, Any], comparison_rows: list[dict[str, Any]]) -> str:
    report = payload["report"]
    previous = report["previous_v09_summary"]
    best = report["best_strategy"]
    strategy_lines = []
    for row in sorted(
        comparison_rows,
        key=lambda item: (-item["positive_role_lift_families"], -item["mean_role_lift_over_best_baseline"], -item["transfer_accuracy_role"], item["strategy_name"]),
    )[:12]:
        strategy_lines.append(
            f"{row['strategy_name']}:pos={row['positive_role_lift_families']}:acc={row['transfer_accuracy_role']:.3f}:lift_best={row['mean_role_lift_over_best_baseline']:.3f}:coverage={row['coverage_rate']:.3f}:low_conf={row['low_confidence_assignment_rate']:.3f}:conclusion={row['scientific_conclusion']}"
        )

    def _fmt_roles(rows: list[dict[str, Any]]) -> str:
        return "; ".join(
            f"{row['role_id']}:{row['role_label_candidate']}:acc={row['transfer_accuracy']:.3f}:succ={row['transfer_success_count']}:fail={row['transfer_failure_count']}"
            for row in rows
        ) if rows else "none"

    return "\n".join(
        [
            "ARC-AGI3 v0.9b-role-transfer-prototype-refinement",
            f"scientific_conclusion={report['scientific_conclusion']}",
            f"v10_gate_cleared={payload['validation']['proceed_to_v10']}",
            f"previous_v09_conclusion={previous['scientific_conclusion']}",
            f"previous_v09_positive_role_lift_family_count={previous['positive_role_lift_family_count']}",
            f"previous_v09_successful_role_candidates_count={previous['successful_role_candidates_count']}",
            f"best_strategy={best['strategy_name']}",
            f"best_strategy_mode={best['prototype_mode']}",
            f"best_strategy_positive_role_lift_families={best['positive_role_lift_families']}",
            f"best_strategy_transfer_accuracy_role={best['transfer_accuracy_role']:.6f}",
            f"best_strategy_transfer_accuracy_raw_m2={best['transfer_accuracy_raw_m2']:.6f}",
            f"best_strategy_transfer_accuracy_appearance_proxy={best['transfer_accuracy_appearance_proxy']:.6f}",
            f"best_strategy_role_transfer_lift_vs_raw_m2={best['role_transfer_lift_vs_raw_m2']:.6f}",
            f"best_strategy_role_transfer_lift_vs_appearance={best['role_transfer_lift_vs_appearance']:.6f}",
            f"best_strategy_mean_role_lift_over_best_baseline={best['mean_role_lift_over_best_baseline']:.6f}",
            f"best_strategy_low_confidence_assignment_rate={best['low_confidence_assignment_rate']:.6f}",
            f"best_strategy_coverage_rate={best['coverage_rate']:.6f}",
            f"best_strategy_precision_on_confident_assignments={best['precision_on_confident_assignments']:.6f}",
            f"h2_support_improved={report['h2_support_improved']}",
            f"breadth_improved={report['breadth_improved']}",
            f"families_newly_improved_vs_v09={','.join(report['families_newly_improved_vs_v09'])}",
            f"families_still_failing={','.join(report['families_still_failing'])}",
            f"roles_transferring_successfully={_fmt_roles(report['roles_transferring_successfully'])}",
            f"roles_failing_transfer={_fmt_roles(report['roles_failing_transfer'])}",
            f"exclude_unknown_helped={report['exclude_unknown_effect']['exclude_unknown_helped']}",
            f"best_include_unknown_strategy={report['exclude_unknown_effect']['best_include_unknown_strategy']}",
            f"best_exclude_unknown_strategy={report['exclude_unknown_effect']['best_exclude_unknown_strategy']}",
            f"confidence_coverage_tradeoff={json.dumps(report['confidence_coverage_tradeoff'], separators=(',', ':'))}",
            "strategy_comparison_top12=",
            *strategy_lines,
        ]
    )
