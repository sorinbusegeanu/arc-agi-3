from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v6.game_sets import GameSetManifest, load_game_set_manifest


@dataclass(frozen=True)
class RoleTransferV09Config:
    m3_input_dir: str = "runs/v6/v08_cd2_extended32_discriminative"
    m2_input_dir: str = "runs/v6/v07_cd2_extended32_expanded"
    m1_input_dir: str = "runs/v6/v06_cd2_extended32"
    output_dir: str = "runs/v6/v09_role_transfer_extended32"
    game_set_manifest: str | None = None
    game_set_name: str | None = None
    split_mode: str = "leave_family_out"
    min_source_role_support: int = 3
    min_target_family_support: int = 3
    workers: int = 25


@dataclass(frozen=True)
class Neighborhood:
    family_id: str
    family_label_candidate: str
    games_present: tuple[str, ...]
    game_families_present: tuple[str, ...]
    support_count: int
    family_coherence: float
    mean_prediction_accuracy: float
    mean_context_lift: float
    dominant_outcome_signature: str
    dominant_motif_candidate: str
    coarse_features: dict[str, float]
    directional_features: dict[str, float]
    future_option_features: dict[str, float]
    local_motif_features: dict[str, float]
    temporal_effect_features: dict[str, float]
    incoming_edge_profile: dict[str, int]
    outgoing_edge_profile: dict[str, int]


@dataclass(frozen=True)
class RoleRecord:
    role_id: str
    role_label_candidate: str
    member_family_ids: tuple[str, ...]
    games_present: tuple[str, ...]
    game_families_present: tuple[str, ...]
    support_count: int
    cross_game_support: int
    cross_game_family_support: int
    role_consistency_score: float
    status: str


@dataclass(frozen=True)
class RolePrototype:
    role_id: str
    role_label_candidate: str
    family_ids: tuple[str, ...]
    all_features: dict[str, float]
    coarse_features: dict[str, float]
    appearance_features: dict[str, float]
    source_games: tuple[str, ...]
    source_families: tuple[str, ...]


def run_role_transfer_v09(config: RoleTransferV09Config) -> dict[str, Any]:
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
    selected_families = tuple(sorted(game_set.families.keys()))
    tasks = [
        (
            heldout_family,
            tuple(game_set.families[heldout_family]),
            neighborhoods,
            roles,
            family_to_role,
            int(config.min_source_role_support),
            int(config.min_target_family_support),
        )
        for heldout_family in selected_families
    ]
    if config.workers <= 1 or len(tasks) <= 1:
        family_results = [_evaluate_heldout_family(*task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = [executor.submit(_evaluate_heldout_family, *task) for task in tasks]
            family_results = [future.result() for future in futures]
    family_results = sorted(family_results, key=lambda item: item["heldout_family"])

    assignment_rows = [row for result in family_results for row in result["assignments"]]
    score_rows = [row for result in family_results for row in result["scores"]]
    failure_rows = [row for result in family_results for row in result["failures"]]
    by_family_rows = build_by_family_rows(family_results)
    by_role_rows = build_by_role_rows(assignment_rows, failure_rows, roles)
    payload = build_payload(
        config=config,
        game_set=game_set,
        roles=roles,
        family_results=family_results,
        assignment_rows=assignment_rows,
        by_family_rows=by_family_rows,
        by_role_rows=by_role_rows,
    )
    write_outputs(
        output_dir=output_dir,
        payload=payload,
        assignment_rows=assignment_rows,
        score_rows=score_rows,
        by_family_rows=by_family_rows,
        by_role_rows=by_role_rows,
        failure_rows=failure_rows,
    )
    return payload


def load_neighborhoods(path: Path) -> dict[str, Neighborhood]:
    df = pd.read_parquet(path)
    output = {}
    for row in df.to_dict(orient="records"):
        output[str(row["family_id"])] = Neighborhood(
            family_id=str(row["family_id"]),
            family_label_candidate=str(row["family_label_candidate"]),
            games_present=tuple(_parse_json_list(row.get("games_present"))),
            game_families_present=tuple(_parse_json_list(row.get("game_families_present"))),
            support_count=int(row["support_count"]),
            family_coherence=float(row["family_coherence"]),
            mean_prediction_accuracy=float(row["mean_prediction_accuracy"]),
            mean_context_lift=float(row["mean_context_lift"]),
            dominant_outcome_signature=str(row["dominant_outcome_signature"]),
            dominant_motif_candidate=str(row["dominant_motif_candidate"]),
            coarse_features=_parse_json_dict(row.get("coarse_features")),
            directional_features=_parse_json_dict(row.get("directional_features")),
            future_option_features=_parse_json_dict(row.get("future_option_features")),
            local_motif_features=_parse_json_dict(row.get("local_motif_features")),
            temporal_effect_features=_parse_json_dict(row.get("temporal_effect_features")),
            incoming_edge_profile=_parse_json_dict(row.get("incoming_edge_profile")),
            outgoing_edge_profile=_parse_json_dict(row.get("outgoing_edge_profile")),
        )
    return output


def load_roles(path: Path) -> list[RoleRecord]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for row in rows:
        output.append(
            RoleRecord(
                role_id=str(row["role_id"]),
                role_label_candidate=str(row["role_label_candidate"]),
                member_family_ids=tuple(_parse_json_list(row.get("member_family_ids"))),
                games_present=tuple(_parse_json_list(row.get("games_present"))),
                game_families_present=tuple(_parse_json_list(row.get("game_families_present"))),
                support_count=int(row["support_count"]),
                cross_game_support=int(row["cross_game_support"]),
                cross_game_family_support=int(row["cross_game_family_support"]),
                role_consistency_score=float(row["role_consistency_score"]),
                status=str(row["status"]),
            )
        )
    return output


def load_family_to_role(roles: list[RoleRecord]) -> dict[str, RoleRecord]:
    mapping = {}
    for role in roles:
        for family_id in role.member_family_ids:
            mapping[family_id] = role
    return mapping


def _evaluate_heldout_family(
    heldout_family: str,
    heldout_games: tuple[str, ...],
    neighborhoods: dict[str, Neighborhood],
    roles: list[RoleRecord],
    family_to_role: dict[str, RoleRecord],
    min_source_role_support: int,
    min_target_family_support: int,
) -> dict[str, Any]:
    target_family_ids = [
        family_id
        for family_id, record in neighborhoods.items()
        if set(record.games_present) & set(heldout_games)
    ]
    source_family_ids = [family_id for family_id in neighborhoods if family_id not in target_family_ids]
    source_neighborhoods = {family_id: neighborhoods[family_id] for family_id in source_family_ids}
    source_role_prototypes = build_source_role_prototypes(roles, source_neighborhoods, min_source_role_support)
    majority_role_id = majority_role(source_role_prototypes)
    assignment_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    per_metric = defaultdict(list)
    for family_id in sorted(target_family_ids):
        target = neighborhoods[family_id]
        target_role = family_to_role.get(family_id)
        if target.support_count < min_target_family_support:
            failure_rows.append(
                {
                    "heldout_family": heldout_family,
                    "target_family_id": family_id,
                    "failure_reason": "insufficient_target_support",
                }
            )
            continue
        if target_role is None or target_role.status != "stable":
            failure_rows.append(
                {
                    "heldout_family": heldout_family,
                    "target_family_id": family_id,
                    "failure_reason": "no_stable_ground_truth_role",
                }
            )
            continue
        if target_role.role_id not in source_role_prototypes:
            failure_rows.append(
                {
                    "heldout_family": heldout_family,
                    "target_family_id": family_id,
                    "failure_reason": "no_transferable_source_role",
                    "ground_truth_role_id": target_role.role_id,
                }
            )
            continue
        assigned_role_id, assigned_similarity = nearest_role(target, source_role_prototypes, mode="all")
        coarse_role_id, coarse_similarity = nearest_role(target, source_role_prototypes, mode="coarse")
        appearance_role_id, appearance_similarity = nearest_role(target, source_role_prototypes, mode="appearance")
        raw_m2_role_id, raw_m2_similarity = nearest_raw_m2(target, source_neighborhoods, family_to_role)
        random_role_id = deterministic_random_role_id(source_role_prototypes, heldout_family, family_id)
        majority_role_id = majority_role_id
        result_row = {
            "heldout_family": heldout_family,
            "target_family_id": family_id,
            "target_games": list(target.games_present),
            "target_game_families": list(target.game_families_present),
            "ground_truth_role_id": target_role.role_id,
            "ground_truth_role_label": target_role.role_label_candidate,
            "assigned_role_id": assigned_role_id,
            "assigned_role_label": source_role_prototypes[assigned_role_id].role_label_candidate,
            "assigned_similarity": assigned_similarity,
            "coarse_role_id": coarse_role_id,
            "coarse_similarity": coarse_similarity,
            "appearance_role_id": appearance_role_id,
            "appearance_similarity": appearance_similarity,
            "raw_m2_role_id": raw_m2_role_id,
            "raw_m2_similarity": raw_m2_similarity,
            "majority_role_id": majority_role_id,
            "random_role_id": random_role_id,
            "role_correct": int(assigned_role_id == target_role.role_id),
            "coarse_correct": int(coarse_role_id == target_role.role_id),
            "appearance_correct": int(appearance_role_id == target_role.role_id),
            "raw_m2_correct": int(raw_m2_role_id == target_role.role_id),
            "majority_correct": int(majority_role_id == target_role.role_id),
            "random_correct": int(random_role_id == target_role.role_id),
            "cross_family_transfer_score": structural_transfer_score(target, source_role_prototypes[assigned_role_id]),
        }
        assignment_rows.append(result_row)
        for metric_name, value in (
            ("role", result_row["role_correct"]),
            ("random", result_row["random_correct"]),
            ("majority", result_row["majority_correct"]),
            ("raw_m2", result_row["raw_m2_correct"]),
            ("coarse", result_row["coarse_correct"]),
            ("appearance", result_row["appearance_correct"]),
        ):
            per_metric[metric_name].append(int(value))
        score_rows.append(
            {
                "heldout_family": heldout_family,
                "target_family_id": family_id,
                "mean_target_similarity_to_assigned_role": assigned_similarity,
                "mean_cross_family_transfer_score": result_row["cross_family_transfer_score"],
            }
        )

    role_accuracy = float(np.mean(per_metric["role"])) if per_metric["role"] else 0.0
    baseline_scores = {key: float(np.mean(values)) if values else 0.0 for key, values in per_metric.items() if key != "role"}
    best_baseline_accuracy = max(baseline_scores.values(), default=0.0)
    return {
        "heldout_family": heldout_family,
        "source_role_candidates_count": len(source_role_prototypes),
        "target_games": sorted({game for row in assignment_rows for game in row["target_games"]}),
        "target_m2_families": len(assignment_rows),
        "transferable_role_candidates_count": len({row["ground_truth_role_id"] for row in assignment_rows}),
        "role_transfer_accuracy": role_accuracy,
        "baseline_scores": baseline_scores,
        "best_baseline_accuracy": best_baseline_accuracy,
        "role_lift_over_best_baseline": role_accuracy - best_baseline_accuracy,
        "assignments": assignment_rows,
        "scores": score_rows,
        "failures": failure_rows,
        "best_transferring_roles": top_roles_for_family(assignment_rows, best=True),
        "weakest_transferring_roles": top_roles_for_family(assignment_rows, best=False),
    }


def build_source_role_prototypes(
    roles: list[RoleRecord],
    source_neighborhoods: dict[str, Neighborhood],
    min_source_role_support: int,
) -> dict[str, RolePrototype]:
    prototypes = {}
    for role in roles:
        if role.status != "stable":
            continue
        member_records = [source_neighborhoods[item] for item in role.member_family_ids if item in source_neighborhoods]
        if len(member_records) < min_source_role_support:
            continue
        prototypes[role.role_id] = RolePrototype(
            role_id=role.role_id,
            role_label_candidate=role.role_label_candidate,
            family_ids=tuple(sorted(record.family_id for record in member_records)),
            all_features=mean_vector([all_features(record) for record in member_records]),
            coarse_features=mean_vector([record.coarse_features for record in member_records]),
            appearance_features=mean_vector([appearance_features(record) for record in member_records]),
            source_games=tuple(sorted({game for record in member_records for game in record.games_present})),
            source_families=tuple(sorted({family for record in member_records for family in record.game_families_present})),
        )
    return prototypes


def nearest_role(target: Neighborhood, prototypes: dict[str, RolePrototype], mode: str) -> tuple[str, float]:
    target_vector = {
        "all": all_features(target),
        "coarse": target.coarse_features,
        "appearance": appearance_features(target),
    }[mode]
    best = None
    for role_id, prototype in sorted(prototypes.items()):
        source_vector = {
            "all": prototype.all_features,
            "coarse": prototype.coarse_features,
            "appearance": prototype.appearance_features,
        }[mode]
        similarity = cosine_similarity(target_vector, source_vector)
        candidate = (similarity, role_id)
        if best is None or candidate > (best[0], best[1]):
            best = (similarity, role_id)
    if best is None:
        return ("", 0.0)
    return best[1], float(best[0])


def nearest_raw_m2(target: Neighborhood, source_neighborhoods: dict[str, Neighborhood], family_to_role: dict[str, RoleRecord]) -> tuple[str, float]:
    best = None
    target_vector = raw_m2_features(target)
    for family_id, source in sorted(source_neighborhoods.items()):
        source_role = family_to_role.get(family_id)
        if source_role is None or source_role.status != "stable":
            continue
        similarity = cosine_similarity(target_vector, raw_m2_features(source))
        candidate = (similarity, source_role.role_id)
        if best is None or candidate > (best[0], best[1]):
            best = (similarity, source_role.role_id)
    if best is None:
        return ("", 0.0)
    return best[1], float(best[0])


def structural_transfer_score(target: Neighborhood, prototype: RolePrototype) -> float:
    return cosine_similarity(all_features(target), prototype.all_features)


def majority_role(prototypes: dict[str, RolePrototype]) -> str:
    if not prototypes:
        return ""
    return max(sorted(prototypes), key=lambda role_id: len(prototypes[role_id].family_ids))


def deterministic_random_role_id(prototypes: dict[str, RolePrototype], heldout_family: str, family_id: str) -> str:
    if not prototypes:
        return ""
    role_ids = sorted(prototypes)
    digest = hashlib.sha256(f"{heldout_family}|{family_id}".encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(role_ids)
    return role_ids[index]


def all_features(record: Neighborhood) -> dict[str, float]:
    output = {}
    for prefix, features in (
        ("coarse", record.coarse_features),
        ("directional", record.directional_features),
        ("future", record.future_option_features),
        ("motif", record.local_motif_features),
        ("temporal", record.temporal_effect_features),
    ):
        for key, value in features.items():
            output[f"{prefix}:{key}"] = float(value)
    return output


def raw_m2_features(record: Neighborhood) -> dict[str, float]:
    output = {
        "family_coherence": record.family_coherence,
        "mean_prediction_accuracy": record.mean_prediction_accuracy,
        "mean_context_lift": record.mean_context_lift,
        f"label::{record.family_label_candidate}": 1.0,
        f"outcome::{record.dominant_outcome_signature}": 1.0,
        f"motif::{record.dominant_motif_candidate}": 1.0,
    }
    return output


def appearance_features(record: Neighborhood) -> dict[str, float]:
    effect = record.temporal_effect_features
    return {
        f"label::{record.family_label_candidate}": 1.0,
        f"outcome::{record.dominant_outcome_signature}": 1.0,
        "no_change_rate": float(effect.get("no_change_rate", 0.0)),
        "position_change_rate": float(effect.get("position_change_rate", 0.0)),
        "discontinuous_position_change_rate": float(effect.get("discontinuous_position_change_rate", 0.0)),
        "terminal_rate": float(effect.get("terminal_rate", 0.0)),
        "multi_cell_change_rate": float(effect.get("multi_cell_change_rate", 0.0)),
        "coverage_change_rate": float(effect.get("coverage_change_rate_if_derivable", 0.0)),
        "repeated_toggle_like_rate": float(effect.get("repeated_toggle_like_rate", 0.0)),
    }


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    lv = np.asarray([float(left.get(key, 0.0)) for key in keys], dtype=float)
    rv = np.asarray([float(right.get(key, 0.0)) for key in keys], dtype=float)
    denom = float(np.linalg.norm(lv) * np.linalg.norm(rv))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(lv, rv) / denom)


def mean_vector(vectors: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for vector in vectors for key in vector})
    return {key: float(np.mean([float(vector.get(key, 0.0)) for vector in vectors])) for key in keys}


def build_by_family_rows(family_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in family_results:
        rows.append(
            {
                "heldout_family": result["heldout_family"],
                "source_family_count": result["source_role_candidates_count"],
                "target_games": result["target_games"],
                "target_m2_families": result["target_m2_families"],
                "evaluable": bool(result["target_m2_families"] > 0),
                "best_transferring_roles": result["best_transferring_roles"],
                "weakest_transferring_roles": result["weakest_transferring_roles"],
                "role_transfer_accuracy": result["role_transfer_accuracy"],
                "best_baseline_accuracy": result["best_baseline_accuracy"],
                "role_lift_over_best_baseline": result["role_lift_over_best_baseline"],
            }
        )
    return rows


def top_roles_for_family(assignment_rows: list[dict[str, Any]], *, best: bool) -> list[dict[str, Any]]:
    by_role = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in assignment_rows:
        role_id = row["ground_truth_role_id"]
        by_role[role_id]["correct"] += int(row["role_correct"])
        by_role[role_id]["total"] += 1
    rows = [
        {
            "role_id": role_id,
            "transfer_accuracy": values["correct"] / max(1, values["total"]),
        }
        for role_id, values in by_role.items()
    ]
    return sorted(rows, key=lambda item: ((-1 if best else 1) * item["transfer_accuracy"], item["role_id"]))[:5]


def build_by_role_rows(assignment_rows: list[dict[str, Any]], failure_rows: list[dict[str, Any]], roles: list[RoleRecord]) -> list[dict[str, Any]]:
    successes = defaultdict(list)
    failures = defaultdict(list)
    for row in assignment_rows:
        key = row["ground_truth_role_id"]
        if row["role_correct"]:
            successes[key].append(row)
        else:
            failures[key].append(row)
    rows = []
    for role in roles:
        if role.status != "stable":
            continue
        success_rows = successes.get(role.role_id, [])
        failure_role_rows = failures.get(role.role_id, [])
        rows.append(
            {
                "role_id": role.role_id,
                "role_label_candidate": role.role_label_candidate,
                "source_games": list(role.games_present),
                "source_families": list(role.game_families_present),
                "target_games_matched": sorted({game for row in success_rows for game in row["target_games"]}),
                "target_families_matched": sorted({row["heldout_family"] for row in success_rows}),
                "transfer_success_count": len(success_rows),
                "transfer_failure_count": len(failure_role_rows),
                "transfer_accuracy": len(success_rows) / max(1, len(success_rows) + len(failure_role_rows)),
                "best_target_examples": success_rows[:5],
                "failure_examples": failure_role_rows[:5],
            }
        )
    return rows


def build_payload(
    *,
    config: RoleTransferV09Config,
    game_set: GameSetManifest,
    roles: list[RoleRecord],
    family_results: list[dict[str, Any]],
    assignment_rows: list[dict[str, Any]],
    by_family_rows: list[dict[str, Any]],
    by_role_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluable_family_rows = [row for row in by_family_rows if row["target_m2_families"] > 0]
    role_acc = mean_metric(family_results, "role_transfer_accuracy")
    random_acc = mean_baseline(family_results, "random")
    majority_acc = mean_baseline(family_results, "majority")
    raw_m2_acc = mean_baseline(family_results, "raw_m2")
    coarse_acc = mean_baseline(family_results, "coarse")
    appearance_acc = mean_baseline(family_results, "appearance")
    positive_families = sum(1 for row in evaluable_family_rows if row["role_lift_over_best_baseline"] > 0)
    successful_roles = sum(1 for row in by_role_rows if row["transfer_success_count"] > row["transfer_failure_count"])
    mean_lift_best = float(np.mean([row["role_lift_over_best_baseline"] for row in evaluable_family_rows])) if evaluable_family_rows else 0.0
    success_counts = [row["transfer_success_count"] for row in by_role_rows]
    dominant_share = (max(success_counts) / max(1, sum(success_counts))) if success_counts else 1.0
    non_evaluable_families = sorted([row["heldout_family"] for row in by_family_rows if row["target_m2_families"] <= 0])
    if (
        role_acc > random_acc
        and role_acc > majority_acc
        and role_acc > raw_m2_acc
        and role_acc > coarse_acc
        and role_acc > appearance_acc
        and positive_families >= 12
        and successful_roles >= 7
        and mean_lift_best >= 0.10
        and dominant_share <= 0.40
    ):
        conclusion = "role_transfer_very_strong"
    elif (
        role_acc > random_acc
        and role_acc > majority_acc
        and role_acc > raw_m2_acc
        and role_acc > coarse_acc
        and role_acc > appearance_acc
        and positive_families >= 8
        and successful_roles >= 5
        and mean_lift_best >= 0.05
    ):
        conclusion = "role_transfer_strong"
    elif (
        role_acc > random_acc
        and role_acc > majority_acc
        and role_acc > appearance_acc
        and positive_families >= 6
        and successful_roles >= 3
    ):
        conclusion = "role_transfer_weak"
    else:
        conclusion = "role_transfer_not_established"

    report = {
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
        "mean_target_similarity_to_assigned_role": float(np.mean([row["assigned_similarity"] for row in assignment_rows])) if assignment_rows else 0.0,
        "mean_cross_family_transfer_score": float(np.mean([row["cross_family_transfer_score"] for row in assignment_rows])) if assignment_rows else 0.0,
        "heldout_family_count": len(family_results),
        "evaluable_heldout_family_count": len(evaluable_family_rows),
        "target_games_count": len({game for row in assignment_rows for game in row["target_games"]}),
        "target_m2_families_count": len(assignment_rows),
        "source_role_candidates_count": len([role for role in roles if role.status == "stable"]),
        "transferable_role_candidates_count": len([row for row in by_role_rows if row["transfer_success_count"] + row["transfer_failure_count"] > 0]),
        "positive_role_lift_family_count": positive_families,
        "successful_role_candidates_count": successful_roles,
        "mean_role_lift_over_best_baseline": mean_lift_best,
        "scientific_conclusion": conclusion,
        "supports_H2": role_acc > appearance_acc and role_acc > raw_m2_acc,
        "h2_interpretation": (
            "supported_on_average_but_not_broad_enough"
            if conclusion == "role_transfer_not_established" and role_acc > appearance_acc and role_acc > raw_m2_acc
            else ("supported" if role_acc > appearance_acc and role_acc > raw_m2_acc else "not_supported")
        ),
        "best_transferring_roles": sorted(by_role_rows, key=lambda item: (-item["transfer_accuracy"], -item["transfer_success_count"], item["role_id"]))[:5],
        "weak_or_non_transferring_roles": sorted(by_role_rows, key=lambda item: (item["transfer_accuracy"], item["role_id"]))[:5],
        "families_where_transfer_works": sorted([row["heldout_family"] for row in evaluable_family_rows if row["role_lift_over_best_baseline"] > 0]),
        "families_where_transfer_fails": sorted([row["heldout_family"] for row in evaluable_family_rows if row["role_lift_over_best_baseline"] <= 0]),
        "families_not_evaluable": non_evaluable_families,
        "cross_family_transfer_dominant": sum(1 for row in by_role_rows if len(row["source_families"]) >= 2 and len(row["target_families_matched"]) >= 1),
        "appearance_proxy_beats_role_anywhere": any(result["baseline_scores"].get("appearance", 0.0) > result["role_transfer_accuracy"] for result in family_results),
        "recommendation_for_v10": (
            "proceed_to_v10" if conclusion in {"role_transfer_weak", "role_transfer_strong", "role_transfer_very_strong"} else "stop_and_redesign_transfer"
        ),
    }
    validation = {
        "diagnostic_success": bool(assignment_rows),
        "scientific_conclusion": conclusion,
    }
    return {
        "config": {
            "m3_input_dir": config.m3_input_dir,
            "m2_input_dir": config.m2_input_dir,
            "m1_input_dir": config.m1_input_dir,
            "output_dir": config.output_dir,
            "game_set_name": game_set.name,
            "split_mode": config.split_mode,
            "workers": int(config.workers),
        },
        "report": report,
        "validation": validation,
    }


def mean_metric(results: list[dict[str, Any]], field_name: str) -> float:
    values = [float(result[field_name]) for result in results if result["target_m2_families"] > 0]
    return float(np.mean(values)) if values else 0.0


def mean_baseline(results: list[dict[str, Any]], baseline_name: str) -> float:
    values = [float(result["baseline_scores"].get(baseline_name, 0.0)) for result in results if result["target_m2_families"] > 0]
    return float(np.mean(values)) if values else 0.0


def write_outputs(
    *,
    output_dir: Path,
    payload: dict[str, Any],
    assignment_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    by_family_rows: list[dict[str, Any]],
    by_role_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> None:
    _write_parquet(output_dir / "role_transfer_assignments.parquet", assignment_rows)
    _write_parquet(output_dir / "role_transfer_scores.parquet", score_rows)
    _write_parquet(output_dir / "role_transfer_by_family.parquet", by_family_rows)
    _write_parquet(output_dir / "role_transfer_by_role.parquet", by_role_rows)
    _write_parquet(output_dir / "role_transfer_failures.parquet", failure_rows)
    (output_dir / "v09_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "v09_report.txt").write_text(format_report(payload), encoding="utf-8")


def format_report(payload: dict[str, Any]) -> str:
    report = payload["report"]
    def _fmt_role_rows(rows: list[dict[str, Any]]) -> str:
        parts = []
        for row in rows:
            parts.append(
                f"{row['role_id']}:{row['role_label_candidate']}:acc={row['transfer_accuracy']:.3f}:succ={row['transfer_success_count']}:fail={row['transfer_failure_count']}"
            )
        return "; ".join(parts) if parts else "none"

    return "\n".join(
        [
            "ARC-AGI3 v0.9-role-transfer-validation",
            f"scientific_conclusion={payload['validation']['scientific_conclusion']}",
            f"supports_H2={report['supports_H2']}",
            f"h2_interpretation={report['h2_interpretation']}",
            "summary=role similarity is evaluated by leave-family-out transfer and compared against random, majority, raw_m2, coarse fingerprint, and appearance baselines",
            f"transfer_accuracy_role={report['transfer_accuracy_role']:.6f}",
            f"transfer_accuracy_random={report['transfer_accuracy_random']:.6f}",
            f"transfer_accuracy_majority={report['transfer_accuracy_majority']:.6f}",
            f"transfer_accuracy_raw_m2={report['transfer_accuracy_raw_m2']:.6f}",
            f"transfer_accuracy_coarse={report['transfer_accuracy_coarse']:.6f}",
            f"transfer_accuracy_appearance_proxy={report['transfer_accuracy_appearance_proxy']:.6f}",
            f"role_transfer_lift_vs_random={report['role_transfer_lift_vs_random']:.6f}",
            f"role_transfer_lift_vs_majority={report['role_transfer_lift_vs_majority']:.6f}",
            f"role_transfer_lift_vs_raw_m2={report['role_transfer_lift_vs_raw_m2']:.6f}",
            f"role_transfer_lift_vs_coarse={report['role_transfer_lift_vs_coarse']:.6f}",
            f"role_transfer_lift_vs_appearance={report['role_transfer_lift_vs_appearance']:.6f}",
            f"heldout_family_count={report['heldout_family_count']}",
            f"evaluable_heldout_family_count={report['evaluable_heldout_family_count']}",
            f"target_m2_families_count={report['target_m2_families_count']}",
            f"successful_role_candidates_count={report['successful_role_candidates_count']}",
            f"positive_role_lift_family_count={report['positive_role_lift_family_count']}",
            f"mean_role_lift_over_best_baseline={report['mean_role_lift_over_best_baseline']:.6f}",
            f"best_transferring_roles={_fmt_role_rows(report['best_transferring_roles'])}",
            f"weak_or_non_transferring_roles={_fmt_role_rows(report['weak_or_non_transferring_roles'])}",
            f"families_where_transfer_works={','.join(report['families_where_transfer_works'])}",
            f"families_where_transfer_fails={','.join(report['families_where_transfer_fails'])}",
            f"families_not_evaluable={','.join(report['families_not_evaluable'])}",
            f"cross_family_transfer_dominant={report['cross_family_transfer_dominant']}",
            f"appearance_proxy_vs_role=appearance_proxy_beats_role_anywhere:{report['appearance_proxy_beats_role_anywhere']}",
            "interpretation=if role similarity beats appearance and raw_m2 on average but fails the breadth threshold, treat this as partial transfer rather than a milestone pass",
            f"appearance_proxy_beats_role_anywhere={report['appearance_proxy_beats_role_anywhere']}",
            f"recommendation_for_v10={report['recommendation_for_v10']}",
        ]
    )


def _write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    normalized = [_normalize_record(record) for record in records]
    table = pa.Table.from_pylist(normalized) if normalized else pa.table({"_empty": pa.array([], type=pa.string())})
    pq.write_table(table, path, compression="zstd")


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for key, value in record.items():
        if isinstance(value, (list, tuple, dict, Counter)):
            output[key] = json.dumps(value)
        else:
            output[key] = value
    return output


def _parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _parse_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): float(value_item) if isinstance(value_item, (int, float)) else value_item for key, value_item in value.items()}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}
