from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from v6.evaluation.broad_game_validation import family_for_game
from v6.game_sets import GameSetManifest, load_game_set_manifest
from v6.memory_types import M3RoleCandidate


@dataclass(frozen=True)
class RoleCandidatesV08dConfig:
    input_dir: str = "runs/v6/v07_cd2_extended32_expanded"
    m1_input_dir: str = "runs/v6/v06_cd2_extended32"
    output_dir: str = "runs/v6/v08_cd2_extended32_discriminative"
    context_depth: int = 2
    min_role_support: int = 3
    role_similarity_threshold: float = 0.70
    workers: int = 25
    partition_by: tuple[str, ...] = ("family_pair", "neighborhood_shard")
    game_set_manifest: str | None = None
    game_set_name: str | None = None
    fingerprint_mode: str = "discriminative"
    weight_coarse: float = 0.25
    weight_directional: float = 0.20
    weight_future_option: float = 0.25
    weight_local_motif: float = 0.20
    weight_temporal_effect: float = 0.10


@dataclass(frozen=True)
class M1SupportRecord:
    contingency_id: str
    game_id: str
    sampler_scope: str
    action: int
    outcome_signature: str
    prediction_accuracy: float
    entropy: float
    support_count: int
    total_count: int
    future_option_motif_candidate: str
    context_signature: tuple[str, ...]
    first_seen_step: int
    last_seen_step: int
    context_lift: float


@dataclass(frozen=True)
class M2FamilyRecord:
    family_id: str
    family_label_candidate: str
    games_present: tuple[str, ...]
    samplers_present: tuple[str, ...]
    contingency_ids: tuple[str, ...]
    support_count: int
    mean_prediction_accuracy: float
    mean_context_lift: float
    dominant_outcome_signature: str
    outcome_signature_distribution: dict[str, int]
    motif_candidate_distribution: dict[str, int]
    family_coherence: float
    compression_ratio: float
    cross_game_presence: int
    stable: bool
    examples: tuple[dict[str, Any], ...]
    notes: dict[str, Any]


@dataclass(frozen=True)
class DiscNeighborhood:
    family_id: str
    family_label_candidate: str
    game_ids: tuple[str, ...]
    game_family_ids: tuple[str, ...]
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
    examples: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PairResult:
    left_family_id: str
    right_family_id: str
    coarse_similarity: float
    directional_similarity: float
    future_option_similarity: float
    local_motif_similarity: float
    temporal_effect_similarity: float
    weighted_similarity: float


@dataclass(frozen=True)
class SimilarityWeights:
    coarse: float
    directional: float
    future_option: float
    local_motif: float
    temporal_effect: float

    @property
    def total(self) -> float:
        return self.coarse + self.directional + self.future_option + self.local_motif + self.temporal_effect


def run_role_candidates_v08d(config: RoleCandidatesV08dConfig) -> dict[str, Any]:
    input_dir = Path(config.input_dir)
    m1_input_dir = Path(config.m1_input_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = SimilarityWeights(
        coarse=float(config.weight_coarse),
        directional=float(config.weight_directional),
        future_option=float(config.weight_future_option),
        local_motif=float(config.weight_local_motif),
        temporal_effect=float(config.weight_temporal_effect),
    )

    game_set = load_game_set(config)
    m2_families = load_m2_families(input_dir)
    m1_support = load_m1_support(m1_input_dir)
    available_games = {game for family in m2_families for game in family.games_present}
    selected_games = tuple(game for game in game_set.games if game in available_games) if game_set.games else tuple(sorted(available_games))
    selected_game_set = set(selected_games)
    m2_families = [family for family in m2_families if selected_game_set.intersection(family.games_present)]
    m1_support = {key: value for key, value in m1_support.items() if value.game_id in selected_game_set}
    game_family_map = build_game_family_map(game_set, selected_games)

    neighborhoods = build_discriminative_neighborhoods(m2_families, m1_support, game_family_map)
    pair_results = evaluate_pairwise_similarity(neighborhoods, weights=weights, threshold=config.role_similarity_threshold, workers=config.workers)
    adjacency = build_similarity_adjacency(pair_results, config.role_similarity_threshold)
    clusters, rejected = cluster_role_candidates(adjacency, pair_results, neighborhoods, config.role_similarity_threshold)
    roles = build_role_candidates(
        clusters=clusters,
        neighborhoods=neighborhoods,
        min_role_support=config.min_role_support,
        role_similarity_threshold=config.role_similarity_threshold,
        pair_results=pair_results,
    )
    graph_nodes, graph_edges = build_graph_outputs(neighborhoods, roles, pair_results, m2_families, m1_support)
    membership_rows = role_membership_rows(roles)
    neighborhood_rows = neighborhood_output_rows(neighborhoods)
    similarity_rows = similarity_output_rows(pair_results)
    cluster_rows = cluster_diagnostics_rows(clusters, neighborhoods, pair_results, rejected)
    payload = build_v08d_payload(
        config=config,
        game_set=game_set,
        selected_games=selected_games,
        neighborhoods=neighborhoods,
        pair_results=pair_results,
        roles=roles,
        rejected_clusters=rejected,
    )

    write_outputs(
        output_dir=output_dir,
        roles=roles,
        membership_rows=membership_rows,
        neighborhood_rows=neighborhood_rows,
        similarity_rows=similarity_rows,
        cluster_rows=cluster_rows,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        payload=payload,
    )
    return payload


def load_game_set(config: RoleCandidatesV08dConfig) -> GameSetManifest:
    return load_game_set_manifest(
        manifest_path=config.game_set_manifest,
        game_set_name=config.game_set_name,
        fallback_games=(),
    )


def build_game_family_map(game_set: GameSetManifest, games: tuple[str, ...]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for family_name, family_games in game_set.families.items():
        for game in family_games:
            mapping.setdefault(game, family_name)
    for game in games:
        mapping.setdefault(game, family_for_game(game))
    return mapping


def load_m2_families(input_dir: Path) -> list[M2FamilyRecord]:
    rows = json.loads((input_dir / "m2_families.json").read_text(encoding="utf-8"))
    output = []
    for row in rows:
        output.append(
            M2FamilyRecord(
                family_id=str(row["family_id"]),
                family_label_candidate=str(row["family_label_candidate"]),
                games_present=tuple(sorted(_parse_json_list(row.get("games_present")))),
                samplers_present=tuple(sorted(_parse_json_list(row.get("samplers_present")))),
                contingency_ids=tuple(_parse_json_list(row.get("contingency_ids"))),
                support_count=int(row["support_count"]),
                mean_prediction_accuracy=float(row["mean_prediction_accuracy"]),
                mean_context_lift=float(row["mean_context_lift"]),
                dominant_outcome_signature=str(row["dominant_outcome_signature"]),
                outcome_signature_distribution=_parse_json_dict(row.get("outcome_signature_distribution")),
                motif_candidate_distribution=_parse_json_dict(row.get("motif_candidate_distribution")),
                family_coherence=float(row["family_coherence"]),
                compression_ratio=float(row["compression_ratio"]),
                cross_game_presence=int(row["cross_game_presence"]),
                stable=bool(row["stable"]),
                examples=tuple(_parse_json_list(row.get("examples"))),
                notes=_parse_json_dict(row.get("notes")),
            )
        )
    return output


def load_m1_support(input_dir: Path) -> dict[str, M1SupportRecord]:
    rows = json.loads((input_dir / "contingencies.json").read_text(encoding="utf-8"))
    baselines = compute_action_baselines(rows)
    output = {}
    for row in rows:
        key = (str(row["game_id"]), int(row["action"]))
        record = M1SupportRecord(
            contingency_id=str(row["contingency_id"]),
            game_id=str(row["game_id"]),
            sampler_scope=str(row["sampler_scope"]),
            action=int(row["action"]),
            outcome_signature=str(row["outcome_signature"]),
            prediction_accuracy=float(row["prediction_accuracy"]),
            entropy=float(row["entropy"]),
            support_count=int(row["support_count"]),
            total_count=int(row["total_count"]),
            future_option_motif_candidate=str(row["future_option_motif_candidate"]),
            context_signature=tuple(_parse_json_list(row.get("context_signature"))),
            first_seen_step=int(row.get("first_seen_step", 0)),
            last_seen_step=int(row.get("last_seen_step", 0)),
            context_lift=float(row["prediction_accuracy"]) - baselines.get(key, 0.0),
        )
        output[record.contingency_id] = record
    return output


def compute_action_baselines(rows: list[dict[str, Any]]) -> dict[tuple[str, int], float]:
    counts: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    totals: Counter[tuple[str, int]] = Counter()
    for row in rows:
        key = (str(row["game_id"]), int(row["action"]))
        counts[key][str(row["outcome_signature"])] += int(row["total_count"])
        totals[key] += int(row["total_count"])
    output = {}
    for key, counter in counts.items():
        output[key] = max(counter.values()) / max(1, totals[key])
    return output


def build_discriminative_neighborhoods(
    families: list[M2FamilyRecord],
    m1_support: dict[str, M1SupportRecord],
    game_family_map: dict[str, str],
) -> dict[str, DiscNeighborhood]:
    family_members = {
        family.family_id: [m1_support[item] for item in family.contingency_ids if item in m1_support]
        for family in families
    }
    game_token_emitters: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for family in families:
        for member in family_members[family.family_id]:
            game_token_emitters[member.game_id][emitted_token(member)][family.family_id] += 1

    predecessor_counts: dict[str, Counter[str]] = defaultdict(Counter)
    successor_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for family in families:
        for member in family_members[family.family_id]:
            for token in member.context_signature:
                for predecessor_family_id, count in game_token_emitters[member.game_id].get(token, {}).items():
                    predecessor_counts[family.family_id][predecessor_family_id] += count
                    successor_counts[predecessor_family_id][family.family_id] += count

    max_step_by_game: dict[str, int] = defaultdict(int)
    for member in m1_support.values():
        max_step_by_game[member.game_id] = max(max_step_by_game[member.game_id], member.last_seen_step)

    precomputed = {}
    for family in families:
        precomputed[family.family_id] = precompute_family_profiles(
            family=family,
            members=family_members[family.family_id],
            predecessors=predecessor_counts[family.family_id],
            successors=successor_counts[family.family_id],
            family_members=family_members,
            family_index={item.family_id: item for item in families},
            game_family_map=game_family_map,
            max_step_by_game=max_step_by_game,
        )

    normalized = normalize_feature_groups(precomputed)
    neighborhoods: dict[str, DiscNeighborhood] = {}
    for family in families:
        members = family_members[family.family_id]
        dominant_motif = dominant_key(family.motif_candidate_distribution)
        games = tuple(sorted(family.games_present))
        game_families = tuple(sorted({game_family_map.get(game, "unknown") for game in games}))
        profiles = normalized[family.family_id]
        neighborhoods[family.family_id] = DiscNeighborhood(
            family_id=family.family_id,
            family_label_candidate=family.family_label_candidate,
            game_ids=games,
            game_family_ids=game_families,
            support_count=family.support_count,
            family_coherence=family.family_coherence,
            mean_prediction_accuracy=family.mean_prediction_accuracy,
            mean_context_lift=family.mean_context_lift,
            dominant_outcome_signature=family.dominant_outcome_signature,
            dominant_motif_candidate=dominant_motif,
            coarse_features=profiles["coarse"],
            directional_features=profiles["directional"],
            future_option_features=profiles["future_option"],
            local_motif_features=profiles["local_motif"],
            temporal_effect_features=profiles["temporal_effect"],
            incoming_edge_profile=dict(sorted(precomputed[family.family_id]["incoming_edge_profile"].items())),
            outgoing_edge_profile=dict(sorted(precomputed[family.family_id]["outgoing_edge_profile"].items())),
            examples=tuple(family.examples),
        )
    return neighborhoods


def precompute_family_profiles(
    *,
    family: M2FamilyRecord,
    members: list[M1SupportRecord],
    predecessors: Counter[str],
    successors: Counter[str],
    family_members: dict[str, list[M1SupportRecord]],
    family_index: dict[str, M2FamilyRecord],
    game_family_map: dict[str, str],
    max_step_by_game: dict[str, int],
) -> dict[str, Any]:
    outcome_dist = normalized_counter(family.outcome_signature_distribution)
    motif_dist = normalized_counter(family.motif_candidate_distribution)
    action_counts = Counter(member.action for member in members)
    context_depth_counts = Counter(len(member.context_signature) for member in members)
    interactive_ratio = sum(count for action, count in action_counts.items() if action in {5, 6}) / max(1, sum(action_counts.values()))
    directional_ratio = sum(count for action, count in action_counts.items() if action in {1, 2, 3, 4}) / max(1, sum(action_counts.values()))
    pred_total = sum(predecessors.values())
    succ_total = sum(successors.values())
    unique_pred = len(predecessors)
    unique_succ = len(successors)
    pred_entropy = entropy(predecessors)
    succ_entropy = entropy(successors)
    reciprocity = reciprocal_successor_ratio(family.family_id, predecessors, successors)
    position_rate = outcome_dist.get("position_like_change", 0.0)
    no_change_rate = outcome_dist.get("blocked_no_change", 0.0) + outcome_dist.get("preserve_no_change", 0.0)
    terminal_rate = outcome_dist.get("terminal_transition", 0.0)
    large_change_rate = outcome_dist.get("large_change", 0.0)
    change_rate = outcome_dist.get("change", 0.0) + large_change_rate
    discontinuous_position_rate = position_rate * (0.8 if family.family_label_candidate in {"teleport_like_family_candidate", "push_like_family_candidate"} else 0.25 + 0.5 * interactive_ratio)
    coverage_change_rate = (
        1.0 if family.family_label_candidate == "coverage_change_family_candidate" else min(1.0, large_change_rate + 0.5 * change_rate * max(0.0, family.mean_context_lift))
    )
    successor_terminal = mean_neighbor_metric(successors, family_index, lambda item: normalized_counter(item.outcome_signature_distribution).get("terminal_transition", 0.0))
    enable_score = max(0.0, (unique_succ - unique_pred) / max(1.0, unique_succ + unique_pred)) + max(0.0, family.mean_context_lift)
    block_score = no_change_rate + max(0.0, (unique_pred - unique_succ) / max(1.0, unique_succ + unique_pred))
    terminate_score = terminal_rate + (1.0 if unique_succ == 0 else 0.0) * 0.5 + successor_terminal * 0.5
    preserve_score = max(0.0, 1.0 - abs(unique_succ - unique_pred) / max(1.0, unique_succ + unique_pred)) * no_change_rate
    reversibility_score = min(1.0, reciprocity + loop_2cycle_score(family.family_id, predecessors, successors))
    chain_position = float(unique_pred > 0 and unique_succ > 0)
    branch_in = float(unique_pred >= 2)
    branch_out = float(unique_succ >= 2)
    bottleneck = float(unique_pred >= 2 and unique_succ >= 2)
    loop2 = loop_2cycle_score(family.family_id, predecessors, successors)
    loop3 = loop_3cycle_score(family.family_id, successors, family_index, family_members)
    sink = float(unique_succ == 0)
    source = float(unique_pred == 0)
    bridge = float(unique_pred > 0 and unique_succ > 0 and not loop2 and not loop3)
    early, mid, late = temporal_position_frequencies(members, max_step_by_game)
    pre_terminal = successor_terminal
    repeated_sequence = repeated_sequence_frequency(members)
    coarse = {
        "mean_prediction_accuracy": family.mean_prediction_accuracy,
        "mean_context_lift": family.mean_context_lift,
        "family_coherence": family.family_coherence,
        "support_log": math.log1p(family.support_count),
        "cross_game_presence": float(family.cross_game_presence),
        "cross_game_family_presence": float(len({game_family_map.get(game, "unknown") for game in family.games_present})),
        "terminal_frequency": terminal_rate,
        "blocked_frequency": no_change_rate,
        "position_change_frequency": position_rate,
        "change_frequency": change_rate,
        "context_dependency": context_dependency_score(context_depth_counts),
        "action_regularity": action_regularity_score(action_counts),
        "entropy_inverse": entropy_inverse(float(np.mean([member.entropy for member in members])) if members else 0.0),
        **label_one_hot(family.family_label_candidate),
    }
    directional = {
        "predecessor_count": float(pred_total),
        "successor_count": float(succ_total),
        "predecessor_successor_ratio": safe_ratio(pred_total, succ_total),
        "incoming_outgoing_ratio": safe_ratio(pred_total, succ_total),
        "unique_predecessor_families": float(unique_pred),
        "unique_successor_families": float(unique_succ),
        "predecessor_entropy": pred_entropy,
        "successor_entropy": succ_entropy,
        "directional_asymmetry_score": abs(unique_succ - unique_pred) / max(1.0, unique_succ + unique_pred),
    }
    future_option = {
        "reachable_before_mean": float(unique_pred),
        "reachable_after_mean": float(unique_succ),
        "reachable_delta_mean": float(unique_succ - unique_pred),
        "reachable_delta_sign_neg": float(unique_succ < unique_pred),
        "reachable_delta_sign_zero": float(unique_succ == unique_pred),
        "reachable_delta_sign_pos": float(unique_succ > unique_pred),
        "enable_score": enable_score,
        "block_score": block_score,
        "terminate_score": terminate_score,
        "preserve_score": preserve_score,
        "reversibility_score": reversibility_score,
    }
    local_motif = {
        "chain_position_count": chain_position,
        "branch_in_count": branch_in,
        "branch_out_count": branch_out,
        "bottleneck_count": bottleneck,
        "loop_2cycle_count": loop2,
        "loop_3cycle_count": loop3,
        "sink_count": sink,
        "source_count": source,
        "bridge_count": bridge,
    }
    temporal_effect = {
        "early_episode_frequency": early,
        "mid_episode_frequency": mid,
        "late_episode_frequency": late,
        "pre_terminal_frequency": pre_terminal,
        "post_unlock_frequency_if_derivable": min(1.0, enable_score * 0.5 + branch_out * 0.5),
        "repeated_sequence_frequency": repeated_sequence,
        "no_change_rate": no_change_rate,
        "position_change_rate": position_rate,
        "discontinuous_position_change_rate": discontinuous_position_rate,
        "terminal_rate": terminal_rate,
        "multi_cell_change_rate": large_change_rate,
        "coverage_change_rate_if_derivable": coverage_change_rate,
        "reversible_effect_rate": reversibility_score,
        "repeated_toggle_like_rate": min(1.0, loop2 + no_change_rate * interactive_ratio),
    }
    incoming_profile = build_profile_counter(predecessors, family_index)
    outgoing_profile = build_profile_counter(successors, family_index)
    return {
        "coarse": coarse,
        "directional": directional,
        "future_option": future_option,
        "local_motif": local_motif,
        "temporal_effect": temporal_effect,
        "incoming_edge_profile": incoming_profile,
        "outgoing_edge_profile": outgoing_profile,
    }


def normalize_feature_groups(precomputed: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    groups = ("coarse", "directional", "future_option", "local_motif", "temporal_effect")
    normalized: dict[str, dict[str, dict[str, float]]] = {family_id: {} for family_id in precomputed}
    for group in groups:
        keys = sorted({key for item in precomputed.values() for key in item[group].keys()})
        for key in keys:
            values = np.asarray([float(precomputed[family_id][group].get(key, 0.0)) for family_id in precomputed], dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values))
            if std <= 1e-9:
                normalized_values = [0.0 for _ in values]
            else:
                normalized_values = [float((value - mean) / std) for value in values]
            for family_id, value in zip(precomputed, normalized_values, strict=True):
                normalized[family_id].setdefault(group, {})[key] = value
    return normalized


def evaluate_pairwise_similarity(
    neighborhoods: dict[str, DiscNeighborhood],
    *,
    weights: SimilarityWeights,
    threshold: float,
    workers: int,
) -> list[PairResult]:
    items = sorted(neighborhoods.values(), key=lambda item: item.family_id)
    pairs = [(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]
    if not pairs:
        return []
    if workers <= 1 or len(pairs) <= 10:
        return [_pair_result(left, right, weights) for left, right in pairs]
    shards = _pair_shards(pairs, min(len(pairs), max(1, workers)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_pair_worker, shard, weights) for shard in shards]
    output: list[PairResult] = []
    for future in futures:
        output.extend(future.result())
    return sorted(output, key=lambda item: (item.left_family_id, item.right_family_id))


def _pair_worker(shard: list[tuple[DiscNeighborhood, DiscNeighborhood]], weights: SimilarityWeights) -> list[PairResult]:
    return [_pair_result(left, right, weights) for left, right in shard]


def _pair_shards(pairs: list[tuple[DiscNeighborhood, DiscNeighborhood]], shard_count: int) -> list[list[tuple[DiscNeighborhood, DiscNeighborhood]]]:
    output = [[] for _ in range(shard_count)]
    for index, pair in enumerate(pairs):
        output[index % shard_count].append(pair)
    return [item for item in output if item]


def _pair_result(left: DiscNeighborhood, right: DiscNeighborhood, weights: SimilarityWeights) -> PairResult:
    coarse = vector_similarity(left.coarse_features, right.coarse_features)
    directional = vector_similarity(left.directional_features, right.directional_features)
    future_option = vector_similarity(left.future_option_features, right.future_option_features)
    local_motif = vector_similarity(left.local_motif_features, right.local_motif_features)
    temporal_effect = vector_similarity(left.temporal_effect_features, right.temporal_effect_features)
    weighted = (
        coarse * weights.coarse
        + directional * weights.directional
        + future_option * weights.future_option
        + local_motif * weights.local_motif
        + temporal_effect * weights.temporal_effect
    ) / max(1e-9, weights.total)
    return PairResult(
        left_family_id=left.family_id,
        right_family_id=right.family_id,
        coarse_similarity=coarse,
        directional_similarity=directional,
        future_option_similarity=future_option,
        local_motif_similarity=local_motif,
        temporal_effect_similarity=temporal_effect,
        weighted_similarity=weighted,
    )


def build_similarity_adjacency(pair_results: list[PairResult], threshold: float) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for pair in pair_results:
        adjacency.setdefault(pair.left_family_id, set())
        adjacency.setdefault(pair.right_family_id, set())
        if pair.weighted_similarity >= threshold:
            adjacency[pair.left_family_id].add(pair.right_family_id)
            adjacency[pair.right_family_id].add(pair.left_family_id)
    return adjacency


def cluster_role_candidates(
    adjacency: dict[str, set[str]],
    pair_results: list[PairResult],
    neighborhoods: dict[str, DiscNeighborhood],
    threshold: float,
) -> tuple[list[list[str]], list[list[str]]]:
    pair_map = {
        (pair.left_family_id, pair.right_family_id): pair
        for pair in pair_results
    }
    remaining = set(adjacency)
    components: list[list[str]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        seen = {start}
        while stack:
            node = stack.pop()
            for neighbor in adjacency.get(node, ()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        remaining -= seen
        components.append(sorted(seen))

    clusters: list[list[str]] = []
    rejected: list[list[str]] = []
    for component in components:
        split = split_mixed_component(component, pair_map, neighborhoods, threshold)
        for cluster in split:
            if len(cluster) > max(1, math.ceil(0.45 * max(1, len(adjacency)))) and component_mean_similarity(cluster, pair_map) < threshold:
                rejected.append(cluster)
            else:
                clusters.append(cluster)
    return sorted(clusters), sorted(rejected)


def split_mixed_component(
    component: list[str],
    pair_map: dict[tuple[str, str], PairResult],
    neighborhoods: dict[str, DiscNeighborhood],
    threshold: float,
) -> list[list[str]]:
    if len(component) <= 1:
        return [sorted(component)]
    if not component_is_mixed(component, pair_map, neighborhoods, threshold):
        return [sorted(component)]
    for key_builder in (
        lambda record: dominant_profile_key(record.future_option_features, ("enable_score", "block_score", "terminate_score", "preserve_score", "reversibility_score")),
        lambda record: dominant_profile_key(record.local_motif_features, ("branch_out_count", "branch_in_count", "bridge_count", "loop_2cycle_count", "sink_count", "source_count")),
        lambda record: asymmetry_band(record.directional_features.get("directional_asymmetry_score", 0.0)),
        lambda record: effect_profile_key(record.temporal_effect_features),
    ):
        groups: dict[str, list[str]] = defaultdict(list)
        for family_id in component:
            groups[key_builder(neighborhoods[family_id])].append(family_id)
        values = [sorted(group) for group in groups.values() if group]
        if len(values) >= 2 and not mostly_singletons(values):
            return [sorted(group) for group in values]
    return [sorted(component)]


def component_is_mixed(
    component: list[str],
    pair_map: dict[tuple[str, str], PairResult],
    neighborhoods: dict[str, DiscNeighborhood],
    threshold: float,
) -> bool:
    if len(component) <= 1:
        return False
    pairs = []
    motifs = Counter()
    terminal_high = False
    reversible_high = False
    enable_high = False
    block_high = False
    for index, left in enumerate(component):
        record = neighborhoods[left]
        motifs[record.dominant_motif_candidate] += 1
        terminal_high = terminal_high or record.temporal_effect_features.get("terminal_rate", 0.0) > 0.5
        reversible_high = reversible_high or record.future_option_features.get("reversibility_score", 0.0) > 0.5
        enable_high = enable_high or record.future_option_features.get("enable_score", 0.0) > 0.5
        block_high = block_high or record.future_option_features.get("block_score", 0.0) > 0.5
        for right in component[index + 1 :]:
            pair = pair_map.get((left, right), pair_map.get((right, left)))
            if pair:
                pairs.append(pair)
    if not pairs:
        return False
    coarse = float(np.mean([pair.coarse_similarity for pair in pairs]))
    future_option = float(np.mean([pair.future_option_similarity for pair in pairs]))
    local_motif = float(np.mean([pair.local_motif_similarity for pair in pairs]))
    return (
        (coarse >= threshold and (future_option < threshold - 0.08 or local_motif < threshold - 0.08))
        or len(motifs) >= 2
        or (terminal_high and reversible_high)
        or (enable_high and block_high)
    )


def build_role_candidates(
    *,
    clusters: list[list[str]],
    neighborhoods: dict[str, DiscNeighborhood],
    min_role_support: int,
    role_similarity_threshold: float,
    pair_results: list[PairResult],
) -> list[M3RoleCandidate]:
    pair_map = {(pair.left_family_id, pair.right_family_id): pair for pair in pair_results}
    output: list[M3RoleCandidate] = []
    for index, cluster in enumerate(sorted(clusters), start=1):
        records = [neighborhoods[item] for item in cluster]
        games = sorted({game for record in records for game in record.game_ids})
        game_families = sorted({game_family for record in records for game_family in record.game_family_ids})
        support_count = sum(record.support_count for record in records)
        member_count = len(cluster)
        coarse_similarity = mean_similarity(cluster, pair_map, "coarse_similarity")
        directional_similarity = mean_similarity(cluster, pair_map, "directional_similarity")
        future_option_similarity = mean_similarity(cluster, pair_map, "future_option_similarity")
        local_motif_similarity = mean_similarity(cluster, pair_map, "local_motif_similarity")
        temporal_effect_similarity = mean_similarity(cluster, pair_map, "temporal_effect_similarity")
        mean_weighted_similarity = mean_similarity(cluster, pair_map, "weighted_similarity")
        mean_coherence = float(np.mean([record.family_coherence for record in records]))
        role_consistency = 0.35 * mean_weighted_similarity + 0.25 * mean_coherence + 0.20 * float(np.mean([record.mean_prediction_accuracy for record in records])) + 0.20 * future_option_similarity
        stable = (
            support_count >= int(min_role_support)
            and role_consistency >= float(role_similarity_threshold)
            and len(games) >= 2
            and mean_coherence >= 0.70
        )
        status = "singleton" if member_count == 1 else ("stable" if stable else "weak")
        label, evidence = label_role_candidate(records)
        output.append(
            M3RoleCandidate(
                role_id=f"m3-{index:04d}",
                role_label_candidate=label,
                member_family_ids=cluster,
                games_present=games,
                game_families_present=game_families,
                support_count=support_count,
                cross_game_support=len(games),
                cross_game_family_support=len(game_families),
                role_consistency_score=role_consistency,
                mean_neighborhood_similarity=mean_weighted_similarity,
                mean_family_coherence=mean_coherence,
                dominant_motif_profile=normalized_counter(Counter(record.dominant_motif_candidate for record in records)),
                incoming_edge_profile=merge_counter_dicts(record.incoming_edge_profile for record in records),
                outgoing_edge_profile=merge_counter_dicts(record.outgoing_edge_profile for record in records),
                future_option_effect_profile={
                    "change": float(np.mean([record.temporal_effect_features.get("coverage_change_rate_if_derivable", 0.0) for record in records])),
                    "position_like_change": float(np.mean([record.temporal_effect_features.get("position_change_rate", 0.0) for record in records])),
                    "terminal_transition": float(np.mean([record.temporal_effect_features.get("terminal_rate", 0.0) for record in records])),
                    "blocked_no_change": float(np.mean([record.temporal_effect_features.get("no_change_rate", 0.0) for record in records])),
                },
                transfer_readiness_score=0.30 * min(1.0, len(games) / 3.0) + 0.25 * min(1.0, len(game_families) / 2.0) + 0.25 * role_consistency + 0.20 * mean_coherence,
                label_evidence=evidence,
                examples=role_examples(records),
                status=status,
                notes={
                    "member_count": member_count,
                    "coarse_similarity": coarse_similarity,
                    "directional_similarity": directional_similarity,
                    "future_option_similarity": future_option_similarity,
                    "local_motif_similarity": local_motif_similarity,
                    "temporal_effect_similarity": temporal_effect_similarity,
                },
            )
        )
    return output


def label_role_candidate(records: list[DiscNeighborhood]) -> tuple[str, dict[str, Any]]:
    means = lambda key, group: float(np.mean([getattr(record, group)[key] for record in records]))
    summary = {
        "enable_score": means("enable_score", "future_option_features"),
        "block_score": means("block_score", "future_option_features"),
        "terminate_score": means("terminate_score", "future_option_features"),
        "preserve_score": means("preserve_score", "future_option_features"),
        "reversibility_score": means("reversibility_score", "future_option_features"),
        "branch_out_count": means("branch_out_count", "local_motif_features"),
        "bridge_count": means("bridge_count", "local_motif_features"),
        "source_count": means("source_count", "local_motif_features"),
        "discontinuous_position_change_rate": means("discontinuous_position_change_rate", "temporal_effect_features"),
        "position_change_rate": means("position_change_rate", "temporal_effect_features"),
        "terminal_rate": means("terminal_rate", "temporal_effect_features"),
        "coverage_change_rate_if_derivable": means("coverage_change_rate_if_derivable", "temporal_effect_features"),
        "repeated_toggle_like_rate": means("repeated_toggle_like_rate", "temporal_effect_features"),
    }
    alternatives = []
    if summary["terminal_rate"] >= 0.40 or summary["terminate_score"] >= 0.55:
        label = "terminator_candidate"
    elif summary["reversibility_score"] >= 0.45 or summary["repeated_toggle_like_rate"] >= 0.45:
        label = "reversible_preserver_candidate"
        alternatives.append("blocker_candidate")
    elif summary["block_score"] >= 0.55 and summary["terminal_rate"] < 0.30:
        label = "blocker_candidate"
        alternatives.append("reversible_preserver_candidate")
    elif summary["enable_score"] >= 0.50 and summary["branch_out_count"] >= 0.25:
        label = "enabler_candidate"
        alternatives.append("connector_candidate")
    elif summary["source_count"] >= 0.30 and summary["branch_out_count"] >= 0.25:
        label = "trigger_candidate"
        alternatives.append("enabler_candidate")
    elif summary["discontinuous_position_change_rate"] >= 0.45:
        label = "transporter_candidate"
        alternatives.append("movement_controller_candidate")
    elif summary["position_change_rate"] >= 0.45:
        label = "movement_controller_candidate"
        alternatives.append("transporter_candidate")
    elif summary["coverage_change_rate_if_derivable"] >= 0.40:
        label = "coverage_expander_candidate"
        alternatives.append("connector_candidate")
    elif summary["bridge_count"] >= 0.20:
        label = "connector_candidate"
        alternatives.append("enabler_candidate")
    else:
        label = "unknown_role_candidate"
    evidence = {
        "top_feature_values": dict(sorted(summary.items(), key=lambda item: (-item[1], item[0]))[:5]),
        "rejected_alternatives": alternatives,
    }
    return label, evidence


def build_graph_outputs(
    neighborhoods: dict[str, DiscNeighborhood],
    roles: list[M3RoleCandidate],
    pair_results: list[PairResult],
    m2_families: list[M2FamilyRecord],
    m1_support: dict[str, M1SupportRecord],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    family_map = {family.family_id: family for family in m2_families}
    for contingency_id, record in m1_support.items():
        nodes[contingency_id] = {"node_id": contingency_id, "node_type": "m1_contingency", "game_id": record.game_id}
    for family_id, record in neighborhoods.items():
        nodes[family_id] = {"node_id": family_id, "node_type": "m2_family", "family_label_candidate": record.family_label_candidate}
        family = family_map.get(family_id)
        if family is not None:
            for contingency_id in family.contingency_ids:
                if contingency_id in m1_support:
                    edges.append({"edge_type": "member_of", "source_id": contingency_id, "target_id": family_id})
    for role in roles:
        nodes[role.role_id] = {"node_id": role.role_id, "node_type": "m3_role_candidate", "role_label_candidate": role.role_label_candidate, "status": role.status}
        for family_id in role.member_family_ids:
            edges.append({"edge_type": "member_of", "source_id": family_id, "target_id": role.role_id})
    for pair in pair_results:
        if pair.weighted_similarity >= 0.55:
            edges.append(
                {
                    "edge_type": "similar_to",
                    "source_id": pair.left_family_id,
                    "target_id": pair.right_family_id,
                    "coarse_similarity": pair.coarse_similarity,
                    "directional_similarity": pair.directional_similarity,
                    "future_option_similarity": pair.future_option_similarity,
                    "local_motif_similarity": pair.local_motif_similarity,
                    "temporal_effect_similarity": pair.temporal_effect_similarity,
                    "weighted_similarity": pair.weighted_similarity,
                }
            )
    return list(nodes.values()), edges


def role_membership_rows(roles: list[M3RoleCandidate]) -> list[dict[str, Any]]:
    rows = []
    for role in roles:
        for family_id in role.member_family_ids:
            rows.append({"role_id": role.role_id, "family_id": family_id, "role_label_candidate": role.role_label_candidate, "status": role.status})
    return rows


def neighborhood_output_rows(neighborhoods: dict[str, DiscNeighborhood]) -> list[dict[str, Any]]:
    rows = []
    for record in sorted(neighborhoods.values(), key=lambda item: item.family_id):
        rows.append(
            {
                "family_id": record.family_id,
                "family_label_candidate": record.family_label_candidate,
                "games_present": list(record.game_ids),
                "game_families_present": list(record.game_family_ids),
                "support_count": record.support_count,
                "family_coherence": record.family_coherence,
                "mean_prediction_accuracy": record.mean_prediction_accuracy,
                "mean_context_lift": record.mean_context_lift,
                "dominant_outcome_signature": record.dominant_outcome_signature,
                "dominant_motif_candidate": record.dominant_motif_candidate,
                "coarse_features": record.coarse_features,
                "directional_features": record.directional_features,
                "future_option_features": record.future_option_features,
                "local_motif_features": record.local_motif_features,
                "temporal_effect_features": record.temporal_effect_features,
                "incoming_edge_profile": record.incoming_edge_profile,
                "outgoing_edge_profile": record.outgoing_edge_profile,
                "examples": list(record.examples),
            }
        )
    return rows


def similarity_output_rows(pair_results: list[PairResult]) -> list[dict[str, Any]]:
    return [
        {
            "source_family_id": pair.left_family_id,
            "target_family_id": pair.right_family_id,
            "coarse_similarity": pair.coarse_similarity,
            "directional_similarity": pair.directional_similarity,
            "future_option_similarity": pair.future_option_similarity,
            "local_motif_similarity": pair.local_motif_similarity,
            "temporal_effect_similarity": pair.temporal_effect_similarity,
            "weighted_similarity": pair.weighted_similarity,
        }
        for pair in pair_results
    ]


def cluster_diagnostics_rows(
    clusters: list[list[str]],
    neighborhoods: dict[str, DiscNeighborhood],
    pair_results: list[PairResult],
    rejected_clusters: list[list[str]],
) -> list[dict[str, Any]]:
    pair_map = {(pair.left_family_id, pair.right_family_id): pair for pair in pair_results}
    rows = []
    for index, cluster in enumerate(clusters, start=1):
        rows.append(
            {
                "cluster_id": f"c{index:04d}",
                "family_ids": cluster,
                "size": len(cluster),
                "coarse_similarity": mean_similarity(cluster, pair_map, "coarse_similarity"),
                "directional_similarity": mean_similarity(cluster, pair_map, "directional_similarity"),
                "future_option_similarity": mean_similarity(cluster, pair_map, "future_option_similarity"),
                "local_motif_similarity": mean_similarity(cluster, pair_map, "local_motif_similarity"),
                "temporal_effect_similarity": mean_similarity(cluster, pair_map, "temporal_effect_similarity"),
                "weighted_similarity": mean_similarity(cluster, pair_map, "weighted_similarity"),
                "mixed_cluster_detected": component_is_mixed(cluster, pair_map, neighborhoods, 0.70),
            }
        )
    for index, cluster in enumerate(rejected_clusters, start=1):
        rows.append({"cluster_id": f"r{index:04d}", "family_ids": cluster, "size": len(cluster), "rejected": True})
    return rows


def build_v08d_payload(
    *,
    config: RoleCandidatesV08dConfig,
    game_set: GameSetManifest,
    selected_games: tuple[str, ...],
    neighborhoods: dict[str, DiscNeighborhood],
    pair_results: list[PairResult],
    roles: list[M3RoleCandidate],
    rejected_clusters: list[list[str]],
) -> dict[str, Any]:
    stable_roles = [role for role in roles if role.status == "stable"]
    weak_roles = [role for role in roles if role.status == "weak"]
    singleton_roles = [role for role in roles if role.status == "singleton"]
    cross_game = [role for role in stable_roles if role.cross_game_support >= 2]
    cross_family = [role for role in stable_roles if role.cross_game_family_support >= 2]
    games_represented = sorted({game for role in stable_roles for game in role.games_present})
    game_families_represented = sorted({family for role in stable_roles for family in role.game_families_present})
    largest_cluster_size = max((len(role.member_family_ids) for role in roles), default=0)
    total_families = max(1, len(neighborhoods))
    largest_cluster_percent = largest_cluster_size / total_families
    mean_rcs = float(np.mean([role.role_consistency_score for role in stable_roles])) if stable_roles else 0.0
    weak = (
        len(stable_roles) >= 4
        and len(games_represented) >= 10
        and len(game_families_represented) >= 6
        and len(cross_family) >= 2
        and mean_rcs >= 0.70
        and largest_cluster_percent < 0.40
    )
    strong = (
        len(stable_roles) >= 6
        and len(games_represented) >= 16
        and len(game_families_represented) >= 8
        and len(cross_family) >= 3
        and mean_rcs >= 0.75
        and largest_cluster_percent <= 0.35
    )
    very_strong = (
        len(stable_roles) >= 8
        and len(games_represented) >= 24
        and len(game_families_represented) >= 12
        and len(cross_family) >= 4
        and mean_rcs >= 0.80
        and largest_cluster_percent <= 0.30
    )
    comparison_original = load_json_if_exists(Path("runs/v6/v08_cd2_extended32/v08_report.json"))
    comparison_expanded = load_json_if_exists(Path("runs/v6/v08_cd2_extended32_expanded/v08_report.json"))
    improvements = []
    old_clusters = comparison_expanded.get("report", {}).get("stable_clusters", 0)
    old_percent = comparison_expanded.get("report", {}).get("largest_cluster_percent", 1.0)
    old_conclusion = comparison_expanded.get("validation", {}).get("scientific_conclusion")
    old_pass = comparison_expanded.get("report", {}).get("extended_validation_pass_level")
    old_labels = {row.get("role_label_candidate") for row in comparison_expanded.get("report", {}).get("top_role_candidates", [])}
    stable_labels = [role.role_label_candidate for role in stable_roles]
    weak_labels = [role.role_label_candidate for role in weak_roles]
    singleton_labels = [role.role_label_candidate for role in singleton_roles]
    new_labels = sorted(set(stable_labels + weak_labels) - old_labels)
    if len(stable_roles) > int(old_clusters):
        improvements.append("stable_clusters")
    if largest_cluster_percent < float(old_percent):
        improvements.append("largest_cluster_percent")
    if "reversible_preserver_candidate" in weak_labels or "reversible_preserver_candidate" in stable_labels:
        improvements.append("reversible_preserver_candidate")
    if any(label in (stable_labels + weak_labels) for label in ("enabler_candidate", "terminator_candidate", "trigger_candidate", "movement_controller_candidate", "coverage_expander_candidate")):
        improvements.append("new_discriminative_label")
    if old_pass != "failed" and old_pass is not None:
        improvements.append("extended_validation_pass_level")
    if old_conclusion and old_conclusion != "m3_extended_not_established":
        improvements.append("scientific_conclusion")

    if very_strong:
        conclusion = "m3_discriminative_very_strong_role_candidates"
    elif strong:
        conclusion = "m3_discriminative_strong_role_candidates"
    elif weak:
        conclusion = "m3_discriminative_weak_role_candidates"
    elif len(set(improvements)) >= 2:
        conclusion = "m3_discriminative_improved_but_not_passed"
    else:
        conclusion = "m3_discriminative_not_established"
    pass_level = "very_strong" if very_strong else "strong" if strong else "weak" if weak else "failed"

    report = {
        "games_analyzed": list(selected_games),
        "game_families_analyzed": sorted({family for record in neighborhoods.values() for family in record.game_family_ids}),
        "families_analyzed": len(neighborhoods),
        "stable_clusters": len(stable_roles),
        "weak_clusters": len(weak_roles),
        "singleton_clusters": len(singleton_roles),
        "cross_game_clusters": len(cross_game),
        "cross_family_clusters": len(cross_family),
        "largest_cluster_percent": largest_cluster_percent,
        "giant_cluster_detected": largest_cluster_percent > 0.40,
        "stable_role_labels": stable_labels,
        "weak_role_labels": weak_labels,
        "singleton_role_labels": singleton_labels,
        "new_labels_vs_v08c": new_labels,
        "reversible_preserver_status": role_status("reversible_preserver_candidate", roles),
        "enabler_status": role_status("enabler_candidate", roles),
        "terminator_status": role_status("terminator_candidate", roles),
        "trigger_status": role_status("trigger_candidate", roles),
        "movement_controller_status": role_status("movement_controller_candidate", roles),
        "coverage_expander_status": role_status("coverage_expander_candidate", roles),
        "scientific_conclusion": conclusion,
        "extended_validation_pass_level": pass_level,
        "proceed_to_v09_role_transfer_testing": conclusion in {
            "m3_discriminative_weak_role_candidates",
            "m3_discriminative_strong_role_candidates",
            "m3_discriminative_very_strong_role_candidates",
        },
        "comparison_original_v08": comparison_summary(comparison_original),
        "comparison_expanded_v08c": comparison_summary(comparison_expanded),
        "top_role_candidates": [role.to_record() for role in sorted(stable_roles, key=lambda item: (-item.role_consistency_score, item.role_id))[:8]],
        "top_weak_role_candidates": [role.to_record() for role in sorted(weak_roles, key=lambda item: (-item.role_consistency_score, item.role_id))[:8]],
        "improvements_detected": sorted(set(improvements)),
    }
    validation = {
        "diagnostic_success": bool(neighborhoods),
        "scientific_conclusion": conclusion,
        "weak_pass": weak,
        "strong_pass": strong,
        "very_strong_pass": very_strong,
        "extended_validation_pass_level": pass_level,
    }
    return {
        "config": {
            "input_dir": config.input_dir,
            "m1_input_dir": config.m1_input_dir,
            "output_dir": config.output_dir,
            "context_depth": int(config.context_depth),
            "min_role_support": int(config.min_role_support),
            "role_similarity_threshold": float(config.role_similarity_threshold),
            "workers": int(config.workers),
            "partition_by": list(config.partition_by),
            "fingerprint_mode": config.fingerprint_mode,
        },
        "report": report,
        "validation": validation,
    }


def write_outputs(
    *,
    output_dir: Path,
    roles: list[M3RoleCandidate],
    membership_rows: list[dict[str, Any]],
    neighborhood_rows: list[dict[str, Any]],
    similarity_rows: list[dict[str, Any]],
    cluster_rows: list[dict[str, Any]],
    graph_nodes: list[dict[str, Any]],
    graph_edges: list[dict[str, Any]],
    payload: dict[str, Any],
) -> None:
    role_rows = [role.to_record() for role in roles]
    (output_dir / "m3_role_candidates.json").write_text(json.dumps(role_rows, indent=2), encoding="utf-8")
    _write_parquet(output_dir / "m3_role_candidates.parquet", role_rows)
    _write_parquet(output_dir / "role_candidate_membership.parquet", membership_rows)
    _write_parquet(output_dir / "role_neighborhoods.parquet", neighborhood_rows)
    _write_parquet(output_dir / "role_similarity_edges.parquet", similarity_rows)
    _write_parquet(output_dir / "role_cluster_diagnostics.parquet", cluster_rows)
    _write_parquet(output_dir / "v08_graph_nodes.parquet", graph_nodes)
    _write_parquet(output_dir / "v08_graph_edges.parquet", graph_edges)
    (output_dir / "v08d_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "v08d_report.txt").write_text(format_v08d_report(payload), encoding="utf-8")
    (output_dir / "v08_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "v08_report.txt").write_text(format_v08d_report(payload), encoding="utf-8")


def format_v08d_report(payload: dict[str, Any]) -> str:
    r = payload["report"]
    lines = [
        "ARC-AGI3 v0.8d-m3-discriminative-fingerprint-refinement",
        f"scientific_conclusion={payload['validation']['scientific_conclusion']}",
        f"families_analyzed={r['families_analyzed']}",
        f"stable_clusters={r['stable_clusters']}",
        f"weak_clusters={r['weak_clusters']}",
        f"singleton_clusters={r['singleton_clusters']}",
        f"cross_game_clusters={r['cross_game_clusters']}",
        f"cross_family_clusters={r['cross_family_clusters']}",
        f"largest_cluster_percent={r['largest_cluster_percent']:.6f}",
        f"giant_cluster_detected={r['giant_cluster_detected']}",
        f"stable_role_labels={','.join(r['stable_role_labels'])}",
        f"weak_role_labels={','.join(r['weak_role_labels'])}",
        f"singleton_role_labels={','.join(r['singleton_role_labels'])}",
        f"new_labels_vs_v08c={','.join(r['new_labels_vs_v08c'])}",
        f"reversible_preserver_status={r['reversible_preserver_status']}",
        f"enabler_status={r['enabler_status']}",
        f"terminator_status={r['terminator_status']}",
        f"trigger_status={r['trigger_status']}",
        f"movement_controller_status={r['movement_controller_status']}",
        f"coverage_expander_status={r['coverage_expander_status']}",
        f"extended_validation_pass_level={r['extended_validation_pass_level']}",
        f"proceed_to_v09_role_transfer_testing={r['proceed_to_v09_role_transfer_testing']}",
        f"improvements_detected={','.join(r['improvements_detected'])}",
    ]
    return "\n".join(lines)


def emitted_token(record: M1SupportRecord) -> str:
    return f"a{record.action}|o{coarse_outcome_class(record.outcome_signature)}"


def coarse_outcome_class(signature: str) -> str:
    if signature in {"blocked_no_change", "preserve_no_change"}:
        return "preserve"
    if signature == "terminal_transition":
        return "terminal"
    if signature in {"position_like_change", "large_change", "change"}:
        return "change"
    return "unknown"


def normalized_counter(counter_like: dict[str, int] | Counter[str]) -> dict[str, float]:
    total = float(sum(counter_like.values()))
    if total <= 0.0:
        return {}
    return {key: float(value) / total for key, value in sorted(counter_like.items())}


def dominant_key(mapping: dict[str, Any]) -> str:
    if not mapping:
        return "unknown"
    return max(mapping.items(), key=lambda item: (float(item[1]), item[0]))[0]


def entropy(counter_like: dict[str, int] | Counter[str]) -> float:
    total = sum(counter_like.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter_like.values() if count > 0)


def vector_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    left_values = np.asarray([float(left.get(key, 0.0)) for key in keys], dtype=float)
    right_values = np.asarray([float(right.get(key, 0.0)) for key in keys], dtype=float)
    numerator = float(np.dot(left_values, right_values))
    denominator = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
    if denominator <= 0.0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def action_regularity_score(action_counts: Counter[int]) -> float:
    total = sum(action_counts.values())
    if total <= 0:
        return 0.0
    dominant = max(action_counts.values())
    return dominant / total


def context_dependency_score(context_counts: Counter[int]) -> float:
    total = sum(context_counts.values())
    if total <= 0:
        return 0.0
    return sum(depth * count for depth, count in context_counts.items()) / total


def entropy_inverse(value: float) -> float:
    return 1.0 / (1.0 + max(0.0, value))


def safe_ratio(left: float, right: float) -> float:
    return float(left) / max(1.0, float(right))


def reciprocal_successor_ratio(family_id: str, predecessors: Counter[str], successors: Counter[str]) -> float:
    reciprocal = len(set(predecessors) & set(successors))
    total = len(set(predecessors) | set(successors))
    return reciprocal / max(1, total)


def loop_2cycle_score(family_id: str, predecessors: Counter[str], successors: Counter[str]) -> float:
    reciprocal = len(set(predecessors) & set(successors))
    return reciprocal / max(1, len(set(successors)))


def loop_3cycle_score(
    family_id: str,
    successors: Counter[str],
    family_index: dict[str, M2FamilyRecord],
    family_members: dict[str, list[M1SupportRecord]],
) -> float:
    if len(successors) < 2:
        return 0.0
    # Approximate 3-cycle potential from successor label diversity rather than explicit triads.
    labels = Counter(family_index[item].family_label_candidate for item in successors if item in family_index)
    return min(1.0, max(0, len(labels) - 1) / 3.0)


def temporal_position_frequencies(members: list[M1SupportRecord], max_step_by_game: dict[str, int]) -> tuple[float, float, float]:
    early = mid = late = 0
    for member in members:
        max_step = max(1, max_step_by_game.get(member.game_id, member.last_seen_step))
        pos = ((member.first_seen_step + member.last_seen_step) / 2.0) / max_step
        if pos < 0.33:
            early += 1
        elif pos < 0.66:
            mid += 1
        else:
            late += 1
    total = max(1, len(members))
    return early / total, mid / total, late / total


def repeated_sequence_frequency(members: list[M1SupportRecord]) -> float:
    counter = Counter(tuple(member.context_signature) for member in members)
    repeated = sum(count for count in counter.values() if count > 1)
    return repeated / max(1, len(members))


def build_profile_counter(neighbors: Counter[str], family_index: dict[str, M2FamilyRecord]) -> Counter[str]:
    output: Counter[str] = Counter()
    for family_id, count in neighbors.items():
        label = family_index[family_id].family_label_candidate if family_id in family_index else "unknown"
        output[label] += int(count)
    return output


def mean_neighbor_metric(neighbors: Counter[str], family_index: dict[str, M2FamilyRecord], fn) -> float:
    if not neighbors:
        return 0.0
    values = []
    for family_id, count in neighbors.items():
        family = family_index.get(family_id)
        if family is None:
            continue
        values.extend([fn(family)] * int(count))
    return 0.0 if not values else float(np.mean(values))


def label_one_hot(label: str) -> dict[str, float]:
    labels = (
        "blocked_no_change_family_candidate",
        "position_like_change_family_candidate",
        "activation_like_family_candidate",
        "teleport_like_family_candidate",
        "push_like_family_candidate",
        "coverage_change_family_candidate",
        "state_change_family_candidate",
        "terminal_family_candidate",
        "unknown_change_family_candidate",
    )
    return {f"label::{item}": 1.0 if label == item else 0.0 for item in labels}


def component_mean_similarity(component: list[str], pair_map: dict[tuple[str, str], PairResult]) -> float:
    return mean_similarity(component, pair_map, "weighted_similarity")


def mean_similarity(component: list[str], pair_map: dict[tuple[str, str], PairResult], field_name: str) -> float:
    if len(component) <= 1:
        return 1.0
    values = []
    for index, left in enumerate(component):
        for right in component[index + 1 :]:
            pair = pair_map.get((left, right), pair_map.get((right, left)))
            if pair is not None:
                values.append(float(getattr(pair, field_name)))
    return 0.0 if not values else float(np.mean(values))


def dominant_profile_key(features: dict[str, float], keys: tuple[str, ...]) -> str:
    filtered = {key: float(features.get(key, 0.0)) for key in keys}
    return dominant_key(filtered)


def asymmetry_band(value: float) -> str:
    if value < 0.15:
        return "balanced"
    if value < 0.35:
        return "moderate"
    return "strong"


def effect_profile_key(features: dict[str, float]) -> str:
    if features.get("terminal_rate", 0.0) >= 0.40:
        return "terminal"
    if features.get("reversible_effect_rate", 0.0) >= 0.40:
        return "reversible"
    if features.get("discontinuous_position_change_rate", 0.0) >= 0.40:
        return "discontinuous_position"
    if features.get("position_change_rate", 0.0) >= 0.40:
        return "position"
    if features.get("no_change_rate", 0.0) >= 0.40:
        return "no_change"
    return "change"


def mostly_singletons(groups: list[list[str]]) -> bool:
    return sum(1 for group in groups if len(group) == 1) > len(groups) / 2


def merge_counter_dicts(counters: Any) -> dict[str, int]:
    output: Counter[str] = Counter()
    for counter in counters:
        output.update({key: int(value) for key, value in counter.items()})
    return dict(sorted(output.items()))


def role_examples(records: list[DiscNeighborhood], limit: int = 5) -> list[dict[str, Any]]:
    output = []
    for record in records:
        output.append(
            {
                "family_id": record.family_id,
                "family_label_candidate": record.family_label_candidate,
                "games_present": list(record.game_ids),
                "game_families_present": list(record.game_family_ids),
                "dominant_outcome_signature": record.dominant_outcome_signature,
                "dominant_motif_candidate": record.dominant_motif_candidate,
                "family_coherence": record.family_coherence,
            }
        )
        if len(output) >= limit:
            break
    return output


def role_status(label: str, roles: list[M3RoleCandidate]) -> str:
    matching = [role for role in roles if role.role_label_candidate == label]
    if not matching:
        return "absent"
    if any(role.status == "stable" for role in matching):
        return "stable"
    if any(role.status == "weak" for role in matching):
        return "weak"
    return "singleton"


def comparison_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    report = payload.get("report", {})
    validation = payload.get("validation", {})
    return {
        "families_analyzed": report.get("families_analyzed"),
        "stable_clusters": report.get("stable_clusters"),
        "largest_cluster_percent": report.get("largest_cluster_percent"),
        "scientific_conclusion": validation.get("scientific_conclusion"),
        "extended_validation_pass_level": report.get("extended_validation_pass_level"),
    }


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}
