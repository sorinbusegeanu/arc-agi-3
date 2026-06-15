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


ALLOWED_ROLE_LABELS = (
    "blocker_candidate",
    "enabler_candidate",
    "trigger_candidate",
    "transporter_candidate",
    "terminator_candidate",
    "connector_candidate",
    "reversible_preserver_candidate",
    "movement_controller_candidate",
    "coverage_expander_candidate",
    "unknown_role_candidate",
)


@dataclass(frozen=True)
class RoleCandidatesV08Config:
    input_dir: str = "runs/v6/v07_cd2"
    m1_input_dir: str = "runs/v6/v06_cd2"
    output_dir: str = "runs/v6/v08_cd2"
    context_depth: int = 2
    min_role_support: int = 3
    role_similarity_threshold: float = 0.70
    workers: int = 25
    partition_by: tuple[str, ...] = ("family_pair", "neighborhood_shard")
    game_set_manifest: str | None = None
    game_set_name: str | None = None


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
class NeighborhoodRecord:
    family_id: str
    game_ids: tuple[str, ...]
    game_family_ids: tuple[str, ...]
    support_count: int
    family_coherence: float
    mean_prediction_accuracy: float
    mean_context_lift: float
    dominant_outcome_signature: str
    dominant_motif_candidate: str
    fingerprint: dict[str, float]
    incoming_edge_profile: dict[str, int]
    outgoing_edge_profile: dict[str, int]
    examples: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PairResult:
    left_family_id: str
    right_family_id: str
    similarity: float
    edge_types: tuple[str, ...]


def run_role_candidates_v08(config: RoleCandidatesV08Config) -> dict[str, Any]:
    input_dir = Path(config.input_dir)
    m1_input_dir = Path(config.m1_input_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    game_set = load_game_set(config, input_dir)
    m2_families = load_m2_families(input_dir)
    m1_support = load_m1_support(m1_input_dir)

    available_games = {game for family in m2_families for game in family.games_present}
    selected_games = tuple(game for game in game_set.games if game in available_games) if game_set.games else tuple(sorted(available_games))
    selected_game_set = set(selected_games)
    m2_families = [family for family in m2_families if selected_game_set.intersection(family.games_present)]
    m1_support = {key: value for key, value in m1_support.items() if value.game_id in selected_game_set}

    game_family_map = build_game_family_map(game_set, selected_games)
    neighborhoods = build_neighborhoods(m2_families, m1_support, game_family_map)
    pair_results = evaluate_pairwise_similarity(
        neighborhoods,
        threshold=config.role_similarity_threshold,
        workers=config.workers,
    )
    adjacency = build_similarity_adjacency(pair_results, config.role_similarity_threshold)
    clusters, rejected_clusters = cluster_role_candidates(adjacency, pair_results, config.role_similarity_threshold)
    roles = build_role_candidates(
        clusters=clusters,
        neighborhoods=neighborhoods,
        min_role_support=config.min_role_support,
        role_similarity_threshold=config.role_similarity_threshold,
    )
    graph_nodes, graph_edges = build_graph_outputs(neighborhoods, roles, pair_results, m2_families, m1_support)
    membership_rows = role_membership_rows(roles)
    neighborhood_rows = neighborhood_output_rows(neighborhoods)
    payload = build_v08_payload(
        config=config,
        game_set=game_set,
        selected_games=selected_games,
        neighborhoods=neighborhoods,
        pair_results=pair_results,
        roles=roles,
        rejected_clusters=rejected_clusters,
    )

    write_v08_outputs(
        output_dir=output_dir,
        roles=roles,
        membership_rows=membership_rows,
        neighborhood_rows=neighborhood_rows,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        payload=payload,
    )
    return payload


def load_game_set(config: RoleCandidatesV08Config, input_dir: Path) -> GameSetManifest:
    if config.game_set_manifest or config.game_set_name:
        return load_game_set_manifest(
            manifest_path=config.game_set_manifest,
            game_set_name=config.game_set_name,
            fallback_games=(),
        )
    report_path = input_dir / "v07_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        games = sorted(report.get("report", {}).get("families_by_game", {}).keys())
    else:
        games = []
    return GameSetManifest(name="core7_fallback", games=tuple(games), families={}, purpose="fallback inferred from v07 output")


def build_game_family_map(game_set: GameSet, games: tuple[str, ...]) -> dict[str, str]:
    output: dict[str, str] = {}
    for family_name, family_games in game_set.families.items():
        for game in family_games:
            output.setdefault(game, family_name)
    for game in games:
        output.setdefault(game, family_for_game(game))
    return output


def load_m2_families(input_dir: Path) -> list[M2FamilyRecord]:
    path = input_dir / "m2_families.json"
    if not path.exists():
        raise FileNotFoundError(f"missing v0.7 input: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
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
    path = input_dir / "contingencies.json"
    if not path.exists():
        raise FileNotFoundError(f"missing v0.6 input: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    output = {}
    for row in rows:
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
        )
        output[record.contingency_id] = record
    return output


def build_neighborhoods(
    families: list[M2FamilyRecord],
    m1_support: dict[str, M1SupportRecord],
    game_family_map: dict[str, str],
) -> dict[str, NeighborhoodRecord]:
    output = {}
    for family in sorted(families, key=lambda item: item.family_id):
        members = [m1_support[item] for item in family.contingency_ids if item in m1_support]
        action_counts = Counter(member.action for member in members)
        context_depth_counts = Counter(len(member.context_signature) for member in members)
        outcome_dist = normalized_counter(family.outcome_signature_distribution)
        motif_dist = normalized_counter(family.motif_candidate_distribution)
        games = tuple(sorted(family.games_present))
        game_families = tuple(sorted({game_family_map.get(game, "unknown") for game in games}))
        dominant_motif = dominant_key(family.motif_candidate_distribution)
        terminal_frequency = outcome_dist.get("terminal_transition", 0.0)
        blocked_frequency = outcome_dist.get("blocked_no_change", 0.0) + outcome_dist.get("preserve_no_change", 0.0)
        position_frequency = outcome_dist.get("position_like_change", 0.0)
        change_frequency = (
            outcome_dist.get("change", 0.0)
            + outcome_dist.get("large_change", 0.0)
            + position_frequency
            + terminal_frequency
        )
        fingerprint = {
            "mean_prediction_accuracy": family.mean_prediction_accuracy,
            "mean_context_lift": family.mean_context_lift,
            "family_coherence": family.family_coherence,
            "support_log": math.log1p(family.support_count),
            "cross_game_presence": float(family.cross_game_presence),
            "cross_game_family_presence": float(len(game_families)),
            "terminal_frequency": terminal_frequency,
            "blocked_frequency": blocked_frequency,
            "position_change_frequency": position_frequency,
            "change_frequency": change_frequency,
            "context_dependency": context_dependency_score(context_depth_counts),
            "action_regularity": action_regularity_score(action_counts),
            "entropy_inverse": entropy_inverse(mean_member_entropy(members)),
            "motif_block": motif_dist.get("block_candidate", 0.0),
            "motif_change": motif_dist.get("change_candidate", 0.0),
            "motif_terminate": motif_dist.get("terminate_candidate", 0.0),
            "motif_preserve": motif_dist.get("preserve_candidate", 0.0),
            "motif_enable": motif_dist.get("enable_candidate_unknown", 0.0),
        }
        output[family.family_id] = NeighborhoodRecord(
            family_id=family.family_id,
            game_ids=games,
            game_family_ids=game_families,
            support_count=family.support_count,
            family_coherence=family.family_coherence,
            mean_prediction_accuracy=family.mean_prediction_accuracy,
            mean_context_lift=family.mean_context_lift,
            dominant_outcome_signature=family.dominant_outcome_signature,
            dominant_motif_candidate=dominant_motif,
            fingerprint=fingerprint,
            incoming_edge_profile={},
            outgoing_edge_profile={},
            examples=family.examples,
        )
    return add_neighbor_profiles(output)


def add_neighbor_profiles(neighborhoods: dict[str, NeighborhoodRecord]) -> dict[str, NeighborhoodRecord]:
    pair_results = evaluate_pairwise_similarity(
        neighborhoods,
        threshold=0.0,
        workers=1,
    )
    incoming: dict[str, Counter[str]] = defaultdict(Counter)
    outgoing: dict[str, Counter[str]] = defaultdict(Counter)
    for pair in pair_results:
        for edge_type in pair.edge_types:
            outgoing[pair.left_family_id][edge_type] += 1
            incoming[pair.right_family_id][edge_type] += 1
            outgoing[pair.right_family_id][edge_type] += 1
            incoming[pair.left_family_id][edge_type] += 1
    output = {}
    for family_id, record in neighborhoods.items():
        fingerprint = dict(record.fingerprint)
        fingerprint.update(
            {
                "incoming_degree": float(sum(incoming[family_id].values())),
                "outgoing_degree": float(sum(outgoing[family_id].values())),
                "incoming_similar": float(incoming[family_id].get("similar_to", 0)),
                "outgoing_similar": float(outgoing[family_id].get("similar_to", 0)),
                "incoming_blocks": float(incoming[family_id].get("candidate_blocks", 0)),
                "outgoing_blocks": float(outgoing[family_id].get("candidate_blocks", 0)),
                "incoming_enables": float(incoming[family_id].get("candidate_enables", 0)),
                "outgoing_enables": float(outgoing[family_id].get("candidate_enables", 0)),
                "incoming_terminates": float(incoming[family_id].get("candidate_terminates", 0)),
                "outgoing_terminates": float(outgoing[family_id].get("candidate_terminates", 0)),
                "incoming_transports": float(incoming[family_id].get("candidate_transports", 0)),
                "outgoing_transports": float(outgoing[family_id].get("candidate_transports", 0)),
            }
        )
        output[family_id] = NeighborhoodRecord(
            family_id=record.family_id,
            game_ids=record.game_ids,
            game_family_ids=record.game_family_ids,
            support_count=record.support_count,
            family_coherence=record.family_coherence,
            mean_prediction_accuracy=record.mean_prediction_accuracy,
            mean_context_lift=record.mean_context_lift,
            dominant_outcome_signature=record.dominant_outcome_signature,
            dominant_motif_candidate=record.dominant_motif_candidate,
            fingerprint=fingerprint,
            incoming_edge_profile=dict(sorted(incoming[family_id].items())),
            outgoing_edge_profile=dict(sorted(outgoing[family_id].items())),
            examples=record.examples,
        )
    return output


def evaluate_pairwise_similarity(
    neighborhoods: dict[str, NeighborhoodRecord],
    *,
    threshold: float,
    workers: int,
) -> list[PairResult]:
    items = sorted(neighborhoods.values(), key=lambda item: item.family_id)
    pairs = [(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]
    if not pairs:
        return []
    if workers <= 1 or len(pairs) <= 3:
        return [_pair_result_from_records(left, right, threshold) for left, right in pairs]
    shards = _pair_shards(pairs, min(len(pairs), max(1, workers)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_pair_worker, shard, threshold) for shard in shards]
    results: list[PairResult] = []
    for future in futures:
        results.extend(future.result())
    return sorted(results, key=lambda item: (item.left_family_id, item.right_family_id))


def _pair_shards(pairs: list[tuple[NeighborhoodRecord, NeighborhoodRecord]], shard_count: int) -> list[list[tuple[NeighborhoodRecord, NeighborhoodRecord]]]:
    output = [[] for _ in range(shard_count)]
    for index, pair in enumerate(pairs):
        output[index % shard_count].append(pair)
    return [shard for shard in output if shard]


def _pair_worker(shard: list[tuple[NeighborhoodRecord, NeighborhoodRecord]], threshold: float) -> list[PairResult]:
    return [_pair_result_from_records(left, right, threshold) for left, right in shard]


def _pair_result_from_records(left: NeighborhoodRecord, right: NeighborhoodRecord, threshold: float) -> PairResult:
    similarity = neighborhood_similarity(left, right)
    edge_types = derive_edge_types(left, right, similarity, threshold)
    return PairResult(
        left_family_id=left.family_id,
        right_family_id=right.family_id,
        similarity=similarity,
        edge_types=tuple(sorted(edge_types)),
    )


def neighborhood_similarity(left: NeighborhoodRecord, right: NeighborhoodRecord) -> float:
    base = vector_similarity(left.fingerprint, right.fingerprint)
    outcome_match = 1.0 if left.dominant_outcome_signature == right.dominant_outcome_signature else 0.35
    motif_match = 1.0 if left.dominant_motif_candidate == right.dominant_motif_candidate else 0.55
    game_overlap = 1.0 if set(left.game_ids) & set(right.game_ids) else 0.85
    return max(0.0, min(1.0, base * outcome_match * motif_match * game_overlap))


def derive_edge_types(left: NeighborhoodRecord, right: NeighborhoodRecord, similarity: float, threshold: float) -> set[str]:
    edge_types: set[str] = set()
    if similarity >= threshold:
        edge_types.add("similar_to")
    if set(left.game_ids) & set(right.game_ids):
        edge_types.add("co_occurs_with")
    if abs(left.mean_context_lift - right.mean_context_lift) <= 0.05:
        edge_types.add("context_depends_on")
    if left.dominant_outcome_signature == "terminal_transition" or right.dominant_outcome_signature == "terminal_transition":
        edge_types.add("candidate_terminates")
    if "transport" in left.dominant_motif_candidate or "transport" in right.dominant_motif_candidate:
        edge_types.add("candidate_transports")
    if left.dominant_outcome_signature in {"blocked_no_change", "preserve_no_change"} or right.dominant_outcome_signature in {
        "blocked_no_change",
        "preserve_no_change",
    }:
        edge_types.add("candidate_blocks")
    if left.mean_context_lift >= 0.10 or right.mean_context_lift >= 0.10:
        edge_types.add("candidate_enables")
    if similarity >= max(0.55, threshold - 0.10):
        edge_types.add("candidate_explains")
    if (
        left.mean_context_lift > right.mean_context_lift + 0.05
        or right.mean_context_lift > left.mean_context_lift + 0.05
    ):
        edge_types.add("precedes")
    return edge_types


def build_similarity_adjacency(pair_results: list[PairResult], threshold: float) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for pair in pair_results:
        adjacency.setdefault(pair.left_family_id, set())
        adjacency.setdefault(pair.right_family_id, set())
        if pair.similarity >= threshold:
            adjacency[pair.left_family_id].add(pair.right_family_id)
            adjacency[pair.right_family_id].add(pair.left_family_id)
    return adjacency


def cluster_role_candidates(
    adjacency: dict[str, set[str]],
    pair_results: list[PairResult],
    threshold: float,
) -> tuple[list[list[str]], list[list[str]]]:
    pair_map = {(pair.left_family_id, pair.right_family_id): pair.similarity for pair in pair_results}
    remaining = {key for key in adjacency}
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
        split_clusters = split_low_coherence_component(component, adjacency, pair_map, threshold)
        for cluster in split_clusters:
            if len(cluster) > max(1, math.ceil(0.4 * max(1, len(adjacency)))) and component_mean_similarity(cluster, pair_map) < threshold:
                rejected.append(cluster)
            else:
                clusters.append(cluster)
    return sorted(clusters), sorted(rejected)


def split_low_coherence_component(
    component: list[str],
    adjacency: dict[str, set[str]],
    pair_map: dict[tuple[str, str], float],
    threshold: float,
) -> list[list[str]]:
    if len(component) <= 1 or component_mean_similarity(component, pair_map) >= threshold:
        return [sorted(component)]
    edges = []
    for index, left in enumerate(component):
        for right in component[index + 1 :]:
            if right in adjacency.get(left, set()):
                edges.append((pair_map.get((left, right), pair_map.get((right, left), 0.0)), left, right))
    if not edges:
        return [[item] for item in sorted(component)]
    _lowest, cut_left, cut_right = min(edges, key=lambda item: (item[0], item[1], item[2]))
    mutated = {node: set(neighbors) for node, neighbors in adjacency.items()}
    mutated[cut_left].discard(cut_right)
    mutated[cut_right].discard(cut_left)
    subcomponents: list[list[str]] = []
    remaining = set(component)
    while remaining:
        start = min(remaining)
        stack = [start]
        seen = {start}
        while stack:
            node = stack.pop()
            for neighbor in mutated.get(node, ()):
                if neighbor in remaining and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        remaining -= seen
        subcomponents.extend(split_low_coherence_component(sorted(seen), mutated, pair_map, threshold))
    return subcomponents


def component_mean_similarity(component: list[str], pair_map: dict[tuple[str, str], float]) -> float:
    if len(component) <= 1:
        return 1.0
    values = []
    for index, left in enumerate(component):
        for right in component[index + 1 :]:
            values.append(pair_map.get((left, right), pair_map.get((right, left), 0.0)))
    return 0.0 if not values else float(np.mean(values))


def build_role_candidates(
    *,
    clusters: list[list[str]],
    neighborhoods: dict[str, NeighborhoodRecord],
    min_role_support: int,
    role_similarity_threshold: float,
) -> list[M3RoleCandidate]:
    output: list[M3RoleCandidate] = []
    for index, cluster in enumerate(sorted(clusters), start=1):
        records = [neighborhoods[family_id] for family_id in cluster]
        games = sorted({game for record in records for game in record.game_ids})
        game_families = sorted({family for record in records for family in record.game_family_ids})
        support_count = sum(record.support_count for record in records)
        member_count = len(cluster)
        mean_similarity = mean_cluster_similarity(records)
        mean_coherence = float(np.mean([record.family_coherence for record in records]))
        role_consistency = (
            0.45 * mean_similarity
            + 0.35 * mean_coherence
            + 0.20 * float(np.mean([record.mean_prediction_accuracy for record in records]))
        )
        dominant_motif_profile = normalized_counter(Counter(record.dominant_motif_candidate for record in records))
        incoming_profile = merge_counter_dicts(record.incoming_edge_profile for record in records)
        outgoing_profile = merge_counter_dicts(record.outgoing_edge_profile for record in records)
        future_effect_profile = normalized_counter(Counter(record.dominant_outcome_signature for record in records))
        stable = (
            support_count >= int(min_role_support)
            and role_consistency >= float(role_similarity_threshold)
            and len(games) >= 2
            and mean_coherence >= 0.70
        )
        status = "stable" if stable else ("singleton" if member_count == 1 else "weak")
        transfer_readiness = (
            0.30 * min(1.0, len(games) / 3.0)
            + 0.25 * min(1.0, len(game_families) / 2.0)
            + 0.25 * role_consistency
            + 0.20 * mean_coherence
        )
        output.append(
            M3RoleCandidate(
                role_id=f"m3-{index:04d}",
                role_label_candidate=label_role_candidate(records, dominant_motif_profile, future_effect_profile),
                member_family_ids=cluster,
                games_present=games,
                game_families_present=game_families,
                support_count=support_count,
                cross_game_support=len(games),
                cross_game_family_support=len(game_families),
                role_consistency_score=role_consistency,
                mean_neighborhood_similarity=mean_similarity,
                mean_family_coherence=mean_coherence,
                dominant_motif_profile=dominant_motif_profile,
                incoming_edge_profile=incoming_profile,
                outgoing_edge_profile=outgoing_profile,
                future_option_effect_profile=future_effect_profile,
                transfer_readiness_score=transfer_readiness,
                examples=role_examples(records),
                status=status,
                notes={
                    "stable": stable,
                    "member_count": member_count,
                },
            )
        )
    return output


def label_role_candidate(
    records: list[NeighborhoodRecord],
    motif_profile: dict[str, float],
    effect_profile: dict[str, float],
) -> str:
    dominant_motif = dominant_key(motif_profile)
    dominant_effect = dominant_key(effect_profile)
    if dominant_effect == "terminal_transition":
        return "terminator_candidate"
    if dominant_effect in {"blocked_no_change", "preserve_no_change"}:
        return "blocker_candidate" if motif_profile.get("block_candidate", 0.0) >= 0.30 else "reversible_preserver_candidate"
    if dominant_effect == "position_like_change":
        return "transporter_candidate" if motif_profile.get("change_candidate", 0.0) >= 0.40 else "movement_controller_candidate"
    if dominant_motif == "enable_candidate_unknown":
        return "enabler_candidate"
    if dominant_motif == "change_candidate":
        return "coverage_expander_candidate" if effect_profile.get("large_change", 0.0) >= 0.25 else "connector_candidate"
    if motif_profile.get("preserve_candidate", 0.0) >= 0.35:
        return "reversible_preserver_candidate"
    if dominant_motif == "terminate_candidate":
        return "trigger_candidate"
    return "unknown_role_candidate"


def role_examples(records: list[NeighborhoodRecord], limit: int = 5) -> list[dict[str, Any]]:
    output = []
    for record in records:
        output.append(
            {
                "family_id": record.family_id,
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


def core7_comparison(stable_roles: list[M3RoleCandidate], game_set_name: str) -> dict[str, Any]:
    core7_path = Path("runs/v6/v08_cd2_core7/v08_report.json")
    if not core7_path.exists() or game_set_name == "core7":
        return {}
    core = json.loads(core7_path.read_text(encoding="utf-8"))
    core_roles = core.get("report", {}).get("top_role_candidates", [])
    core_labels = {row.get("role_label_candidate") for row in core_roles}
    current_labels = {role.role_label_candidate for role in stable_roles}
    survived = sorted(core_labels & current_labels)
    new_only = sorted(current_labels - core_labels)
    return {
        "core7_scientific_conclusion": core.get("validation", {}).get("scientific_conclusion"),
        "core7_stable_clusters": core.get("report", {}).get("stable_clusters"),
        "extended_stable_clusters": len(stable_roles),
        "stable_role_candidates_survived_from_core7": survived,
        "new_role_candidates_only_in_extended": new_only,
        "blocker_candidate_survives": "blocker_candidate" in current_labels,
        "transporter_candidate_survives": "transporter_candidate" in current_labels,
        "connector_candidate_survives": "connector_candidate" in current_labels,
    }


def build_graph_outputs(
    neighborhoods: dict[str, NeighborhoodRecord],
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
        nodes[family_id] = {
            "node_id": family_id,
            "node_type": "m2_family",
            "dominant_outcome_signature": record.dominant_outcome_signature,
        }
        family = family_map.get(family_id)
        if family is not None:
            for contingency_id in family.contingency_ids:
                if contingency_id in m1_support:
                    edges.append({"edge_type": "member_of", "source_id": contingency_id, "target_id": family_id})
    for role in roles:
        nodes[role.role_id] = {
            "node_id": role.role_id,
            "node_type": "m3_role_candidate",
            "role_label_candidate": role.role_label_candidate,
            "status": role.status,
        }
        for family_id in role.member_family_ids:
            edges.append({"edge_type": "member_of", "source_id": family_id, "target_id": role.role_id})
    for pair in pair_results:
        for edge_type in pair.edge_types:
            edges.append(
                {
                    "edge_type": edge_type,
                    "source_id": pair.left_family_id,
                    "target_id": pair.right_family_id,
                    "similarity": pair.similarity,
                }
            )
    return list(nodes.values()), edges


def role_membership_rows(roles: list[M3RoleCandidate]) -> list[dict[str, Any]]:
    rows = []
    for role in roles:
        for family_id in role.member_family_ids:
            rows.append(
                {
                    "role_id": role.role_id,
                    "family_id": family_id,
                    "role_label_candidate": role.role_label_candidate,
                    "status": role.status,
                }
            )
    return rows


def neighborhood_output_rows(neighborhoods: dict[str, NeighborhoodRecord]) -> list[dict[str, Any]]:
    rows = []
    for record in sorted(neighborhoods.values(), key=lambda item: item.family_id):
        rows.append(
            {
                "family_id": record.family_id,
                "games_present": list(record.game_ids),
                "game_families_present": list(record.game_family_ids),
                "support_count": record.support_count,
                "family_coherence": record.family_coherence,
                "mean_prediction_accuracy": record.mean_prediction_accuracy,
                "mean_context_lift": record.mean_context_lift,
                "dominant_outcome_signature": record.dominant_outcome_signature,
                "dominant_motif_candidate": record.dominant_motif_candidate,
                "fingerprint": record.fingerprint,
                "incoming_edge_profile": record.incoming_edge_profile,
                "outgoing_edge_profile": record.outgoing_edge_profile,
                "examples": list(record.examples),
            }
        )
    return rows


def build_v08_payload(
    *,
    config: RoleCandidatesV08Config,
    game_set: GameSetManifest,
    selected_games: tuple[str, ...],
    neighborhoods: dict[str, NeighborhoodRecord],
    pair_results: list[PairResult],
    roles: list[M3RoleCandidate],
    rejected_clusters: list[list[str]],
) -> dict[str, Any]:
    stable_roles = [role for role in roles if role.status == "stable"]
    singleton_roles = [role for role in roles if role.status == "singleton"]
    weak_roles = [role for role in roles if role.status == "weak"]
    cross_game = [role for role in stable_roles if role.cross_game_support >= 2]
    cross_family = [role for role in stable_roles if role.cross_game_family_support >= 2]
    games_represented = sorted({game for role in stable_roles for game in role.games_present})
    game_families_represented = sorted({family for role in stable_roles for family in role.game_families_present})
    missing_manifest_games = sorted(set(game_set.games) - set(selected_games))
    missing_manifest_families = sorted(
        family_name for family_name, games in game_set.families.items() if not (set(selected_games) & set(games))
    )
    core7_anchor_games_present = sorted(set(game_set.core7_anchors) & set(selected_games))
    largest_cluster_size = max((len(role.member_family_ids) for role in roles), default=0)
    total_families = max(1, len(neighborhoods))
    largest_cluster_percent = largest_cluster_size / total_families
    mean_rcs = float(np.mean([role.role_consistency_score for role in stable_roles])) if stable_roles else 0.0
    is_extended = game_set.name == "extended32_v08" or len(game_set.games) >= 32
    if is_extended:
        weak = (
            len(stable_roles) >= 4
            and len(games_represented) >= 10
            and len(game_families_represented) >= 6
            and len(cross_family) >= 2
            and mean_rcs >= 0.70
        )
        strong = (
            len(stable_roles) >= 6
            and len(games_represented) >= 16
            and len(game_families_represented) >= 8
            and len(cross_family) >= 3
            and mean_rcs >= 0.75
            and largest_cluster_percent <= 0.40
        )
        very_strong = (
            len(stable_roles) >= 8
            and len(games_represented) >= 24
            and len(game_families_represented) >= 12
            and len(cross_family) >= 4
            and mean_rcs >= 0.80
            and largest_cluster_percent <= 0.35
        )
    else:
        weak = len(stable_roles) >= 3 and len(games_represented) >= 3 and mean_rcs >= 0.70
        strong = (
            len(stable_roles) >= 5
            and len(games_represented) >= 5
            and len(game_families_represented) >= 2
            and len(cross_game) >= 3
            and mean_rcs >= 0.75
        )
        very_strong = (
            len(stable_roles) >= 6
            and len(game_families_represented) >= 3
            and len(cross_game) >= 4
            and len(cross_family) >= 2
            and mean_rcs >= 0.80
            and largest_cluster_percent <= 0.40
        )
    if very_strong:
        conclusion = "m3_extended_very_strong_role_candidates" if is_extended else "m3_very_strong_role_candidates"
    elif strong:
        conclusion = "m3_extended_strong_role_candidates" if is_extended else "m3_strong_role_candidates"
    elif weak:
        conclusion = "m3_extended_weak_role_candidates" if is_extended else "m3_weak_role_candidates"
    else:
        conclusion = "m3_extended_not_established" if is_extended else "m3_not_established"
    role_candidates_by_family = Counter()
    for role in stable_roles:
        for family in role.game_families_present:
            role_candidates_by_family[family] += 1
    extended_validation_pass_level = (
        "very_strong" if very_strong else "strong" if strong else "weak" if weak else "failed"
    )
    comparison = core7_comparison(stable_roles, game_set.name)
    report = {
        "games_analyzed": list(selected_games),
        "game_families_analyzed": sorted({family for record in neighborhoods.values() for family in record.game_family_ids}),
        "core7_anchor_games_present": core7_anchor_games_present,
        "missing_manifest_games": missing_manifest_games,
        "missing_manifest_families": missing_manifest_families,
        "families_analyzed": len(neighborhoods),
        "similarity_pairs_evaluated": len(pair_results),
        "similarity_edges_created": sum(1 for pair in pair_results if "similar_to" in pair.edge_types),
        "clusters_total": len(roles),
        "stable_clusters": len(stable_roles),
        "stable_role_candidates": len(stable_roles),
        "weak_clusters": len(weak_roles),
        "singleton_clusters": len(singleton_roles),
        "rejected_clusters": len(rejected_clusters),
        "cross_game_clusters": len(cross_game),
        "cross_family_clusters": len(cross_family),
        "cross_game_role_candidates": len(cross_game),
        "cross_family_role_candidates": len(cross_family),
        "role_candidates_by_family": dict(sorted(role_candidates_by_family.items())),
        "largest_cluster_size": largest_cluster_size,
        "largest_cluster_percent": largest_cluster_percent,
        "top_role_candidates": [role.to_record() for role in sorted(stable_roles, key=lambda item: (-item.role_consistency_score, item.role_id))[:5]],
        "top_cross_family_role_candidates": [
            role.to_record()
            for role in sorted(cross_family, key=lambda item: (-item.cross_game_family_support, -item.role_consistency_score, item.role_id))[:5]
        ],
        "fragmentation_analysis": {
            "singleton_ratio": len(singleton_roles) / max(1, len(roles)),
            "weak_ratio": len(weak_roles) / max(1, len(roles)),
            "rejected_ratio": len(rejected_clusters) / max(1, len(roles) + len(rejected_clusters)),
        },
        "giant_cluster_analysis": {
            "largest_cluster_size": largest_cluster_size,
            "largest_cluster_percent": largest_cluster_percent,
            "giant_cluster_detected": largest_cluster_percent > 0.40,
        },
        "conclusion": conclusion,
        "extended_validation_pass_level": extended_validation_pass_level,
        "extended_validation_pending": is_extended and bool(missing_manifest_games),
        "game_set_name": game_set.name,
        "game_set_purpose": game_set.purpose,
        "core7_comparison": comparison,
    }
    validation = {
        "diagnostic_success": bool(neighborhoods),
        "weak_pass": weak,
        "strong_pass": strong,
        "very_strong_pass": very_strong,
        "scientific_conclusion": conclusion,
        "context_depth": int(config.context_depth),
        "extended_validation_pending": report["extended_validation_pending"],
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
            "game_set_name": game_set.name,
        },
        "validation": validation,
        "report": report,
    }


def write_v08_outputs(
    *,
    output_dir: Path,
    roles: list[M3RoleCandidate],
    membership_rows: list[dict[str, Any]],
    neighborhood_rows: list[dict[str, Any]],
    graph_nodes: list[dict[str, Any]],
    graph_edges: list[dict[str, Any]],
    payload: dict[str, Any],
) -> None:
    role_rows = [role.to_record() for role in roles]
    (output_dir / "m3_role_candidates.json").write_text(json.dumps(role_rows, indent=2), encoding="utf-8")
    _write_parquet(output_dir / "m3_role_candidates.parquet", role_rows)
    _write_parquet(output_dir / "role_candidate_membership.parquet", membership_rows)
    _write_parquet(output_dir / "role_neighborhoods.parquet", neighborhood_rows)
    _write_parquet(output_dir / "v08_graph_nodes.parquet", graph_nodes)
    _write_parquet(output_dir / "v08_graph_edges.parquet", graph_edges)
    (output_dir / "v08_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "v08_report.txt").write_text(format_v08_report(payload), encoding="utf-8")


def format_v08_report(payload: dict[str, Any]) -> str:
    report = payload["report"]
    validation = payload["validation"]
    lines = [
        "ARC-AGI3 v0.8-m3-role-candidate-graph-neighborhood-discovery",
        f"scientific_conclusion={validation['scientific_conclusion']}",
        f"game_set_name={report['game_set_name']}",
        f"games_analyzed={','.join(report['games_analyzed'])}",
        f"game_families_analyzed={','.join(report['game_families_analyzed'])}",
        f"missing_manifest_games={','.join(report.get('missing_manifest_games', []))}",
        f"missing_manifest_families={','.join(report.get('missing_manifest_families', []))}",
        f"families_analyzed={report['families_analyzed']}",
        f"stable_clusters={report['stable_clusters']}",
        f"cross_game_clusters={report['cross_game_clusters']}",
        f"cross_family_clusters={report['cross_family_clusters']}",
        f"largest_cluster_percent={report['largest_cluster_percent']:.6f}",
        "",
        "Top Role Candidates:",
    ]
    for row in report["top_role_candidates"]:
        lines.append(
            f"{row['role_id']} label={row['role_label_candidate']} support={row['support_count']} "
            f"cross_game={row['cross_game_support']} cross_family={row['cross_game_family_support']} "
            f"rcs={row['role_consistency_score']:.3f}"
        )
    comparison = report.get("core7_comparison", {})
    if comparison:
        lines.extend(
            [
                "",
                "Core7 Comparison:",
                f"core7_scientific_conclusion={comparison.get('core7_scientific_conclusion')}",
                f"core7_stable_clusters={comparison.get('core7_stable_clusters')}",
                f"extended_stable_clusters={comparison.get('extended_stable_clusters')}",
                f"stable_role_candidates_survived_from_core7={','.join(comparison.get('stable_role_candidates_survived_from_core7', []))}",
                f"new_role_candidates_only_in_extended={','.join(comparison.get('new_role_candidates_only_in_extended', []))}",
                f"blocker_candidate_survives={comparison.get('blocker_candidate_survives')}",
                f"transporter_candidate_survives={comparison.get('transporter_candidate_survives')}",
                f"connector_candidate_survives={comparison.get('connector_candidate_survives')}",
            ]
        )
    lines.extend(
        [
            "",
            "Analysis:",
            f"giant_cluster_detected={report['giant_cluster_analysis']['giant_cluster_detected']}",
            f"cross_family_role_candidates={report['cross_family_role_candidates']}",
            f"extended_validation_pass_level={report.get('extended_validation_pass_level')}",
            f"proceed_to_v09_role_transfer_testing={validation['scientific_conclusion'] in {'m3_extended_weak_role_candidates','m3_extended_strong_role_candidates','m3_extended_very_strong_role_candidates'}}",
        ]
    )
    lines.extend(
        [
            "",
            "Interpretation:",
            "M3 tests role-candidate discovery only.",
            "No transfer, concept, or Hydra claim is made here.",
        ]
    )
    return "\n".join(lines)


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


def mean_cluster_similarity(records: list[NeighborhoodRecord]) -> float:
    if len(records) <= 1:
        return 1.0
    values = []
    for index, left in enumerate(records):
        for right in records[index + 1 :]:
            values.append(vector_similarity(left.fingerprint, right.fingerprint))
    return 0.0 if not values else float(np.mean(values))


def merge_counter_dicts(items: Any) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in items:
        counter.update({str(key): int(value) for key, value in item.items()})
    return dict(sorted(counter.items()))


def normalized_counter(counter_like: dict[str, int] | Counter[str]) -> dict[str, float]:
    counter = Counter(counter_like)
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {str(key): float(value) / total for key, value in sorted(counter.items())}


def dominant_key(counter_like: dict[str, Any]) -> str:
    if not counter_like:
        return "unknown"
    return max(counter_like.items(), key=lambda item: (float(item[1]), str(item[0])))[0]


def mean_member_entropy(members: list[M1SupportRecord]) -> float:
    if not members:
        return 0.0
    return float(np.mean([member.entropy for member in members]))


def entropy_inverse(value: float) -> float:
    return 1.0 / (1.0 + max(0.0, float(value)))


def context_dependency_score(depth_counts: Counter[int]) -> float:
    total = sum(depth_counts.values())
    if total <= 0:
        return 0.0
    return sum(depth * count for depth, count in depth_counts.items()) / total / max(1, max(depth_counts))


def action_regularity_score(action_counts: Counter[int]) -> float:
    total = sum(action_counts.values())
    if total <= 0:
        return 0.0
    return max(action_counts.values()) / total


def _parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
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


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for key, value in record.items():
        if isinstance(value, (list, tuple, dict)):
            output[key] = json.dumps(value)
        else:
            output[key] = value
    return output


def _write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    normalized = [_normalize_record(record) for record in records]
    table = pa.Table.from_pylist(normalized) if normalized else pa.table({"_empty": pa.array([], type=pa.string())})
    pq.write_table(table, path, compression="zstd")
