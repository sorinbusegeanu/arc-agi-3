from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from v6.game_sets import load_game_set_manifest
from v6.memory_types import M2TransformationFamily


@dataclass(frozen=True)
class TransformationFamiliesV07Config:
    input_dir: str = "runs/v6/v06"
    output_dir: str = "runs/v6/v07"
    min_family_support: int = 5
    similarity_threshold: float = 0.70
    game_set_manifest: str | None = None
    game_set_name: str | None = None


@dataclass(frozen=True)
class M1Record:
    contingency_id: str
    game_id: str
    sampler_scope: str
    context_signature: tuple[str, ...]
    action: int
    outcome_signature: str
    support_count: int
    total_count: int
    prediction_accuracy: float
    prediction_error_rate: float
    entropy: float
    confidence: float
    future_option_motif_candidate: str
    discovered: bool
    notes: dict[str, Any]
    context_lift: float
    family_label_candidate: str


@dataclass
class FamilyAccumulator:
    family_id: str
    label_candidate: str
    members: list[M1Record]
    prototype: dict[str, float]
    prototype_sum: dict[str, float]


def run_transformation_families_v07(config: TransformationFamiliesV07Config) -> dict[str, Any]:
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    game_set = load_game_set_manifest(
        manifest_path=config.game_set_manifest,
        game_set_name=config.game_set_name,
        fallback_games=(),
    )

    contingencies = load_m1_contingencies(input_dir)
    requested_games = set(game_set.games)
    if requested_games:
        contingencies = [item for item in contingencies if item.game_id in requested_games]
    families = build_m2_families(
        contingencies,
        min_family_support=config.min_family_support,
        similarity_threshold=config.similarity_threshold,
    )
    mappings = contingency_family_rows(families)
    nodes, edges = graph_rows(families, similarity_threshold=config.similarity_threshold)

    stable = [family for family in families if family.stable]
    by_game = families_by_game(families)
    cross_game = [family.to_record() for family in stable if family.cross_game_presence > 1]
    compression_ratio = len(contingencies) / max(1, len(families))
    mean_coherence = 0.0 if not families else float(np.mean([family.family_coherence for family in families]))
    validation = validation_summary(
        families=families,
        contingencies=contingencies,
        compression_ratio=compression_ratio,
    )
    payload = {
        "config": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "min_family_support": int(config.min_family_support),
            "similarity_threshold": float(config.similarity_threshold),
            "game_set_name": game_set.name,
        },
        "validation": validation,
        "report": {
            "total_m1_contingencies_loaded": len(contingencies),
            "total_m2_family_candidates": len(families),
            "stable_m2_families": len(stable),
            "compression_ratio": compression_ratio,
            "families_by_game": by_game,
            "cross_game_families": cross_game,
            "mean_family_coherence": mean_coherence,
            "manifest_games_requested": list(game_set.games),
            "games_loaded": sorted({game for family in families for game in family.games_present}),
            "missing_manifest_games": sorted(requested_games - {game for family in families for game in family.games_present}),
            "missing_manifest_families": sorted(
                family_name
                for family_name, games in game_set.families.items()
                if not ({game for family in families for game in family.games_present} & set(games))
            ),
            "tested_theory_components": {
                "M0": "tested previously",
                "M1": "established by v0.6",
                "M2": "tested by v0.7",
                "M3": "no",
                "M4": "no",
            },
        },
    }
    write_v07_outputs(
        output_dir=output_dir,
        families=families,
        mappings=mappings,
        nodes=nodes,
        edges=edges,
        payload=payload,
    )
    return payload


def load_m1_contingencies(input_dir: Path) -> list[M1Record]:
    path = input_dir / "contingencies.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    action_baselines = compute_action_baselines(rows)
    output = []
    for row in rows:
        context_signature = tuple(parse_json_list(row.get("context_signature")))
        context_lift = float(row["prediction_accuracy"]) - action_baselines[(str(row["game_id"]), int(row["action"]))]
        label = label_candidate_for_row(
            outcome_signature=str(row["outcome_signature"]),
            motif_candidate=str(row["future_option_motif_candidate"]),
            action=int(row["action"]),
            context_signature=context_signature,
            prediction_accuracy=float(row["prediction_accuracy"]),
            context_lift=context_lift,
        )
        output.append(
            M1Record(
                contingency_id=str(row["contingency_id"]),
                game_id=str(row["game_id"]),
                sampler_scope=str(row["sampler_scope"]),
                context_signature=context_signature,
                action=int(row["action"]),
                outcome_signature=str(row["outcome_signature"]),
                support_count=int(row["support_count"]),
                total_count=int(row["total_count"]),
                prediction_accuracy=float(row["prediction_accuracy"]),
                prediction_error_rate=float(row["prediction_error_rate"]),
                entropy=float(row["entropy"]),
                confidence=float(row["confidence"]),
                future_option_motif_candidate=str(row["future_option_motif_candidate"]),
                discovered=bool(row["discovered"]),
                notes=parse_json_dict(row.get("notes")),
                context_lift=context_lift,
                family_label_candidate=label,
            )
        )
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


def label_candidate_for_row(
    *,
    outcome_signature: str,
    motif_candidate: str,
    action: int,
    context_signature: tuple[str, ...],
    prediction_accuracy: float,
    context_lift: float,
) -> str:
    if outcome_signature == "terminal_transition" or motif_candidate == "terminate_candidate":
        return "terminal_family_candidate"
    if outcome_signature in {"blocked_no_change", "preserve_no_change"}:
        return "blocked_no_change_family_candidate"
    if outcome_signature == "position_like_change":
        if context_lift >= 0.15:
            return "teleport_like_family_candidate"
        if action in {5, 6} or context_lift >= 0.08:
            return "push_like_family_candidate"
        return "position_like_change_family_candidate"
    if outcome_signature == "large_change":
        if context_lift >= 0.10:
            return "activation_like_family_candidate"
        return "coverage_change_family_candidate"
    if outcome_signature == "change":
        if context_lift >= 0.12:
            return "activation_like_family_candidate"
        if context_signature and prediction_accuracy >= 0.90:
            return "coverage_change_family_candidate"
        return "state_change_family_candidate"
    return "unknown_change_family_candidate"


def build_m2_families(
    contingencies: list[M1Record],
    *,
    min_family_support: int,
    similarity_threshold: float,
) -> list[M2TransformationFamily]:
    by_label: dict[str, list[M1Record]] = defaultdict(list)
    for contingency in contingencies:
        by_label[contingency.family_label_candidate].append(contingency)

    accumulators: list[FamilyAccumulator] = []
    family_index = 0
    for label, rows in sorted(by_label.items()):
        for row in sorted(rows, key=lambda item: (item.game_id, item.sampler_scope, item.contingency_id)):
            vector = contingency_vector(row)
            assigned = False
            for family in accumulators:
                if family.label_candidate != label:
                    continue
                similarity = vector_similarity(vector, family.prototype)
                if similarity >= float(similarity_threshold):
                    family.members.append(row)
                    update_family_prototype(family, vector)
                    assigned = True
                    break
            if not assigned:
                family_index += 1
                accumulators.append(
                    FamilyAccumulator(
                        family_id=f"m2-{family_index:04d}",
                        label_candidate=label,
                        members=[row],
                        prototype=vector,
                        prototype_sum=dict(vector),
                    )
                )

    total_families = len(accumulators)
    return [
        finalize_family(
            family,
            similarity_threshold=similarity_threshold,
            min_family_support=min_family_support,
            compression_ratio=len(family.members) / max(1, total_families),
        )
        for family in accumulators
    ]


def contingency_vector(row: M1Record) -> dict[str, float]:
    return {
        "outcome_position_like": 1.0 if row.outcome_signature == "position_like_change" else 0.0,
        "outcome_blocked": 1.0 if row.outcome_signature in {"blocked_no_change", "preserve_no_change"} else 0.0,
        "outcome_terminal": 1.0 if row.outcome_signature == "terminal_transition" else 0.0,
        "outcome_change": 1.0 if row.outcome_signature in {"change", "large_change"} else 0.0,
        "motif_block": 1.0 if row.future_option_motif_candidate == "block_candidate" else 0.0,
        "motif_change": 1.0 if row.future_option_motif_candidate == "change_candidate" else 0.0,
        "motif_terminate": 1.0 if row.future_option_motif_candidate == "terminate_candidate" else 0.0,
        "directional_action": 1.0 if row.action in {1, 2, 3, 4} else 0.0,
        "interactive_action": 1.0 if row.action in {5, 6} else 0.0,
        "context_depth": float(len(row.context_signature)),
        "prediction_accuracy": float(row.prediction_accuracy),
        "context_lift": float(row.context_lift),
        "entropy": float(row.entropy),
        "support_log": math.log1p(float(row.support_count)),
    }


def mean_vector(vectors: list[dict[str, float]]) -> dict[str, float]:
    keys = vectors[0].keys()
    return {key: float(np.mean([vector[key] for vector in vectors])) for key in keys}


def update_family_prototype(family: FamilyAccumulator, vector: dict[str, float]) -> None:
    count = len(family.members)
    for key, value in vector.items():
        family.prototype_sum[key] = float(family.prototype_sum.get(key, 0.0) + float(value))
    family.prototype = {key: family.prototype_sum[key] / count for key in family.prototype_sum}


def vector_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    left_values = np.asarray([float(left.get(key, 0.0)) for key in keys], dtype=float)
    right_values = np.asarray([float(right.get(key, 0.0)) for key in keys], dtype=float)
    numerator = float(np.dot(left_values, right_values))
    denominator = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
    cosine = 0.0 if denominator <= 0.0 else numerator / denominator
    return max(0.0, min(1.0, cosine))


def finalize_family(
    family: FamilyAccumulator,
    *,
    similarity_threshold: float,
    min_family_support: int,
    compression_ratio: float,
) -> M2TransformationFamily:
    members = family.members
    outcome_counts = Counter(member.outcome_signature for member in members)
    motif_counts = Counter(member.future_option_motif_candidate for member in members)
    dominant_outcome, dominant_count = max(outcome_counts.items(), key=lambda item: (item[1], item[0]))
    pairwise = pairwise_similarity(members)
    outcome_entropy = entropy(outcome_counts)
    motif_entropy = entropy(motif_counts)
    dominant_ratio = dominant_count / max(1, len(members))
    coherence = (
        0.4 * dominant_ratio
        + 0.4 * pairwise
        + 0.1 * entropy_to_coherence(outcome_entropy, len(outcome_counts))
        + 0.1 * entropy_to_coherence(motif_entropy, len(motif_counts))
    )
    support_total = sum(member.support_count for member in members)
    mean_accuracy = float(np.mean([member.prediction_accuracy for member in members]))
    mean_context_lift = float(np.mean([member.context_lift for member in members]))
    stable = support_total >= int(min_family_support) and coherence >= float(similarity_threshold) and mean_accuracy >= 0.75
    return M2TransformationFamily(
        family_id=family.family_id,
        family_label_candidate=family.label_candidate,
        games_present=sorted({member.game_id for member in members}),
        samplers_present=sorted({member.sampler_scope for member in members}),
        contingency_ids=[member.contingency_id for member in members],
        support_count=support_total,
        mean_prediction_accuracy=mean_accuracy,
        mean_context_lift=mean_context_lift,
        dominant_outcome_signature=dominant_outcome,
        outcome_signature_distribution=dict(sorted(outcome_counts.items())),
        motif_candidate_distribution=dict(sorted(motif_counts.items())),
        family_coherence=coherence,
        compression_ratio=float(compression_ratio),
        cross_game_presence=len({member.game_id for member in members}),
        stable=stable,
        examples=[
            {
                "contingency_id": member.contingency_id,
                "game_id": member.game_id,
                "action": member.action,
                "outcome_signature": member.outcome_signature,
                "context_signature": list(member.context_signature),
            }
            for member in members[:5]
        ],
        notes={
            "dominant_outcome_ratio": dominant_ratio,
            "mean_pairwise_similarity": pairwise,
            "outcome_entropy": outcome_entropy,
            "motif_entropy": motif_entropy,
        },
    )


def pairwise_similarity(members: list[M1Record]) -> float:
    if len(members) < 2:
        return 1.0
    vectors = [contingency_vector(member) for member in members]
    max_pairs = 2000
    step = max(1, math.ceil((len(vectors) * (len(vectors) - 1) / 2) / max_pairs))
    similarities = []
    pair_index = 0
    for index, left in enumerate(vectors):
        for right in vectors[index + 1 :]:
            if pair_index % step == 0:
                similarities.append(vector_similarity(left, right))
            pair_index += 1
    return 1.0 if not similarities else float(np.mean(similarities))


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values() if count > 0)


def entropy_to_coherence(value: float, classes: int) -> float:
    if classes <= 1:
        return 1.0
    max_entropy = math.log2(classes)
    return 1.0 - min(1.0, value / max_entropy)


def contingency_family_rows(families: list[M2TransformationFamily]) -> list[dict[str, Any]]:
    rows = []
    for family in families:
        for contingency_id in family.contingency_ids:
            rows.append(
                {
                    "contingency_id": contingency_id,
                    "family_id": family.family_id,
                    "family_label_candidate": family.family_label_candidate,
                    "stable": bool(family.stable),
                }
            )
    return rows


def families_by_game(families: list[M2TransformationFamily]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for family in families:
        for game in family.games_present:
            item = rows.setdefault(game, {"total_families": 0, "stable_families": 0, "m1_count": 0})
            item["total_families"] += 1
            item["m1_count"] += sum(1 for contingency_id in family.contingency_ids if contingency_id.startswith(f"m1-{game}-"))
            if family.stable:
                item["stable_families"] += 1
    for game, item in rows.items():
        item["game_compression_ratio"] = item["m1_count"] / max(1, item["total_families"])
    return dict(sorted(rows.items()))


def graph_rows(families: list[M2TransformationFamily], *, similarity_threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = []
    edges = []
    for family in families:
        nodes.append({"node_id": family.family_id, "node_type": "m2_family", "label_candidate": family.family_label_candidate})
        for contingency_id in family.contingency_ids:
            nodes.append({"node_id": contingency_id, "node_type": "m1_contingency"})
            edges.append({"edge_type": "member_of", "source_id": contingency_id, "target_id": family.family_id})
    for index, left in enumerate(families):
        for right in families[index + 1 :]:
            similarity = family_similarity(left, right)
            if similarity >= float(similarity_threshold):
                edges.append({"edge_type": "similar_to", "source_id": left.family_id, "target_id": right.family_id, "similarity": similarity})
    dedup_nodes = {node["node_id"]: node for node in nodes}
    return list(dedup_nodes.values()), edges


def family_similarity(left: M2TransformationFamily, right: M2TransformationFamily) -> float:
    left_vector = {
        "accuracy": left.mean_prediction_accuracy,
        "context_lift": left.mean_context_lift,
        "coherence": left.family_coherence,
        "terminal": 1.0 if left.dominant_outcome_signature == "terminal_transition" else 0.0,
        "blocked": 1.0 if left.dominant_outcome_signature in {"blocked_no_change", "preserve_no_change"} else 0.0,
    }
    right_vector = {
        "accuracy": right.mean_prediction_accuracy,
        "context_lift": right.mean_context_lift,
        "coherence": right.family_coherence,
        "terminal": 1.0 if right.dominant_outcome_signature == "terminal_transition" else 0.0,
        "blocked": 1.0 if right.dominant_outcome_signature in {"blocked_no_change", "preserve_no_change"} else 0.0,
    }
    return vector_similarity(left_vector, right_vector)


def validation_summary(
    *,
    families: list[M2TransformationFamily],
    contingencies: list[M1Record],
    compression_ratio: float,
) -> dict[str, Any]:
    stable = [family for family in families if family.stable]
    games_with_stable = {game for family in stable for game in family.games_present}
    cross_game = [family for family in stable if family.cross_game_presence > 1]
    weak = len(stable) >= 5 and len(games_with_stable) >= 4
    strong = len(stable) >= 8 and len(games_with_stable) >= 7 and compression_ratio > 2.0
    very_strong = len(stable) >= 10 and len(games_with_stable) >= 7 and compression_ratio > 3.0 and len(cross_game) >= 3
    if very_strong:
        conclusion = "m2_very_strong"
    elif strong:
        conclusion = "m2_strong"
    elif weak:
        conclusion = "m2_weak"
    else:
        conclusion = "m2_not_established"
    return {
        "diagnostic_success": bool(contingencies),
        "weak_pass": weak,
        "strong_pass": strong,
        "very_strong_pass": very_strong,
        "scientific_conclusion": conclusion,
        "M0_status": "tested previously",
        "M1_status": "established by v0.6",
        "M2_status": "tested by v0.7",
        "M3_status": "not tested",
        "M4_status": "not tested",
    }


def write_v07_outputs(
    *,
    output_dir: Path,
    families: list[M2TransformationFamily],
    mappings: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    payload: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    family_rows = [family.to_record() for family in families]
    (output_dir / "m2_families.json").write_text(json.dumps(family_rows, indent=2), encoding="utf-8")
    _write_parquet(output_dir / "m2_families.parquet", family_rows)
    _write_parquet(output_dir / "contingency_to_family.parquet", mappings)
    _write_parquet(output_dir / "m2_graph_nodes.parquet", nodes)
    _write_parquet(output_dir / "m2_graph_edges.parquet", edges)
    (output_dir / "v07_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "v07_report.txt").write_text(format_v07_report(payload), encoding="utf-8")


def _write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    normalized = [_normalize_record(record) for record in records]
    table = pa.Table.from_pylist(normalized) if normalized else pa.table({"_empty": pa.array([], type=pa.string())})
    pq.write_table(table, path, compression="zstd")


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for key, value in record.items():
        if isinstance(value, (list, tuple, dict)):
            output[key] = json.dumps(value)
        else:
            output[key] = value
    return output


def parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


def parse_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def format_v07_report(payload: dict[str, Any]) -> str:
    report = payload["report"]
    validation = payload["validation"]
    lines = [
        "ARC-AGI3 v0.7-m2-transformation-family-compression",
        f"scientific_conclusion={validation['scientific_conclusion']}",
        f"total_m1_contingencies_loaded={report['total_m1_contingencies_loaded']}",
        f"total_m2_family_candidates={report['total_m2_family_candidates']}",
        f"stable_m2_families={report['stable_m2_families']}",
        f"compression_ratio={report['compression_ratio']:.6f}",
        f"mean_family_coherence={report['mean_family_coherence']:.6f}",
        "",
        "Families By Game:",
    ]
    for game, row in sorted(report["families_by_game"].items()):
        lines.append(
            f"{game} total={row['total_families']} stable={row['stable_families']} "
            f"game_compression_ratio={row['game_compression_ratio']:.3f}"
        )
    lines.extend(
        [
            "",
            "Theory Status:",
            f"M0={validation['M0_status']}",
            f"M1={validation['M1_status']}",
            f"M2={validation['M2_status']}",
            f"M3={validation['M3_status']}",
            f"M4={validation['M4_status']}",
        ]
    )
    return "\n".join(lines)
