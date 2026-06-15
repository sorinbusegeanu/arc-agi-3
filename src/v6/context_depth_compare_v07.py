from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ContextDepthCompareConfig:
    runs: tuple[str, ...]
    labels: tuple[str, ...]
    output_dir: str = "runs/v6/v07_context_compare"


def run_context_depth_compare_v07(config: ContextDepthCompareConfig) -> dict[str, Any]:
    if len(config.runs) != len(config.labels):
        raise ValueError("--runs and --labels must have the same length")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    by_game_rows = []
    for run_path, label in zip(config.runs, config.labels, strict=True):
        row, game_rows = load_context_depth_run(Path(run_path), label)
        rows.append(row)
        by_game_rows.extend(game_rows)

    enrich_scores(rows)
    best = select_best_context_depth(rows)
    statuses = classify_context_depth_statuses(rows, best_label=best["label"])
    extended_best = select_best_context_depth(rows) if any(int(row.get("game_count", 0)) > 7 for row in rows) else None
    payload = {
        "runs": rows,
        "best_context_depth": best["context_depth"],
        "best_context_depth_core": best["context_depth"],
        "best_context_depth_extended": None if extended_best is None else extended_best["context_depth"],
        "context_depth_0_status": statuses.get(0, "inconclusive"),
        "context_depth_1_status": statuses.get(1, "inconclusive"),
        "context_depth_2_status": statuses.get(2, "inconclusive"),
        "cd3_status": statuses.get(3, "inconclusive"),
        "recommended_context_depth_for_v08": best["context_depth"],
        "recommended_context_depth_for_extended_games": None if extended_best is None else extended_best["context_depth"],
        "reason": best["reason"],
    }
    write_context_depth_outputs(output_dir, payload, by_game_rows)
    return payload


def load_context_depth_run(run_dir: Path, label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report_v07 = json.loads((run_dir / "v07_report.json").read_text(encoding="utf-8"))
    v06_input = Path(report_v07["config"]["input_dir"])
    report_v06 = json.loads((v06_input / "v06_report.json").read_text(encoding="utf-8")) if (v06_input / "v06_report.json").exists() else None
    families = json.loads((run_dir / "m2_families.json").read_text(encoding="utf-8"))

    context_depth = int(report_v06["config"]["context_depth"]) if report_v06 else infer_depth_from_label(label)
    stable_families = [family for family in families if bool(family.get("stable"))]
    family_sizes = [len(parse_json_list(family.get("contingency_ids"))) for family in families]
    tiny_ratio = 0.0 if not families else sum(1 for size in family_sizes if size <= 2) / len(families)
    unknown_ratio = 0.0 if not families else sum(1 for family in families if family["family_label_candidate"] == "unknown_change_family_candidate") / len(families)
    entropies = [family_entropy(family) for family in families]
    mean_contingencies_per_family = 0.0 if not family_sizes else float(np.mean(family_sizes))
    m1_game_rows = report_v06["report"]["game_summary"] if report_v06 else []
    mean_prediction_accuracy = 0.0 if not m1_game_rows else float(np.mean([row["mean_prediction_accuracy"] for row in m1_game_rows]))
    mean_context_lift = 0.0 if not m1_game_rows else float(np.mean([row["context_lift"] for row in m1_game_rows]))
    sparse_ratio = sparse_contingency_ratio(v06_input, report_v06)
    cross_game_family_count = sum(1 for family in stable_families if int(family["cross_game_presence"]) > 1)
    all_game_count = len(report_v06["config"]["games"]) if report_v06 else 7
    families_present_in_all_games = sum(1 for family in stable_families if int(family["cross_game_presence"]) == all_game_count)

    row = {
        "label": label,
        "run_dir": str(run_dir),
        "input_dir": str(v06_input),
        "context_depth": context_depth,
        "total_contingency_candidates": int(report_v06["report"]["total_contingency_candidates"]) if report_v06 else 0,
        "total_discovered_contingencies": int(report_v06["report"]["total_discovered_contingencies"]) if report_v06 else 0,
        "mean_prediction_accuracy": mean_prediction_accuracy,
        "mean_context_lift": mean_context_lift,
        "discovered_contingencies_by_game": report_v06["report"]["discovered_contingencies_by_game"] if report_v06 else {},
        "sparse_contingency_ratio": sparse_ratio,
        "total_m2_family_candidates": int(report_v07["report"]["total_m2_family_candidates"]),
        "stable_m2_families": int(report_v07["report"]["stable_m2_families"]),
        "compression_ratio": float(report_v07["report"]["compression_ratio"]),
        "mean_family_coherence": float(report_v07["report"]["mean_family_coherence"]),
        "median_family_coherence": 0.0 if not families else float(np.median([family["family_coherence"] for family in families])),
        "cross_game_family_count": cross_game_family_count,
        "families_present_in_all_games": families_present_in_all_games,
        "singletons_or_tiny_family_ratio": tiny_ratio,
        "unknown_family_ratio": unknown_ratio,
        "family_entropy_mean": 0.0 if not entropies else float(np.mean(entropies)),
        "mean_contingencies_per_family": mean_contingencies_per_family,
        "game_count": all_game_count,
    }
    if context_depth == 3:
        cd2_reference = None
        # filled later in enrich_scores
        row["cd3_m1_candidate_growth_vs_cd2"] = None
        row["cd3_discovered_growth_vs_cd2"] = None
        row["cd3_stable_family_growth_vs_cd2"] = None
        row["cd3_compression_ratio_vs_cd2"] = None
        row["cd3_mean_family_coherence_vs_cd2"] = None
        row["cd3_cross_game_family_count_vs_cd2"] = None
        row["cd3_singleton_or_tiny_family_ratio_vs_cd2"] = None
        row["cd3_unknown_family_ratio_vs_cd2"] = None
    by_game_rows = [
        {
            "label": label,
            "context_depth": context_depth,
            "game": row_v06["game"],
            "discovered_contingencies": row_v06["discovered_contingency_count"],
            "mean_prediction_accuracy": row_v06["mean_prediction_accuracy"],
            "context_lift": row_v06["context_lift"],
            "stable_families": report_v07["report"]["families_by_game"].get(row_v06["game"], {}).get("stable_families", 0),
            "total_families": report_v07["report"]["families_by_game"].get(row_v06["game"], {}).get("total_families", 0),
            "game_compression_ratio": report_v07["report"]["families_by_game"].get(row_v06["game"], {}).get("game_compression_ratio", 0.0),
        }
        for row_v06 in m1_game_rows
    ]
    return row, by_game_rows


def sparse_contingency_ratio(v06_input: Path, report_v06: dict[str, Any] | None) -> float:
    if report_v06 is None:
        return 0.0
    path = v06_input / "contingencies.json"
    if not path.exists():
        return 0.0
    rows = json.loads(path.read_text(encoding="utf-8"))
    min_support = int(report_v06["config"]["min_support"])
    if not rows:
        return 0.0
    sparse = sum(1 for row in rows if int(row["support_count"]) < min_support)
    return sparse / len(rows)


def family_entropy(family: dict[str, Any]) -> float:
    distribution = parse_json_dict(family.get("outcome_signature_distribution"))
    total = sum(int(value) for value in distribution.values())
    if total <= 0:
        return 0.0
    return -sum((int(value) / total) * math.log2(int(value) / total) for value in distribution.values() if int(value) > 0)


def enrich_scores(rows: list[dict[str, Any]]) -> None:
    metrics = {
        "mean_prediction_accuracy": True,
        "mean_context_lift": True,
        "mean_family_coherence": True,
        "compression_ratio": True,
        "cross_game_family_count": True,
        "singletons_or_tiny_family_ratio": False,
        "unknown_family_ratio": False,
    }
    z_values: dict[str, dict[int, float]] = {}
    for key in metrics:
        values = [float(row[key]) for row in rows]
        z_values[key] = z_score(values)
    for index, row in enumerate(rows):
        row["overfragmentation_score"] = (
            z_values["singletons_or_tiny_family_ratio"][index]
            + z_values["unknown_family_ratio"][index]
            - z_values["compression_ratio"][index]
            - z_values["cross_game_family_count"][index]
            - z_values["mean_family_coherence"][index]
        )
    overfrag_norm = normalize([row["overfragmentation_score"] for row in rows], higher_is_better=False)
    acc_norm = normalize([row["mean_prediction_accuracy"] for row in rows], higher_is_better=True)
    lift_norm = normalize([row["mean_context_lift"] for row in rows], higher_is_better=True)
    coh_norm = normalize([row["mean_family_coherence"] for row in rows], higher_is_better=True)
    comp_norm = normalize([row["compression_ratio"] for row in rows], higher_is_better=True)
    cross_norm = normalize([row["cross_game_family_count"] for row in rows], higher_is_better=True)
    for index, row in enumerate(rows):
        row["context_depth_score"] = (
            0.25 * acc_norm[index]
            + 0.20 * lift_norm[index]
            + 0.25 * coh_norm[index]
            + 0.20 * comp_norm[index]
            + 0.10 * cross_norm[index]
            - 0.20 * overfrag_norm[index]
        )
    attach_cd3_deltas(rows)


def attach_cd3_deltas(rows: list[dict[str, Any]]) -> None:
    cd2 = next((row for row in rows if int(row["context_depth"]) == 2), None)
    cd3 = next((row for row in rows if int(row["context_depth"]) == 3), None)
    if not cd2 or not cd3:
        return
    cd3["cd3_m1_candidate_growth_vs_cd2"] = ratio_delta(cd3["total_contingency_candidates"], cd2["total_contingency_candidates"])
    cd3["cd3_discovered_growth_vs_cd2"] = ratio_delta(cd3["total_discovered_contingencies"], cd2["total_discovered_contingencies"])
    cd3["cd3_stable_family_growth_vs_cd2"] = ratio_delta(cd3["stable_m2_families"], cd2["stable_m2_families"])
    cd3["cd3_compression_ratio_vs_cd2"] = ratio_delta(cd3["compression_ratio"], cd2["compression_ratio"])
    cd3["cd3_mean_family_coherence_vs_cd2"] = ratio_delta(cd3["mean_family_coherence"], cd2["mean_family_coherence"])
    cd3["cd3_cross_game_family_count_vs_cd2"] = ratio_delta(cd3["cross_game_family_count"], cd2["cross_game_family_count"])
    cd3["cd3_singleton_or_tiny_family_ratio_vs_cd2"] = ratio_delta(cd3["singletons_or_tiny_family_ratio"], cd2["singletons_or_tiny_family_ratio"], zero_default=0.0)
    cd3["cd3_unknown_family_ratio_vs_cd2"] = ratio_delta(cd3["unknown_family_ratio"], cd2["unknown_family_ratio"], zero_default=0.0)


def ratio_delta(value: float, reference: float, *, zero_default: float | None = None) -> float:
    value = float(value)
    reference = float(reference)
    if math.isclose(reference, 0.0):
        if zero_default is not None:
            return value - zero_default
        return 0.0 if math.isclose(value, 0.0) else 1.0
    return (value - reference) / abs(reference)


def z_score(values: list[float]) -> dict[int, float]:
    mean = float(np.mean(values)) if values else 0.0
    std = float(np.std(values)) if values else 0.0
    if std <= 0.0:
        return {index: 0.0 for index in range(len(values))}
    return {index: (value - mean) / std for index, value in enumerate(values)}


def normalize(values: list[float], *, higher_is_better: bool) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [0.5 for _ in values]
    if higher_is_better:
        return [(value - low) / (high - low) for value in values]
    return [(high - value) / (high - low) for value in values]


def select_best_context_depth(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (-row["context_depth_score"], row["context_depth"]))
    best = ordered[0]
    cd1 = next((row for row in rows if int(row["context_depth"]) == 1), None)
    cd2 = next((row for row in rows if int(row["context_depth"]) == 2), None)
    cd3 = next((row for row in rows if int(row["context_depth"]) == 3), None)
    if cd2 and cd3:
        if best["label"] == cd3["label"]:
            if not cd3_acceptable(cd2, cd3):
                best = cd2
                reason = "cd3 improves prediction specificity but does not meet M2 abstraction criteria; keep cd2"
                return {"label": best["label"], "context_depth": 2, "reason": reason}
            improvement = cd3["context_depth_score"] - cd2["context_depth_score"]
            if improvement > 0.0 and improvement < 0.05:
                return {"label": cd2["label"], "context_depth": 2, "reason": "cd3 is less than 5% better than cd2; keep simpler cd2"}
    if cd1 and cd2:
        if best["label"] == cd2["label"]:
            improvement = cd2["context_depth_score"] - cd1["context_depth_score"]
            if improvement > 0.0 and improvement < 0.05:
                return {"label": cd1["label"], "context_depth": 1, "reason": "cd2 improves less than 5% over cd1; prefer cd1"}
            if cd1["context_depth_score"] > next((row["context_depth_score"] for row in rows if int(row["context_depth"]) == 0), -1.0):
                if is_over_fragmented(cd2) and not is_over_fragmented(cd1):
                    return {"label": cd1["label"], "context_depth": 1, "reason": "cd1 improves over cd0 while cd2 over-fragments families"}
    return {"label": best["label"], "context_depth": int(best["context_depth"]), "reason": f"highest context_depth_score={best['context_depth_score']:.6f}"}


def cd3_acceptable(cd2: dict[str, Any], cd3: dict[str, Any]) -> bool:
    if int(cd3.get("game_count", 0)) <= 7:
        # Extended-game requirement cannot be validated on the core-only set.
        return False
    lift_gain = ratio_delta(cd3["mean_context_lift"], cd2["mean_context_lift"], zero_default=0.0)
    coherence_drop = ratio_delta(cd3["mean_family_coherence"], cd2["mean_family_coherence"], zero_default=0.0)
    tiny_delta = float(cd3["singletons_or_tiny_family_ratio"]) - float(cd2["singletons_or_tiny_family_ratio"])
    unknown_delta = float(cd3["unknown_family_ratio"]) - float(cd2["unknown_family_ratio"])
    return (
        lift_gain >= 0.05
        and coherence_drop >= -0.02
        and int(cd3["cross_game_family_count"]) >= int(cd2["cross_game_family_count"])
        and tiny_delta <= 0.05
        and unknown_delta <= 0.05
        and int(cd3["stable_m2_families"]) >= int(cd2["stable_m2_families"])
    )


def is_over_fragmented(row: dict[str, Any]) -> bool:
    return (
        row["singletons_or_tiny_family_ratio"] > 0.30
        or row["unknown_family_ratio"] > 0.30
        or row["mean_family_coherence"] < 0.80
    )


def classify_context_depth_statuses(rows: list[dict[str, Any]], *, best_label: str) -> dict[int, str]:
    statuses: dict[int, str] = {}
    for row in rows:
        depth = int(row["context_depth"])
        if row["label"] == best_label:
            statuses[depth] = "balanced"
        elif depth == 0 and row["mean_context_lift"] < 0.05:
            statuses[depth] = "under_contextualized"
        elif depth == 3:
            cd2 = next((item for item in rows if int(item["context_depth"]) == 2), None)
            if cd2 and (
                float(row["mean_prediction_accuracy"]) > float(cd2["mean_prediction_accuracy"])
                and (
                    float(row["compression_ratio"]) < float(cd2["compression_ratio"])
                    or float(row["mean_family_coherence"]) < float(cd2["mean_family_coherence"])
                )
            ):
                statuses[depth] = "over_contextualized"
            elif is_over_fragmented(row):
                statuses[depth] = "over_fragmented"
            else:
                statuses[depth] = "inconclusive"
        elif depth in {2, 3} and is_over_fragmented(row):
            statuses[depth] = "over_fragmented"
        else:
            statuses[depth] = "inconclusive"
    return statuses


def write_context_depth_outputs(output_dir: Path, payload: dict[str, Any], by_game_rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "context_depth_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "context_depth_comparison.txt").write_text(format_context_depth_comparison(payload), encoding="utf-8")
    write_parquet(output_dir / "context_depth_by_game.parquet", by_game_rows)


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    normalized = [_normalize_row(row) for row in rows]
    table = pa.Table.from_pylist(normalized) if normalized else pa.table({"_empty": pa.array([], type=pa.string())})
    pq.write_table(table, path, compression="zstd")


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for key, value in row.items():
        if isinstance(value, (list, tuple, dict)):
            output[key] = json.dumps(value)
        else:
            output[key] = value
    return output


def format_context_depth_comparison(payload: dict[str, Any]) -> str:
    lines = [
        "ARC-AGI3 v0.7 context-depth comparison",
        f"best_context_depth={payload['best_context_depth']}",
        f"best_context_depth_core={payload['best_context_depth_core']}",
        f"best_context_depth_extended={payload['best_context_depth_extended']}",
        f"recommended_context_depth_for_v08={payload['recommended_context_depth_for_v08']}",
        f"recommended_context_depth_for_extended_games={payload['recommended_context_depth_for_extended_games']}",
        f"reason={payload['reason']}",
        f"context_depth_0_status={payload['context_depth_0_status']}",
        f"context_depth_1_status={payload['context_depth_1_status']}",
        f"context_depth_2_status={payload['context_depth_2_status']}",
        f"cd3_status={payload['cd3_status']}",
        "",
        "Runs:",
    ]
    for row in sorted(payload["runs"], key=lambda item: int(item["context_depth"])):
        lines.append(
            f"{row['label']} depth={row['context_depth']} score={row['context_depth_score']:.6f} "
            f"m1_acc={row['mean_prediction_accuracy']:.6f} lift={row['mean_context_lift']:.6f} "
            f"m2_coherence={row['mean_family_coherence']:.6f} compression={row['compression_ratio']:.6f} "
            f"cross_game={row['cross_game_family_count']} overfrag={row['overfragmentation_score']:.6f}"
        )
    return "\n".join(lines)


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


def infer_depth_from_label(label: str) -> int:
    match = "".join(ch for ch in label if ch.isdigit())
    return int(match) if match else 0
