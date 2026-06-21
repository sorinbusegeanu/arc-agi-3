from __future__ import annotations

import gc
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v6.concept_candidates_v10fixb import (
    annotate_projection_outcomes,
    apply_target_metrics,
    build_raw_candidate_row,
    build_source_role_map,
    canonical_role_fingerprint,
    future_option_behavior_features,
    graph_position_features,
    mean_metric,
    merge_exact_candidates,
    predecessor_successor_profile,
    score_concept_against_target_family,
)
from v6.m4_failure_diagnostics import count_by_reason, ensure_failure_buckets, merge_reason_counts
from v6.role_transfer_v09 import _write_parquet, appearance_features, cosine_similarity, mean_vector
from v6.role_transfer_v09c import (
    RoleTransferV09cConfig,
    SingleFamilyContext,
    build_single_family_context,
    detect_available_memory_bytes,
    list_heldout_families,
)

PROCESS_POOL_CLASS = ProcessPoolExecutor


@dataclass(frozen=True)
class M4RoleConceptsV10eConfig:
    m3_input_dir: str = "runs/v6/v08d_cd2_extended32_sourceclean"
    transfer_input_dir: str = "runs/v6/v09c_transfer_hardened_extended32"
    m2_input_dir: str = "runs/v6/v07_cd2_extended32_expanded"
    m1_input_dir: str = "runs/v6/v06_cd2_extended32"
    previous_v09b_dir: str = "runs/v6/v09b_role_transfer_refined_sourceclean_extended32"
    output_dir: str = "runs/v6/v10e_role_based_m4_extended32"
    game_set_manifest: str | None = None
    game_set_name: str | None = None
    workers: int = 1
    min_games: int = 2
    min_manifest_families: int = 1
    min_role_count: int = 2
    max_role_count: int = 3
    max_role_items_per_family: int = 128
    max_candidates_per_heldout: int = 250000
    candidate_chunk_size: int = 5000
    worker_memory_gib: int = 12


def run_m4_role_concepts_v10e(config: M4RoleConceptsV10eConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    transfer_report = json.loads((Path(config.transfer_input_dir) / "v09c_report.json").read_text(encoding="utf-8"))
    transfer_rows = pd.read_parquet(Path(config.transfer_input_dir) / "v09c_hardened_assignments.parquet").to_dict(orient="records")
    transfer_by_heldout = defaultdict(list)
    for row in transfer_rows:
        transfer_by_heldout[str(row.get("heldout_family", ""))].append(row)

    role_config = RoleTransferV09cConfig(
        m2_input_dir=config.m2_input_dir,
        m1_input_dir=config.m1_input_dir,
        previous_v09b_dir=config.previous_v09b_dir,
        output_dir=config.output_dir,
        game_set_manifest=config.game_set_manifest,
        game_set_name=config.game_set_name,
        workers=1,
    )
    heldout_families = list_heldout_families(role_config)
    effective_workers = choose_v10e_worker_count(config.workers, len(heldout_families), config.worker_memory_gib)
    multiprocessing_used = effective_workers > 1 and len(heldout_families) > 1

    family_results = []
    if multiprocessing_used:
        with PROCESS_POOL_CLASS(max_workers=effective_workers) as executor:
            futures = [
                executor.submit(
                    run_single_family_v10e,
                    heldout_family,
                    role_config,
                    transfer_by_heldout.get(heldout_family, []),
                    config,
                    str(shards_dir),
                )
                for heldout_family in heldout_families
            ]
            family_results = [future.result() for future in futures]
    else:
        for heldout_family in heldout_families:
            family_results.append(
                run_single_family_v10e(
                    heldout_family,
                    role_config,
                    transfer_by_heldout.get(heldout_family, []),
                    config,
                    str(shards_dir),
                )
            )
            gc.collect()

    family_results = sorted(family_results, key=lambda row: row["heldout_family"])
    family_failure_rows = [row["failure_diagnostics"] for row in family_results]
    failure_diagnostics = _build_v10e_failure_diagnostics(family_failure_rows, merge_diag={})
    raw_candidate_rows = load_v10e_shards(shards_dir, "role_based_candidates")
    transfer_score_rows = load_v10e_shards(shards_dir, "role_based_transfer_scores")
    target_family_rows = load_v10e_shards(shards_dir, "role_based_target_family_scores")
    fallback_rows = load_v10e_shards(shards_dir, "fallback_diagnostic_candidates")
    fallback_score_rows = load_v10e_shards(shards_dir, "fallback_diagnostic_scores")
    by_family_rows = load_v10e_shards(shards_dir, "family_summary")

    exact_rows, merge_diag = merge_exact_candidates(
        raw_candidate_rows,
        min_games=config.min_games,
        min_manifest_families=config.min_manifest_families,
        min_role_count=config.min_role_count,
    )
    failure_diagnostics = _build_v10e_failure_diagnostics(family_failure_rows, merge_diag=merge_diag)
    concept_rows = apply_target_metrics(exact_rows, transfer_score_rows)
    concept_rows = annotate_projection_outcomes(concept_rows, transfer_score_rows)
    concept_rows = apply_role_based_gates(concept_rows, transfer_score_rows)
    concept_rows = sorted(concept_rows, key=lambda row: row["concept_id"])

    fallback_only_signal_detected = any(
        float(row.get("target_family_score", 0.0)) > float(row.get("best_individual_role_baseline_raw", 0.0))
        and float(row.get("target_family_score", 0.0)) > float(row.get("best_surface_raw_baseline", 0.0))
        for row in fallback_score_rows
    )
    transferable = [
        row
        for row in concept_rows
        if row.get("candidate_source") == "stable_role" and is_transferable_role_based_concept(row)
    ]

    rejected_rows = build_rejected_candidate_rows(concept_rows)
    identity_rows = build_concept_identity_rows(raw_candidate_rows)
    compression_rows = build_compression_rows(concept_rows)
    future_rows = build_future_option_rows(target_family_rows)
    baseline_rows = build_baseline_rows(target_family_rows)
    failure_decomposition_rows = build_failure_decomposition_rows(concept_rows, target_family_rows)
    generator_failure_rows = build_generator_failure_summary_rows(failure_decomposition_rows)
    projection_audit_rows = build_projection_audit_rows(concept_rows, target_family_rows)
    baseline_dominance_rows = build_baseline_dominance_rows(failure_decomposition_rows)
    representation_loss_rows = build_representation_loss_rows(target_family_rows)
    closest_candidate_rows = build_closest_candidate_rows(concept_rows)
    generator_rows = build_candidate_generator_rows(by_family_rows)
    multiprocessing_rows = build_multiprocessing_rows(family_results, requested_workers=config.workers, effective_workers=effective_workers, multiprocessing_used=multiprocessing_used)
    cap_rows = build_candidate_cap_rows(by_family_rows)

    payload = build_v10e_report(
        config=config,
        transfer_report=transfer_report,
        concept_rows=concept_rows,
        transferable_rows=transferable,
        transfer_rows=transfer_score_rows,
        fallback_rows=fallback_rows,
        fallback_score_rows=fallback_score_rows,
        by_family_rows=by_family_rows,
        merge_diag=merge_diag,
        fallback_only_signal_detected=fallback_only_signal_detected,
        requested_workers=config.workers,
        effective_workers=effective_workers,
        multiprocessing_used=multiprocessing_used,
        generator_rows=generator_rows,
        cap_rows=cap_rows,
        failure_decomposition_rows=failure_decomposition_rows,
        generator_failure_rows=generator_failure_rows,
        projection_audit_rows=projection_audit_rows,
        baseline_dominance_rows=baseline_dominance_rows,
        closest_candidate_rows=closest_candidate_rows,
        representation_loss_rows=representation_loss_rows,
        failure_diagnostics=failure_diagnostics,
    )

    _write_parquet(output_dir / "role_based_m4_concepts.parquet", concept_rows)
    (output_dir / "role_based_m4_concepts.json").write_text(json.dumps(concept_rows, indent=2), encoding="utf-8")
    _write_parquet(output_dir / "role_based_concept_transfer_scores.parquet", transfer_score_rows)
    _write_parquet(output_dir / "role_based_concept_by_family.parquet", by_family_rows)
    _write_parquet(output_dir / "role_based_concept_compression_scores.parquet", compression_rows)
    _write_parquet(output_dir / "role_based_future_option_prediction_scores.parquet", future_rows)
    _write_parquet(output_dir / "role_based_baseline_comparison.parquet", baseline_rows)
    _write_parquet(output_dir / "fallback_diagnostic_candidates.parquet", fallback_rows)
    _write_parquet(output_dir / "fallback_diagnostic_scores.parquet", fallback_score_rows)
    _write_parquet(output_dir / "rejected_candidate_diagnostics.parquet", rejected_rows)
    _write_parquet(output_dir / "concept_identity_diagnostics.parquet", identity_rows)
    _write_parquet(output_dir / "candidate_generator_diagnostics.parquet", generator_rows)
    _write_parquet(output_dir / "multiprocessing_diagnostics.parquet", multiprocessing_rows)
    _write_parquet(output_dir / "candidate_cap_diagnostics.parquet", cap_rows)
    _write_parquet(output_dir / "v10e_failure_decomposition.parquet", failure_decomposition_rows)
    _write_parquet(output_dir / "v10e_generator_failure_summary.parquet", generator_failure_rows)
    _write_parquet(output_dir / "v10e_projection_audit.parquet", projection_audit_rows)
    _write_parquet(output_dir / "v10e_baseline_dominance_audit.parquet", baseline_dominance_rows)
    _write_parquet(output_dir / "v10e_representation_loss_audit.parquet", representation_loss_rows)
    _write_parquet(output_dir / "v10e_closest_candidates.parquet", closest_candidate_rows)
    (output_dir / "m4_failure_diagnostics.json").write_text(json.dumps({"m4_failure_diagnostics": failure_diagnostics}, indent=2), encoding="utf-8")
    (output_dir / "v10e_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "v10e_report.txt").write_text(format_v10e_report(payload), encoding="utf-8")
    return payload


def run_single_family_v10e(
    heldout_family: str,
    role_config: RoleTransferV09cConfig,
    target_rows: list[dict[str, Any]],
    config: M4RoleConceptsV10eConfig,
    shards_dir: str,
) -> dict[str, Any]:
    context = build_single_family_context(role_config, heldout_family)
    shard_root = Path(shards_dir)

    source_role_map = build_source_role_map(context.source_roles)
    stable_items = build_stable_role_items(context, max_role_items=config.max_role_items_per_family)
    fallback_rows = build_fallback_diagnostic_rows(context, source_role_map)
    raw_candidate_rows, generator_counts, cap_hit, dropped_due_to_cap = generate_role_based_candidates(
        context,
        stable_items,
        max_role_count=config.max_role_count,
        max_candidates=config.max_candidates_per_heldout,
    )

    local_exact_rows, _ = merge_exact_candidates(
        raw_candidate_rows,
        min_games=1,
        min_manifest_families=1,
        min_role_count=config.min_role_count,
    )
    transfer_rows: list[dict[str, Any]] = []
    local_concept_rows: list[dict[str, Any]] = []
    target_family_rows: list[dict[str, Any]] = []
    for chunk_start in range(0, len(local_exact_rows), max(1, config.candidate_chunk_size)):
        for concept in local_exact_rows[chunk_start : chunk_start + max(1, config.candidate_chunk_size)]:
            projection, per_target_rows = evaluate_role_based_projection_by_family(concept, context, target_rows)
            enriched = {"heldout_family": context.heldout_family, "candidate_source": "stable_role", **concept, **projection}
            transfer_rows.append(enriched)
            local_concept_rows.append(enriched)
            target_family_rows.extend(
                {
                    "heldout_family": context.heldout_family,
                    "candidate_source": "stable_role",
                    "concept_id": concept["concept_id"],
                    **row,
                }
                for row in per_target_rows
            )

    fallback_score_rows = score_fallback_rows(context, fallback_rows, target_rows)
    rejected_rows = [row for row in build_rejected_candidate_rows(local_concept_rows) if row.get("rejection_reasons")]
    family_failure = _build_v10e_family_failure_row(
        heldout_family=context.heldout_family,
        context=context,
        source_role_map=source_role_map,
        stable_items=stable_items,
        raw_candidate_rows=raw_candidate_rows,
        projection_rows=local_concept_rows,
        target_family_rows=target_family_rows,
        rejected_rows=rejected_rows,
        fallback_rows=fallback_rows,
        min_role_count=config.min_role_count,
    )
    summary = {
        "heldout_family": context.heldout_family,
        "source_only_concept_discovery": True,
        "role_based_candidate_count": len(local_exact_rows),
        "raw_candidate_row_count": len(raw_candidate_rows),
        "fallback_candidate_count": len(fallback_rows),
        "positive_lift_families": sum(
            1
            for row in transfer_rows
            if row.get("projection_used")
            and row.get("passes_role_based_gate")
            and float(row.get("target_mean_concept_lift_vs_role_raw", 0.0)) > 0.0
            and float(row.get("target_mean_concept_lift_vs_role_bag", 0.0)) > 0.0
            and float(row.get("target_mean_concept_lift_vs_surface_raw", 0.0)) > 0.0
        ),
        "fallback_only_signal_detected": any(
            float(row.get("target_family_score", 0.0)) > float(row.get("best_individual_role_baseline_raw", 0.0))
            and float(row.get("target_family_score", 0.0)) > float(row.get("best_surface_raw_baseline", 0.0))
            for row in fallback_score_rows
        ),
        "candidate_generator_counts_json": json.dumps(dict(generator_counts), sort_keys=True),
        "candidate_cap_hit": bool(cap_hit),
        "candidates_dropped_due_to_cap": int(dropped_due_to_cap),
        "requested_stable_items": len(context.source_roles),
        "stable_items_used": len(stable_items),
        **family_failure,
    }

    _write_parquet(shard_root / f"role_based_candidates__{heldout_family}.parquet", raw_candidate_rows)
    _write_parquet(shard_root / f"role_based_transfer_scores__{heldout_family}.parquet", transfer_rows)
    _write_parquet(shard_root / f"role_based_target_family_scores__{heldout_family}.parquet", target_family_rows)
    _write_parquet(shard_root / f"fallback_diagnostic_candidates__{heldout_family}.parquet", fallback_rows)
    _write_parquet(shard_root / f"fallback_diagnostic_scores__{heldout_family}.parquet", fallback_score_rows)
    _write_parquet(shard_root / f"family_summary__{heldout_family}.parquet", [summary])
    del context
    gc.collect()
    return {
        "heldout_family": heldout_family,
        "candidate_cap_hit": bool(cap_hit),
        "candidates_dropped_due_to_cap": int(dropped_due_to_cap),
        "generator_counts": dict(generator_counts),
        "transfer_row_count": len(transfer_rows),
        "failure_diagnostics": family_failure,
    }


def choose_v10e_worker_count(requested_workers: int, heldout_family_count: int, worker_memory_gib: int) -> int:
    if requested_workers <= 1 or heldout_family_count <= 1:
        return 1
    cpu_cap = os.cpu_count() or 1
    worker_cap = max(1, min(requested_workers, heldout_family_count, cpu_cap))
    available_memory = detect_available_memory_bytes()
    if available_memory is None:
        return worker_cap
    safe_budget = int(available_memory * 0.50)
    per_worker_bytes = max(1, worker_memory_gib) * 1024 * 1024 * 1024
    memory_cap = max(1, safe_budget // per_worker_bytes)
    return max(1, min(worker_cap, memory_cap))


def load_v10e_shards(shards_dir: Path, stem: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(shards_dir.glob(f"{stem}__*.parquet")):
        for row in pd.read_parquet(path).to_dict(orient="records"):
            rows.append(_normalize_shard_row(row))
    return rows


def build_stable_role_items(context: Any, *, max_role_items: int) -> list[dict[str, Any]]:
    items = []
    for role_id, role_info in sorted(context.source_roles.items()):
        records = [
            context.source_neighborhoods[family_id]
            for family_id in role_info.get("member_family_ids", ())
            if family_id in context.source_neighborhoods
        ]
        if not records:
            continue
        record = sorted(records, key=lambda item: str(item.family_id))[0]
        fingerprint = canonical_role_fingerprint(role_info["role_label_candidate"], record)
        items.append(
            {
                "source_fold": context.heldout_family,
                "family_id": str(record.family_id),
                "record": record,
                "role_id": role_id,
                "role_label": role_info["role_label_candidate"],
                "canonical_role_fingerprint_hash": fingerprint["canonical_role_fingerprint_hash"],
                "canonical_role_signature_json": fingerprint["canonical_role_signature_json"],
                "canonical_role_label_or_family": fingerprint["canonical_role_label_or_family"],
                "canonical_role_similarity_vector": fingerprint["canonical_role_similarity_vector"],
                "unknown_role_flag": False,
                "source_manifest_families": sorted({family for member in records for family in getattr(member, "game_family_ids", ())}) or ["source_clean_roles"],
                "source_game_support": len({game for member in records for game in getattr(member, "game_ids", ())}),
                "source_role_support": len(records),
            }
        )
    ordered = sorted(items, key=lambda item: _graph_sort_key(item["record"], item["role_id"]))
    return ordered[: max(1, max_role_items)]


def build_fallback_diagnostic_rows(context: Any, source_role_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for family_id, record in sorted(context.source_neighborhoods.items()):
        if family_id in source_role_map:
            continue
        fingerprint = canonical_role_fingerprint("unknown_role_candidate", record)
        row = build_raw_candidate_row(
            source_fold=context.heldout_family,
            heldout_family=context.heldout_family,
            manifest_family="fallback_diagnostic",
            local_candidate_id=f"fallback-{family_id}",
            generator_type="fallback_diagnostic_only",
            items=[
                {
                    "source_fold": context.heldout_family,
                    "family_id": family_id,
                    "record": record,
                    "role_id": family_id,
                    "role_label": "unknown_role_candidate",
                    "canonical_role_fingerprint_hash": fingerprint["canonical_role_fingerprint_hash"],
                    "canonical_role_signature_json": fingerprint["canonical_role_signature_json"],
                    "canonical_role_label_or_family": fingerprint["canonical_role_label_or_family"],
                    "canonical_role_similarity_vector": fingerprint["canonical_role_similarity_vector"],
                    "source_manifest_families": sorted(getattr(record, "game_family_ids", ()) or ("unknown_manifest_family",)),
                    "source_game_support": len(getattr(record, "game_ids", ()) or ()),
                    "source_role_support": 1,
                }
            ],
        )
        row["candidate_source"] = "fallback_diagnostic_only"
        row["role_count"] = 1
        row["source_role_support"] = 1
        row["source_game_support"] = len(row.get("source_games_present", ()))
        rows.append(row)
    return rows


def generate_role_based_candidates(
    context: Any,
    stable_items: list[dict[str, Any]],
    *,
    max_role_count: int,
    max_candidates: int,
) -> tuple[list[dict[str, Any]], Counter, bool, int]:
    if len(stable_items) < 2:
        return [], Counter(), False, 0
    generators = []
    generators.extend(
        [
            ("adjacent_graph_pair", _generate_adjacent_pairs(stable_items)),
            ("adjacent_graph_triple", _generate_adjacent_triples(stable_items)),
            ("graph_chain_window", _generate_graph_windows(stable_items)),
            ("graph_motif_bridge", _generate_motif_candidates(stable_items)),
            ("future_option_pair", _generate_future_option_pairs(stable_items)),
            ("future_option_triple", _generate_future_option_triples(stable_items)),
            ("high_contrast_role_pair", _generate_high_contrast_pairs(stable_items)),
            ("high_coherence_role_triple", _generate_high_coherence_triples(stable_items)),
        ]
    )
    if len(stable_items) <= 8:
        generators.append(("all_combination", _generate_all_combinations(stable_items, max_role_count=max_role_count)))

    seen_local_ids: set[tuple[str, ...]] = set()
    rows: list[dict[str, Any]] = []
    counts: Counter = Counter()
    dropped = 0
    for generator_type, combos in generators:
        for combo_index, combo in enumerate(combos, start=1):
            role_key = tuple(sorted(item["role_id"] for item in combo))
            if len(role_key) < 2 or role_key in seen_local_ids:
                continue
            if len(rows) >= max_candidates:
                dropped += 1
                continue
            seen_local_ids.add(role_key)
            row = build_role_based_candidate_row(
                source_fold=context.heldout_family,
                heldout_family=context.heldout_family,
                local_candidate_id=f"{generator_type}-{combo_index:04d}",
                items=list(combo),
                generator_type=generator_type,
            )
            rows.append(row)
            counts[generator_type] += 1
    cap_hit = dropped > 0
    return sorted(rows, key=lambda row: (row["concept_id"], row["local_candidate_id"])), counts, cap_hit, dropped


def _generate_all_combinations(stable_items: list[dict[str, Any]], *, max_role_count: int):
    for size in range(2, min(len(stable_items), max_role_count) + 1):
        for combo in combinations(stable_items, size):
            yield combo


def _generate_adjacent_pairs(stable_items: list[dict[str, Any]]):
    for index in range(len(stable_items) - 1):
        yield stable_items[index : index + 2]


def _generate_adjacent_triples(stable_items: list[dict[str, Any]]):
    for index in range(len(stable_items) - 2):
        yield stable_items[index : index + 3]


def _generate_graph_windows(stable_items: list[dict[str, Any]]):
    for size in range(2, min(5, len(stable_items)) + 1):
        for index in range(len(stable_items) - size + 1):
            yield stable_items[index : index + size]


def _generate_motif_candidates(stable_items: list[dict[str, Any]]):
    outgoing = sorted(stable_items, key=lambda item: float(getattr(item["record"], "directional_features", {}).get("successor_count", 0.0)), reverse=True)
    incoming = sorted(stable_items, key=lambda item: float(getattr(item["record"], "directional_features", {}).get("predecessor_count", 0.0)), reverse=True)
    middles = stable_items[:]
    for left in outgoing[:3]:
        for middle in middles[:3]:
            for right in incoming[:3]:
                role_ids = {left["role_id"], middle["role_id"], right["role_id"]}
                if len(role_ids) >= 2:
                    yield [left, middle, right]


def _generate_future_option_pairs(stable_items: list[dict[str, Any]]):
    ranked = sorted(
        stable_items,
        key=lambda item: abs(float(getattr(item["record"], "future_option_features", {}).get("enable_score", 0.0)))
        + abs(float(getattr(item["record"], "future_option_features", {}).get("block_score", 0.0))),
        reverse=True,
    )
    for combo in combinations(ranked[:6], 2):
        yield combo


def _generate_future_option_triples(stable_items: list[dict[str, Any]]):
    ranked = sorted(
        stable_items,
        key=lambda item: abs(float(getattr(item["record"], "future_option_features", {}).get("preserve_score", 0.0)))
        + abs(float(getattr(item["record"], "future_option_features", {}).get("enable_score", 0.0))),
        reverse=True,
    )
    for combo in combinations(ranked[:6], 3):
        yield combo


def _generate_high_contrast_pairs(stable_items: list[dict[str, Any]]):
    scored = []
    for left, right in combinations(stable_items[:10], 2):
        left_delta = float(getattr(left["record"], "future_option_features", {}).get("reachable_delta_mean", 0.0))
        right_delta = float(getattr(right["record"], "future_option_features", {}).get("reachable_delta_mean", 0.0))
        scored.append((abs(left_delta - right_delta), (left, right)))
    for _, combo in sorted(scored, key=lambda item: item[0], reverse=True)[:8]:
        yield combo


def _generate_high_coherence_triples(stable_items: list[dict[str, Any]]):
    scored = []
    for combo in combinations(stable_items[:10], 3):
        profiles = [future_option_behavior_features(item["record"]) for item in combo]
        coherence = float(np.mean([cosine_similarity(profiles[0], profile) for profile in profiles[1:]]))
        scored.append((coherence, combo))
    for _, combo in sorted(scored, key=lambda item: item[0], reverse=True)[:8]:
        yield combo


def build_role_based_candidate_row(
    *,
    source_fold: str,
    heldout_family: str,
    local_candidate_id: str,
    items: list[dict[str, Any]],
    generator_type: str = "stable_role_composition",
) -> dict[str, Any]:
    row = build_raw_candidate_row(
        source_fold=source_fold,
        heldout_family=heldout_family,
        manifest_family="role_based",
        local_candidate_id=local_candidate_id,
        generator_type=generator_type,
        items=items,
    )
    row["candidate_source"] = "stable_role"
    row["role_count"] = len(items)
    row["source_role_support"] = int(sum(int(item.get("source_role_support", 1)) for item in items))
    row["source_game_support"] = int(len({game for item in items for game in getattr(item["record"], "game_ids", ())}))
    row["source_manifest_families_present"] = sorted({family for item in items for family in item.get("source_manifest_families", [])})
    return row


def evaluate_role_based_projection_by_family(
    concept: dict[str, Any],
    context: Any,
    target_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_target_rows = []
    target_rows_by_family = defaultdict(list)
    for row in target_rows:
        target_rows_by_family[str(row.get("target_family_id", ""))].append(row)
    target_records = _target_records(context)
    for family in sorted(context.target_families, key=lambda item: item.family_id):
        target_record = target_records.get(family.family_id)
        if target_record is None:
            continue
        per_target_rows.append(
            score_role_based_concept_against_target_family(
                concept,
                context,
                family.family_id,
                target_record,
                target_rows_by_family.get(family.family_id, []),
            )
        )
    if not per_target_rows:
        return (
            {
                "concept_id": concept["concept_id"],
                "projection_used": False,
                "failure_reason": "missing_target_rows",
                "target_family_count": 0,
                "target_concept_prediction_score": 0.0,
                "target_best_match_score": 0.0,
                "target_top3_mean_score": 0.0,
                "target_mean_score": 0.0,
                "best_target_family_id": "",
                "local_match_lost_by_averaging": False,
                "target_mean_concept_lift_vs_role_raw": 0.0,
                "target_mean_concept_lift_vs_role_bag": 0.0,
                "target_mean_concept_lift_vs_surface_raw": 0.0,
                "target_mean_concept_lift_vs_graph_no_label": 0.0,
                "target_mean_concept_lift_vs_m2": 0.0,
                "target_mean_future_option_prediction_lift": 0.0,
                "mean_compression_gain": 0.0,
                "explained_m2_family_count": 0,
                "positive_lift_family_count": 0,
                "passes_role_based_gate": False,
            },
            [],
        )
    ordered_rows = sorted(per_target_rows, key=lambda row: row["target_family_score"], reverse=True)
    ordered_scores = [row["target_family_score"] for row in ordered_rows]
    best_score = float(ordered_scores[0])
    top3_mean = float(np.mean(ordered_scores[:3]))
    mean_score = float(np.mean(ordered_scores))
    projection = {
        "concept_id": concept["concept_id"],
        "projection_used": True,
        "failure_reason": "",
        "target_family_count": len(per_target_rows),
        "target_concept_prediction_score": mean_score,
        "target_best_match_score": best_score,
        "target_top3_mean_score": top3_mean,
        "target_mean_score": mean_score,
        "best_target_family_id": str(ordered_rows[0]["target_family_id"]),
        "local_match_lost_by_averaging": bool(best_score - mean_score > 0.05),
        "target_mean_concept_lift_vs_role_raw": float(np.mean([row["target_family_score"] - row["best_individual_role_baseline_raw"] for row in per_target_rows])),
        "target_mean_concept_lift_vs_role_bag": float(np.mean([row["target_family_score"] - row["unordered_role_bag_baseline"] for row in per_target_rows])),
        "target_mean_concept_lift_vs_surface_raw": float(np.mean([row["target_family_score"] - row["best_surface_raw_baseline"] for row in per_target_rows])),
        "target_mean_concept_lift_vs_graph_no_label": float(np.mean([row["target_family_score"] - row["graph_no_label_baseline"] for row in per_target_rows])),
        "target_mean_concept_lift_vs_m2": float(np.mean([row["target_family_score"] - row["best_raw_m2_baseline"] for row in per_target_rows])),
        "target_mean_future_option_prediction_lift": float(np.mean([row["future_option_prediction_lift"] for row in per_target_rows])),
        "mean_compression_gain": float(np.mean([row["compression_gain"] for row in per_target_rows])),
        "explained_m2_family_count": sum(1 for row in per_target_rows if row["compression_gain"] > 0.0 and row["target_family_score"] > row["best_raw_m2_baseline"]),
        "positive_lift_family_count": sum(
            1
            for row in per_target_rows
            if row["target_family_score"] > row["best_individual_role_baseline_raw"]
            and row["target_family_score"] > row["unordered_role_bag_baseline"]
            and row["target_family_score"] > row["best_surface_raw_baseline"]
            and row["compression_gain"] > 0.0
            and row["future_option_prediction_lift"] > 0.0
        ),
        "passes_role_based_gate": False,
    }
    projection["passes_role_based_gate"] = is_transferable_role_based_concept({**concept, **projection, "candidate_source": concept.get("candidate_source", "stable_role")})
    return projection, per_target_rows


def score_role_based_concept_against_target_family(
    concept: dict[str, Any],
    context: Any,
    target_family_id: str,
    target_record: Any,
    target_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    base = score_concept_against_target_family(concept, _base_scoring_context(context), target_family_id, target_record, target_rows)
    target_future = future_option_behavior_features(target_record)
    target_graph = graph_position_features(target_record)
    target_predsucc = predecessor_successor_profile(target_record)
    target_surface = appearance_features(target_record)
    concept_future_similarity = cosine_similarity(concept["future_option_delta_profile"], target_future)
    concept_graph_similarity = float(cosine_similarity(concept["graph_position_profile"], target_graph))
    concept_predsucc_similarity = float(cosine_similarity(concept["predecessor_successor_profile"], target_predsucc))
    concept_surface_similarity = float(cosine_similarity(appearance_features(target_record), target_surface))
    future_role_score = best_future_option_role_score(context, target_record)
    separate_role_score = separate_role_explanation_score(concept, context, target_record)
    raw_m2_matches = best_raw_m2_component_matches(context, target_record)
    graph_no_label_matches = best_graph_no_label_component_matches(context, target_record)
    return {
        **base,
        "unordered_role_bag_baseline": unordered_role_bag_score(concept, context, target_record),
        "graph_no_label_baseline": best_graph_no_label_score(context, target_record),
        "future_option_prediction_lift": float(concept_future_similarity - future_role_score),
        "best_future_option_role_score": float(future_role_score),
        "target_future_similarity": float(concept_future_similarity),
        "target_graph_similarity": concept_graph_similarity,
        "target_predsucc_similarity": concept_predsucc_similarity,
        "target_surface_similarity": concept_surface_similarity,
        "separate_role_explanation_score": float(separate_role_score),
        "composed_concept_explanation_score": float(base["target_family_score"]),
        "compression_gain": float(base["target_family_score"] - separate_role_score),
        "lift_vs_best_role": float(base["target_family_score"] - base["best_individual_role_baseline_raw"]),
        "lift_vs_unordered_bag": float(base["target_family_score"] - unordered_role_bag_score(concept, context, target_record)),
        "lift_vs_surface_effect": float(base["target_family_score"] - base["best_surface_raw_baseline"]),
        "lift_vs_raw_m2": float(base["target_family_score"] - base["best_raw_m2_baseline"]),
        "lift_vs_graph_no_label": float(base["target_family_score"] - best_graph_no_label_score(context, target_record)),
        "best_raw_m2_future_similarity": raw_m2_matches["future_similarity"],
        "best_raw_m2_graph_similarity": raw_m2_matches["graph_similarity"],
        "best_raw_m2_predsucc_similarity": raw_m2_matches["predsucc_similarity"],
        "best_raw_m2_surface_similarity": raw_m2_matches["surface_similarity"],
        "best_graph_no_label_graph_similarity": graph_no_label_matches["graph_similarity"],
        "target_role_overlap_diagnostic": float(
            len(set(concept.get("fold_local_role_ids", ())) & {str(row.get("assigned_role_id", "")) for row in target_rows if row.get("assigned_role_id")})
            / max(1, len(set(concept.get("fold_local_role_ids", ()))))
        ),
    }


def unordered_role_bag_score(concept: dict[str, Any], context: Any, target_record: Any) -> float:
    target_future = future_option_behavior_features(target_record)
    target_graph = graph_position_features(target_record)
    role_futures = []
    role_graphs = []
    for role_id in concept.get("fold_local_role_ids", ()):
        role = context.source_roles.get(role_id)
        if role is None:
            continue
        role_futures.append(_subset_prefixed(role.get("all_features", {}), "future:"))
        role_graphs.append(_subset_prefixed(role.get("all_features", {}), "directional:"))
    if not role_futures:
        return 0.0
    return float(0.6 * cosine_similarity(target_future, mean_vector(role_futures)) + 0.4 * cosine_similarity(target_graph, mean_vector(role_graphs) if role_graphs else {}))


def best_graph_no_label_score(context: Any, target_record: Any) -> float:
    target_graph = graph_position_features(target_record)
    best = 0.0
    for record in context.source_neighborhoods.values():
        best = max(best, float(cosine_similarity(target_graph, graph_position_features(record))))
    return best


def best_graph_no_label_component_matches(context: Any, target_record: Any) -> dict[str, float]:
    target_graph = graph_position_features(target_record)
    best = 0.0
    for record in context.source_neighborhoods.values():
        best = max(best, float(cosine_similarity(target_graph, graph_position_features(record))))
    return {"graph_similarity": float(best)}


def best_raw_m2_component_matches(context: Any, target_record: Any) -> dict[str, float]:
    target_future = future_option_behavior_features(target_record)
    target_graph = graph_position_features(target_record)
    target_predsucc = predecessor_successor_profile(target_record)
    target_surface = appearance_features(target_record)
    best = {
        "combined_score": 0.0,
        "future_similarity": 0.0,
        "graph_similarity": 0.0,
        "predsucc_similarity": 0.0,
        "surface_similarity": 0.0,
    }
    for record in context.source_neighborhoods.values():
        future_similarity = float(cosine_similarity(target_future, future_option_behavior_features(record)))
        graph_similarity = float(cosine_similarity(target_graph, graph_position_features(record)))
        predsucc_similarity = float(cosine_similarity(target_predsucc, predecessor_successor_profile(record)))
        surface_similarity = float(cosine_similarity(target_surface, appearance_features(record)))
        combined = 0.6 * future_similarity + 0.4 * graph_similarity
        if combined > best["combined_score"]:
            best = {
                "combined_score": float(combined),
                "future_similarity": future_similarity,
                "graph_similarity": graph_similarity,
                "predsucc_similarity": predsucc_similarity,
                "surface_similarity": surface_similarity,
            }
    return best


def best_future_option_role_score(context: Any, target_record: Any) -> float:
    target_future = future_option_behavior_features(target_record)
    best = 0.0
    for role in context.source_roles.values():
        best = max(best, float(cosine_similarity(target_future, _subset_prefixed(role.get("all_features", {}), "future:"))))
    return best


def separate_role_explanation_score(concept: dict[str, Any], context: Any, target_record: Any) -> float:
    target_future = future_option_behavior_features(target_record)
    target_graph = graph_position_features(target_record)
    scores = []
    for role_id in concept.get("fold_local_role_ids", ()):
        role = context.source_roles.get(role_id)
        if role is None:
            continue
        scores.append(
            0.6 * cosine_similarity(target_future, _subset_prefixed(role.get("all_features", {}), "future:"))
            + 0.4 * cosine_similarity(target_graph, _subset_prefixed(role.get("all_features", {}), "directional:"))
        )
    return float(np.mean(scores)) if scores else 0.0


def apply_role_based_gates(concept_rows: list[dict[str, Any]], transfer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in transfer_rows:
        if row.get("projection_used"):
            grouped[row["concept_id"]].append(row)
    output = []
    for row in concept_rows:
        projections = grouped.get(row["concept_id"], [])
        updated = dict(row)
        updated["candidate_source"] = updated.get("candidate_source", "stable_role")
        updated["target_mean_concept_lift_vs_role_bag"] = mean_metric(projections, "target_mean_concept_lift_vs_role_bag")
        updated["target_mean_concept_lift_vs_graph_no_label"] = mean_metric(projections, "target_mean_concept_lift_vs_graph_no_label")
        updated["target_mean_future_option_prediction_lift"] = mean_metric(projections, "target_mean_future_option_prediction_lift")
        updated["mean_compression_gain"] = mean_metric(projections, "mean_compression_gain")
        updated["explained_m2_family_count"] = int(sum(int(item.get("explained_m2_family_count", 0)) for item in projections))
        updated["passes_role_based_gate"] = is_transferable_role_based_concept(updated)
        output.append(updated)
    return output


def is_transferable_role_based_concept(row: dict[str, Any]) -> bool:
    if row.get("candidate_source", "stable_role") != "stable_role":
        return False
    return (
        len(row.get("canonical_role_fingerprint_hashes", [])) >= 2
        and float(row.get("target_mean_concept_lift_vs_role_raw", 0.0)) > 0.0
        and float(row.get("target_mean_concept_lift_vs_role_bag", 0.0)) > 0.0
        and float(row.get("target_mean_concept_lift_vs_surface_raw", 0.0)) > 0.0
        and float(row.get("mean_compression_gain", 0.0)) > 0.0
        and float(row.get("target_mean_future_option_prediction_lift", 0.0)) > 0.0
        and int(row.get("positive_lift_family_count", 0)) >= 1
        and int(row.get("explained_m2_family_count", 0)) >= 2
    )


def score_fallback_rows(context: Any, fallback_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_rows_by_family = defaultdict(list)
    for row in target_rows:
        target_rows_by_family[str(row.get("target_family_id", ""))].append(row)
    rows = []
    for candidate in fallback_rows:
        for family in sorted(context.target_families, key=lambda item: item.family_id):
            target_record = _target_records(context).get(family.family_id)
            if target_record is None:
                continue
            rows.append(
                {
                    "heldout_family": context.heldout_family,
                    "candidate_source": "fallback_diagnostic_only",
                    "concept_id": candidate["concept_id"],
                    **score_role_based_concept_against_target_family(candidate, context, family.family_id, target_record, target_rows_by_family.get(family.family_id, [])),
                }
            )
    return rows


def build_rejected_candidate_rows(concept_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in concept_rows:
        reasons = []
        if row.get("candidate_source") != "stable_role":
            reasons.append("diagnostic_only_candidate_source")
        if float(row.get("target_mean_concept_lift_vs_role_raw", 0.0)) <= 0.0:
            reasons.append("no_lift_vs_best_individual_role")
        if float(row.get("target_mean_concept_lift_vs_role_bag", 0.0)) <= 0.0:
            reasons.append("no_lift_vs_unordered_role_bag")
        if float(row.get("target_mean_concept_lift_vs_surface_raw", 0.0)) <= 0.0:
            reasons.append("no_lift_vs_surface_effect_raw")
        if float(row.get("mean_compression_gain", 0.0)) <= 0.0:
            reasons.append("no_positive_compression_gain")
        if float(row.get("target_mean_future_option_prediction_lift", 0.0)) <= 0.0:
            reasons.append("no_future_option_prediction_lift")
        if int(row.get("explained_m2_family_count", 0)) < 2:
            reasons.append("insufficient_explained_m2_families")
        if int(row.get("positive_lift_family_count", 0)) < 1:
            reasons.append("insufficient_positive_lift_families")
        output.append(
            {
                "concept_id": row["concept_id"],
                "candidate_source": row.get("candidate_source", "stable_role"),
                "passes_role_based_gate": bool(row.get("passes_role_based_gate")),
                "rejection_reasons": reasons,
            }
        )
    return output


def _build_v10e_family_failure_row(
    *,
    heldout_family: str,
    context: Any,
    source_role_map: dict[str, dict[str, Any]],
    stable_items: list[dict[str, Any]],
    raw_candidate_rows: list[dict[str, Any]],
    projection_rows: list[dict[str, Any]],
    target_family_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    fallback_rows: list[dict[str, Any]],
    min_role_count: int,
) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    if not context.source_neighborhoods:
        reason_counts["no_source_neighborhoods"] += 1
    if not context.source_roles:
        reason_counts["no_source_roles"] += 1
    if not source_role_map:
        reason_counts["source_role_map_family_id_mismatch"] += 1
    if len(stable_items) < int(min_role_count):
        reason_counts["insufficient_stable_role_items"] += 1
    if len(stable_items) >= 2 and not raw_candidate_rows:
        reason_counts["subcomposition_generation_failure"] += 1
    projection_used_count = sum(1 for row in projection_rows if row.get("projection_used"))
    if raw_candidate_rows and projection_used_count == 0:
        reason_counts["no_target_projection"] += len(raw_candidate_rows)
    reason_counts.update(
        merge_reason_counts(
            count_by_reason(rejected_rows, "rejection_reasons"),
            count_by_reason(rejected_rows, "rejection_reason"),
            count_by_reason(rejected_rows, "reason"),
        )
    )
    projected_target_families = {str(row.get("target_family_id", "")) for row in target_family_rows if row.get("target_family_id")}
    stable_candidate_count = len(projection_rows)
    transferable_candidate_count = sum(1 for row in projection_rows if row.get("passes_role_based_gate"))
    return {
        "heldout_family_id": heldout_family,
        "family_loaded": True,
        "source_neighborhoods_available": bool(context.source_neighborhoods),
        "source_roles_available": bool(context.source_roles),
        "source_role_overlap_ok": bool(source_role_map),
        "stable_role_items_count": len(stable_items),
        "raw_candidates_count": len(raw_candidate_rows),
        "projected_target_families_count": len(projected_target_families),
        "projection_used_count": projection_used_count,
        "rejected_candidate_count": len(rejected_rows),
        "stable_candidate_count": stable_candidate_count,
        "transferable_candidate_count": transferable_candidate_count,
        "fallback_candidate_count": len(fallback_rows),
        "mixed_candidate_count": 0,
        "unknown_manifest_candidate_count": 0,
        "failure_reason_counts": ensure_failure_buckets(reason_counts),
    }


def _build_v10e_failure_diagnostics(family_failure_rows: list[dict[str, Any]], *, merge_diag: dict[str, int]) -> dict[str, Any]:
    total_reason_counts = merge_reason_counts(*(row.get("failure_reason_counts", {}) for row in family_failure_rows))
    total_reason_counts = merge_reason_counts(
        total_reason_counts,
        {
            "insufficient_games": int(merge_diag.get("rejected_due_to_min_games", 0)),
            "insufficient_manifest_families": int(merge_diag.get("rejected_due_to_min_families", 0)),
        },
    )
    attrition_totals = {
        "families_loaded": sum(1 for row in family_failure_rows if row.get("family_loaded")),
        "families_with_source_neighborhoods": sum(1 for row in family_failure_rows if row.get("source_neighborhoods_available")),
        "families_with_source_roles": sum(1 for row in family_failure_rows if row.get("source_roles_available")),
        "families_with_source_role_overlap": sum(1 for row in family_failure_rows if row.get("source_role_overlap_ok")),
        "stable_role_items_total": sum(int(row.get("stable_role_items_count", 0)) for row in family_failure_rows),
        "raw_candidates_total": sum(int(row.get("raw_candidates_count", 0)) for row in family_failure_rows),
        "projected_target_families_total": sum(int(row.get("projected_target_families_count", 0)) for row in family_failure_rows),
        "projection_used_total": sum(int(row.get("projection_used_count", 0)) for row in family_failure_rows),
        "rejected_candidate_total": sum(int(row.get("rejected_candidate_count", 0)) for row in family_failure_rows),
        "stable_candidate_total": sum(int(row.get("stable_candidate_count", 0)) for row in family_failure_rows),
        "transferable_candidate_total": sum(int(row.get("transferable_candidate_count", 0)) for row in family_failure_rows),
        "fallback_candidate_total": sum(int(row.get("fallback_candidate_count", 0)) for row in family_failure_rows),
        "mixed_candidate_total": sum(int(row.get("mixed_candidate_count", 0)) for row in family_failure_rows),
        "unknown_manifest_candidate_total": sum(int(row.get("unknown_manifest_candidate_count", 0)) for row in family_failure_rows),
    }
    return {
        "per_family": family_failure_rows,
        "total_failure_reason_counts": ensure_failure_buckets(total_reason_counts),
        "attrition_totals": attrition_totals,
    }


def build_concept_identity_rows(raw_candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "concept_id": row["concept_id"],
            "concept_signature_json": row["concept_signature_json"],
            "local_candidate_id": row["local_candidate_id"],
            "generator_type": row.get("generator_type", ""),
            "candidate_source": row.get("candidate_source", ""),
            "source_fold": row["source_fold"],
            "heldout_family": row["heldout_family"],
            "canonical_role_fingerprint_hashes": row["canonical_role_fingerprint_hashes"],
        }
        for row in raw_candidate_rows
    ]


def build_compression_rows(concept_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"concept_id": row["concept_id"], "candidate_source": row.get("candidate_source", "stable_role"), "mean_compression_gain": float(row.get("mean_compression_gain", 0.0)), "explained_m2_family_count": int(row.get("explained_m2_family_count", 0)), "positive_lift_family_count": int(row.get("positive_lift_family_count", 0))} for row in concept_rows]


def build_future_option_rows(target_family_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"heldout_family": row["heldout_family"], "concept_id": row["concept_id"], "candidate_source": row.get("candidate_source", "stable_role"), "target_family_id": row["target_family_id"], "target_future_similarity": float(row.get("target_future_similarity", 0.0)), "best_future_option_role_score": float(row.get("best_future_option_role_score", 0.0)), "future_option_prediction_lift": float(row.get("future_option_prediction_lift", 0.0))} for row in target_family_rows]


def build_baseline_rows(target_family_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"heldout_family": row["heldout_family"], "concept_id": row["concept_id"], "candidate_source": row.get("candidate_source", "stable_role"), "target_family_id": row["target_family_id"], "target_family_score": float(row.get("target_family_score", 0.0)), "best_individual_role_baseline_raw": float(row.get("best_individual_role_baseline_raw", 0.0)), "unordered_role_bag_baseline": float(row.get("unordered_role_bag_baseline", 0.0)), "best_raw_m2_baseline": float(row.get("best_raw_m2_baseline", 0.0)), "best_surface_raw_baseline": float(row.get("best_surface_raw_baseline", 0.0)), "graph_no_label_baseline": float(row.get("graph_no_label_baseline", 0.0)), "compression_gain": float(row.get("compression_gain", 0.0))} for row in target_family_rows]


def build_candidate_generator_rows(by_family_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in by_family_rows:
        counts = json.loads(row.get("candidate_generator_counts_json", "{}"))
        for generator_type, count in sorted(counts.items()):
            rows.append({"heldout_family": row["heldout_family"], "generator_type": generator_type, "candidate_count": int(count)})
    return rows


def build_multiprocessing_rows(family_results: list[dict[str, Any]], *, requested_workers: int, effective_workers: int, multiprocessing_used: bool) -> list[dict[str, Any]]:
    return [
        {
            "heldout_family": row["heldout_family"],
            "requested_workers": requested_workers,
            "effective_workers": effective_workers,
            "multiprocessing_used": multiprocessing_used,
            "transfer_row_count": int(row.get("transfer_row_count", 0)),
        }
        for row in family_results
    ]


def build_candidate_cap_rows(by_family_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"heldout_family": row["heldout_family"], "candidate_cap_hit": bool(row.get("candidate_cap_hit", False)), "candidates_dropped_due_to_cap": int(row.get("candidates_dropped_due_to_cap", 0))} for row in by_family_rows]


def build_failure_decomposition_rows(concept_rows: list[dict[str, Any]], target_family_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    concept_map = {row["concept_id"]: row for row in concept_rows}
    rows = []
    for row in target_family_rows:
        concept = concept_map.get(row["concept_id"], {})
        family_coverage_distance = min(
            float(concept.get("positive_lift_family_count", 0)) - 1.0,
            float(concept.get("explained_m2_family_count", 0)) - 2.0,
        )
        failed = []
        if float(row.get("lift_vs_surface_effect", row.get("target_family_score", 0.0) - row.get("best_surface_raw_baseline", 0.0))) <= 0.0:
            failed.append("surface_effect")
        if float(row.get("future_option_prediction_lift", 0.0)) <= 0.0:
            failed.append("future_option")
        if float(row.get("compression_gain", 0.0)) <= 0.0:
            failed.append("compression")
        if float(row.get("lift_vs_raw_m2", row.get("target_family_score", 0.0) - row.get("best_raw_m2_baseline", 0.0))) <= 0.0:
            failed.append("raw_m2")
        if float(row.get("lift_vs_graph_no_label", row.get("target_family_score", 0.0) - row.get("graph_no_label_baseline", 0.0))) <= 0.0:
            failed.append("graph_no_label")
        if family_coverage_distance <= 0.0:
            failed.append("family_coverage")
        if float(row.get("lift_vs_best_role", row.get("target_family_score", 0.0) - row.get("best_individual_role_baseline_raw", 0.0))) <= 0.0:
            failed.append("best_role")
        if float(row.get("lift_vs_unordered_bag", row.get("target_family_score", 0.0) - row.get("unordered_role_bag_baseline", 0.0))) <= 0.0:
            failed.append("unordered_bag")
        distance_map = {
            "surface_effect": float(row.get("lift_vs_surface_effect", row.get("target_family_score", 0.0) - row.get("best_surface_raw_baseline", 0.0))),
            "future_option": float(row.get("future_option_prediction_lift", 0.0)),
            "compression": float(row.get("compression_gain", 0.0)),
            "raw_m2": float(row.get("lift_vs_raw_m2", row.get("target_family_score", 0.0) - row.get("best_raw_m2_baseline", 0.0))),
            "graph_no_label": float(row.get("lift_vs_graph_no_label", row.get("target_family_score", 0.0) - row.get("graph_no_label_baseline", 0.0))),
            "family_coverage": family_coverage_distance,
        }
        closest_failed_gate = max(((gate, dist) for gate, dist in distance_map.items() if dist <= 0.0), key=lambda item: item[1], default=("none", 0.0))[0]
        rows.append(
            {
                "concept_id": row["concept_id"],
                "generator_type": concept.get("generator_type", ""),
                "motif_type": concept.get("motif_type", ""),
                "heldout_family": row["heldout_family"],
                "target_family_id": row["target_family_id"],
                "target_family_score": float(row.get("target_family_score", 0.0)),
                "best_individual_role_baseline_raw": float(row.get("best_individual_role_baseline_raw", 0.0)),
                "unordered_role_bag_baseline": float(row.get("unordered_role_bag_baseline", 0.0)),
                "best_surface_raw_baseline": float(row.get("best_surface_raw_baseline", 0.0)),
                "best_raw_m2_baseline": float(row.get("best_raw_m2_baseline", 0.0)),
                "graph_no_label_baseline": float(row.get("graph_no_label_baseline", 0.0)),
                "future_option_prediction_lift": float(row.get("future_option_prediction_lift", 0.0)),
                "compression_gain": float(row.get("compression_gain", 0.0)),
                "lift_vs_best_role": float(row.get("lift_vs_best_role", row.get("target_family_score", 0.0) - row.get("best_individual_role_baseline_raw", 0.0))),
                "lift_vs_unordered_bag": float(row.get("lift_vs_unordered_bag", row.get("target_family_score", 0.0) - row.get("unordered_role_bag_baseline", 0.0))),
                "lift_vs_surface_effect": float(row.get("lift_vs_surface_effect", row.get("target_family_score", 0.0) - row.get("best_surface_raw_baseline", 0.0))),
                "lift_vs_raw_m2": float(row.get("lift_vs_raw_m2", row.get("target_family_score", 0.0) - row.get("best_raw_m2_baseline", 0.0))),
                "lift_vs_graph_no_label": float(row.get("lift_vs_graph_no_label", row.get("target_family_score", 0.0) - row.get("graph_no_label_baseline", 0.0))),
                "failed_gates": failed,
                "closest_failed_gate": closest_failed_gate,
                "distance_to_surface_gate": distance_map["surface_effect"],
                "distance_to_future_option_gate": distance_map["future_option"],
                "distance_to_raw_m2_gate": distance_map["raw_m2"],
                "distance_to_graph_no_label_gate": distance_map["graph_no_label"],
                "distance_to_family_coverage_gate": distance_map["family_coverage"],
                "surface_effect_failed": "surface_effect" in failed,
                "future_option_failed": "future_option" in failed,
                "compression_failed": "compression" in failed,
                "raw_m2_failed": "raw_m2" in failed,
                "graph_no_label_failed": "graph_no_label" in failed,
                "family_coverage_failed": "family_coverage" in failed,
            }
        )
    return rows


def build_generator_failure_summary_rows(failure_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not failure_rows:
        return []
    df = pd.DataFrame(failure_rows)
    rows = []
    for generator_type, group in df.groupby("generator_type", dropna=False):
        best_idx = group["lift_vs_best_role"].astype(float).idxmax()
        best_row = group.loc[best_idx]
        rows.append(
            {
                "generator_type": generator_type,
                "candidate_count": int(group["concept_id"].nunique()),
                "target_family_score_count": int(len(group)),
                "mean_lift_vs_best_role": float(group["lift_vs_best_role"].mean()),
                "max_lift_vs_best_role": float(group["lift_vs_best_role"].max()),
                "mean_lift_vs_unordered_bag": float(group["lift_vs_unordered_bag"].mean()),
                "max_lift_vs_unordered_bag": float(group["lift_vs_unordered_bag"].max()),
                "mean_lift_vs_surface_effect": float(group["lift_vs_surface_effect"].mean()),
                "max_lift_vs_surface_effect": float(group["lift_vs_surface_effect"].max()),
                "mean_lift_vs_future_option": float(group["future_option_prediction_lift"].mean()),
                "max_lift_vs_future_option": float(group["future_option_prediction_lift"].max()),
                "mean_lift_vs_raw_m2": float(group["lift_vs_raw_m2"].mean()),
                "max_lift_vs_raw_m2": float(group["lift_vs_raw_m2"].max()),
                "mean_lift_vs_graph_no_label": float(group["lift_vs_graph_no_label"].mean()),
                "max_lift_vs_graph_no_label": float(group["lift_vs_graph_no_label"].max()),
                "mean_family_coverage": float(np.mean([1.0 if not item else 0.0 for item in group["family_coverage_failed"]])),
                "max_family_coverage": float(np.max([1.0 if not item else 0.0 for item in group["family_coverage_failed"]])),
                "best_candidate_id": str(best_row["concept_id"]),
                "best_candidate_score": float(best_row["target_family_score"]),
                "closest_failure_gate": str(best_row["closest_failed_gate"]),
                "dominant_failure_pattern": _dominant_pattern_from_series(group["failed_gates"]),
            }
        )
    return sorted(rows, key=lambda row: (-row["max_lift_vs_best_role"], row["generator_type"]))


def build_projection_audit_rows(concept_rows: list[dict[str, Any]], target_family_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    concept_map = {row["concept_id"]: row for row in concept_rows}
    grouped = defaultdict(list)
    for row in target_family_rows:
        grouped[row["concept_id"]].append(row)
    rows = []
    for concept_id, items in grouped.items():
        ordered = sorted(items, key=lambda row: row["target_family_score"], reverse=True)
        scores = [float(row["target_family_score"]) for row in ordered]
        concept = concept_map.get(concept_id, {})
        rows.append(
            {
                "concept_id": concept_id,
                "generator_type": concept.get("generator_type", ""),
                "heldout_family": concept.get("heldout_family", ""),
                "target_best_match_score": float(scores[0]),
                "target_top3_mean_score": float(np.mean(scores[:3])),
                "target_mean_score": float(np.mean(scores)),
                "best_target_family_id": str(ordered[0]["target_family_id"]),
                "local_match_lost_by_averaging": float(scores[0] - float(np.mean(scores))),
            }
        )
    return rows


def build_baseline_dominance_rows(failure_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in failure_rows:
        distances = {
            "raw_m2": float(row["lift_vs_raw_m2"]),
            "graph_no_label": float(row["lift_vs_graph_no_label"]),
            "surface_effect": float(row["lift_vs_surface_effect"]),
            "future_option": float(row["future_option_prediction_lift"]),
            "best_role": float(row["lift_vs_best_role"]),
        }
        dominant = min(distances.items(), key=lambda item: item[1])[0]
        rows.append(
            {
                "concept_id": row["concept_id"],
                "target_family_id": row["target_family_id"],
                "raw_m2_dominates": dominant == "raw_m2",
                "graph_no_label_dominates": dominant == "graph_no_label",
                "surface_effect_dominates": dominant == "surface_effect",
                "future_option_dominates": dominant == "future_option",
                "best_role_dominates": dominant == "best_role",
                "dominant_baseline": dominant,
            }
        )
    return rows


def build_representation_loss_rows(target_family_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in target_family_rows:
        rows.append(
            {
                "concept_id": row["concept_id"],
                "heldout_family": row["heldout_family"],
                "target_family_id": row["target_family_id"],
                "target_future_similarity": float(row.get("target_future_similarity", 0.0)),
                "target_graph_similarity": float(row.get("target_graph_similarity", 0.0)),
                "target_predsucc_similarity": float(row.get("target_predsucc_similarity", 0.0)),
                "best_raw_m2_future_similarity": float(row.get("best_raw_m2_future_similarity", 0.0)),
                "best_raw_m2_graph_similarity": float(row.get("best_raw_m2_graph_similarity", 0.0)),
                "best_raw_m2_predsucc_similarity": float(row.get("best_raw_m2_predsucc_similarity", 0.0)),
                "best_raw_m2_surface_similarity": float(row.get("best_raw_m2_surface_similarity", 0.0)),
                "best_graph_no_label_graph_similarity": float(row.get("best_graph_no_label_graph_similarity", 0.0)),
                "loses_future_option_detail": float(row.get("target_future_similarity", 0.0)) < float(row.get("best_raw_m2_future_similarity", 0.0)),
                "loses_graph_position_detail": float(row.get("target_graph_similarity", 0.0)) < max(float(row.get("best_raw_m2_graph_similarity", 0.0)), float(row.get("best_graph_no_label_graph_similarity", 0.0))),
                "loses_surface_effect_detail": float(row.get("lift_vs_surface_effect", row.get("target_family_score", 0.0) - row.get("best_surface_raw_baseline", 0.0))) <= 0.0,
                "loses_predecessor_successor_detail": float(row.get("target_predsucc_similarity", 0.0)) < float(row.get("best_raw_m2_predsucc_similarity", 0.0)),
            }
        )
    return rows


def build_closest_candidate_rows(concept_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in concept_rows:
        gap_vector = {
            "best_role": min(0.0, float(row.get("target_mean_concept_lift_vs_role_raw", 0.0))),
            "unordered_bag": min(0.0, float(row.get("target_mean_concept_lift_vs_role_bag", 0.0))),
            "surface_effect": min(0.0, float(row.get("target_mean_concept_lift_vs_surface_raw", 0.0))),
            "future_option": min(0.0, float(row.get("target_mean_future_option_prediction_lift", 0.0))),
            "compression": min(0.0, float(row.get("mean_compression_gain", 0.0))),
            "family_coverage": min(
                0.0,
                min(float(row.get("positive_lift_family_count", 0)) - 1.0, float(row.get("explained_m2_family_count", 0)) - 2.0),
            ),
        }
        rows.append(
            {
                "concept_id": row["concept_id"],
                "generator_type": row.get("generator_type", ""),
                "candidate_source": row.get("candidate_source", ""),
                "target_mean_concept_lift_vs_role_raw": float(row.get("target_mean_concept_lift_vs_role_raw", 0.0)),
                "target_mean_concept_lift_vs_role_bag": float(row.get("target_mean_concept_lift_vs_role_bag", 0.0)),
                "target_mean_concept_lift_vs_surface_raw": float(row.get("target_mean_concept_lift_vs_surface_raw", 0.0)),
                "target_mean_future_option_prediction_lift": float(row.get("target_mean_future_option_prediction_lift", 0.0)),
                "mean_compression_gain": float(row.get("mean_compression_gain", 0.0)),
                "positive_lift_family_count": int(row.get("positive_lift_family_count", 0)),
                "explained_m2_family_count": int(row.get("explained_m2_family_count", 0)),
                "closest_failure_gate": max(gap_vector.items(), key=lambda item: item[1])[0],
                "transfer_gap_score": float(sum(abs(value) for value in gap_vector.values())),
            }
        )
    return sorted(rows, key=lambda row: (row["transfer_gap_score"], -row["target_mean_concept_lift_vs_role_raw"]))[:50]


def build_v10e_report(
    *,
    config: M4RoleConceptsV10eConfig,
    transfer_report: dict[str, Any],
    concept_rows: list[dict[str, Any]],
    transferable_rows: list[dict[str, Any]],
    transfer_rows: list[dict[str, Any]],
    fallback_rows: list[dict[str, Any]],
    fallback_score_rows: list[dict[str, Any]],
    by_family_rows: list[dict[str, Any]],
    merge_diag: dict[str, int],
    fallback_only_signal_detected: bool,
    requested_workers: int,
    effective_workers: int,
    multiprocessing_used: bool,
    generator_rows: list[dict[str, Any]],
    cap_rows: list[dict[str, Any]],
    failure_decomposition_rows: list[dict[str, Any]],
    generator_failure_rows: list[dict[str, Any]],
    projection_audit_rows: list[dict[str, Any]],
    baseline_dominance_rows: list[dict[str, Any]],
    closest_candidate_rows: list[dict[str, Any]],
    representation_loss_rows: list[dict[str, Any]],
    failure_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    stable_rows = [row for row in concept_rows if row.get("candidate_source") == "stable_role"]
    role_based_transferable = [row for row in transferable_rows if row.get("candidate_source") == "stable_role"]
    mean_lift_vs_role = mean_metric(role_based_transferable, "target_mean_concept_lift_vs_role_raw")
    mean_lift_vs_bag = mean_metric(role_based_transferable, "target_mean_concept_lift_vs_role_bag")
    mean_lift_vs_surface = mean_metric(role_based_transferable, "target_mean_concept_lift_vs_surface_raw")
    mean_lift_vs_graph = mean_metric(role_based_transferable, "target_mean_concept_lift_vs_graph_no_label")
    mean_lift_vs_m2 = mean_metric(role_based_transferable, "target_mean_concept_lift_vs_m2")
    mean_compression = mean_metric(role_based_transferable, "mean_compression_gain")
    future_lift = mean_metric(role_based_transferable, "target_mean_future_option_prediction_lift")
    positive_families = sum(1 for row in by_family_rows if int(row.get("positive_lift_families", 0)) > 0)
    target_family_score_count = sum(1 for row in transfer_rows if row.get("projection_used"))
    dominant_transfer_share = _dominant_transfer_share(role_based_transferable)
    dominant_failure_pattern = _dominant_failure_pattern(failure_decomposition_rows)
    best_generator_by_role = best_generator_by_metric(generator_failure_rows, "max_lift_vs_best_role")
    best_generator_by_surface = best_generator_by_metric(generator_failure_rows, "max_lift_vs_surface_effect")
    best_generator_by_future = best_generator_by_metric(generator_failure_rows, "max_lift_vs_future_option")
    best_generator_by_raw_m2 = best_generator_by_metric(generator_failure_rows, "max_lift_vs_raw_m2")
    best_generator_by_graph = best_generator_by_metric(generator_failure_rows, "max_lift_vs_graph_no_label")
    closest_candidate = closest_candidate_rows[0] if closest_candidate_rows else {}
    projection_averaging_loss_detected = any(float(row.get("local_match_lost_by_averaging", 0.0)) > 0.05 for row in projection_audit_rows)
    mean_projection_averaging_loss = float(np.mean([float(row.get("local_match_lost_by_averaging", 0.0)) for row in projection_audit_rows])) if projection_audit_rows else 0.0
    max_projection_averaging_loss = float(max((float(row.get("local_match_lost_by_averaging", 0.0)) for row in projection_audit_rows), default=0.0))
    worst_projection_row = max(projection_audit_rows, key=lambda row: float(row.get("local_match_lost_by_averaging", 0.0)), default={})
    best_local_match_row = max(projection_audit_rows, key=lambda row: float(row.get("target_best_match_score", 0.0)), default={})
    raw_m2_dominance_rate = _rate(baseline_dominance_rows, "raw_m2_dominates")
    graph_no_label_dominance_rate = _rate(baseline_dominance_rows, "graph_no_label_dominates")
    raw_m2_failure_rate = _failed_gate_rate(failure_decomposition_rows, "raw_m2")
    graph_no_label_failure_rate = _failed_gate_rate(failure_decomposition_rows, "graph_no_label")
    best_role_failure_rate = _failed_gate_rate(failure_decomposition_rows, "best_role")
    unordered_bag_failure_rate = _failed_gate_rate(failure_decomposition_rows, "unordered_bag")
    compression_failure_rate = _failed_gate_rate(failure_decomposition_rows, "compression")
    future_option_failure_rate = _rate(failure_decomposition_rows, "future_option_failed")
    surface_effect_failure_rate = _rate(failure_decomposition_rows, "surface_effect_failed")
    family_coverage_failure_rate = _rate(failure_decomposition_rows, "family_coverage_failed")
    diagnostic_conclusion = build_diagnostic_conclusion(
        family_coverage_failure_rate=family_coverage_failure_rate,
        raw_m2_failure_rate=raw_m2_failure_rate,
        graph_no_label_failure_rate=graph_no_label_failure_rate,
        raw_m2_dominance_rate=raw_m2_dominance_rate,
        graph_no_label_dominance_rate=graph_no_label_dominance_rate,
        projection_averaging_loss_detected=projection_averaging_loss_detected,
        future_option_failure_rate=future_option_failure_rate,
        surface_effect_failure_rate=surface_effect_failure_rate,
    )
    best_candidate_by_role_lift = _candidate_summary(failure_decomposition_rows, "lift_vs_best_role", prefer_max=True)
    best_candidate_by_surface_lift = _candidate_summary(failure_decomposition_rows, "lift_vs_surface_effect", prefer_max=True)
    best_candidate_by_future_lift = _candidate_summary(failure_decomposition_rows, "future_option_prediction_lift", prefer_max=True)
    best_candidate_by_raw_m2_lift = _candidate_summary(failure_decomposition_rows, "lift_vs_raw_m2", prefer_max=True)
    best_candidate_by_graph_no_label_lift = _candidate_summary(failure_decomposition_rows, "lift_vs_graph_no_label", prefer_max=True)
    worst_candidate_by_surface_lift = _candidate_summary(failure_decomposition_rows, "lift_vs_surface_effect", prefer_max=False)
    worst_candidate_by_future_lift = _candidate_summary(failure_decomposition_rows, "future_option_prediction_lift", prefer_max=False)

    conclusion = "m4_role_based_not_established"
    if not stable_rows and fallback_only_signal_detected:
        conclusion = "m4_fallback_signal_only_m3_bottleneck"
    elif not stable_rows:
        conclusion = "m4_role_based_pipeline_not_diagnostic"
    elif not role_based_transferable and fallback_only_signal_detected:
        conclusion = "m4_fallback_signal_only_m3_bottleneck"
    elif len(role_based_transferable) >= 5 and mean_lift_vs_role >= 0.10 and mean_lift_vs_bag >= 0.10 and mean_lift_vs_surface >= 0.10 and mean_compression >= 0.10 and future_lift > 0.0 and positive_families >= 12 and dominant_transfer_share <= 0.40:
        conclusion = "m4_role_based_very_strong"
    elif len(role_based_transferable) >= 3 and mean_lift_vs_role >= 0.05 and mean_lift_vs_bag >= 0.05 and mean_lift_vs_surface >= 0.05 and mean_compression >= 0.05 and future_lift > 0.0 and positive_families >= 8:
        conclusion = "m4_role_based_strong"
    elif len(role_based_transferable) >= 2 and mean_lift_vs_role > 0.0 and mean_lift_vs_bag > 0.0 and mean_lift_vs_surface > 0.0 and mean_compression > 0.0 and future_lift > 0.0 and positive_families >= 6:
        conclusion = "m4_role_based_weak"

    report = {
        "transfer_report_summary": transfer_report.get("report", {}),
        "requested_workers": requested_workers,
        "effective_workers": effective_workers,
        "multiprocessing_used": multiprocessing_used,
        "role_based_stable_concepts": len(stable_rows),
        "role_based_transferable_concepts": len(role_based_transferable),
        "transferable_role_based_concepts": len(role_based_transferable),
        "role_based_candidate_count": len(concept_rows),
        "fallback_diagnostic_candidate_count": len(fallback_rows),
        "fallback_diagnostic_score_count": len(fallback_score_rows),
        "candidate_evidence_lanes": {
            "eligible_stable": len(concept_rows),
            "diagnostic_fallback": len(fallback_rows),
            "diagnostic_mixed": 0,
            "diagnostic_unknown_manifest": 0,
            "diagnostic_unresolved_manifest": 0,
            "diagnostic_unclassified": 0,
        },
        "eligible_candidate_count": len(concept_rows),
        "diagnostic_only_candidate_count": len(fallback_rows),
        "accepted_candidate_count": len(role_based_transferable),
        "diagnostic_only_policy": {
            "fallback_candidates_promoted": False,
            "mixed_candidates_promoted": False,
            "unknown_manifest_candidates_promoted": False,
            "unresolved_manifest_candidates_promoted": False,
        },
        "target_family_score_count": target_family_score_count,
        "mean_lift_vs_best_individual_m3_role": mean_lift_vs_role,
        "mean_lift_vs_unordered_role_bag": mean_lift_vs_bag,
        "mean_lift_vs_surface_effect_raw": mean_lift_vs_surface,
        "mean_lift_vs_graph_no_label": mean_lift_vs_graph,
        "mean_lift_vs_raw_m2": mean_lift_vs_m2,
        "mean_compression_gain": mean_compression,
        "future_option_prediction_lift_vs_best_role": future_lift,
        "positive_lift_families": positive_families,
        "fallback_only_signal_detected": fallback_only_signal_detected,
        "candidate_generator_counts": dict(Counter({row["generator_type"]: row["candidate_count"] for row in generator_rows})),
        "candidate_cap_hit_count": sum(1 for row in cap_rows if row["candidate_cap_hit"]),
        "dominant_failure_pattern": dominant_failure_pattern,
        "best_generator_by_role_lift": best_generator_by_role,
        "best_generator_by_surface_lift": best_generator_by_surface,
        "best_generator_by_future_lift": best_generator_by_future,
        "best_generator_by_raw_m2_lift": best_generator_by_raw_m2,
        "best_generator_by_graph_no_label_lift": best_generator_by_graph,
        "closest_candidate_to_transfer": closest_candidate,
        "projection_averaging_loss_detected": projection_averaging_loss_detected,
        "mean_projection_averaging_loss": mean_projection_averaging_loss,
        "max_projection_averaging_loss": max_projection_averaging_loss,
        "worst_projection_averaging_candidate": str(worst_projection_row.get("concept_id", "")),
        "best_local_match_candidate": str(best_local_match_row.get("concept_id", "")),
        "best_local_match_score": float(best_local_match_row.get("target_best_match_score", 0.0)),
        "best_local_match_target_family": str(best_local_match_row.get("best_target_family_id", "")),
        "raw_m2_dominance_rate": raw_m2_dominance_rate,
        "graph_no_label_dominance_rate": graph_no_label_dominance_rate,
        "raw_m2_failure_rate": raw_m2_failure_rate,
        "graph_no_label_failure_rate": graph_no_label_failure_rate,
        "best_role_failure_rate": best_role_failure_rate,
        "unordered_bag_failure_rate": unordered_bag_failure_rate,
        "compression_failure_rate": compression_failure_rate,
        "future_option_failure_rate": future_option_failure_rate,
        "surface_effect_failure_rate": surface_effect_failure_rate,
        "family_coverage_failure_rate": family_coverage_failure_rate,
        "failure_decomposition_rows": len(failure_decomposition_rows),
        "generator_failure_summary_rows": len(generator_failure_rows),
        "projection_audit_rows": len(projection_audit_rows),
        "baseline_dominance_rows": len(baseline_dominance_rows),
        "representation_loss_rows": len(representation_loss_rows),
        "closest_candidate_rows": len(closest_candidate_rows),
        "best_candidate_by_role_lift": best_candidate_by_role_lift,
        "best_candidate_by_surface_lift": best_candidate_by_surface_lift,
        "best_candidate_by_future_lift": best_candidate_by_future_lift,
        "best_candidate_by_raw_m2_lift": best_candidate_by_raw_m2_lift,
        "best_candidate_by_graph_no_label_lift": best_candidate_by_graph_no_label_lift,
        "worst_candidate_by_surface_lift": worst_candidate_by_surface_lift,
        "worst_candidate_by_future_lift": worst_candidate_by_future_lift,
        "diagnostic_conclusion": diagnostic_conclusion,
        "scientific_conclusion": conclusion,
        "v10a_can_proceed": conclusion in {"m4_role_based_weak", "m4_role_based_strong", "m4_role_based_very_strong"},
        "source_clean_context_mode": "single_family_streaming",
        "target_role_overlap_diagnostic_only": True,
        "manifest_family_used_for_grouping": False,
        "graph_no_label_source_clean": True,
        "merge_diag": merge_diag,
    }
    return {
        "config": {
            "output_dir": config.output_dir,
            "workers": config.workers,
            "previous_v09b_dir": config.previous_v09b_dir,
        },
        "report": report,
        "m4_failure_diagnostics": failure_diagnostics,
        "validation": {"scientific_conclusion": conclusion, "proceed_to_v10a": report["v10a_can_proceed"]},
    }


def format_v10e_report(payload: dict[str, Any]) -> str:
    report = payload["report"]
    return "\n".join(
        [
            "ARC-AGI3 v0.10e-b: strict role-based M4 validation",
            "",
            f"scientific_conclusion={report['scientific_conclusion']}",
            f"requested_workers={report['requested_workers']}",
            f"effective_workers={report['effective_workers']}",
            f"multiprocessing_used={report['multiprocessing_used']}",
            f"role_based_stable_concepts={report['role_based_stable_concepts']}",
            f"role_based_transferable_concepts={report['role_based_transferable_concepts']}",
            f"role_based_candidate_count={report['role_based_candidate_count']}",
            f"fallback_diagnostic_candidate_count={report['fallback_diagnostic_candidate_count']}",
            f"target_family_score_count={report['target_family_score_count']}",
            f"mean_lift_vs_best_individual_m3_role={report['mean_lift_vs_best_individual_m3_role']}",
            f"mean_lift_vs_unordered_role_bag={report['mean_lift_vs_unordered_role_bag']}",
            f"mean_lift_vs_surface_effect_raw={report['mean_lift_vs_surface_effect_raw']}",
            f"mean_lift_vs_graph_no_label={report['mean_lift_vs_graph_no_label']}",
            f"mean_lift_vs_raw_m2={report['mean_lift_vs_raw_m2']}",
            f"mean_compression_gain={report['mean_compression_gain']}",
            f"future_option_prediction_lift_vs_best_role={report['future_option_prediction_lift_vs_best_role']}",
            f"positive_lift_families={report['positive_lift_families']}",
            f"candidate_cap_hit_count={report['candidate_cap_hit_count']}",
            f"dominant_failure_pattern={report['dominant_failure_pattern']}",
            f"best_generator_by_role_lift={report['best_generator_by_role_lift']}",
            f"best_generator_by_surface_lift={report['best_generator_by_surface_lift']}",
            f"best_generator_by_future_lift={report['best_generator_by_future_lift']}",
            f"best_generator_by_raw_m2_lift={report['best_generator_by_raw_m2_lift']}",
            f"best_generator_by_graph_no_label_lift={report['best_generator_by_graph_no_label_lift']}",
            f"closest_candidate_to_transfer={report['closest_candidate_to_transfer']}",
            f"projection_averaging_loss_detected={report['projection_averaging_loss_detected']}",
            f"mean_projection_averaging_loss={report['mean_projection_averaging_loss']}",
            f"max_projection_averaging_loss={report['max_projection_averaging_loss']}",
            f"worst_projection_averaging_candidate={report['worst_projection_averaging_candidate']}",
            f"best_local_match_candidate={report['best_local_match_candidate']}",
            f"best_local_match_score={report['best_local_match_score']}",
            f"best_local_match_target_family={report['best_local_match_target_family']}",
            f"raw_m2_dominance_rate={report['raw_m2_dominance_rate']}",
            f"graph_no_label_dominance_rate={report['graph_no_label_dominance_rate']}",
            f"raw_m2_failure_rate={report['raw_m2_failure_rate']}",
            f"graph_no_label_failure_rate={report['graph_no_label_failure_rate']}",
            f"best_role_failure_rate={report['best_role_failure_rate']}",
            f"unordered_bag_failure_rate={report['unordered_bag_failure_rate']}",
            f"compression_failure_rate={report['compression_failure_rate']}",
            f"future_option_failure_rate={report['future_option_failure_rate']}",
            f"surface_effect_failure_rate={report['surface_effect_failure_rate']}",
            f"family_coverage_failure_rate={report['family_coverage_failure_rate']}",
            f"failure_decomposition_rows={report['failure_decomposition_rows']}",
            f"generator_failure_summary_rows={report['generator_failure_summary_rows']}",
            f"projection_audit_rows={report['projection_audit_rows']}",
            f"baseline_dominance_rows={report['baseline_dominance_rows']}",
            f"representation_loss_rows={report['representation_loss_rows']}",
            f"closest_candidate_rows={report['closest_candidate_rows']}",
            f"diagnostic_conclusion={report['diagnostic_conclusion']}",
            f"v10a_can_proceed={report['v10a_can_proceed']}",
        ]
    )


def _base_scoring_context(context: Any) -> Any:
    if hasattr(context, "full_neighborhoods"):
        return context

    class _Proxy:
        pass

    proxy = _Proxy()
    proxy.source_neighborhoods = context.source_neighborhoods
    proxy.source_roles = context.source_roles
    proxy.target_families = context.target_families
    proxy.full_neighborhoods = _target_records(context)
    proxy.full_no_label_neighborhoods = dict(context.source_neighborhoods)
    return proxy


def _target_records(context: Any) -> dict[str, Any]:
    if hasattr(context, "target_neighborhoods"):
        return context.target_neighborhoods
    return getattr(context, "full_neighborhoods", {})


def _subset_prefixed(features: dict[str, Any], prefix: str) -> dict[str, float]:
    return {str(key).removeprefix(prefix): float(value) for key, value in features.items() if str(key).startswith(prefix)}


def _graph_sort_key(record: Any, role_id: str) -> tuple[float, float, str]:
    directional = getattr(record, "directional_features", {})
    return (
        -float(directional.get("predecessor_count", 0.0)),
        float(directional.get("successor_count", 0.0)),
        role_id,
    )


def _dominant_transfer_share(rows: list[dict[str, Any]]) -> float:
    denominator = sum(max(0, int(row.get("positive_lift_family_count", 0))) for row in rows) or 1
    return max((int(row.get("positive_lift_family_count", 0)) / denominator for row in rows), default=0.0)


def _dominant_failure_pattern(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    patterns = Counter(tuple(row.get("failed_gates", ())) for row in rows)
    return json.dumps(list(patterns.most_common(1)[0][0]))


def _dominant_pattern_from_series(series: pd.Series) -> str:
    patterns = Counter(tuple(item) for item in series if item)
    if not patterns:
        return json.dumps([])
    return json.dumps(list(patterns.most_common(1)[0][0]))


def best_generator_by_metric(rows: list[dict[str, Any]], metric: str) -> str:
    if not rows:
        return ""
    valid = [
        row
        for row in rows
        if metric in row and row.get(metric) is not None and str(row.get("generator_type", "")).strip()
    ]
    if not valid:
        return ""
    return str(max(valid, key=lambda row: float(row.get(metric, 0.0))).get("generator_type", ""))


def _rate(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([1.0 if row.get(field) else 0.0 for row in rows]))


def _failed_gate_rate(rows: list[dict[str, Any]], gate: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([1.0 if gate in row.get("failed_gates", []) else 0.0 for row in rows]))


def _candidate_summary(rows: list[dict[str, Any]], metric: str, *, prefer_max: bool) -> dict[str, Any]:
    if not rows:
        return {}
    valid = [row for row in rows if metric in row and row.get(metric) is not None]
    if not valid:
        return {}
    chosen = max(valid, key=lambda row: float(row.get(metric, 0.0))) if prefer_max else min(valid, key=lambda row: float(row.get(metric, 0.0)))
    return {
        "concept_id": str(chosen.get("concept_id", "")),
        "generator_type": str(chosen.get("generator_type", "")),
        "target_family_id": str(chosen.get("target_family_id", "")),
        "value": float(chosen.get(metric, 0.0)),
    }


def build_diagnostic_conclusion(
    *,
    family_coverage_failure_rate: float,
    raw_m2_failure_rate: float,
    graph_no_label_failure_rate: float,
    raw_m2_dominance_rate: float,
    graph_no_label_dominance_rate: float,
    projection_averaging_loss_detected: bool,
    future_option_failure_rate: float,
    surface_effect_failure_rate: float,
) -> str:
    if family_coverage_failure_rate == 1.0:
        return "m4_failure_due_to_family_coverage"
    elif raw_m2_failure_rate >= 0.80 and graph_no_label_failure_rate >= 0.80:
        return "m4_failure_due_to_representation_loss_against_m2_and_graph"
    elif raw_m2_dominance_rate >= 0.80 or graph_no_label_dominance_rate >= 0.80:
        return "m4_failure_due_to_baseline_dominance"
    elif projection_averaging_loss_detected:
        return "m4_failure_due_to_projection_averaging"
    elif future_option_failure_rate >= 0.80 or surface_effect_failure_rate >= 0.80:
        return "m4_failure_due_to_m3_representation_loss"
    else:
        return "m4_failure_mixed_causes"


def _normalize_shard_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for key in (
        "future_option_delta_profile",
        "graph_position_profile",
        "local_motif_profile",
        "predecessor_successor_profile",
        "temporal_profile",
        "effect_residual_profile",
        "concept_structural_fingerprint",
        "failure_reason_counts",
    ):
        value = normalized.get(key)
        if isinstance(value, str) and value.startswith("{"):
            normalized[key] = json.loads(value)
    for key in (
        "role_ids_fold_local",
        "fold_local_role_ids",
        "role_labels",
        "canonical_role_fingerprint_hashes",
        "canonical_role_signatures_json",
        "canonical_role_label_or_family",
        "graph_ordered_role_pattern",
        "source_games_present",
        "source_manifest_families_present",
        "source_manifest_family_support_signature",
        "heldout_families_seen",
        "episode_ordered_role_sequence",
    ):
        value = normalized.get(key)
        if isinstance(value, str) and value.startswith("["):
            normalized[key] = json.loads(value)
    if isinstance(normalized.get("role_similarity_vectors"), str) and normalized["role_similarity_vectors"].startswith("["):
        normalized["role_similarity_vectors"] = json.loads(normalized["role_similarity_vectors"])
    return normalized
