from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from v6.evaluation.broad_game_validation import family_for_game
from v6.game_sets import GameSetManifest, load_game_set_manifest
from v6.memory_types import M2TransformationFamily
from v6.transformation_families_v07 import M1Record, entropy, entropy_to_coherence, graph_rows, load_m1_contingencies, pairwise_similarity


@dataclass(frozen=True)
class M2ExpandV08cConfig:
    input_dir: str = "runs/v6/v07_cd2_extended32"
    m1_input_dir: str = "runs/v6/v06_cd2_extended32"
    output_dir: str = "runs/v6/v07_cd2_extended32_expanded"
    game_set_manifest: str | None = None
    game_set_name: str | None = None
    min_family_support: int = 3
    max_family_share: float = 0.25
    min_expanded_families: int = 40
    target_expanded_families: int = 60


@dataclass(frozen=True)
class FamilyDiagnostics:
    family_id: str
    family_label_candidate: str
    member_count: int
    family_share: float
    games_present: tuple[str, ...]
    manifest_families_present: tuple[str, ...]
    cross_game_presence: int
    cross_manifest_family_presence: int
    outcome_signature_distribution: dict[str, int]
    motif_candidate_distribution: dict[str, int]
    outcome_signature_entropy: float
    motif_entropy: float
    context_lift_band_entropy: float
    terminal_frequency_band_entropy: float
    no_change_frequency_band_entropy: float
    position_change_frequency_band_entropy: float
    family_coherence: float
    mean_prediction_accuracy: float
    mean_context_lift: float
    overcompressed: bool
    split_reasons: tuple[str, ...]


SplitKey = Callable[[M1Record, dict[str, str]], str]


def run_m2_expand_v08c(config: M2ExpandV08cConfig) -> dict[str, Any]:
    input_dir = Path(config.input_dir)
    m1_input_dir = Path(config.m1_input_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    game_set = load_game_set_manifest(
        manifest_path=config.game_set_manifest,
        game_set_name=config.game_set_name,
        fallback_games=(),
    )
    game_family_map = build_game_family_map(game_set)
    contingencies = load_m1_contingencies(m1_input_dir)
    if game_set.games:
        requested = set(game_set.games)
        contingencies = [row for row in contingencies if row.game_id in requested]
    contingency_map = {row.contingency_id: row for row in contingencies}

    original_families = load_m2_families(input_dir)
    if game_set.games:
        selected = set(game_set.games)
        original_families = [row for row in original_families if selected.intersection(row.games_present)]

    total_contingencies = max(1, sum(len(family.contingency_ids) for family in original_families))
    original_diagnostics = [
        family_diagnostics(family, [contingency_map[item] for item in family.contingency_ids if item in contingency_map], total_contingencies, game_family_map, config.max_family_share)
        for family in original_families
    ]

    expanded_groups: list[ExpandedGroup] = []
    for family, diag in zip(original_families, original_diagnostics, strict=True):
        members = [contingency_map[item] for item in family.contingency_ids if item in contingency_map]
        expanded_groups.extend(
            expand_family_groups(
                family=family,
                members=members,
                total_contingencies=total_contingencies,
                game_family_map=game_family_map,
                config=config,
                overcompressed=diag.overcompressed,
            )
        )

    if len(expanded_groups) < config.min_expanded_families:
        expanded_groups = secondary_expansion_pass(
            groups=expanded_groups,
            total_contingencies=total_contingencies,
            game_family_map=game_family_map,
            config=config,
        )

    expanded_families, expansion_rows = finalize_expanded_families(
        groups=expanded_groups,
        total_contingencies=total_contingencies,
        game_family_map=game_family_map,
        min_family_support=config.min_family_support,
    )
    mappings = contingency_family_rows(expanded_families)
    nodes, edges = graph_rows(expanded_families, similarity_threshold=0.70)

    original_report = load_json(input_dir / "v07_report.json")
    diagnostics_before = summarize_family_population(
        families=original_families,
        contingencies=contingencies,
        game_family_map=game_family_map,
        overcompressed_ids={diag.family_id for diag in original_diagnostics if diag.overcompressed},
        min_family_support=config.min_family_support,
    )
    diagnostics_after = summarize_family_population(
        families=expanded_families,
        contingencies=contingencies,
        game_family_map=game_family_map,
        overcompressed_ids=set(),
        min_family_support=config.min_family_support,
    )
    payload = build_expansion_payload(
        config=config,
        game_set=game_set,
        original_report=original_report,
        diagnostics_before=diagnostics_before,
        diagnostics_after=diagnostics_after,
        expansion_rows=expansion_rows,
        total_contingencies=len(contingencies),
    )

    write_outputs(
        output_dir=output_dir,
        families=expanded_families,
        mappings=mappings,
        nodes=nodes,
        edges=edges,
        expansion_rows=expansion_rows,
        payload=payload,
    )
    return payload


@dataclass
class ExpandedGroup:
    original_family_id: str
    label_candidate: str
    members: list[M1Record]
    split_reasons: list[str]
    split_key_values: dict[str, Any]


def build_game_family_map(game_set: GameSetManifest) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for family_name, games in game_set.families.items():
        for game in games:
            mapping.setdefault(game, family_name)
    for game in game_set.games:
        mapping.setdefault(game, family_for_game(game))
    return mapping


def load_m2_families(input_dir: Path) -> list[M2TransformationFamily]:
    rows = load_json(input_dir / "m2_families.json")
    output: list[M2TransformationFamily] = []
    for row in rows:
        output.append(
            M2TransformationFamily(
                family_id=str(row["family_id"]),
                family_label_candidate=str(row["family_label_candidate"]),
                games_present=list(_parse_json_list(row.get("games_present"))),
                samplers_present=list(_parse_json_list(row.get("samplers_present"))),
                contingency_ids=list(_parse_json_list(row.get("contingency_ids"))),
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
                examples=list(_parse_json_list(row.get("examples"))),
                notes=_parse_json_dict(row.get("notes")),
            )
        )
    return output


def family_diagnostics(
    family: M2TransformationFamily,
    members: list[M1Record],
    total_contingencies: int,
    game_family_map: dict[str, str],
    max_family_share: float,
) -> FamilyDiagnostics:
    family_share = len(members) / max(1, total_contingencies)
    manifest_families = tuple(sorted({game_family_map.get(member.game_id, "unknown") for member in members}))
    outcome_counts = Counter(member.outcome_signature for member in members)
    motif_counts = Counter(member.future_option_motif_candidate for member in members)
    lift_bands = Counter(context_lift_band(member.context_lift) for member in members)
    terminal_bands = Counter(terminal_band(member) for member in members)
    no_change_bands = Counter(no_change_band(member) for member in members)
    position_bands = Counter(position_change_band(member) for member in members)
    outcome_entropy = entropy(outcome_counts)
    motif_value = entropy(motif_counts)
    coherence = family.family_coherence if members else 0.0
    low_internal_coherence = coherence < 0.95 or float(family.notes.get("mean_pairwise_similarity", coherence)) < 0.96
    mixed_profiles = len(manifest_families) >= 6 and (
        len(outcome_counts) > 1
        or len(motif_counts) > 1
        or len(lift_bands) > 2
        or len(terminal_bands) > 1
        or len(no_change_bands) > 1
        or len(position_bands) > 1
    )
    overcompressed = (
        family_share > max_family_share
        or (len(members) >= 3000 and len(manifest_families) >= 6)
        or (len(members) >= 5000 and len(manifest_families) >= 4 and (low_internal_coherence or mixed_profiles))
    )
    reasons: list[str] = []
    if family_share > max_family_share:
        reasons.append("family_share_exceeds_max")
    if len(manifest_families) >= 6 and low_internal_coherence:
        reasons.append("cross_manifest_low_coherence")
    if mixed_profiles:
        reasons.append("mixed_structural_profiles")
    return FamilyDiagnostics(
        family_id=family.family_id,
        family_label_candidate=family.family_label_candidate,
        member_count=len(members),
        family_share=family_share,
        games_present=tuple(sorted({member.game_id for member in members})),
        manifest_families_present=manifest_families,
        cross_game_presence=len({member.game_id for member in members}),
        cross_manifest_family_presence=len(manifest_families),
        outcome_signature_distribution=dict(sorted(outcome_counts.items())),
        motif_candidate_distribution=dict(sorted(motif_counts.items())),
        outcome_signature_entropy=outcome_entropy,
        motif_entropy=motif_value,
        context_lift_band_entropy=entropy(lift_bands),
        terminal_frequency_band_entropy=entropy(terminal_bands),
        no_change_frequency_band_entropy=entropy(no_change_bands),
        position_change_frequency_band_entropy=entropy(position_bands),
        family_coherence=coherence,
        mean_prediction_accuracy=float(np.mean([member.prediction_accuracy for member in members])) if members else 0.0,
        mean_context_lift=float(np.mean([member.context_lift for member in members])) if members else 0.0,
        overcompressed=overcompressed,
        split_reasons=tuple(reasons),
    )


def expand_family_groups(
    *,
    family: M2TransformationFamily,
    members: list[M1Record],
    total_contingencies: int,
    game_family_map: dict[str, str],
    config: M2ExpandV08cConfig,
    overcompressed: bool,
) -> list[ExpandedGroup]:
    groups = [
        ExpandedGroup(
            original_family_id=family.family_id,
            label_candidate=family.family_label_candidate,
            members=sorted(members, key=lambda item: item.contingency_id),
            split_reasons=[],
            split_key_values={},
        )
    ]
    splitters: list[tuple[str, SplitKey]] = [("manifest_family", lambda row, gf: gf.get(row.game_id, "unknown"))]
    for split_name, splitter in splitters:
        next_groups: list[ExpandedGroup] = []
        changed = False
        for group in groups:
            if not should_split_group(group, total_contingencies, config, overcompressed):
                next_groups.append(group)
                continue
            candidate_groups = grouped_members(group.members, game_family_map, splitter)
            if not split_is_useful(candidate_groups, config.min_family_support):
                next_groups.append(group)
                continue
            changed = True
            for key, rows in sorted(candidate_groups.items()):
                next_groups.append(
                    ExpandedGroup(
                        original_family_id=group.original_family_id,
                        label_candidate=group.label_candidate,
                        members=rows,
                        split_reasons=group.split_reasons + [split_name],
                        split_key_values={**group.split_key_values, split_name: key},
                    )
                )
        groups = next_groups
        if split_name == "manifest_family" and all(
            len(group.members) / max(1, total_contingencies) <= config.max_family_share and len(group.members) < 3000
            for group in groups
        ):
            break
        if len(groups) >= config.target_expanded_families and all(
            len(group.members) / max(1, total_contingencies) <= config.max_family_share for group in groups
        ):
            break
        if not changed:
            continue
    return groups


def secondary_expansion_pass(
    *,
    groups: list[ExpandedGroup],
    total_contingencies: int,
    game_family_map: dict[str, str],
    config: M2ExpandV08cConfig,
) -> list[ExpandedGroup]:
    expanded = list(groups)
    for split_name, splitter in [("manifest_family", lambda row, gf: gf.get(row.game_id, "unknown"))]:
        if len(expanded) >= config.min_expanded_families:
            break
        next_groups: list[ExpandedGroup] = []
        changed = False
        for group in sorted(expanded, key=lambda item: (-len(item.members), item.original_family_id)):
            if len(expanded) + 1 >= config.target_expanded_families or len(group.members) < max(config.min_family_support * 16, 120):
                next_groups.append(group)
                continue
            candidate_groups = grouped_members(group.members, game_family_map, splitter)
            if not split_is_useful(candidate_groups, config.min_family_support):
                next_groups.append(group)
                continue
            changed = True
            for key, rows in sorted(candidate_groups.items()):
                next_groups.append(
                    ExpandedGroup(
                        original_family_id=group.original_family_id,
                        label_candidate=group.label_candidate,
                        members=rows,
                        split_reasons=group.split_reasons + [split_name],
                        split_key_values={**group.split_key_values, split_name: key},
                    )
                )
            if len(next_groups) >= config.target_expanded_families:
                next_groups.extend(expanded[expanded.index(group) + 1 :])
                break
        expanded = next_groups if changed else expanded
    return expanded


def should_split_group(
    group: ExpandedGroup,
    total_contingencies: int,
    config: M2ExpandV08cConfig,
    overcompressed: bool,
) -> bool:
    member_count = len(group.members)
    share = member_count / max(1, total_contingencies)
    return overcompressed or share > config.max_family_share or member_count >= 3000


def grouped_members(
    members: list[M1Record],
    game_family_map: dict[str, str],
    splitter: SplitKey,
) -> dict[str, list[M1Record]]:
    groups: dict[str, list[M1Record]] = defaultdict(list)
    for member in members:
        groups[splitter(member, game_family_map)].append(member)
    return {key: sorted(rows, key=lambda item: item.contingency_id) for key, rows in groups.items()}


def split_is_useful(groups: dict[str, list[M1Record]], min_family_support: int) -> bool:
    if len(groups) <= 1:
        return False
    sizes = [len(rows) for rows in groups.values()]
    viable = sum(1 for size in sizes if size >= min_family_support)
    tiny = sum(1 for size in sizes if size < min_family_support)
    return viable >= 2 and tiny <= viable


def outcome_profile(row: M1Record) -> str:
    if row.outcome_signature == "terminal_transition":
        return "terminal"
    if row.outcome_signature in {"blocked_no_change", "preserve_no_change"}:
        return "no_change"
    if row.outcome_signature == "position_like_change":
        return "position"
    if row.outcome_signature == "large_change":
        return "large_change"
    return "change"


def action_bucket(action: int) -> str:
    if action in {1, 2, 3, 4}:
        return "directional"
    if action in {5, 6}:
        return "interactive"
    return "other"


def prediction_band(value: float) -> str:
    if value < 0.80:
        return "acc_low"
    if value < 0.90:
        return "acc_mid"
    if value < 0.97:
        return "acc_high"
    return "acc_very_high"


def support_band(value: int) -> str:
    if value < 3:
        return "support_low"
    if value < 10:
        return "support_mid"
    if value < 30:
        return "support_high"
    return "support_very_high"


def context_lift_band(value: float) -> str:
    if value < 0.05:
        return "lift_low"
    if value < 0.10:
        return "lift_mid"
    if value < 0.20:
        return "lift_high"
    return "lift_very_high"


def terminal_band(row: M1Record) -> str:
    return "terminal" if row.outcome_signature == "terminal_transition" else "non_terminal"


def no_change_band(row: M1Record) -> str:
    return "no_change" if row.outcome_signature in {"blocked_no_change", "preserve_no_change"} else "not_no_change"


def position_change_band(row: M1Record) -> str:
    return "position_change" if row.outcome_signature == "position_like_change" else "not_position_change"


def finalize_expanded_families(
    *,
    groups: list[ExpandedGroup],
    total_contingencies: int,
    game_family_map: dict[str, str],
    min_family_support: int,
) -> tuple[list[M2TransformationFamily], list[dict[str, Any]]]:
    families: list[M2TransformationFamily] = []
    expansion_rows: list[dict[str, Any]] = []
    total_families = max(1, len(groups))
    sorted_groups = sorted(
        groups,
        key=lambda item: (
            item.original_family_id,
            tuple(item.split_reasons),
            json.dumps(item.split_key_values, sort_keys=True),
            item.members[0].contingency_id if item.members else "",
        ),
    )
    for index, group in enumerate(sorted_groups, start=1):
        family_id = f"m2x-{index:04d}"
        family = build_family_from_members(
            family_id=family_id,
            label_candidate=group.label_candidate,
            members=group.members,
            total_families=total_families,
            min_family_support=min_family_support,
        )
        families.append(family)
        manifest_families = sorted({game_family_map.get(member.game_id, "unknown") for member in group.members})
        expansion_rows.append(
            {
                "original_family_id": group.original_family_id,
                "expanded_family_id": family_id,
                "split_reason": ",".join(group.split_reasons) if group.split_reasons else "unchanged",
                "split_key_values": dict(sorted(group.split_key_values.items())),
                "support_count": family.support_count,
                "games_present": family.games_present,
                "manifest_families_present": manifest_families,
                "family_coherence": family.family_coherence,
                "motif_profile": family.motif_candidate_distribution,
                "outcome_signature_profile": family.outcome_signature_distribution,
            }
        )
    return families, expansion_rows


def build_family_from_members(
    *,
    family_id: str,
    label_candidate: str,
    members: list[M1Record],
    total_families: int,
    min_family_support: int,
) -> M2TransformationFamily:
    outcome_counts = Counter(member.outcome_signature for member in members)
    motif_counts = Counter(member.future_option_motif_candidate for member in members)
    dominant_outcome, dominant_count = max(outcome_counts.items(), key=lambda item: (item[1], item[0]))
    pairwise = pairwise_similarity(members)
    outcome_entropy = entropy(outcome_counts)
    motif_entropy_value = entropy(motif_counts)
    dominant_ratio = dominant_count / max(1, len(members))
    coherence = (
        0.4 * dominant_ratio
        + 0.4 * pairwise
        + 0.1 * entropy_to_coherence(outcome_entropy, len(outcome_counts))
        + 0.1 * entropy_to_coherence(motif_entropy_value, len(motif_counts))
    )
    support_total = sum(member.support_count for member in members)
    mean_accuracy = float(np.mean([member.prediction_accuracy for member in members]))
    mean_context_lift = float(np.mean([member.context_lift for member in members]))
    stable = support_total >= int(min_family_support) and coherence >= 0.70 and mean_accuracy >= 0.75
    return M2TransformationFamily(
        family_id=family_id,
        family_label_candidate=label_candidate,
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
        compression_ratio=len(members) / max(1, total_families),
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
            "motif_entropy": motif_entropy_value,
        },
    )


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


def summarize_family_population(
    *,
    families: list[M2TransformationFamily],
    contingencies: list[M1Record],
    game_family_map: dict[str, str],
    overcompressed_ids: set[str],
    min_family_support: int,
) -> dict[str, Any]:
    total_contingencies = max(1, len(contingencies))
    sizes = sorted(len(family.contingency_ids) for family in families)
    family_by_id = {family.family_id: family for family in families}
    contingency_map = {item.contingency_id: item for item in contingencies}
    families_by_game = Counter()
    families_by_manifest_family = Counter()
    cross_game_count = 0
    cross_manifest_count = 0
    motif_entropy_values = []
    outcome_entropy_values = []
    lift_entropy_values = []
    terminal_entropy_values = []
    no_change_entropy_values = []
    position_entropy_values = []
    for family in families:
        members = [contingency_map[item] for item in family.contingency_ids if item in contingency_map]
        games = sorted({member.game_id for member in members})
        manifest_families = sorted({game_family_map.get(game, "unknown") for game in games})
        for game in games:
            families_by_game[game] += 1
        for manifest_family in manifest_families:
            families_by_manifest_family[manifest_family] += 1
        cross_game_count += int(len(games) > 1)
        cross_manifest_count += int(len(manifest_families) > 1)
        motif_entropy_values.append(entropy(Counter(member.future_option_motif_candidate for member in members)))
        outcome_entropy_values.append(entropy(Counter(member.outcome_signature for member in members)))
        lift_entropy_values.append(entropy(Counter(context_lift_band(member.context_lift) for member in members)))
        terminal_entropy_values.append(entropy(Counter(terminal_band(member) for member in members)))
        no_change_entropy_values.append(entropy(Counter(no_change_band(member) for member in members)))
        position_entropy_values.append(entropy(Counter(position_change_band(member) for member in members)))
    size_distribution = Counter(sizes)
    return {
        "m2_family_count": len(families),
        "contingencies_per_family_min": int(min(sizes)) if sizes else 0,
        "contingencies_per_family_median": float(np.median(sizes)) if sizes else 0.0,
        "contingencies_per_family_mean": float(np.mean(sizes)) if sizes else 0.0,
        "contingencies_per_family_max": int(max(sizes)) if sizes else 0,
        "largest_family_size": int(max(sizes)) if sizes else 0,
        "largest_family_percent": (max(sizes) / total_contingencies) if sizes else 0.0,
        "singleton_family_count": sum(1 for size in sizes if size == 1),
        "tiny_family_count": sum(1 for size in sizes if size < min_family_support),
        "overcompressed_family_count": len(overcompressed_ids),
        "families_by_game": dict(sorted(families_by_game.items())),
        "families_by_manifest_family": dict(sorted(families_by_manifest_family.items())),
        "cross_game_family_count": cross_game_count,
        "cross_manifest_family_count": cross_manifest_count,
        "family_entropy": entropy(size_distribution),
        "motif_entropy": float(np.mean(motif_entropy_values)) if motif_entropy_values else 0.0,
        "outcome_signature_entropy": float(np.mean(outcome_entropy_values)) if outcome_entropy_values else 0.0,
        "context_lift_band_entropy": float(np.mean(lift_entropy_values)) if lift_entropy_values else 0.0,
        "terminal_frequency_band_entropy": float(np.mean(terminal_entropy_values)) if terminal_entropy_values else 0.0,
        "no_change_frequency_band_entropy": float(np.mean(no_change_entropy_values)) if no_change_entropy_values else 0.0,
        "position_change_frequency_band_entropy": float(np.mean(position_entropy_values)) if position_entropy_values else 0.0,
        "expanded_games_covered": len({member.game_id for member in contingencies}),
        "expanded_manifest_families_covered": len({game_family_map.get(member.game_id, "unknown") for member in contingencies}),
    }


def build_expansion_payload(
    *,
    config: M2ExpandV08cConfig,
    game_set: GameSetManifest,
    original_report: dict[str, Any],
    diagnostics_before: dict[str, Any],
    diagnostics_after: dict[str, Any],
    expansion_rows: list[dict[str, Any]],
    total_contingencies: int,
) -> dict[str, Any]:
    expansion_succeeds = (
        diagnostics_after["m2_family_count"] >= config.min_expanded_families
        and diagnostics_after["largest_family_percent"] <= config.max_family_share
        and diagnostics_after["tiny_family_count"] <= diagnostics_after["m2_family_count"] / 2
        and diagnostics_after["expanded_games_covered"] >= 24
        and diagnostics_after["expanded_manifest_families_covered"] >= 12
    )
    payload = {
        "config": {
            "input_dir": config.input_dir,
            "m1_input_dir": config.m1_input_dir,
            "output_dir": config.output_dir,
            "min_family_support": config.min_family_support,
            "max_family_share": config.max_family_share,
            "min_expanded_families": config.min_expanded_families,
            "target_expanded_families": config.target_expanded_families,
            "game_set_name": game_set.name,
        },
        "report": {
            "original_v07_report": original_report.get("report", {}),
            "original_m2_family_count": diagnostics_before["m2_family_count"],
            "expanded_m2_family_count": diagnostics_after["m2_family_count"],
            "largest_family_percent_before": diagnostics_before["largest_family_percent"],
            "largest_family_percent_after": diagnostics_after["largest_family_percent"],
            "diagnostics_before": diagnostics_before,
            "diagnostics_after": diagnostics_after,
            "overcompressed_families_detected": sorted({row["original_family_id"] for row in expansion_rows if row["split_reason"] != "unchanged"}),
            "split_reasons": Counter(row["split_reason"] for row in expansion_rows if row["split_reason"] != "unchanged"),
            "target_40_to_60_useful_families_achieved": config.min_expanded_families <= diagnostics_after["m2_family_count"] <= config.target_expanded_families,
            "expansion_suitable_for_v08_retry": expansion_succeeds,
            "total_contingencies_loaded": total_contingencies,
        },
        "validation": {
            "diagnostic_success": total_contingencies > 0,
            "expansion_suitable_for_v08_retry": expansion_succeeds,
        },
    }
    return payload


def write_outputs(
    *,
    output_dir: Path,
    families: list[M2TransformationFamily],
    mappings: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    expansion_rows: list[dict[str, Any]],
    payload: dict[str, Any],
) -> None:
    family_rows = [family.to_record() for family in families]
    (output_dir / "m2_families.json").write_text(json.dumps(family_rows, indent=2), encoding="utf-8")
    _write_parquet(output_dir / "m2_families.parquet", family_rows)
    _write_parquet(output_dir / "contingency_to_family.parquet", mappings)
    _write_parquet(output_dir / "m2_graph_nodes.parquet", nodes)
    _write_parquet(output_dir / "m2_graph_edges.parquet", edges)
    _write_parquet(output_dir / "m2_expansion_map.parquet", expansion_rows)
    (output_dir / "m2_expansion_diagnostics.json").write_text(json.dumps(payload["report"], indent=2), encoding="utf-8")
    (output_dir / "v07_expanded_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "v07_expanded_report.txt").write_text(format_v07_expanded_report(payload), encoding="utf-8")
    (output_dir / "v07_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "v07_report.txt").write_text(format_v07_expanded_report(payload), encoding="utf-8")


def format_v07_expanded_report(payload: dict[str, Any]) -> str:
    report = payload["report"]
    before = report["diagnostics_before"]
    after = report["diagnostics_after"]
    split_reasons = report["split_reasons"]
    lines = [
        "ARC-AGI3 v0.8c-m2-expansion-anti-collapse",
        f"original_m2_family_count={report['original_m2_family_count']}",
        f"expanded_m2_family_count={report['expanded_m2_family_count']}",
        f"largest_family_percent_before={report['largest_family_percent_before']:.6f}",
        f"largest_family_percent_after={report['largest_family_percent_after']:.6f}",
        f"overcompressed_families_detected={','.join(report['overcompressed_families_detected'])}",
        f"split_reasons={dict(split_reasons)}",
        f"singleton_family_count_after={after['singleton_family_count']}",
        f"tiny_family_count_after={after['tiny_family_count']}",
        f"target_40_to_60_useful_families_achieved={report['target_40_to_60_useful_families_achieved']}",
        f"expansion_suitable_for_v08_retry={report['expansion_suitable_for_v08_retry']}",
        "",
        "Before:",
        f"m2_family_count={before['m2_family_count']} largest_family_size={before['largest_family_size']} largest_family_percent={before['largest_family_percent']:.6f}",
        "",
        "After:",
        f"m2_family_count={after['m2_family_count']} largest_family_size={after['largest_family_size']} largest_family_percent={after['largest_family_percent']:.6f}",
        f"cross_game_family_count={after['cross_game_family_count']}",
        f"cross_manifest_family_count={after['cross_manifest_family_count']}",
    ]
    return "\n".join(lines)


def load_json(path: Path) -> Any:
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
