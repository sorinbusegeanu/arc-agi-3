from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from v5_0.contracts.avatar_types import (
    AdaptiveSolveReport,
    AvatarIdentificationReport,
    CampaignRunReport,
    ContactExperimentReport,
    CrossResetPOIEvidence,
    CrossResetHUDEvidence,
    HUDEpisodeReport,
    HUDDetectionReport,
    HUDCellSample,
    MultiResetAvatarReport,
    HUDHintReport,
    GameLevelBatchReport,
    LevelSolution,
    MechanicReport,
    POIMechanicEvidence,
    SavedLevelTrace,
    SolveReport,
    TraceOptimizationReport,
    POIDiscoveryReport,
    POIEpisode,
    ProbeTransitionRecord,
)
from v5_0.memory.trace_store import get_global_trace_store_path
from v5_0.render.palette import get_render_palette as _get_shared_render_palette, render_value_to_rgb


def get_game_root_output_dir(base_output_dir: str, game_id: str) -> str:
    return str(Path(str(base_output_dir)) / str(game_id))


def resolve_run_dir(base_output_dir: str | None, game_id: str) -> Path:
    if base_output_dir:
        base = Path(base_output_dir)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = Path("runs_v5_0") / timestamp
    run_dir = base / game_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_artifacts(
    *,
    run_dir: Path,
    transitions: tuple[ProbeTransitionRecord, ...],
    report: AvatarIdentificationReport,
    write_montage: bool,
) -> dict[str, str]:
    transitions_path = run_dir / "bootstrap_transitions.json"
    candidates_path = run_dir / "avatar_candidates.json"
    summary_path = run_dir / "avatar_summary.json"

    transitions_payload = [record.to_dict() for record in transitions]
    candidates_payload = [candidate.to_dict() for candidate in report.candidates]
    summary_payload = {
        "selected": report.selected.to_dict(),
        "diagnostics": report.diagnostics.to_dict(),
    }

    transitions_path.write_text(json.dumps(transitions_payload, indent=2), encoding="utf-8")
    candidates_path.write_text(json.dumps(candidates_payload, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    output = {
        "bootstrap_transitions.json": str(transitions_path),
        "avatar_candidates.json": str(candidates_path),
        "avatar_summary.json": str(summary_path),
    }

    if write_montage:
        montage_path = run_dir / "probe_montage.png"
        if _write_probe_montage(montage_path, transitions):
            output["probe_montage.png"] = str(montage_path)
    return output


def _write_probe_montage(path: Path, transitions: tuple[ProbeTransitionRecord, ...]) -> bool:
    try:
        from PIL import Image
    except Exception:
        return False

    tiles: list[Image.Image] = []
    for record in transitions:
        if record.pre_frame is None or record.post_frame is None:
            continue
        pre = _frame_to_image(record.pre_frame)
        post = _frame_to_image(record.post_frame)
        row = Image.new("RGB", (pre.width + post.width + 1, max(pre.height, post.height)), color=(15, 15, 15))
        row.paste(pre, (0, 0))
        row.paste(post, (pre.width + 1, 0))
        tiles.append(row)

    if not tiles:
        return False

    width = max(tile.width for tile in tiles)
    height = sum(tile.height for tile in tiles) + (len(tiles) - 1)
    montage = Image.new("RGB", (width, height), color=(0, 0, 0))
    y = 0
    for tile in tiles:
        montage.paste(tile, (0, y))
        y += tile.height + 1
    montage = montage.resize((montage.width * 8, montage.height * 8), resample=Image.Resampling.NEAREST)
    montage.save(path)
    return True


def _frame_to_image(frame: tuple[tuple[int, ...], ...]):
    return frame_grid_to_pil_image(frame)


def frame_grid_to_json_list(frame: tuple[tuple[int, ...], ...]) -> list[list[int]]:
    return [[int(value) for value in row] for row in tuple(frame)]


def frame_grid_to_pil_image(frame: tuple[tuple[int, ...], ...]):
    from PIL import Image

    height = len(frame)
    width = len(frame[0]) if height else 0
    image = Image.new("RGB", (width, height), color=(0, 0, 0))
    pixels = image.load()
    palette = get_render_palette()
    for y, row in enumerate(frame):
        for x, value in enumerate(row):
            pixels[x, y] = render_value_to_rgb(int(value), palette)
    return image


def get_render_palette() -> dict[int, tuple[int, int, int]]:
    return _get_shared_render_palette()


def json_ready(payload: Any) -> str:
    return json.dumps(payload, indent=2)


def write_multi_reset_artifacts(
    *,
    run_dir: Path,
    report: MultiResetAvatarReport,
    write_montage: bool,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "multi_reset_summary.json"
    evidence_path = run_dir / "cross_reset_evidence.json"
    index_path = run_dir / "episode_index.json"

    summary_payload = {
        "selected": report.selected.to_dict(),
        "diagnostics": report.diagnostics.to_dict(),
    }
    evidence_payload = [item.to_dict() for item in report.cross_reset_evidence]

    episode_index_payload: dict[str, dict[str, Any]] = {}
    for episode in report.episodes:
        episode_dir = run_dir / f"episode_{episode.episode_index:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        episode_artifacts = write_artifacts(
            run_dir=episode_dir,
            transitions=episode.transitions,
            report=episode.report,
            write_montage=write_montage,
        )
        episode_index_payload[str(episode.episode_index)] = {
            "seed": int(episode.seed),
            "artifact_paths": episode_artifacts,
            "failure_reason": episode.report.selected.failure_reason,
            "confidence": float(episode.report.selected.confidence),
        }

    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    evidence_path.write_text(json.dumps(evidence_payload, indent=2), encoding="utf-8")
    index_path.write_text(json.dumps(episode_index_payload, indent=2), encoding="utf-8")

    return {
        "multi_reset_summary.json": str(summary_path),
        "cross_reset_evidence.json": str(evidence_path),
        "episode_index.json": str(index_path),
    }


def write_poi_artifacts(
    *,
    run_dir: Path,
    poi_report: POIDiscoveryReport,
    cross_reset_poi_evidence: tuple[CrossResetPOIEvidence, ...],
    episode_poi_reports: tuple[POIEpisode, ...] = (),
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    poi_candidates_path = run_dir / "poi_candidates.json"
    poi_summary_path = run_dir / "poi_summary.json"
    poi_logs_path = run_dir / "poi_contact_logs.json"
    cross_reset_poi_path = run_dir / "cross_reset_poi_evidence.json"

    poi_candidates_path.write_text(
        json.dumps([candidate for candidate in [item.__dict__ for item in poi_report.candidates]], indent=2),
        encoding="utf-8",
    )
    poi_summary_path.write_text(
        json.dumps(
            {
                "selected": poi_report.selected.to_dict(),
                "diagnostics": poi_report.diagnostics.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    poi_logs_path.write_text(json.dumps(list(poi_report.contact_logs), indent=2), encoding="utf-8")
    cross_reset_poi_path.write_text(
        json.dumps([item.to_dict() for item in cross_reset_poi_evidence], indent=2),
        encoding="utf-8",
    )

    for episode in episode_poi_reports:
        episode_dir = run_dir / f"episode_{episode.episode_index:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "poi_candidates.json").write_text(
            json.dumps([item.__dict__ for item in episode.poi_report.candidates], indent=2),
            encoding="utf-8",
        )
        (episode_dir / "poi_summary.json").write_text(
            json.dumps(
                {
                    "selected": episode.poi_report.selected.to_dict(),
                    "diagnostics": episode.poi_report.diagnostics.to_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (episode_dir / "poi_contact_logs.json").write_text(
            json.dumps(list(episode.poi_report.contact_logs), indent=2),
            encoding="utf-8",
        )

    return {
        "poi_candidates.json": str(poi_candidates_path),
        "poi_summary.json": str(poi_summary_path),
        "poi_contact_logs.json": str(poi_logs_path),
        "cross_reset_poi_evidence.json": str(cross_reset_poi_path),
    }


def write_contact_experiment_artifacts(
    *,
    run_dir: Path,
    report: ContactExperimentReport,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "contact_experiments_summary.json"
    tested_path = run_dir / "tested_pois.json"
    outcomes_path = run_dir / "contact_outcomes.json"

    summary_payload = {
        "tested_poi_ids": [item.poi_id for item in report.tested_pois],
        "episode_indices": sorted({item.episode_index for item in report.tested_pois}),
        "outcome_type_counts": dict(report.diagnostics.get("outcome_type_counts", {})),
        "level_change_count": int(report.diagnostics.get("level_change_count", 0)),
        "terminal_count": int(report.diagnostics.get("terminal_count", 0)),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    tested_path.write_text(
        json.dumps([item.to_dict() for item in report.tested_pois], indent=2),
        encoding="utf-8",
    )
    outcomes_path.write_text(
        json.dumps([item.outcome.to_dict() for item in report.tested_pois], indent=2),
        encoding="utf-8",
    )

    for episode in report.episodes:
        for index, tested in enumerate(episode.tested_pois):
            poi_dir = run_dir / f"episode_{episode.episode_index:03d}" / f"contact_poi_{index:03d}"
            poi_dir.mkdir(parents=True, exist_ok=True)
            (poi_dir / "policy.json").write_text(json.dumps(tested.policy.to_dict(), indent=2), encoding="utf-8")
            (poi_dir / "contact_steps.json").write_text(
                json.dumps([item.to_dict() for item in tested.steps], indent=2),
                encoding="utf-8",
            )
            (poi_dir / "contact_outcome.json").write_text(
                json.dumps(tested.outcome.to_dict(), indent=2),
                encoding="utf-8",
            )

    return {
        "contact_experiments_summary.json": str(summary_path),
        "tested_pois.json": str(tested_path),
        "contact_outcomes.json": str(outcomes_path),
    }


def write_hud_artifacts(
    *,
    run_dir: Path,
    hud_report: HUDDetectionReport,
    cross_reset_hud_evidence: tuple[CrossResetHUDEvidence, ...],
    episode_hud_reports: tuple[HUDEpisodeReport, ...] = (),
    hud_value_samples: tuple[HUDCellSample, ...] = (),
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "hud_summary.json"
    regions_path = run_dir / "hud_regions.json"
    mask_path = run_dir / "hud_mask.json"
    evidence_path = run_dir / "cross_reset_hud_evidence.json"

    summary_path.write_text(
        json.dumps(
            {
                "failure_reason": hud_report.failure_reason,
                "diagnostics": hud_report.diagnostics.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    regions_path.write_text(
        json.dumps([item.to_dict() for item in hud_report.regions], indent=2),
        encoding="utf-8",
    )
    mask_path.write_text(
        json.dumps(hud_report.mask.to_dict(), indent=2),
        encoding="utf-8",
    )
    evidence_path.write_text(
        json.dumps([item.to_dict() for item in cross_reset_hud_evidence], indent=2),
        encoding="utf-8",
    )

    for episode in episode_hud_reports:
        episode_dir = run_dir / f"episode_{episode.episode_index:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "hud_summary.json").write_text(
            json.dumps(
                {
                    "failure_reason": episode.hud_report.failure_reason,
                    "diagnostics": episode.hud_report.diagnostics.to_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (episode_dir / "hud_regions.json").write_text(
            json.dumps([item.to_dict() for item in episode.hud_report.regions], indent=2),
            encoding="utf-8",
        )
        (episode_dir / "hud_mask.json").write_text(
            json.dumps(episode.hud_report.mask.to_dict(), indent=2),
            encoding="utf-8",
        )

    output = {
        "hud_summary.json": str(summary_path),
        "hud_regions.json": str(regions_path),
        "hud_mask.json": str(mask_path),
        "cross_reset_hud_evidence.json": str(evidence_path),
    }
    if hud_value_samples:
        samples_path = run_dir / "hud_value_samples.json"
        samples_path.write_text(
            json.dumps([item.to_dict() for item in hud_value_samples], indent=2),
            encoding="utf-8",
        )
        output["hud_value_samples.json"] = str(samples_path)
    return output


def write_full_analysis_index(
    *,
    run_dir: Path,
    game_id: str,
    episode_count: int,
    phase_status: dict[str, str],
    artifact_paths: dict[str, str],
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    index_path = run_dir / "full_analysis_index.json"
    payload = {
        "run_dir": str(run_dir),
        "game_id": str(game_id),
        "episode_count": int(episode_count),
        "phase_status": dict(phase_status),
        "artifact_paths": {
            "avatar": {
                key: value
                for key, value in artifact_paths.items()
                if key in {"multi_reset_summary.json", "cross_reset_evidence.json", "episode_index.json"}
            },
            "poi": {
                key: value
                for key, value in artifact_paths.items()
                if key in {"poi_candidates.json", "poi_summary.json", "poi_contact_logs.json", "cross_reset_poi_evidence.json"}
            },
            "contact": {
                key: value
                for key, value in artifact_paths.items()
                if key in {"contact_experiments_summary.json", "tested_pois.json", "contact_outcomes.json"}
            },
            "hud": {
                key: value
                for key, value in artifact_paths.items()
                if key in {"hud_summary.json", "hud_regions.json", "hud_mask.json", "cross_reset_hud_evidence.json"}
            },
        },
    }
    index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"full_analysis_index.json": str(index_path)}


def write_hud_hint_artifacts(
    *,
    run_dir: Path,
    report: HUDHintReport,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "hud_hint_summary.json"
    matches_path = run_dir / "hud_poi_matches.json"
    target_path = run_dir / "hud_target_selection.json"

    summary_path.write_text(
        json.dumps([item.to_dict() for item in report.hud_hints], indent=2),
        encoding="utf-8",
    )
    matches_path.write_text(
        json.dumps([item.to_dict() for item in report.matches], indent=2),
        encoding="utf-8",
    )
    target_path.write_text(
        json.dumps(
            {
                "selected": report.selected.to_dict(),
                "diagnostics": report.diagnostics.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "hud_hint_summary.json": str(summary_path),
        "hud_poi_matches.json": str(matches_path),
        "hud_target_selection.json": str(target_path),
    }


def write_solve_artifacts(
    *,
    run_dir: Path,
    report: SolveReport,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "solve_summary.json"
    diagnostics_path = run_dir / "solve_diagnostics.json"
    steps_path = run_dir / "solve_steps.json"

    summary_path.write_text(
        json.dumps(
            {
                "selected_target_id": report.selected_target_id,
                "solved": bool(report.solved),
                "failure_reason": report.failure_reason,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    diagnostics_path.write_text(
        json.dumps(report.diagnostics.to_dict(), indent=2),
        encoding="utf-8",
    )
    all_steps = []
    for episode in report.episodes:
        for step in episode.steps:
            payload = step.to_dict()
            payload["episode_index"] = int(episode.episode_index)
            all_steps.append(payload)
    all_steps.sort(key=lambda item: (int(item.get("episode_index", 0)), int(item.get("step_index", 0))))
    steps_path.write_text(json.dumps(all_steps, indent=2), encoding="utf-8")

    for episode in report.episodes:
        episode_dir = run_dir / f"episode_{int(episode.episode_index):03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "solve_steps.json").write_text(
            json.dumps([item.to_dict() for item in episode.steps], indent=2),
            encoding="utf-8",
        )

    return {
        "solve_summary.json": str(summary_path),
        "solve_diagnostics.json": str(diagnostics_path),
        "solve_steps.json": str(steps_path),
    }


def write_mechanic_artifacts(
    *,
    run_dir: Path,
    report: MechanicReport,
    evidence: tuple[POIMechanicEvidence, ...] = (),
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "mechanic_summary.json"
    memory_path = run_dir / "mechanic_memory.json"
    evidence_path = run_dir / "mechanic_evidence.json"
    diagnostics_path = run_dir / "mechanic_diagnostics.json"

    summary_path.write_text(
        json.dumps(
            {
                "selected_poi_id": report.memory.selected_poi_id,
                "retired_poi_ids": list(report.memory.retired_poi_ids),
                "failure_reason": report.failure_reason,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    memory_path.write_text(json.dumps(report.memory.to_dict(), indent=2), encoding="utf-8")
    evidence_path.write_text(json.dumps([item.to_dict() for item in tuple(evidence)], indent=2), encoding="utf-8")
    diagnostics_path.write_text(json.dumps(report.diagnostics.to_dict(), indent=2), encoding="utf-8")
    return {
        "mechanic_summary.json": str(summary_path),
        "mechanic_memory.json": str(memory_path),
        "mechanic_evidence.json": str(evidence_path),
        "mechanic_diagnostics.json": str(diagnostics_path),
    }


def write_adaptive_solve_artifacts(
    *,
    run_dir: Path,
    report: AdaptiveSolveReport,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "adaptive_solve_summary.json"
    diagnostics_path = run_dir / "adaptive_solve_diagnostics.json"
    steps_path = run_dir / "adaptive_solve_steps.json"
    partial_path = run_dir / "partial_successful_action_sequences.json"

    summary_path.write_text(
        json.dumps(
            {
                "selected_target_id": report.selected_target_id,
                "solved": bool(report.solved),
                "failure_reason": report.failure_reason,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    diagnostics_path.write_text(json.dumps(report.diagnostics.to_dict(), indent=2), encoding="utf-8")

    all_steps = []
    for episode in report.episodes:
        for step in episode.steps:
            payload = step.to_dict()
            payload["episode_index"] = int(episode.episode_index)
            all_steps.append(payload)
    all_steps.sort(key=lambda item: (int(item.get("episode_index", 0)), int(item.get("step_index", 0))))
    steps_path.write_text(json.dumps(all_steps, indent=2), encoding="utf-8")
    partial_sequences = tuple(getattr(report, "partial_successful_action_sequences", ()) or ())
    partial_path.write_text(json.dumps(list(partial_sequences), indent=2), encoding="utf-8")

    for episode in report.episodes:
        episode_dir = run_dir / f"episode_{int(episode.episode_index):03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "adaptive_solve_steps.json").write_text(
            json.dumps([item.to_dict() for item in episode.steps], indent=2),
            encoding="utf-8",
        )

    return {
        "adaptive_solve_summary.json": str(summary_path),
        "adaptive_solve_diagnostics.json": str(diagnostics_path),
        "adaptive_solve_steps.json": str(steps_path),
        "partial_successful_action_sequences.json": str(partial_path),
    }


def write_level_solution_artifacts(
    *,
    run_dir: Path,
    solution: LevelSolution,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "level_solution.json"
    actions_path = run_dir / "level_solution_actions.json"

    summary_path.write_text(
        json.dumps(
            {
                "game_id": solution.game_id,
                "level_id": solution.level_id,
                "solved": bool(solution.solved),
                "step_count": int(solution.step_count),
                "terminal": bool(solution.terminal),
                "level_transition": bool(solution.level_transition),
                "failure_reason": solution.failure_reason,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    actions_path.write_text(
        json.dumps([item.to_dict() for item in solution.action_trace], indent=2),
        encoding="utf-8",
    )
    return {
        "level_solution.json": str(summary_path),
        "level_solution_actions.json": str(actions_path),
    }


def write_game_level_batch_artifacts(
    *,
    run_dir: Path,
    report: GameLevelBatchReport,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "game_level_batch_summary.json"
    diagnostics_path = run_dir / "game_level_batch_diagnostics.json"
    index_path = run_dir / "game_level_index.json"

    summary_path.write_text(
        json.dumps(
            {
                "game_id": report.game_id,
                "level_count": len(report.levels),
                "solved_level_count": int(report.diagnostics.solved_level_count),
                "failed_level_count": int(report.diagnostics.failed_level_count),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    diagnostics_path.write_text(json.dumps(report.diagnostics.to_dict(), indent=2), encoding="utf-8")

    level_index = {}
    for level in sorted(report.levels, key=lambda item: str(item.level_id)):
        level_index[str(level.level_id)] = {
            "solved": bool(level.solved),
            "failure_reason": level.failure_reason,
            "run_directory": str(Path(next(iter(level.artifact_paths.values()), "")).parent) if level.artifact_paths else "",
            "solution_artifacts": {
                key: value
                for key, value in level.artifact_paths.items()
                if key in {"level_solution.json", "level_solution_actions.json"}
            },
        }
    index_path.write_text(json.dumps(level_index, indent=2), encoding="utf-8")
    return {
        "game_level_batch_summary.json": str(summary_path),
        "game_level_batch_diagnostics.json": str(diagnostics_path),
        "game_level_index.json": str(index_path),
    }


def write_campaign_artifacts(
    *,
    run_dir: Path,
    report: CampaignRunReport,
    campaign_step_trace: tuple[Any, ...] | list[Any] | None = None,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "campaign_summary.json"
    levels_path = run_dir / "campaign_levels.json"
    trace_path = run_dir / "campaign_action_trace.json"
    step_trace_path = run_dir / "campaign_step_trace.json"

    summary_path.write_text(
        json.dumps(
            {
                "game_id": report.game_id,
                "solved": bool(report.solved),
                "highest_reached_level_id": report.highest_reached_level_id,
                "failure_reason": report.failure_reason,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    level_index = {}
    for level in sorted(report.levels, key=lambda item: str(item.level_id)):
        trace_path_value = None
        replay_verified = False
        best_step_count = None
        if level.solution is not None:
            best_step_count = int(level.solution.step_count)
        level_index[str(level.level_id)] = {
            "solved": bool(level.solved),
            "best_trace_path": trace_path_value,
            "best_step_count": best_step_count,
            "replay_verified": bool(replay_verified),
            "failure_reason": level.failure_reason,
        }
    levels_path.write_text(json.dumps(level_index, indent=2), encoding="utf-8")
    trace_path.write_text(
        json.dumps([item.to_dict() for item in report.global_action_trace], indent=2),
        encoding="utf-8",
    )
    step_source = tuple(campaign_step_trace) if campaign_step_trace is not None else tuple(report.global_action_trace)
    step_trace_path.write_text(
        json.dumps([item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in step_source], indent=2),
        encoding="utf-8",
    )
    return {
        "campaign_summary.json": str(summary_path),
        "campaign_levels.json": str(levels_path),
        "campaign_action_trace.json": str(trace_path),
        "campaign_step_trace.json": str(step_trace_path),
    }


def write_saved_level_trace(
    *,
    run_dir: Path,
    trace: SavedLevelTrace,
    step_trace: tuple[Any, ...] | list[Any] | None = None,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "saved_level_trace.json"
    actions_path = run_dir / "saved_level_trace_actions.json"
    steps_path = run_dir / "saved_level_trace_steps.json"
    summary_path.write_text(
        json.dumps(trace.to_dict(), indent=2),
        encoding="utf-8",
    )
    actions_path.write_text(json.dumps(list(trace.action_trace), indent=2), encoding="utf-8")
    out = {
        "saved_level_trace.json": str(summary_path),
        "saved_level_trace_actions.json": str(actions_path),
    }
    if step_trace is not None:
        rows = []
        for item in tuple(step_trace):
            if hasattr(item, "to_dict"):
                rows.append(item.to_dict())
            elif isinstance(item, dict):
                rows.append(dict(item))
        steps_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        out["saved_level_trace_steps.json"] = str(steps_path)
    return out


def write_generated_trajectories(
    *,
    run_dir: Path,
    generated_trajectories: tuple[Any, ...] | list[Any],
    rejected_trajectories: tuple[Any, ...] | list[Any] = (),
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "generated_trajectories.json"
    valid_path = run_dir / "generated_trajectories_valid.json"
    rejected_path = run_dir / "generated_trajectories_rejected.json"
    rows = []
    for item in tuple(generated_trajectories or ()):
        if hasattr(item, "to_dict"):
            rows.append(item.to_dict())
        elif isinstance(item, dict):
            rows.append(dict(item))
    rejected_rows = []
    for item in tuple(rejected_trajectories or ()):
        if hasattr(item, "to_dict"):
            rejected_rows.append(item.to_dict())
        elif isinstance(item, dict):
            rejected_rows.append(dict(item))
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    valid_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    rejected_path.write_text(json.dumps(rejected_rows, indent=2), encoding="utf-8")
    return {
        "generated_trajectories.json": str(path),
        "generated_trajectories_valid.json": str(valid_path),
        "generated_trajectories_rejected.json": str(rejected_path),
    }


def write_trajectory_attempts(
    *,
    run_dir: Path,
    trajectory_attempts: tuple[Any, ...] | list[Any],
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "trajectory_attempts.json"
    rows = []
    for item in tuple(trajectory_attempts or ()):
        if hasattr(item, "to_dict"):
            rows.append(item.to_dict())
        elif isinstance(item, dict):
            rows.append(dict(item))
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return {"trajectory_attempts.json": str(path)}


def write_trajectory_stats(
    *,
    run_dir: Path,
    trajectory_stats: Any,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "trajectory_stats.json"
    if hasattr(trajectory_stats, "to_dict"):
        payload = trajectory_stats.to_dict()
    elif isinstance(trajectory_stats, dict):
        payload = dict(trajectory_stats)
    elif isinstance(trajectory_stats, (list, tuple)):
        payload = [
            item.to_dict() if hasattr(item, "to_dict") else dict(item) if isinstance(item, dict) else item
            for item in trajectory_stats
        ]
    else:
        payload = {}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"trajectory_stats.json": str(path)}


def write_trace_optimization_artifacts(
    *,
    run_dir: Path,
    report: TraceOptimizationReport,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "trace_optimization_summary.json"
    candidates_path = run_dir / "trace_optimization_candidates.json"
    optimized_path = run_dir / "optimized_trace.json"
    optimized_actions_path = run_dir / "optimized_trace_actions.json"

    summary_path.write_text(
        json.dumps(
            {
                "game_id": report.game_id,
                "level_id": report.level_id,
                "failure_reason": report.failure_reason,
                "best_step_count": int(report.best_candidate.step_count),
                "baseline_step_count": int(report.baseline_trace.step_count),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    candidates_path.write_text(
        json.dumps([item.to_dict() for item in report.candidates], indent=2),
        encoding="utf-8",
    )
    optimized_path.write_text(
        json.dumps(
            {
                "game_id": report.game_id,
                "level_id": report.level_id,
                "action_trace": list(report.best_candidate.action_trace),
                "step_count": int(report.best_candidate.step_count),
                "verified": bool(report.best_candidate.verified),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    optimized_actions_path.write_text(
        json.dumps(list(report.best_candidate.action_trace), indent=2),
        encoding="utf-8",
    )
    return {
        "trace_optimization_summary.json": str(summary_path),
        "trace_optimization_candidates.json": str(candidates_path),
        "optimized_trace.json": str(optimized_path),
        "optimized_trace_actions.json": str(optimized_actions_path),
    }


def write_trace_analysis_batch_artifacts(
    *,
    run_dir: Path,
    game_id: str,
    reports: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    trace_db_path: str | None = None,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "trace_analysis_summary.json"
    reports_path = run_dir / "trace_analysis_reports.json"
    index_path = run_dir / "trace_analysis_index.json"
    report_items = list(reports)
    resolved_trace_db_path = str(trace_db_path or get_global_trace_store_path())
    summary_path.write_text(
        json.dumps(
            {
                "game_id": game_id,
                "report_count": len(report_items),
                "trace_store_db": resolved_trace_db_path,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    reports_path.write_text(json.dumps(report_items, indent=2), encoding="utf-8")
    index_payload: dict[str, Any] = {}
    for item in report_items:
        if not isinstance(item, dict):
            continue
        level_id = str(item.get("level_id", ""))
        if not level_id:
            continue
        diagnostics = dict(item.get("diagnostics", {}))
        baseline_step_count = int(item.get("baseline_step_count", diagnostics.get("baseline_step_count", 0)) or 0)
        optimized_step_count = int(item.get("best_step_count", diagnostics.get("best_step_count", baseline_step_count)) or baseline_step_count)
        index_payload[level_id] = {
            "baseline_trace_id": diagnostics.get("baseline_trace_id"),
            "baseline_step_count": baseline_step_count,
            "optimized_trace_id": diagnostics.get("optimized_trace_id"),
            "optimized_step_count": optimized_step_count,
            "improvement": int(baseline_step_count - optimized_step_count),
            "verified": bool(
                item.get("verified", False)
                or diagnostics.get("db_updated", False)
                or (optimized_step_count <= baseline_step_count and diagnostics.get("optimized_trace_id") is not None)
            ),
            "optimized_at": item.get("optimized_at") or diagnostics.get("optimized_at"),
        }
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    return {
        "trace_analysis_summary.json": str(summary_path),
        "trace_analysis_reports.json": str(reports_path),
        "trace_analysis_index.json": str(index_path),
    }


def write_trace_store_index_artifacts(
    *,
    run_dir: Path,
    game_id: str,
    solved_levels: tuple[str, ...] | list[str],
    trace_db_path: str,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    index_path = run_dir / "trace_store_index.json"
    level_summary: dict[str, Any] = {}
    try:
        import sqlite3

        with sqlite3.connect(trace_db_path) as conn:
            rows = conn.execute(
                """
                SELECT level_id, MIN(step_count) AS best_step_count
                FROM level_traces
                WHERE game_id = ? AND solved = 1 AND replay_verified = 1
                GROUP BY level_id
                ORDER BY level_id
                """,
                (game_id,),
            ).fetchall()
        level_summary = {
            str(level_id): {
                "best_step_count": int(best_step_count),
                "best_trace_id": None,
                "optimization_status": "known",
            }
            for level_id, best_step_count in rows
        }
    except Exception:
        level_summary = {
            str(level_id): {
                "best_step_count": None,
                "best_trace_id": None,
                "optimization_status": "unknown",
            }
            for level_id in tuple(solved_levels)
        }
    payload = {
        "game_id": game_id,
        "solved_levels": list(tuple(solved_levels)),
        "trace_store_db": str(trace_db_path),
        "levels": level_summary,
    }
    index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"trace_store_index.json": str(index_path)}
