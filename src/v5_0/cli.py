from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from v5_0.io.artifact_writer import get_game_root_output_dir
from v5_0.io.final_video_builder import build_final_game_video
from v5_0.runtime.run_avatar_bootstrap import (
    SUPPORTED_GAMES,
    replay_saved_level_solution,
    run_full_campaign_analysis,
    run_trace_analysis_for_game,
    run_avatar_and_poi_bootstrap_multi_reset,
    run_avatar_poi_hud_bootstrap_multi_reset,
    run_avatar_poi_contact_bootstrap_multi_reset,
    run_avatar_bootstrap,
    run_avatar_bootstrap_multi_reset,
    run_full_analysis_for_game_levels,
    run_full_analysis_for_level,
    run_full_bootstrap_analysis,
    run_full_bootstrap_analysis_with_hud_targeting,
    run_full_bootstrap_analysis_with_adaptive_solve,
    run_full_bootstrap_analysis_with_mechanics,
    run_full_bootstrap_analysis_with_solve,
)
from v5_0.memory.trace_store import get_global_trace_store_path
from v5_0.replay.player import replay_saved_trace

FULL_ANALYSIS_STDOUT_ARTIFACT_KEYS: tuple[str, ...] = (
    "multi_reset_summary.json",
    "cross_reset_evidence.json",
    "episode_index.json",
    "poi_candidates.json",
    "poi_summary.json",
    "poi_contact_logs.json",
    "cross_reset_poi_evidence.json",
    "contact_experiments_summary.json",
    "tested_pois.json",
    "contact_outcomes.json",
    "hud_summary.json",
    "hud_regions.json",
    "hud_mask.json",
    "cross_reset_hud_evidence.json",
    "hud_value_samples.json",
    "full_analysis_index.json",
    "solve_summary.json",
    "solve_diagnostics.json",
    "solve_steps.json",
    "adaptive_solve_summary.json",
    "adaptive_solve_diagnostics.json",
    "adaptive_solve_steps.json",
    "mechanic_summary.json",
    "mechanic_memory.json",
    "mechanic_evidence.json",
    "mechanic_diagnostics.json",
)
STDOUT_HIDE_KEYS: frozenset[str] = frozenset(
    {
        "plan",
        "selected",
        "diagnostics",
        "reliable_rank1",
        "levels",
        "global_trace_store_path",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v5.0 avatar bootstrap CLI")
    parser.add_argument("--game-id", required=True, help=f"Supported: {', '.join(SUPPORTED_GAMES)}")
    parser.add_argument("--output-dir", default=None, help="Output root (defaults under runs_v5_0)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-montage", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-debug-all", action="store_true")
    parser.add_argument("--render-terminal", action="store_true")
    parser.add_argument("--multi-reset", action="store_true")
    parser.add_argument("--episode-count", type=int, default=1)
    parser.add_argument("--full-analysis", action="store_true")
    parser.add_argument("--interpret-hud-target", action="store_true")
    parser.add_argument("--solve", action="store_true")
    parser.add_argument("--adaptive-solve", action="store_true")
    parser.add_argument("--mechanic-aware-solve", action="store_true")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--level-id", default=None)
    parser.add_argument("--all-levels", action="store_true")
    parser.add_argument("--replay-solution", action="store_true")
    parser.add_argument("--solution-file", default=None)
    parser.add_argument("--campaign-solve", action="store_true")
    parser.add_argument("--use-solutions", "--use-solution", dest="use_solutions", action="store_true")
    parser.add_argument("--optimize-traces", action="store_true")
    parser.add_argument("--replay-trace", action="store_true")
    parser.add_argument("--trace-file", default=None)
    parser.add_argument("--trace-analysis", action="store_true")
    parser.add_argument("--discover-pois", action="store_true")
    parser.add_argument("--run-contact-experiments", action="store_true")
    parser.add_argument("--detect-hud", action="store_true")
    return parser


def _infer_game_root(
    *,
    game_id: str,
    output_dir: str | None,
    artifact_paths: dict[str, str] | None = None,
) -> Path | None:
    if output_dir:
        return Path(output_dir) / game_id
    if not artifact_paths:
        return None
    for value in artifact_paths.values():
        path = Path(str(value)).resolve()
        parts = path.parts
        if game_id in parts:
            idx = parts.index(game_id)
            return Path(*parts[: idx + 1])
    return None


def _maybe_build_video_artifacts(
    *,
    enabled: bool,
    debug_all: bool = False,
    game_id: str,
    output_dir: str | None,
    artifact_paths: dict[str, str] | None = None,
) -> dict[str, object]:
    if not enabled:
        return {}
    base_output_dir = str(output_dir) if output_dir else "runs_v5_0"
    root = _infer_game_root(game_id=game_id, output_dir=None, artifact_paths=artifact_paths)
    if root is None:
        root = Path(get_game_root_output_dir(base_output_dir, game_id))
    try:
        video = build_final_game_video(game_root_dir=root, fps=2)
        payload = {
            "video_frames_dir": str(root / "video_frames"),
            "final_video_path": video.get("video_path"),
            "final_video_failure_reason": video.get("failure_reason"),
            "final_video_frame_count": int(video.get("frame_count", 0)),
            "final_video_explanation": (
                "no_frame_bearing_artifacts_found_under_game_root"
                if int(video.get("frame_count", 0)) == 0 and str(video.get("failure_reason")) == "no_renderable_frames"
                else None
            ),
        }
        if debug_all:
            payload["video_debug_root_dir"] = str(root)
        return payload
    except Exception as exc:
        payload = {
            "video_frames_dir": str(root / "video_frames"),
            "final_video_path": None,
            "final_video_failure_reason": str(exc),
            "final_video_frame_count": 0,
            "final_video_explanation": None,
        }
        if debug_all:
            payload["video_debug_root_dir"] = str(root)
        return payload


def _clear_game_output_dir(*, game_id: str, output_dir: str | None) -> Path:
    base = Path(output_dir) if output_dir else Path("runs_v5_0")
    game_root = base / str(game_id)
    if game_root.exists():
        for entry in game_root.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                try:
                    entry.unlink()
                except FileNotFoundError:
                    pass
    game_root.mkdir(parents=True, exist_ok=True)
    return game_root


def _sanitize_stdout_payload(payload):
    if isinstance(payload, dict):
        return {
            key: _sanitize_stdout_payload(value)
            for key, value in payload.items()
            if str(key) not in STDOUT_HIDE_KEYS
        }
    if isinstance(payload, list):
        return [_sanitize_stdout_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return tuple(_sanitize_stdout_payload(item) for item in payload)
    return payload


def _print_json_stdout(payload) -> None:
    print(json.dumps(_sanitize_stdout_payload(payload), indent=2))


def main() -> int:
    args = build_parser().parse_args()
    campaign_mode = bool(args.campaign_solve)
    cli_episode_count = int(args.episode_count)

    if args.run_contact_experiments and not args.multi_reset and not campaign_mode:
        raise SystemExit("--run-contact-experiments requires --multi-reset")
    if args.discover_pois and not args.multi_reset and not campaign_mode:
        raise SystemExit("--discover-pois requires --multi-reset")
    if args.detect_hud and not args.multi_reset and not campaign_mode:
        raise SystemExit("--detect-hud requires --multi-reset")
    if args.full_analysis and not args.multi_reset and not campaign_mode:
        raise SystemExit("--full-analysis requires --multi-reset")
    if args.interpret_hud_target and not args.full_analysis and not campaign_mode:
        raise SystemExit("--interpret-hud-target requires --full-analysis")
    if args.solve and not args.full_analysis and not campaign_mode:
        raise SystemExit("--solve requires --full-analysis")
    if args.adaptive_solve and not args.full_analysis and not campaign_mode:
        raise SystemExit("--adaptive-solve requires --full-analysis")
    if args.mechanic_aware_solve and not args.full_analysis and not campaign_mode:
        raise SystemExit("--mechanic-aware-solve requires --full-analysis")
    if args.all_levels and args.level_id:
        raise SystemExit("--all-levels cannot be combined with --level-id")
    if args.all_levels and not args.full_analysis:
        raise SystemExit("--all-levels requires --full-analysis")
    if args.replay_solution and not args.level_id:
        raise SystemExit("--replay-solution requires --level-id")
    if args.replay_solution and not args.solution_file:
        raise SystemExit("--replay-solution requires --solution-file")
    if args.replay_trace and not args.trace_file:
        raise SystemExit("--replay-trace requires --trace-file")
    if args.trace_analysis and not args.game_id:
        raise SystemExit("--trace-analysis requires --game-id")
    if args.use_solutions and not args.campaign_solve:
        raise SystemExit("--use-solutions is only valid with --campaign-solve")

    if not args.replay_solution and not args.replay_trace:
        _clear_game_output_dir(game_id=args.game_id, output_dir=args.output_dir)

    if args.campaign_solve:
        result = run_full_campaign_analysis(
            game_id=args.game_id,
            output_dir=args.output_dir,
            seed=args.seed,
            episode_count=cli_episode_count,
            probe_montage=bool(args.probe_montage),
            render_terminal=bool(args.render_terminal),
            max_steps=int(args.max_steps),
            use_solutions=bool(args.use_solutions),
        )
        stdout_result = dict(result)
        stdout_result.pop("global_action_trace", None)
        stdout_result["global_trace_store_path"] = get_global_trace_store_path()
        video_artifacts = _maybe_build_video_artifacts(
            enabled=bool(args.video),
            debug_all=bool(args.video_debug_all),
            game_id=args.game_id,
            output_dir=args.output_dir,
            artifact_paths=dict(result.get("artifact_paths", {})),
        )
        if video_artifacts:
            artifacts = dict(stdout_result.get("artifact_paths", {}))
            if video_artifacts.get("video_frames_dir") is not None:
                artifacts["video_frames_dir"] = str(video_artifacts.get("video_frames_dir"))
            if video_artifacts.get("final_video_path") is not None:
                artifacts["final_run.mp4"] = str(video_artifacts.get("final_video_path"))
            stdout_result["artifact_paths"] = artifacts
        _print_json_stdout(stdout_result)
        return 0 if bool(result.get("solved", False)) else 2

    if args.trace_analysis or args.optimize_traces:
        try:
            payload = run_trace_analysis_for_game(
                game_id=args.game_id,
                render_terminal=bool(args.render_terminal),
                output_dir=args.output_dir,
            )
            payload["global_trace_store_path"] = get_global_trace_store_path()
            _print_json_stdout(payload)
            return 0
        except Exception as exc:
            _print_json_stdout(
                {
                    "game_id": args.game_id,
                    "global_trace_store_path": get_global_trace_store_path(),
                    "solved_level_count": 0,
                    "analyzed_level_count": 0,
                    "reports": [],
                    "failure_reason": str(exc),
                }
            )
            return 2

    if args.replay_trace:
        trace_path = Path(str(args.trace_file))
        if not trace_path.exists():
            _print_json_stdout({"success": False, "divergence": True, "failure_reason": "trace_file_missing"})
            return 2
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        actions = payload.get("action_trace")
        if actions is None:
            actions_path = trace_path.with_name("saved_level_trace_actions.json")
            actions = json.loads(actions_path.read_text(encoding="utf-8")) if actions_path.exists() else []
        result = replay_saved_trace(
            game_id=args.game_id,
            action_trace=tuple(str(item) for item in actions),
            render_terminal=bool(args.render_terminal),
        )
        _print_json_stdout(result)
        return 0 if bool(result.get("success", False)) and not bool(result.get("divergence", False)) else 2

    if args.replay_solution:
        solution_path = Path(str(args.solution_file))
        payload = json.loads(solution_path.read_text(encoding="utf-8"))
        if "action_trace" not in payload:
            actions_path = solution_path.with_name("level_solution_actions.json")
            if actions_path.exists():
                payload["action_trace"] = json.loads(actions_path.read_text(encoding="utf-8"))
        result = replay_saved_level_solution(
            game_id=args.game_id,
            level_id=str(args.level_id),
            solution=payload,
            render_terminal=bool(args.render_terminal),
        )
        _print_json_stdout(result)
        return 0 if bool(result.get("solved", False)) else 2

    if args.all_levels:
        result = run_full_analysis_for_game_levels(
            game_id=args.game_id,
            level_ids=None,
            output_dir=args.output_dir,
            seed=args.seed,
            episode_count=args.episode_count,
            probe_montage=bool(args.probe_montage),
            render_terminal=bool(args.render_terminal),
            max_steps=int(args.max_steps),
        )
        _maybe_build_video_artifacts(
            enabled=bool(args.video),
            debug_all=bool(args.video_debug_all),
            game_id=args.game_id,
            output_dir=args.output_dir,
            artifact_paths=dict(result.get("artifact_paths", {})),
        )
        _print_json_stdout(result)
        solved = int(result.get("diagnostics", {}).get("solved_level_count", 0))
        return 0 if solved > 0 else 2

    if args.level_id and args.full_analysis:
        result = run_full_analysis_for_level(
            game_id=args.game_id,
            level_id=str(args.level_id),
            output_dir=args.output_dir,
            seed=args.seed,
            episode_count=args.episode_count,
            probe_montage=bool(args.probe_montage),
            render_terminal=bool(args.render_terminal),
            max_steps=int(args.max_steps),
        )
        _maybe_build_video_artifacts(
            enabled=bool(args.video),
            debug_all=bool(args.video_debug_all),
            game_id=args.game_id,
            output_dir=args.output_dir,
            artifact_paths=dict(result.get("artifact_paths", {})),
        )
        _print_json_stdout(result)
        return 0 if bool(result.get("solved", False)) else 2

    if args.full_analysis:
        if args.adaptive_solve:
            result = run_full_bootstrap_analysis_with_adaptive_solve(
                game_id=args.game_id,
                output_dir=args.output_dir,
                seed=args.seed,
                episode_count=args.episode_count,
                probe_montage=bool(args.probe_montage),
                render_terminal=bool(args.render_terminal),
                max_steps=int(args.max_steps),
            )
            payload = {
                "artifact_paths": {
                    key: value
                    for key, value in dict(result.get("artifact_paths", {})).items()
                    if key in FULL_ANALYSIS_STDOUT_ARTIFACT_KEYS
                },
                "phase_status": dict(result.get("phase_status", {})),
            }
            _maybe_build_video_artifacts(
                enabled=bool(args.video),
                debug_all=bool(args.video_debug_all),
                game_id=args.game_id,
                output_dir=args.output_dir,
                artifact_paths=dict(result.get("artifact_paths", {})),
            )
            _print_json_stdout(payload)
            return 0 if bool(result.get("adaptive_solve", {}).get("solved", False)) else 2

        if args.mechanic_aware_solve:
            result = run_full_bootstrap_analysis_with_mechanics(
                game_id=args.game_id,
                output_dir=args.output_dir,
                seed=args.seed,
                episode_count=args.episode_count,
                probe_montage=bool(args.probe_montage),
                render_terminal=bool(args.render_terminal),
                max_steps=int(args.max_steps),
            )
            payload = {
                "artifact_paths": {
                    key: value
                    for key, value in dict(result.get("artifact_paths", {})).items()
                    if key in FULL_ANALYSIS_STDOUT_ARTIFACT_KEYS
                },
                "phase_status": dict(result.get("phase_status", {})),
            }
            _maybe_build_video_artifacts(
                enabled=bool(args.video),
                debug_all=bool(args.video_debug_all),
                game_id=args.game_id,
                output_dir=args.output_dir,
                artifact_paths=dict(result.get("artifact_paths", {})),
            )
            _print_json_stdout(payload)
            return 0 if bool(result.get("solve", {}).get("solved", False)) else 2

        if args.solve:
            result = run_full_bootstrap_analysis_with_solve(
                game_id=args.game_id,
                output_dir=args.output_dir,
                seed=args.seed,
                episode_count=args.episode_count,
                probe_montage=bool(args.probe_montage),
                render_terminal=bool(args.render_terminal),
                max_steps=int(args.max_steps),
            )
            payload = {
                "artifact_paths": {
                    key: value
                    for key, value in dict(result.get("artifact_paths", {})).items()
                    if key in FULL_ANALYSIS_STDOUT_ARTIFACT_KEYS
                },
                "phase_status": dict(result.get("phase_status", {})),
            }
            _maybe_build_video_artifacts(
                enabled=bool(args.video),
                debug_all=bool(args.video_debug_all),
                game_id=args.game_id,
                output_dir=args.output_dir,
                artifact_paths=dict(result.get("artifact_paths", {})),
            )
            _print_json_stdout(payload)
            return 0 if bool(result.get("solve", {}).get("solved", False)) else 2

        if args.interpret_hud_target:
            result = run_full_bootstrap_analysis_with_hud_targeting(
                game_id=args.game_id,
                output_dir=args.output_dir,
                seed=args.seed,
                episode_count=args.episode_count,
                probe_montage=bool(args.probe_montage),
                render_terminal=bool(args.render_terminal),
            )
            payload = {
                "artifact_paths": {
                    key: value
                    for key, value in dict(result.get("artifact_paths", {})).items()
                    if key in FULL_ANALYSIS_STDOUT_ARTIFACT_KEYS
                },
                "phase_status": dict(result.get("phase_status", {})),
            }
            _maybe_build_video_artifacts(
                enabled=bool(args.video),
                debug_all=bool(args.video_debug_all),
                game_id=args.game_id,
                output_dir=args.output_dir,
                artifact_paths=dict(result.get("artifact_paths", {})),
            )
            _print_json_stdout(payload)
            hud_targeting = dict(result.get("hud_targeting", {}))
            selected_poi_id = hud_targeting.get("selected_poi_id")
            ambiguous = bool(hud_targeting.get("ambiguous", True))
            return 0 if selected_poi_id is not None and not ambiguous else 2

        result = run_full_bootstrap_analysis(
            game_id=args.game_id,
            output_dir=args.output_dir,
            seed=args.seed,
            episode_count=args.episode_count,
            probe_montage=bool(args.probe_montage),
            render_terminal=bool(args.render_terminal),
        )
        payload = {
            "artifact_paths": {
                key: value
                for key, value in dict(result.get("artifact_paths", {})).items()
                if key in FULL_ANALYSIS_STDOUT_ARTIFACT_KEYS
            },
            "phase_status": dict(result.get("phase_status", {})),
        }
        _maybe_build_video_artifacts(
            enabled=bool(args.video),
            debug_all=bool(args.video_debug_all),
            game_id=args.game_id,
            output_dir=args.output_dir,
            artifact_paths=dict(result.get("artifact_paths", {})),
        )
        _print_json_stdout(payload)
        return 0 if result.get("phase_status", {}).get("avatar") == "ok" else 2

    if args.run_contact_experiments:
        result = run_avatar_poi_contact_bootstrap_multi_reset(
            game_id=args.game_id,
            output_dir=args.output_dir,
            seed=args.seed,
            episode_count=args.episode_count,
            probe_montage=bool(args.probe_montage),
            render_terminal=bool(args.render_terminal),
        )
        _maybe_build_video_artifacts(
            enabled=bool(args.video),
            debug_all=bool(args.video_debug_all),
            game_id=args.game_id,
            output_dir=args.output_dir,
            artifact_paths=dict(result.get("artifact_paths", {})),
        )
        _print_json_stdout(result)
        stable_avatar = result.get("selected_avatar", {}).get("failure_reason") is None
        contact_ok = bool(result.get("artifact_paths", {}).get("contact_experiments_summary.json"))
        return 0 if stable_avatar and contact_ok else 2

    if args.discover_pois:
        result = run_avatar_and_poi_bootstrap_multi_reset(
            game_id=args.game_id,
            output_dir=args.output_dir,
            seed=args.seed,
            episode_count=args.episode_count,
            probe_montage=bool(args.probe_montage),
            render_terminal=bool(args.render_terminal),
        )
        _maybe_build_video_artifacts(
            enabled=bool(args.video),
            debug_all=bool(args.video_debug_all),
            game_id=args.game_id,
            output_dir=args.output_dir,
            artifact_paths=dict(result.get("artifact_paths", {})),
        )
        _print_json_stdout(result)
        stable_avatar = result.get("selected_avatar", {}).get("failure_reason") is None
        poi_ok = bool(result.get("artifact_paths", {}).get("poi_summary.json"))
        return 0 if stable_avatar and poi_ok else 2

    if args.detect_hud:
        result = run_avatar_poi_hud_bootstrap_multi_reset(
            game_id=args.game_id,
            output_dir=args.output_dir,
            seed=args.seed,
            episode_count=args.episode_count,
            probe_montage=bool(args.probe_montage),
            render_terminal=bool(args.render_terminal),
        )
        _maybe_build_video_artifacts(
            enabled=bool(args.video),
            debug_all=bool(args.video_debug_all),
            game_id=args.game_id,
            output_dir=args.output_dir,
            artifact_paths=dict(result.get("artifact_paths", {})),
        )
        _print_json_stdout(result)
        hud_ok = bool(result.get("artifact_paths", {}).get("hud_summary.json"))
        hud_failed = result.get("hud_diagnostics", {}).get("failure_reason") is not None
        return 0 if hud_ok and not hud_failed else 2

    if args.multi_reset:
        result = run_avatar_bootstrap_multi_reset(
            game_id=args.game_id,
            output_dir=args.output_dir,
            seed=args.seed,
            episode_count=args.episode_count,
            probe_montage=bool(args.probe_montage),
            render_terminal=bool(args.render_terminal),
        )
        _maybe_build_video_artifacts(
            enabled=bool(args.video),
            debug_all=bool(args.video_debug_all),
            game_id=args.game_id,
            output_dir=args.output_dir,
            artifact_paths=dict(result.get("artifact_paths", {})),
        )
        _print_json_stdout(result)
        return 0 if result.get("stable_avatar_found", False) else 2

    result = run_avatar_bootstrap(
        game_id=args.game_id,
        output_dir=args.output_dir,
        seed=args.seed,
        probe_montage=bool(args.probe_montage),
        render_terminal=bool(args.render_terminal),
    )
    _maybe_build_video_artifacts(
        enabled=bool(args.video),
        debug_all=bool(args.video_debug_all),
        game_id=args.game_id,
        output_dir=args.output_dir,
        artifact_paths=dict(result.get("artifact_paths", {})),
    )
    _print_json_stdout(result)
    if not result.get("reliable_rank1", False):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
