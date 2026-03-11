from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from codex_baseline_v2.adapters.trajectory_import import import_legacy_from_path
from codex_baseline_v2.analyst.analyst import analyze_episodes
from codex_baseline_v2.controller.controller import select_instruction
from codex_baseline_v2.executor.executor import execute_instruction_offline
from dataclasses import replace

from codex_baseline_v2.memory.store import append_round_report, load_blackboard, save_blackboard
from codex_baseline_v2.shared.config import V2Config, load_config
from codex_baseline_v2.shared.logging_utils import log_event
from codex_baseline_v2.shared.metrics import compute_round_metrics
from codex_baseline_v2.shared.schemas import (
    BlackboardStateV2,
    CandidatePOIV2,
    ConsequenceRecordV2,
    ObjectRecordV2,
    ReachabilityRecordV2,
    SCHEMA_VERSION,
)
from codex_baseline_v2.shared.storage import StoragePathsV2
from codex_baseline_v2.shared.utils import BBox
from codex_baseline_v2.trajectory_analysis.analyzer import analyze_trajectories


def _load_config(path: str) -> V2Config:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return load_config(payload)


def _save_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)


def _save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _parse_bbox(payload: Dict[str, Any]) -> BBox:
    return BBox(int(payload["x1"]), int(payload["y1"]), int(payload["x2"]), int(payload["y2"]))


def _parse_poi(payload: Dict[str, Any]) -> CandidatePOIV2:
    return CandidatePOIV2(
        schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        poi_id=str(payload["poi_id"]),
        game_id=str(payload["game_id"]),
        source_type=str(payload["source_type"]),
        bbox=_parse_bbox(payload["bbox"]),
        centroid=tuple(payload["centroid"]),
        object_class=str(payload["object_class"]),
        reachable_now=str(payload.get("reachable_now", "uncertain")),
        confidence=float(payload.get("confidence", 0.0)),
        expected_information_gain=float(payload.get("expected_information_gain", 0.0)),
        expected_interaction_type=str(payload.get("expected_interaction_type", "unknown")),
        evidence_count=int(payload.get("evidence_count", 0)),
        observation_count=int(payload.get("observation_count", payload.get("evidence_count", 0))),
        first_seen_episode=payload.get("first_seen_episode"),
        last_seen_episode=payload.get("last_seen_episode"),
        last_seen_step=payload.get("last_seen_step"),
        first_seen_ref=payload.get("first_seen_ref"),
        last_seen_ref=payload.get("last_seen_ref"),
        type_confidence=float(payload.get("type_confidence", 0.5)),
        utility_confidence=float(payload.get("utility_confidence", 0.5)),
        rejection_reasons=list(payload.get("rejection_reasons", [])),
        demotion_reasons=list(payload.get("demotion_reasons", [])),
    )


def _parse_reachability(payload: Dict[str, Any]) -> ReachabilityRecordV2:
    return ReachabilityRecordV2(
        schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        game_id=str(payload["game_id"]),
        poi_id=str(payload["poi_id"]),
        status=str(payload.get("status", "uncertain")),
        confidence=float(payload.get("confidence", 0.0)),
        distance_estimate=payload.get("distance_estimate"),
        evidence_refs=list(payload.get("evidence_refs", [])),
        reason_code=payload.get("reason_code"),
    )


def _parse_consequence(payload: Dict[str, Any]) -> ConsequenceRecordV2:
    return ConsequenceRecordV2(
        schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        game_id=str(payload["game_id"]),
        poi_id=str(payload["poi_id"]),
        round_id=int(payload.get("round_id", 0)),
        episode_id=str(payload.get("episode_id", "")),
        instruction_id=payload.get("instruction_id"),
        target_poi_id=payload.get("target_poi_id"),
        distance_decreased=bool(payload.get("distance_decreased", False)),
        reached=bool(payload.get("reached", False)),
        contact=bool(payload.get("contact", False)),
        local_change_magnitude=float(payload.get("local_change_magnitude", 0.0)),
        global_change_magnitude=float(payload.get("global_change_magnitude", 0.0)),
        reward_delta=payload.get("reward_delta"),
        terminal_flag_changed=bool(payload.get("terminal_flag_changed", False)),
        object_change_summary=str(payload.get("object_change_summary", "")),
        followup_poi_ids=list(payload.get("followup_poi_ids", [])),
        consequence_class=str(payload.get("consequence_class", "ambiguous")),
    )


def _parse_object(payload: Dict[str, Any]) -> ObjectRecordV2:
    return ObjectRecordV2(
        schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        object_id=str(payload["object_id"]),
        game_id=str(payload["game_id"]),
        episode_id=str(payload["episode_id"]),
        bbox=_parse_bbox(payload["bbox"]),
        centroid=tuple(payload["centroid"]),
        color=int(payload["color"]),
        area=int(payload["area"]),
        aspect_ratio=float(payload["aspect_ratio"]),
        object_class=str(payload.get("object_class", "unknown")),
        confidence=float(payload.get("confidence", 0.0)),
        evidence_refs=list(payload.get("evidence_refs", [])),
        first_seen_ref=payload.get("first_seen_ref"),
        last_seen_ref=payload.get("last_seen_ref"),
    )


def _blackboard_from_dict(payload: Dict[str, Any]) -> BlackboardStateV2:
    return BlackboardStateV2.from_dict(payload)


def cmd_init(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    storage = StoragePathsV2(cfg.memory.storage_dir)
    storage.ensure_round_dirs(cfg.game_id, 0)
    log_event(cfg.logging.log_dir, "v2_init", {"game_id": cfg.game_id})


def cmd_import_analyze(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    storage = StoragePathsV2(cfg.memory.storage_dir)
    paths = storage.ensure_round_dirs(cfg.game_id, args.round_id)

    trajectory_path = args.trajectory_path or cfg.dataset_or_rollout_source.trajectory_path
    if not trajectory_path:
        raise SystemExit("trajectory_path is required (flag or config.dataset_or_rollout_source.trajectory_path).")
    episodes = import_legacy_from_path(trajectory_path, game_id_override=cfg.game_id)
    analyzed = analyze_episodes(episodes, cfg.analyst)
    prior = load_blackboard(storage, cfg.game_id) if args.round_id > 0 else None
    blackboard = analyze_trajectories(analyzed, cfg.trajectory_analysis, round_id=args.round_id, prior_blackboard=prior)

    _save_jsonl(os.path.join(paths["normalized_trajectories"], "episodes.jsonl"), [ep.to_dict() for ep in analyzed])
    _save_json(os.path.join(paths["blackboard_snapshots"], "blackboard.json"), blackboard.to_dict())
    save_blackboard(cfg.memory, storage, blackboard)

    report = {
        "round_id": args.round_id,
        "game_id": cfg.game_id,
        "poi_count": len(blackboard.poi_table),
        "reachability_count": len(blackboard.reachability_table),
    }
    append_round_report(storage, cfg.game_id, args.round_id, report)
    log_event(cfg.logging.log_dir, "v2_import_analyze", report)


def cmd_directed_round(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    storage = StoragePathsV2(cfg.memory.storage_dir)
    paths = storage.ensure_round_dirs(cfg.game_id, args.round_id)
    prior = load_blackboard(storage, cfg.game_id)
    if prior is None:
        raise SystemExit("No blackboard state found. Run import/analyze first.")
    blackboard = prior

    instruction = select_instruction(blackboard, cfg.controller, cfg.scoring, args.round_id)
    trajectory_path = args.trajectory_path or cfg.dataset_or_rollout_source.trajectory_path
    if not trajectory_path:
        raise SystemExit("trajectory_path is required (flag or config.dataset_or_rollout_source.trajectory_path).")
    episodes = import_legacy_from_path(trajectory_path, game_id_override=cfg.game_id)
    analyzed = analyze_episodes(episodes, cfg.analyst)
    outcome = execute_instruction_offline(analyzed, instruction, cfg.executor)

    updated_blackboard = analyze_trajectories(analyzed, cfg.trajectory_analysis, round_id=args.round_id, prior_blackboard=blackboard)
    updated_blackboard = replace(
        updated_blackboard,
        consequence_table=list(updated_blackboard.consequence_table) + list(outcome.consequence_records),
    )
    _save_jsonl(os.path.join(paths["normalized_trajectories"], "episodes.jsonl"), [ep.to_dict() for ep in analyzed])
    _save_json(os.path.join(paths["controller_decisions"], "instruction.json"), instruction.to_dict())
    _save_json(os.path.join(paths["executor_outcomes"], "outcome.json"), outcome.to_dict())
    _save_json(os.path.join(paths["blackboard_snapshots"], "blackboard.json"), updated_blackboard.to_dict())
    save_blackboard(cfg.memory, storage, updated_blackboard)

    reach_lookup = {r.poi_id: r.status for r in updated_blackboard.reachability_table}
    metrics = compute_round_metrics(
        analyzed,
        updated_blackboard.poi_table,
        reach_lookup,
        updated_blackboard.consequence_table,
        [instruction.mode],
        len(updated_blackboard.avatar_hypotheses),
        outcome.target_progress,
        blackboard=updated_blackboard,
        executor_outcomes=[outcome],
    )
    report = {
        "round_id": args.round_id,
        "game_id": cfg.game_id,
        "instruction": instruction.to_dict(),
        "metrics": metrics.to_dict(),
    }
    append_round_report(storage, cfg.game_id, args.round_id, report)
    log_event(cfg.logging.log_dir, "v2_directed_round", {"round_id": args.round_id})


def cmd_loop(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    storage = StoragePathsV2(cfg.memory.storage_dir)
    trajectory_path = args.trajectory_path or cfg.dataset_or_rollout_source.trajectory_path
    if not trajectory_path:
        raise SystemExit("trajectory_path is required (flag or config.dataset_or_rollout_source.trajectory_path).")
    for round_id in range(cfg.rounds):
        paths = storage.ensure_round_dirs(cfg.game_id, round_id)
        episodes = import_legacy_from_path(trajectory_path, game_id_override=cfg.game_id)
        analyzed = analyze_episodes(episodes, cfg.analyst)
        if round_id == 0:
            blackboard = analyze_trajectories(analyzed, cfg.trajectory_analysis, round_id=round_id)
            save_blackboard(cfg.memory, storage, blackboard)
            _save_jsonl(os.path.join(paths["normalized_trajectories"], "episodes.jsonl"), [ep.to_dict() for ep in analyzed])
            _save_json(os.path.join(paths["blackboard_snapshots"], "blackboard.json"), blackboard.to_dict())
        else:
            prior = load_blackboard(storage, cfg.game_id)
            if prior is None:
                raise SystemExit("Missing blackboard state")
            blackboard = prior
            instruction = select_instruction(blackboard, cfg.controller, cfg.scoring, round_id)
            outcome = execute_instruction_offline(analyzed, instruction, cfg.executor)
            blackboard = analyze_trajectories(analyzed, cfg.trajectory_analysis, round_id=round_id, prior_blackboard=blackboard)
            blackboard = replace(
                blackboard,
                consequence_table=list(blackboard.consequence_table) + list(outcome.consequence_records),
            )
            _save_json(os.path.join(paths["controller_decisions"], "instruction.json"), instruction.to_dict())
            _save_json(os.path.join(paths["executor_outcomes"], "outcome.json"), outcome.to_dict())
            _save_json(os.path.join(paths["blackboard_snapshots"], "blackboard.json"), blackboard.to_dict())
            save_blackboard(cfg.memory, storage, blackboard)
        report = {"round_id": round_id, "game_id": cfg.game_id, "poi_count": len(blackboard.poi_table)}
        append_round_report(storage, cfg.game_id, round_id, report)
        log_event(cfg.logging.log_dir, "v2_round_complete", report)


def cmd_print_blackboard(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    storage = StoragePathsV2(cfg.memory.storage_dir)
    payload = load_blackboard(storage, cfg.game_id)
    if payload is None:
        raise SystemExit("No blackboard state found.")
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_export_reports(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    storage = StoragePathsV2(cfg.memory.storage_dir)
    path = os.path.join(storage.category_path(cfg.game_id, args.round_id, "round_reports"), "round_report.json")
    if not os.path.exists(path):
        raise SystemExit("No round report found.")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    out_path = args.out_path or os.path.join(storage.category_path(cfg.game_id, args.round_id, "exports"), "round_report.json")
    _save_json(out_path, payload)


def cmd_run_autonomous_game(args: argparse.Namespace) -> None:
    import sys
    from codex_baseline_v2.cli import run_autonomous_game

    argv = [
        "run_autonomous_game",
        "--config",
        args.config,
        "--game-id",
        args.game_id,
        "--storage-root",
        args.storage_root,
        "--env-factory",
        args.env_factory,
        "--env-id",
        args.env_id,
        "--env-root",
        args.env_root,
        "--workers",
        str(args.workers),
    ]
    if getattr(args, "render_terminal", False):
        argv.append("--render-terminal")
    prev = sys.argv[:]
    try:
        sys.argv = argv
        run_autonomous_game.main()
    finally:
        sys.argv = prev


def cmd_collect_trajectories(args: argparse.Namespace) -> None:
    import sys
    from codex_baseline_v2.cli import collect_trajectories

    argv = [
        "collect_trajectories",
        "--config",
        args.config,
        "--env-factory",
        args.env_factory,
        "--mode",
        args.mode,
        "--round-id",
        str(args.round_id),
        "--env-id",
        args.env_id,
        "--env-root",
        args.env_root,
        "--workers",
        str(args.workers),
    ]
    prev = sys.argv[:]
    try:
        sys.argv = argv
        collect_trajectories.main()
    finally:
        sys.argv = prev


def cmd_analyze_trajectories(args: argparse.Namespace) -> None:
    import sys
    from codex_baseline_v2.cli import analyze_trajectories

    argv = [
        "analyze_trajectories",
        "--config",
        args.config,
        "--trajectory-path",
        args.trajectory_path,
        "--round-id",
        str(args.round_id),
    ]
    if getattr(args, "prior_blackboard", None):
        argv.extend(["--prior-blackboard", args.prior_blackboard])
    prev = sys.argv[:]
    try:
        sys.argv = argv
        analyze_trajectories.main()
    finally:
        sys.argv = prev


def cmd_analyze_causal_world(args: argparse.Namespace) -> None:
    import sys
    from codex_baseline_v2.cli import analyze_causal_world

    argv = [
        "analyze_causal_world",
        "--config",
        args.config,
    ]
    prev = sys.argv[:]
    try:
        sys.argv = argv
        analyze_causal_world.main()
    finally:
        sys.argv = prev


def cmd_plan_with_options(args: argparse.Namespace) -> None:
    import sys
    from codex_baseline_v2.cli import plan_with_options

    argv = [
        "plan_with_options",
        "--config",
        args.config,
    ]
    prev = sys.argv[:]
    try:
        sys.argv = argv
        plan_with_options.main()
    finally:
        sys.argv = prev


def cmd_build_visual_prototypes(args: argparse.Namespace) -> None:
    import sys
    from codex_baseline_v2.cli import build_visual_prototypes

    argv = [
        "build_visual_prototypes",
        "--config",
        args.config,
    ]
    prev = sys.argv[:]
    try:
        sys.argv = argv
        build_visual_prototypes.main()
    finally:
        sys.argv = prev


def cmd_train_rankers(args: argparse.Namespace) -> None:
    import sys
    from codex_baseline_v2.cli import train_rankers

    argv = [
        "train_rankers",
        "--config",
        args.config,
    ]
    prev = sys.argv[:]
    try:
        sys.argv = argv
        train_rankers.main()
    finally:
        sys.argv = prev


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Codex Baseline V2 CLI")
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    init_cmd = sub.add_parser("init")
    init_cmd.set_defaults(func=cmd_init)

    import_cmd = sub.add_parser("import_analyze")
    import_cmd.add_argument("--trajectory-path", default=None)
    import_cmd.add_argument("--round-id", type=int, default=0)
    import_cmd.set_defaults(func=cmd_import_analyze)

    directed_cmd = sub.add_parser("directed_round")
    directed_cmd.add_argument("--trajectory-path", default=None)
    directed_cmd.add_argument("--round-id", type=int, default=1)
    directed_cmd.set_defaults(func=cmd_directed_round)

    loop_cmd = sub.add_parser("loop")
    loop_cmd.add_argument("--trajectory-path", default=None)
    loop_cmd.set_defaults(func=cmd_loop)

    print_cmd = sub.add_parser("print_blackboard")
    print_cmd.set_defaults(func=cmd_print_blackboard)

    export_cmd = sub.add_parser("export_reports")
    export_cmd.add_argument("--round-id", type=int, default=0)
    export_cmd.add_argument("--out-path", default=None)
    export_cmd.set_defaults(func=cmd_export_reports)

    autonomous_cmd = sub.add_parser("run_autonomous_game")
    autonomous_cmd.add_argument("--game-id", default=None)
    autonomous_cmd.add_argument("--storage-root", default=None)
    autonomous_cmd.add_argument("--env-factory", default=None)
    autonomous_cmd.add_argument("--env-id", default=None)
    autonomous_cmd.add_argument("--env-root", default=None)
    autonomous_cmd.add_argument("--workers", type=int, default=1)
    autonomous_cmd.add_argument("--render-terminal", action="store_true")
    autonomous_cmd.set_defaults(func=cmd_run_autonomous_game)

    collect_cmd = sub.add_parser("collect_trajectories")
    collect_cmd.add_argument("--env-factory", required=True)
    collect_cmd.add_argument("--mode", required=True, choices=["random_probe", "unguided_probe", "instructed_execution"])
    collect_cmd.add_argument("--round-id", type=int, default=0)
    collect_cmd.add_argument("--env-id", default=None)
    collect_cmd.add_argument("--env-root", default=None)
    collect_cmd.add_argument("--workers", type=int, default=1)
    collect_cmd.set_defaults(func=cmd_collect_trajectories)

    analyze_cmd = sub.add_parser("analyze_trajectories")
    analyze_cmd.add_argument("--trajectory-path", required=True)
    analyze_cmd.add_argument("--round-id", type=int, default=0)
    analyze_cmd.add_argument("--prior-blackboard", default=None)
    analyze_cmd.set_defaults(func=cmd_analyze_trajectories)

    causal_cmd = sub.add_parser("analyze_causal_world")
    causal_cmd.set_defaults(func=cmd_analyze_causal_world)

    plan_cmd = sub.add_parser("plan_with_options")
    plan_cmd.set_defaults(func=cmd_plan_with_options)

    vision_cmd = sub.add_parser("build_visual_prototypes")
    vision_cmd.set_defaults(func=cmd_build_visual_prototypes)

    train_cmd = sub.add_parser("train_rankers")
    train_cmd.set_defaults(func=cmd_train_rankers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
