from __future__ import annotations

import argparse
import json
import os

from codex_baseline_v2.analyst.analyst import analyze_episodes
from codex_baseline_v2.adapters.trajectory_import import import_legacy_from_path
from codex_baseline_v2.memory.store import append_round_report, load_blackboard, save_blackboard
from codex_baseline_v2.shared.config import load_config
from codex_baseline_v2.shared.storage import StoragePathsV2
from codex_baseline_v2.trajectory_analysis.analyzer import analyze_trajectories
from codex_baseline_v2.trajectory_analysis.avatar_hypothesis_debug import export_avatar_debug


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze stored V2 trajectories")
    parser.add_argument("--config", required=True)
    parser.add_argument("--round-id", type=int, default=0)
    parser.add_argument("--trajectory-path", required=True)
    parser.add_argument("--prior-blackboard", default=None)
    parser.add_argument("--debug-export", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg_payload = json.load(handle)
    cfg = load_config(cfg_payload)

    storage = StoragePathsV2(cfg.memory.storage_dir)
    storage.ensure_round_dirs(cfg.game_id, args.round_id)

    episodes = import_legacy_from_path(args.trajectory_path, game_id_override=cfg.game_id)
    analyzed = analyze_episodes(episodes, cfg.analyst)
    prior = None
    if args.prior_blackboard:
        with open(args.prior_blackboard, "r", encoding="utf-8") as handle:
            prior = BlackboardStateV2.from_dict(json.load(handle))
    elif args.round_id > 0:
        prior = load_blackboard(storage, cfg.game_id)
    blackboard = analyze_trajectories(analyzed, cfg.trajectory_analysis, round_id=args.round_id, prior_blackboard=prior)
    save_blackboard(cfg.memory, storage, blackboard)

    report = {"round_id": args.round_id, "game_id": cfg.game_id, "poi_count": len(blackboard.poi_table)}
    append_round_report(storage, cfg.game_id, args.round_id, report)
    if args.debug_export:
        paths = storage.ensure_round_dirs(cfg.game_id, args.round_id)
        debug_summary = _build_debug_summary(analyzed, blackboard)
        with open(os.path.join(paths["exports"], "debug_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(debug_summary, handle, sort_keys=True)


if __name__ == "__main__":
    main()


def _build_debug_summary(episodes, blackboard):
    state_hashes = set()
    invalid_state_count = 0
    states_observed = 0
    for ep in episodes:
        for step in ep.steps:
            for state_hash in (step.pre_state_hash, step.post_state_hash):
                if state_hash:
                    state_hashes.add(state_hash)
                    states_observed += 1
                else:
                    invalid_state_count += 1
    reachability_reasons = {}
    for record in blackboard.reachability_table:
        reason = record.reason_code or "unknown"
        reachability_reasons[reason] = reachability_reasons.get(reason, 0) + 1
    invalid_target_links = sum(1 for c in blackboard.consequence_table if c.consequence_class == "invalid_target_link")
    poi_rejections = {}
    poi_demotions = {}
    for poi in blackboard.poi_table:
        for reason in poi.rejection_reasons:
            poi_rejections[reason] = poi_rejections.get(reason, 0) + 1
        for reason in poi.demotion_reasons:
            poi_demotions[reason] = poi_demotions.get(reason, 0) + 1
    instruction_counts = {"targeted": 0, "untargeted": 0}
    missing_linkages = 0
    for ep in episodes:
        mode = ep.metadata.get("mode", "unknown") if isinstance(ep.metadata, dict) else "unknown"
        if mode in {"instructed_execution", "poi_approach", "poi_interaction", "exploit_route"}:
            instruction_counts["targeted"] += 1
        else:
            instruction_counts["untargeted"] += 1
        for step in ep.steps:
            if step.target_poi_id and not step.instruction_id:
                missing_linkages += 1
    return {
        "state_identity_summary": {
            "states_observed": states_observed,
            "unique_states": len(state_hashes),
            "invalid_state_count": invalid_state_count,
        },
        "avatar_candidate_summary": export_avatar_debug(episodes),
        "reachability_reason_histogram": reachability_reasons,
        "controller_target_selection_summary": instruction_counts,
        "executor_linkage_summary": {"missing_linkages": missing_linkages},
        "poi_rejection_summary": {"rejections": poi_rejections, "demotions": poi_demotions},
        "invalid_target_link_count": invalid_target_links,
    }
