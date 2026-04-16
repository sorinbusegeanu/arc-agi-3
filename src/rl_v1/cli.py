from __future__ import annotations

import argparse
import json

from rl_v1.eval.eval_main import eval_main
from rl_v1.training.train_main import train_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="rl_v1 V1 entrypoint")
    parser.add_argument("--config", default=None)
    parser.add_argument("--preset", default="v2", help="Optional preset override. Defaults to v2.")
    parser.add_argument(
        "--mode",
        choices=("train", "eval", "pretrain_world", "train_rl", "eval_policy"),
        default="train_rl",
    )
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--world-updates", type=int, default=None)
    parser.add_argument("--world-batch-size", type=int, default=None)
    parser.add_argument("--world-unroll-length", type=int, default=None)
    parser.add_argument("--world-eval-every", type=int, default=None)
    parser.add_argument("--world-save-every", type=int, default=None)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--freeze-recurrent", action="store_true")
    parser.add_argument("--init-from-checkpoint", default=None)
    parser.add_argument("--log-gameplay-metrics-during-pretrain", action="store_true")
    parser.add_argument("--game", action="append", dest="games", default=None, help="Override game list. Repeat for multiple games.")
    parser.add_argument("--workers", type=int, default=None, help="Deprecated alias: overrides runtime.rollout_processes.")
    parser.add_argument("--rollout-processes", type=int, default=None, help="Override runtime.rollout_processes.")
    parser.add_argument("--accelerator", default=None, help="Override runtime.accelerator.")
    parser.add_argument("--devices", type=int, default=None, help="Override runtime.devices.")
    parser.add_argument("--execution-mode", default=None, help="Override env.execution_mode.")
    parser.add_argument("--render", default=None, help="Override env.render_mode.")
    parser.add_argument("--video", action="store_true", help="Enable environment recording output.")
    parser.add_argument("--debug", default=None, help="Write per-step reward debug lines to this file path (e.g. runs/debug.log).")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint.restore_path.")
    parser.add_argument("--eval-kind", choices=("policy", "world"), default=None)
    parser.add_argument("--per-game", action="store_true")
    parser.add_argument("--world-metrics-only", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--acting-mode", choices=("policy_only", "planner_eval_only", "planner_act"), default=None)
    parser.add_argument("--deterministic-eval", type=int, choices=(0, 1), default=None)
    parser.add_argument("--compare-policy-vs-configured", type=int, choices=(0, 1), default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    mode = _normalize_mode(args.mode)
    shared = {
        "config_path": args.config,
        "preset": args.preset,
        "game_ids": args.games,
        "num_workers": args.workers,
        "rollout_processes": args.rollout_processes,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "execution_mode": args.execution_mode,
        "render_mode": args.render,
        "video": bool(args.video),
        "debug_log_path": args.debug,
        "checkpoint_path": args.checkpoint,
        "eval_kind": args.eval_kind,
        "per_game": bool(args.per_game),
        "world_metrics_only": bool(args.world_metrics_only),
        "eval_episodes": args.eval_episodes,
        "acting_mode_override": args.acting_mode,
        "deterministic_eval_override": None if args.deterministic_eval is None else bool(args.deterministic_eval),
        "compare_policy_vs_configured_override": None if args.compare_policy_vs_configured is None else bool(args.compare_policy_vs_configured),
        "world_updates": args.world_updates,
        "world_batch_size": args.world_batch_size,
        "world_unroll_length": args.world_unroll_length,
        "world_eval_every": args.world_eval_every,
        "world_save_every": args.world_save_every,
        "freeze_encoder": bool(args.freeze_encoder),
        "freeze_recurrent": bool(args.freeze_recurrent),
        "init_from_checkpoint": args.init_from_checkpoint,
        "log_gameplay_metrics_during_pretrain": bool(args.log_gameplay_metrics_during_pretrain),
        "dry_run": bool(args.dry_run),
        "smoke_test": bool(args.smoke_test),
        "mode": mode,
    }
    if mode == "eval_policy":
        result = eval_main(**shared)
    else:
        result = train_main(updates=args.updates, **shared)
    print(json.dumps(result, indent=2, default=str))
    return 0


def _normalize_mode(mode: str) -> str:
    if mode == "train":
        return "train_rl"
    if mode == "eval":
        return "eval_policy"
    return mode


if __name__ == "__main__":
    raise SystemExit(main())
