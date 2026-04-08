from __future__ import annotations

import argparse
import json

from rl_v1.eval.eval_main import eval_main
from rl_v1.training.train_main import train_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="rl_v1 V1 entrypoint")
    parser.add_argument("--config", default=None)
    parser.add_argument("--preset", default="v2", help="Optional preset override. Defaults to v2.")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--game", action="append", dest="games", default=None, help="Override game list. Repeat for multiple games.")
    parser.add_argument("--workers", type=int, default=None, help="Deprecated alias: overrides runtime.rollout_processes.")
    parser.add_argument("--rollout-processes", type=int, default=None, help="Override runtime.rollout_processes.")
    parser.add_argument("--accelerator", default=None, help="Override runtime.accelerator.")
    parser.add_argument("--devices", type=int, default=None, help="Override runtime.devices.")
    parser.add_argument("--execution-mode", default=None, help="Override env.execution_mode.")
    parser.add_argument("--render", default=None, help="Override env.render_mode.")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint.restore_path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
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
        "checkpoint_path": args.checkpoint,
    }
    result = train_main(updates=args.updates, **shared) if args.mode == "train" else eval_main(**shared)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
