from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from .config import DEFAULT_ENV_FACTORY_PATH, DEFAULT_ENV_ROOT, build_config, default_session_id
from .loop_controller import LoopController


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VLM v2 branching loop")
    parser.add_argument("--env-factory-path", default=DEFAULT_ENV_FACTORY_PATH)
    parser.add_argument("--games", "--env-id", dest="env_id", required=True)
    parser.add_argument("--outdir", "--output-dir", dest="output_dir", default="runs/vlm_v2")
    parser.add_argument("--prompt-config", default="src/vlm_v2/prompt_config.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--max_actions_budget", "--max-actions-budget", dest="max_actions_budget", type=int, default=1000)
    parser.add_argument("--action-list", required=True, help="Bootstrap LRUD sequence applied at the start of every level, e.g. RRDD")
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--max-prompt-frames", type=int, default=6)
    parser.add_argument("--max-parallel-branches", type=int, default=8)
    parser.add_argument("--render-terminal", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def _configure_debug_logging(*, output_dir: str, enabled: bool) -> None:
    if not enabled:
        return
    root = logging.getLogger("vlm_v2")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.propagate = False
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(Path(output_dir) / "debug.log", mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    root.addHandler(handler)


def _capture_tty_state() -> str | None:
    try:
        if not os.isatty(sys.stdin.fileno()):
            return None
    except Exception:
        return None
    try:
        proc = subprocess.run(
            ["stty", "-g"],
            check=True,
            capture_output=True,
            text=True,
        )
        state = proc.stdout.strip()
        return state or None
    except Exception:
        return None


def _restore_tty_state(state: str | None) -> None:
    if not state:
        return
    try:
        subprocess.run(
            ["stty", state],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    tty_state = _capture_tty_state()
    _configure_debug_logging(output_dir=args.output_dir, enabled=bool(args.debug))
    try:
        config = build_config(
            output_root=args.output_dir,
            env_factory_path=args.env_factory_path,
            env_id=args.env_id,
            env_root=DEFAULT_ENV_ROOT,
            seed=args.seed,
            render_terminal=bool(args.render_terminal),
            fps=args.fps,
            max_actions_budget=args.max_actions_budget,
            action_list=args.action_list,
            prompt_config_path=args.prompt_config,
            timeout_sec=args.timeout_sec,
            retry_count=args.retry_count,
            max_prompt_frames=args.max_prompt_frames,
            max_parallel_branches=args.max_parallel_branches,
            debug=bool(args.debug),
        )
        controller = LoopController(config=config, session_id=default_session_id())
        summary = controller.run_episode()
        print(json.dumps(summary, indent=2))
        return 0
    finally:
        _restore_tty_state(tty_state)


if __name__ == "__main__":
    raise SystemExit(main())
