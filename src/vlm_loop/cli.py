from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import build_config, default_session_id
from .loop_controller import LoopController

DEFAULT_ENV_ROOT = "/home/zodrak/zod/environment_files"
DEFAULT_ENV_FACTORY_PATH = "arc_agi_agent.envs.arc_env_factory:create_env"


def _config_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal VLM closed-loop runner")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run-seeds", "run-loop", "replay-sequence"):
        cmd = sub.add_parser(name)
        _add_common_args(cmd)
        if name == "replay-sequence":
            cmd.add_argument("--sequence", required=True, help="Comma-separated actions, for example UP,LEFT,DOWN")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    prompt_config = _load_prompt_config_file(args.prompt_config)
    config = build_config(
        output_root=args.output_dir,
        fps=args.fps,
        max_steps=args.max_actions,
        max_iterations=args.iters,
        max_sequences_per_iter=args.workers,
        max_returned_sequences_per_video=args.max_returned_sequences_per_video,
        action_length_cap=args.action_length_cap or args.max_actions,
        timeout_sec=args.timeout_sec,
        agents_per_iteration=args.workers,
        retry_count=args.retry_count,
        seed=args.seed,
        task_prompt=args.task_prompt,
        llm_backend=args.llm_backend or str(prompt_config.get("llm_backend") or "vllm"),
        ollama_url=args.ollama_url or str(prompt_config.get("ollama_url") or "http://192.168.0.51:11434"),
        ollama_model=args.ollama_model or str(prompt_config.get("ollama_model") or "qwen3-vl:8b"),
        ollama_num_ctx=args.ollama_num_ctx or int(prompt_config.get("ollama_num_ctx") or 16384),
        vllm_url=args.vllm_url or str(prompt_config.get("vllm_url") or "http://192.168.0.51:8000"),
        vllm_model=args.vllm_model or str(prompt_config.get("vllm_model") or "Qwen/Qwen3-VL-8B-Instruct-FP8"),
        disable_thinking=_config_bool(prompt_config.get("disable_thinking"), True),
        greedy=_config_bool(prompt_config.get("greedy"), False),
        top_p=float(prompt_config.get("top_p") or 0.8),
        top_k=int(prompt_config.get("top_k") or 20),
        temperature=float(prompt_config.get("temperature") or 0.7),
        repetition_penalty=float(prompt_config.get("repetition_penalty") or 1.0),
        presence_penalty=float(prompt_config.get("presence_penalty") or 1.5),
        out_seq_length=int(prompt_config.get("out_seq_length") or 16384),
        prompt_config_path=args.prompt_config,
        max_prompt_frames=args.max_prompt_frames,
        render_terminal=args.render_terminal,
        debug=args.debug,
    )
    controller = LoopController(
        env_factory_path=args.env_factory_path,
        env_id=args.env_id,
        env_root=DEFAULT_ENV_ROOT,
        config=config,
        session_id=default_session_id(),
    )
    if args.command == "run-seeds":
        controller.run_seeds()
        return 0
    if args.command == "run-loop":
        controller.run_loop()
        return 0
    actions = [token.strip().upper() for token in args.sequence.split(",") if token.strip()]
    result = controller.replay_sequence(actions)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--env-factory-path",
        default=DEFAULT_ENV_FACTORY_PATH,
        help=f"Factory path like package.module:create_env. Default: {DEFAULT_ENV_FACTORY_PATH}",
    )
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--output-dir", default="runs/vlm_loop")
    parser.add_argument("--iters", type=int, default=2, help="Number of episodes to run in run-loop")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=200)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--llm-backend", choices=["ollama", "vllm"], default=None)
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--ollama-num-ctx", type=int, default=None)
    parser.add_argument("--vllm-url", default=None)
    parser.add_argument("--vllm-model", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-returned-sequences-per-video", type=int, default=4)
    parser.add_argument("--action-length-cap", type=int, default=None)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--task-prompt", default="")
    parser.add_argument("--prompt-config", default="src/vlm_loop/prompt_config.json")
    parser.add_argument("--max-prompt-frames", type=int, default=2)
    parser.add_argument("--render-terminal", action="store_true")
    parser.add_argument("--debug", action="store_true")


def _load_prompt_config_file(path: str) -> dict:
    config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("prompt config must be a JSON object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
