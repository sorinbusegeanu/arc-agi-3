from __future__ import annotations

import argparse
import importlib
import json

from codex_baseline_v2.runtime.environment_session import EnvironmentSessionV2
from codex_baseline_v2.runtime.trajectory_collector import CollectionConfigV2, TrajectoryCollectorV2
from codex_baseline_v2.shared.config import load_config
from codex_baseline_v2.shared.storage import StoragePathsV2


def _load_env_factory(path: str):
    module_name, func_name = path.rsplit(":", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, func_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect V2 trajectories")
    parser.add_argument("--config", required=True)
    parser.add_argument("--env-factory", required=True)
    parser.add_argument("--mode", required=True, choices=["random_probe", "unguided_probe", "instructed_execution"])
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--env-root", default=None)
    parser.add_argument("--round-id", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg_payload = json.load(handle)
    cfg = load_config(cfg_payload)
    storage = StoragePathsV2(cfg.memory.storage_dir)

    env_cfg = cfg_payload.get("env", {}) if isinstance(cfg_payload, dict) else {}
    env_id = args.env_id or env_cfg.get("env_id")
    env_root = args.env_root or env_cfg.get("env_root")
    if not env_id or not env_root:
        raise SystemExit("env-id and env-root are required (flag or config.env.*)")
    env_factory = _load_env_factory(args.env_factory)
    try:
        env = env_factory(env_id=env_id, env_root=env_root)
    except TypeError:
        env = env_factory()
    session = EnvironmentSessionV2(env, cfg.game_id)

    collector = TrajectoryCollectorV2(
        storage,
        CollectionConfigV2(
            episodes=cfg.dataset_or_rollout_source.episodes_per_round,
            max_steps_per_episode=cfg.dataset_or_rollout_source.max_steps_per_episode,
            max_steps_per_instruction=cfg.executor.max_steps,
            seed=cfg.controller.random_seed,
            keep_invalid_steps_for_debug=cfg.debug.keep_invalid_steps_for_debug,
        ),
    )
    if args.workers > 1 and args.env_factory:
        episodes = collector.collect_round_parallel(
            env_factory_path=args.env_factory,
            env_id=env_id,
            env_root=env_root,
            mode=args.mode,
            instruction=None,
            round_id=args.round_id,
            workers=int(args.workers),
        )
    else:
        episodes = collector.collect_round(session, args.mode, None, args.round_id)
    collector.write_artifacts(cfg.game_id, args.round_id, episodes)


if __name__ == "__main__":
    main()
