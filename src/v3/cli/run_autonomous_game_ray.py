from __future__ import annotations

import argparse
import json

from codex_baseline_v2.shared.config import load_config

from v3.runtime_ray.bootstrap import bootstrap_runtime, default_runtime_config


def main() -> None:
    import ray

    parser = argparse.ArgumentParser(description="Run V3 Ray autonomous game")
    parser.add_argument("--config", required=True)
    parser.add_argument("--game-id", default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cfg = load_config(payload)
    runtime = bootstrap_runtime(
        default_runtime_config(workers=int(args.workers)),
        actor_kwargs={
            "env_worker": {
                "env_factory": cfg.env.env_factory,
                "env_id": cfg.env.env_id,
                "env_root": cfg.env.env_root,
                "collection_cfg": cfg.collection.__dict__,
            }
        },
    )
    result = ray.get(
        runtime["actors"]["coordinator"].run_session.remote(
            cfg_dict=cfg.to_dict(),
            actors=runtime["actors"],
            env_workers=runtime["env_workers"],
            episode_analyzer_workers=runtime["episode_analyzer_workers"],
            planning_helper_workers=runtime["planning_helper_workers"],
            game_id=args.game_id or cfg.game_id,
            rounds=cfg.runtime.max_rounds,
        )
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
