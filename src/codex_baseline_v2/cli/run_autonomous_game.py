from __future__ import annotations

import argparse
import importlib
import json
from typing import Any, Callable

from codex_baseline_v2.runtime.round_orchestrator import run_autonomous_rounds
from codex_baseline_v2.shared.config import load_config


def _load_env_factory(path: str) -> Callable[[], Any]:
    module_name, func_name = path.rsplit(":", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, func_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V2 autonomous game")
    parser.add_argument("--config", required=True)
    parser.add_argument("--game-id", default=None)
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--env-factory", default=None, help="module:function returning env instance")
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--env-root", default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg_payload = json.load(handle)
    game_id = args.game_id or cfg_payload.get("game_id")
    storage_root = args.storage_root or cfg_payload.get("storage", {}).get("root_dir") or cfg_payload.get("memory", {}).get("storage_dir")
    if not game_id or not storage_root:
        raise SystemExit("game-id and storage-root are required (flag or config)")
    cfg_payload["game_id"] = game_id
    cfg_payload.setdefault("memory", {})["storage_dir"] = storage_root
    cfg = load_config(cfg_payload)

    env_cfg = cfg_payload.get("env", {}) if isinstance(cfg_payload, dict) else {}
    env_factory_path = args.env_factory or env_cfg.get("env_factory")
    env_id = args.env_id or env_cfg.get("env_id")
    env_root = args.env_root or env_cfg.get("env_root")
    if not env_factory_path or not env_id or not env_root:
        raise SystemExit("env-factory, env-id, and env-root are required (flag or config.env.*)")

    env_factory = _load_env_factory(env_factory_path)

    def factory_wrapper():
        try:
            return env_factory(env_id=env_id, env_root=env_root)
        except TypeError:
            return env_factory()

    print(f"[v2] autonomous_start game_id={game_id} storage_root={storage_root}", flush=True)
    run_autonomous_rounds(cfg, factory_wrapper, env_factory_path=env_factory_path, workers=int(args.workers))
    print(f"[v2] autonomous_done game_id={game_id}", flush=True)


if __name__ == "__main__":
    main()
