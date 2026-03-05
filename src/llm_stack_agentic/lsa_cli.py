from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any, Dict

from .lsa_agent_catalog import build_agent_catalog
from .lsa_env_adapter import default_env_adapter


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _import_adapter(adapter_path: str) -> Any:
    if ":" not in adapter_path:
        raise SystemExit("--adapter must be in module:attr format")
    module_name, attr_name = adapter_path.split(":", 1)
    module = importlib.import_module(module_name)
    if not hasattr(module, attr_name):
        raise SystemExit(f"Adapter attribute not found: {adapter_path}")
    return getattr(module, attr_name)


def _build_adapter(adapter_path: str, adapter_args: Dict[str, Any]) -> Any:
    adapter_factory = _import_adapter(adapter_path)
    if callable(adapter_factory):
        return adapter_factory(**adapter_args) if adapter_args else adapter_factory()
    return adapter_factory


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM Stack Agentic runner")
    parser.add_argument("--list-agents", action="store_true", help="List available agents")
    parser.add_argument("--agent", help="Agent name")
    parser.add_argument("--adapter", help="Env adapter path module:attr (overrides adapter config)")
    parser.add_argument("--adapter-args", help="JSON string of adapter kwargs")
    parser.add_argument("--adapter-args-file", help="JSON file of adapter kwargs")
    parser.add_argument("--adapter-config", help="JSON file with adapter config")
    parser.add_argument("--adapter-config-json", help="JSON string with adapter config")
    parser.add_argument("--episode-id", default="episode_0", help="Episode id")
    parser.add_argument("--game-id", default="game_0", help="Game id")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--probe-steps", type=int, default=4, help="Probe steps (bootstrap_explorer)")
    parser.add_argument("--policy-config", help="JSON file with policy_config")
    parser.add_argument("--policy-config-json", help="JSON string with policy_config")
    parser.add_argument("--logging-config", help="JSON file with logging_config")
    parser.add_argument("--logging-config-json", help="JSON string with logging_config")
    parser.add_argument("--outdir", help="Output directory for JSON outputs")
    parser.add_argument("--print-output", action="store_true", help="Print stdout JSON output")
    parser.add_argument("--probe-trace", help="Path to probe_trace.json (visual_describer)")
    parser.add_argument("--max-pois", type=int, default=5, help="Max POIs (visual_describer)")
    parser.add_argument("--model-config", help="JSON file with model_config (visual_describer)")
    parser.add_argument("--model-config-json", help="JSON string with model_config (visual_describer)")
    parser.add_argument("--controller-config", help="JSON file with controller_config (controller)")
    parser.add_argument("--controller-config-json", help="JSON string with controller_config (controller)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to runs/debug.log")
    parser.add_argument("--png-factor", type=int, default=1, help="Scale factor for saved PNG frames (visual_describer)")

    args = parser.parse_args()

    catalog = build_agent_catalog()
    if args.list_agents:
        for name in catalog.keys():
            print(name)
        return 0

    if not args.agent:
        raise SystemExit("--agent is required")
    if args.agent not in catalog:
        raise SystemExit(f"Unknown agent: {args.agent}")

    adapter_args: Dict[str, Any] = {}
    if args.adapter_args:
        adapter_args = json.loads(args.adapter_args)
    if args.adapter_args_file:
        adapter_args = _load_json(args.adapter_args_file)

    adapter_config: Dict[str, Any] = {}
    if args.adapter_config:
        adapter_config = _load_json(args.adapter_config)
    if args.adapter_config_json:
        adapter_config = json.loads(args.adapter_config_json)

    adapter_name = str(adapter_config.get("adapter", "default"))
    adapter_path = adapter_config.get("adapter_path")
    adapter_config_args = adapter_config.get("adapter_args")
    if isinstance(adapter_config_args, dict) and not adapter_args:
        adapter_args = adapter_config_args

    policy_config: Dict[str, Any] = {}
    if args.policy_config:
        policy_config = _load_json(args.policy_config)
    if args.policy_config_json:
        policy_config = json.loads(args.policy_config_json)

    logging_config: Dict[str, Any] = {}
    if args.logging_config:
        logging_config = _load_json(args.logging_config)
    if args.logging_config_json:
        logging_config = json.loads(args.logging_config_json)

    env_adapter = None
    if args.agent == "bootstrap_explorer":
        if args.adapter:
            env_adapter = _build_adapter(args.adapter, adapter_args)
        elif adapter_path:
            env_adapter = _build_adapter(str(adapter_path), adapter_args)
        else:
            if adapter_name not in ("default", "arc_agi_default"):
                raise SystemExit(f"Unknown adapter name: {adapter_name}")
            env_adapter = default_env_adapter(**(adapter_args or {}))

    debug_log_path = os.path.join("runs", "debug.log")
    if args.debug:
        logging_config = dict(logging_config)
        logging_config["debug"] = True
        logging_config["debug_log_path"] = debug_log_path
        os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)
        with open(debug_log_path, "w", encoding="utf-8") as f:
            f.write("")
    if args.png_factor and args.png_factor != 1:
        logging_config = dict(logging_config)
        logging_config["png_factor"] = max(1, int(args.png_factor))

    if args.agent == "bootstrap_explorer":
        result = catalog[args.agent].run(
            episode_id=args.episode_id,
            game_id=args.game_id,
            seed=args.seed,
            env_adapter=env_adapter,
            probe_steps=args.probe_steps,
            policy_config=policy_config or None,
            logging_config=logging_config or None,
        )
    elif args.agent == "visual_describer":
        if not args.probe_trace:
            raise SystemExit("--probe-trace is required for visual_describer")
        probe_trace = _load_json(args.probe_trace)
        model_config: Dict[str, Any] = {}
        if args.model_config:
            model_config = _load_json(args.model_config)
        if args.model_config_json:
            model_config = json.loads(args.model_config_json)
        result = catalog[args.agent].run(
            episode_id=args.episode_id,
            game_id=args.game_id,
            seed=args.seed,
            probe_trace=probe_trace,
            max_pois=args.max_pois,
            model_config=model_config or None,
            logging_config=logging_config or None,
        )
    elif args.agent == "controller":
        controller_config: Dict[str, Any] = {}
        if args.controller_config:
            controller_config = _load_json(args.controller_config)
        if args.controller_config_json:
            controller_config = json.loads(args.controller_config_json)
        if args.debug:
            controller_config = dict(controller_config)
            controller_config["debug"] = True
            controller_config["debug_log_path"] = debug_log_path
        if args.png_factor and args.png_factor != 1:
            controller_config = dict(controller_config)
            controller_config["png_factor"] = max(1, int(args.png_factor))
        if args.adapter:
            env_adapter = _build_adapter(args.adapter, adapter_args)
        elif adapter_path:
            env_adapter = _build_adapter(str(adapter_path), adapter_args)
        else:
            if adapter_name not in ("default", "arc_agi_default"):
                raise SystemExit(f"Unknown adapter name: {adapter_name}")
            env_adapter = default_env_adapter(**(adapter_args or {}))
        result = catalog[args.agent].run(
            game_id=args.game_id,
            episode_id=args.episode_id,
            seed=args.seed,
            controller_config=controller_config,
            env_adapter=env_adapter,
            agents=build_agent_catalog(),
            services={},
        )
    else:
        result = catalog[args.agent].run(
            episode_id=args.episode_id,
            game_id=args.game_id,
            seed=args.seed,
            trace_id=logging_config.get("trace_id"),
            timestamp_step=logging_config.get("timestamp_step"),
        )

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        if args.agent == "bootstrap_explorer":
            trace, report = result
            _write_json(os.path.join(args.outdir, "probe_trace.json"), trace)
            _write_json(os.path.join(args.outdir, "bootstrap_report.json"), report)
        else:
            _write_json(os.path.join(args.outdir, f"{args.agent}_output.json"), result)
    if args.print_output and not args.outdir:
        if args.agent == "bootstrap_explorer":
            trace, report = result
            print(json.dumps({"probe_trace": trace, "bootstrap_report": report}, indent=2))
        else:
            print(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
