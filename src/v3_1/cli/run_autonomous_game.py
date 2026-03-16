from __future__ import annotations

import argparse
import uuid
from dataclasses import replace

from v3_1.config.loader import load_config
from v3_1.runtime.bootstrap import bootstrap_services
from v3_1.runtime.orchestrator import Orchestrator
from v3_1.runtime.run_context import RunContext
from v3_1.runtime.snapshot_registry import SnapshotRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--game-id", default="game")
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--render-terminal", action="store_true")
    parser.add_argument("--llm-debug", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.llm_debug:
        config = replace(
            config,
            hypothesis_generation=replace(config.hypothesis_generation, llm_emit_raw_debug=True),
        )
    session_id = f"session:{uuid.uuid4().hex[:8]}"
    context = RunContext(session_id=session_id, run_id=f"run:{uuid.uuid4().hex[:8]}", game_id=args.game_id)
    services = bootstrap_services(config, session_id=context.session_id, game_id=context.game_id, render_terminal=args.render_terminal)
    result = Orchestrator(config=config, context=context, services=services, snapshot_registry=SnapshotRegistry()).run(export_png=args.png)
    print(result)


if __name__ == "__main__":
    main()
