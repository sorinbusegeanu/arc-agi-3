from __future__ import annotations

import argparse
import uuid

from v3_1.config.loader import load_config
from v3_1.runtime.bootstrap import bootstrap_services
from v3_1.runtime.orchestrator import Orchestrator
from v3_1.runtime.run_context import RunContext
from v3_1.runtime.snapshot_registry import SnapshotRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--game-id", default="game")
    args = parser.parse_args()
    config = load_config(args.config)
    session_id = f"session:{uuid.uuid4().hex[:8]}"
    context = RunContext(session_id=session_id, run_id=f"run:{uuid.uuid4().hex[:8]}", game_id=args.game_id)
    services = bootstrap_services(config, session_id=context.session_id, game_id=context.game_id)
    result = Orchestrator(config=config, context=context, services=services, snapshot_registry=SnapshotRegistry()).run()
    print(result)


if __name__ == "__main__":
    main()

