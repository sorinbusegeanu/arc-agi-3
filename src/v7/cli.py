from __future__ import annotations

import argparse
import json
from pathlib import Path

from v7.derivation.scientific import EpisodeEvidence
from v7.runtime import V7Runtime, V7RuntimeConfig


def _episode(row: dict[str, object]) -> EpisodeEvidence:
    return EpisodeEvidence(
        context_signature=int(row["context_signature"]),
        action_id=int(row["action_id"]),
        outcome_signature=int(row["outcome_signature"]),
        success=bool(row["success"]),
        prediction_error=float(row.get("prediction_error", 0.0)),
        future_option_delta=float(row.get("future_option_delta", 0.0)),
        source_game=None if row.get("source_game") is None else str(row["source_game"]),
        source_context=None if row.get("source_context") is None else str(row["source_context"]),
        source_global_step=None if row.get("source_global_step") is None else int(row["source_global_step"]),
    )


def run_events(root: str | Path, events_path: str | Path, *, no_restore: bool = False) -> dict[str, int]:
    runtime = V7Runtime(V7RuntimeConfig.from_path(root, restore=not no_restore))
    count = 0
    try:
        with Path(events_path).open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                runtime.observe(_episode(json.loads(line)))
                count += 1
        result = runtime.commit()
        return {"events": count, "generation": int(result.state.generation_id), "memories": len(result.view.nodes)}
    finally:
        runtime.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arc-agi3-v7")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--root", required=True)
    run.add_argument("--events", required=True)
    run.add_argument("--no-restore", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run":
        print(json.dumps(run_events(args.root, args.events, no_restore=args.no_restore), sort_keys=True))
        return 0
    return 2
