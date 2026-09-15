from __future__ import annotations

import argparse
from pathlib import Path

from .grounding_h16 import H16RunState, evaluate_h16, run_synthetic_h16_controls, save_h16_evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the matched v9.7.8 H16 C0-C3 grounding experiment")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--environment-config-id", type=int, default=1)
    parser.add_argument("--interaction-budget", type=int, default=128)
    parser.add_argument("--evaluation-id", type=int, default=1)
    parser.add_argument("--scientific-config-id", default="")
    parser.add_argument("--model-version", default="untrained")
    parser.add_argument("--snapshot-id", type=int, default=0)
    parser.add_argument("--source-state-id", default="")
    args = parser.parse_args(argv)
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        parser.error("--seeds must contain at least one integer")
    source_state_id = args.source_state_id or f"h16:{args.environment_config_id}:{args.evaluation_id}:snapshot:{args.snapshot_id}"
    run_state = H16RunState(
        scientific_config_id=str(args.scientific_config_id),
        model_version=str(args.model_version),
        snapshot_id=int(args.snapshot_id),
        symbol_schema_version=2,
        source_state_id=source_state_id,
    )
    trials = run_synthetic_h16_controls(
        seeds=seeds,
        environment_config_id=int(args.environment_config_id),
        interaction_budget=int(args.interaction_budget),
        evaluation_id=int(args.evaluation_id),
        run_state=run_state,
    )
    report = evaluate_h16(trials)
    save_h16_evidence(Path(args.output), trials, report)
    print(f"H16 status={report.result.status.value} matched={int(report.matched)} effect={report.result.effect} causal={report.causal_effect}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
