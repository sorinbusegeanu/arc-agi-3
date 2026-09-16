from __future__ import annotations

import argparse
from pathlib import Path

from .grounding_h16 import evaluate_h16, run_synthetic_h16_controls, save_h16_evidence
from .h16_reproducible import run_reproducible_hydra_h16_controls


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the matched v9.7.8 H16 C0-C3 grounding experiment")
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=None, help="Hydra/HGT condition roots; defaults beside --output")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--environment-config-id", type=int, default=1)
    parser.add_argument("--interaction-budget", type=int, default=128)
    parser.add_argument("--evaluation-id", type=int, default=1)
    parser.add_argument("--no-hgt", action="store_true", help="Run Hydra memory controls without HGT training")
    parser.add_argument("--reference-synthetic", action="store_true", help="Use the cheap non-Hydra reference control")
    args = parser.parse_args(argv)

    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        parser.error("--seeds must contain at least one integer")
    output = Path(args.output)

    if args.reference_synthetic:
        trials = run_synthetic_h16_controls(
            seeds=seeds,
            environment_config_id=int(args.environment_config_id),
            interaction_budget=int(args.interaction_budget),
            evaluation_id=int(args.evaluation_id),
        )
    else:
        runtime_root = Path(args.root) if args.root else output.parent / "hydra-runtime"
        trials = run_reproducible_hydra_h16_controls(
            root=runtime_root,
            seeds=seeds,
            environment_config_id=int(args.environment_config_id),
            interaction_budget=int(args.interaction_budget),
            evaluation_id=int(args.evaluation_id),
            train_hgt=not bool(args.no_hgt),
        )

    report = evaluate_h16(trials)
    save_h16_evidence(output, trials, report)
    print(
        f"H16 status={report.result.status.value} matched={int(report.matched)} "
        f"effect={report.result.effect} causal={report.causal_effect} "
        f"runtime_backed={int(all(row.run_state.runtime_backed for row in trials))} "
        f"hgt_used={int(any(row.run_state.hgt_used for row in trials))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
