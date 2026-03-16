from __future__ import annotations

import json
from pathlib import Path

from v3_1.eval.run_hypothesis_comparison import run_hypothesis_comparison


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--game-ids", default="")
    parser.add_argument("--game-set", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--round-budget", type=int, default=0)
    parser.add_argument("--deterministic-only-flag", action="store_true")
    parser.add_argument("--llm-enable-flag", action="store_true")
    parser.add_argument("--provider-config-override", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deterministic-only-report", default=None)
    parser.add_argument("--deterministic-plus-llm-report", default=None)
    parser.add_argument("--llm-gated-tight-report", default=None)
    args = parser.parse_args()

    deterministic_only = json.loads(Path(args.deterministic_only_report).read_text(encoding="utf-8")) if args.deterministic_only_report else {}
    deterministic_plus_llm = json.loads(Path(args.deterministic_plus_llm_report).read_text(encoding="utf-8")) if args.deterministic_plus_llm_report else {}
    llm_gated_tight_control = json.loads(Path(args.llm_gated_tight_report).read_text(encoding="utf-8")) if args.llm_gated_tight_report else {}
    report = run_hypothesis_comparison(
        deterministic_only=deterministic_only,
        deterministic_plus_llm=deterministic_plus_llm,
        llm_gated_tight_control=llm_gated_tight_control,
    )
    report["requested_eval"] = {
        "game_ids": [value for value in str(args.game_ids or "").split(",") if value],
        "game_set": str(args.game_set or ""),
        "seeds": [value for value in str(args.seeds or "").split(",") if value],
        "round_budget": int(args.round_budget or 0),
        "deterministic_only_flag": bool(args.deterministic_only_flag),
        "llm_enable_flag": bool(args.llm_enable_flag),
        "provider_config_override": str(args.provider_config_override or ""),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hypothesis_comparison_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
