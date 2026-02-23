from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Check run quality before training")
    p.add_argument("--summary", type=str, required=True, help="run summary.json path")
    args = p.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    results = summary.get("results", [])
    if not isinstance(results, list):
        results = []

    episodes = len(results)
    unique_ok = [r for r in results if int(r.get("unique_states", 0)) > 1]
    wins = [r for r in results if bool(r.get("won", False))]

    report = {
        "episodes": episodes,
        "episodes_unique_states_gt1": len(unique_ok),
        "episodes_won": len(wins),
        "ready_for_training": len(unique_ok) > 0 and len(wins) > 0,
        "notes": [],
    }
    if len(unique_ok) == 0:
        report["notes"].append("No state changes detected (unique_states <= 1).")
    if len(wins) == 0:
        report["notes"].append("No successful episodes yet; ranker positives will be zero.")

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
