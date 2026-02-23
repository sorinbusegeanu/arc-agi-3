from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from zod01.src.learned_models import LearnedCriticModel


def main() -> None:
    p = argparse.ArgumentParser(description="Train zod01 learned critic")
    p.add_argument("--dataset", type=str, default="zod01/datasets/critic.jsonl")
    p.add_argument("--out", type=str, default="zod01/models/critic.json")
    args = p.parse_args()

    totals: dict[str, int] = defaultdict(int)
    risk_sum: dict[str, float] = defaultdict(float)

    for line in Path(args.dataset).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        action = str(r.get("action", ""))
        totals[action] += 1
        risk_sum[action] += (
            0.5 * float(r.get("loop_risk", 0))
            + 0.5 * float(r.get("irreversible_risk", 0))
            + 0.25 * float(r.get("dead_end", 0))
        )

    action_risk = {
        a: (risk_sum[a] / totals[a]) if totals[a] else 0.0
        for a in totals
    }
    model = LearnedCriticModel(action_risk=action_risk)
    model.save(args.out)
    print(json.dumps({"actions": len(action_risk), "model_out": args.out}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
