from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from zod01.src.learned_models import MechanicClassifierModel


def main() -> None:
    p = argparse.ArgumentParser(description="Train zod01 mechanic classifier (lightweight)")
    p.add_argument("--dataset", type=str, default="zod01/datasets/mechanic.jsonl")
    p.add_argument("--out", type=str, default="zod01/models/mechanic.json")
    args = p.parse_args()

    change: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)

    for line in Path(args.dataset).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        action = str(r.get("action", ""))
        total[action] += 1
        label = str(r.get("label", ""))
        if label == "sig:change":
            change[action] += 1

    action_bias = {
        a: (2.0 * change[a] / total[a] - 1.0) if total[a] else 0.0
        for a in total
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model = MechanicClassifierModel(action_bias=action_bias)
    Path(args.out).write_text(json.dumps({"action_bias": action_bias}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"actions": len(action_bias), "model_out": args.out}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
