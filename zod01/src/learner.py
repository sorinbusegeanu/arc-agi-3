from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class ActionFrequencyLearner:
    """Tiny offline learner baseline: action prior from successful trajectories."""

    def train(self, log_files: list[str], output_path: str = "zod01/data/action_priors.json") -> dict[str, float]:
        success_counts: Counter[str] = Counter()
        total = 0
        for file in log_files:
            for line in Path(file).read_text(encoding="utf-8").splitlines():
                ev = json.loads(line)
                if ev.get("type") != "step":
                    continue
                if not ev.get("won", False):
                    continue
                action = ev.get("action")
                if isinstance(action, str):
                    success_counts[action] += 1
                    total += 1

        priors = {k: (v / total) for k, v in success_counts.items()} if total else {}
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(priors, sort_keys=True, indent=2), encoding="utf-8")
        return priors
