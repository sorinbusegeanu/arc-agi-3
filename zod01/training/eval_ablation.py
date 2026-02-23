from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from zod01.src.eval_runner import run_eval


def summarize(results: list[dict[str, object]]) -> dict[str, object]:
    n = len(results)
    wins = [r for r in results if bool(r.get("won", False))]
    actions_on_wins = [int(r.get("actions", 0)) for r in wins]
    unique_per_1k = []
    for r in results:
        actions = max(1, int(r.get("actions", 1)))
        unique = int(r.get("unique_states", 0))
        unique_per_1k.append(unique * 1000.0 / actions)
    irreversible_eps = 0
    loop_eps = 0
    for r in results:
        log_path = str(r.get("log_path", ""))
        if not log_path or not Path(log_path).exists():
            continue
        has_irrev = False
        has_loop = False
        for line in Path(log_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            if ev.get("type") != "step":
                continue
            tags = ev.get("tags", [])
            if isinstance(tags, list) and "cycle-risk" in tags:
                has_loop = True
            delta = ev.get("delta", {})
            if isinstance(delta, dict):
                ratio = float(delta.get("change_ratio", 0.0))
                if ratio >= 0.5:
                    has_irrev = True
            if isinstance(tags, list) and "complex-risk" in tags:
                has_irrev = True
        irreversible_eps += int(has_irrev)
        loop_eps += int(has_loop)

    return {
        "episodes": n,
        "win_rate": 0.0 if n == 0 else len(wins) / n,
        "median_actions_on_wins": None if not actions_on_wins else statistics.median(actions_on_wins),
        "unique_states_per_1k_steps": 0.0 if not unique_per_1k else statistics.mean(unique_per_1k),
        "irreversible_rate": 0.0 if n == 0 else irreversible_eps / n,
        "loop_thrash_rate": 0.0 if n == 0 else loop_eps / n,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Ablation runner for learned modules")
    p.add_argument("--games", type=str, required=True)
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--max-actions", type=int, default=80)
    p.add_argument("--log-dir", type=str, default="zod01/logs_ablation")
    p.add_argument("--ranker-model", type=str, default="zod01/models/ranker.json")
    p.add_argument("--critic-model", type=str, default="zod01/models/critic.json")
    p.add_argument("--mechanic-model", type=str, default="zod01/models/mechanic.json")
    args = p.parse_args()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    variants = {
        "baseline": {},
        "ranker": {"use_ranker": True},
        "critic": {"use_learned_critic": True},
        "mechanic": {"use_mechanic_classifier": True},
        "all": {"use_ranker": True, "use_learned_critic": True, "use_mechanic_classifier": True},
    }

    report: dict[str, object] = {}
    for name, flags in variants.items():
        rows: list[dict[str, object]] = []
        for seed in seeds:
            rows.extend(
                run_eval(
                    games,
                    seed=seed,
                    max_actions=args.max_actions,
                    variant_id=name,
                    log_dir=args.log_dir,
                    ranker_model_path=args.ranker_model,
                    critic_model_path=args.critic_model,
                    mechanic_model_path=args.mechanic_model,
                    **flags,
                )
            )
        report[name] = summarize(rows)

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
