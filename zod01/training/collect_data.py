from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from zod01.src.eval_runner import run_eval


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def variant_flags(variant: str) -> dict[str, object]:
    v = variant.strip().lower()
    if v == "baseline":
        return {}
    if v == "ranker":
        return {"use_ranker": True}
    if v == "critic":
        return {"use_learned_critic": True}
    if v == "mechanic":
        return {"use_mechanic_classifier": True}
    if v == "all":
        return {
            "use_ranker": True,
            "use_learned_critic": True,
            "use_mechanic_classifier": True,
        }
    return {}


def main() -> None:
    p = argparse.ArgumentParser(description="Collect zod01 training logs")
    p.add_argument("--games", type=str, required=True, help="Comma-separated game ids")
    p.add_argument("--seeds", type=str, default="0,1,2,3")
    p.add_argument("--variants", type=str, default="baseline")
    p.add_argument("--max-actions", type=int, default=80)
    p.add_argument("--run-dir", type=str, default="zod01/runs")
    p.add_argument("--ranker-model", type=str, default="zod01/models/ranker.json")
    p.add_argument("--critic-model", type=str, default="zod01/models/critic.json")
    p.add_argument("--mechanic-model", type=str, default="zod01/models/mechanic.json")
    args = p.parse_args()

    seeds = parse_int_list(args.seeds)
    games = [g.strip() for g in args.games.split(",") if g.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    run_id = f"run_{uuid.uuid4().hex[:10]}"
    run_dir = Path(args.run_dir) / run_id
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, object]] = []
    for variant in variants:
        flags = variant_flags(variant)
        for seed in seeds:
            results = run_eval(
                games,
                seed=seed,
                max_actions=args.max_actions,
                variant_id=variant,
                log_dir=str(log_dir),
                ranker_model_path=args.ranker_model,
                critic_model_path=args.critic_model,
                mechanic_model_path=args.mechanic_model,
                **flags,
            )
            for r in results:
                r["seed"] = seed
                r["variant_id"] = variant
            all_results.extend(results)

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "games": games,
        "seeds": seeds,
        "variants": variants,
        "max_actions": args.max_actions,
        "episodes": len(all_results),
        "results": all_results,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
