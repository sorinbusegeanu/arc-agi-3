from __future__ import annotations

import argparse
import json

from zod01.src.eval_runner import run_eval


def main() -> None:
    p = argparse.ArgumentParser(description="Run zod01 ARC-AGI-3 agent")
    p.add_argument("--games", type=str, default="ls20", help="Comma-separated game ids")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-actions", type=int, default=80)
    p.add_argument("--variant-id", type=str, default="baseline")
    p.add_argument("--log-dir", type=str, default="zod01/logs")
    p.add_argument("--use-ranker", action="store_true")
    p.add_argument("--use-learned-critic", action="store_true")
    p.add_argument("--use-mechanic-classifier", action="store_true")
    p.add_argument("--ranker-model", type=str, default="zod01/models/ranker.json")
    p.add_argument("--critic-model", type=str, default="zod01/models/critic.json")
    p.add_argument("--mechanic-model", type=str, default="zod01/models/mechanic.json")
    p.add_argument("--w-ranker", type=float, default=0.5)
    p.add_argument("--w-risk", type=float, default=0.5)
    p.add_argument("--w-safety", type=float, default=1.0)
    args = p.parse_args()

    results = run_eval(
        args.games.split(","),
        seed=args.seed,
        max_actions=args.max_actions,
        variant_id=args.variant_id,
        log_dir=args.log_dir,
        use_ranker=args.use_ranker,
        use_learned_critic=args.use_learned_critic,
        use_mechanic_classifier=args.use_mechanic_classifier,
        ranker_model_path=args.ranker_model,
        critic_model_path=args.critic_model,
        mechanic_model_path=args.mechanic_model,
        w_ranker=args.w_ranker,
        w_risk=args.w_risk,
        w_safety=args.w_safety,
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
