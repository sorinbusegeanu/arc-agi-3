from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random

from v8 import ContinuousMemoryRuntime, V8RuntimeConfig
from v8.environments import ChessAdapter, GymDiscreteAdapter
from v8.model import stable_u64


def make_adapter(name: str, *, seed: int, slippery: bool = False, chess_opponent: str = "random"):
    key = str(name).strip().lower()
    if key in {"frozenlake", "frozenlake-v1", "gym:frozenlake-v1"}:
        return GymDiscreteAdapter(
            "FrozenLake-v1",
            seed=int(seed),
            make_kwargs={"is_slippery": bool(slippery)},
        )
    if key in {"chess", "arcagi/chess-v0", "gym:arcagi/chess-v0"}:
        return ChessAdapter(seed=int(seed), opponent=str(chess_opponent))
    raise ValueError(f"unsupported environment {name!r}; choices=frozenlake,chess")


def _choose_action(view, context: int, actions: tuple[int, ...], rng: Random, epsilon: float) -> int:
    if not actions:
        raise ValueError("cannot choose from an empty action set")
    scores = tuple(view.score_actions(int(context), tuple(actions)))
    if not scores or rng.random() < float(epsilon):
        return int(actions[rng.randrange(len(actions))])
    unseen = [int(row.action_id) for row in scores if int(row.support_count) <= 0]
    if unseen:
        return int(unseen[rng.randrange(len(unseen))])
    best = min(
        scores,
        key=lambda row: (-float(row.score), -int(row.support_count), int(row.action_id)),
    )
    return int(best.action_id)


def run_environment(
    *,
    environment: str,
    root: str | Path,
    steps: int,
    seed: int = 0,
    epsilon: float = 0.10,
    slippery: bool = False,
    chess_opponent: str = "random",
    shards: int = 2,
    stage_workers: int = 1,
    enable_peers: bool = True,
) -> dict[str, object]:
    if int(steps) <= 0:
        raise ValueError("steps must be positive")
    adapter = make_adapter(
        environment,
        seed=int(seed),
        slippery=bool(slippery),
        chess_opponent=str(chess_opponent),
    )
    runtime = ContinuousMemoryRuntime(
        V8RuntimeConfig.from_path(
            root,
            shards=int(shards),
            stage_workers=int(stage_workers),
            enable_snapshots=False,
            restore=True,
            enable_peers=bool(enable_peers),
        )
    )
    rng = Random(int(seed))
    episodes = wins = losses = draws = submitted = 0
    trajectory = stable_u64(adapter.identity.source_hash, seed, person=b"v8.58-env-trajectory")
    try:
        runtime.start()
        for sequence in range(1, int(steps) + 1):
            before = adapter.observe()
            before_actions = tuple(map(int, adapter.available_actions()))
            if not before_actions:
                adapter.reset()
                episodes += 1
                trajectory = stable_u64(
                    adapter.identity.source_hash,
                    seed,
                    episodes,
                    person=b"v8.58-env-trajectory",
                )
                continue
            context = int(adapter.observation_signature(before))
            action = _choose_action(runtime.read_view, context, before_actions, rng, epsilon)
            distribution = runtime.read_view.outcome_distribution(context, action)
            after = adapter.step(action)
            after_actions = tuple(map(int, adapter.available_actions()))
            after_context = int(adapter.observation_signature(after))
            outcome = int(adapter.cognitive_transition_signature(before, after))
            family = int(adapter.cognitive_family_signature(before, after))
            changed = max(0, int(adapter.cognitive_changed_extent(before, after)))
            boundary = adapter.cognitive_boundary_event()
            prediction_error = (
                0.0
                if not distribution
                else max(0.0, 1.0 - float(distribution.get(outcome, 0.0)))
            )
            trajectory = stable_u64(
                trajectory,
                context,
                action,
                outcome,
                person=b"v8.58-env-trajectory",
            )
            observation_schema = getattr(adapter, "observation_schema", None)
            if observation_schema is None:
                observation_schema = adapter.observation_codec.schema
            action_schema = getattr(adapter, "action_schema", None)
            if action_schema is None:
                action_schema = adapter.action_codec.schema
            carrier = stable_u64(
                int(observation_schema.schema_id),
                int(action_schema.schema_id),
                person=b"v8.58-env-schema-carrier",
            )
            event = runtime.make_experience(
                producer_id=1,
                producer_sequence=sequence,
                source_game_hash=int(adapter.identity.source_hash),
                global_step=max(0, runtime.watermark),
                context_signature=context,
                action_id=action,
                outcome_signature=outcome,
                family_signature=family,
                carrier_signature=carrier,
                future_option_delta=float(len(after_actions) - len(before_actions)),
                changed_cells=changed,
                terminal_polarity=int(boundary.primary_valence),
                trajectory_signature=trajectory,
                next_context_signature=after_context,
                prediction_error=prediction_error,
            )
            runtime.submit(event)
            submitted += 1
            if not boundary.continuation:
                episodes += 1
                if boundary.primary_valence > 0:
                    wins += 1
                elif boundary.primary_valence < 0:
                    losses += 1
                else:
                    draws += 1
                adapter.reset()
                trajectory = stable_u64(
                    adapter.identity.source_hash,
                    seed,
                    episodes,
                    person=b"v8.58-env-trajectory",
                )
        runtime.wait_quiescent(timeout=120.0)
        metrics = runtime.metrics()
        summary = {
            "environment": environment,
            "environment_family": adapter.identity.family,
            "environment_type": adapter.identity.environment_type,
            "source_hash": int(adapter.identity.source_hash),
            "steps_requested": int(steps),
            "events_submitted": int(submitted),
            "episodes": int(episodes),
            "wins": int(wins),
            "losses": int(losses),
            "draws": int(draws),
            "memories": int(metrics.get("memories", 0)),
            "edges": int(metrics.get("edges", 0)),
        }
        target = Path(root) / "multi_environment_summary.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary
    finally:
        try:
            adapter.close()
        finally:
            runtime.close(normal=True, timeout=120.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m v8.multi_environment_run")
    parser.add_argument("--environment", choices=("frozenlake", "chess"), required=True)
    parser.add_argument("--root", default="runs/v8/multi-environment")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--slippery", action="store_true")
    parser.add_argument("--chess-opponent", choices=("random", "first"), default="random")
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--stage-workers", type=int, default=1)
    parser.add_argument("--no-peers", action="store_true")
    args = parser.parse_args(argv)
    summary = run_environment(
        environment=args.environment,
        root=args.root,
        steps=args.steps,
        seed=args.seed,
        epsilon=args.epsilon,
        slippery=args.slippery,
        chess_opponent=args.chess_opponent,
        shards=args.shards,
        stage_workers=args.stage_workers,
        enable_peers=not args.no_peers,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
