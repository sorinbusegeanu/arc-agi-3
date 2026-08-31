from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean

from v8 import ContinuousMemoryRuntime, V8RuntimeConfig
from v8.environments.synthetic_symbolic import SyntheticSymbolicEnvironment
from v8.grounding import GroundingMaturity
from v8.model import MemoryLevel, MemoryUid, stable_u64
from v8.modalities.symbols import DeterministicSymbolCodec
from v8.research.grounding_controls import CONTROL_SPECS, GroundingCondition, synthetic_config


@dataclass(frozen=True, slots=True)
class H16TrialMetrics:
    scientific_config_id: str
    condition: GroundingCondition
    seed: int
    trials: int
    prediction_improvement: float
    action_selection_improvement: float
    held_out_transfer: float
    false_transfer: float
    g4_count: int
    g5_count: int
    time_to_g4: int
    time_to_g5: int
    cross_modal_memory_count: int
    memory_growth: int
    candidate_search_compute: int
    negative_grounding_evidence: int
    held_out_causal_effect: float


@dataclass(frozen=True, slots=True)
class H16Report:
    scientific_config_id: str
    decision: str
    reason: str
    trials: tuple[H16TrialMetrics, ...]


def _higher_for_source(view, source_uid: MemoryUid) -> MemoryUid | None:
    by_uid = {row.uid: row for row in view.node_records()}
    candidates = []
    for edge in view.edge_records():
        if edge.target_uid != source_uid:
            continue
        row = by_uid.get(edge.source_uid)
        if row is not None and int(row.level) in {int(MemoryLevel.M2), int(MemoryLevel.M3)}:
            candidates.append(row)
    if not candidates:
        return None
    return min(candidates, key=lambda row: (-int(row.support_count), int(row.level), row.uid)).uid


def _run_stream(runtime, env: SyntheticSymbolicEnvironment, *, condition: GroundingCondition, seed: int, steps: int, producer_id: int) -> tuple[int, int, int]:
    spec = CONTROL_SPECS[condition]
    codec = DeterministicSymbolCodec("synthetic-symbols")
    runtime.hydra_v9.register_environment(env.identity)
    runtime.hydra_v9.start_episode(env.identity)
    wins = action_correct = actions_total = 0
    trajectory = stable_u64(env.identity.source_hash, seed, person=b"v9-h16-trajectory")
    for sequence in range(1, max(1, int(steps)) + 1):
        before = env.observe()
        context = int(env.cognitive_context_signature())
        if spec.publish_symbols:
            raw_tokens = tuple(env.passive_symbol_tokens())
            symbols = codec.encode_stream(raw_tokens, stream_name="instruction")
            runtime.hydra_v9.observe_symbol_observations(
                symbols,
                environment_identity=env.identity,
                context_signature=context,
                producer_id=producer_id + 100_000,
                raw_payload=" ".join(raw_tokens).encode("utf-8"),
            )
        actions = tuple(map(int, env.available_actions()))
        selected = runtime.read_view.best_action(context, actions) if spec.publish_world else None
        action = int(selected) if selected in actions else int(actions[(sequence + seed) % len(actions)])
        action_correct += int(action == 0)
        actions_total += 1
        before_actions = actions
        after = env.step(action)
        after_actions = tuple(map(int, env.available_actions()))
        boundary = env.cognitive_boundary_event()
        if spec.publish_world:
            outcome = int(env.cognitive_transition_signature(before, after))
            trajectory = stable_u64(trajectory, context, action, outcome, person=b"v9-h16-trajectory")
            event = runtime.make_experience(
                producer_id=int(producer_id),
                producer_sequence=int(sequence),
                source_game_hash=int(env.identity.source_hash),
                global_step=max(0, int(runtime.watermark)),
                context_signature=context,
                action_id=action,
                outcome_signature=outcome,
                family_signature=stable_u64(env.config.mechanic, person=b"v9-h16-family"),
                carrier_signature=stable_u64(env.observation_schema.schema_id, env.action_schema.schema_id, person=b"v9-h16-carrier"),
                future_option_delta=float(len(after_actions) - len(before_actions)),
                changed_cells=int(before != after),
                terminal_polarity=int(boundary.primary_valence),
                trajectory_signature=trajectory,
                next_context_signature=int(env.cognitive_context_signature()),
                prediction_error=0.0,
            )
            runtime.submit(event)
        wins += int(boundary.positive)
        if not boundary.continuation:
            env.reset()
            runtime.hydra_v9.start_episode(env.identity)
    return wins, action_correct, actions_total


def _maturity_summary(runtime) -> tuple[int, int, int, int, int]:
    rows = tuple(runtime.hydra_v9.grounding.states.values())
    g4 = [row for row in rows if row.maturity >= GroundingMaturity.G4]
    g5 = [row for row in rows if row.maturity >= GroundingMaturity.G5]
    first_g4 = min((row.first_g4_watermark for row in g4 if row.first_g4_watermark > 0), default=0)
    first_g5 = min((row.first_g5_watermark for row in g5 if row.first_g5_watermark > 0), default=0)
    negatives = sum(row.negative_evidence for row in rows)
    return len(g4), len(g5), first_g4, first_g5, negatives


def run_condition(root: Path, condition: GroundingCondition, *, seed: int, steps: int = 96, causal_intervention: bool = True) -> H16TrialMetrics:
    run_root = root / f"{condition.value}-seed{int(seed)}"
    config = V8RuntimeConfig.from_path(run_root, shards=2, stage_workers=1, enable_snapshots=False, restore=False, enable_peers=True)
    runtime = ContinuousMemoryRuntime(config)
    try:
        runtime.start()
        start_memories = int(runtime.metrics().get("memories", 0))
        env = SyntheticSymbolicEnvironment(synthetic_config(condition, seed=seed))
        _wins, action_correct, action_total = _run_stream(runtime, env, condition=condition, seed=seed, steps=steps, producer_id=10_000 + int(seed))
        runtime.wait_quiescent(timeout=60.0)
        if runtime.peers is not None:
            runtime.peers.run_once()
            runtime.wait_quiescent(timeout=60.0)
        shadow = runtime.hydra_v9.shadow_metrics()
        held_out_effect = 0.0

        if causal_intervention and condition is GroundingCondition.C2_ALIGNED and float(shadow.get("prediction_improvement", 0.0)) > 0.0:
            candidates = sorted(runtime.hydra_v9.grounding.states.values(), key=lambda row: (-int(row.maturity), -row.positive_evidence, row.key))
            for row in candidates:
                higher = _higher_for_source(runtime.read_view, row.key.source_symbol_uid)
                if higher is None:
                    continue
                runtime.hydra_v9.record_grounding_intervention(
                    source_symbol_uid=row.key.source_symbol_uid,
                    target_interaction_uid=row.key.target_interaction_uid,
                    environment_instance_id=row.key.environment_instance_id,
                    context_signature=row.key.context_scope_id,
                    effect=float(shadow["prediction_improvement"]),
                    held_out=False,
                    higher_memory_uid=higher,
                )
                break
            runtime.wait_quiescent(timeout=60.0)

            before_index = len(runtime.hydra_v9._shadow)
            heldout = SyntheticSymbolicEnvironment(synthetic_config(condition, seed=seed, held_out=True))
            _run_stream(runtime, heldout, condition=condition, seed=seed + 31, steps=max(24, steps // 3), producer_id=30_000 + int(seed))
            runtime.wait_quiescent(timeout=60.0)
            held_rows = runtime.hydra_v9._shadow[before_index:]
            if held_rows:
                baseline = mean(float(row.baseline_correct) for row in held_rows)
                symbol = mean(float(row.symbol_correct) for row in held_rows)
                held_out_effect = symbol - baseline
                if held_out_effect > 0.0:
                    for row in sorted(runtime.hydra_v9.grounding.states.values(), key=lambda item: (-int(item.maturity), -item.positive_evidence, item.key)):
                        higher = _higher_for_source(runtime.read_view, row.key.source_symbol_uid)
                        if higher is None:
                            continue
                        runtime.hydra_v9.record_grounding_intervention(
                            source_symbol_uid=row.key.source_symbol_uid,
                            target_interaction_uid=row.key.target_interaction_uid,
                            environment_instance_id=heldout.identity.source_hash,
                            context_signature=row.key.context_scope_id,
                            effect=held_out_effect,
                            held_out=True,
                            higher_memory_uid=higher,
                            source_environment_instance_id=row.key.environment_instance_id,
                        )
                        break
                    runtime.wait_quiescent(timeout=60.0)

        metrics = runtime.metrics()
        hydra = runtime.hydra_v9.metrics()
        g4, g5, t4, t5, negative = _maturity_summary(runtime)
        action_accuracy = action_correct / max(1, action_total)
        return H16TrialMetrics(
            scientific_config_id=runtime.hydra_v9.config.config_id,
            condition=condition,
            seed=int(seed),
            trials=1,
            prediction_improvement=float(shadow.get("prediction_improvement", 0.0)),
            action_selection_improvement=float(action_accuracy - 0.5),
            held_out_transfer=float(max(0.0, held_out_effect)),
            false_transfer=float(max(0.0, -held_out_effect)),
            g4_count=g4,
            g5_count=g5,
            time_to_g4=t4,
            time_to_g5=t5,
            cross_modal_memory_count=int(hydra.get("cross_modal_m1n_nodes", 0)),
            memory_growth=max(0, int(metrics.get("memories", 0)) - start_memories),
            candidate_search_compute=int(hydra.get("progressive_search", {}).get("candidate_search_compute", 0)),
            negative_grounding_evidence=negative,
            held_out_causal_effect=float(held_out_effect),
        )
    finally:
        runtime.close(normal=True, timeout=60.0)


def evaluate_h16(rows: tuple[H16TrialMetrics, ...]) -> H16Report:
    if not rows:
        return H16Report("", "INSUFFICIENT_EVIDENCE", "no H16 trials", ())
    config_ids = {row.scientific_config_id for row in rows}
    if len(config_ids) != 1:
        return H16Report("MIXED", "INVALID", "ScientificConfigId differs across conditions", rows)
    grouped = {condition: [row for row in rows if row.condition is condition] for condition in GroundingCondition}
    if any(not grouped[condition] for condition in GroundingCondition):
        return H16Report(next(iter(config_ids)), "INSUFFICIENT_EVIDENCE", "all C0-C3 conditions are required", rows)

    def effect(condition: GroundingCondition) -> float:
        values = grouped[condition]
        return mean(row.prediction_improvement + row.action_selection_improvement + row.held_out_transfer - row.false_transfer for row in values)

    c2 = effect(GroundingCondition.C2_ALIGNED)
    controls = max(effect(GroundingCondition.C0_INTERACTION_ONLY), effect(GroundingCondition.C1_SYMBOLS_ONLY), effect(GroundingCondition.C3_SHUFFLED))
    causal = any(row.g4_count > 0 and row.held_out_causal_effect > 0.0 for row in grouped[GroundingCondition.C2_ALIGNED])
    heldout = any(row.g5_count > 0 or row.held_out_transfer > 0.0 for row in grouped[GroundingCondition.C2_ALIGNED])
    if c2 > controls and causal and heldout:
        decision, reason = "VALID", "C2 separates C0/C1/C3 with causal and held-out interaction evidence"
    elif c2 > controls:
        decision, reason = "PARTIALLY_VALID", "C2 separates controls but causal/held-out grounding is incomplete"
    else:
        decision, reason = "INVALID", "aligned symbols do not outperform matched controls"
    return H16Report(next(iter(config_ids)), decision, reason, rows)


def run_h16(root: str | Path, *, seeds: tuple[int, ...] = (0, 1, 2), steps: int = 96) -> H16Report:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = tuple(run_condition(root, condition, seed=seed, steps=steps) for seed in seeds for condition in GroundingCondition)
    report = evaluate_h16(rows)
    target = root / "h16_grounding.json"
    target.write_text(json.dumps({"scientific_config_id": report.scientific_config_id, "decision": report.decision, "reason": report.reason, "trials": [{**asdict(row), "condition": row.condition.value} for row in report.trials]}, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m v8.research.h16_grounding")
    parser.add_argument("--root", default="runs/v8/h16-grounding")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=96)
    args = parser.parse_args(argv)
    seeds = tuple(int(value.strip()) for value in str(args.seeds).split(",") if value.strip())
    report = run_h16(args.root, seeds=seeds, steps=args.steps)
    print(json.dumps({"decision": report.decision, "reason": report.reason, "scientific_config_id": report.scientific_config_id}, sort_keys=True), flush=True)
    return 0 if report.decision in {"VALID", "PARTIALLY_VALID", "INSUFFICIENT_EVIDENCE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
