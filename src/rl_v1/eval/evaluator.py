from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

from rl_v1.data.rollout_collector import RolloutCollector
from rl_v1.eval.video_export import write_episode_video_from_observations
from rl_v1.metrics.metric_keys import GAME_WIN_RATE, LEVEL_COMPLETION_RATE, MEAN_LEVELS_REACHED, MEAN_STEPS_PER_COMPLETED_LEVEL
from rl_v1.metrics.multilevel_metrics import MultiLevelMetricAccumulator
from rl_v1.model.action_selector import ActionSelector, resolve_acting_mode
from rl_v1.training.parallel_collector import ParallelRolloutManager
from rl_v1.utils.artifact_writer import ArtifactWriter
from rl_v1.utils.run_summary import build_run_summary


class Evaluator:
    def __init__(self, cfg=None, model=None, wandb_logger=None, step: int = 0, parallel_rollout_manager=None) -> None:
        self.cfg = cfg
        self.model = model
        self.wandb_logger = wandb_logger
        self.step = step
        self.parallel_rollout_manager = parallel_rollout_manager
        self._owns_parallel_rollout_manager = False
        if (
            self.cfg is not None
            and self.model is not None
            and self.cfg.env.execution_mode == "parallel_workers"
            and self.parallel_rollout_manager is None
        ):
            # env.num_workers is deprecated for multiprocessing; runtime.rollout_processes is the active knob.
            self.parallel_rollout_manager = ParallelRolloutManager(self.cfg, self.cfg)
            self.parallel_rollout_manager.start()
            self._owns_parallel_rollout_manager = True

    def evaluate(self, episodes=None) -> dict:
        return self.evaluate_policy(episodes=episodes)

    def evaluate_policy(self, episodes=None, *, per_game: bool = False) -> dict:
        try:
            if self.cfg is None or self.model is None:
                accumulator = MultiLevelMetricAccumulator()
                for episode in episodes or []:
                    accumulator.on_game_start()
                    current_level_index = None
                    steps_in_level = 0
                    for step in episode:
                        level_index = step.current_level_index
                        if current_level_index is None or level_index != current_level_index:
                            accumulator.on_level_enter(level_index)
                            current_level_index = level_index
                            steps_in_level = 0
                        steps_in_level += 1
                        if step.level_completed:
                            accumulator.on_level_complete(level_index, steps_in_level)
                            steps_in_level = 0
                    if episode:
                        final_step = episode[-1]
                        accumulator.on_game_end(final_step.game_won, final_step.deepest_level_index)
                return accumulator.compute()

            artifact_writer = ArtifactWriter(Path(self.cfg.logging.output_dir) / self.cfg.logging.run_name / "eval")
            if self.cfg.acting.mode in {"planner_eval_only", "planner_act"}:
                assert (
                    self.cfg.planner.enabled and not self.cfg.ablations.disable_planner
                ), "planner must be available when acting.mode is planner_eval_only or planner_act"

            def collect_mode(mode_name: str):
                selector_cfg = copy.deepcopy(self.cfg)
                selector_cfg.acting.mode = "planner_act" if mode_name == "planner_act" else "policy_only"
                sequences = []
                try:
                    if self.cfg.env.execution_mode == "parallel_workers":
                        game_episode_counts = {
                            str(game_id): int(self.cfg.evaluation.episodes)
                            for game_id in self.cfg.env.game_ids
                        }
                        # Training can weight collection via env.game_episode_multipliers; evaluation is intentionally unweighted.
                        if self.parallel_rollout_manager is None:
                            raise RuntimeError("parallel rollout manager is not initialized for evaluator")
                        sequences = self.parallel_rollout_manager.collect(
                            self.model,
                            game_episode_counts=game_episode_counts,
                            deterministic=self.cfg.evaluation.deterministic,
                            evaluation=True,
                            acting_mode=selector_cfg.acting.mode,
                        )
                    else:
                        selector = ActionSelector(selector_cfg)
                        collector = RolloutCollector(self.cfg.rollout, selector)
                        from rl_v1.env.adapter import ArcEnvironmentAdapter

                        for worker_game_ids in _partition_game_ids(self.cfg.env.game_ids, self.cfg.runtime.rollout_processes):
                            for game_id in worker_game_ids:
                                env = ArcEnvironmentAdapter(self.cfg.env, self.cfg.model, game_id, reward_cfg=self.cfg.reward)
                                try:
                                    # Training can weight collection via env.game_episode_multipliers; evaluation is intentionally unweighted.
                                    sequences.extend(collector.collect(self.model, env, episodes=self.cfg.evaluation.episodes, deterministic=self.cfg.evaluation.deterministic, evaluation=True))
                                finally:
                                    if hasattr(env, "close"):
                                        env.close()
                except Exception as exc:
                    sys.stdout.write(f"eval collection failure mode={mode_name}: {exc}\n")
                    sys.stdout.flush()
                    raise
                video_root = None
                if bool(getattr(self.cfg.env, "save_recording", False)):
                    video_root = Path(self.cfg.logging.output_dir) / self.cfg.logging.run_name / "eval" / f"{mode_name}_videos"
                summary, episode_rows, planner_traces = _summarize_sequences(sequences, video_root=video_root)
                for row in episode_rows:
                    row["deterministic"] = bool(self.cfg.evaluation.deterministic)
                    row["checkpoint_path"] = self.cfg.checkpoint.restore_path
                payload = build_run_summary(self.cfg, summary | {"mode": "eval_policy"})
                artifact_writer.write_eval_summary(mode_name, payload)
                artifact_writer.write_episode_summaries(mode_name, episode_rows)
                if planner_traces:
                    artifact_writer.write_planner_trace(mode_name, planner_traces)
                return summary

            configured_mode = resolve_acting_mode(self.cfg, evaluation=True)
            run_paired_modes = bool(getattr(self.cfg.evaluation, "compare_policy_vs_configured", False))
            if run_paired_modes and configured_mode != "policy_only" and not self.cfg.ablations.disable_planner:
                policy = collect_mode("policy_only")
                configured = collect_mode("planner_act")
                summary = {
                    "policy_only_game_win_rate": policy[GAME_WIN_RATE],
                    "policy_only_level_completion_rate": policy[LEVEL_COMPLETION_RATE],
                    "policy_only_mean_levels_reached": policy[MEAN_LEVELS_REACHED],
                    "policy_only_mean_steps_per_completed_level": policy[MEAN_STEPS_PER_COMPLETED_LEVEL],
                    "configured_mode_game_win_rate": configured[GAME_WIN_RATE],
                    "configured_mode_level_completion_rate": configured[LEVEL_COMPLETION_RATE],
                    "configured_mode_mean_levels_reached": configured[MEAN_LEVELS_REACHED],
                    "configured_mode_mean_steps_per_completed_level": configured[MEAN_STEPS_PER_COMPLETED_LEVEL],
                }
                if self.wandb_logger is not None:
                    self.wandb_logger.log_metrics(_wandb_eval_metrics(summary), step=self.step)
                _print_eval_summary(summary)
                if per_game:
                    summary["per_game"] = _per_game_metrics(episode_rows)
                return summary
            summary = collect_mode("policy_only" if configured_mode == "policy_only" else "planner_act")
            if self.wandb_logger is not None:
                self.wandb_logger.log_metrics(_wandb_eval_metrics(summary), step=self.step)
            _print_eval_summary(summary)
            return summary
        finally:
            self.shutdown()

    def evaluate_world_model(self, *, metrics_only: bool = False) -> dict:
        # Teacher-forced world-model validation placeholder:
        # use existing rollout collection and compute lightweight prediction diagnostics when available.
        policy_summary = self.evaluate_policy(episodes=None, per_game=False)
        world_summary = {
            "latent_transition_mse": 0.0,
            "change_mask_loss": 0.0,
            "reward_prediction_mse": 0.0,
            "done_prediction_accuracy": 0.0,
        }
        if not metrics_only:
            world_summary["policy_summary"] = policy_summary
        return world_summary

    def shutdown(self) -> None:
        if self.parallel_rollout_manager is not None and self._owns_parallel_rollout_manager:
            self.parallel_rollout_manager.close()
            self.parallel_rollout_manager = None


def _summarize_sequences(sequences, *, video_root: Path | None = None):
    accumulator = MultiLevelMetricAccumulator()
    episode_rows = []
    planner_traces = []
    empty_valid_action_mask_count = 0
    by_episode = {}
    for sequence in sequences:
        by_episode.setdefault(sequence.episode_id, []).append(sequence)
    for episode_id in sorted(by_episode):
        episode_sequences = sorted(by_episode[episode_id], key=lambda seq: _sequence_sort_key(seq.sequence_id))
        game_ids = {seq.game_id for seq in episode_sequences}
        env_instance_ids = {seq.env_instance_id for seq in episode_sequences}
        assert len(game_ids) == 1, f"grouped episode {episode_id} contains multiple game ids: {game_ids}"
        assert len(env_instance_ids) == 1, f"grouped episode {episode_id} contains multiple env instance ids: {env_instance_ids}"
        game_id = next(iter(game_ids))
        env_instance_id = next(iter(env_instance_ids))
        accumulator.on_game_start()
        starting_level = None
        current_level = None
        steps_in_level = 0
        levels_completed = 0
        total_steps = 0
        deepest_level_reached = None
        terminal_level_index = None
        game_won = False
        planner = {}
        step_diagnostics = []
        observed_level_indices: list[int] = []
        previous_levels_completed_meta: int | None = None
        for sequence in episode_sequences:
            for step in sequence.timesteps:
                level_index = int(step.observation.current_level_index)
                next_level_index = int(step.next_observation.current_level_index)
                levels_completed_meta = step.next_observation.raw_metadata.get("levels_completed")
                levels_completed_meta = int(levels_completed_meta) if isinstance(levels_completed_meta, (int, float)) else None
                observed_level_indices.append(level_index)
                observed_level_indices.append(next_level_index)
                if starting_level is None:
                    starting_level = level_index
                if deepest_level_reached is None:
                    deepest_level_reached = level_index
                else:
                    deepest_level_reached = max(deepest_level_reached, level_index, next_level_index)
                if current_level is None:
                    accumulator.on_level_enter(level_index)
                    current_level = level_index
                    steps_in_level = 0
                # Keep episode-level progression monotonic; noisy observation-level regressions are ignored.
                level_index = max(level_index, current_level)
                steps_in_level += 1
                total_steps += 1
                if next_level_index < current_level:
                    raise ValueError(
                        f"invalid level regression in episode {episode_id}: "
                        f"current_level={current_level}, next_level={next_level_index}, "
                        f"raw={step.next_observation.raw_metadata}"
                    )
                completion_via_level_completed = bool(step.next_observation.level_completed)
                completion_via_meta = (
                    levels_completed_meta is not None
                    and previous_levels_completed_meta is not None
                    and levels_completed_meta > previous_levels_completed_meta
                )
                if completion_via_level_completed and not bool(step.next_observation.game_won):
                    if next_level_index <= current_level and not completion_via_meta:
                        raise ValueError(
                            f"invalid level progression in episode {episode_id}: "
                            f"level_completed=True without upward transition "
                            f"(current={current_level}, next={next_level_index}), "
                            f"raw={step.next_observation.raw_metadata}"
                        )
                if completion_via_level_completed:
                    accumulator.on_level_complete(current_level, steps_in_level)
                    levels_completed += 1
                    steps_in_level = 0
                if next_level_index > current_level:
                    delta = next_level_index - current_level
                    already_counted = 1 if completion_via_level_completed else 0
                    remaining = max(0, delta - already_counted)
                    if remaining > 0:
                        for _ in range(remaining):
                            accumulator.on_level_complete(current_level, max(1, steps_in_level))
                        levels_completed += remaining
                    current_level = next_level_index
                    accumulator.on_level_enter(current_level)
                    steps_in_level = 0
                terminal_level_index = next_level_index
                game_won = game_won or bool(step.next_observation.game_won)
                if step.next_observation.game_won and not step.next_observation.terminal:
                    raise ValueError(
                        f"inconsistent game_won terminal state in episode {episode_id}: "
                        f"raw={step.next_observation.raw_metadata}"
                    )
                if not planner:
                    planner = step.extras.get("diagnostics", {}).get("planner", {})
                valid_action_mask_empty = bool(step.extras.get("diagnostics", {}).get("valid_action_mask_empty", False))
                if valid_action_mask_empty:
                    empty_valid_action_mask_count += 1
                top1_action_id, top1_action_prob, top2_action_id, top2_action_prob = _extract_top2_action_probs(
                    step.extras.get("policy_logits")
                )
                step_diagnostics.append(
                    {
                        "step_idx": int(step.extras.get("step_idx", 0)),
                        "top1_action_id": top1_action_id,
                        "top1_action_prob": top1_action_prob,
                        "top2_action_id": top2_action_id,
                        "top2_action_prob": top2_action_prob,
                        "valid_action_mask_empty": valid_action_mask_empty,
                        "revisit_match_count": step.next_observation.raw_metadata.get("revisit_match_count"),
                        "same_action_streak": step.next_observation.raw_metadata.get("same_action_streak"),
                        "revisit_window_size": step.next_observation.raw_metadata.get("revisit_window_size"),
                        "repeat_penalty_applied": step.next_observation.raw_metadata.get("repeat_penalty_applied"),
                        "same_action_penalty_applied": step.next_observation.raw_metadata.get("same_action_penalty_applied"),
                    }
                )
                if levels_completed_meta is not None:
                    previous_levels_completed_meta = levels_completed_meta
        if starting_level is not None and deepest_level_reached is not None:
            if observed_level_indices:
                deepest_level_reached = max(observed_level_indices + [deepest_level_reached, terminal_level_index if terminal_level_index is not None else deepest_level_reached])
            assert levels_completed == 0 or deepest_level_reached >= starting_level
            assert not game_won or deepest_level_reached >= starting_level
            if levels_completed > max(0, deepest_level_reached - starting_level) and not game_won:
                raise ValueError(
                    f"inconsistent level metrics in episode {episode_id}: "
                    f"levels_completed={levels_completed}, starting_level={starting_level}, "
                    f"deepest_level_reached={deepest_level_reached}"
                )
            accumulator.on_game_end(game_won, deepest_level_reached)
            episode_row = {
                "game_id": game_id,
                "env_instance_id": env_instance_id,
                "seed": episode_sequences[0].timesteps[0].observation.raw_metadata.get("seed"),
                "won": game_won,
                "starting_level_index": starting_level,
                "deepest_level_reached": deepest_level_reached,
                "levels_completed": levels_completed,
                "terminal_level_index": terminal_level_index,
                "total_steps": total_steps,
                "acting_mode": episode_sequences[0].timesteps[0].extras.get("diagnostics", {}).get("acting_mode"),
                "deterministic": None,
                "checkpoint_path": None,
            }
            if video_root is not None:
                try:
                    observations = _episode_observation_stream(episode_sequences)
                    video_path = write_episode_video_from_observations(
                        observations,
                        output_root=video_root / _safe_episode_name(episode_id),
                        fps=2,
                    )
                    episode_row["video_path"] = video_path
                except Exception as exc:
                    episode_row["video_error"] = str(exc)
            episode_rows.append(episode_row)
            if planner:
                planner_traces.append(
                    {
                        "chosen_first_action": planner.get("chosen_first_action"),
                        "branch_length": planner.get("branch_length"),
                        "planner_score": planner.get("planner_score"),
                        "step_diagnostics": step_diagnostics,
                    }
                )
    metrics = accumulator.compute()
    assert not (metrics[LEVEL_COMPLETION_RATE] > 0.0 and metrics[MEAN_LEVELS_REACHED] == 0.0)
    summary = {
        GAME_WIN_RATE: metrics[GAME_WIN_RATE],
        LEVEL_COMPLETION_RATE: metrics[LEVEL_COMPLETION_RATE],
        MEAN_LEVELS_REACHED: metrics[MEAN_LEVELS_REACHED],
        MEAN_STEPS_PER_COMPLETED_LEVEL: metrics[MEAN_STEPS_PER_COMPLETED_LEVEL],
    }
    summary["empty_valid_action_mask_count"] = int(empty_valid_action_mask_count)
    return summary, episode_rows, planner_traces


def _episode_observation_stream(episode_sequences):
    stream = []
    first_added = False
    for sequence in episode_sequences:
        for step in sequence.timesteps:
            if not first_added:
                stream.append(step.observation)
                first_added = True
            stream.append(step.next_observation)
    return stream


def _safe_episode_name(episode_id: str) -> str:
    out = []
    for ch in str(episode_id):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _partition_game_ids(game_ids, num_workers: int):
    worker_count = max(1, min(int(num_workers), len(game_ids)))
    buckets = [[] for _ in range(worker_count)]
    for idx, game_id in enumerate(game_ids):
        buckets[idx % worker_count].append(game_id)
    return buckets


def _sequence_sort_key(sequence_id: str):
    if sequence_id.startswith("seq-"):
        try:
            return int(sequence_id.split("-", 1)[1])
        except ValueError:
            return sequence_id
    return sequence_id


def _wandb_eval_metrics(summary: dict) -> dict:
    allowed = {
        "game_win_rate",
        "level_completion_rate",
        "mean_levels_reached",
        "mean_steps_per_completed_level",
        "policy_only_game_win_rate",
        "policy_only_level_completion_rate",
        "policy_only_mean_levels_reached",
        "policy_only_mean_steps_per_completed_level",
        "configured_mode_game_win_rate",
        "configured_mode_level_completion_rate",
        "configured_mode_mean_levels_reached",
        "configured_mode_mean_steps_per_completed_level",
    }
    return {key: value for key, value in summary.items() if key in allowed and isinstance(value, (int, float))}


def _print_eval_summary(summary: dict) -> None:
    label_map = {
        "game_win_rate": "gwr",
        "level_completion_rate": "lcr",
        "mean_levels_reached": "mlr",
        "mean_steps_per_completed_level": "mspl",
        "policy_only_game_win_rate": "po_gwr",
        "policy_only_level_completion_rate": "po_lcr",
        "policy_only_mean_levels_reached": "po_mlr",
        "policy_only_mean_steps_per_completed_level": "po_mspl",
        "configured_mode_game_win_rate": "cfg_gwr",
        "configured_mode_level_completion_rate": "cfg_lcr",
        "configured_mode_mean_levels_reached": "cfg_mlr",
        "configured_mode_mean_steps_per_completed_level": "cfg_mspl",
    }
    ordered_keys = list(label_map.keys())
    parts = [
        f"{label_map[key]}={summary[key]:.4f}"
        for key in ordered_keys
        if key in summary and isinstance(summary[key], (int, float))
    ]
    sys.stdout.write("eval " + " ".join(parts) + "\n")
    sys.stdout.flush()


def _extract_top2_action_probs(policy_logits) -> tuple[int | None, float, int | None, float]:
    if policy_logits is None:
        return None, 0.0, None, 0.0
    logits = torch.tensor(policy_logits, dtype=torch.float32).reshape(-1)
    probs = torch.softmax(logits, dim=0)
    topk = min(2, probs.numel())
    values, indices = torch.topk(probs, k=topk)
    top1_id = int(indices[0].item()) if topk >= 1 else None
    top1_prob = float(values[0].item()) if topk >= 1 else 0.0
    if topk >= 2:
        top2_id = int(indices[1].item())
        top2_prob = float(values[1].item())
    else:
        top2_id = None
        top2_prob = 0.0
    if topk == 1:
        top2_id = None
        top2_prob = 0.0
    return top1_id, top1_prob, top2_id, top2_prob


def _per_game_metrics(episode_rows: list[dict]) -> dict[str, dict[str, float]]:
    by_game: dict[str, dict[str, float]] = {}
    for row in episode_rows:
        game_id = str(row.get("game_id", "unknown"))
        bucket = by_game.setdefault(
            game_id,
            {
                "episodes": 0.0,
                "wins": 0.0,
                "levels_completed": 0.0,
                "levels_entered_proxy": 0.0,
                "levels_reached_sum": 0.0,
                "steps_completed_sum": 0.0,
                "completed_level_count": 0.0,
            },
        )
        bucket["episodes"] += 1.0
        bucket["wins"] += 1.0 if bool(row.get("won", False)) else 0.0
        completed = float(row.get("levels_completed", 0))
        bucket["levels_completed"] += completed
        start = float(row.get("starting_level_index", 0))
        deepest = float(row.get("deepest_level_reached", start))
        bucket["levels_reached_sum"] += deepest
        bucket["levels_entered_proxy"] += max(1.0, deepest - start + 1.0)
        if completed > 0:
            bucket["steps_completed_sum"] += float(row.get("total_steps", 0))
            bucket["completed_level_count"] += completed
    out: dict[str, dict[str, float]] = {}
    for game_id, b in by_game.items():
        episodes = max(1.0, b["episodes"])
        entered = max(1.0, b["levels_entered_proxy"])
        completed_count = max(1.0, b["completed_level_count"])
        out[game_id] = {
            "game_win_rate": b["wins"] / episodes,
            "level_completion_rate": b["levels_completed"] / entered,
            "mean_levels_reached": b["levels_reached_sum"] / episodes,
            "mean_steps_per_completed_level": b["steps_completed_sum"] / completed_count if b["completed_level_count"] > 0 else 0.0,
        }
    return out
