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
    def __init__(
        self,
        cfg=None,
        model=None,
        wandb_logger=None,
        step: int = 0,
        parallel_rollout_manager=None,
        *,
        training_mode: str = "eval_policy",
        eval_kind: str = "policy",
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.wandb_logger = wandb_logger
        self.step = step
        self.training_mode = str(training_mode)
        self.eval_kind = str(eval_kind)
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
                return summary
            summary = collect_mode("policy_only" if configured_mode == "policy_only" else "planner_act")
            if self.wandb_logger is not None:
                self.wandb_logger.log_metrics(_wandb_eval_metrics(summary), step=self.step)
            _print_eval_summary(summary)
            return summary
        finally:
            self.shutdown()

    def evaluate_world_model(self, *, metrics_only: bool = True, per_game: bool = False) -> dict:
        try:
            artifact_writer = ArtifactWriter(Path(self.cfg.logging.output_dir) / self.cfg.logging.run_name / "eval")
            sequences = self._collect_sequences_for_mode("policy_only")
            world_rows = []
            by_game_rows: dict[str, list[dict[str, float]]] = {}
            baseline_changed = []
            baseline_reward = []
            baseline_done = []
            for sequence in sequences:
                device = next(self.model.parameters()).device
                hidden = sequence.initial_hidden_state.to(device)
                for step in sequence.timesteps:
                    obs = step.observation
                    encoded = self.model.encode_observation(
                        obs,
                        prev_action_id=step.previous_action.action_id,
                        prev_reward=step.previous_reward,
                        prev_done=step.previous_done,
                        hidden=hidden,
                    )
                    hidden = encoded.hidden.detach()
                    world = self.model.transition(encoded.latent, step.chosen_action.action_id)
                    if isinstance(world, tuple) and len(world) >= 5:
                        next_latent_pred, change_mask_logits, next_frame_logits, reward_pred, done_logit = world[:5]
                    else:
                        next_latent_pred, reward_pred, done_logit = world
                        change_mask_logits = None
                        next_frame_logits = None
                    row: dict[str, float] = {}
                    if change_mask_logits is not None and step.changed_cell_mask is not None:
                        target = step.changed_cell_mask.to(change_mask_logits.device).unsqueeze(0).to(dtype=torch.float32)
                        cm_loss = torch.nn.functional.binary_cross_entropy_with_logits(change_mask_logits, target)
                        pred = torch.sigmoid(change_mask_logits) >= 0.5
                        tgt = target >= 0.5
                        tp = (pred & tgt).sum().float()
                        fp = (pred & ~tgt).sum().float()
                        fn = (~pred & tgt).sum().float()
                        precision = tp / (tp + fp + 1e-8)
                        recall = tp / (tp + fn + 1e-8)
                        f1 = (2.0 * precision * recall) / (precision + recall + 1e-8)
                        row["change_mask_loss"] = float(cm_loss.detach().cpu())
                        row["change_mask_f1"] = float(f1.detach().cpu())
                        baseline_pred = torch.zeros_like(target, dtype=torch.bool)
                        b_tp = (baseline_pred & tgt).sum().float()
                        b_fp = (baseline_pred & ~tgt).sum().float()
                        b_fn = (~baseline_pred & tgt).sum().float()
                        b_prec = b_tp / (b_tp + b_fp + 1e-8)
                        b_rec = b_tp / (b_tp + b_fn + 1e-8)
                        b_f1 = (2.0 * b_prec * b_rec) / (b_prec + b_rec + 1e-8)
                        baseline_changed.append(float(b_f1.detach().cpu()))
                    if next_latent_pred is not None:
                        row["transition_loss"] = float(((next_latent_pred - encoded.latent.detach()) ** 2).mean().detach().cpu())
                    reward_t = torch.tensor([float(step.reward)], device=reward_pred.device, dtype=torch.float32)
                    reward_loss = torch.nn.functional.mse_loss(reward_pred.view(-1), reward_t.view(-1))
                    row["reward_prediction_loss"] = float(reward_loss.detach().cpu())
                    baseline_reward.append(float((reward_t**2).mean().detach().cpu()))
                    done_t = torch.tensor([1.0 if step.done else 0.0], device=done_logit.device, dtype=torch.float32)
                    done_loss = torch.nn.functional.binary_cross_entropy_with_logits(done_logit.view(-1), done_t.view(-1))
                    done_acc = float(((torch.sigmoid(done_logit.view(-1)) >= 0.5).float() == done_t.view(-1)).float().mean().detach().cpu())
                    row["done_prediction_loss"] = float(done_loss.detach().cpu())
                    row["done_accuracy"] = done_acc
                    baseline_done.append(float((done_t == 0.0).float().mean().detach().cpu()))
                    if next_frame_logits is not None:
                        next_frame_t = step.next_observation.current_frame.to(next_frame_logits.device).unsqueeze(0).to(dtype=torch.float32)
                        nf_loss = torch.nn.functional.mse_loss(next_frame_logits, next_frame_t)
                        row["next_frame_loss"] = float(nf_loss.detach().cpu())
                    world_rows.append(row)
                    by_game_rows.setdefault(str(sequence.game_id), []).append(row)
            summary = _mean_rows(world_rows)
            world_total = 0.0
            for key in ("transition_loss", "change_mask_loss", "next_frame_loss", "reward_prediction_loss", "done_prediction_loss"):
                if key in summary:
                    world_total += float(summary[key])
            summary["world_total_loss"] = float(world_total)
            summary["baseline_unchanged_board_change_mask_f1"] = float(sum(baseline_changed) / len(baseline_changed)) if baseline_changed else 0.0
            summary["baseline_always_zero_reward_mse"] = float(sum(baseline_reward) / len(baseline_reward)) if baseline_reward else 0.0
            summary["baseline_always_not_done_accuracy"] = float(sum(baseline_done) / len(baseline_done)) if baseline_done else 0.0
            per_game_summary = {gid: _mean_rows(rows) for gid, rows in by_game_rows.items()}
            if per_game:
                summary["per_game"] = per_game_summary
            artifact_writer.write_world_eval_summary(build_run_summary(self.cfg, summary | {"mode": self.training_mode}, mode=self.training_mode))
            artifact_writer.write_world_eval_per_game(per_game_summary)
            if self.wandb_logger is not None:
                self.wandb_logger.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))}, step=self.step)
            _print_world_eval_summary(summary)
            if not metrics_only:
                summary["policy_summary"] = self.evaluate_policy(episodes=None, per_game=False)
            return summary
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self.parallel_rollout_manager is not None and self._owns_parallel_rollout_manager:
            self.parallel_rollout_manager.close()
            self.parallel_rollout_manager = None

    def _collect_sequences_for_mode(self, mode_name: str):
        selector_cfg = copy.deepcopy(self.cfg)
        selector_cfg.acting.mode = "planner_act" if mode_name == "planner_act" else "policy_only"
        sequences = []
        if self.cfg.env.execution_mode == "parallel_workers":
            game_episode_counts = {str(game_id): int(self.cfg.evaluation.episodes) for game_id in self.cfg.env.game_ids}
            if self.parallel_rollout_manager is None:
                raise RuntimeError("parallel rollout manager is not initialized for evaluator")
            sequences = self.parallel_rollout_manager.collect(
                self.model,
                game_episode_counts=game_episode_counts,
                deterministic=self.cfg.evaluation.deterministic,
                evaluation=True,
                acting_mode=selector_cfg.acting.mode,
                collection_mode="rl",
            )
        else:
            selector = ActionSelector(selector_cfg)
            collector = RolloutCollector(self.cfg.rollout, selector)
            from rl_v1.env.adapter import ArcEnvironmentAdapter

            for worker_game_ids in _partition_game_ids(self.cfg.env.game_ids, self.cfg.runtime.rollout_processes):
                for game_id in worker_game_ids:
                    env = ArcEnvironmentAdapter(self.cfg.env, self.cfg.model, game_id, reward_cfg=self.cfg.reward)
                    try:
                        sequences.extend(
                            collector.collect(
                                self.model,
                                env,
                                episodes=self.cfg.evaluation.episodes,
                                deterministic=self.cfg.evaluation.deterministic,
                                evaluation=True,
                                collection_mode="rl",
                            )
                        )
                    finally:
                        if hasattr(env, "close"):
                            env.close()
        return sequences


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


def _print_world_eval_summary(summary: dict) -> None:
    label_map = {
        "world_total_loss": "loss",
        "transition_loss": "tr",
        "change_mask_loss": "cm",
        "change_mask_f1": "cm_f1",
        "reward_prediction_loss": "rw",
        "done_prediction_loss": "done",
        "next_frame_loss": "nf",
    }
    parts = []
    for key in ("world_total_loss", "transition_loss", "change_mask_loss", "change_mask_f1", "reward_prediction_loss", "done_prediction_loss", "next_frame_loss"):
        if key in summary and isinstance(summary[key], (int, float)):
            parts.append(f"{label_map[key]}={summary[key]:.4f}")
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


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = set().union(*(row.keys() for row in rows))
    return {
        key: float(sum(float(row.get(key, 0.0)) for row in rows) / len(rows))
        for key in keys
    }


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
