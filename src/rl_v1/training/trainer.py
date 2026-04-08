from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from rl_v1.data.rollout_collector import RolloutCollector
from rl_v1.env.adapter import ArcEnvironmentAdapter
from rl_v1.eval.evaluator import Evaluator
from rl_v1.metrics.metric_keys import LEVEL_COMPLETION_RATE
from rl_v1.metrics.training_loss_meter import TrainingLossMeter
from rl_v1.model.action_selector import ActionSelector
from rl_v1.model.latent_targets import NextLatentTargetBuilder
from rl_v1.model.model_factory import build_model
from rl_v1.training.fabric import build_fabric
from rl_v1.training.losses import compute_gae, compute_losses
from rl_v1.training.parallel_collector import ParallelRolloutManager
from rl_v1.utils.artifact_writer import ArtifactWriter
from rl_v1.utils.checkpoint_manager import CheckpointManager
from rl_v1.utils.io import ensure_dir
from rl_v1.utils.run_summary import build_run_summary


class Trainer:
    def __init__(self, cfg, model=None, wandb_logger=None) -> None:
        self.cfg = cfg
        self.wandb_logger = wandb_logger
        accelerator = _resolve_runtime_accelerator(cfg.runtime.accelerator)
        self.fabric = build_fabric(
            accelerator=accelerator,
            precision=cfg.runtime.precision,
            devices=cfg.runtime.devices,
        )
        self.model = build_model(cfg) if model is None else model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.optimization.learning_rate, weight_decay=cfg.optimization.weight_decay)
        self.scheduler = None
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.action_selector = ActionSelector(cfg)
        self.collector = RolloutCollector(cfg.rollout, self.action_selector)
        self.target_builder = NextLatentTargetBuilder()
        self.training_loss_meter = TrainingLossMeter()
        self.run_dir = ensure_dir(Path(cfg.logging.output_dir) / cfg.logging.run_name)
        self.artifacts = ArtifactWriter(self.run_dir / "artifacts")
        self.checkpoints = CheckpointManager()
        self.update_idx = 0
        self.best_level_completion_rate = float("-inf")
        self._envs = None
        self.parallel_rollout_manager = None
        self._closed = False
        if self.cfg.env.execution_mode == "parallel_workers":
            # runtime.rollout_processes is the active multiprocessing worker-count knob.
            self.parallel_rollout_manager = ParallelRolloutManager(cfg, self.action_selector)
            self.parallel_rollout_manager.start()

    def build_envs(self):
        envs = []
        for worker_game_ids in _partition_game_ids(self.cfg.env.game_ids, self.cfg.runtime.rollout_processes):
            for game_id in worker_game_ids:
                envs.append(ArcEnvironmentAdapter(self.cfg.env, self.cfg.model, game_id, reward_cfg=self.cfg.reward))
        return envs

    def maybe_restore(self):
        if self.cfg.checkpoint.restore_path:
            self.checkpoints.load(self.cfg.checkpoint.restore_path, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, cfg=self.cfg)

    def train(self, updates: int = 1, mode: str = "train_rl"):
        self.maybe_restore()
        self._ensure_envs()
        summary = {}
        collected_steps_total = 0
        try:
            for _ in range(updates):
                self.update_idx += 1
                collection_start = time.perf_counter()
                sequences = self._collect_sequences(
                    deterministic=False,
                    evaluation=False,
                    collection_mode="world_pretrain" if mode == "world_pretrain" else "rl",
                )
                collection_seconds = time.perf_counter() - collection_start
                collected_steps = sum(len(sequence.timesteps) for sequence in sequences)
                collected_steps_total += collected_steps
                summary = self._train_on_sequences(sequences, mode=mode)
                summary["collection_seconds"] = float(collection_seconds)
                _print_training_progress(self.update_idx, updates, summary.get("training_loss"))
                if self.update_idx % self.cfg.logging.log_every_updates == 0:
                    payload = build_run_summary(self.cfg, self.training_loss_meter.compute() | {"mode": mode})
                    payload["update_idx"] = self.update_idx
                    payload["effective_training_episodes_per_game"] = {
                        str(game_id): _episodes_to_collect_for_game(self.cfg, str(game_id))
                        for game_id in self.cfg.env.game_ids
                    }
                    for key in (
                        "collection_seconds",
                        "target_build_seconds",
                        "train_update_seconds",
                        "evaluation_seconds",
                        "optimizer_batches",
                        "valid_timesteps_per_update",
                        "mean_valid_timesteps_per_optimizer_batch",
                    ):
                        if key in summary:
                            payload[key] = summary[key]
                    self.artifacts.write_eval_summary(f"train_update_{self.update_idx:06d}", payload)
                    if self.wandb_logger is not None and self.update_idx % self.cfg.wandb.log_every_updates == 0:
                        self.wandb_logger.log_metrics(_wandb_training_metrics(summary), step=self.update_idx)
                    self.training_loss_meter.reset()
                if self.cfg.checkpoint.enabled and self.update_idx % self.cfg.checkpoint.save_every_updates == 0:
                    self.checkpoints.save(
                        self.run_dir / "last.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        cfg=self.cfg,
                        update_idx=self.update_idx,
                        model_variant=self.cfg.model.variant,
                        training_mode=mode,
                        seed=self.cfg.runtime.training_seed if mode != "world_pretrain" else self.cfg.runtime.world_pretrain_seed,
                    )
                if self.update_idx % self.cfg.logging.eval_every_updates == 0:
                    eval_start = time.perf_counter()
                    summary["evaluation"] = Evaluator(
                        self.cfg,
                        self.model,
                        wandb_logger=self.wandb_logger,
                        step=self.update_idx,
                        parallel_rollout_manager=self.parallel_rollout_manager,
                    ).evaluate()
                    summary["evaluation_seconds"] = float(time.perf_counter() - eval_start)
                    if self.cfg.checkpoint.enabled:
                        self.checkpoints.save(
                            self.run_dir / "last.pt",
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            cfg=self.cfg,
                            update_idx=self.update_idx,
                            model_variant=self.cfg.model.variant,
                            training_mode=mode,
                            seed=self.cfg.runtime.training_seed if mode != "world_pretrain" else self.cfg.runtime.world_pretrain_seed,
                            evaluation_deterministic=bool(self.cfg.evaluation.deterministic),
                        )
                    self._maybe_save_best_checkpoint(summary["evaluation"])
                else:
                    summary["evaluation_seconds"] = 0.0
            if updates > 0:
                sys.stdout.write(f"collected_steps={collected_steps_total}\n")
                sys.stdout.write("\n")
                sys.stdout.flush()
            return summary
        finally:
            self.shutdown()

    def _train_on_sequences(self, sequences, mode: str = "train_rl"):
        self.model.train()
        if not sequences:
            return self.training_loss_meter.compute()
        # Previous implementation used nested per-sequence/per-timestep Python loops.
        # This path batches across sequences and valid timesteps to improve GPU utilization.
        shuffled_sequences = list(sequences)
        random.shuffle(shuffled_sequences)
        seq_batches = list(_batched_sequences(shuffled_sequences, self.cfg.optimization.batch_size))
        target_cache = []
        target_build_start = time.perf_counter()
        for seq_batch in seq_batches:
            targets_padded, target_mask, target_extras = _build_batched_targets_for_seq_batch(
                self.target_builder,
                seq_batch,
                self.model,
                self.fabric.device,
            )
            target_cache.append((targets_padded, target_mask, target_extras))
        target_build_seconds = float(time.perf_counter() - target_build_start)
        rows = []
        optimizer_batches = 0
        valid_timesteps_per_update = 0
        train_update_start = time.perf_counter()
        for _epoch_idx in range(self.cfg.optimization.ppo_epochs):
            for batch_idx, seq_batch in enumerate(seq_batches):
                self.optimizer.zero_grad()
                packed = _pack_sequence_batch(seq_batch, self.fabric.device, self.cfg.optimization)
                if packed["timestep_mask"].sum().item() == 0:
                    continue
                model_out = _forward_sequence_batch_compat(self.model, packed)
                timestep_mask = packed["timestep_mask"]
                flat_valid = timestep_mask.view(-1)
                policy_logits_valid = model_out["policy_logits"].reshape(-1, model_out["policy_logits"].shape[-1])[flat_valid]
                values_valid = model_out["values"].reshape(-1)[flat_valid]
                latents_valid = model_out["latents"].reshape(-1, model_out["latents"].shape[-1])[flat_valid]
                chosen_actions_valid = packed["chosen_action_ids"].reshape(-1)[flat_valid].long()
                old_logprobs_valid = packed["old_action_logprobs"].reshape(-1)[flat_valid]
                returns_valid = packed["returns"].reshape(-1)[flat_valid]
                advantages_valid = packed["advantages"].reshape(-1)[flat_valid]
                reward_targets_valid = packed["rewards"].reshape(-1)[flat_valid]
                done_targets_valid = packed["done_flags"].reshape(-1)[flat_valid]
                for tensor_name, tensor in {
                    "policy_logits_valid": policy_logits_valid,
                    "values_valid": values_valid,
                    "latents_valid": latents_valid,
                    "returns_valid": returns_valid,
                    "advantages_valid": advantages_valid,
                }.items():
                    if not torch.isfinite(tensor).all().item():
                        raise ValueError(f"non-finite tensor after flattening: {tensor_name}")
                dist = torch.distributions.Categorical(logits=policy_logits_valid)
                new_logprob = dist.log_prob(chosen_actions_valid)
                entropy = dist.entropy()
                transition_pred = reward_pred = done_logit = transition_target = None
                if self.model.dynamics is not None:
                    transition_pred, reward_pred, done_logit = _transition_batch_compat(
                        self.model,
                        latents_valid,
                        chosen_actions_valid,
                    )
                    target_padded, target_mask, target_extras = target_cache[batch_idx]
                    if target_padded.shape[:2] != timestep_mask.shape:
                        raise ValueError(
                            "target latent/timestep mismatch for batched training: "
                            f"target_shape={tuple(target_padded.shape)}, mask_shape={tuple(timestep_mask.shape)}"
                        )
                    if not torch.equal(target_mask, timestep_mask):
                        raise ValueError("target latent mask does not align with timestep mask")
                    transition_target = target_padded.reshape(-1, target_padded.shape[-1])[flat_valid]
                    change_mask_target_valid = None
                    next_frame_target_valid = None
                    if "change_mask_target" in target_extras:
                        change_mask_target_valid = target_extras["change_mask_target"].reshape(
                            -1,
                            *target_extras["change_mask_target"].shape[2:],
                        )[flat_valid]
                    if "next_frame_target" in target_extras:
                        next_frame_target_valid = target_extras["next_frame_target"].reshape(
                            -1,
                            *target_extras["next_frame_target"].shape[2:],
                        )[flat_valid]
                else:
                    change_mask_target_valid = None
                    next_frame_target_valid = None
                if self.cfg.ablations.disable_transition_loss:
                    transition_pred = None
                    transition_target = None
                change_mask_logits_valid = None
                next_frame_logits_valid = None
                if model_out.get("change_mask_logits", None) is not None:
                    cm = model_out["change_mask_logits"]
                    change_mask_logits_valid = cm.reshape(-1, *cm.shape[2:])[flat_valid]
                if model_out.get("next_frame_logits", None) is not None:
                    nf = model_out["next_frame_logits"]
                    next_frame_logits_valid = nf.reshape(-1, *nf.shape[2:])[flat_valid]
                if mode == "world_pretrain":
                    batch_total, parts = compute_losses(
                        new_logprob=new_logprob.detach() * 0.0,
                        old_logprob=old_logprobs_valid.detach() * 0.0,
                        value_pred=values_valid.detach() * 0.0,
                        returns=returns_valid.detach() * 0.0,
                        advantages=advantages_valid.detach() * 0.0,
                        entropy=entropy.detach() * 0.0,
                        transition_pred=transition_pred,
                        transition_target=transition_target,
                        reward_pred=reward_pred,
                        reward_target=reward_targets_valid,
                        done_logit=done_logit,
                        done_target=done_targets_valid,
                        change_mask_logits=change_mask_logits_valid,
                        change_mask_target=change_mask_target_valid,
                        next_frame_logits=next_frame_logits_valid,
                        next_frame_target=next_frame_target_valid,
                        optimization_cfg=self.cfg.optimization,
                        loss_weights_cfg=self.cfg.loss_weights,
                        mode="world_pretrain",
                    )
                else:
                    batch_total, parts = compute_losses(
                        new_logprob=new_logprob,
                        old_logprob=old_logprobs_valid,
                        value_pred=values_valid,
                        returns=returns_valid,
                        advantages=advantages_valid,
                        entropy=entropy,
                        transition_pred=transition_pred,
                        transition_target=transition_target,
                        reward_pred=reward_pred,
                        reward_target=reward_targets_valid,
                        done_logit=done_logit,
                        done_target=done_targets_valid,
                        change_mask_logits=change_mask_logits_valid,
                        change_mask_target=change_mask_target_valid,
                        next_frame_logits=next_frame_logits_valid,
                        next_frame_target=next_frame_target_valid,
                        optimization_cfg=self.cfg.optimization,
                        loss_weights_cfg=self.cfg.loss_weights,
                        mode="train_rl",
                    )
                if not torch.isfinite(batch_total).item():
                    raise ValueError(f"non-finite batch loss: {float(batch_total.detach().cpu())}, parts={parts}")
                if not torch.isfinite(batch_total).item():
                    raise ValueError(f"non-finite batch_total before backward: {float(batch_total.detach().cpu())}")
                self.fabric.backward(batch_total)
                grad_total_sq = 0.0
                grad_max_abs = 0.0
                grad_nonfinite = 0
                for parameter in self.model.parameters():
                    if parameter.grad is None:
                        continue
                    grad = parameter.grad.detach()
                    finite = torch.isfinite(grad)
                    if not finite.all().item():
                        grad_nonfinite += 1
                    grad_total_sq += float((grad[finite] ** 2).sum().item()) if finite.any().item() else 0.0
                    if grad.numel() > 0:
                        grad_max_abs = max(grad_max_abs, float(grad.abs().max().item()))
                if grad_nonfinite > 0:
                    raise ValueError(f"non-finite gradients detected before optimizer step: count={grad_nonfinite}")
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optimization.grad_clip_norm)
                self.optimizer.step()
                for name, parameter in self.model.named_parameters():
                    if parameter is None:
                        continue
                    if not torch.isfinite(parameter.detach()).all().item():
                        raise ValueError(f"non-finite model parameter after optimizer step: {name}")
                self.training_loss_meter.update(float(batch_total.detach().cpu()))
                rows.append(parts)
                rows.append(
                    {
                        "grad_total_norm": grad_total_sq ** 0.5,
                        "grad_max_abs": grad_max_abs,
                        "nonfinite_grad_params": float(grad_nonfinite),
                    }
                )
                optimizer_batches += 1
                valid_timesteps = int(flat_valid.sum().item())
                valid_timesteps_per_update += valid_timesteps
        train_update_seconds = float(time.perf_counter() - train_update_start)
        summary = _mean_rows(rows) | self.training_loss_meter.compute()
        summary["target_build_seconds"] = target_build_seconds
        summary["train_update_seconds"] = train_update_seconds
        summary["optimizer_batches"] = optimizer_batches
        summary["valid_timesteps_per_update"] = valid_timesteps_per_update
        summary["mean_valid_timesteps_per_optimizer_batch"] = (
            float(valid_timesteps_per_update / optimizer_batches) if optimizer_batches > 0 else 0.0
        )
        return summary

    def shutdown(self) -> None:
        if self._closed:
            return
        if self.parallel_rollout_manager is not None:
            self.parallel_rollout_manager.close()
        else:
            for env in self._envs or []:
                if hasattr(env, "close"):
                    env.close()
        self._closed = True

    def _ensure_envs(self) -> None:
        if self.cfg.env.execution_mode == "parallel_workers":
            return
        if self._envs is None:
            self._envs = self.build_envs()

    def _collect_sequences(self, *, deterministic: bool, evaluation: bool, collection_mode: str = "rl"):
        if self.cfg.env.execution_mode == "parallel_workers":
            game_episode_counts = {
                str(game_id): _episodes_to_collect_for_game(self.cfg, str(game_id))
                for game_id in self.cfg.env.game_ids
            }
            if self.parallel_rollout_manager is None:
                raise RuntimeError("parallel rollout manager is not initialized")
            return self.parallel_rollout_manager.collect(
                self.model,
                game_episode_counts=game_episode_counts,
                deterministic=deterministic,
                evaluation=evaluation,
                collection_mode=collection_mode,
            )
        sequences = []
        for env in self._envs or []:
            episodes_to_collect = _episodes_to_collect_for_game(self.cfg, env.game_id)
            sequences.extend(
                    self.collector.collect(
                        self.model,
                        env,
                        episodes=episodes_to_collect,
                        deterministic=deterministic,
                        evaluation=evaluation,
                        collection_mode=collection_mode,
                    )
            )
        return sequences

    def _maybe_save_best_checkpoint(self, evaluation_summary: dict) -> None:
        if not self.cfg.checkpoint.enabled:
            return
        metric = _extract_level_completion_rate(evaluation_summary)
        if metric is None or metric <= self.best_level_completion_rate:
            return
        self.best_level_completion_rate = float(metric)
        self.checkpoints.save(
            self.run_dir / "best_level_completion_rate.pt",
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            cfg=self.cfg,
            update_idx=self.update_idx,
            model_variant=self.cfg.model.variant,
            training_mode="train_rl",
            seed=self.cfg.runtime.training_seed,
            evaluation_deterministic=bool(self.cfg.evaluation.deterministic),
        )


def _mean_rows(rows):
    if not rows:
        return {}
    keys = set().union(*(row.keys() for row in rows))
    return {key: sum(row.get(key, 0.0) for row in rows) / len(rows) for key in keys}


def _partition_game_ids(game_ids, num_workers: int):
    worker_count = max(1, min(int(num_workers), len(game_ids)))
    buckets = [[] for _ in range(worker_count)]
    for idx, game_id in enumerate(game_ids):
        buckets[idx % worker_count].append(game_id)
    return buckets


def _resolve_runtime_accelerator(runtime_value: str) -> str:
    value = str(runtime_value).lower()
    if value == "gpu":
        value = "cuda"
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU was requested but torch.cuda.is_available() is False")
        return "cuda"
    if value == "cpu":
        return "cpu"
    raise ValueError(f"unsupported runtime.accelerator value: {runtime_value}")


def _batched_sequences(sequences, batch_size: int):
    size = max(1, int(batch_size))
    for idx in range(0, len(sequences), size):
        yield sequences[idx : idx + size]


def _stack_observations(observations, device) -> dict[str, torch.Tensor]:
    if not observations:
        return {}
    return {
        "current_frame": torch.stack([obs.current_frame for obs in observations], dim=0).to(device),
        "previous_frame_1": torch.stack([obs.previous_frame_1 for obs in observations], dim=0).to(device),
        "previous_frame_2": torch.stack([obs.previous_frame_2 for obs in observations], dim=0).to(device),
        "valid_action_mask": torch.stack([obs.valid_action_mask for obs in observations], dim=0).to(device),
        "valid_pixel_mask": torch.stack([obs.valid_pixel_mask for obs in observations], dim=0).to(device),
        "game_id_index": torch.tensor([int(obs.game_id_index) for obs in observations], dtype=torch.long, device=device),
        "current_level_index": torch.tensor([int(obs.current_level_index) for obs in observations], dtype=torch.long, device=device),
        "step_count": torch.tensor([int(getattr(obs, "step_count", 0)) for obs in observations], dtype=torch.long, device=device),
        "changed_cell_mask": torch.stack(
            [
                (obs.changed_cell_mask if getattr(obs, "changed_cell_mask", None) is not None else torch.zeros_like(obs.current_frame))
                for obs in observations
            ],
            dim=0,
        ).to(device),
    }


def _pack_sequence_batch(seq_batch, device, optimization_cfg=None) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    gamma = 0.99 if optimization_cfg is None else float(optimization_cfg.gamma)
    gae_lambda = 0.95 if optimization_cfg is None else float(optimization_cfg.gae_lambda)
    batch_size = len(seq_batch)
    max_t = max((len(sequence.timesteps) for sequence in seq_batch), default=0)
    timestep_mask = torch.zeros((batch_size, max_t), dtype=torch.bool, device=device)
    previous_action_ids = torch.zeros((batch_size, max_t), dtype=torch.long, device=device)
    game_id_indices = torch.zeros((batch_size, max_t), dtype=torch.long, device=device)
    previous_rewards = torch.zeros((batch_size, max_t), dtype=torch.float32, device=device)
    previous_done_flags = torch.zeros((batch_size, max_t), dtype=torch.float32, device=device)
    current_level_indices = torch.zeros((batch_size, max_t), dtype=torch.long, device=device)
    step_counts = torch.zeros((batch_size, max_t), dtype=torch.long, device=device)
    chosen_action_ids = torch.zeros((batch_size, max_t), dtype=torch.long, device=device)
    old_action_logprobs = torch.zeros((batch_size, max_t), dtype=torch.float32, device=device)
    rewards = torch.zeros((batch_size, max_t), dtype=torch.float32, device=device)
    done_flags = torch.zeros((batch_size, max_t), dtype=torch.float32, device=device)
    returns = torch.zeros((batch_size, max_t), dtype=torch.float32, device=device)
    advantages = torch.zeros((batch_size, max_t), dtype=torch.float32, device=device)
    current_obs_list = []
    next_obs_list = []
    seq_lengths = []
    for sequence in seq_batch:
        seq_lengths.append(len(sequence.timesteps))
        for step in sequence.timesteps:
            current_obs_list.append(step.observation)
            next_obs_list.append(step.next_observation)
    current_obs_stacked = _stack_observations(current_obs_list, device=device)
    next_obs_stacked = _stack_observations(next_obs_list, device=device)
    if current_obs_stacked:
        frame_shape = current_obs_stacked["current_frame"].shape[1:]
        action_shape = current_obs_stacked["valid_action_mask"].shape[1:]
        pixel_shape = current_obs_stacked["valid_pixel_mask"].shape[1:]
        current_obs = {
            "current_frame": torch.zeros((batch_size, max_t, *frame_shape), dtype=current_obs_stacked["current_frame"].dtype, device=device),
            "previous_frame_1": torch.zeros((batch_size, max_t, *frame_shape), dtype=current_obs_stacked["previous_frame_1"].dtype, device=device),
            "previous_frame_2": torch.zeros((batch_size, max_t, *frame_shape), dtype=current_obs_stacked["previous_frame_2"].dtype, device=device),
            "valid_action_mask": torch.zeros((batch_size, max_t, *action_shape), dtype=torch.bool, device=device),
            "valid_pixel_mask": torch.zeros((batch_size, max_t, *pixel_shape), dtype=torch.bool, device=device),
            "changed_cell_mask": torch.zeros((batch_size, max_t, *frame_shape), dtype=torch.float32, device=device),
        }
        next_obs = {
            "current_frame": torch.zeros((batch_size, max_t, *frame_shape), dtype=next_obs_stacked["current_frame"].dtype, device=device),
            "previous_frame_1": torch.zeros((batch_size, max_t, *frame_shape), dtype=next_obs_stacked["previous_frame_1"].dtype, device=device),
            "previous_frame_2": torch.zeros((batch_size, max_t, *frame_shape), dtype=next_obs_stacked["previous_frame_2"].dtype, device=device),
            "valid_action_mask": torch.zeros((batch_size, max_t, *action_shape), dtype=torch.bool, device=device),
            "valid_pixel_mask": torch.zeros((batch_size, max_t, *pixel_shape), dtype=torch.bool, device=device),
            "changed_cell_mask": torch.zeros((batch_size, max_t, *frame_shape), dtype=torch.float32, device=device),
        }
    else:
        current_obs = {}
        next_obs = {}
    obs_idx = 0
    for b_idx, sequence in enumerate(seq_batch):
        rewards_list = [step.reward for step in sequence.timesteps]
        values_list = [step.value_estimate for step in sequence.timesteps]
        dones_list = [step.done for step in sequence.timesteps]
        terminal_chunk = sequence.chunk_end_reason == "terminal"
        seq_advantages, seq_returns = compute_gae(
            rewards_list,
            values_list,
            dones_list,
            gamma,
            gae_lambda,
            bootstrap_value=sequence.bootstrap_value,
            terminal_chunk=terminal_chunk,
        )
        for t_idx, step in enumerate(sequence.timesteps):
            timestep_mask[b_idx, t_idx] = True
            if current_obs:
                for key in current_obs.keys():
                    current_obs[key][b_idx, t_idx] = current_obs_stacked[key][obs_idx]
                    next_obs[key][b_idx, t_idx] = next_obs_stacked[key][obs_idx]
            previous_action_ids[b_idx, t_idx] = int(step.previous_action.action_id)
            game_id_indices[b_idx, t_idx] = int(current_obs_stacked["game_id_index"][obs_idx].item())
            previous_rewards[b_idx, t_idx] = float(step.previous_reward)
            previous_done_flags[b_idx, t_idx] = 1.0 if step.previous_done else 0.0
            current_level_indices[b_idx, t_idx] = int(current_obs_stacked["current_level_index"][obs_idx].item())
            step_counts[b_idx, t_idx] = int(current_obs_stacked["step_count"][obs_idx].item())
            chosen_action_ids[b_idx, t_idx] = int(step.chosen_action.action_id)
            old_action_logprobs[b_idx, t_idx] = float(step.action_logprob)
            rewards[b_idx, t_idx] = float(step.reward)
            done_flags[b_idx, t_idx] = 1.0 if step.done else 0.0
            returns[b_idx, t_idx] = float(seq_returns[t_idx].item())
            advantages[b_idx, t_idx] = float(seq_advantages[t_idx].item())
            obs_idx += 1
    initial_hidden_states = torch.stack(
        [sequence.initial_hidden_state.to(device).reshape(-1) for sequence in seq_batch],
        dim=0,
    )
    return {
        "current_obs": current_obs,
        "next_obs": next_obs,
        "timestep_mask": timestep_mask,
        "previous_action_ids": previous_action_ids,
        "game_id_indices": game_id_indices,
        "previous_rewards": previous_rewards,
        "previous_done_flags": previous_done_flags,
        "current_level_indices": current_level_indices,
        "step_counts": step_counts,
        "chosen_action_ids": chosen_action_ids,
        "old_action_logprobs": old_action_logprobs,
        "rewards": rewards,
        "done_flags": done_flags,
        "returns": returns,
        "advantages": advantages,
        "initial_hidden_states": initial_hidden_states,
    }


def _build_batched_targets_for_seq_batch(target_builder, seq_batch, model, device):
    if hasattr(target_builder, "build_next_latents_for_sequence_batch"):
        built = target_builder.build_next_latents_for_sequence_batch(seq_batch, model, device)
        if isinstance(built, tuple) and len(built) == 3:
            return built
        return built[0], built[1], {}
    batch_size = len(seq_batch)
    max_t = max((len(sequence.timesteps) for sequence in seq_batch), default=0)
    hidden_dim = int(getattr(model.cfg, "hidden_dim", getattr(model.cfg, "latent_dim", 0)))
    targets_padded = torch.zeros((batch_size, max_t, hidden_dim), dtype=torch.float32, device=device)
    timestep_mask = torch.zeros((batch_size, max_t), dtype=torch.bool, device=device)
    if hasattr(target_builder, "build_next_latents_for_sequence"):
        for b_idx, sequence in enumerate(seq_batch):
            targets = target_builder.build_next_latents_for_sequence(sequence, model).to(device).detach()
            if targets.shape[0] != len(sequence.timesteps):
                raise ValueError(
                    "target latent/timestep mismatch for sequence_id="
                    f"{sequence.sequence_id}: target_len={targets.shape[0]}, "
                    f"timestep_len={len(sequence.timesteps)}"
                )
            length = len(sequence.timesteps)
            if length > 0:
                targets_padded[b_idx, :length] = targets
                timestep_mask[b_idx, :length] = True
        return targets_padded, timestep_mask, {}
    if hasattr(target_builder, "build_next_latent_targets"):
        hidden_state_bundle = {
            sequence.sequence_id: sequence.initial_hidden_state
            for sequence in seq_batch
        }
        targets, _ = target_builder.build_next_latent_targets(seq_batch, model, hidden_state_bundle)
        targets = targets.to(device).detach()
        row_idx = 0
        for b_idx, sequence in enumerate(seq_batch):
            length = len(sequence.timesteps)
            if row_idx + length > targets.shape[0]:
                raise ValueError(
                    "target latent/timestep mismatch for sequence_id="
                    f"{sequence.sequence_id}: target_len={targets.shape[0]}, "
                    f"timestep_len={length}"
                )
            if length > 0:
                targets_padded[b_idx, :length] = targets[row_idx : row_idx + length]
                timestep_mask[b_idx, :length] = True
                row_idx += length
        return targets_padded, timestep_mask, {}
    raise AttributeError("target_builder does not expose a supported target-latent construction method")


def _forward_sequence_batch_compat(model, packed):
    if hasattr(model, "forward_sequence_batch"):
        return model.forward_sequence_batch(
            current_frame=packed["current_obs"]["current_frame"],
            previous_frame_1=packed["current_obs"]["previous_frame_1"],
            previous_frame_2=packed["current_obs"]["previous_frame_2"],
            valid_action_mask=packed["current_obs"]["valid_action_mask"],
            valid_pixel_mask=packed["current_obs"]["valid_pixel_mask"],
            prev_action_ids=packed["previous_action_ids"],
            game_id_indices=packed["game_id_indices"],
            prev_rewards=packed["previous_rewards"],
            prev_dones=packed["previous_done_flags"],
            current_level_indices=packed["current_level_indices"],
            step_counts=packed["step_counts"],
            chosen_action_ids=packed["chosen_action_ids"],
            initial_hidden=packed["initial_hidden_states"],
        )
    batch_size, timesteps = packed["timestep_mask"].shape
    hidden = packed["initial_hidden_states"]
    logits_steps = []
    values_steps = []
    latents_steps = []
    hiddens_steps = []
    for t_idx in range(timesteps):
        step_logits = []
        step_values = []
        step_latents = []
        step_hiddens = []
        for b_idx in range(batch_size):
            obs = SimpleNamespace(
                current_frame=packed["current_obs"]["current_frame"][b_idx, t_idx],
                previous_frame_1=packed["current_obs"]["previous_frame_1"][b_idx, t_idx],
                previous_frame_2=packed["current_obs"]["previous_frame_2"][b_idx, t_idx],
                valid_action_mask=packed["current_obs"]["valid_action_mask"][b_idx, t_idx],
                valid_pixel_mask=packed["current_obs"]["valid_pixel_mask"][b_idx, t_idx],
                game_id_index=int(packed["game_id_indices"][b_idx, t_idx].item()),
                current_level_index=int(packed["current_level_indices"][b_idx, t_idx].item()),
                step_count=int(packed["step_counts"][b_idx, t_idx].item()),
            )
            encoded = model.encode_observation(
                obs,
                int(packed["previous_action_ids"][b_idx, t_idx].item()),
                float(packed["previous_rewards"][b_idx, t_idx].item()),
                bool(packed["previous_done_flags"][b_idx, t_idx].item() > 0.5),
                hidden[b_idx : b_idx + 1],
            )
            step_logits.append(encoded.policy_logits.squeeze(0))
            step_values.append(encoded.value.reshape(-1)[0])
            step_latents.append(encoded.latent.squeeze(0))
            step_hiddens.append(encoded.hidden.squeeze(0))
        hidden = torch.stack(step_hiddens, dim=0)
        logits_steps.append(torch.stack(step_logits, dim=0))
        values_steps.append(torch.stack(step_values, dim=0))
        latents_steps.append(torch.stack(step_latents, dim=0))
        hiddens_steps.append(hidden)
    return {
        "policy_logits": torch.stack(logits_steps, dim=1),
        "values": torch.stack(values_steps, dim=1),
        "latents": torch.stack(latents_steps, dim=1),
        "hiddens": torch.stack(hiddens_steps, dim=1),
        "spatial_features": None,
        "click_logits": None,
    }


def _transition_batch_compat(model, latents: torch.Tensor, action_ids: torch.Tensor):
    if hasattr(model, "transition_batch"):
        outputs = model.transition_batch(latents, action_ids)
        if isinstance(outputs, tuple) and len(outputs) >= 5:
            next_latent, _change_mask, _next_frame, reward_pred, done_logit = outputs[:5]
            return next_latent, reward_pred, done_logit
        return outputs
    next_latents = []
    rewards = []
    done_logits = []
    for idx in range(latents.shape[0]):
        next_latent, reward, done_logit = model.transition(latents[idx : idx + 1], int(action_ids[idx].item()))
        next_latents.append(next_latent.reshape(1, -1))
        rewards.append(reward.reshape(-1)[0])
        done_logits.append(done_logit.reshape(-1)[0])
    return torch.cat(next_latents, dim=0), torch.stack(rewards, dim=0), torch.stack(done_logits, dim=0)


def _wandb_training_metrics(summary: dict) -> dict:
    mapping = {
        "training_loss": "training_loss",
        "policy_loss": "policy_loss",
        "value_loss": "value_loss",
        "entropy_bonus": "entropy",
        "latent_transition_loss": "transition_loss",
        "reward_prediction_loss": "reward_loss",
        "done_prediction_loss": "done_loss",
    }
    return {target: summary[source] for source, target in mapping.items() if source in summary}


def _print_training_progress(current_update: int, total_updates: int, training_loss) -> None:
    width = 24
    completed = 0 if total_updates <= 0 else int(width * current_update / total_updates)
    bar = "#" * completed + "-" * (width - completed)
    suffix = f" loss={float(training_loss):.4f}" if training_loss is not None else ""
    sys.stdout.write(f"\rtrain [{bar}] {current_update}/{total_updates}{suffix}")
    sys.stdout.flush()


def _extract_level_completion_rate(summary: dict) -> float | None:
    if LEVEL_COMPLETION_RATE in summary:
        return float(summary[LEVEL_COMPLETION_RATE])
    if "configured_mode_level_completion_rate" in summary:
        return float(summary["configured_mode_level_completion_rate"])
    return None


def _episodes_to_collect_for_game(cfg, game_id: str) -> int:
    base = cfg.rollout.num_episodes_per_collect
    multiplier = cfg.env.game_episode_multipliers.get(game_id, 1)
    return base * multiplier
