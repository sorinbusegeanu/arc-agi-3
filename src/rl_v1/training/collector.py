from __future__ import annotations

import hashlib
from dataclasses import replace

import torch

from rl_v1.configs.schema import RolloutConfig
from rl_v1.data.contracts import RolloutSequence, RolloutTimestep, V1Action
from rl_v1.env.adapter import ArcEnvironmentAdapter


class RolloutCollector:
    def __init__(self, rollout_cfg: RolloutConfig, action_selector_or_legacy) -> None:
        self.cfg = rollout_cfg
        self.action_selector = action_selector_or_legacy
        self._episode_counters_by_env: dict[str, int] = {}

    def collect(
        self,
        model,
        env: ArcEnvironmentAdapter,
        episodes: int,
        deterministic: bool = False,
        evaluation: bool = False,
        eval_episode_start_idx: int | None = None,
    ) -> list[RolloutSequence]:
        sequences: list[RolloutSequence] = []
        sequence_counter = 0
        max_steps = int(getattr(self.cfg, "max_steps_per_level", getattr(self.cfg, "max_steps", 0)))
        model_device = next(model.parameters()).device
        for episode_idx in range(int(episodes)):
            global_episode_idx = (
                int(eval_episode_start_idx) + int(episode_idx)
                if eval_episode_start_idx is not None
                else None
            )
            env_instance_id = str(getattr(env, "env_instance_id", "env-unknown"))
            game_id = str(getattr(env, "game_id", "game-unknown"))
            episode_counter = self._next_episode_counter(
                env_instance_id,
                evaluation=evaluation,
                eval_episode_idx=0 if global_episode_idx is None else global_episode_idx,
            )
            reset_seed = self._build_reset_seed(
                base_seed=int(env.env_cfg.seed),
                env_instance_id=env_instance_id,
                game_id=game_id,
                episode_counter=episode_counter,
                evaluation=evaluation,
                eval_episode_idx=0 if global_episode_idx is None else global_episode_idx,
                global_episode_idx=global_episode_idx,
            )
            torch.manual_seed(int(reset_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(reset_seed))
            obs = env.reset(seed=reset_seed)
            episode_id = f"{env_instance_id}:{game_id}:ep{episode_counter}"
            hidden = model.initial_hidden(1, model_device).detach()
            prev_action = V1Action(action_id=0)
            prev_reward = 0.0
            prev_done = True
            active = RolloutSequence(
                sequence_id=f"seq-{sequence_counter}",
                episode_id=episode_id,
                env_instance_id=env_instance_id,
                game_id=game_id,
                episode_counter=episode_counter,
                sequence_start=True,
                initial_hidden_state=hidden.detach().cpu(),
            )
            for step_idx in range(max_steps):
                action, policy_output, next_hidden, diagnostics, entropy = self._select_action(
                    model=model,
                    observation=obs,
                    prev_action=prev_action,
                    prev_reward=prev_reward,
                    prev_done=prev_done,
                    hidden=hidden,
                    deterministic=deterministic,
                    evaluation=evaluation,
                )
                next_obs = env.step(action)
                timestep = RolloutTimestep(
                    observation=obs,
                    previous_action=prev_action,
                    previous_reward=prev_reward,
                    previous_done=prev_done,
                    chosen_action=action,
                    action_logprob=float(policy_output.action_logprob.item()),
                    value_estimate=float(policy_output.value.item()),
                    reward=float(next_obs.reward),
                    done=bool(next_obs.terminal),
                    next_observation=next_obs,
                    hidden_state=hidden.detach().cpu(),
                    extras={
                        "step_idx": step_idx,
                        "diagnostics": diagnostics,
                        "policy_logits": (
                            policy_output.action_logits.detach().cpu().reshape(-1).tolist()
                            if getattr(policy_output, "action_logits", None) is not None
                            else None
                        ),
                        "entropy": float(entropy.detach().cpu()),
                        "episode_id": episode_id,
                        "env_instance_id": env_instance_id,
                        "game_id": game_id,
                        "reset_seed": reset_seed,
                    },
                )
                active.timesteps.append(timestep)
                prev_action = action
                prev_reward = timestep.reward
                prev_done = timestep.done
                obs = next_obs
                hidden = next_hidden.detach()
                if len(active.timesteps) >= self.cfg.unroll_length and not timestep.done:
                    active.truncated = True
                    active.chunk_end_reason = "truncation"
                    active.bootstrap_value = self._bootstrap_value(
                        model=model,
                        observation=obs,
                        prev_action=prev_action,
                        prev_reward=prev_reward,
                        prev_done=prev_done,
                        hidden=hidden,
                    )
                    sequences.append(active)
                    sequence_counter += 1
                    active = RolloutSequence(
                        sequence_id=f"seq-{sequence_counter}",
                        episode_id=episode_id,
                        env_instance_id=env_instance_id,
                        game_id=game_id,
                        episode_counter=episode_counter,
                        sequence_start=False,
                        initial_hidden_state=hidden.detach().cpu(),
                    )
                if timestep.done:
                    active.chunk_end_reason = "terminal"
                    active.bootstrap_value = 0.0
                    break
            if active.timesteps:
                if not active.chunk_end_reason:
                    if active.timesteps[-1].done:
                        active.chunk_end_reason = "terminal"
                        active.bootstrap_value = 0.0
                    else:
                        active.truncated = True
                        active.chunk_end_reason = "truncation"
                        active.bootstrap_value = self._bootstrap_value(
                            model=model,
                            observation=obs,
                            prev_action=prev_action,
                            prev_reward=prev_reward,
                            prev_done=prev_done,
                            hidden=hidden,
                        )
                sequences.append(active)
                sequence_counter += 1
        return sequences

    def _select_action(
        self,
        *,
        model,
        observation,
        prev_action,
        prev_reward,
        prev_done,
        hidden,
        deterministic: bool,
        evaluation: bool,
    ):
        if hasattr(self.action_selector, "select_action"):
            inference_observation = _observation_to_device(observation, hidden.device)
            return self.action_selector.select_action(
                model,
                inference_observation,
                prev_action.action_id,
                prev_reward,
                prev_done,
                hidden,
                deterministic=deterministic,
                evaluation=evaluation,
            )
        action, policy_output, next_hidden, diagnostics = model.act(
            observation=_observation_to_device(observation, hidden.device),
            prev_action_id=prev_action.action_id,
            prev_reward=prev_reward,
            prev_done=prev_done,
            hidden=hidden,
            acting_mode=getattr(self.action_selector, "acting_mode", "policy_only"),
            deterministic=deterministic,
        )
        entropy = torch.distributions.Categorical(logits=policy_output.action_logits).entropy()
        return action, policy_output, next_hidden, diagnostics, entropy

    def _bootstrap_value(self, *, model, observation, prev_action, prev_reward, prev_done, hidden) -> float:
        with torch.no_grad():
            encoded = model.encode_observation(
                _observation_to_device(observation, hidden.device),
                prev_action_id=prev_action.action_id,
                prev_reward=prev_reward,
                prev_done=prev_done,
                hidden=hidden,
            )
            return float(encoded.value.detach().cpu().item())

    def _next_episode_counter(self, env_instance_id: str, *, evaluation: bool, eval_episode_idx: int) -> int:
        if evaluation:
            return eval_episode_idx + 1
        current = self._episode_counters_by_env.get(env_instance_id, 0) + 1
        self._episode_counters_by_env[env_instance_id] = current
        return current

    def _build_reset_seed(
        self,
        *,
        base_seed: int,
        env_instance_id: str,
        game_id: str,
        episode_counter: int,
        evaluation: bool,
        eval_episode_idx: int,
        global_episode_idx: int | None = None,
    ) -> int:
        if evaluation:
            stable = _stable_seed_component(f"eval:{game_id}:{eval_episode_idx}")
            return int(base_seed + stable)
        if global_episode_idx is not None:
            stable = _stable_seed_component(f"train:{game_id}:{global_episode_idx}")
            return int(base_seed + stable)
        stable = _stable_seed_component(f"train:{env_instance_id}:{game_id}:{episode_counter}")
        return int(base_seed + stable)


def _stable_seed_component(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 1_000_000


def _observation_to_device(observation, device):
    return replace(
        observation,
        current_frame=observation.current_frame.to(device),
        previous_frame_1=observation.previous_frame_1.to(device),
        previous_frame_2=observation.previous_frame_2.to(device),
        valid_action_mask=observation.valid_action_mask.to(device),
        valid_pixel_mask=observation.valid_pixel_mask.to(device),
    )
