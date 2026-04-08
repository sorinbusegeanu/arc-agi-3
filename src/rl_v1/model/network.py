from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from rl_v1.configs.schema import ModelConfig, PlannerConfig
from rl_v1.data.contracts import ObservationPackage, PolicyOutput, V1Action
from rl_v1.model.components import (
    ClickPolicyHead,
    DynamicsModel,
    MetadataEncoder,
    PolicyHead,
    RecurrentCore,
    SlotAttention,
    SlotPooler,
    SlotRelationBlock,
    ValueHead,
    VisualEncoder,
)
from rl_v1.planner.beam_search import BeamSearchPlanner


@dataclass
class StepOutput:
    latent: torch.Tensor
    hidden: torch.Tensor
    spatial: torch.Tensor
    tokens: torch.Tensor
    raw_slots: torch.Tensor | None
    slots: torch.Tensor | None
    pooled: torch.Tensor
    metadata: torch.Tensor | None
    policy_logits: torch.Tensor
    value: torch.Tensor
    click_logits: torch.Tensor | None
    change_mask_logits: torch.Tensor | None
    next_frame_logits: torch.Tensor | None
    reward_pred: torch.Tensor | None
    done_logit: torch.Tensor | None
    diagnostics: dict[str, Any]


class RLV1Model(nn.Module):
    def __init__(self, model_cfg: ModelConfig, planner_cfg: PlannerConfig) -> None:
        super().__init__()
        self.cfg = model_cfg
        self.encoder = VisualEncoder(3, model_cfg.encoder_dim, tuple(model_cfg.encoder_channels))
        self.use_slots = bool(model_cfg.use_slots)
        if self.use_slots:
            self.slot_attention = SlotAttention(model_cfg.slot_count, model_cfg.slot_dim)
            self.slot_relation = SlotRelationBlock(model_cfg.slot_dim, model_cfg.relation_heads, model_cfg.relation_layers)
            self.pooler = SlotPooler(model_cfg.slot_dim, mode=model_cfg.pooling)
            scene_dim = model_cfg.slot_dim
        else:
            self.slot_attention = None
            self.slot_relation = None
            self.pooler = None
            self.direct_pool = nn.Sequential(nn.Linear(model_cfg.encoder_dim, model_cfg.hidden_dim), nn.ReLU(), nn.Linear(model_cfg.hidden_dim, model_cfg.hidden_dim))
            scene_dim = model_cfg.hidden_dim
        self.use_recurrent_memory = bool(model_cfg.use_recurrent_memory)
        self.metadata_encoder = MetadataEncoder(
            max_game_ids=model_cfg.max_game_ids,
            game_embed_dim=model_cfg.game_embed_dim,
            action_vocab=8,
            action_embed_dim=model_cfg.action_embed_dim,
            step_count_embed_dim=model_cfg.step_count_embed_dim,
            max_step_count=model_cfg.max_step_count,
            metadata_embed_dim=model_cfg.metadata_embed_dim,
            max_level_index=model_cfg.max_level_index,
        )
        self.recurrent = RecurrentCore(
            scene_dim,
            model_cfg.hidden_dim,
            metadata_dim=model_cfg.metadata_embed_dim,
        )
        self.direct_latent = nn.Sequential(nn.Linear(scene_dim, model_cfg.hidden_dim), nn.Tanh())
        self.policy_head = PolicyHead(model_cfg.hidden_dim, action_vocab=8)
        self.value_head = ValueHead(model_cfg.hidden_dim)
        self.use_click_branch = bool(model_cfg.use_click_branch)
        self.click_head = ClickPolicyHead(model_cfg.hidden_dim, model_cfg.encoder_dim) if self.use_click_branch else None
        self.use_transition_model = bool(model_cfg.use_transition_model)
        self.dynamics = (
            DynamicsModel(
                model_cfg.hidden_dim,
                action_vocab=8,
                board_height=model_cfg.canvas_height,
                board_width=model_cfg.canvas_width,
                predict_next_frame=bool(model_cfg.predict_next_frame),
            )
            if self.use_transition_model
            else None
        )
        self.planner = BeamSearchPlanner(planner_cfg) if planner_cfg.enabled else None

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.recurrent.initial_state(batch_size, device)

    def encode_observation(
        self,
        observation: ObservationPackage,
        prev_action_id: int,
        prev_reward: float,
        prev_done: bool,
        hidden: torch.Tensor,
    ) -> StepOutput:
        frames = observation.stacked_frames().unsqueeze(0).to(hidden.device)
        valid_pixel_mask = observation.valid_pixel_mask.unsqueeze(0).to(hidden.device)
        spatial, tokens = self.encoder(frames)
        diagnostics: dict[str, Any] = {}
        if self.use_slots:
            raw_slots, slot_diag = self.slot_attention(tokens)
            slots = self.slot_relation(raw_slots)
            pooled, pool_diag = self.pooler(slots)
            diagnostics.update(slot_diag)
            diagnostics.update(pool_diag)
        else:
            raw_slots = None
            slots = None
            pooled = self.direct_pool(tokens.mean(dim=1))
            diagnostics["pooling_mode"] = "direct_mean"
        if self.use_recurrent_memory:
            prev_action = torch.tensor([int(prev_action_id)], device=hidden.device, dtype=torch.long)
            game_id_index = torch.tensor([int(observation.game_id_index)], device=hidden.device, dtype=torch.long)
            prev_reward_t = torch.tensor([float(prev_reward)], device=hidden.device, dtype=torch.float32)
            prev_done_t = torch.tensor([1.0 if prev_done else 0.0], device=hidden.device, dtype=torch.float32)
            current_level_index = torch.tensor([int(observation.current_level_index)], device=hidden.device, dtype=torch.long)
            step_count = torch.tensor([int(getattr(observation, "step_count", 0))], device=hidden.device, dtype=torch.long)
            metadata = self.metadata_encoder(
                game_id_index=game_id_index,
                current_level_index=current_level_index,
                step_count=step_count,
                prev_action=prev_action,
                prev_reward=prev_reward_t,
                prev_done=prev_done_t,
            )
            next_hidden, latent = self.recurrent(
                pooled,
                metadata,
                hidden,
            )
        else:
            next_hidden = hidden
            latent = self.direct_latent(pooled)
            metadata = None
        action_mask = observation.valid_action_mask.unsqueeze(0).to(hidden.device)
        policy_logits = self.policy_head(latent, action_mask)
        value = self.value_head(latent)
        click_logits = self.click_head(latent, spatial, valid_pixel_mask) if self.click_head is not None else None
        change_mask_logits = None
        next_frame_logits = None
        reward_pred = None
        done_logit = None
        if self.dynamics is not None:
            transition = self.dynamics(latent, torch.tensor([int(prev_action_id)], device=hidden.device, dtype=torch.long))
            _, change_mask_logits, next_frame_logits, reward_pred, done_logit = transition
        diagnostics["action_mask_count"] = int(action_mask.sum().item())
        return StepOutput(
            latent=latent,
            hidden=next_hidden,
            spatial=spatial,
            tokens=tokens,
            raw_slots=raw_slots,
            slots=slots,
            pooled=pooled,
            metadata=metadata,
            policy_logits=policy_logits,
            value=value,
            click_logits=click_logits,
            change_mask_logits=change_mask_logits,
            next_frame_logits=next_frame_logits,
            reward_pred=reward_pred,
            done_logit=done_logit,
            diagnostics=diagnostics,
        )

    def act(
        self,
        observation: ObservationPackage,
        prev_action_id: int,
        prev_reward: float,
        prev_done: bool,
        hidden: torch.Tensor,
        acting_mode: str = "policy_only",
        deterministic: bool = False,
    ) -> tuple[V1Action, PolicyOutput, torch.Tensor, dict[str, Any]]:
        step = self.encode_observation(observation, prev_action_id, prev_reward, prev_done, hidden)
        planner_diag = {}
        planned_action = None
        if acting_mode in {"planner_act", "planner_eval_only"} and self.planner is not None and self.dynamics is not None:
            planned_action, planner_diag = self.planner.plan(
                latent=step.latent.detach(),
                action_mask=observation.valid_action_mask.unsqueeze(0).to(step.latent.device),
                policy_logits=step.policy_logits.detach(),
                value_head=self.value_head,
                dynamics=self.dynamics,
            )
        if acting_mode == "planner_act" and planned_action is not None:
            action = planned_action
            chosen = int(action.action_id)
        else:
            distribution = torch.distributions.Categorical(logits=step.policy_logits)
            if deterministic:
                chosen = int(step.policy_logits.argmax(dim=-1).item())
            else:
                chosen = int(distribution.sample().item())
        click_action = self._resolve_click(step, observation, chosen)
        final_action = click_action if click_action is not None else V1Action(action_id=chosen)
        action_logprob = self._action_logprob(step, final_action)
        diagnostics = dict(step.diagnostics)
        diagnostics["acting_mode"] = acting_mode
        diagnostics["planner"] = planner_diag
        diagnostics["planner_first_action"] = None if planned_action is None else planned_action.action_id
        return (
            final_action,
            PolicyOutput(
                action_logits=step.policy_logits,
                action_logprob=action_logprob,
                value=step.value,
                click_logits=step.click_logits,
                diagnostics=diagnostics,
            ),
            step.hidden,
            diagnostics,
        )

    def _resolve_click(self, step: StepOutput, observation: ObservationPackage, chosen: int) -> V1Action | None:
        if chosen != 6:
            return None
        if step.click_logits is None:
            return V1Action(action_id=6, x=0, y=0)
        flat_index = int(step.click_logits.view(1, -1).argmax(dim=-1).item())
        width = step.click_logits.shape[-1]
        y = flat_index // width
        x = flat_index % width
        return V1Action(action_id=6, x=x, y=y)

    def _action_logprob(self, step: StepOutput, action: V1Action) -> torch.Tensor:
        dist = torch.distributions.Categorical(logits=step.policy_logits)
        action_id = torch.tensor([action.action_id], device=step.policy_logits.device)
        logprob = dist.log_prob(action_id)
        if action.action_id == 6 and step.click_logits is not None and action.x is not None and action.y is not None:
            click_logp = F.log_softmax(step.click_logits.view(1, -1), dim=-1)
            flat = action.y * step.click_logits.shape[-1] + action.x
            logprob = logprob + click_logp[:, flat]
        return logprob

    def transition(self, latent: torch.Tensor, action_id: int):
        if self.dynamics is None:
            raise RuntimeError("transition model disabled")
        action_ids = torch.tensor([action_id], device=latent.device, dtype=torch.long)
        return self.dynamics(latent, action_ids)

    def transition_batch(self, latents: torch.Tensor, action_ids: torch.Tensor):
        if self.dynamics is None:
            raise RuntimeError("transition model disabled")
        return self.dynamics(latents, action_ids.long())

    def forward_sequence_batch(
        self,
        *,
        current_frame: torch.Tensor,
        previous_frame_1: torch.Tensor,
        previous_frame_2: torch.Tensor,
        valid_action_mask: torch.Tensor,
        valid_pixel_mask: torch.Tensor,
        prev_action_ids: torch.Tensor,
        game_id_indices: torch.Tensor,
        prev_rewards: torch.Tensor,
        prev_dones: torch.Tensor,
        current_level_indices: torch.Tensor,
        step_counts: torch.Tensor,
        chosen_action_ids: torch.Tensor | None = None,
        initial_hidden: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, timesteps = prev_action_ids.shape
        hidden = initial_hidden
        policy_logits_steps = []
        values_steps = []
        latents_steps = []
        hiddens_steps = []
        spatial_steps = []
        click_steps = []
        metadata_steps = []
        transition_latent_steps = []
        reward_steps = []
        done_steps = []
        change_mask_steps = []
        next_frame_steps = []
        for t in range(timesteps):
            # Recurrent semantics are preserved across time; batching is across sequences.
            frames_t = torch.cat(
                [
                    current_frame[:, t],
                    previous_frame_1[:, t],
                    previous_frame_2[:, t],
                ],
                dim=1,
            )
            action_mask_t = valid_action_mask[:, t]
            valid_pixel_mask_t = valid_pixel_mask[:, t]
            spatial_t, tokens_t = self.encoder(frames_t)
            if self.use_slots:
                raw_slots_t, _ = self.slot_attention(tokens_t)
                slots_t = self.slot_relation(raw_slots_t)
                pooled_t, _ = self.pooler(slots_t)
            else:
                pooled_t = self.direct_pool(tokens_t.mean(dim=1))
            if self.use_recurrent_memory:
                metadata_t = self.metadata_encoder(
                    game_id_index=game_id_indices[:, t].long(),
                    current_level_index=current_level_indices[:, t].long(),
                    step_count=step_counts[:, t].long(),
                    prev_action=prev_action_ids[:, t].long(),
                    prev_reward=prev_rewards[:, t].float(),
                    prev_done=prev_dones[:, t].float(),
                )
                hidden, latent_t = self.recurrent(
                    pooled_t,
                    metadata_t,
                    hidden,
                )
            else:
                latent_t = self.direct_latent(pooled_t)
                metadata_t = torch.zeros(
                    (batch_size, self.cfg.metadata_embed_dim),
                    dtype=latent_t.dtype,
                    device=latent_t.device,
                )
            logits_t = self.policy_head(latent_t, action_mask_t)
            value_t = self.value_head(latent_t)
            policy_logits_steps.append(logits_t)
            values_steps.append(value_t)
            latents_steps.append(latent_t)
            hiddens_steps.append(hidden)
            spatial_steps.append(spatial_t)
            metadata_steps.append(metadata_t)
            if self.click_head is not None:
                click_steps.append(self.click_head(latent_t, spatial_t, valid_pixel_mask_t))
            if self.dynamics is not None and chosen_action_ids is not None:
                (
                    next_latent_pred_t,
                    change_mask_logits_t,
                    next_frame_logits_t,
                    reward_pred_t,
                    done_logit_t,
                ) = self.transition_batch(latent_t, chosen_action_ids[:, t].long())
                transition_latent_steps.append(next_latent_pred_t)
                reward_steps.append(reward_pred_t)
                done_steps.append(done_logit_t)
                change_mask_steps.append(change_mask_logits_t)
                if next_frame_logits_t is not None:
                    next_frame_steps.append(next_frame_logits_t)
        out: dict[str, torch.Tensor | None] = {
            "policy_logits": torch.stack(policy_logits_steps, dim=1),
            "values": torch.stack(values_steps, dim=1),
            "latents": torch.stack(latents_steps, dim=1),
            "hiddens": torch.stack(hiddens_steps, dim=1),
            "spatial_features": torch.stack(spatial_steps, dim=1),
            "metadata": torch.stack(metadata_steps, dim=1),
        }
        if self.click_head is not None:
            out["click_logits"] = torch.stack(click_steps, dim=1)
        else:
            out["click_logits"] = None
        out["next_latent_pred"] = torch.stack(transition_latent_steps, dim=1) if transition_latent_steps else None
        out["reward_pred"] = torch.stack(reward_steps, dim=1) if reward_steps else None
        out["done_logit"] = torch.stack(done_steps, dim=1) if done_steps else None
        out["change_mask_logits"] = torch.stack(change_mask_steps, dim=1) if change_mask_steps else None
        out["next_frame_logits"] = torch.stack(next_frame_steps, dim=1) if next_frame_steps else None
        return out


class RecurrentBaselineModel(RLV1Model):
    def __init__(self, model_cfg: ModelConfig, planner_cfg: PlannerConfig) -> None:
        baseline_cfg = ModelConfig(**{**model_cfg.__dict__, "use_slots": False, "use_transition_model": False, "use_click_branch": False, "baseline": True})
        baseline_planner = PlannerConfig(**{**planner_cfg.__dict__, "enabled": False})
        super().__init__(baseline_cfg, baseline_planner)
