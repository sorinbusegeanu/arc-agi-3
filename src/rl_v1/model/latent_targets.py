from __future__ import annotations

import torch


class NextLatentTargetBuilder:
    def build_next_latents_for_sequence(self, sequence, model) -> torch.Tensor:
        device = next(model.parameters()).device
        hidden = sequence.initial_hidden_state.to(device)
        outputs = []
        for step in sequence.timesteps:
            encoded = model.encode_observation(
                step.next_observation,
                prev_action_id=step.chosen_action.action_id,
                prev_reward=step.reward,
                prev_done=step.done,
                hidden=hidden,
            )
            outputs.append(_single_latent_row(encoded.latent.detach()))
            hidden = encoded.hidden.detach()
        if not outputs:
            return torch.zeros((0, model.cfg.hidden_dim), device=device)
        return torch.cat(outputs, dim=0)

    def build_next_latents_for_sequence_batch(self, seq_batch, model, device):
        if not seq_batch:
            hidden_dim = int(getattr(model.cfg, "hidden_dim", getattr(model.cfg, "latent_dim", 0)))
            return (
                torch.zeros((0, 0, hidden_dim), device=device),
                torch.zeros((0, 0), dtype=torch.bool, device=device),
                {},
            )
        max_t = max(len(sequence.timesteps) for sequence in seq_batch)
        batch_size = len(seq_batch)
        timestep_mask = torch.zeros((batch_size, max_t), dtype=torch.bool, device=device)
        next_obs = [step.next_observation for sequence in seq_batch for step in sequence.timesteps]
        obs_tensors = _stack_observations(next_obs, device=device)
        if not obs_tensors:
            hidden_dim = int(getattr(model.cfg, "hidden_dim", getattr(model.cfg, "latent_dim", 0)))
            return torch.zeros((batch_size, max_t, hidden_dim), device=device), timestep_mask, {}
        frame_shape = obs_tensors["current_frame"].shape[1:]
        action_shape = obs_tensors["valid_action_mask"].shape[1:]
        pixel_shape = obs_tensors["valid_pixel_mask"].shape[1:]
        packed = {
            "current_frame": torch.zeros((batch_size, max_t, *frame_shape), dtype=obs_tensors["current_frame"].dtype, device=device),
            "previous_frame_1": torch.zeros((batch_size, max_t, *frame_shape), dtype=obs_tensors["previous_frame_1"].dtype, device=device),
            "previous_frame_2": torch.zeros((batch_size, max_t, *frame_shape), dtype=obs_tensors["previous_frame_2"].dtype, device=device),
            "valid_action_mask": torch.zeros((batch_size, max_t, *action_shape), dtype=torch.bool, device=device),
            "valid_pixel_mask": torch.zeros((batch_size, max_t, *pixel_shape), dtype=torch.bool, device=device),
            "prev_action_ids": torch.zeros((batch_size, max_t), dtype=torch.long, device=device),
            "game_id_indices": torch.zeros((batch_size, max_t), dtype=torch.long, device=device),
            "prev_rewards": torch.zeros((batch_size, max_t), dtype=torch.float32, device=device),
            "prev_dones": torch.zeros((batch_size, max_t), dtype=torch.float32, device=device),
            "current_level_indices": torch.zeros((batch_size, max_t), dtype=torch.long, device=device),
            "step_counts": torch.zeros((batch_size, max_t), dtype=torch.long, device=device),
            "chosen_action_ids": torch.zeros((batch_size, max_t), dtype=torch.long, device=device),
            "initial_hidden": torch.stack([sequence.initial_hidden_state.to(device) for sequence in seq_batch], dim=0).squeeze(1),
        }
        change_mask_target = torch.zeros((batch_size, max_t, *frame_shape), dtype=torch.float32, device=device)
        next_frame_target = torch.zeros((batch_size, max_t, *frame_shape), dtype=torch.float32, device=device)
        reward_target = torch.zeros((batch_size, max_t), dtype=torch.float32, device=device)
        done_target = torch.zeros((batch_size, max_t), dtype=torch.float32, device=device)
        obs_index = 0
        for b_idx, sequence in enumerate(seq_batch):
            for t_idx, step in enumerate(sequence.timesteps):
                timestep_mask[b_idx, t_idx] = True
                packed["current_frame"][b_idx, t_idx] = obs_tensors["current_frame"][obs_index]
                packed["previous_frame_1"][b_idx, t_idx] = obs_tensors["previous_frame_1"][obs_index]
                packed["previous_frame_2"][b_idx, t_idx] = obs_tensors["previous_frame_2"][obs_index]
                packed["valid_action_mask"][b_idx, t_idx] = obs_tensors["valid_action_mask"][obs_index]
                packed["valid_pixel_mask"][b_idx, t_idx] = obs_tensors["valid_pixel_mask"][obs_index]
                packed["prev_action_ids"][b_idx, t_idx] = int(step.chosen_action.action_id)
                packed["game_id_indices"][b_idx, t_idx] = int(obs_tensors["game_id_index"][obs_index].item())
                packed["prev_rewards"][b_idx, t_idx] = float(step.reward)
                packed["prev_dones"][b_idx, t_idx] = 1.0 if step.done else 0.0
                packed["current_level_indices"][b_idx, t_idx] = int(obs_tensors["current_level_index"][obs_index].item())
                packed["step_counts"][b_idx, t_idx] = int(obs_tensors["step_count"][obs_index].item())
                packed["chosen_action_ids"][b_idx, t_idx] = int(step.chosen_action.action_id)
                change_mask_target[b_idx, t_idx] = obs_tensors["changed_cell_mask"][obs_index].to(dtype=torch.float32)
                next_frame_target[b_idx, t_idx] = obs_tensors["current_frame"][obs_index]
                reward_target[b_idx, t_idx] = float(step.reward)
                done_target[b_idx, t_idx] = 1.0 if step.done else 0.0
                obs_index += 1
        with torch.no_grad():
            outputs = model.forward_sequence_batch(**packed)
            targets = outputs["latents"].detach()
        return (
            targets,
            timestep_mask,
            {
                "change_mask_target": change_mask_target.detach(),
                "next_frame_target": next_frame_target.detach(),
                "reward_target": reward_target.detach(),
                "done_target": done_target.detach(),
            },
        )

    def build_next_latent_targets(self, batch, model, hidden_state_bundle):
        outputs = []
        masks = []
        device = next(model.parameters()).device
        by_episode = {}
        for sequence in batch:
            by_episode.setdefault(sequence.episode_id, []).append(sequence)
        for episode_id in sorted(by_episode):
            sequences = sorted(by_episode[episode_id], key=lambda seq: _sequence_sort_key(seq.sequence_id))
            first_sequence = sequences[0]
            hidden = hidden_state_bundle[first_sequence.sequence_id].to(device)
            for seq_idx, sequence in enumerate(sequences):
                if seq_idx == 0 and sequence.sequence_start:
                    hidden = hidden_state_bundle[sequence.sequence_id].to(device)
                for step in sequence.timesteps:
                    encoded = model.encode_observation(
                        step.next_observation,
                        prev_action_id=step.chosen_action.action_id,
                        prev_reward=step.reward,
                        prev_done=step.done,
                        hidden=hidden,
                    )
                    hidden = encoded.hidden.detach()
                    outputs.append(_single_latent_row(encoded.latent.detach()))
                    masks.append(1.0)
        if not outputs:
            return torch.zeros(0, model.cfg.hidden_dim), torch.zeros(0)
        return torch.cat(outputs, dim=0), torch.tensor(masks, dtype=torch.float32, device=outputs[0].device)


def _sequence_sort_key(sequence_id: str):
    if sequence_id.startswith("seq-"):
        try:
            return int(sequence_id.split("-", 1)[1])
        except ValueError:
            return sequence_id
    return sequence_id


def _single_latent_row(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim == 1:
        return latent.unsqueeze(0)
    if latent.ndim == 2:
        if latent.shape[0] == 1:
            return latent
        return latent.mean(dim=0, keepdim=True)
    flat = latent.reshape(latent.shape[0], -1)
    if flat.shape[0] == 1:
        return flat
    return flat.mean(dim=0, keepdim=True)


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
