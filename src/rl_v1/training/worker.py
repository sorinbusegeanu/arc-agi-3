from __future__ import annotations

import io

import torch

from rl_v1.configs.load import _from_dict
from rl_v1.data.rollout_collector import RolloutCollector
from rl_v1.env.adapter import ArcEnvironmentAdapter
from rl_v1.model.action_selector import ActionSelector
from rl_v1.model.model_factory import build_model


def rollout_worker_main(command_conn) -> None:
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass
    initialized = False
    worker_id = None
    cfg = None
    collector = None
    envs_by_game_id = {}
    assigned_game_ids = []
    model = None
    model_device = None
    model_variant_signature = None
    while True:
        command = command_conn.recv()
        command_type = command.get("type")
        if command_type == "init":
            try:
                payload = command["payload"]
                cfg = _from_dict(payload["config_dict"])
                worker_id = int(payload["worker_id"])
                assigned_game_ids = [str(game_id) for game_id in payload["game_ids"]]
                selector = ActionSelector(cfg)
                collector = RolloutCollector(cfg.rollout, selector)
                model_device = _resolve_worker_device(cfg)
                model_variant_signature = _build_model_signature(cfg)
                model = build_model(cfg).to(model_device)
                model.eval()
                envs_by_game_id = {}
                for game_id in assigned_game_ids:
                    env = ArcEnvironmentAdapter(cfg.env, cfg.model, game_id, reward_cfg=cfg.reward)
                    env.env_instance_id = f"worker-{worker_id}-{env.env_instance_id}"
                    envs_by_game_id[game_id] = env
                initialized = True
                command_conn.send({"ok": True})
            except Exception as exc:  # pragma: no cover - defensive cross-process error forwarding
                command_conn.send({"ok": False, "error": repr(exc)})
            continue
        if command_type == "close":
            for env in envs_by_game_id.values():
                if hasattr(env, "close"):
                    env.close()
            envs_by_game_id.clear()
            command_conn.send({"ok": True})
            break
        if command_type != "collect":
            command_conn.send({"ok": False, "error": f"unsupported command type: {command_type}"})
            continue
        if not initialized or cfg is None or collector is None:
            command_conn.send({"ok": False, "error": "worker collect called before init"})
            continue
        try:
            payload = command["payload"]
            if "worker_id" in payload and int(payload["worker_id"]) != int(worker_id):
                raise ValueError(f"worker id mismatch: expected {worker_id}, got {payload['worker_id']}")
            incoming_signature = payload.get("model_signature")
            if incoming_signature is not None and incoming_signature != model_variant_signature:
                raise ValueError(
                    "model signature mismatch in worker collect: "
                    f"expected={model_variant_signature}, got={incoming_signature}"
                )
            per_game_episode_counts = {str(key): int(value) for key, value in payload["per_game_episode_counts"].items()}
            per_game_episode_offsets = {
                str(key): int(value)
                for key, value in payload.get("per_game_episode_offsets", {}).items()
            }
            deterministic = bool(payload["deterministic"])
            evaluation = bool(payload["evaluation"])
            acting_mode = payload.get("acting_mode")
            if acting_mode is not None and hasattr(collector, "action_selector") and hasattr(collector.action_selector, "cfg"):
                collector.action_selector.cfg.acting.mode = str(acting_mode)
            if "model_state_bytes" in payload:
                model_state = torch.load(io.BytesIO(payload["model_state_bytes"]), map_location="cpu")
            elif "model_state_dict" in payload:
                model_state = payload["model_state_dict"]
            else:
                raise ValueError("collect payload missing model_state_bytes/model_state_dict")
            model.load_state_dict(model_state)
            model.to(model_device)
            model.eval()
            sequences = []
            for game_id in dict.fromkeys(assigned_game_ids):
                episodes = int(per_game_episode_counts.get(game_id, 0))
                if episodes <= 0:
                    continue
                env = envs_by_game_id.get(game_id)
                if env is None:
                    raise ValueError(f"missing persistent env for game_id={game_id}")
                sequences.extend(
                    collector.collect(
                        model,
                        env,
                        episodes=episodes,
                        deterministic=deterministic,
                        evaluation=evaluation,
                        eval_episode_start_idx=per_game_episode_offsets.get(game_id, 0),
                    )
                )
            _sanitize_sequences_for_ipc(sequences)
            payload_buffer = io.BytesIO()
            torch.save(sequences, payload_buffer)
            command_conn.send({"ok": True, "sequences_bytes": payload_buffer.getvalue()})
        except Exception as exc:  # pragma: no cover - defensive cross-process error forwarding
            command_conn.send({"ok": False, "error": repr(exc)})


def _sanitize_sequences_for_ipc(sequences) -> None:
    for sequence in sequences:
        for step in sequence.timesteps:
            if hasattr(step.observation, "raw_response"):
                step.observation.raw_response = None
            if hasattr(step.next_observation, "raw_response"):
                step.next_observation.raw_response = None


def _resolve_worker_device(cfg) -> torch.device:
    accelerator = str(getattr(cfg.runtime, "accelerator", "auto")).lower()
    if accelerator == "gpu":
        accelerator = "cuda"
    if accelerator in {"auto", "cuda"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("worker requires CUDA/GPU but torch.cuda.is_available() is False")
    raise RuntimeError(f"worker requires CUDA/GPU; unsupported accelerator value: {cfg.runtime.accelerator}")


def _build_model_signature(cfg) -> tuple:
    model_cfg = cfg.model
    return (
        model_cfg.variant,
        tuple(model_cfg.encoder_channels),
        model_cfg.encoder_dim,
        model_cfg.num_slots,
        model_cfg.slot_dim,
        model_cfg.slot_iters,
        model_cfg.slot_transformer_layers,
        model_cfg.slot_transformer_heads,
        model_cfg.gru_hidden_size,
        model_cfg.action_embed_dim,
        model_cfg.latent_dim,
    )
