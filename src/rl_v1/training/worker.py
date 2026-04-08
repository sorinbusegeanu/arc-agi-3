from __future__ import annotations

import io

import torch

from rl_v1.configs.load import _from_dict
from rl_v1.data.rollout_collector import RolloutCollector
from rl_v1.env.adapter import ArcEnvironmentAdapter
from rl_v1.model.action_selector import ActionSelector
from rl_v1.model.model_factory import build_model
from rl_v1.utils.runtime import ensure_arc_paths


def rollout_worker_main(command_conn) -> None:
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass
    initialized = False
    worker_id = None
    cfg = None
    collector = None
    assigned_game_ids = []
    model = None
    model_device = None
    model_variant_signature = None
    model_ready = False
    while True:
        command = command_conn.recv()
        command_type = command.get("type")
        if command_type == "init":
            try:
                payload = command["payload"]
                ensure_arc_paths()
                cfg = _from_dict(payload["config_dict"])
                worker_id = int(payload["worker_id"])
                assigned_game_ids = [str(game_id) for game_id in payload.get("game_ids", [])]
                selector = ActionSelector(cfg)
                collector = RolloutCollector(cfg.rollout, selector)
                model_device = _resolve_worker_device(cfg)
                model_variant_signature = _build_model_signature(cfg)
                model = build_model(cfg).to(model_device)
                model.eval()
                model_ready = False
                initialized = True
                command_conn.send({"ok": True})
            except Exception as exc:  # pragma: no cover - defensive cross-process error forwarding
                command_conn.send({"ok": False, "error": repr(exc)})
            continue
        if command_type == "set_model":
            if not initialized or cfg is None or collector is None or model is None:
                command_conn.send({"ok": False, "error": "worker set_model called before init"})
                continue
            try:
                payload = command["payload"]
                if "worker_id" in payload and int(payload["worker_id"]) != int(worker_id):
                    raise ValueError(f"worker id mismatch: expected {worker_id}, got {payload['worker_id']}")
                incoming_signature = payload.get("model_signature")
                if incoming_signature is not None and incoming_signature != model_variant_signature:
                    raise ValueError(
                        "model signature mismatch in worker set_model: "
                        f"expected={model_variant_signature}, got={incoming_signature}"
                    )
                acting_mode = payload.get("acting_mode")
                if acting_mode is not None and hasattr(collector, "action_selector") and hasattr(collector.action_selector, "cfg"):
                    collector.action_selector.cfg.acting.mode = str(acting_mode)
                if "model_state_bytes" in payload:
                    model_state = torch.load(io.BytesIO(payload["model_state_bytes"]), map_location="cpu")
                elif "model_state_dict" in payload:
                    model_state = payload["model_state_dict"]
                else:
                    raise ValueError("set_model payload missing model_state_bytes/model_state_dict")
                model.load_state_dict(model_state)
                model.to(model_device)
                model.eval()
                model_ready = True
                command_conn.send({"ok": True})
            except Exception as exc:  # pragma: no cover - defensive cross-process error forwarding
                command_conn.send({"ok": False, "error": repr(exc)})
            continue
        if command_type == "close":
            command_conn.send({"ok": True})
            break
        if command_type not in {"collect", "collect_task"}:
            command_conn.send({"ok": False, "error": f"unsupported command type: {command_type}"})
            continue
        if not initialized or cfg is None or collector is None:
            command_conn.send({"ok": False, "error": "worker collect called before init"})
            continue
        if not model_ready and command_type != "collect":
            command_conn.send({"ok": False, "error": "worker collect called before set_model"})
            continue
        try:
            payload = command["payload"]
            if "worker_id" in payload and int(payload["worker_id"]) != int(worker_id):
                raise ValueError(f"worker id mismatch: expected {worker_id}, got {payload['worker_id']}")
            if command_type == "collect_task":
                task = dict(payload.get("task", {}))
                game_tasks = [task] if task else []
            else:
                incoming_signature = payload.get("model_signature")
                if incoming_signature is not None and incoming_signature != model_variant_signature:
                    raise ValueError(
                        "model signature mismatch in worker collect: "
                        f"expected={model_variant_signature}, got={incoming_signature}"
                    )
                game_tasks = list(payload.get("game_tasks", []))
                if "model_state_bytes" in payload:
                    model_state = torch.load(io.BytesIO(payload["model_state_bytes"]), map_location="cpu")
                    model.load_state_dict(model_state)
                    model.to(model_device)
                    model.eval()
                    model_ready = True
                elif "model_state_dict" in payload:
                    model_state = payload["model_state_dict"]
                    model.load_state_dict(model_state)
                    model.to(model_device)
                    model.eval()
                    model_ready = True
                elif not model_ready:
                    raise ValueError("worker collect called before set_model")
            deterministic = bool(payload["deterministic"])
            evaluation = bool(payload["evaluation"])
            collection_mode = str(payload.get("collection_mode", "rl"))
            sequences = []
            for task in game_tasks:
                game_id = str(task["game_id"])
                episodes = int(task["episodes"])
                episode_offset = int(task.get("episode_offset", 0))
                if episodes <= 0:
                    continue
                env = ArcEnvironmentAdapter(cfg.env, cfg.model, game_id, reward_cfg=cfg.reward)
                # Keep env id stable across task-local env recreation for persistent episode counters/seeds.
                env.env_instance_id = f"worker-{worker_id}-game-{game_id}"
                try:
                    sequences.extend(
                        collector.collect(
                            model,
                            env,
                            episodes=episodes,
                            deterministic=deterministic,
                            evaluation=evaluation,
                            eval_episode_start_idx=episode_offset,
                            collection_mode=collection_mode,
                        )
                    )
                finally:
                    if hasattr(env, "close"):
                        env.close()
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
    inference_device = str(getattr(cfg.runtime, "inference_device", "gpu")).lower()
    if inference_device == "cuda" or inference_device == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError(
            "worker inference_device requires CUDA/GPU but torch.cuda.is_available() is False"
        )
    if inference_device == "cpu":
        return torch.device("cpu")
    raise RuntimeError(
        f"unsupported runtime.inference_device value: {getattr(cfg.runtime, 'inference_device', None)}"
    )


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
