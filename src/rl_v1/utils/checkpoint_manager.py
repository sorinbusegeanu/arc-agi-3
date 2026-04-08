from __future__ import annotations

import torch


class CheckpointManager:
    def save(self, path, *, model, optimizer, scheduler, cfg, update_idx: int, model_variant: str) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": None if optimizer is None else optimizer.state_dict(),
                "scheduler_state": None if scheduler is None else scheduler.state_dict(),
                "config_snapshot": cfg.to_dict(),
                "update_idx": int(update_idx),
                "model_variant": model_variant,
            },
            path,
        )

    def load(self, path, *, model, optimizer=None, scheduler=None, cfg=None):
        payload = torch.load(path, map_location="cpu")
        if cfg is not None:
            if payload.get("model_variant") != cfg.model.variant:
                raise ValueError("checkpoint model variant does not match requested model variant")
            snap_model = payload.get("config_snapshot", {}).get("model", {})
            for key in ("encoder_dim", "latent_dim", "gru_hidden_size"):
                if key in snap_model and snap_model[key] != getattr(cfg.model, key):
                    raise ValueError(f"checkpoint incompatible for model.{key}")
        model.load_state_dict(payload["model_state"])
        if optimizer is not None and payload.get("optimizer_state") is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and payload.get("scheduler_state") is not None:
            scheduler.load_state_dict(payload["scheduler_state"])
        return payload
