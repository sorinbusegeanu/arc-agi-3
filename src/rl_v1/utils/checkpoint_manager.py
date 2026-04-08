from __future__ import annotations

import torch


class CheckpointManager:
    def save(
        self,
        path,
        *,
        model,
        optimizer,
        scheduler,
        cfg,
        update_idx: int,
        model_variant: str,
        training_mode: str = "train_rl",
        seed: int | None = None,
        evaluation_deterministic: bool | None = None,
    ) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": None if optimizer is None else optimizer.state_dict(),
                "scheduler_state": None if scheduler is None else scheduler.state_dict(),
                "config_snapshot": cfg.to_dict(),
                "update_idx": int(update_idx),
                "model_variant": model_variant,
                "training_mode": str(training_mode),
                "effective_game_ids": list(getattr(cfg.env, "game_ids", [])),
                "seed": int(seed) if seed is not None else None,
                "acting_mode": getattr(cfg.acting, "mode", None),
                "evaluation_deterministic": (
                    bool(evaluation_deterministic)
                    if evaluation_deterministic is not None
                    else None
                ),
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

    def read_metadata(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return {
            "model_variant": payload.get("model_variant"),
            "config_snapshot": payload.get("config_snapshot", {}),
            "training_mode": payload.get("training_mode"),
            "update_idx": payload.get("update_idx"),
            "effective_game_ids": payload.get("effective_game_ids"),
            "seed": payload.get("seed"),
            "acting_mode": payload.get("acting_mode"),
            "evaluation_deterministic": payload.get("evaluation_deterministic"),
        }
