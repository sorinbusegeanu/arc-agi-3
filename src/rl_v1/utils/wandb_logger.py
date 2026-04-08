from __future__ import annotations

from collections.abc import Mapping


class WandbLogger:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._enabled = bool(cfg.wandb.enabled) and cfg.wandb.mode != "disabled"
        self._run = None
        self._backend = None
        self._config_logged = False
        if not self._enabled:
            return
        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise RuntimeError("wandb is enabled in config but the wandb package is not installed") from exc
        self._backend = wandb
        self._run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            tags=list(cfg.wandb.tags),
            mode=cfg.wandb.mode,
        )

    def log_metrics(self, metrics: dict, step: int):
        if not self._enabled or self._run is None:
            return
        scalars = {}
        for key, value in metrics.items():
            if isinstance(value, bool):
                scalars[key] = int(value)
            elif isinstance(value, (int, float)):
                scalars[key] = value
        if scalars:
            self._run.log(scalars, step=int(step))

    def log_config(self, cfg_dict: dict):
        if not self._enabled or self._run is None or self._config_logged:
            return
        self._run.config.update(_sanitize_for_wandb(cfg_dict), allow_val_change=True)
        self._config_logged = True

    def finish(self):
        if not self._enabled or self._run is None:
            return
        self._run.finish()


def _sanitize_for_wandb(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _sanitize_for_wandb(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_wandb(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_wandb(item) for item in value]
    return str(value)
