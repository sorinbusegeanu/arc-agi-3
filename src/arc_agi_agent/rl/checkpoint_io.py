from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import torch


def save_checkpoint(path: str, modules: Dict[str, Any], optim: Optional[Any], cfg: Dict[str, Any], iter_idx: int) -> None:
    payload = {
        "iter": int(iter_idx),
        "cfg": cfg,
        "encoder": modules["encoder"].state_dict(),
        "memory": modules["memory"].state_dict(),
        "policy": modules["policy"].state_dict(),
        "value": modules["value"].state_dict(),
        "optim": optim.state_dict() if optim is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, modules: Dict[str, Any], optim: Optional[Any] = None) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    modules["encoder"].load_state_dict(payload.get("encoder", {}))
    modules["memory"].load_state_dict(payload.get("memory", {}))
    modules["policy"].load_state_dict(payload.get("policy", {}))
    modules["value"].load_state_dict(payload.get("value", {}))
    if optim is not None and payload.get("optim") is not None:
        optim.load_state_dict(payload["optim"])
    return payload
