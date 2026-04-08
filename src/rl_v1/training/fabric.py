from __future__ import annotations

from contextlib import nullcontext

import torch


class FabricShim:
    def __init__(self, accelerator: str = "cpu", precision: str = "32-true", devices: int = 1) -> None:
        self.accelerator = accelerator
        self.precision = precision
        self.devices = devices
        self.device = torch.device("cuda" if accelerator == "cuda" and torch.cuda.is_available() else "cpu")

    def setup(self, model, optimizer):
        model.to(self.device)
        return model, optimizer

    def backward(self, loss):
        loss.backward()

    def autocast(self):
        return nullcontext()


def build_fabric(accelerator: str = "cpu", precision: str = "32-true", devices: int = 1):
    try:
        from lightning.fabric import Fabric  # type: ignore

        fabric = Fabric(accelerator=accelerator, precision=precision, devices=devices)
        fabric.launch()
        return fabric
    except Exception:
        return FabricShim(accelerator=accelerator, precision=precision, devices=devices)
