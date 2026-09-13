from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GPUSnapshot:
    memory_used_bytes: int = 0
    utilization_percent: float = 0.0


def read_gpu_snapshot(device_index: int = 0) -> GPUSnapshot:
    try:
        import pynvml
    except ImportError:
        return GPUSnapshot()

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(device_index))
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return GPUSnapshot(
            memory_used_bytes=int(memory.used),
            utilization_percent=float(utilization.gpu),
        )
    except Exception:
        return GPUSnapshot()
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
