from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class MemoryGovernorState(str, Enum):
    NORMAL = "NORMAL"
    COMPACTING = "COMPACTING"
    HARD_PRESSURE_DRAIN = "HARD_PRESSURE_DRAIN"
    RECOVERING = "RECOVERING"


@dataclass(frozen=True, slots=True)
class ProcessMemorySnapshot:
    rss_bytes: int
    uss_bytes: int
    swap_bytes: int
    state: MemoryGovernorState


def _proc_status_bytes(name: str) -> int:
    path = Path("/proc/self/status")
    if not path.exists():
        return 0
    prefix = f"{name}:"
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith(prefix):
                continue
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) * 1024
    except (OSError, ValueError):
        return 0
    return 0


def _uss_bytes() -> int:
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_full_info().uss)
    except (ImportError, AttributeError, OSError):
        return 0


class RuntimeMemoryGovernor:
    """Process-memory feedback for v9.7.9 resident-memory control."""

    def __init__(self, scientific: Any) -> None:
        self.high_watermark = int(scientific.memory_rss_high_watermark_bytes)
        self.hard_watermark = int(scientific.memory_rss_hard_watermark_bytes)
        self.swap_high_watermark = int(scientific.memory_swap_high_watermark_bytes)
        if not 0 < self.high_watermark < self.hard_watermark:
            raise ValueError("memory governor requires high < hard RSS watermarks")
        self.state = MemoryGovernorState.NORMAL
        self.peak_rss_bytes = 0
        self.peak_swap_bytes = 0
        self.transitions = 0

    def sample(self, *, backlog: int = 0) -> ProcessMemorySnapshot:
        rss = _proc_status_bytes("VmRSS")
        swap = _proc_status_bytes("VmSwap")
        uss = _uss_bytes()
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        self.peak_swap_bytes = max(self.peak_swap_bytes, swap)

        previous = self.state
        if rss >= self.hard_watermark or swap >= self.swap_high_watermark:
            state = MemoryGovernorState.HARD_PRESSURE_DRAIN
        elif rss >= self.high_watermark or backlog > 0:
            state = MemoryGovernorState.COMPACTING
        elif previous in {MemoryGovernorState.COMPACTING, MemoryGovernorState.HARD_PRESSURE_DRAIN}:
            recovery = int(self.high_watermark * 0.90)
            state = MemoryGovernorState.RECOVERING if rss > recovery else MemoryGovernorState.NORMAL
        elif previous is MemoryGovernorState.RECOVERING:
            state = MemoryGovernorState.RECOVERING if rss > int(self.high_watermark * 0.90) else MemoryGovernorState.NORMAL
        else:
            state = MemoryGovernorState.NORMAL

        if state is not previous:
            self.transitions += 1
        self.state = state
        return ProcessMemorySnapshot(rss, uss, swap, state)

    def state_dict(self) -> dict[str, int | str]:
        return {
            "memory_governor_state": self.state.value,
            "process_rss_peak_bytes": int(self.peak_rss_bytes),
            "process_swap_peak_bytes": int(self.peak_swap_bytes),
            "memory_governor_transitions": int(self.transitions),
        }
