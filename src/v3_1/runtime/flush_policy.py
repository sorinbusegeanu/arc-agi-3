from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlushDecision:
    should_flush: bool
    reason: str | None = None


@dataclass
class FlushPolicy:
    config: object

    def should_flush_periodic(self, *, round_id: int, pending_status: dict | None, won: bool = False) -> FlushDecision:
        if not self.config.storage.enable_persistent_memory:
            return FlushDecision(False)
        cadence = int(self.config.storage.persistent_memory_flush_every_n_rounds or 0)
        if cadence <= 0 or round_id % cadence != 0:
            return FlushDecision(False)
        status = dict(pending_status or {})
        pending_batch_count = int(status.get("pending_batch_count", 0) or 0)
        eligible_row_count = int(status.get("eligible_row_count", 0) or 0)
        mature_family_count = int(status.get("mature_family_count", 0) or 0)
        meaningful_delta = bool(status.get("has_meaningful_delta"))
        if won and eligible_row_count > 0:
            return FlushDecision(True, "periodic_win")
        if eligible_row_count <= 0:
            return FlushDecision(False)
        if pending_batch_count >= 2 or mature_family_count >= 2 or meaningful_delta:
            return FlushDecision(True, "periodic")
        return FlushDecision(False)

    def should_flush_end_of_session(self, *, pending_status: dict | None) -> FlushDecision:
        if not self.config.storage.enable_persistent_memory:
            return FlushDecision(False)
        status = dict(pending_status or {})
        if int(status.get("pending_batch_count", 0) or 0) <= 0:
            return FlushDecision(False)
        return FlushDecision(True, "end_of_session")
