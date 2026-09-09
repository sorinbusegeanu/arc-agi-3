from __future__ import annotations

"""v8.83: make final stabilization invoke the full-cut developmental authority.

Final stabilization runs after peers are paused and idle. It must not re-enter the
public active-runtime ``run_once`` wrapper chain because that chain contains bounded
scheduling/input-token guards which may legally return without producing a cut.

This layer therefore calls the underlying v8.2 full-cut authority directly while
holding the v8.45 state lock. A proven no-op may still be retried until the existing
stabilization timeout; partial advancement remains fatal to avoid duplicate writes.
"""

import time
from collections import Counter

from v8.model import MemoryLevel, MemoryProposal, MemoryType, MemoryUid


_INSTALLED = False
_BASE_RUN_UNTIL_STABLE = None
_BASE_FULL_CUT_RUN_ONCE = None
_RETRY_POLL_SECONDS = 0.005


def _run_full_cut_once(self) -> None:
    runner = _BASE_FULL_CUT_RUN_ONCE
    if not callable(runner):
        raise RuntimeError("v8 full-cut developmental authority is unavailable")
    state_lock = getattr(self, "_v845_state_lock", None)
    if state_lock is None:
        runner(self)
        return
    with state_lock:
        runner(self)


def _run_until_stable_v883(
    self,
    max_cycles: int = 8,
    *,
    commit_proposals,
    timeout: float = 60.0,
) -> str:
    from v8 import information_flow_diagnostics as flow

    limit = min(8, max(1, int(max_cycles)))
    deadline = time.monotonic() + max(0.0, float(timeout))
    was_paused = self._pause.is_set()
    self.pause()
    cancel = getattr(self, "_v841_peer_cancel", None)
    was_cancelled = bool(cancel is not None and cancel.is_set())
    submit = self.submit_proposal
    prior_stabilizing = getattr(self, "_v82_stabilizing", False)
    try:
        if not self.wait_idle(max(0.0, deadline - time.monotonic())):
            raise TimeoutError("v8 peers did not become idle before stabilization")
        if cancel is not None:
            cancel.clear()
        self._v82_stabilizing = True
        commit_proposals()

        for cycle in range(1, limit + 1):
            submitted: dict[MemoryUid, str] = {}

            def track(proposal: MemoryProposal) -> None:
                submit(proposal)
                level = int(proposal.level)
                kind = int(proposal.memory_type)
                if level == int(MemoryLevel.M2):
                    submitted[proposal.uid] = "new_m2_count"
                elif level == int(MemoryLevel.M3):
                    if kind == int(MemoryType.CARRIER):
                        submitted[proposal.uid] = "new_m3_carrier_count"
                    elif kind == int(MemoryType.ROLE):
                        submitted[proposal.uid] = "new_m3_role_count"
                elif level == int(MemoryLevel.M4):
                    submitted[proposal.uid] = "new_m4_count"

            while True:
                before_cut = self.last_developmental_cut
                before_cycles = self._cycles
                self._v841_last_input_token = None
                self.submit_proposal = track
                try:
                    _run_full_cut_once(self)
                finally:
                    self.submit_proposal = submit

                cut = self.last_developmental_cut
                cycles_changed = self._cycles != before_cycles
                cut_changed = cut is not before_cut

                if cut is not None and cycles_changed and cut_changed:
                    break
                if cycles_changed or cut_changed:
                    raise RuntimeError(
                        "v8 stabilization produced a partial developmental cut"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "v8 stabilization timed out waiting for a developmental cut"
                    )
                time.sleep(_RETRY_POLL_SECONDS)

            commit_proposals()
            existing = {row.uid for row in cut.nodes}
            counts = Counter(kind for uid, kind in submitted.items() if uid not in existing)
            total = sum(counts.values())
            reason = "stable" if total == 0 else "max_cycles" if cycle == limit else None
            fields: dict[str, object] = {
                "stabilization_cycle": cycle,
                "new_m2_count": counts["new_m2_count"],
                "new_m3_carrier_count": counts["new_m3_carrier_count"],
                "new_m3_role_count": counts["new_m3_role_count"],
                "new_m4_count": counts["new_m4_count"],
            }
            if reason is not None:
                fields["stop_reason"] = reason
            flow.emit(
                "developmental",
                "stabilization",
                input_count=len(cut.nodes),
                output_count=total,
                fields=fields,
            )
            if reason is not None:
                return reason
    finally:
        self.submit_proposal = submit
        self._v82_stabilizing = prior_stabilizing
        if cancel is not None and was_cancelled:
            cancel.set()
        if not was_paused:
            self.resume()


def install_stabilization_noop_retry_v883() -> None:
    global _INSTALLED, _BASE_RUN_UNTIL_STABLE, _BASE_FULL_CUT_RUN_ONCE
    if _INSTALLED:
        return

    from v8 import peers_v82
    from v8 import snapshot_state_consistency_v845 as v845

    cls = peers_v82.V82DevelopmentalPeerSupervisor
    _BASE_RUN_UNTIL_STABLE = cls.run_until_stable
    _BASE_FULL_CUT_RUN_ONCE = v845._BASE_PEER_RUN_ONCE
    if not callable(_BASE_FULL_CUT_RUN_ONCE):
        raise RuntimeError("v8.45 did not expose the underlying full-cut peer authority")
    cls.run_until_stable = _run_until_stable_v883
    _INSTALLED = True
