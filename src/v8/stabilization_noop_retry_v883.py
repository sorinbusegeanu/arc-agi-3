from __future__ import annotations

"""v8.83: drain formation cheaply, then invoke the full-cut authority once.

Final stabilization runs after peers are paused and idle. It must not re-enter the
public active-runtime ``run_once`` wrapper chain because that chain contains bounded
scheduling/input-token guards which may legally return without producing a cut.

Repeated full analysis of the same large graph is both unnecessary and capable of
exhausting the stabilization timeout.  Formation-only cuts therefore advance the
developmental hierarchy until M2--M7 are quiescent.  One full cut then evaluates
the stable hierarchy.  A proven full-cut no-op may still be retried until the
existing stabilization timeout; partial advancement remains fatal.
"""

import time
from collections import Counter

from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryProposal,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
    proposal_fingerprint,
)


_INSTALLED = False
_BASE_RUN_UNTIL_STABLE = None
_BASE_FULL_CUT_RUN_ONCE = None
_RETRY_POLL_SECONDS = 0.005
_NOOP_RETRY_LIMIT = 2


def _process_role_formation(self, cut, frozen) -> None:
    """Run only the generic operator that advances carriers into M3 roles."""
    by_uid = {row.uid: row for row in cut.nodes}
    candidates = self.roles.propose(cut.nodes)
    for candidate in candidates[: self.candidate_budget]:
        cancel = getattr(self, "_v841_peer_cancel", None)
        if cancel is not None and cancel.is_set():
            return
        watermarks = [
            by_uid[uid].updated_watermark
            for uid in candidate.carriers
            if uid in by_uid
        ]
        if not watermarks or not self._fresh("role", candidate.uid, max(watermarks)):
            continue
        exact_games: set[int] = set()
        for carrier in candidate.carriers:
            exact_games.update(frozen.source_games(carrier))
        proposal = MemoryProposal(
            uid=candidate.uid,
            fingerprint=proposal_fingerprint(
                MemoryLevel.M3,
                MemoryType.ROLE,
                candidate.key_parts,
            ),
            event_id=self._event_id(),
            watermark=int(cut.watermark),
            level=MemoryLevel.M3,
            memory_type=MemoryType.ROLE,
            key_parts=candidate.key_parts,
            support_delta=len(candidate.carriers),
            explanatory_sum=float(len(candidate.carriers)),
            transfer_prior_sum=min(1.0, len(exact_games) / 2.0),
            score_weight=1.0,
            parent_uid=candidate.carriers[0],
            relation_type=RelationType.EXPLAINS,
            cognitive_state=int(CognitiveState.ACTIVE),
            validation_state=int(ValidationState.STRUCTURAL),
        )
        self._submit(proposal)
        source = by_uid.get(candidate.carriers[0])
        if source is not None:
            self._append_evidence(
                "role_emergence",
                source,
                1.0,
                validation_state=int(ValidationState.STRUCTURAL),
            )


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


def _run_formation_cut_once(self, *, include_correspondence: bool = True) -> None:
    """Run one developmental cut without the expensive generic peer analyses."""
    from v8.developmental_cut import capture_developmental_cut
    from v8.peers_v82 import _FrozenCutReadView

    state_lock = getattr(self, "_v845_state_lock", None)
    run_lock = getattr(self, "_v82_run_lock", None)

    def run() -> None:
        if run_lock is not None:
            run_lock.acquire()
        live_read_view = self.read_view
        try:
            cancel = getattr(self, "_v841_peer_cancel", None)
            if cancel is not None and cancel.is_set():
                return
            cut = capture_developmental_cut(
                live_read_view,
                generation=int(self.current_generation()),
                watermark=int(self.current_watermark()),
            )
            self._last_developmental_cut = cut
            frozen = _FrozenCutReadView(cut, cancel_event=cancel)
            if frozen.cancelled or (cancel is not None and cancel.is_set()):
                return
            self.read_view = frozen
            self._process_formation(cut, frozen)
            if cancel is None or not cancel.is_set():
                _process_role_formation(self, cut, frozen)
            if include_correspondence and (cancel is None or not cancel.is_set()):
                self._process_correspondence(cut, frozen)
            if cancel is None or not cancel.is_set():
                self._cycles += 1
        finally:
            self.read_view = live_read_view
            if run_lock is not None:
                run_lock.release()

    if state_lock is None:
        run()
    else:
        with state_lock:
            run()


def _run_until_stable_v883(
    self,
    max_cycles: int = 8,
    *,
    commit_proposals,
    timeout: float = 60.0,
) -> str:
    from v8 import information_flow_diagnostics as flow

    # Preserve dependency-injected test/specialist supervisors. Production
    # instances use the bounded immutable-cut implementation below.
    if "run_once" in getattr(self, "__dict__", {}):
        return _BASE_RUN_UNTIL_STABLE(
            self,
            max_cycles=max_cycles,
            commit_proposals=commit_proposals,
            timeout=timeout,
        )

    limit = min(8, max(1, int(max_cycles)))
    operation_timeout = max(0.0, float(timeout))
    operation_deadline = time.monotonic() + operation_timeout

    def remaining_timeout() -> float:
        return max(0.0, operation_deadline - time.monotonic())
    was_paused = self._pause.is_set()
    self.pause()
    cancel = getattr(self, "_v841_peer_cancel", None)
    was_cancelled = bool(cancel is not None and cancel.is_set())
    submit = self.submit_proposal
    prior_stabilizing = getattr(self, "_v82_stabilizing", False)
    try:
        if not self.wait_idle(remaining_timeout()):
            flow.emit(
                "developmental",
                "stabilization",
                input_count=0,
                output_count=0,
                rejection_counts={"peer_cycle_not_idle": 1},
                fields={"stop_reason": "timeout_waiting_for_idle"},
            )
            return "timeout_waiting_for_idle"
        if cancel is not None:
            cancel.clear()
        self._v82_stabilizing = True
        commit_proposals()

        formation_only = True
        for cycle in range(1, limit + 1):
            if remaining_timeout() <= 0.0:
                flow.emit(
                    "developmental",
                    "stabilization",
                    input_count=0,
                    output_count=0,
                    rejection_counts={"overall_timeout_before_next_cut": 1},
                    fields={
                        "stabilization_cycle": cycle,
                        "stabilization_phase": (
                            "formation" if formation_only else "full_analysis"
                        ),
                        "stop_reason": "timeout",
                    },
                )
                return "timeout"
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
                elif level == int(MemoryLevel.M5):
                    submitted[proposal.uid] = "new_m5_count"
                elif level == int(MemoryLevel.M6):
                    submitted[proposal.uid] = "new_m6_count"
                elif level == int(MemoryLevel.M7):
                    submitted[proposal.uid] = "new_m7_count"

            noop_attempts = 0
            while True:
                before_cut = self.last_developmental_cut
                before_cycles = self._cycles
                self._v841_last_input_token = None
                self.submit_proposal = track
                try:
                    if formation_only:
                        _run_formation_cut_once(self)
                    else:
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
                noop_attempts += 1
                if remaining_timeout() <= 0.0:
                    flow.emit(
                        "developmental",
                        "stabilization",
                        input_count=0,
                        output_count=0,
                        rejection_counts={"developmental_cut_timeout": 1},
                        fields={
                            "stabilization_cycle": cycle,
                            "stabilization_phase": (
                                "formation" if formation_only else "full_analysis"
                            ),
                            "noop_attempts": noop_attempts,
                            "stop_reason": "timeout",
                        },
                    )
                    return "timeout"
                if noop_attempts >= _NOOP_RETRY_LIMIT:
                    # A wrapper-level no-op is deterministic while peers are
                    # paused and the graph is unchanged. Retrying it for the full
                    # drain timeout previously reran post-cut H13 diagnostics for
                    # minutes and then crashed the otherwise-complete run.
                    flow.emit(
                        "developmental",
                        "stabilization",
                        input_count=0,
                        output_count=0,
                        rejection_counts={"developmental_cut_not_started": 1},
                        fields={
                            "stabilization_cycle": cycle,
                            "stabilization_phase": (
                                "formation" if formation_only else "full_analysis"
                            ),
                            "noop_attempts": noop_attempts,
                            "stop_reason": "incomplete_cut",
                        },
                    )
                    return "incomplete_cut"
                time.sleep(_RETRY_POLL_SECONDS)

            commit_proposals()
            existing = {row.uid for row in cut.nodes}
            counts = Counter(kind for uid, kind in submitted.items() if uid not in existing)
            total = sum(counts.values())
            phase = "formation" if formation_only else "full_analysis"
            if formation_only and total == 0 and cycle < limit:
                formation_only = False
                reason = None
            elif not formation_only and total == 0:
                reason = "stable"
            elif cycle == limit:
                reason = "max_cycles"
            else:
                formation_only = True
                reason = None
            fields: dict[str, object] = {
                "stabilization_cycle": cycle,
                "stabilization_phase": phase,
                "new_m2_count": counts["new_m2_count"],
                "new_m3_carrier_count": counts["new_m3_carrier_count"],
                "new_m3_role_count": counts["new_m3_role_count"],
                "new_m4_count": counts["new_m4_count"],
                "new_m5_count": counts["new_m5_count"],
                "new_m6_count": counts["new_m6_count"],
                "new_m7_count": counts["new_m7_count"],
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
