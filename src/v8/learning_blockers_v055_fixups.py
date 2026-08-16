from __future__ import annotations

import os
from collections import defaultdict

from v8.model import MemoryUid


_BASE_V055_ACTOR_WORKER = None
_INSTALLED = False


def _actor_worker_v055(*, job, **kwargs):
    """Picklable process entry point for the v8.5 game-scoped control context."""
    if _BASE_V055_ACTOR_WORKER is None:
        raise RuntimeError("v8.5 actor worker is not installed")
    prior = os.environ.get("ARC_AGI3_V8_CONTROL_SCOPE")
    os.environ["ARC_AGI3_V8_CONTROL_SCOPE"] = str(job.game_id)
    try:
        return _BASE_V055_ACTOR_WORKER(job=job, **kwargs)
    finally:
        if prior is None:
            os.environ.pop("ARC_AGI3_V8_CONTROL_SCOPE", None)
        else:
            os.environ["ARC_AGI3_V8_CONTROL_SCOPE"] = prior


def _restore_repeatable_preference_evidence() -> None:
    """Allow independent repeated probes; v8.5 removes duplication at the recorder path."""
    from v8 import preference as preference_module

    Probe = preference_module.PreferenceProbe

    def record_probe(
        self,
        *,
        outcome_a: MemoryUid,
        outcome_b: MemoryUid,
        context_bucket: int,
        chosen_outcome: MemoryUid,
        both_reachable: bool,
        preference_influenced: bool,
    ) -> bool:
        if outcome_a == outcome_b:
            return False
        if chosen_outcome not in {outcome_a, outcome_b}:
            return False
        if not both_reachable or preference_influenced:
            return False
        self._probes.append(
            Probe(
                outcome_a,
                outcome_b,
                int(context_bucket),
                chosen_outcome,
                True,
                False,
            )
        )
        return True

    def load_state(self, state: dict[str, object] | None) -> None:
        if not state:
            return
        for raw in state.get("probes", []):
            if not isinstance(raw, dict):
                continue
            a = raw.get("outcome_a", [0, 0])
            b = raw.get("outcome_b", [0, 0])
            chosen = raw.get("chosen_outcome", [0, 0])
            self._probes.append(
                Probe(
                    MemoryUid(int(a[0]), int(a[1])),
                    MemoryUid(int(b[0]), int(b[1])),
                    int(raw.get("context_bucket", 0)),
                    MemoryUid(int(chosen[0]), int(chosen[1])),
                    bool(raw.get("both_reachable", True)),
                    bool(raw.get("preference_influenced", False)),
                )
            )

    preference_module.PreferenceEstimator.record_probe = record_probe
    preference_module.PreferenceEstimator.load_state = load_state


def install_learning_blockers_v055_fixups() -> None:
    global _BASE_V055_ACTOR_WORKER, _INSTALLED
    if _INSTALLED:
        return
    from v8 import actor as actor_module

    # The v8.5 installer wrapped the actor with a closure. Keep its behavior but
    # expose a module-level target so spawn/forkserver can pickle it.
    _BASE_V055_ACTOR_WORKER = actor_module.actor_worker
    actor_module.actor_worker = _actor_worker_v055

    # Repeated independently observed preference probes are legitimate evidence.
    # The blocker was duplicate publication through two runtime recorder paths,
    # which v8.5 already removes; do not globally deduplicate observations.
    _restore_repeatable_preference_evidence()
    _INSTALLED = True
