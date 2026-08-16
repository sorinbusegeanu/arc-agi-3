from __future__ import annotations

import os
from collections import defaultdict

from v8.model import MemoryLevel, MemoryType, MemoryUid
from v8.world_model import WorldModelComponent


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


def _world_model_propose_v055(self, rows):
    """Aggregate only matching consequence structure and primary-valence direction."""
    grouped = defaultdict(list)
    for row in rows:
        if int(row.level) != int(MemoryLevel.M5):
            continue
        if int(row.memory_type) != int(MemoryType.CONSEQUENCE) or len(row.key_parts) < 4:
            continue
        weight = max(0.0, float(getattr(row, "primary_valence_weight", 0.0)))
        value = 0.0 if weight <= 0.0 else float(getattr(row, "expected_primary_valence", 0.0))
        valence_bucket = 1 if value > 1e-9 else -1 if value < -1e-9 else 0
        key = (int(row.key_parts[2]), int(row.key_parts[3]), int(valence_bucket))
        grouped[key].append(row)

    result = []
    for key, members in grouped.items():
        distinct_concepts = {
            (int(row.key_parts[0]), int(row.key_parts[1])) for row in members
        }
        if len(distinct_concepts) < int(self.min_consequences):
            continue
        mask = 0
        for row in members:
            mask |= int(row.game_mask)
        result.append(
            WorldModelComponent(
                MemoryUid.from_key(MemoryLevel.M5, MemoryType.WORLD_MODEL, key),
                key,
                tuple(sorted(row.uid for row in members)),
                sum(max(0, int(row.support_count)) for row in members),
                mask.bit_count(),
            )
        )
    return tuple(result)


def install_learning_blockers_v055_fixups() -> None:
    global _BASE_V055_ACTOR_WORKER, _INSTALLED
    if _INSTALLED:
        return
    from v8 import actor as actor_module
    from v8.world_model import WorldModelEstimator

    # The v8.5 installer wrapped the actor with a closure. Keep its behavior but
    # expose a module-level target so spawn/forkserver can pickle it.
    _BASE_V055_ACTOR_WORKER = actor_module.actor_worker
    actor_module.actor_worker = _actor_worker_v055

    # Repeated independently observed preference probes are legitimate evidence.
    # The blocker was duplicate publication through two runtime recorder paths,
    # which v8.5 already removes; do not globally deduplicate observations.
    _restore_repeatable_preference_evidence()

    # Preserve structural consequence identity instead of collapsing every M5 row
    # with the same future/valence direction, while still separating valence sign.
    WorldModelEstimator.propose = _world_model_propose_v055
    _INSTALLED = True
