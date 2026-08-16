from __future__ import annotations

from dataclasses import replace

from v8.model import MemoryLevel, stable_u64
from v8.normalized_memory_v086 import (
    MAX_NORMALIZED_FACTS_PER_EVENT,
    V86PipelineEvent,
    _BASE_PIPELINE_EVENT,
    _BASE_PIPELINE_PACKET_SIZE,
    stage_worker_v086,
)
from v8.structural_events import is_normalized_fact_token, native_action_set_signature


_INSTALLED = False
_CURRENT_ACTION_SET_SIGNATURE = 0


def _pipeline_post_init(self) -> None:
    """Validate the slotted dataclass without zero-argument super()."""
    base_post_init = getattr(_BASE_PIPELINE_EVENT, "__post_init__", None)
    if base_post_init is not None:
        base_post_init(self)
    elif int(self.multiplicity) <= 0:
        raise ValueError("pipeline multiplicity must be positive")
    if int(self.elapsed_since_change) < 0:
        raise ValueError("elapsed_since_change cannot be negative")
    if len(self.normalized_facts) > MAX_NORMALIZED_FACTS_PER_EVENT:
        raise ValueError("too many normalized M1N facts")
    for value in self.normalized_facts:
        if not is_normalized_fact_token(int(value)):
            raise ValueError("invalid normalized M1N fact token")


def install_normalized_memory_v086_fixups() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v7.environment import arc_adapter as adapter
    from v7.environment import encoding
    from v8 import development
    from v8 import model
    from v8 import runtime

    V86PipelineEvent.__post_init__ = _pipeline_post_init

    # The legacy decoder captured by v8.6 consults model.PIPELINE_PACKET_SIZE at
    # call time. Keep that model-level constant at the legacy size while the
    # runtime ring uses runtime.PIPELINE_PACKET_SIZE for the larger v8.6 packet.
    model.PIPELINE_PACKET_SIZE = int(_BASE_PIPELINE_PACKET_SIZE)

    base_available = adapter.ArcGridEnvironment.available_actions
    base_structural = encoding.structural_grid_signature

    def available_actions(self):
        global _CURRENT_ACTION_SET_SIGNATURE
        values = list(base_available(self))
        _CURRENT_ACTION_SET_SIGNATURE = native_action_set_signature(values)
        return values

    def structural_grid_signature(grid):
        return stable_u64(
            int(base_structural(grid)),
            int(_CURRENT_ACTION_SET_SIGNATURE),
            person=b"v8.6-action-context",
        )

    adapter.ArcGridEnvironment.available_actions = available_actions
    encoding.structural_grid_signature = structural_grid_signature

    # Preserve the existing primary-valence invariant: a grounded terminal event
    # remains maximally significant. v8.6 removes changed-cell magnitude as the
    # importance proxy; it does not weaken primitive valence.
    base_derive = development.derive_proposal

    def derive_proposal(level, event):
        proposal = base_derive(level, event)
        if (
            int(level) == int(MemoryLevel.M1)
            and len(proposal.key_parts) >= 4
            and int(event.experience.terminal_polarity) != 0
        ):
            multiplicity = max(1, int(event.multiplicity))
            proposal = replace(
                proposal,
                significance_sum=float(multiplicity),
                learning_value_sum=max(
                    float(proposal.learning_value_sum), float(multiplicity)
                ),
            )
        return proposal

    development.derive_proposal = derive_proposal

    # Runtime already points at the v8.6 worker. Keep the development entry point
    # consistent for direct tests/tools and future runtime constructors.
    development.stage_worker = stage_worker_v086
    runtime.stage_worker = stage_worker_v086

    _INSTALLED = True
