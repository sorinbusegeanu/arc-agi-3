from __future__ import annotations

from dataclasses import replace

from v8 import model as _model
from v8 import primary_valence as _primary


_SCHEMA_FIXED = False
_RUNTIME_FIXED = False


def install_schema_fixups() -> None:
    """Finalize schema invariants before any worker/runtime imports."""
    global _SCHEMA_FIXED
    if _SCHEMA_FIXED:
        return

    # dataclass(slots=True) creates a replacement class, so zero-argument
    # super() in a slotted subclass can retain the pre-decoration __class__ cell.
    # Call the captured canonical base explicitly.
    def proposal_post_init(self) -> None:
        _primary._BASE_MEMORY_PROPOSAL.__post_init__(self)
        if self.primary_valence_weight < 0.0:
            raise ValueError("primary_valence_weight cannot be negative")
        if self.primary_valence_sq_sum < 0.0:
            raise ValueError("primary_valence_sq_sum cannot be negative")
        if self.positive_valence_count < 0.0 or self.negative_valence_count < 0.0:
            raise ValueError("primary valence counts cannot be negative")

    _model.MemoryProposal.__post_init__ = proposal_post_init
    _primary.MemoryProposal.__post_init__ = proposal_post_init

    # Primary valence is the admitted primitive. Generic task-semantic reward
    # fields remain outside the observation contract so they cannot smuggle in
    # task interpretation. The signed terminal_polarity/primary_valence channel
    # is intentionally not forbidden.
    from v8 import observation_contract as contract_module

    current = contract_module.ARC_GRID_CONTRACT
    forbidden = tuple(
        dict.fromkeys(
            (*current.forbidden_semantic_fields, "reward", "win_value", "terminal_value")
        )
    )
    contract_module.ARC_GRID_CONTRACT = replace(
        current,
        forbidden_semantic_fields=forbidden,
    )
    _SCHEMA_FIXED = True


def install_runtime_fixups() -> None:
    """Keep batch transport lossless while live preference learning stays valence-grounded."""
    global _RUNTIME_FIXED
    if _RUNTIME_FIXED:
        return

    from v8 import actor as actor_module

    primary_merge = actor_module._merge_learning_batches

    def merge_learning_batches(rows):
        rows = tuple(rows)
        merged = primary_merge(rows)
        probes_by_actor: dict[tuple[int, str], list[object]] = {}
        for row in rows:
            probes_by_actor.setdefault((int(row.actor_id), str(row.game_id)), []).extend(
                tuple(row.preference_probes)
            )
        return tuple(
            replace(
                row,
                preference_probes=tuple(
                    probes_by_actor.get((int(row.actor_id), str(row.game_id)), ())
                ),
            )
            for row in merged
        )

    actor_module._merge_learning_batches = merge_learning_batches
    _RUNTIME_FIXED = True
