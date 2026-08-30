from __future__ import annotations

"""v8.78 final research-integrity authority.

This layer removes the remaining ambiguity exposed by a clean research_1 run:

* normalized M2 formation emits only the canonical ``family_compression`` evidence;
* a current-run verified WIN is visible to both lease dispatch and final telemetry;
* evidence provenance counts are named as source/target worlds, not game counts;
* optimizer counters carry their actual stage scopes in runtime/research summaries.

No learning threshold, formation gate, transfer rule, or sampling weight is changed.
"""

from dataclasses import replace


_INSTALLED = False
_BASE_CHOOSE_MODE = None
_BASE_RUNTIME_METRICS = None
_BASE_EVIDENCE_DIGEST = None
_BASE_SUMMARY_STATE = None

_OPTIMIZER_SCOPE_NOTE = (
    "candidates_generated, trajectories_seen, and validated_variants describe the "
    "optimization-edit pipeline; validations and validation_successes also include "
    "source-validation replay attempts and are not expected to be one-to-one."
)
_PROVENANCE_SCOPE_NOTE = (
    "source/target hashes identify provenance worlds, not selected game IDs: ARC "
    "sources are game-scoped, while generic sources are environment-instance/seed-scoped."
)


def _install_canonical_family_compression() -> None:
    from v8 import evaluation
    from v8 import intelligence_loop_v087 as intelligence
    from v8 import mixed_research_runtime_integrity_v875 as v875

    # v8.75 already defines the canonical conversion. v8.76 temporarily restored the
    # older label; make the production authority canonical again.
    intelligence._compression_to_candidate = v875._compression_to_candidate_v875

    # H03 has one scientific evidence name. Do not keep two spellings active.
    contracts = []
    for contract in evaluation.CONTRACTS:
        if contract.hypothesis_id == "H03":
            contract = replace(contract, required_kinds=("family_compression",))
        contracts.append(contract)
    evaluation.CONTRACTS = tuple(contracts)


def _promote_current_run_win(coordinator, game_id: str) -> bool:
    from v8 import runtime_win_optimization_v834 as runtime_win

    if not all(
        hasattr(coordinator, name)
        for name in ("_lock", "_game_won", "_records")
    ):
        return False
    return bool(runtime_win._promote_runtime_win_if_present(coordinator, str(game_id)))


def _telemetry_game_state_v878(self, game_id: str):
    """Expose fresh current-run competence without forcing a graph refresh."""
    from v8 import adaptive_learning_allocation_v819 as allocation
    from v8 import adaptive_learning_allocation_v819_solve_fix as solve_fix
    from v8 import lifecycle_competence_integration_v827 as lifecycle

    game = str(game_id)
    if _promote_current_run_win(self, game):
        # The raw coordinator state sees the just-promoted current-run WIN but does
        # not rebuild the lifecycle graph. A fresh verified success is valid current-
        # run competence even before a canonical M7 frontier exists.
        state = lifecycle._BASE_GAME_STATE(self, game)
        if state != allocation.GameLearningState.UNSOLVED:
            return state
    return solve_fix._cached_game_state(self, game)


def _choose_mode_v878(self, game_id: str):
    """Promote a current-run WIN before v8.43 performs its cheap raw-state gate."""
    _promote_current_run_win(self, str(game_id))
    return _BASE_CHOOSE_MODE(self, str(game_id))


def _optimizer_with_scope(value):
    if not isinstance(value, dict):
        try:
            value = dict(value)
        except (TypeError, ValueError):
            return value
    payload = dict(value)
    if payload:
        payload["counter_scope_note"] = _OPTIMIZER_SCOPE_NOTE
    return payload


def _runtime_metrics_v878(self):
    payload = dict(_BASE_RUNTIME_METRICS(self))
    if "trajectory_optimizer" in payload:
        payload["trajectory_optimizer"] = _optimizer_with_scope(
            payload.get("trajectory_optimizer", {})
        )
    return payload


def _summary_state_v878(summary):
    payload = dict(_BASE_SUMMARY_STATE(summary))
    payload["trajectory_optimizer"] = _optimizer_with_scope(
        payload.get("trajectory_optimizer", {})
    )
    return payload


def _evidence_digest_v878(path, *, start_offset: int = 0):
    payload = dict(_BASE_EVIDENCE_DIGEST(path, start_offset=start_offset))
    if "distinct_source_games" in payload:
        payload["distinct_source_worlds"] = payload.pop("distinct_source_games")
    if "distinct_target_games" in payload:
        payload["distinct_target_worlds"] = payload.pop("distinct_target_games")
    if payload.get("available"):
        payload["provenance_scope_note"] = _PROVENANCE_SCOPE_NOTE
    return payload


def install_research_integrity_v878() -> None:
    global _INSTALLED
    global _BASE_CHOOSE_MODE, _BASE_RUNTIME_METRICS
    global _BASE_EVIDENCE_DIGEST, _BASE_SUMMARY_STATE
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819 as allocation
    from v8 import lease_dispatch_lifecycle_v843 as dispatch
    from v8 import mixed_research_runtime_integrity_v875 as v875
    from v8.research import experiment_artifacts
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    _install_canonical_family_compression()

    allocation.AdaptiveLearningCoordinator._v819_telemetry_game_state = (
        _telemetry_game_state_v878
    )
    v875._authoritative_telemetry_game_state_v875 = _telemetry_game_state_v878

    _BASE_CHOOSE_MODE = allocation.AdaptiveLearningCoordinator.choose_mode
    dispatch._choose_mode_v843 = _choose_mode_v878
    allocation.AdaptiveLearningCoordinator.choose_mode = dispatch._choose_mode_v843

    _BASE_RUNTIME_METRICS = V82ContinuousMemoryRuntime.metrics
    V82ContinuousMemoryRuntime.metrics = _runtime_metrics_v878

    _BASE_EVIDENCE_DIGEST = experiment_artifacts._evidence_digest
    experiment_artifacts._evidence_digest = _evidence_digest_v878
    _BASE_SUMMARY_STATE = experiment_artifacts._summary_state
    experiment_artifacts._summary_state = _summary_state_v878

    _INSTALLED = True
