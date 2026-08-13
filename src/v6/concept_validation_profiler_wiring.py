from __future__ import annotations

_INSTALLED = False


def install_concept_validation_profiler_wiring() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v6 import concept_validation_profiler_context_fix as context_fix
    from v6 import concept_validation_sparse_cache as sparse

    # Sparse-cache creates the outer profiling context. Make its wrapped
    # validator reuse that context instead of entering the fastpath validator
    # that creates a second, hidden context. This exposes the real per-concept
    # workload and lets the sparse caches cover it.
    sparse._ORIGINAL = context_fix._validate_with_active_profile
    _INSTALLED = True
