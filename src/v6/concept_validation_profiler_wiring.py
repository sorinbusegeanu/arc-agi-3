from __future__ import annotations

from typing import Any

_INSTALLED = False


def _sparse_profile_bridge(*args: Any, **kwargs: Any):
    from v6 import concept_validation_fastpath as fast
    from v6 import concept_validation_profiler_context_fix as context_fix

    config = kwargs.get("config")
    if config is None and len(args) > 1:
        config = args[1]
    if config is not None and not bool(getattr(config, "enabled", False)):
        return fast._ORIGINALS["validate_incremental_promotions_only"](*args, **kwargs)
    return context_fix._validate_with_active_profile(*args, **kwargs)


def install_concept_validation_profiler_wiring() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v6 import concept_validation_sparse_cache as sparse

    # Sparse-cache creates the outer profiling context. Reuse it for enabled
    # validation, while preserving the legacy no-profile result when validation
    # is disabled.
    sparse._ORIGINAL = _sparse_profile_bridge
    _INSTALLED = True
