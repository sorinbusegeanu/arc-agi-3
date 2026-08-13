from __future__ import annotations

from typing import Any

_INSTALLED = False
_ORIGINAL_CONTEXT_VALIDATE: Any = None


def _sparse_profile_bridge(*args: Any, **kwargs: Any):
    from v6 import concept_validation_fastpath as fast

    config = kwargs.get("config")
    if config is None and len(args) > 1:
        config = args[1]
    if config is not None and not bool(getattr(config, "enabled", False)):
        return fast._ORIGINALS["validate_incremental_promotions_only"](*args, **kwargs)
    return _ORIGINAL_CONTEXT_VALIDATE(*args, **kwargs)


def install_concept_validation_profiler_wiring() -> None:
    global _INSTALLED, _ORIGINAL_CONTEXT_VALIDATE
    if _INSTALLED:
        return

    from v6 import concept_validation_profiler_context_fix as context_fix
    from v6 import concept_validation_sparse_cache as sparse

    _ORIGINAL_CONTEXT_VALIDATE = context_fix._validate_with_active_profile
    context_fix._validate_with_active_profile = _sparse_profile_bridge
    sparse._ORIGINAL = _sparse_profile_bridge
    _INSTALLED = True
