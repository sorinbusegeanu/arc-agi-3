from __future__ import annotations

from v6 import concept_validation_profiler_context_fix as context_fix
from v6 import concept_validation_sparse_cache as sparse


def test_sparse_validator_reuses_context_bridge() -> None:
    assert sparse._ORIGINAL is context_fix._validate_with_active_profile
