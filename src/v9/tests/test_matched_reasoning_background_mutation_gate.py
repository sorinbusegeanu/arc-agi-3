from __future__ import annotations

import pytest

from v9.runtime.developmental_cut import DevelopmentalMutationGate
from v9.runtime.scientific_modes import ScientificVisibilityMode


def test_matched_mode_rejects_background_developmental_publication() -> None:
    gate = DevelopmentalMutationGate(ScientificVisibilityMode.MATCHED_REASONING)
    with pytest.raises(RuntimeError, match="outside DevelopmentalCut"):
        gate.assert_publication_allowed()
    with gate.allow_cut("cut-1"):
        gate.assert_publication_allowed()


def test_async_mode_requires_declared_async_origin() -> None:
    gate = DevelopmentalMutationGate(ScientificVisibilityMode.ASYNC_DEVELOPMENT)
    gate.assert_publication_allowed(async_origin=True)
