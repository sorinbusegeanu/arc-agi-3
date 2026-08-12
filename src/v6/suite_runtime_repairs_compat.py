from __future__ import annotations

from typing import Any

_INSTALLED = False
_REAL_WORLD_MODEL_DERIVER: Any = None
_REPAIRED_PREPARE: Any = None


def install_suite_runtime_repairs_compat() -> None:
    """Preserve test/plugin monkeypatch semantics without disabling runtime refresh."""
    global _INSTALLED, _REAL_WORLD_MODEL_DERIVER, _REPAIRED_PREPARE
    if _INSTALLED:
        return
    from v6 import hypothesis_suite_report as suite

    _REAL_WORLD_MODEL_DERIVER = suite.derive_world_model_components_only
    _REPAIRED_PREPARE = suite.prepare_hypothesis_evidence
    suite.prepare_hypothesis_evidence = _prepare_compat
    _INSTALLED = True


def _prepare_compat(**kwargs: Any) -> dict[str, Any]:
    from v6 import hypothesis_suite_report as suite
    from v6 import suite_runtime_repairs as repairs

    # Tests and external plugins may temporarily replace the canonical M5
    # deriver to inspect call order. In that case retain the historical
    # one-call orchestration contract; real production runs use the repaired
    # post-future refresh path.
    if suite.derive_world_model_components_only is not _REAL_WORLD_MODEL_DERIVER:
        return repairs._ORIGINAL_PREPARE(**kwargs)
    return _REPAIRED_PREPARE(**kwargs)
