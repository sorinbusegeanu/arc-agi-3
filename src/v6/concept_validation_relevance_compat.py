from __future__ import annotations

from typing import Any

_INSTALLED = False
_ORIGINAL_DIAGNOSTICS: Any = None


def _diagnostics_with_population_accounting(*args: Any, **kwargs: Any):
    result = _ORIGINAL_DIAGNOSTICS(*args, **kwargs)
    if not isinstance(result, tuple) or len(result) != 3:
        return result
    events, diagnostics, state = result
    if not isinstance(diagnostics, dict):
        return result

    total = int(diagnostics.get("total_later_event_count", 0) or 0)
    relevant = int(diagnostics.get("relevant_heldout_event_count", 0) or 0)
    invalid = int(diagnostics.get("invalid_explanation_event_count", 0) or 0)
    derived_unrelated = max(0, total - relevant - invalid)
    diagnostics["unrelated_event_count"] = max(
        int(diagnostics.get("unrelated_event_count", 0) or 0),
        derived_unrelated,
    )
    return events, diagnostics, state


def install_concept_validation_relevance_compat() -> None:
    global _INSTALLED, _ORIGINAL_DIAGNOSTICS
    if _INSTALLED:
        return

    from v6 import concept_validation_fastpath_compat as compat
    from v6 import higher_order_substrate as substrate

    # Existing integration tests and callers treat the compatibility module's
    # future-option helper as the public installed helper. Keep that binding
    # while letting the implementation be the relevance-pruned version.
    compat._future_option_motif_explanation_events = substrate._future_option_motif_explanation_events

    _ORIGINAL_DIAGNOSTICS = substrate._build_functional_explanation_diagnostics
    substrate._build_functional_explanation_diagnostics = _diagnostics_with_population_accounting
    _INSTALLED = True
