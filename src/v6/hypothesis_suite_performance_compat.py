from __future__ import annotations

from typing import Any, Callable


def apply_hypothesis_suite_performance_compatibility() -> None:
    from v6 import hypothesis_h11_report as h11
    from v6 import hypothesis_suite_performance as perf
    from v6 import hypothesis_suite_report as suite

    # Preserve the public H11 API: direct callers historically requested the
    # full provenance JSONL unless they explicitly disabled it. The suite still
    # passes False explicitly for scalable epoch reporting.
    perf._h11_streaming_large.__kwdefaults__["write_full_provenance_jsonl"] = True

    original_call_supported = suite._call_supported
    if getattr(original_call_supported, "_v64_cache_compat", False):
        return

    h11_functions = {
        perf._h11_streaming_large,
        h11.evaluate_h11_future_option_transfer_concepts,
        suite.evaluate_h11_future_option_transfer_concepts,
    }

    def call_supported(function: Callable[..., Any], **kwargs: Any) -> Any:
        # Some legacy evaluators expose **kwargs and forward them internally.
        # Keep the shared cache suite-local unless the evaluator explicitly
        # participates in the cache contract (currently the large H11 path).
        if function not in h11_functions:
            kwargs.pop("suite_evidence_cache", None)
        return original_call_supported(function, **kwargs)

    call_supported._v64_cache_compat = True  # type: ignore[attr-defined]
    suite._call_supported = call_supported
