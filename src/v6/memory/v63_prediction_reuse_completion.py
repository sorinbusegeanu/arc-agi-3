from __future__ import annotations

from typing import Any


_INSTALLED = False


def install_v63_prediction_reuse_completion() -> None:
    """Reuse the exact prediction produced while ranking the selected action."""
    global _INSTALLED
    if _INSTALLED:
        return

    from v6.memory import v63_performance_completion as completion
    from v6.memory.v621_runtime import (
        V621MemoryController,
        V621SnapshotMemoryQueryEngine,
    )

    completion._cache_prediction = _cache_prediction_by_action
    V621MemoryController.predict = _controller_predict_reuse
    V621SnapshotMemoryQueryEngine.score_action = _snapshot_score_capture_prediction
    _INSTALLED = True


def _context_key(context_signatures: dict[int, tuple], action: int) -> tuple[Any, ...]:
    return (
        int(action),
        tuple(
            (int(level), tuple(context_signatures[level]))
            for level in sorted(context_signatures)
        ),
    )


def _cache_prediction_by_action(
    engine: Any,
    context_signatures: dict[int, tuple],
    action: int,
    prediction: Any,
) -> None:
    cache = getattr(engine, "_v63_prediction_cache_by_action", None)
    if cache is None:
        cache = {}
        engine._v63_prediction_cache_by_action = cache
    cache[int(action)] = (_context_key(context_signatures, action), prediction)


def _controller_predict_reuse(
    self: Any,
    context_signatures: dict[int, tuple],
    action: int,
    *,
    record_query: bool = False,
) -> Any:
    if not record_query:
        cache = getattr(self.query_engine, "_v63_prediction_cache_by_action", None)
        cached = None if not isinstance(cache, dict) else cache.get(int(action))
        if cached is not None and cached[0] == _context_key(context_signatures, action):
            return cached[1]
    from v6.memory import v63_performance_completion as completion

    return completion._ORIGINAL_CONTROLLER_PREDICT(
        self,
        context_signatures,
        action,
        record_query=record_query,
    )


def _snapshot_score_capture_prediction(
    self: Any,
    context_signatures: dict[int, tuple],
    action: int,
    available_actions: list[int],
    *,
    record_query: bool = False,
) -> Any:
    from v6.memory import query_engine as query_module
    from v6.memory import v63_performance_completion as completion

    captured: dict[str, Any] = {}
    original_compute = query_module.compute_memory_action_score

    def capture_compute(*args: Any, **kwargs: Any) -> Any:
        prediction = kwargs.get("prediction")
        if prediction is not None:
            captured["prediction"] = prediction
        return original_compute(*args, **kwargs)

    query_module.compute_memory_action_score = capture_compute
    try:
        score = completion._ORIGINAL_SNAPSHOT_SCORE_ACTION(
            self,
            context_signatures,
            action,
            available_actions,
            record_query=record_query,
        )
    finally:
        query_module.compute_memory_action_score = original_compute

    prediction = captured.get("prediction")
    if prediction is not None:
        _cache_prediction_by_action(self, context_signatures, action, prediction)
    return score
