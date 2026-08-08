from __future__ import annotations

from types import SimpleNamespace

from v6.memory.v63_performance import (
    _choose_with_sampler_prior_cached,
    _emit_live_memory_event_deduplicated,
    install_v63_validation_performance,
)


class _Queue:
    def __init__(self) -> None:
        self.items = []
        self.qsize_calls = 0

    def put_nowait(self, item) -> None:
        self.items.append(item)

    def qsize(self) -> int:
        self.qsize_calls += 1
        return len(self.items)


def test_live_memory_projection_deduplicates_identical_payload() -> None:
    queue = _Queue()
    system = SimpleNamespace(
        live_memory_queue=queue,
        config=SimpleNamespace(
            shared_live_memory_mode="readwrite",
            live_memory_worker_id="worker",
        ),
        live_memory_queue_block_seconds=0.0,
        live_memory_queue_peak_size=0,
        live_memory_events_emitted=0,
        live_memory_events_dropped_queue_full=0,
        live_memory_events_dropped_error=0,
    )
    args = (
        "stable_contingency",
        "stable:1",
        10,
        0.8,
        {"key": "stable:1", "support_count": 20},
    )
    _emit_live_memory_event_deduplicated(system, *args)
    _emit_live_memory_event_deduplicated(system, *args)
    _emit_live_memory_event_deduplicated(
        system,
        "stable_contingency",
        "stable:1",
        11,
        0.8,
        {"key": "stable:1", "support_count": 21},
    )

    assert len(queue.items) == 2
    assert system.live_memory_events_emitted == 2
    assert system._v63_live_events_deduplicated == 1
    assert queue.qsize_calls == 0


def test_sampler_override_reuses_precomputed_ranking() -> None:
    ranked = [
        SimpleNamespace(action=2, score=0.9),
        SimpleNamespace(action=1, score=0.4),
    ]

    class Controller:
        def __init__(self) -> None:
            self._v63_precomputed_sampler_ranking = ranked
            self._last_ranked_scores = {}
            self.calls = 0
            self.audit = []

        def choose_action_candidates(self, *_args, **_kwargs):
            self.calls += 1
            return ranked

        def _audit(self, event_type, *, owner_id, payload):
            self.audit.append((event_type, owner_id, payload))

    controller = Controller()
    selected = _choose_with_sampler_prior_cached(
        controller,
        context_signatures_by_action={1: {0: (1,)}, 2: {0: (2,)}},
        available_actions=[1, 2],
        sampler_action=1,
        override_margin=0.15,
    )

    assert selected == 2
    assert controller.calls == 0
    assert controller._v63_precomputed_sampler_ranking is None
    assert controller.audit


def test_validation_transfer_history_rate_cache() -> None:
    from v6 import higher_order_substrate as substrate

    install_v63_validation_performance()
    series = substrate._TransferHistorySeries(
        steps=(2, 4, 6),
        success_prefix=(0, 1, 1, 2),
    )
    index = substrate._TransferHistoryIndex(
        by_role={"r": series},
        by_source_target_scope={},
        all_rows=series,
    )

    first = index.rate_before(role="r", step=6)
    second = index.rate_before(role="r", step=6)

    assert first == second == (0.5, 2)
    assert len(index._v63_rate_cache) == 1
