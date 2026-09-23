from __future__ import annotations

import hashlib

import pytest

from v9.runtime.shared_batch_transport import ProducerCausalAdmission, TransportSlabDescriptor


def _descriptor(start: int, end: int, *, producer: int = 1) -> TransportSlabDescriptor:
    payload = f"{start}:{end}".encode()
    return TransportSlabDescriptor("slab", 0, 0, len(payload), end - start + 1, producer, start, end, 0, hashlib.sha256(payload).hexdigest())


def test_causal_admission_holds_gaps_and_releases_contiguously() -> None:
    admission = ProducerCausalAdmission()
    late = _descriptor(3, 4)
    assert admission.admit(late, now=0.0) == ()
    assert admission.admit(_descriptor(1, 2), now=0.1) == (_descriptor(1, 2), late)
    with pytest.raises(RuntimeError, match="duplicate/stale"):
        admission.admit(_descriptor(1, 1), now=0.2)


def test_causal_gap_timeout_is_explicit() -> None:
    admission = ProducerCausalAdmission(gap_timeout_seconds=1.0)
    admission.admit(_descriptor(2, 2), now=0.0)
    with pytest.raises(TimeoutError, match="gap timeout"):
        admission.check_gap_timeouts(now=2.0)


def test_gap_budget_is_global_across_producers() -> None:
    admission = ProducerCausalAdmission(gap_rows_limit=2, gap_bytes_limit=100)
    admission.admit(_descriptor(2, 2, producer=1), now=0.0)
    admission.admit(_descriptor(2, 2, producer=2), now=0.0)
    with pytest.raises(OverflowError, match="gap window"):
        admission.admit(_descriptor(2, 2, producer=3), now=0.0)


def test_producer_identity_state_has_a_hard_ceiling() -> None:
    admission = ProducerCausalAdmission(producer_limit=2)
    admission.admit(_descriptor(1, 1, producer=1))
    admission.admit(_descriptor(1, 1, producer=2))
    with pytest.raises(OverflowError, match="identity ceiling"):
        admission.admit(_descriptor(1, 1, producer=3))
