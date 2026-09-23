from __future__ import annotations

import hashlib
import struct
from threading import Event, Thread

from v9.runtime.canonical_wal import (
    CanonicalCommitWAL,
    _FOOTER,
    _HEADER,
)


def test_group_is_visible_only_with_valid_footer(tmp_path) -> None:
    path = tmp_path / "canonical.wal"
    wal = CanonicalCommitWAL(path)
    wal.append_group(({"transaction_id": "first"}, {"transaction_id": "second"}))
    payload = path.read_bytes()
    path.write_bytes(payload[:-20])
    recovery = wal.recover(truncate=True)
    assert recovery.frames == ()
    assert recovery.wal_durable_lsn == 0
    assert path.read_bytes() == b""


def test_recovery_rejects_checksummed_non_contiguous_group_identity(tmp_path) -> None:
    path = tmp_path / "canonical.wal"
    wal = CanonicalCommitWAL(path)
    wal.append_group(({"transaction_id": "first"},))
    first_group_bytes = path.stat().st_size
    wal.append_group(({"transaction_id": "second"},))

    payload = bytearray(path.read_bytes())
    header = list(_HEADER.unpack_from(payload, first_group_bytes))
    header[2] = 9
    core = struct.pack(">8sIQQIQ", *header[:6])
    header[6] = hashlib.sha256(core).digest()
    payload[first_group_bytes : first_group_bytes + _HEADER.size] = _HEADER.pack(*header)
    footer_offset = len(payload) - _FOOTER.size
    footer = list(_FOOTER.unpack_from(payload, footer_offset))
    footer[1] = 9
    footer[3] = hashlib.sha256(payload[first_group_bytes:footer_offset]).digest()
    payload[footer_offset:] = _FOOTER.pack(*footer)
    path.write_bytes(payload)

    recovery = wal.recover(truncate=True)
    assert tuple(frame.wal_tx_id for frame in recovery.frames) == ("first",)
    assert recovery.wal_durable_lsn == 1
    assert path.stat().st_size == first_group_bytes


def test_stalled_fsync_exposes_bounded_pending_state_and_backpressures(tmp_path) -> None:
    entered = Event()
    release = Event()
    calls = 0

    def slow_first_fsync(_descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(timeout=2.0)

    wal = CanonicalCommitWAL(
        tmp_path / "canonical.wal", max_pending_bytes=4096, fsync=slow_first_fsync
    )
    first = Thread(
        target=lambda: wal.append_group(({"transaction_id": "first"},)), daemon=True
    )
    second_done = Event()
    second = Thread(
        target=lambda: (
            wal.append_group(({"transaction_id": "second"},)), second_done.set()
        ),
        daemon=True,
    )
    first.start()
    assert entered.wait(timeout=1.0)
    assert 0 < wal.pending_bytes <= wal.max_pending_bytes
    assert wal.pending_age_seconds >= 0.0
    second.start()
    assert not second_done.wait(timeout=0.05)
    release.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)
    assert second_done.is_set()
    assert wal.pending_bytes == 0
    assert wal.pending_age_seconds == 0.0
