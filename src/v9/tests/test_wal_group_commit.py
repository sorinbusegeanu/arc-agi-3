from __future__ import annotations

import hashlib
import struct

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
