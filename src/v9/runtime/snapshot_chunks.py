from __future__ import annotations

import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

CHUNK_BYTES = 4 * 1024 * 1024


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_chunks(root: Path, payload: bytes) -> list[dict[str, Any]]:
    directory = root / "snapshot_chunks"
    directory.mkdir(parents=True, exist_ok=True)
    result: list[dict[str, Any]] = []
    for offset in range(0, len(payload), CHUNK_BYTES):
        chunk = payload[offset : offset + CHUNK_BYTES]
        digest = sha256(chunk)
        target = directory / f"{digest}.bin"
        if not target.exists():
            with NamedTemporaryFile(
                "wb",
                dir=directory,
                prefix=f".{digest}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink(missing_ok=True)
        result.append({"sha256": digest, "bytes": len(chunk)})
    return result


def read_chunks(root: Path, chunks: list[dict[str, Any]]) -> bytes:
    payload = bytearray()
    for spec in chunks:
        digest = str(spec["sha256"])
        expected_bytes = int(spec["bytes"])
        chunk = (root / "snapshot_chunks" / f"{digest}.bin").read_bytes()
        if len(chunk) != expected_bytes or sha256(chunk) != digest:
            raise RuntimeError(f"snapshot chunk checksum mismatch {digest}")
        payload.extend(chunk)
    return bytes(payload)
