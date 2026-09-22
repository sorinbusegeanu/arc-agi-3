from __future__ import annotations

import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

CHUNK_BYTES = 4 * 1024 * 1024


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _persist_chunk(directory: Path, chunk: bytes) -> dict[str, Any]:
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
    return {"sha256": digest, "bytes": len(chunk)}


def write_stream_chunks(
    root: Path, blocks: Any
) -> tuple[list[dict[str, Any]], int, str]:
    """Persist a byte stream in bounded chunks without materializing the stream."""
    directory = root / "snapshot_chunks"
    directory.mkdir(parents=True, exist_ok=True)
    result: list[dict[str, Any]] = []
    pending = bytearray()
    full_digest = hashlib.sha256()
    total = 0

    def flush_pending() -> None:
        nonlocal total
        if not pending:
            return
        chunk = bytes(pending)
        pending.clear()
        full_digest.update(chunk)
        total += len(chunk)
        result.append(_persist_chunk(directory, chunk))

    for block in blocks:
        if not block:
            continue
        view = memoryview(block)
        offset = 0
        while offset < len(view):
            take = min(CHUNK_BYTES - len(pending), len(view) - offset)
            pending.extend(view[offset : offset + take])
            offset += take
            if len(pending) == CHUNK_BYTES:
                flush_pending()
    flush_pending()
    return result, total, full_digest.hexdigest()


def iter_file_blocks(path: Path, *, block_bytes: int = 1024 * 1024):
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(max(1, int(block_bytes)))
            if not block:
                break
            yield block


def write_file_chunks(root: Path, path: Path) -> tuple[list[dict[str, Any]], int, str]:
    return write_stream_chunks(root, iter_file_blocks(path))


def write_chunks(root: Path, payload: bytes) -> list[dict[str, Any]]:
    chunks, _total, _digest = write_stream_chunks(root, (payload,))
    return chunks


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
