from pathlib import Path

from . import chunked_snapshot
from . import snapshot as legacy

SnapshotResult = chunked_snapshot.SnapshotResult
assert_native_root = legacy.assert_native_root


def latest_snapshot(root: Path):
    old = legacy.latest_snapshot(root)
    new = chunked_snapshot.latest_snapshot(root)
    if old is None:
        return new
    if new is None:
        return old
    old_id = int(old.stem.rsplit("-", 1)[1])
    new_id = int(new.name.rsplit("-", 1)[1])
    return new if new_id >= old_id else old


def load_snapshot(path: Path, *, expected_config_id: str):
    if path.is_file():
        return legacy.load_snapshot(path, expected_config_id=expected_config_id)
    return chunked_snapshot.load_snapshot(path, expected_config_id=expected_config_id)


write_snapshot = chunked_snapshot.write_snapshot


def load_snapshot_direct(path: Path, *, expected_config_id: str):
    if path.is_file():
        return None
    return chunked_snapshot.load_snapshot_parts(path, expected_config_id=expected_config_id)

decode_graph_shard = chunked_snapshot._decode_graph_shard

load_graph_shard = chunked_snapshot.load_graph_shard
