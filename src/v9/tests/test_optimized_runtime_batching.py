from __future__ import annotations

from v9.memory import CanonicalNode, MemoryLevel, MemoryType
from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig


def test_derivation_publication_batches_respect_read_set_bound(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False, enable_snapshots=False))
    rows = []
    for index in range(80):
        node = CanonicalNode.build(MemoryLevel.M2, MemoryType.FAMILY, (index,), index + 1)
        parents = [[index * 100 + offset, index * 100 + offset + 1] for offset in range(8)]
        rows.append((node, {"parents": parents}, ()))

    published = []
    runtime._publish_group = lambda batch: published.append(tuple(batch))
    runtime._publish_derivation_rows(rows)

    assert len(published) > 1
    maximum = runtime.config.scientific.maximum_read_set_size
    for batch in published:
        dependency_count = sum(runtime._publication_row_dependency_cost(row) for row in batch)
        assert dependency_count <= maximum
