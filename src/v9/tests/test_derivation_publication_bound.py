from pathlib import Path

from v9 import ContinuousMemoryRuntime, RuntimeConfig
from v9.memory.identity import MemoryUid
from v9.memory.m2_family import M2TransformationFamily
from v9.memory.model import MemoryLevel, MemoryType
from v9.memory.provenance import DerivationProvenance
from v9.memory.relations import RelationType
from v9.runtime.memory_pipeline import DerivationResult


def test_oversized_derivation_provenance_is_published_in_bounded_chunks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(tmp_path / "runtime", restore=False, enable_snapshots=False)
    )
    maximum = int(runtime.config.scientific.maximum_read_set_size)
    parents = tuple(
        MemoryUid.from_key(MemoryLevel.M1, MemoryType.NORMALIZED_RELATION, (index,))
        for index in range(maximum * 2 + 17)
    )
    structural_signature = 0xD3A1_0001
    family_uid = MemoryUid.from_key(
        MemoryLevel.M2,
        MemoryType.FAMILY,
        (structural_signature,),
    )
    family = M2TransformationFamily(
        family_uid,
        structural_signature,
        DerivationProvenance(parents, parents[:2]),
        len(parents),
        1.0,
    )
    result = DerivationResult(
        1,
        structural_signature,
        len(parents),
        family,
        (),
        (),
        1,
    )

    observed_read_set_sizes: list[int] = []
    original_publish_batch = runtime.graph.publish_batch

    def checked_publish_batch(proposals):
        observed_read_set_sizes.extend(len(proposal.read_set.dependencies) for proposal in proposals)
        return original_publish_batch(proposals)

    monkeypatch.setattr(runtime.graph, "publish_batch", checked_publish_batch)

    runtime.apply_derivation_results_batch((result,))

    assert family_uid in runtime.graph.nodes
    assert {tuple(row) for row in runtime.graph.payloads[family_uid]["parents"]} == {
        (uid.hi, uid.lo) for uid in parents
    }
    assert {
        edge.target
        for edge in runtime.graph.edges.values()
        if edge.source == family_uid and edge.relation is RelationType.PROVENANCE
    } == set(parents)
    assert len(observed_read_set_sizes) >= 3
    assert max(observed_read_set_sizes) <= maximum
