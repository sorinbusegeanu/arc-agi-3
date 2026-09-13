from .config import RuntimeConfig, ScientificConfig, ScientificConfigId
from .publication import CanonicalGraph
from .read_view import ReadView
from .optimized_runtime import ContinuousMemoryRuntime as _OptimizedContinuousMemoryRuntime


class ContinuousMemoryRuntime(_OptimizedContinuousMemoryRuntime):
    """Public v9 runtime without an artificial canonical graph record ceiling."""

    def __init__(self, config):
        super().__init__(config)
        # v9 uses dynamic Python graph storage. The former arena-style capacities
        # were inherited from an earlier fixed-storage design and became a hard
        # 1,000,000-node rejection ceiling at the default four shards. They are
        # not a scientific constraint and are intentionally disabled, including
        # for restored snapshots that persisted the old limits.
        self.graph.node_capacity_per_partition = None
        self.graph.edge_capacity_per_partition = None


__all__ = ["CanonicalGraph", "ContinuousMemoryRuntime", "ReadView", "RuntimeConfig", "ScientificConfig", "ScientificConfigId"]
