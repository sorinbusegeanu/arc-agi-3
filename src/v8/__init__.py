"""ARC-AGI-3 v8.2 RAM-authoritative developmental memory runtime."""

# runtime.py remains the v8.1 RAM/concurrency authority.  During its import, bind
# only the scientific-semantic extension points: raw stage topology, peer supervisor,
# and evaluator.  Direct derivation helpers retain the complete level table for
# migration/unit-boundary compatibility but live raw events stop at M1.
from v8 import development as _development
from v8 import evaluation as _evaluation
from v8 import peers as _peers
from v8.evaluation_v82 import V82ScientificHypothesisEvaluator
from v8.model import EventId, ExperienceEvent, MemoryLevel, MemoryType, MemoryUid
from v8.peers_v82 import V82DevelopmentalPeerSupervisor

_full_stages = _development.STAGES
_development.STAGES = _development.RAW_STAGES
_peers.DevelopmentalPeerSupervisor = V82DevelopmentalPeerSupervisor
_evaluation.ScientificHypothesisEvaluator = V82ScientificHypothesisEvaluator
try:
    import v8.runtime as _runtime
finally:
    _development.STAGES = _full_stages

from v8.runtime_v82 import V82ContinuousMemoryRuntime
from v8.behavior_recovery import install_behavior_recovery as _install_behavior_recovery

# Install the behavioral half of the v8.2 semantics after the RAM/concurrency
# authority has loaded.  This keeps the runtime architecture unchanged while
# enforcing causal strategy formation, empirical planner admission, canonical
# outcome recognition and actor exploration.
_install_behavior_recovery()

# Preserve existing import paths: `from v8.runtime import ContinuousMemoryRuntime`
# and the package-level API both resolve to the v8.2 semantic layer.
_runtime.ContinuousMemoryRuntime = V82ContinuousMemoryRuntime
ContinuousMemoryRuntime = V82ContinuousMemoryRuntime
V8RuntimeConfig = _runtime.V8RuntimeConfig

__all__ = [
    "ContinuousMemoryRuntime",
    "EventId",
    "ExperienceEvent",
    "MemoryLevel",
    "MemoryType",
    "MemoryUid",
    "V8RuntimeConfig",
]
