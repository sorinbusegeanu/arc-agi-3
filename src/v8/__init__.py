"""ARC-AGI-3 v8.5 learning-capability runtime."""

# Install the v0.5.3 memory schema before importing development/runtime modules so
# proposal packets, shared arenas and snapshot compatibility agree on one layout.
from v8.primary_valence import (
    install_primary_valence_runtime as _install_primary_valence_runtime,
    install_primary_valence_schema as _install_primary_valence_schema,
)

_install_primary_valence_schema()
from v8.primary_valence_fixups import install_schema_fixups as _install_schema_fixups
_install_schema_fixups()

# runtime.py remains the v8.1 RAM/concurrency authority. During its import, bind
# only the scientific-semantic extension points: raw stage topology, peer supervisor,
# and evaluator. Direct raw events still instantiate only M0/M1; M2-M7 emerge from
# accumulated memory state.
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
from v8.primary_valence_fixups import install_runtime_fixups as _install_runtime_fixups
from v8.trajectory_efficiency_v054 import (
    install_trajectory_efficiency_v054 as _install_trajectory_efficiency_v054,
)
from v8.progress_reporting_v054 import (
    install_progress_reporting_v054 as _install_progress_reporting_v054,
)
from v8.hypothesis_validation_v054 import (
    install_hypothesis_validation_v054 as _install_hypothesis_validation_v054,
)
from v8.learning_blockers_v055 import (
    install_learning_blockers_v055 as _install_learning_blockers_v055,
)
from v8.learning_blockers_v055_fixups import (
    install_learning_blockers_v055_fixups as _install_learning_blockers_v055_fixups,
)

# Install semantic layers in chronological order. v8.5 is last because it tightens
# control-state/action representation, multi-action planning, causal validation and
# exploration semantics established by the preceding compatibility layers.
_install_behavior_recovery()
_install_primary_valence_runtime()
_install_runtime_fixups()
_install_trajectory_efficiency_v054()
_install_progress_reporting_v054()
_install_hypothesis_validation_v054()
_install_learning_blockers_v055()
_install_learning_blockers_v055_fixups()

# Preserve existing import paths: `from v8.runtime import ContinuousMemoryRuntime`
# and the package-level API both resolve to the current semantic layer.
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
