"""ARC-AGI-3 v8.15 memory-to-intelligence runtime."""

# Install the v0.5.3 memory schema before importing development/runtime modules so
# proposal packets, shared arenas and snapshot compatibility agree on one layout.
from v8.primary_valence import (
    install_primary_valence_runtime as _install_primary_valence_runtime,
    install_primary_valence_schema as _install_primary_valence_schema,
)

_install_primary_valence_schema()
from v8.primary_valence_fixups import install_schema_fixups as _install_schema_fixups
_install_schema_fixups()

# runtime.py remains the v8.1 RAM/concurrency authority. Bind the v8.2 semantic
# extensions around it, then make the raw runtime topology explicit: ExperienceEvent
# traffic has exactly two raw stages (M0/M1). M2-M7 are peer/evidence formed.
from v8 import development as _development
from v8 import evaluation as _evaluation
from v8 import peers as _peers
from v8.evaluation_v82 import V82ScientificHypothesisEvaluator
from v8.model import EventId, ExperienceEvent, MemoryLevel, MemoryType, MemoryUid
from v8.peers_v82 import V82DevelopmentalPeerSupervisor

_peers.DevelopmentalPeerSupervisor = V82DevelopmentalPeerSupervisor
_evaluation.ScientificHypothesisEvaluator = V82ScientificHypothesisEvaluator
import v8.runtime as _runtime
_runtime.STAGES = _development.RAW_STAGES

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
from v8.normalized_memory_v086 import (
    install_normalized_memory_v086 as _install_normalized_memory_v086,
)
from v8.normalized_memory_v086_fixups import (
    install_normalized_memory_v086_fixups as _install_normalized_memory_v086_fixups,
)
from v8.intelligence_loop_v087 import (
    install_intelligence_loop_v087 as _install_intelligence_loop_v087,
)
from v8.intelligence_loop_v087_fixups import (
    install_intelligence_loop_v087_fixups as _install_intelligence_loop_v087_fixups,
)
from v8.learning_fixes_v088 import (
    install_learning_fixes_v088 as _install_learning_fixes_v088,
)
from v8.learning_fixes_v088_fixups import (
    install_learning_fixes_v088_fixups as _install_learning_fixes_v088_fixups,
)
from v8.shutdown_semantics_v089 import (
    install_shutdown_semantics_v089 as _install_shutdown_semantics_v089,
)
from v8.action_targeting_v810 import (
    install_action_targeting_v810 as _install_action_targeting_v810,
)
from v8.action_targeting_v810_fixups import (
    install_action_targeting_v810_fixups as _install_action_targeting_v810_fixups,
)
from v8.final_save_lifecycle_v812 import (
    install_final_save_lifecycle_v812 as _install_final_save_lifecycle_v812,
)
from v8.final_save_lifecycle_v812_fixups import (
    install_final_save_lifecycle_v812_fixups as _install_final_save_lifecycle_v812_fixups,
)
from v8.lifecycle_progress_v812 import (
    install_lifecycle_progress_v812 as _install_lifecycle_progress_v812,
)
from v8.dedicated_lifecycle_v813 import (
    install_dedicated_lifecycle_v813 as _install_dedicated_lifecycle_v813,
)
from v8.trajectory_optimizer_v814 import (
    install_trajectory_optimizer_v814 as _install_trajectory_optimizer_v814,
)
from v8.trajectory_optimizer_v814_fixups import (
    install_trajectory_optimizer_v814_fixups as _install_trajectory_optimizer_v814_fixups,
)
from v8.restart_memory_v815 import (
    install_restart_memory_v815 as _install_restart_memory_v815,
)
from v8.restart_memory_v815_fixups import (
    install_restart_memory_v815_fixups as _install_restart_memory_v815_fixups,
)

# Install semantic layers in chronological order. v8.15 is last so restart-memory
# fallback, session retention and trajectory phase reuse see the final policy stack.
_install_behavior_recovery()
_install_primary_valence_runtime()
_install_runtime_fixups()
_install_trajectory_efficiency_v054()
_install_progress_reporting_v054()
_install_hypothesis_validation_v054()
_install_learning_blockers_v055()
_install_learning_blockers_v055_fixups()
_install_normalized_memory_v086()
_install_normalized_memory_v086_fixups()
_install_intelligence_loop_v087()
_install_intelligence_loop_v087_fixups()
_install_learning_fixes_v088()
_install_learning_fixes_v088_fixups()
_install_shutdown_semantics_v089()
_install_action_targeting_v810()
_install_action_targeting_v810_fixups()
_install_final_save_lifecycle_v812()
_install_final_save_lifecycle_v812_fixups()
_install_lifecycle_progress_v812()
_install_dedicated_lifecycle_v813()
_install_trajectory_optimizer_v814()
_install_trajectory_optimizer_v814_fixups()
_install_restart_memory_v815()
_install_restart_memory_v815_fixups()

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
