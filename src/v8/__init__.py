"""ARC-AGI-3 v8.2 RAM-authoritative developmental memory runtime."""

# runtime.py intentionally remains the v8.1 RAM/concurrency authority.  During its
# import, bind its stage topology to the v8.2 raw M0/M1 ingress and its peer symbol
# to the v8.2 scientific-semantics supervisor.  Restore development.STAGES
# immediately afterward so direct derivation helpers remain available for migration
# and unit-boundary compatibility without participating in live raw-event flow.
from v8 import development as _development
from v8 import peers as _peers
from v8.model import EventId, ExperienceEvent, MemoryLevel, MemoryType, MemoryUid
from v8.peers_v82 import V82DevelopmentalPeerSupervisor

_full_stages = _development.STAGES
_development.STAGES = _development.RAW_STAGES
_peers.DevelopmentalPeerSupervisor = V82DevelopmentalPeerSupervisor
try:
    from v8.runtime import ContinuousMemoryRuntime, V8RuntimeConfig
finally:
    _development.STAGES = _full_stages

__all__ = [
    "ContinuousMemoryRuntime",
    "EventId",
    "ExperienceEvent",
    "MemoryLevel",
    "MemoryType",
    "MemoryUid",
    "V8RuntimeConfig",
]
