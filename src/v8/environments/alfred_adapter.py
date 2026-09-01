from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from v8.environment_contract import BoundaryEvent
from v8.environments.schemas import ActionSchema, EnvironmentIdentity, ObservationSchema
from v8.model import stable_u64
from v8.modalities.symbols import DeterministicSymbolCodec, SymbolObservation
from v8.residency import M0ProvenanceRecord, PayloadAvailabilityState, PayloadResidencyStore, PayloadUid


class AlfredBackend(Protocol):
    environment_name: str
    def reset(self) -> tuple[Any, str | bytes]: ...
    def available_actions(self) -> tuple[int, ...] | list[int]: ...
    def step(self, action: int) -> tuple[Any, str | bytes, BoundaryEvent]: ...
    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class AlfredObservation:
    world: Any
    world_signature: int
    instruction_bytes: bytes
    payload_uid: PayloadUid


class AlfredAdapter:
    """Backend-neutral ALFRED boundary; no pretrained parser/embedding path."""

    def __init__(self, backend: AlfredBackend, *, payload_store: PayloadResidencyStore | None = None, vocabulary: str = "alfred-bytes") -> None:
        self.backend = backend
        self.payload_store = payload_store
        self.identity = EnvironmentIdentity("alfred", str(getattr(backend, "environment_name", "ALFRED")), "raw-symbols", "default")
        self.observation_schema = ObservationSchema("alfred-world", "opaque backend world payload")
        self.action_schema = ActionSchema("alfred-target-local", "backend available actions")
        self.codec = DeterministicSymbolCodec(vocabulary)
        self._last: AlfredObservation | None = None
        self._boundary = BoundaryEvent()
        self._capture_ordinal = 0

    def _capture(self, world: Any, instruction: str | bytes) -> AlfredObservation:
        raw_instruction = instruction.encode("utf-8") if isinstance(instruction, str) else bytes(instruction)
        payload = bytes(world) if isinstance(world, (bytes, bytearray, memoryview)) else repr(world).encode("utf-8")
        digest = PayloadResidencyStore.digest(payload)
        payload_uid = PayloadUid(digest)
        signature = stable_u64(self.observation_schema.schema_id, digest, person=b"v9-alfred-world")
        row = AlfredObservation(world, signature, raw_instruction, payload_uid)
        self._capture_ordinal += 1
        if self.payload_store is not None:
            m0 = __import__("v8.model", fromlist=["MemoryUid"]).MemoryUid.from_key(0, 1, (stable_u64(self.identity.source_hash, self._capture_ordinal, person=b"v9-alfred-m0-hi"), stable_u64(self._capture_ordinal, person=b"v9-alfred-m0-lo")))
            self.payload_store.register(M0ProvenanceRecord(m0, self.identity.source_hash, 0, self._capture_ordinal, signature, payload_uid, digest, PayloadAvailabilityState.HOT, 1), payload)
        self._last = row
        return row

    def reset(self) -> AlfredObservation:
        world, instruction = self.backend.reset()
        self._boundary = BoundaryEvent()
        return self._capture(world, instruction)

    def observe(self) -> AlfredObservation:
        if self._last is None:
            raise RuntimeError("ALFRED adapter must be reset before observation")
        return self._last

    def available_actions(self) -> tuple[int, ...]:
        return tuple(sorted({int(v) for v in self.backend.available_actions()}))

    def instruction_symbols(self, *, stream_name: str = "instruction") -> tuple[SymbolObservation, ...]:
        return self.codec.encode_stream(tuple(self.observe().instruction_bytes), stream_name=stream_name)

    def observation_signature(self, observation: AlfredObservation | None = None) -> int:
        return int((self.observe() if observation is None else observation).world_signature)

    def step(self, action: int) -> AlfredObservation:
        action = int(action)
        if action not in set(self.available_actions()):
            raise ValueError("action is not available in target ALFRED environment")
        world, instruction, boundary = self.backend.step(action)
        if not isinstance(boundary, BoundaryEvent):
            raise ValueError("ALFRED backend must return BoundaryEvent")
        self._boundary = boundary
        return self._capture(world, instruction)

    def cognitive_boundary_event(self) -> BoundaryEvent:
        return self._boundary

    def cognitive_context_signature(self) -> int:
        return self.observation_signature()

    def cognitive_transition_signature(self, before: AlfredObservation, after: AlfredObservation) -> int:
        return stable_u64(before.world_signature, after.world_signature, person=b"v9-alfred-transition")

    def cognitive_family_signature(self, before: AlfredObservation, after: AlfredObservation) -> int:
        return stable_u64(self.identity.type_id, person=b"v9-alfred-family")

    def cognitive_changed_extent(self, before: AlfredObservation, after: AlfredObservation) -> int:
        return int(before.world_signature != after.world_signature)

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close): close()
