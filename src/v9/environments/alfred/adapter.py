from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from v9.environments.base import StructuralAdapter
from v9.environments.contract import BoundaryEvent, WithinActionFrame, WithinActionTrace
from v9.environments.schemas import ActionSchema, EnvironmentIdentity, ObservationSchema
from v9.memory.identity import MemoryUid, stable_u64
from v9.memory.residency import PayloadStore
from v9.modalities.symbols import DeterministicSymbolCodec, SymbolObservation


class AlfredBackend(Protocol):
    def reset(self) -> tuple[Any, str | bytes]: ...
    def available_actions(self) -> tuple[int, ...]: ...
    def step(self, action: int) -> tuple[Any, str | bytes, BoundaryEvent]: ...


@dataclass(frozen=True, slots=True)
class AlfredObservation:
    world_signature: int
    payload_uid: int
    instruction_bytes: bytes


class AlfredAdapter(StructuralAdapter):
    def __init__(self, backend: AlfredBackend, *, payload_store: PayloadStore | None = None, vocabulary: str = "alfred-bytes") -> None:
        self.backend, self.payload_store = backend, payload_store or PayloadStore()
        self.codec = DeterministicSymbolCodec(vocabulary)
        self._identity = EnvironmentIdentity("alfred", str(getattr(backend, "environment_name", "ALFRED")), "raw-symbols", "default")
        self._observation_schema = ObservationSchema("alfred-world", "external-payload")
        self._action_schema = ActionSchema("environment-local", "backend-actions")
        self._boundary = BoundaryEvent()
        self._last_trace = None
        self._last: AlfredObservation | None = None
        self._counter = 0

    def _capture(self, world: Any, instruction: str | bytes) -> AlfredObservation:
        payload = bytes(world) if isinstance(world, (bytes, bytearray, memoryview)) else repr(world).encode("utf-8")
        self._counter += 1
        source = MemoryUid.derive("alfred-observation", self._identity.instance_id.value, self._counter)
        stored = self.payload_store.put(payload, source)
        raw_instruction = instruction.encode("utf-8") if isinstance(instruction, str) else bytes(instruction)
        self._last = AlfredObservation(stable_u64(self._observation_schema.schema_id, stored.digest, person=b"v9-alfred-world"), stored.payload_uid, raw_instruction)
        return self._last

    def reset(self) -> AlfredObservation:
        world, instruction = self.backend.reset()
        self._boundary = BoundaryEvent()
        self._last_trace = None
        return self._capture(world, instruction)

    def observe(self) -> AlfredObservation:
        if self._last is None:
            raise RuntimeError("ALFRED adapter must be reset before observation")
        return self._last

    def available_actions(self) -> tuple[int, ...]:
        return () if not self._boundary.continuation else tuple(int(value) for value in self.backend.available_actions())

    def optional_symbol_stream(self) -> tuple[object, ...]:
        return tuple(self.observe().instruction_bytes)

    def instruction_symbols(self, stream_name: str = "instruction") -> tuple[SymbolObservation, ...]:
        return self.codec.encode_stream(self.optional_symbol_stream(), stream_name=stream_name)

    def step(self, native_action: Any) -> AlfredObservation:
        before = self.observe()
        action = int(native_action)
        if action not in self.available_actions():
            raise ValueError("ALFRED action is unavailable")
        world, instruction, boundary = self.backend.step(action)
        if not isinstance(boundary, BoundaryEvent):
            raise ValueError("ALFRED backend must return a v9 BoundaryEvent")
        self._boundary = boundary
        after = self._capture(world, instruction)
        self._last_trace = WithinActionTrace(before, (WithinActionFrame(after, 0),), after)
        return after

