from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from v8.environment_contract import BoundaryEvent
from v8.environments.schemas import ActionSchema, EnvironmentIdentity, ObservationSchema
from v8.model import stable_u64
from v9.memory import PayloadAvailabilityState, PayloadUid
from v9.modalities.symbols import DeterministicSymbolCodec, SymbolObservation
from v9.residency import PayloadProvenance, PayloadStore


@dataclass(frozen=True, slots=True)
class AlfredObservation:
    world_signature: int
    payload_uid: PayloadUid | None
    instruction_bytes: bytes


class AlfredAdapter:
    """Backend-neutral ALFRED boundary with raw instruction symbols only."""

    def __init__(
        self,
        backend,
        *,
        payload_store: PayloadStore | None = None,
        vocabulary: str = "alfred-bytes",
    ) -> None:
        self.backend = backend
        self.payload_store = payload_store
        self.codec = DeterministicSymbolCodec(vocabulary)
        self.identity = EnvironmentIdentity(
            "alfred",
            str(getattr(backend, "environment_name", "ALFRED")),
            "raw-symbols",
            "default",
        )
        self.observation_schema = ObservationSchema(
            "alfred-world", "backend observation signature + external payload"
        )
        self.action_schema = ActionSchema(
            "alfred-target-local", "backend available actions"
        )
        self._last: AlfredObservation | None = None
        self._boundary = BoundaryEvent()
        self._m0_counter = 0

    def _capture(self, world: Any, instruction: str | bytes) -> AlfredObservation:
        raw_instruction = (
            instruction.encode("utf-8") if isinstance(instruction, str) else bytes(instruction)
        )
        payload = (
            bytes(world)
            if isinstance(world, (bytes, bytearray, memoryview))
            else repr(world).encode("utf-8")
        )
        digest = stable_u64(payload, person=b"v9-payload-digest")
        payload_uid = PayloadUid(digest)
        if self.payload_store is not None:
            self._m0_counter += 1
            provenance = PayloadProvenance(
                self._m0_counter,
                self.identity.source_hash,
                0,
                self._m0_counter,
                0,
                payload_uid,
                digest,
                PayloadAvailabilityState.EXTERNAL,
            )
            self.payload_store.register(provenance, payload)
        observation = AlfredObservation(
            stable_u64(
                self.observation_schema.schema_id,
                digest,
                person=b"v9-alfred-world",
            ),
            payload_uid,
            raw_instruction,
        )
        self._last = observation
        return observation

    def reset(self) -> AlfredObservation:
        world, instruction = self.backend.reset()
        self._boundary = BoundaryEvent()
        return self._capture(world, instruction)

    def observe(self) -> AlfredObservation:
        if self._last is None:
            raise RuntimeError("ALFRED adapter must be reset before observation")
        return self._last

    def available_actions(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.backend.available_actions())

    def instruction_symbols(
        self, *, stream_name: str = "instruction"
    ) -> tuple[SymbolObservation, ...]:
        return self.codec.encode_stream(
            tuple(self.observe().instruction_bytes), stream_name=stream_name
        )

    def step(self, action: int) -> AlfredObservation:
        if int(action) not in set(self.available_actions()):
            raise ValueError("action is not available in target ALFRED environment")
        world, instruction, boundary = self.backend.step(int(action))
        if not isinstance(boundary, BoundaryEvent):
            raise ValueError("ALFRED backend must return BoundaryEvent")
        self._boundary = boundary
        return self._capture(world, instruction)

    def cognitive_boundary_event(self) -> BoundaryEvent:
        return self._boundary
