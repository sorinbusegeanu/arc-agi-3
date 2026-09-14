from __future__ import annotations

from typing import Any

from v9.memory.identity import stable_u64

from .contract import BoundaryEvent, EnvironmentTransition, TaskProgress, WithinActionTrace
from .schemas import ActionSchema, EnvironmentIdentity, ObservationSchema


class StructuralAdapter:
    _identity: EnvironmentIdentity
    _observation_schema: ObservationSchema
    _action_schema: ActionSchema
    _boundary: BoundaryEvent
    _last_trace: WithinActionTrace | None

    def identity(self) -> EnvironmentIdentity:
        return self._identity

    def observation_schema(self) -> ObservationSchema:
        return self._observation_schema

    def action_schema(self) -> ActionSchema:
        return self._action_schema

    def boundary_event(self) -> BoundaryEvent:
        return self._boundary

    def task_progress(self) -> TaskProgress:
        boundary = self._boundary
        return TaskProgress(
            game_id=str(self._identity.environment_type),
            terminal=not bool(boundary.continuation),
            success=bool(boundary.primary_valence > 0 and not boundary.continuation),
            failure=bool(boundary.primary_valence < 0 and not boundary.continuation),
            truncated=bool(not boundary.continuation and boundary.primary_valence == 0),
            score=float(boundary.primary_valence),
        )

    def optional_micro_trace(self) -> WithinActionTrace | None:
        return self._last_trace

    def optional_symbol_stream(self) -> tuple[object, ...]:
        return ()

    def encode_observation(self, observation: Any) -> int:
        try:
            import numpy as np
            array = np.asarray(observation)
            raw = array.tobytes() + str(array.shape).encode("ascii")
        except Exception:
            raw = repr(observation).encode("utf-8")
        return stable_u64(self._observation_schema.schema_id, raw, person=b"v9-env-observe")

    def encode_action(self, action: Any) -> int:
        return int(action)

    def transition(self, before: Any, after: Any, action: Any) -> EnvironmentTransition:
        before_actions = tuple(self.available_actions())
        before_signature = self.encode_observation(before)
        after_signature = self.encode_observation(after)
        return EnvironmentTransition(
            before, after, action, self.encode_action(action), before_actions,
            tuple(self.available_actions()),
            {"before_signature": before_signature, "after_signature": after_signature},
            self._boundary, before_signature, after_signature, self._last_trace,
        )

