from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from v9.memory.identity import EpisodeId, EnvironmentInstanceId, stable_u64

from .contract import EnvironmentCognitionAdapter
from .schemas import EnvironmentIdentity


Factory = Callable[..., EnvironmentCognitionAdapter]


class EnvironmentRegistry:
    SCHEMA_VERSION = 1

    def __init__(self) -> None:
        self._identities: dict[int, EnvironmentIdentity] = {}
        self._episodes: dict[int, int] = {}
        self._factories: dict[str, Factory] = {}

    def register_factory(self, family: str, factory: Factory) -> None:
        if not family or family in self._factories:
            raise ValueError(f"duplicate or empty environment family {family!r}")
        self._factories[family] = factory

    def create(self, family: str, **kwargs: object) -> EnvironmentCognitionAdapter:
        try:
            adapter = self._factories[family](**kwargs)
        except KeyError as exc:
            raise KeyError(f"unknown environment family {family!r}") from exc
        self.register(adapter.identity())
        return adapter

    def register(self, identity: EnvironmentIdentity) -> EnvironmentInstanceId:
        instance_id = identity.instance_id
        existing = self._identities.get(instance_id.value)
        if existing is not None and existing != identity:
            raise ValueError("environment instance identity collision")
        self._identities[instance_id.value] = identity
        self._episodes.setdefault(instance_id.value, 0)
        return instance_id

    def resolve(self, instance_id: EnvironmentInstanceId | int) -> EnvironmentIdentity:
        key = instance_id.value if isinstance(instance_id, EnvironmentInstanceId) else int(instance_id)
        if key not in self._identities:
            raise KeyError(f"unknown environment instance id {key}")
        return self._identities[key]

    def next_episode(self, instance_id: EnvironmentInstanceId | int) -> EpisodeId:
        key = instance_id.value if isinstance(instance_id, EnvironmentInstanceId) else int(instance_id)
        self.resolve(key)
        ordinal = self._episodes[key] + 1
        self._episodes[key] = ordinal
        return EpisodeId(stable_u64(key, ordinal, person=b"v9-episode"))

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "identities": [
                {"family": row.family, "environment_type": row.environment_type,
                 "config": row.config, "instance": row.instance,
                 "instance_id": key, "episode_counter": self._episodes[key]}
                for key, row in sorted(self._identities.items())
            ],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "EnvironmentRegistry":
        if int(state.get("schema_version", 0)) != cls.SCHEMA_VERSION:
            raise ValueError("unsupported environment registry schema")
        result = cls()
        for raw in state.get("identities", []):
            if not isinstance(raw, dict):
                raise ValueError("invalid environment registry row")
            identity = EnvironmentIdentity(str(raw["family"]), str(raw["environment_type"]), str(raw["config"]), str(raw["instance"]))
            instance_id = result.register(identity)
            if instance_id.value != int(raw["instance_id"]):
                raise ValueError("environment identity does not reproduce persisted ID")
            result._episodes[instance_id.value] = int(raw.get("episode_counter", 0))
        return result

