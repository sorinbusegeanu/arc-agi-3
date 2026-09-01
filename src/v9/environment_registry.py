from __future__ import annotations

from dataclasses import dataclass

from v8.environments.schemas import EnvironmentIdentity
from v8.model import stable_u64


@dataclass(frozen=True, slots=True, order=True)
class EpisodeId:
    value: int


class EnvironmentIdentityRegistry:
    VERSION = 1

    def __init__(self) -> None:
        self._identities: dict[int, EnvironmentIdentity] = {}
        self._episode_counters: dict[int, int] = {}

    def register(self, identity: EnvironmentIdentity) -> int:
        instance_id = int(identity.instance_id)
        existing = self._identities.get(instance_id)
        if existing is not None and existing != identity:
            raise ValueError("environment instance identity collision")
        self._identities[instance_id] = identity
        self._episode_counters.setdefault(instance_id, 0)
        return instance_id

    def resolve(self, instance_id: int) -> EnvironmentIdentity:
        key = int(instance_id)
        if key not in self._identities:
            raise KeyError(f"unknown environment instance id {key}")
        return self._identities[key]

    def resolve_source_hash(self, source_game_hash: int) -> EnvironmentIdentity:
        return self.resolve(int(source_game_hash))

    def next_episode(self, instance_id: int) -> EpisodeId:
        key = int(instance_id)
        self.resolve(key)
        ordinal = self._episode_counters.get(key, 0) + 1
        self._episode_counters[key] = ordinal
        return EpisodeId(stable_u64(key, ordinal, person=b"v9-episode-id"))

    def state_dict(self) -> dict[str, object]:
        identities = []
        for instance_id in sorted(self._identities):
            identity = self._identities[instance_id]
            identities.append(
                {
                    "family": identity.family,
                    "environment_type": identity.environment_type,
                    "config": identity.config,
                    "instance": identity.instance,
                    "family_id": int(identity.family_id),
                    "type_id": int(identity.type_id),
                    "config_id": int(identity.config_id),
                    "instance_id": int(identity.instance_id),
                    "episode_counter": int(self._episode_counters.get(instance_id, 0)),
                }
            )
        return {"version": self.VERSION, "identities": identities}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "EnvironmentIdentityRegistry":
        if int(state.get("version", 0)) != cls.VERSION:
            raise ValueError("unsupported environment registry state version")
        registry = cls()
        rows = state.get("identities", [])
        if not isinstance(rows, list):
            raise ValueError("environment registry identities must be a list")
        for raw in rows:
            if not isinstance(raw, dict):
                raise ValueError("invalid environment registry row")
            identity = EnvironmentIdentity(
                family=str(raw["family"]),
                environment_type=str(raw["environment_type"]),
                config=str(raw["config"]),
                instance=str(raw["instance"]),
            )
            instance_id = registry.register(identity)
            for name, actual in (
                ("family_id", identity.family_id),
                ("type_id", identity.type_id),
                ("config_id", identity.config_id),
                ("instance_id", identity.instance_id),
            ):
                if int(raw[name]) != int(actual):
                    raise ValueError(f"environment registry {name} mismatch")
            registry._episode_counters[instance_id] = int(raw.get("episode_counter", 0))
        return registry
