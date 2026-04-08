from __future__ import annotations

from dataclasses import replace

from v4_5.contracts import POIRecord, POIRegistry, POIUpdate, SCHEMA_VERSION
from v4_5.contracts.constants import (
    POI_STATUS_ACTIVE_TARGET,
    POI_STATUS_CANDIDATE,
    POI_STATUS_CLOSED,
    POI_STATUS_INVALIDATED,
)


class POIRegistryStore:
    agent_name = "POIRegistryStore"

    def create(self, *, round_id: str) -> POIRegistry:
        return POIRegistry(schema_version=SCHEMA_VERSION, agent_name=self.agent_name, round_id=round_id, records=())

    def insert_candidate_poi(self, registry: POIRegistry, record: POIRecord) -> POIRegistry:
        if any(item.poi_id == record.poi_id for item in registry.records):
            return registry
        return replace(registry, records=tuple(registry.records) + (record,))

    def update_status(self, registry: POIRegistry, *, poi_id: str, status: str, agent_name: str, round_id: str) -> tuple[POIRegistry, POIUpdate]:
        records = []
        target = None
        for item in registry.records:
            if item.poi_id == poi_id:
                item = replace(item, status=status, round_id=round_id, agent_name=agent_name)
                target = item
            records.append(item)
        update = POIUpdate(
            schema_version=SCHEMA_VERSION,
            agent_name=agent_name,
            round_id=round_id,
            poi_id=poi_id,
            status=status,
            last_effect_type=(target.last_effect_type if target else ""),
            rationale_codes=("STATUS_UPDATE",),
        )
        return replace(registry, records=tuple(records), round_id=round_id, agent_name=agent_name), update

    def increment_target_counter(self, registry: POIRegistry, *, poi_id: str, agent_name: str, round_id: str) -> POIRegistry:
        return self._mutate_counter(registry, poi_id=poi_id, field_name="times_targeted", agent_name=agent_name, round_id=round_id)

    def increment_reach_counter(self, registry: POIRegistry, *, poi_id: str, agent_name: str, round_id: str) -> POIRegistry:
        return self._mutate_counter(registry, poi_id=poi_id, field_name="times_reached", agent_name=agent_name, round_id=round_id)

    def mark_effect_type(self, registry: POIRegistry, *, poi_id: str, effect_type: str, agent_name: str, round_id: str) -> POIRegistry:
        records = []
        for item in registry.records:
            if item.poi_id == poi_id:
                item = replace(item, last_effect_type=effect_type, agent_name=agent_name, round_id=round_id)
            records.append(item)
        return replace(registry, records=tuple(records), round_id=round_id, agent_name=agent_name)

    def fetch_active_open_pois(self, registry: POIRegistry) -> tuple[POIRecord, ...]:
        return tuple(
            item for item in registry.records if item.status not in {POI_STATUS_CLOSED, POI_STATUS_INVALIDATED}
        )

    def fetch_by_status(self, registry: POIRegistry, status: str) -> tuple[POIRecord, ...]:
        return tuple(item for item in registry.records if item.status == status)

    def invalidate(self, registry: POIRegistry, *, poi_id: str, agent_name: str, round_id: str) -> tuple[POIRegistry, POIUpdate]:
        return self.update_status(registry, poi_id=poi_id, status=POI_STATUS_INVALIDATED, agent_name=agent_name, round_id=round_id)

    def close(self, registry: POIRegistry, *, poi_id: str, agent_name: str, round_id: str) -> tuple[POIRegistry, POIUpdate]:
        return self.update_status(registry, poi_id=poi_id, status=POI_STATUS_CLOSED, agent_name=agent_name, round_id=round_id)

    def activate(self, registry: POIRegistry, *, poi_id: str, agent_name: str, round_id: str) -> tuple[POIRegistry, POIUpdate]:
        return self.update_status(registry, poi_id=poi_id, status=POI_STATUS_ACTIVE_TARGET, agent_name=agent_name, round_id=round_id)

    def _mutate_counter(self, registry: POIRegistry, *, poi_id: str, field_name: str, agent_name: str, round_id: str) -> POIRegistry:
        records = []
        for item in registry.records:
            if item.poi_id == poi_id:
                value = int(getattr(item, field_name)) + 1
                item = replace(item, **{field_name: value}, agent_name=agent_name, round_id=round_id)
            records.append(item)
        return replace(registry, records=tuple(records), round_id=round_id, agent_name=agent_name)
