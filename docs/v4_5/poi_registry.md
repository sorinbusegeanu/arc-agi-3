# v4.5 POI Registry

## Purpose

Track points of interest as first-class entities across rounds.

## Owners

- Discovery Agent creates and updates
- Outcome Agent updates lifecycle after execution
- Planner consumes

## Entities

- `POIRecord`
- `POIRegistry`
- `POIUpdate`

## Statuses

- `candidate`
- `reachable_candidate`
- `active_target`
- `reached`
- `tested_no_effect`
- `tested_effect`
- `blocked`
- `invalidated`
- `deferred`
- `closed`

## Required Fields

- `poi_id`
- `game_id`
- `level_index`
- `type_hint`
- `position_or_region`
- `source`
- `confidence`
- `status`
- `times_targeted`
- `times_reached`
- `last_effect_type`
- `linked_hypotheses`
- `rationale_codes`
