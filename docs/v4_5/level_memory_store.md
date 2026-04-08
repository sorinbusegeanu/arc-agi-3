Level memory store design

- Storage backend is SQLite.
- The key is exactly:
  - `game_id`
  - `level_id`
- The store keeps level priors only. It does not store authoritative runtime state.
- Stored objects are:
  - avatar
  - HUD regions
  - life regions
  - progress regions
  - POIs
  - exit regions
- Each stored region contains:
  - `bbox`
  - `center`
  - `colors`
  - `description` optional
  - `hint` optional
- Retrieval happens at level start.
- Write happens after Discovery completes for that level.
- Retrieved memory is advisory input, not authority.

Validation-state design

- Memory records have two states:
  - `hypothesis`
  - `validated`
- Outputs saved after Discovery are `hypothesis`.
- Outputs become `validated` only if they contributed to solving the level.
- `validated` memory is stronger than `hypothesis` memory.
- Retrieval returns both, but runtime must distinguish them.
