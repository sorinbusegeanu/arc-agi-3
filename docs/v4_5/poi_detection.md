POI detection for movement-only bootstrap

- Avatar handling is separate from POI detection.
- HUD, life, and progress region handling are separate from POI detection.
- Click handling is out of scope here.
- POI output contains only:
  - `bbox`
  - `center`
  - `colors`
- No shape field is included.
- No evidence tags are included.
- No code-generated natural-language hint is included.
- Deterministic POI analysis, LLM text POI analysis, and VLM video POI analysis run in parallel.
- Deterministic POI analysis is the primary authority in v1.
- Candidates are generated from persistent components and changed regions.
- Exclusions include avatar, HUD, life, and progress regions.
- Output remains `bbox`, `center`, and `colors` only.
- The default selected POI set is always the deterministic result.

Rejection behavior

- Border-only noise is rejected.
- Candidates overlapping excluded regions are rejected.
- Large background-like regions are rejected.
- An empty POI result is valid unless stricter caller logic rejects it.
