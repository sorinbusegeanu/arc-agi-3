# v5.0 Milestones A+B Implementation Note

## Mined v4_5 files

- `src/v4_5/bootstrap/bootstrapSequenceBuilder.py`
- `src/v4_5/config/bootstrapConfig.py`
- `src/v4_5/bootstrap/bootstrapCapture.py`
- `src/v4_5/bootstrap/pngExporter.py`
- `src/v4_5/bootstrap/runtimeFactory.py`
- `src/v4_5/perception/board_builder/avatarExtractor.py`
- `src/v4_5/agents/avatarDetector.py`
- `src/v4_5/runtime/liveGameRunner.py` (bootstrap step execution pattern)
- `src/v4_5/cli/outputPaths.py`

## Reused logic (rewritten in v5_0)

- Deterministic fixed probe sequence planning.
- Per-step changed-cell extraction and connected-component splitting.
- Direction/magnitude/shape style scoring signals for movement evidence.
- Short-track linking by spatial continuity and value/shape stability.
- Ranked-candidate output with explicit ambiguity/insufficient-support handling.
- Artifact path convention and JSON artifact writing patterns.

## Intentionally not reused

- Any POI/HUD/traversability/mechanic-typing logic.
- Discovery/orchestrator/planner/hypothesis agent wiring.
- `v4_5` state assumptions on symbolic value ids.
- Plugin systems and broad runtime stage orchestration.
- Any non-deterministic fallback strategy.

## Scope status

- Implemented: Milestone A (pure transition-record -> ranked candidates).
- Implemented: Milestone B (`ez01` bootstrap wiring + artifacts).
- Implemented: Milestone C (`ez01..ez04` runtime support).
- Implemented: Milestone D (artifact writing and deterministic tests).
