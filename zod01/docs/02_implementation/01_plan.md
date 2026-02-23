# zod01 ARC-AGI-3 Agent Implementation Plan

## Goal
Build a robust ARC-AGI-3 agent in `zod01` using a staged path that prioritizes deterministic behavior, fast debugging, and measurable progress.

## Scope
- Build a custom agent architecture for ARC-AGI-3.
- Use `arc-agi` for environment interaction.
- Follow `ARC-AGI-3-Agents` interface patterns for agent lifecycle.
- Deliver modules incrementally with per-stage validation metrics.

## Non-goals (initial phases)
- No end-to-end learned policy in early stages.
- No heavy model training before deterministic core is stable.
- No broad prompt engineering focus until planner/explorer baseline exists.

## Hard Constraints
- Respect dynamic `available_actions` every step.
- Support complex action payloads for `ACTION6` (`x`, `y` in 0..63).
- Deterministic replay with fixed seeds.
- Keep controller stable while introducing one new scoring/module term at a time.

## Architecture (v0)
1. Environment I/O Adapter
2. Observation Parser
3. State Abstraction + Canonicalization
4. Transition/Diff Engine
5. Episodic Memory
6. Safety Guard
7. Controller
8. Empirical World Model
9. Planner (BFS, then A*)
10. Goal Detector + Progress Heuristics
11. Explorer
12. Logger/Dataset Builder
13. Evaluator Runner

## Stage Plan

### Stage 0 - Runner skeleton
Deliverables:
- Agent entrypoint in `zod01` with run loop.
- Environment adapter wrapping `Arcade.make`, `reset`, `step`, `action_space`, `observation_space`.
- Structured JSONL logging per episode.

Acceptance criteria:
- Agent can run one game end-to-end without crashing.
- Same seed gives same action trace in deterministic mode.
- Invalid action emissions are blocked before `step`.

Primary metric:
- Replay determinism rate.

### Stage 1 - Deterministic state core
Deliverables:
- Observation parser for frame + metadata.
- Canonical state object and stable `state_hash`.
- Transition diff classifier (`no-op`, change types, simple reversibility tags).

Acceptance criteria:
- Same observation always maps to same canonical serialization/hash.
- Transition logs are human-readable and compact.

Primary metric:
- Replay determinism rate + hash stability checks.

### Stage 2 - Episodic graph baseline
Deliverables:
- Visited-state index keyed by `state_hash`.
- Transition map `(state_hash, action) -> next_state_hash + delta + stats`.
- Loop detector and cycle breaker.
- Safety guard (max no-op streak, max loop streak, fallback behavior).

Acceptance criteria:
- Agent explores without getting permanently stuck in short cycles.
- Graph grows over steps in most levels.

Primary metric:
- Unique states discovered per 1k steps.

### Stage 3 - Empirical world model
Deliverables:
- Known-edge lookup model from episodic memory.
- Unknown-edge marking for exploration pressure.

Acceptance criteria:
- If a transition is known, controller reuses it deterministically.
- Unknown edges are explicitly tracked.

Primary metric:
- Known-edge exploitation rate.

### Stage 4 - Planner-first policy
Deliverables:
- BFS planner over discovered graph.
- Replan trigger on mismatch between predicted and observed transition.

Acceptance criteria:
- When goal state exists in graph, planner reaches it reliably.

Primary metric:
- Success rate conditioned on goal-known episodes.

### Stage 5 - Goal/progress shaping
Deliverables:
- Terminal detector from environment state.
- Progress heuristic to rank partial progress.
- A* option using heuristic score.

Acceptance criteria:
- Better late-episode exploitation versus BFS-only baseline.

Primary metric:
- Median steps-to-solve on solved levels.

### Stage 6 - Explorer
Deliverables:
- Exploration scoring over action candidates using:
  - novelty
  - uncertainty reduction
  - hypothesis discrimination proxy
  - loop/no-op penalties
- Time/step budget split: early exploration, late exploitation.

Acceptance criteria:
- More useful graph coverage under fixed action budget.

Primary metric:
- Unique states discovered per 1k steps (vs Stage 4/5).

### Stage 7 - Options (macro-actions)
Deliverables:
- Initial option library:
  - navigate-to
  - interact-near
  - select-tool-and-click
- Option interface: `status`, `steps_used`, `result_state_hash`.

Acceptance criteria:
- Reduced action count for repeated micro-sequences.

Primary metric:
- Median actions per solved episode.

### Stage 8 - Mechanic inference v0 + semantic memory v0
Deliverables:
- Rule-based mechanic belief tracker from transition tokens.
- Cross-level semantic memory store (signature -> priors).

Acceptance criteria:
- Faster early decisions on repeated mechanic families.

Primary metric:
- Early-episode efficiency delta vs no-transfer baseline.

### Stage 9 - Critic v0
Deliverables:
- Rule-based critic to penalize:
  - thrashing/loops
  - irreversible-risk actions
  - dead-end likelihood
- Critic can propose one alternative action.

Acceptance criteria:
- Reduced catastrophic mistakes without suppressing progress.

Primary metric:
- Regression delta with critic on/off.

### Stage 10 - Offline learner modules
Deliverables:
- Dataset extraction from logs.
- Train one learned component at a time (start with action ranker or mechanic classifier).
- Ablation harness.

Acceptance criteria:
- Learned module beats heuristic baseline on held-out levels.

Primary metric:
- Ablation wins and confidence intervals.

## Implementation Work Breakdown

### A. Code structure in `zod01` (target)
- `zod01/src/env_adapter.py`
- `zod01/src/observation_parser.py`
- `zod01/src/state_abstract.py`
- `zod01/src/transition_diff.py`
- `zod01/src/memory_episodic.py`
- `zod01/src/world_model.py`
- `zod01/src/planner.py`
- `zod01/src/explorer.py`
- `zod01/src/safety.py`
- `zod01/src/controller.py`
- `zod01/src/agent.py`
- `zod01/src/logger.py`
- `zod01/src/eval_runner.py`

### B. Interfaces to lock early
- Canonical action format (simple + complex payload).
- Canonical state serialization/hash format.
- Transition token schema.
- Controller candidate-scoring contract.

### C. Test harness
- Deterministic replay tests.
- Parser/hash golden tests.
- Transition diff property tests.
- Planner correctness tests on toy graphs.
- Smoke tests against one local ARC game.

## Milestones
- M1: Stage 0-1 complete (deterministic trace + stable state hash)
- M2: Stage 2-4 complete (episodic graph + planner baseline)
- M3: Stage 5-7 complete (progress shaping + explorer + options)
- M4: Stage 8-10 complete (transfer + critic + first learned module)

## Risks and Mitigations
- Risk: Invalid/unsupported action emissions.
  - Mitigation: strict action-space gating in controller.
- Risk: State-hash instability.
  - Mitigation: canonical serialization tests and deterministic sort rules.
- Risk: Exploration thrashing.
  - Mitigation: explicit loop/no-op penalties and cycle breaker.
- Risk: Hard-to-debug regressions.
  - Mitigation: swap one module/term at a time with ablation logging.

## Immediate Next Tasks
1. Scaffold Stage 0 files and dataclasses.
2. Implement deterministic Env Adapter and action validator.
3. Implement structured logger with trajectory hash.
4. Add first smoke test for one game run.
