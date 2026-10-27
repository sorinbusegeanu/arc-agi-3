# V10 Implementation Status

**Updated 26 October 2026 · Current design: [DESIGN.md](DESIGN.md) v0.10**

This document records the implemented architecture and default live configuration. The design document is authoritative for runtime behavior; this file replaces the older pattern-ledger implementation notes, which no longer describe action guidance or review output.

## Implemented ownership

- `src/v10/live.py` owns live startup, level bootstrapping, action prompts, review and post-mortem triggers, complete screen sequencing, shared attempt initialization, and execution orchestration.
- `src/v10/analysis.py` owns the action-only JSON contract, direction-name conversion, and decision validation.
- `src/v10/tools/game.py` owns environment calls, legal-action validation, action budgets, unrestricted justified resets, and transition evidence.
- `src/v10/tools/solutions.py` owns verified route persistence, starting-state compatibility, route selection, and reuse. Successful routes are saved immediately and do not depend on a successful review.
- `src/v10/tools/memory.py` owns persistent paths, Markdown validation and atomic saves, and the append-only historical ledger. Game and global `MEMORY.md` files are the authoritative durable memory read into action prompts; `memory.jsonl` is not sent to the action model.
- `src/v10/strata_client.py` owns requests to the OpenAI-compatible Strata endpoint, image encoding, response parsing, and optional request/response logging.

## Current live defaults

| Setting | Default |
| --- | --- |
| Strata endpoint | `http://127.0.0.1:8080` |
| Model | `qwen3.8-flash-next-iq3_s` |
| Reasoning effort | `medium` |
| Reasoning token cap | none |
| Maximum generated tokens | `6144` |
| Request timeout | `300` seconds |
| Action budget | `30` |
| Plan horizon (`--plan-horizon`) | `8` actions (accepted range 1–8) |
| Action-set review | after more than `9` actions since the last review, checked at set end |
| Game/global Markdown limits | `3000` / `4500` characters |
| Review memory quality | Reject transient run status, position, action progress, and item counts; allow useful fixed level-layout counts; reject general blockage claims inferred from state-specific no-ops; repair once with error-specific guidance |
| Request and response logging | off |

Requests use temperature zero. Action plans and memory consolidation use JSON response mode; game-over post-mortems return plain Markdown. CLI defaults can be changed with the options documented in [README.md](README.md). The offline `analyze` and `ablate` commands retain their separate horizon limit of four; the live horizon is eight. Offline memory limits select complete Markdown content blocks, retaining their standalone headings; zero omits memory. Live action prompts load both complete saved memory documents. Game Markdown defaults to 3000 characters. Global Markdown defaults to 4500 with a hard save-path cap of 4500; reviews target 4000 where possible.

## Implemented behavior

- The model returns only an `actions` list. Movement actions are spelled `up`, `down`, `left`, and `right`; Python maps them to environment IDs. A reset decision is a dedicated question in each normal action prompt. Bootstrap prompts say not to reset.
- An exact saved-route match replays directly on the initial level. If no route is available, the runner bootstraps, resets, and requests an exploratory plan from the restored screen. Later levels run the same bootstrap/reset flow, then replay verified routes for completed levels before asking the model to act on the new level.
- Live plans can contain up to eight actions. The action context is independently bounded to the eight most recent transitions in the active level and includes all intermediate frames in order. After eight uninterrupted actions, this yields at least nine ordered screens plus environment-provided intermediate frames. Execution stops on terminal or level transitions, errors, budget limits, illegal actions, and known same-screen/action/argument no-ops within multi-action plans. Single-action experiments may repeat an input. A no-visible-change action can interrupt the remaining model plan for reassessment.
- Reviews run on level completion or WIN, and after more than 9 active-play actions since the previous review when an action set ends. Interval reviews do not interrupt a plan and are skipped if the budget expires. Completion reviews attach at most the latest 20 screens from the completed-level sequence. Game-over post-mortems receive every screen and action from the failed level, including intermediate environment frames. Action records identify their source screen and all screens produced, using the screen labels' shared numbering. The next normal prompt asks whether to continue with a fresh plan or reset. A model-selected reset re-enters normal route handling: exact matches replay, other saved routes guide, and bootstrap runs only when the level has no verified route. `GAME_OVER` receives a separate post-mortem before an automatic restart when action budget remains. Budget expiration alone does not trigger a review or assessment.
- Request/response logs are opt-in with `--log-request` and `--log-response`. They record retries as well as validation repairs in the run directory, with request IDs, UTC timestamps, payload size/image counts, and response latency. IDs link calls to action decisions, reviews, and post-mortems.
- On interruption, completed transitions and verified successes remain persisted. Metrics record `stop_reason=interrupted`; shutdown does not start a review or mark unfinished routes successful.

## Correctness review fixes

- Native temporal stacks are preserved as ordered intermediate frames plus the final observation; ACTION6 accepts validated coordinates.
- Bootstrap completions save routes immediately, and a bootstrap WIN stops without resetting.
- Replay divergence suspends the failing step and gives the action model control, including unchanged-screen divergence.
- Historical import normalizes directions and validates reset/audit chronology without joining routes across resets.
- `--reset-memory` suppresses historical recovery for the entire run.
- No-op checks distinguish arguments, so one failed coordinate does not exclude other clicks. Single-action experiments can test repeated-input effects.
- Later-level restoration compares the original starting fingerprint and drops stale bootstrap context on mismatch.
- Offline memory-limit experiments change the actual Markdown context.
- One shared routine clears per-attempt state for model, automatic, and game-over resets.
- Existing ledgers archive Markdown revisions; manifests record settings and initial memory hashes. Request metadata correlates calls with decisions and costs.
- Model screen context omits internal image IDs and duplicate mappings. Legacy pattern evaluation is gated to stores containing legacy patterns.
- Run-directory clearing rejects overlap with persistent memory and shared instructions.

## Validation

The correctness review passes 158 v10 tests. A native ez01 smoke check successfully observed a 64×64 screen and executed one up action through `ArcadeGameEnv` and `GameTools`; it did not invoke the model.

Run the focused v10 suite with the project's Python environment:

```sh
PYTHONPATH=src python -m pytest -q tests/test_v10_replay.py tests/test_v10_analysis.py tests/test_v10_memory_integrity.py tests/test_v10_live.py tests/test_v10_solutions.py
```
