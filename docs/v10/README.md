# Hydra Next — Replay, Analysis, and Live Runs

This package implements replay, offline analysis, bounded live play, and incremental game memory described in `DESIGN.md` v0.5. It converts traces to evidence, sends prefix-bounded multimodal context to a configured OpenAI-compatible Strata endpoint, validates JSON decisions, and executes live actions only through budgeted game tools.

## Requirements

Python 3.12+, `numpy`, and Pillow (provided by the ARC-AGI dependency). No MCP server dependency is required; the tool contracts can be wrapped by MCP later.

## Tool APIs

- `GameTools(env, run_dir).observe()` / `.act(action, args)` / `.reset(justification=...)`
- `pixel_diff(before, after)` / `delta_image(before, after)`
- `History(run_dir).append(record)` / `.query(action=..., limit=..., through_step=...)`
- `GameMemory(run_dir, store_dir=...).read()` / `.propose_update(patches)`
- `validate_decision(payload, legal_actions)`
- `replay_trace(trace_path, run_dir, through_action=None)`
- `build_analysis_input(run_dir, through_action=4)` / `run_analysis(run_dir, client, through_action=4)`
- `ArcadeGameEnv(game_id, seed=0)` / `run_live(env, run_dir, client, game_id=...)`
- `run_ablation(run_dir, client)` / `run_optimization(run_dir, client_factory)`
- `StrataClient(base_url, model, api_key=None).complete(...)`

The live environment adapter expects `observe() -> mapping` with a `frame` or `image` pixel grid and `legal_actions`, plus `act(action_id, args) -> mapping`. Action descriptors may include a `parameters` JSON-schema-like object for argument validation. Reset is explicit and requires a non-empty justification.

## Offline workflow

```sh
PYTHONPATH=src python -m v10.cli replay \
  --trace docs/v10/ls20-b09374e5-7bf8-48f4-ab6f-94d61c8c334f.json \
  --run-dir runs/v10/ls20-replay

PYTHONPATH=src python -m v10.cli analyze \
  --run-dir runs/v10/ls20-replay \
  --through-action 4 \
  --base-url https://strata-endpoint.example \
  --model <deployed-qwen-model>
```

The client reads `STRATA_API_KEY` from the environment. Analysis and live runs default to `http://127.0.0.1:8080` and model `qwen3.8-flash-next-iq3_s`; `--base-url` and `--model` remain available to override them. Replay stores ordered transition evidence, palette-rendered frames, intermediate frames, and same-shape pixel deltas. Analysis sends the previous and current actual frames, recent action outcomes, visible state, and compact read-only memory. Continuation evidence is used only after a valid model response; action comparison stops at the first divergence or a level/terminal interruption.

## Persistent skill and game memory

The shared/global action instructions are `skills/SKILL.md` and are referenced by hash in each run manifest; they are no longer copied into disposable run directories. Shared reusable lessons live in `skills/global/MEMORY.md`; game memory lives under `skills/games/<game-id>/`. Reviews return concise game and global Markdown fields; Python validates their sizes and atomically saves each file. These saved files are authoritative for action and replay guidance and are read verbatim. The append-only `memory.jsonl` ledger retains historical revisions, run state, legacy review text, and review checkpoints; it is not rendered into prompts. Exact successful routes and state compatibility remain in `level_solutions.jsonl`. Review checkpoints store covered event counts so repeated reviews remain idempotent. `V10_STATE_DIR` or `--state-dir` can override the persistent root. Project-local memory files can be versioned with Git.

## Phase 4: ablations

Compare pixels/actions with and without up to five read-only memory entries using one output-token budget and one replay cut:

```sh
PYTHONPATH=src python -m v10.cli ablate --run-dir runs/v10/ez01-replay \
  --through-action 4 --horizon 2 --max-output-tokens 1024
```

By default, each condition is also run with and without earlier **structured decision summaries** from this replay. Raw private reasoning is not stored or replayed. Results append to `ablation_runs.jsonl`; ablation arms do not update persistent memory or the normal decisions log.

## Phase 5: optimization

Sweep the plan horizon, output cap, memory limit, and conditioning profile on the same offline cut:

```sh
PYTHONPATH=src python -m v10.cli optimize --run-dir runs/v10/ez01-replay \
  --through-action 4 --horizons 1,2,4 --output-budgets 1024,4096 \
  --memory-limits 0,5
```

The Cartesian product is evaluated and appended to `optimization_runs.jsonl`. Each arm records action matches, token usage, and latency. Offline action matches do not measure policy quality or counterfactual live performance. Sweep lists can be shortened to limit model calls.

## Live workflow

```sh
PYTHONPATH=src:other_repos/arc-agi:other_repos/ARCEngine python -m v10.cli live \
  --game-id ez01 \
  --run-dir runs/v10/ez01-live \
  --seed 0 --max-actions 30 --max-resets 1 --plan-horizon 4 --review-interval 50 \
  --reasoning-effort medium --timeout-seconds 300 --max-output-tokens 4096
```

The live command clears existing contents from its `--run-dir` before starting; it refuses protected locations and symlink run directories. `--game-id` selects the game, `--max-actions` sets the total action limit, `--plan-horizon` sets how many actions the model may return at once (1–4), and `--review-interval` sets the action count between experience reviews (default: 50). `--game-memory-max-chars` and `--global-memory-max-chars` set the maximum accepted review-output sizes (defaults: 2000 and 1000). Reviews also run after level completion, `WIN`, unexpected legacy pattern predictions, or stagnation (at least three actions without progress plus repeated states/actions). Stagnation reviews require at least three fresh actions before retriggering. Simultaneous triggers produce one review. A review interrupts the remaining planned actions; the runner asks for a fresh action decision after review. Review does not advance or reset the game. It stores review triggers, transition coverage, result status, and timing in `reviews.jsonl`; `metrics.json` reports action and review usage separately. Invalid or failed reviews leave prior Markdown files intact and the game loop continues. Successful level routes are saved before review and remain available if review fails.

The runner prints the current level, action count, each planned and executed action, and inference time to stderr while running. Startup first reuses an exact saved route for the current level and starting screen if one is available. Otherwise, it runs the standard directional bootstrap, records its observations, resets to the initial screen, then makes one action-only model call. That call receives the complete ordered bootstrap screens and saved game/global Markdown and is told to plan from the reset initial screen, not from the bootstrap's final screen. Later action prompts also receive the complete ordered screens from recent actions, including every intermediate environment frame; review keeps its separate image selection. The default `--max-resets` is 1 to cover the bootstrap reset; setting it to 0 disables reset-based startup. All later prompts use the ordinary action prompt. Directional actions are shown as `up`, `down`, `left`, and `right` in console output and run logs; the adapter still sends the corresponding environment IDs. It stores a run manifest, rendered frames, append-only transition history, decisions and plan outcomes, and `metrics.json`; the action model receives recent outcomes and the saved game/global Markdown as read-only context, with no duplicate strategy payload. Review calls write the two Markdown files and append a JSONL checkpoint. The model returns only an `actions` array with legal action names and arguments. The four movement actions are presented as `up`, `down`, `left`, and `right`, and the model must return those exact lowercase words; the harness converts them to environment IDs `ACTION1`–`ACTION4`. Other legal actions, including `RESET`, retain their listed names. Actions are revalidated against each observation; plans stop on level changes, terminal states, action errors, budget exhaustion, or illegal next actions. Verified successful routes are stored per level and reused only from an exact observed starting state; cached actions stop on divergence or illegality and return to model selection. When reset budget is available, `RESET` is listed as a legal pseudo-action and must be returned alone with a justification in `args`. If the model decision and its single repair attempt both fail validation, the runner logs the rejection, takes a seeded-random legal action when its required arguments can be generated from the action schema, records that transition, and resumes the model loop. If no safe fallback action can be generated, the run stops. On interruption, completed evidence and saved routes remain intact and `metrics.json` records `stop_reason=interrupted` without launching a shutdown review.

Pass `--reset-memory` to a live run for a clean memory restart. It removes `global/MEMORY.md` and all game-specific memory and solution stores under `<state-dir>/games/` before starting; it leaves `skills/SKILL.md`, unrelated files under `global/`, and the run directory alone.

Strata's per-request `reasoning_effort` defaults to `medium`, which enables the model's thinking phase. Use `--reasoning-effort none` to disable thinking, or choose `low` or `high`. Set `--reasoning-budget-tokens` to a positive value for a hard cap on thinking. `--max-output-tokens` caps reasoning plus the final JSON response. If Strata rejects a generated answer as malformed in JSON mode, the client retries once without server-side JSON validation; v10 then validates the returned text and can request its existing corrective response.

For an individual `live`, `analyze`, `ablate`, or `optimize` invocation, add `--log-request` and/or `--log-response` to save each actual HTTP attempt in the run directory as `model_requests.jsonl` and `model_responses.jsonl`. Both are off by default. Request logs include prompts and base64-encoded image payloads; response logs retain the raw response body, including HTTP error bodies. Authorization headers and API keys are not logged. Logging includes retries and local validation repair calls.

## Regression tests

```sh
PYTHONPATH=src pytest -q tests/test_v10_replay.py tests/test_v10_analysis.py tests/test_v10_memory_integrity.py tests/test_v10_live.py
```
