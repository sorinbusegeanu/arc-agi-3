# V10 — Live Runs, Replay, and Memory

This package implements replay, offline analysis, bounded live play, and persistent game memory described in [`DESIGN.md`](DESIGN.md) v0.10. Version 10.0.2 adds an append-only action conversation for Strata-compatible prefix reuse. It converts traces to evidence, validates JSON decisions, and executes live actions only through budgeted game tools.

## Requirements

Python 3.12+, `numpy`, and Pillow (provided by the ARC-AGI dependency). No MCP server dependency is required; the tool contracts can be wrapped by MCP later.

## Tool APIs

- `GameTools(env, run_dir).observe()` / `.act(action, args)` / `.reset(justification=...)`
- `pixel_diff(before, after)` / `delta_image(before, after)`
- `History(run_dir).append(record)` / `.query(action=..., limit=..., through_step=...)`
- `save_game_memory(store_dir, markdown)` / `save_global_memory(state_dir, markdown)`
- `GameMemory(...).record_review_checkpoint(...)` / `.record_markdown_revision(...)` (historical ledger)
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

The shared/global action instructions are `skills/SKILL.md` and are referenced by hash in each run manifest; they are no longer copied into disposable run directories. Shared reusable lessons live in `skills/global/MEMORY.md`; game memory lives under `skills/games/<game-id>/`. Reviews return concise game and global Markdown fields; Python validates their sizes and atomically saves each file. These saved files are authoritative for action guidance. Live action prompts load both files in full; offline replay analysis uses the configured excerpt of game Markdown and does not currently load global Markdown. The append-only `memory.jsonl` ledger retains exact game/global Markdown revisions, run state, legacy review text, and review checkpoints; it is not rendered into prompts. Exact successful routes and state compatibility remain in `level_solutions.jsonl`. Review checkpoints store covered event counts so repeated reviews remain idempotent. `V10_STATE_DIR` or `--state-dir` can override the persistent root. Project-local memory files can be versioned with Git.

## Phase 4: ablations

Compare pixels/actions with and without up to five complete Markdown blocks of read-only memory using one output-token budget and one replay cut:

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
  --seed 0 --max-actions 30 --plan-horizon 8 \
  --reasoning-effort medium --timeout-seconds 300 --max-output-tokens 6144
```

The repository launcher forwards options to the same live CLI:

```sh
src/start_live.sh --game-id ez01 --run-dir runs/v10/ez01_100 \
  --max-actions 100 --plan-horizon 8 --log-request --log-response
```

### Planning horizon

`--plan-horizon N` sets the maximum number of actions the model may propose per call. The default is **8**, and the accepted range is **1–8**. A plan can contain fewer actions and can end early on a level change, terminal state, invalid next action, replay divergence, or a known no-op in a multi-action plan. A single-action experiment may repeat an input to test delayed effects.

The plan horizon controls actions per decision and is independent of the persistent action conversation. After the initial request, each call adds only new actual outcomes and the current screen; prior messages and screenshots remain in the conversation. Memory reviews run after a level completion or WIN. During active play, a review also runs at the end of an action set when more than 9 actions have occurred since the previous review; it does not interrupt a plan and is skipped at terminal state or budget exhaustion. Stagnation alone does not trigger a review. Game-over post-mortems remain a separate recovery step.

Live CLI defaults:

| Option | Default | Behavior |
| --- | --- | --- |
| `--base-url` | `http://127.0.0.1:8080` | Strata OpenAI-compatible endpoint |
| `--model` | `qwen3.8-flash-next-iq3_s` | Deployed model identifier |
| `--seed` | `0` | Seed for environment and safe random fallback actions |
| `--max-actions` | `30` | Total action budget, including bootstrap and route replay |
| `--plan-horizon` | `8` | Maximum actions per plan; accepted range is 1–8 |
| `--timeout-seconds` | `300` | Per-request timeout |
| `--max-output-tokens` | `6144` | Maximum generated tokens, including reasoning and JSON output |
| `--reasoning-effort` | `medium` | Strata reasoning effort (`none`, `low`, `medium`, or `high`) |
| `--reasoning-budget-tokens` | unset | Optional positive cap on reasoning tokens |
| `--game-memory-max-chars` | `3000` | Maximum accepted game Markdown size |
| `--global-memory-max-chars` | `6000` | Maximum accepted global Markdown size |
| `--memory-context-entries` | `5` | Compatibility option (0–5); recorded in metrics and does not limit saved Markdown |
| `--log-request`, `--log-response` | off | Write request and/or raw response attempts into the run directory |
| `--context-window-tokens` | detected from Strata | Override the context size used to start a fresh action conversation before compaction |

Requests use temperature 0. Action plans and memory reviews use JSON response mode; game-over post-mortems return plain Markdown. `--reset-memory` clears game memory, saved routes, and global memory before startup. `--state-dir` overrides the default project `skills/` persistence root; `--skill` selects another shared action-instructions file.

The live command clears existing contents from its `--run-dir` before starting; it refuses protected locations, symlink run directories, and overlap with game/global persistence or the skill source. `--game-id` selects the game, `--max-actions` sets the total action limit, and `--plan-horizon` sets how many actions the model may return at once (default 8, range 1–8). A level-completion review receives the complete ordered action/outcome history for that level and attaches at most the latest 20 available screens in chronological order, including intermediate frames and the completion transition. Active-play reviews are checked only after an action set finishes and trigger after more than 9 actions since the prior review; they do not interrupt the set. Budget exhaustion does not trigger a review or assessment. `GAME_OVER` uses its separate post-mortem before restart when budget remains. Stagnation still guides action selection and limits plans to one action while recovering; it does not independently trigger a review. After a review, the next normal action prompt asks whether to reset or continue: continuing returns a fresh plan for the current screen; choosing reset restarts the normal startup flow. An exact verified-route match is replayed, a route for a different starting screen is provided as guidance, and bootstrap runs only when no verified route exists for the level. Startup bootstrap planning itself retains its no-reset instruction. `--game-memory-max-chars` and `--global-memory-max-chars` set the maximum accepted review-output sizes (defaults: 3000 and 6000). A nonterminal level replayed entirely from a verified route may skip its prose review when persistent game memory already exists; terminal `WIN` always receives final consolidation. Review does not advance or reset the game. It stores review triggers, transition coverage, result status, and timing in `reviews.jsonl`; `metrics.json` reports action and review usage separately. Invalid or failed reviews leave prior Markdown files intact and the game loop continues. Successful level routes are saved before review and remain available if review fails.

The runner prints the current level, action count, each planned and executed action, and inference time to stderr while running. On the initial level, an exact verified-route match is replayed immediately; a route for the same level from another starting screen is supplied as guidance without replay. If no route exists, the runner bootstraps, resets, then calls the model with ordered bootstrap screens and the restored initial screen. On every later level, it also bootstraps, resets the game, replays exact verified routes for earlier solved levels, then calls the model from the restored start of the bootstrapped level. Missing routes or replay divergence stop restoration rather than presenting a mismatched screen. Bootstrap and replay actions count against `--max-actions`. Resets have no separate budget; `RESET` is available whenever the environment supports it. A `GAME_OVER` triggers a dedicated post-mortem (not a memory review) with the complete active-level action sequence and every available screen in chronological order, including intermediate environment frames, with action entries cross-referencing their source and produced screen numbers; its diagnosis is logged and supplied to the retry prompts before the game resets when budget remains. `WIN` stops without restarting. `metrics.json` includes post-mortem usage and `game_over_restarts`. Level-completion reviews include the full ordered action history and attach at most the latest 20 available screens in chronological order, with shared action/screen references. Game-over post-mortems still include every available screen. All later prompts use the ordinary action prompt. Directional actions are shown as `up`, `down`, `left`, and `right` in console output and run logs; the adapter still sends the corresponding environment IDs. It stores a run manifest, rendered frames, append-only transition history, decisions and plan outcomes, and `metrics.json`; the action model receives saved game/global Markdown as read-only context, with no duplicate strategy payload. Review calls write the two Markdown files and append a JSONL checkpoint. The model returns only an `actions` array with legal action names and arguments. The four movement actions are presented as `up`, `down`, `left`, and `right`, and the model must return those exact lowercase words; the harness converts them to environment IDs `ACTION1`–`ACTION4`. Other legal actions, including `RESET`, retain their listed names. Actions are revalidated against each observation; plans stop on level changes, terminal states, action errors, budget exhaustion, or illegal next actions. Verified successful trajectories are stored per level and exact starting state; the shortest verified success for that start is replayed, and complete prior winning run logs are validated and imported at startup. Same-level routes from another start guide the model but are never replayed. Cached actions stop on divergence or illegality and return to model selection. Diverged steps are suspended from ordinary route selection for the rest of the run. `RESET` is listed as a legal pseudo-action whenever the environment supports it, has no separate budget, and must be returned alone with a justification in `args`. Stagnation summaries expose repeated states and ineffective actions; after stagnation the prompt directs the model to try a different action or reset. Object roles, goal identity, and completion rules stay scoped to levels where they were tested. Before each planned action in a multi-action sequence, the harness checks the full active-level history and stops before a known same-screen no-op with the same arguments; single-action experiments can repeat inputs to test delayed effects; if model output remains invalid and every real parameterless action is a known no-op, fallback recovery resets the game. If the model decision and its single repair attempt both fail validation, the runner logs the rejection, takes a seeded-random legal action when its required arguments can be generated from the action schema, records that transition, and resumes the model loop. If no safe fallback action can be generated, the run stops. On interruption, completed evidence and saved routes remain intact and `metrics.json` records `stop_reason=interrupted` without launching a shutdown review.

The eight-action horizon controls how many actions the model may plan before the next decision. Action calls use an append-only conversation: after the first request, the next user turn adds only new actual outcomes, current state/legal actions, and the current screenshot. Earlier submitted images and accepted model responses remain in the message history; full screen/action evidence remains in run logs and is still attached to reviews and post-mortems. A game reset or memory revision appends reset/recovery details or the latest authoritative Markdown to the same conversation. A fresh conversation starts after a genuine level change or before context compaction. `metrics.json` reports conversation resets and Strata prompt/cache/prefill token metrics when provided by the server. Strata must have conversation caching enabled to report cache reuse; v10 does not infer reuse from repeated message prefixes.

Pass `--reset-memory` to a live run for a clean memory restart. Historical route import and legacy ledger migration are disabled for that run, so old logs cannot immediately restore cleared knowledge. It removes `global/MEMORY.md` and all game-specific memory and solution stores under `<state-dir>/games/` before starting; it leaves `skills/SKILL.md`, unrelated files under `global/`, and the run directory alone.

Strata's per-request `reasoning_effort` defaults to `medium`, which enables the model's thinking phase. Use `--reasoning-effort none` to disable thinking, or choose `low` or `high`. Set `--reasoning-budget-tokens` to a positive value for a hard cap on thinking. `--max-output-tokens` caps reasoning plus the final JSON response. If Strata rejects a generated answer as malformed in JSON mode, the client retries once without server-side JSON validation; v10 then validates the returned text and can request its existing corrective response.

For an individual `live`, `analyze`, `ablate`, or `optimize` invocation, add `--log-request` and/or `--log-response` to save each actual HTTP attempt in the run directory as `model_requests.jsonl` and `model_responses.jsonl`. Both are off by default. Request logs include prompts and base64-encoded image payloads; response logs retain the raw response body, including HTTP error bodies. Authorization headers and API keys are not logged. Logging includes retries and local validation repair calls.

## Regression tests

```sh
PYTHONPATH=src pytest -q tests/test_v10_replay.py tests/test_v10_analysis.py tests/test_v10_memory_integrity.py tests/test_v10_live.py
```

### Reading the sequence and logs

Screens and actions are numbered separately within each supplied sequence. Each action record includes `action_number`, `starts_from_screen_number`, and `produces_screen_numbers`. The produced list contains every intermediate screen followed by the final result, so the following action starts from its last screen.

| Action | Starts from | Produces, in order |
| --- | --- | --- |
| Action 1: `up` | Screen 1 | Screen 2 (intermediate), Screen 3 (result) |
| Action 2: `left` | Screen 3 | Screen 4 (result/current) |

After bootstrap reset and restoration, the restored CURRENT SCREEN is a separate observation. It is not presented as an output of the last bootstrap action. The next plan starts from that restored screen. If restoration reaches a different board, stale bootstrap screens are discarded.

| Artifact | Contents |
| --- | --- |
| `run_manifest.json` | Effective model and run settings, skill hash, starting memory hashes, persistence paths |
| `evidence.jsonl` and `frames/` | Completed transitions, resets, observations, and all persisted intermediate frames |
| `decisions.jsonl` | Accepted/rejected decisions, replay/fallback sources, executed actions, interruptions, request IDs |
| `reviews.jsonl` | Review triggers, coverage, image labels, save/repair status, usage, request IDs |
| `post_mortems.jsonl` | Failed-level actions/screens, diagnosis, request IDs, usage, and retry guidance |
| `metrics.json` | Actions, resets, progress, stop reason, and separate action/review/post-mortem costs |
| `model_requests.jsonl` | With `--log-request`: complete payloads, payload bytes, image count, request IDs, UTC timestamps |
| `model_responses.jsonl` | With `--log-response`: raw bodies/errors, status, request IDs, UTC timestamps, latency |

Successful exact routes remain in `skills/games/<game-id>/level_solutions.jsonl`; Markdown revisions are historical records in that game's existing `memory.jsonl`. The action model receives saved Markdown and applicable route guidance, rather than raw ledgers.

### Correctness and recovery

The native adapter preserves every intermediate environment frame and exposes ACTION6 x/y coordinate schemas. Probe completions are saved immediately; a bootstrap WIN is retained without resetting. Restoring a later level checks its starting fingerprint, and discards stale bootstrap context when the board differs.

Winning-log recovery accepts logged direction names, reset episodes, and intervening audit records while checking the complete evidence chain before writing routes. All attempt restarts share one initialization path.

The existing game `memory.jsonl` archives exact previous and replacement game/global Markdown with hashes and revisions. These snapshots do not enter action prompts. Run manifests capture model settings and initial memory hashes; optional model logs record UTC timestamps, payload size, image count, request IDs, and response latency.

Offline `--memory-context-entries` and optimization `--memory-limits` select complete Markdown content blocks (standalone headings do not consume the limit); zero omits memory. The live option remains a compatibility setting and does not trim saved Markdown.
