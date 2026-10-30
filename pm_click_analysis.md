# Click-only game run investigation

## Follow-up: newest v11 run (`sy013`)

The v11 artifact resolves the run-identity mismatch above: `runs/v11/sy013` is a click-only run. Its manifest identifies run `5df19ae5fbd54b468a0f6ffa2c8b19e4`, game `sy01`, version `11.1.0`, action profile `a6`, and action signature `ACTION6`. The saved metrics show 172 clicks, zero of five levels completed, one GAME_OVER/restart, 328 actions remaining, and `game_state: NOT_FINISHED`. The metrics file still says `stop_reason: running`; no v11 CLI process for this run is currently present, so the artifact does not record a clean final stop reason.

### Main cause of the unsuccessful run

The click coordinate adapter worked. The model-facing legal action was `click` with integer `x` and `y` bounded to `[0,63]` on the 64×64 native frame. Evidence records canonical `ACTION6` with those same arguments and click-effect measurements; `invalid_click_argument_decisions` is zero. The model-facing prompt explicitly defines zero-based native `x`/`y`, provides a concrete one-click JSON example, and includes up to 12 candidate click sites.

The dominant observed failure was decision-schema rejection followed by automated candidate fallbacks:

- `decisions.jsonl` contains 125 rejected decisions, 55 accepted model decisions, and two accepted bootstrap model decisions. There are 116 executed fallback clicks: 111 caused by invalid model decisions and five by invalid bootstrap plans.
- The most common validation failures among rejected decisions were malformed/missing `obstacles[0]` fields (40), malformed/missing `target_candidates[0]` fields (36), invalid obstacle status labels (24), invalid target certainty labels (12), and invalid player certainty labels (5). Three outputs exceeded the one-action horizon. These are schema errors, not coordinate errors.
- Thus 116 of 172 clicks were fallbacks, not accepted model choices. The fallback policy kept the run moving while obscuring model failure as apparent action activity. Several fallbacks revisited candidate points; overall there were 52 distinct exact coordinates and 34 transitions with zero changed pixels. No click completed a level.
- The first 59-click attempt ended in GAME_OVER at step 59 on `click(x=21,y=31)`. The post-mortem found that this coordinate had produced the same visible effect on earlier clicks without ending the game, and hypothesized that a changing corner indicator represented an independent time or move budget. The evidence supports the correlation and repeated-click observations, but does not prove what the indicator measures or why the terminal state occurred.

The request already showed the exact constrained response example, but the model repeatedly returned extra/missing object keys and categorical labels outside the allowed values. The decision validator correctly rejected them; the retry/repair flow did not reliably bring the response into contract. The run then spent hundreds of model calls and substantial time on analysis and repeated fallback clicks instead of reliably executing deliberate decisions.

### Order of model analysis and post-mortem

The initial click-only prompt did not force movement semantics: it says an avatar may not exist, gives the native coordinate convention, and asks for one informative click while uncertain. It includes candidate suggestions and then the core Observe/Hypothesize/Evaluate/Experiment/Act reasoning procedure. Discovery tool calls are allowed before the one-action decision. The coordinate action profile adds a click-specific `Observe → Model → Compare → Click → Verify` procedure, advises testing empty/background cells as well as objects, makes candidates optional, and explains local versus remote click-effect counts. So v11 is materially more click-aware than v10.

When model output failed validation, the harness sent validation errors and requested a corrected structured answer; after retries were exhausted it executed a legal candidate fallback. The actual invalid field categories persisted across the run. The one GAME_OVER post-mortem was generated after the first 59-click attempt from the ordered click list and labeled chronological screens. Its findings were then included in later action prompts (verified in subsequent request records). It called out no-op retries and coordinate oscillation, and recommended one test on the right side of the divider. The next post-reset action sequence did not start with that suggested point, and later actions continued to revisit coordinates; there is no evidence that the post-mortem was durable memory.

### Skill and memory specialization

`skills/v11/SKILL.md` is still a shared, mostly generic game procedure. It explicitly allows a missing avatar and tells the model to use the model-facing action name and arguments. Click-specific guidance is supplied separately by `skills/v11/profile_capabilities/coordinate_action6.md`, selected through the `a6` profile. That file is specialized to coordinate-action games, but it is action-interface guidance, not sy01 puzzle knowledge.

There was no usable sy01 or a6 memory at run start: the manifest hashes are null/empty, and the prompt's `memory` and `profile_memory` fields were empty. At inspection, `skills/v11/state/games/sy01/MEMORY.md` and `skills/v11/state/profiles/a6/MEMORY.md` did not exist; `memory.jsonl` contained run-state snapshots, not learned mechanics. All seven recorded reviews failed with HTTP 400 because their prompts were 313,182–391,641 tokens against a 65,536-token context window. None saved game memory or profile candidates. That left the action procedure specialized, but no accumulated game-specific guidance to use after the first attempt.

### Assessment

The adapter and coordinate conversion were correct for this run. The click profile and per-turn prompt were specialized enough to describe how to reason about coordinates, so the failure was not that the model was given only movement concepts. The biggest demonstrated blocker is the high rate of invalid decision JSON and the fallback policy converting those rejections into 116 real clicks. The second is that memory review never completed, preventing durable learning. Gameplay evidence also shows repeated click sites and no-ops; the saved post-mortem recognized some of this but did not produce a successful route. The available evidence does not prove the exact level objective or the fatal click's causal mechanic.

Evidence is in `runs/v11/sy013/{run_manifest.json,metrics.json,model_requests.jsonl,model_responses.jsonl,decisions.jsonl,evidence.jsonl,reviews.jsonl,post_mortems.jsonl}` and the active prompt sources are `skills/v11/SKILL.md`, `skills/v11/profile_capabilities/coordinate_action6.md`, `src/v11/analysis.py`, and `src/v11/live.py`.

## Scope and run identity

The newest run artifact under `runs/v10` is `runs/v10/va03`, run ID `88a9eee7e4584e0b89652f7309d9d0b4` (metrics modified 2026-10-28 20:02:35 local time). Its manifest records v10 `10.0.29`, model `qwen3.8-flash-next-iq3_s`, a 500-action budget, `stop_reason: interrupted`, 177 executed actions, four levels completed out of five, and `won: false`.

**This artifact is not evidence of a click-only game run.** Its reset/legal-action evidence exposes `ACTION1` through `ACTION4`, with no coordinate arguments. The prompt presented those as `up`, `down`, `left`, and `right`; evidence and decision logs contain those same directions. There is no `ACTION6`, `x/y` click, or click-only action in this run. I checked the reset legal-action lists and executed actions in every available `runs/v10` evidence log; they all contain only the four directional actions. No click-only run is present in this artifact set. The report therefore separates what the latest recorded run proves from how the current code would handle a click-only game.

## What happened in the latest run

- The run began with an automatic `up, left, down, right` bootstrap probe, reset, and then model-driven play. The bootstrap is visible in the first decision record and in the first model request's screen labels.
- The game reached level 4 of 5, with 177 live actions, 21 resets, 3 GAME_OVER restarts, 11 reviews, and 3 post-mortems. It stopped because the run was interrupted, not because the action budget ran out.
- The three post-mortems cover failures after 6 actions on level 1 (twice) and after 21 actions on level 4 (once). The first diagnosis initially guessed that a cyan meter might be a move budget; the second corrected this using the labeled screens and sequence, identifying budget depletion and wasted backtracking as the better-supported explanation. The level 4 diagnosis also called out a long route that exhausted a resource before confirming the goal.
- This was an unsuccessful overall run, but not an “easy click-only” failure: it completed four levels and was interrupted before a fifth-level win.

## Prompt and analysis order

The action system prompt is broad and movement-oriented. It requires Observe → Hypothesize → Evaluate → Experiment → Act, asks for a first-action prediction, and includes extensive guidance about player location, movement, routes, obstacles, no-ops, and grid interpretation. It does not identify the game as click-only or provide click-target selection guidance. The model sees the legal action schema, so an `ACTION6` schema with `x/y` would at least expose the coordinates as required arguments, but the shared prose does not explain how to choose them.

For this run, the actual sequence was:

1. Start routine / bootstrap movement probe.
2. Reset and replay the probe context into model action selection.
3. Model action selection, with native discovery tool calls where used; discovery outcomes are appended before the final action decision.
4. Execute actions and write transitions to `evidence.jsonl`.
5. Periodic review calls consolidate durable game and global guidance (11 saved reviews in this run).
6. On GAME_OVER, build a post-mortem input from the complete active-level action sequence, chronological before/after/intermediate screens, and current game/global memory; save diagnosis and expose the latest diagnosis as transient guidance on a subsequent action turn when the conversation continues.
7. Reset and run the normal startup/bootstrap flow again.

Post-mortem handling is evidence-ordered and generally sound in design: the implementation labels screens and associates each action with the screen it starts from and produces, sends the full sequence, and tells the model to distinguish observations from hypotheses. The concrete outputs show one weakness: the first level 1 report speculated about a move budget before the next report corrected it. The post-mortem itself is not a memory write; periodic review is the durable-memory path. The model action prompt asks it to check same-level history and avoid repeating experiments, but this remains prompt guidance rather than a guaranteed solver.

## Is the skill or memory click-specialized?

`skills/SKILL.md` is a general v10 procedure, not a click-action skill. It explicitly says to use lowercase movement names `up/down/left/right`, discusses movement and grids, and contains no dedicated click-only section. `skills/global/MEMORY.md` is also generic exploration guidance. It advises comparing frames, testing one interaction, and not inferring from appearance; it does not teach coordinate targeting, native-frame click coordinates, clicking empty cells, remote effects, or state-specific click retries.

The game memory recorded for this latest run (`skills/games/va03/MEMORY.md`) is highly specific to va03's movement/trail mechanics and levels. It is not a click profile. The run manifest also records no starting game-memory file/hash, while later action requests can receive the accumulated reviewed memory. In general, the memory architecture scopes game Markdown by game ID, but the shared skill and global memory are generic.

## Action-space and adapter findings

`src/v10/arcade_env.py` builds legal actions from the environment's actual available action IDs. For `ACTION6`, it adds required integer `x` and `y` arguments bounded by the native frame width and height, then forwards those arguments to the engine. The generic game tool validates the action against the current legal-action schema and passes the canonical environment ID and args to `env.act`. This is a structurally correct path for a native `ACTION6(x,y)` click.

There are two material gaps for click-only games:

1. `src/v10/analysis.py` aliases `ACTION1`–`ACTION4` to directions unconditionally when those IDs are legal, but has no `ACTION6` ↔ `click` alias. A game whose actual legal click is `ACTION6` will still be shown `ACTION6` plus its args, not a semantic `click` action. This is not a coordinate conversion error by itself, but it leaves click support opaque and makes any nonstandard use of the first four IDs unsafe to interpret as movement.
2. The default startup/bootstrap routine only emits the four directional actions. If a click-only game exposes only `ACTION6`, `_standard_start_routine` returns no actions; the per-level bootstrap explicitly skips with “no legal directional routine actions.” The initial startup flow likewise disables its probe when no directional routine is available. So click-only games do not get the normal bootstrap learning sequence, and the prompt has no click-specific replacement procedure.

Accordingly: **the native adapter can carry an `ACTION6` coordinate action, but the end-to-end click-only interaction path is incomplete and not click-specialized.** The current v10.0.29 path does not provide a `click` alias, click candidate generation, click-aware bootstrap, click metrics, or click-specific skill guidance. `docs/v10/v10.1.0.md` describes these as planned phases and lists the alias and adaptive click-only bootstrap as unchecked; that document is a design/plan, not proof they were implemented.

## Likely explanation and limits

The latest v10 run failed as a whole because it did not reach the fifth-level win before being interrupted. Its in-run evidence points to exploratory and route-planning failures on movement levels, including repeated GAME_OVER on level 1 and budget exhaustion on level 4. It cannot establish why a click-only game failed, because no click-only game appears in the available v10 run artifacts.

If the intended target was a click-only attempt, the first concrete issue to resolve is run identity: the newest `runs/v10` artifact is va03 and uses four directional controls. Independently, source inspection predicts a click-only weakness: the action schema can represent `ACTION6(x,y)`, while startup, aliases, prompts, and shared skill remain movement-centric. The v10.1 design document itself recognizes this gap and proposes click-only fixtures and adaptive bootstrap; those planned changes are not present in the inspected 10.0.29 implementation.

## Evidence pointers

- Run identity and aggregate result: `runs/v10/va03/run_manifest.json`, `runs/v10/va03/metrics.json`.
- Exact prompt and visible legal-action schema: first entries of `runs/v10/va03/model_requests.jsonl`; actual decisions: `runs/v10/va03/decisions.jsonl`.
- Ordered transitions and environment legal actions: `runs/v10/va03/evidence.jsonl`.
- Review outputs and post-mortem screen/action sequences: `runs/v10/va03/reviews.jsonl`, `runs/v10/va03/post_mortems.jsonl`.
- Prompt construction and GAME_OVER flow: `src/v10/live.py` (`_POST_MORTEM_SYSTEM_PROMPT`, `_perform_game_over_post_mortem`, `prepare_action_turn`, and GAME_OVER restart loop).
- Native action schema and mapping: `src/v10/arcade_env.py`, `src/v10/analysis.py`, `src/v10/tools/game.py`.
- Shared and game-specific guidance: `skills/SKILL.md`, `skills/global/MEMORY.md`, `skills/games/va03/MEMORY.md`.
- Click-only support gap documented as planned work: `docs/v10/v10.1.0.md`, especially the click alias, adaptive bootstrap, click candidate, and release-gate sections.
