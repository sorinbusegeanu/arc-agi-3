# Hydra v11 — Visual Click, Mixed-Action, and Action-Profile Harness

**Status:** v11.0 click/mixed-action implementation remains the baseline; v11.1 profile design is implemented. Live model benchmarks remain outstanding.
**Current release:** 11.1.0
**Fork baseline:** v10.0.29 source copied into `src/v11`; v10 remains unchanged
**Scope:** Fork the v10 LLM/vision/tool harness into v11, then add click-only and mixed click/movement ARC-AGI-3 support while preserving movement-game behavior and leaving v10 frozen.

## Version boundary and implementation order

- `src/v11` is a standalone copy of `src/v10` made before any click-feature code changes. V11 imports no runtime modules from `v10`.
- Keep v10 source, tests, docs, skill, and persistent state unchanged. The unchanged `v10.reasoned_action/1` response and evidence/route formats may remain compatible inside v11; new v11-only records use `v11.*` schema names.
- V11 started at `11.0.0`; action-set profiles and scoped profile learning ship in `11.1.0`. Root package metadata and the v11 runtime constant report `11.1.0`; v10 retains its `10.0.29` runtime constant.
- V11 defaults to `skills/v11/SKILL.md` and `skills/v11/state/` for game/profile memory and routes. It does not automatically read v10 memory, routes, or historical runs. `--state-dir` remains an explicit override; sharing requires the user to choose it intentionally.
- The original fork order above documents the completed v11.0 baseline. For v11.1 changes, keep the common harness authoritative and add only the profile composition and isolated profile-learning responsibilities.

## 1. Objective

Hydra shall use one observation–hypothesis–experiment–action loop for games offering any legal combination of directional movement, coordinate clicks, interaction/wait actions, and reset. The model observes native game frames, proposes testable interactions, and learns the relationships between actions and visual/state changes. Python provides exact execution, validation, candidate-location measurements, replay, budgeting, and evidence persistence.

The primary success metric is **environment-confirmed levels solved per action and per model call**, with valid-action rate and run reproducibility as supporting measures.

### Acceptance outcomes

1. Click-only environments can progress from their initial screen, select valid click coordinates, discover an interactive region, and solve a deterministic fixture.
2. Mixed environments can learn and execute sequences such as `click → move → click`, with arguments preserved in successful-route replay.
3. Movement-only tests, including the existing fixed directional bootstrap where applicable, remain behaviorally compatible.
4. The agent can accurately describe a game with no identified player without inventing an avatar.
5. Every click, including its exact coordinates, is validated, logged, reviewable, and replayable.

## 2. Verified baseline and gaps

The existing implementation already provides much of the required infrastructure:

| Existing owner | Current capability | v11.0.0 requirement |
| --- | --- | --- |
| `src/v11/arcade_env.py` | Exposes `ACTION6` with integer `x`/`y` bounds from the native frame | Preserve; verify coordinate convention and dynamic availability |
| `src/v11/tools/game.py` | Validates action arguments and executes one action with evidence | Reuse as the only execution/validation authority |
| `src/v11/tools/discovery.py` | Isolated `test_hypotheses` actions with arguments | Reuse for comparing coordinate hypotheses |
| `src/v11/tools/vision.py` | Pixel differences, changed boxes and candidate displacement | Add click-site proposals and local/remote effect measurements |
| `src/v11/live.py` | Image-grounded decisions; `_frame_regions`; remembered no-ops | Replace movement-only startup assumptions, improve click exploration and fallback |
| `src/v11/analysis.py` | Direction aliases, structured action contract, fixed `player` field | Add `click` alias and playerless guidance without breaking the response contract |
| `src/v11/tools/solutions.py` | Exact state/action/argument route recording and replay | Retain existing persistence format; cover mixed-action regressions |
| `skills/v11/SKILL.md` | General visual reasoning and action-testing instructions | Explain click targeting, coordinates, interaction effects and mixed strategies |

**Specific movement assumptions to address:**

- `_standard_start_routine()` in `live.py` probes `up`, `left`, `down`, `right` only.
- Bootstrap planning uses `directional_only=True`, and its fallback filters to directions/wait.
- `analysis.py` maps only `ACTION1`–`ACTION4` to model-friendly names and shows directional examples.
- `state_assessment.player` is always required even when a game has no moving player. Its existing `unknown` value provides a compatible representation.
- `_random_legal_action()` samples coordinate arguments blindly; it should prefer grounded, untried candidates.
- Some stagnation/equivalence records in `live.py` distinguish actions by name without consistently distinguishing click arguments; they must use the full `(action, args)` key.
- `_frame_regions()` currently drops regions smaller than four pixels, which can hide meaningful click sites.

This release extends those authorities; it does not introduce a second agent loop or a separate click-game implementation.

## 3. Architecture and responsibilities

```text
ARC environment (native 2-D grid + legal actions)
        |
        v
GameTools / ArcadeGameEnv  ----> evidence, frames, action budget
        |
        +--> Vision: current-frame click candidates and transition measurements
        |         |
        |         v
        +--> Action context: legal schemas, coordinates, candidates, history
                    |
                    v
             VLM agent (single reasoning/decision loop)
                    |
          click(x,y) / up / down / left / right / other legal action
                    |
                    v
         Existing validation -> execution -> exact transition evidence
                    |
         discovery tests / memory review / successful-route store
```

**Single owners:** environment calls in `GameTools` and `ArcadeGameEnv`; action naming and response validation in `analysis.py`; candidate selection measurements in `tools/vision.py`; bootstrap and run control in `live.py`; persistent solutions in `tools/solutions.py`. New behavior belongs in these modules unless a genuinely new responsibility requires a module.

### Invariants

- The current native frame is authoritative. Candidate overlays and region labels are derived visual aids.
- Legal actions come from the current observation and are revalidated immediately before execution.
- `x` denotes the zero-based column and `y` the zero-based row of the **native** grid: `0 <= x < width`, `0 <= y < height`.
- Model-visible aliases are normalized to canonical environment IDs before execution and storage. Existing solution records retain `ACTION6` plus exact `{"x": ..., "y": ...}` arguments.
- Candidate click points are suggestions, **not** an exclusive list of legal coordinates. The model may choose any in-bounds coordinate justified by the frame.
- Hypotheses and visual interpretations are model judgments; pixel measurements and environment-confirmed outcomes are evidence.
- Any change in native frame/state invalidates screen-specific click suggestions; previously tested coordinates remain historical evidence with their source-state reference.
- `RESET` remains a single action with its existing justification and reproducible-route safeguards.

## 4. Action capabilities and response contract

The run shall derive capabilities from `current_state.legal_actions` at each decision, rather than assigning a permanent game type from its ID.

| Legal action set | Exploration policy |
| --- | --- |
| Directional actions only | Preserve existing movement bootstrap and trajectory planning |
| `ACTION6` only | Vision-guided click candidates and one-click experiments |
| Both directional and `ACTION6` | Interleave informative movement and click tests; learn causal dependencies |
| Other parameterized actions | Preserve generic schema validation and use the same action loop |

### Canonical action aliases

- `ACTION1` → `up`; `ACTION2` → `down`; `ACTION3` → `left`; `ACTION4` → `right`.
- `ACTION6` → `click`, **only when `ACTION6` is present in legal actions**.
- `ACTION5` and other actions retain their advertised names unless a tested alias already exists.
- Mapping and validation must work even if no directional action is legal (fix the current early-return behavior in `_canonicalize_model_actions`).

Example model action list:

```json
{
  "actions": [
    {"action": "click", "args": {"x": 24, "y": 15}},
    {"action": "right", "args": {}},
    {"action": "click", "args": {"x": 31, "y": 15}}
  ]
}
```

This is an example of the **actions field**; the existing `v10.reasoned_action/1` wrapper retains `state_assessment`, `hypothesis`, `trajectory_rationale`, `prediction`, and `actions`.

**Compatibility decision:** keep `v10.reasoned_action/1` for v11.0.0. Its mandatory `player` entry may state `location: "unknown"`, `certainty: "unknown"`, and an evidence string such as `"No controllable avatar identified"`. Update prompt descriptions and examples so the agent is instructed to identify controllable *regions or objects*, including cases where there is no moving player. A separate response-schema migration can follow only if this representation proves insufficient in evaluated click games.

Generate the displayed action examples from currently legal action schemas (or show a mode-appropriate example). Validate missing, out-of-bounds, negative, fractional, boolean and extra click parameters through the existing argument validator. Log canonical actions with all arguments.

## 5. Coordinate-grounded visual candidate proposals

Extend the current connected-region logic with a bounded, deterministic candidate generator in `tools/vision.py` and expose results through the existing `live.py` context builder.

### Candidate generation

1. Read the latest native `HxW` integer grid and its stable frame ID.
2. Extract connected components, including meaningful single-pixel and narrow components. Keep component statistics independent of semantic labels.
3. Produce candidate sites such as component medoids/interior points, centers, boundary-adjacent sites, salient color changes, and candidate empty-space/grid locations when the image suggests board-cell clicking.
4. Deduplicate identical and nearby sites, preserve coverage across regions and colors, and rank for visual diversity and information gain. Default maximum: **12 candidate points** per current frame.
5. Prefer points not already tested from the **same exact observed state**. A prior click from another state does not mark a point as globally ineffective.
6. If visually salient targets are exhausted, offer a small coarse-to-fine spatial sampling proposal. This is a bounded fallback, not a 64×64 exhaustive scan.

Example derived context:

```json
{
  "click_candidates": {
            "schema": "v11.click_candidates/1",
    "source_frame_id": "F00021",
    "frame_size": {"width": 64, "height": 64},
    "coordinate_system": "native_zero_based_xy",
    "items": [
      {"id": "C1", "x": 12, "y": 18, "bounds_yxyx": [15, 9, 22, 17], "basis": "distinct_pixel_region"},
      {"id": "C2", "x": 42, "y": 31, "bounds_yxyx": [28, 38, 36, 47], "basis": "distinct_pixel_region"}
    ]
  }
}
```

`bounds_yxyx` is a half-open rectangle `[y0, x0, y1, x1]`; target points must lie within the frame. `basis` describes a measured visual feature, never a guessed game role. IDs are local to the frame and must not be treated as persistent object identities.

### Optional visual overlay

Attach a secondary annotated PNG with numbered candidate markers and an explicit native-coordinate legend when it improves VLM target selection. Keep the unannotated **CURRENT SCREEN** present and authoritative. Ensure resizing, padding, and marker placement cannot change the coordinate values passed to `ACTION6`. Overlay images are derived artifacts, recorded separately from observation frames.

### Critical limitation

Some click games require selecting empty cells, drawing paths, or interacting with regions that are not separated by color. Candidate generation must support exploratory locations beyond connected objects and should never hard-restrict the legal click space.

## 6. Capability-aware bootstrap and planning

### 6.1 Startup policy

Preserve exact verified-route lookup and replay before exploratory bootstrap. At an unsolved level:

- **Movement-only:** retain the existing bounded up/left/down/right bootstrap for regression compatibility.
- **Click-only:** start with an image-grounded single click, selected by the VLM using candidate suggestions and the full legal schema; inspect the result before a second speculative click.
- **Mixed:** select an informative legal probe across capabilities. Prefer testing one move and one click when each has a visually supported hypothesis; the actual order depends on the observed screen.

Default exploratory bootstrap cap: **four executed actions per unsolved level** (subject to remaining action budget). Stop probing on WIN, completion, terminal state, or sufficient evidence. Probe results are evidence, not automatically successful strategies.

For games with a reproducible reset, preserve the current probe → reset → verified restoration approach. Validate that the restored frame fingerprint matches the probe's start before reusing bootstrap conclusions. Where reset is unsupported or restoration cannot be verified, continue from the actual current observation with the gathered evidence rather than assuming the original screen was restored.

### 6.2 Bootstrap action validation

Replace the special `directional_only` gate with **bootstrap-specific legal-action validation**, which accepts any currently legal action and its required parameters while retaining the existing reset restrictions. The startup prompt shall say "choose informative legal actions" and include native coordinates for `click`.

Do not synthesize `{}` for a required-coordinate action. If a model bootstrap plan is invalid, repair it using the same schema. On exhausted repair attempts, prefer an untried grounded click candidate or legal movement action. Keep a safe budget-aware fallback for cases where no candidate exists.

### 6.3 Plan horizon

- Unverified click interaction: normally **one action**, followed by observation.
- Unverified mixed-action dependency: test the prerequisite and observe before proceeding.
- Verified, predictable click or mixed route: retain the existing multi-action plan horizon (up to eight).
- Stagnation: retain the current one-action recovery horizon.

Stop a queued plan at any observed level/terminal transition, invalid next action, unexpected state divergence, or existing multi-action no-op guard. Maintain current first-action prediction behavior.

## 7. Click effects, causal hypotheses and memory

Every click should produce the existing transition evidence plus optional measurements calculated from the native before/after frames:

```json
{
  "click_effect": {
    "schema": "v11.click_effect/1",
    "x": 24,
    "y": 15,
    "radius": 2,
    "local_changed_pixels": 3,
    "remote_changed_pixels": 11,
    "total_changed_pixels": 14,
    "level_changed": false,
    "terminal_changed": false
  }
}
```

The local neighborhood uses a documented bounded region (default Chebyshev radius 2); remote changed pixels are changes outside that region. Preserve intermediate frames for effects that occur between the before/after endpoints. These measurements do not by themselves prove that the click caused a specific semantic event, especially when a game changes autonomously.

`test_hypotheses` already supports parameterized branch actions. The model may compare distinct coordinates from the same reproducible base state, subject to the existing four-branch selection-cycle limit. Promote only observations supported by outcomes; use controls or repeated tests where spontaneous changes could confound an inference.

Memory reviews shall capture reusable conditional mechanics such as:

> On level N, clicking one of the observed blue regions changed the remote barrier; moving through the opened area then advanced progress. Evidence: transitions X and Y.

Store **conditions and evidence IDs**, not unsupported object semantics. Record successful mixed-action routes with exact arguments. Route replay uses canonical action IDs and the existing state fingerprint checks.

## 8. Stagnation, failed clicks and coordinate safety

Represent a state/action attempt with a stable key:

```text
(exact_source_state_fingerprint, canonical_action_id, canonical_json(args))
```

Use the full key for ineffective-action tracking, state/action equivalence, recovery decisions, and other action-specific comparisons. In particular, a failed `click(0,0)` must not suppress `click(1,0)` from the same state; the same coordinate in a changed state remains eligible. Keep the existing rule that a no-visible-change outcome is evidence of one attempt, not proof of global impossibility or absence of delayed effects.

Coordinate validation shall use `width` and `height` from the same current observation that supplied the legal schema. Every action in a queued plan is rechecked against the next live observation. Any candidate list generated for an earlier frame must be refreshed before the next model turn.

For fallback, prioritize untested visual sites and coarse-to-fine coverage over uniform random pixel selection. Record fallback selection and its justification separately from model decisions.

## 9. Evidence, metrics and compatibility

Keep the existing `evidence.jsonl`, `decisions.jsonl`, `frames/`, `level_solutions.jsonl`, and review-memory formats. New additive fields are versioned and optional for older runs. Historical movement-only logs remain replayable. For click actions, action records and route edges contain canonical `ACTION6`, `args.x`, `args.y`, source-frame ID and the observed outcome.

Add metrics:

- `click_actions_executed`, `unique_click_sites_by_state`, `click_actions_with_visible_change`;
- `click_level_completions`, `invalid_click_argument_decisions`, `candidate_click_fallbacks`;
- `movement_actions_executed`, `mixed_action_transitions`, `bootstrap_actions_by_type`;
- existing model calls, inference latency, action budgets, solved levels and route-replay counters.

Count click effectiveness by **state-specific** actions; distinguish exact no-ops from unverified/delayed effects. Do not infer that visually changed pixels equal game progress.

### Release/versioning note

The copied v10 runtime reports `10.0.29`. V11 is a separate feature line and started at `11.0.0`; action-set profiles and scoped profile learning ship in `11.1.0`. Keep root package metadata and the v11 runtime constant synchronized; leave v10's runtime version unchanged.

## Action-set profiles and scoped learning (11.1)

V11 remains one harness. At initialization and whenever legal actions change, it resolves the sorted environment action IDs to one of the nine documented signatures or `generic`. `RESET` is excluded. The resolver also fingerprints argument schemas, while prompt composition loads only the capability fragments that apply to the active signature from `skills/v11/profile_capabilities/`. Neither action names nor catalog descriptions are treated as proof of behavior. If the signature or schema changes during a run, the run logs the change and starts a fresh model conversation while retaining game evidence and game memory.

The general procedure belongs in `skills/v11/SKILL.md`; action-family knowledge belongs in the capability fragments; current state and constraints belong in the live prompt; game-specific findings belong in `games/<game-id>/MEMORY.md`. Approved transferable methods belong in `profiles/<profile-id>/MEMORY.md`. The former shared `global/MEMORY.md` is removed once per state root during migration, with a marker in `migrations/v11-profile-memory.json`.

Model review and discovery may only enqueue profile candidates linked to direct transition evidence. They never edit approved profile memory. The `v11 profiles` CLI lists, rejects, and promotes candidates. Promotion requires matching normalized claims from at least two source games and a held-out report that identifies a distinct test set, candidate and baseline revisions, measured metrics, and an improvement. The proposed replacement Markdown is human-curated and recorded in an append-only promotion audit. `--reset-memory` removes learned game/profile state while preserving migration markers and tracked capability prompts.

Each run manifest, profile-switch event, review, and model action context records profile ID, sorted action signature, argument-schema fingerprint, core skill revision, composed capability-prompt revision, and game/profile memory revisions. Reproducibility logs must not contain secrets or raw model credentials.

## 10. Risks and design decisions

| Risk | Control |
| --- | --- |
| Native 64×64 coordinates differ from resized screenshot positions | Always return integer coordinates in native-frame axes; annotate optional overlay |
| Small or empty-cell click targets are missed | Include single-pixel regions and a coarse-to-fine fallback |
| Clicks have remote or delayed effects | Report local/remote changes, intermediate frames, and conditional hypotheses |
| Mixed game requires a sequence before a click works | Preserve short causal probes and longer verified mixed sequences |
| Startup wastes actions or resets after a successful probe | Bound the probe, stop on completion/WIN, retain verified restoration rules |
| All clicks treated as one action in stagnation checks | Key by canonical action **and** complete arguments |
| Playerless game causes hallucinated player observations | Explicitly permit `player: unknown` in the existing response contract |
| Old movement routes, logs, and review flows regress | Preserve compatible formats, add v11 fixture regressions, and run both the v11 tests and frozen v10 suite |

## 11. Implementation plan

This chapter defines implementation order, affected files, tests and acceptance gates. Implement each phase in the **existing owning module**, following `AGENTS.md` and adding regression tests alongside behavior changes.

### Phase 0 — Freeze the baseline and add fixtures

**Changes:** Add deterministic fixture environments in `tests/` for (a) click-only with one valid hotspot, (b) click-only with an empty-cell target, (c) click then movement prerequisite, (d) movement then click, (e) a click that changes a remote region, and (f) a no-op at one coordinate but success at another.

**Record:** Baseline v10.0.29 test results and existing ez01–ez04 replay/solve metrics before copying. Include at least one non-square frame to expose coordinate reversals.

**Gate:** Fixtures independently establish exact legal actions, expected native-coordinate effects, reset behavior, and completion outcomes.

### Phase 1 — Normalize click actions and preserve the response schema

**Owner:** `src/v11/analysis.py`; tests `test_v11_analysis.py`, `test_v11_live.py`.

1. Add legal-gated `ACTION6` ↔ `click` mapping to naming, prompt-facing action specifications and canonicalization.
2. Ensure canonicalization works with click-only legal actions.
3. Generate legal-action examples dynamically, including required `x`/`y` fields.
4. Clarify that `state_assessment.player` may be `unknown`, retaining the compatible `v10.reasoned_action/1` response structure.
5. Check existing offline analysis paths and logged-route import with click actions.

**Gate:** Click-only and mixed decisions validate; missing/out-of-range coordinates reject; movement outputs remain unchanged; no change to the solution-store format.

### Phase 2 — Click targets and effects

**Owners:** `src/v11/tools/vision.py`, `src/v11/live.py`; vision/live tests.

1. Implement `propose_click_candidates(frame, frame_id, previous_attempts, max_candidates=12)` with bounded runtime and deterministic ordering.
2. Include small regions and fall back to spatial sampling when salient sites are insufficient.
3. Attach a size-bounded `v11.click_candidates/1` block to initialized and incremental model contexts when click is legal.
4. Optionally render a secondary indexed overlay while preserving the raw current frame.
5. Implement `v11.click_effect/1` native-frame local/remote measurements and add them to transition context without rewriting historical records.

**Gate:** All suggestions are in bounds, reproducible and frame-scoped; candidate points include a one-pixel target; local and remote delta tests pass; no image rescaling changes native coordinates.

### Phase 3 — Capability-aware bootstrap and fallback

**Owner:** `src/v11/live.py`; tests `test_v11_live.py`.

1. Preserve the present movement-only standard start routine.
2. For click-only and mixed games, request model-guided legal probes using current screenshots and candidates.
3. Replace `directional_only` and direction-filtered repair/fallback with capability-aware validation.
4. Bound exploratory bootstrap to four environment actions and preserve completion, RESET, restoration, and route-replay logic.
5. Prefer grounded untried click locations over random `(x,y)` fallback, while allowing any valid model-selected coordinate.
6. Default to a single action for unverified interactions; permit longer replayable plans after verification.

**Gate:** Click-only fixtures advance under default startup settings; a mixed fixture executes click and direction in one run; verified-route replay and existing per-level bootstrap tests remain green.

### Phase 4 — Generalize stagnation, reviews and mixed trajectories

**Owners:** `src/v11/live.py`, `src/v11/tools/solutions.py` only where required, `skills/v11/SKILL.md`.

1. Audit action comparisons for arguments: no-op tracking, equivalence candidates, recovery, replay suspension, history and summaries.
2. Treat the `(action, args)` tuple as the atomic attempted interaction in the same source state.
3. Update skill guidance to allow playerless games, clicked empty cells, remote effects, interaction prerequisites, and evidence-grounded combinations.
4. Ensure review summaries refer to exact click coordinates/regions and scoped hypotheses.
5. Add click/movement-specific metrics and include them in the existing run manifest/metrics output.

**Gate:** A failed click at one coordinate cannot exclude a different coordinate; mixed successful routes replay exactly; reviews can describe a click-enabled movement condition without claiming unsupported semantics.

### Phase 5 — End-to-end validation and release

**Owners:** existing v10 tests, docs and release metadata.

1. Run the complete v11 test suite and relevant environment adapter tests; also run the frozen v10 suite as an independent regression gate.
2. Execute deterministic click-only and mixed fixtures with model-driven selection and with cached replay.
3. Re-run ez01–ez04 movement benchmarks with comparable seed, model and budget, investigating any regression.
4. Verify evidence replay, game-over/restart behavior, action budget accounting, prompt validity and coordinate logs.
5. Update `docs/v11/DESIGN.md`, `docs/v11/README.md`, `docs/v11/IMPLEMENTATION_PLAN.md`, and release metadata for v11.1.0.

**Release gate:** All deterministic fixtures solved and replayable, zero invalid executed coordinates, existing movement suite green, and no regression in stored-route compatibility. Model-backed benchmark results (solve rate, actions and model calls) are reported separately from deterministic correctness tests.

### Implementation checklist

- [x] `click` alias and click-only canonicalization implemented.
- [x] Existing action response schema retained and prompts generalized for click coordinates.
- [x] Native-frame candidate sites generated and bounded.
- [x] Candidate context provided to the VLM; optional overlay remains deferred.
- [x] Local/remote click-effect measurements logged.
- [x] Adaptive click-only and mixed bootstrap implemented.
- [x] Coordinate-aware fallback and stagnation comparisons implemented.
- [x] Mixed click/movement route argument preservation verified in both action orders.
- [x] Complete automated v11 tests and movement regressions pass; v10 source/tests remain unchanged.
- [x] V11 runtime and package metadata report 11.1.0; v10 remains at 10.0.29.
- [ ] Model-backed movement benchmarks and live click-game runs recorded.

## 12. Out of scope for this release

Keep the initial release focused on click and mixed-action games. Full visual object tracking, semantic object ontologies, learning a universal click policy, drag/hold gestures not exposed by the environment, exhaustive pixel-search algorithms, and new agent orchestration frameworks remain later extensions. The current review cadence and model-provider architecture stay unchanged unless regression data demonstrates a concrete need.

---

**Baseline reviewed:** [`src/v10`](../v10/README.md), [`skills/SKILL.md`](../../skills/SKILL.md), and [`AGENTS.md`](../../AGENTS.md). V11 copies the source baseline only after this design is established; implementation and validation are tracked separately.
