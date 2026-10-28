# V11 Design Addendum — Action-Set Profiles, Prompt Layers, and Scoped Global Memory

**Status:** Proposed v11 extension  
**Scope:** Action-set classification, prompt composition, scoped global memory, profile experimentation, and compatibility  
**Basis:** The repository's 249-entry ARC interactive game catalog and the previously discussed v11 all-action harness  
**Integration note:** The public repository branch inspected contains v10, not v11. Names below are *proposed logical interfaces*, not assertions about existing v11 source files.

## A1. Decision and objective

Keep **one v11 agent/harness implementation** that supports every legal ARC action. Introduce a **profile** for each distinct action-set combination. At game initialization, the harness identifies the available action IDs, resolves an exact action-set profile, and assembles its prompt and memory context. Game-specific memory remains attached to the game and is not shared across games merely because their action sets match.

This is **not** nine independent agents, nine code forks, or nine distinct cognitive pipelines. A profile is a configuration of instructions, optional tools, exploration defaults, and shared *method* memory. The user can develop and evaluate each profile one at a time without affecting the others.

### Invariants

1. **The environment is authoritative:** determine actions from the runtime's actual legal-action specifications, not game names, catalog descriptions, or a model's guesses.
2. **One execution contract:** existing v11 action validation, frame capture, hypothesis testing, replay, review, and persistence remain shared.
3. **No semantic presumption:** `ACTION1–4` may move a player, move a cursor, select options, or be no-ops; `ACTION5` is context-specific; `ACTION6` is coordinate-based if specified by the environment; `ACTION7` may mean undo but is not assumed universally.
4. **Scoped global guidance:** a profile's global memory records transferable *methods*, never unverified game-specific facts.
5. **Game memory isolation:** previously learned mechanics, state, route solutions, and hypotheses stay game-specific; existing evidence and solution formats remain readable.
6. **Incremental learning:** shared profile guidance changes only through explicit, auditable promotion from multiple independent game runs; it is not casually overwritten by a single review.
7. **Experiment reproducibility:** each model call records exact profile, prompt and memory revisions, legal-action signature, and game-memory revision.

## A2. Nine action-set profiles

The following counts are **catalog-documentation counts**, not audited runtime availability. Exact sets are formed from action IDs, not their perceived meanings.

| Profile ID | Documented available actions | Games | Scope |
|---|---|---:|---|
| `a1234` | 1, 2, 3, 4 | 103 | Four directional/discrete controls; learn their effect |
| `a12346` | 1, 2, 3, 4, 6 | 43 | Four controls plus parameterized spatial click |
| `a12345` | 1, 2, 3, 4, 5 | 33 | Four controls plus a context-specific interaction |
| `a6` | 6 | 31 | Coordinate interaction without non-click controls |
| `a56` | 5, 6 | 19 | Context-specific interaction plus spatial click |
| `a123456` | 1, 2, 3, 4, 5, 6 | 15 | Full regular action set |
| `a5` | 5 | 2 | One non-coordinate game-specific action |
| `a1235` | 1, 2, 3, 5 | 2 | Three discrete controls plus interaction |
| `a1234567` | 1, 2, 3, 4, 5, 6, 7 | 1 | Full set plus game-specific ACTION7 |
| **Total** | | **249** | |

**Important:** The catalog includes games where 1–4 are declared but are explicit no-ops. Such games still use the matching *action-interface* profile. After empirical tests, the agent can attach a **provisional behavioral tag** (`spatial_move`, `cursor_move`, `selection`, `no_visible_effect`, `unknown`) to each action. Behavioral tags affect subsequent strategy, but **do not change the canonical action-set profile**. If runtime legality genuinely changes across levels or states, recompute the signature, switch the active profile as required, and preserve the same game memory.

For missing/unrecognized runtime signatures, use `generic` as a safe fallback; never force an incompatible known profile. Treat `RESET` as a separate harness lifecycle operation unless the environment explicitly publishes it as an action ID. Do not conflate legal action IDs with temporary action *applicability* (for example, a button that is legal but ineffective in a particular state).

## A3. Prompt assembly

Construct every action-selection prompt from layered components in this order:

```text
1. CORE INSTRUCTIONS                   one shared v11 reasoning/execution contract
2. ACTION-PROFILE INSTRUCTIONS          exact action-set profile + capability modules
3. SCOPED GLOBAL MEMORY                transferable profile-specific methods only
4. GAME MEMORY                         existing game-specific hypotheses/mechanics/solutions
5. CURRENT CONTEXT                     current frame(s), deltas, recent actions, legal specs
6. RESPONSE CONTRACT                   existing structured v11 decision schema
```

**Design recommendation:** Maintain nine independently versioned profile entrypoints but **compose them from reusable capabilities**, rather than copying a long prompt nine times. Primitive modules may include `discrete_control`, `context_action5`, `coordinate_action6`, `reversible_action7`, and `cross_action_dependency`. These modules prescribe *how to test* controls, not what controls must mean. For a click-only profile, no avatar/pathfinding or direction-planning guidance is loaded. For `a12346`, add experiments that determine whether click changes movement accessibility and vice versa. For `a56`, explain how to isolate click effects from ACTION5 effects. For `a1234567`, include a controlled reversal/undo experiment, without assuming undo is always safe.

Example `a6` prompt obligations:

- Use the provided image dimensions and legal coordinate bounds; `x` is column, `y` is row, according to the adapter contract.
- Propose **visible target points** and, when justified, empty-region candidates. Label these as *candidates* rather than inferred clickable objects.
- Before a diagnostic click, state a falsifiable expectation; after the click, inspect **local and remote** changes and progress indicators.
- Treat a click with no immediate pixel change as inconclusive about hidden effects unless relevant evidence rules those out.
- Avoid repeating the same coordinate in an equivalent observed state without new justification.
- Keep speculative click sequences short; extend only when intermediate effects have been verified.

The profile layer must not change the established output schema. Legal action IDs and coordinate requirements come from the adapter and are validated by the harness, independent of prompting. Any action alias is display-only and resolves back to the canonical action ID.

## A4. Memory model and ownership

Separate **three scopes** with explicit write permissions:

| Layer | Scope | Typical content | Writer |
|---|---|---|---|
| `core` | All games/profiles | Evidence hygiene, experiment design, prediction/verification discipline | Maintainer-controlled; rarely revised |
| `profile` | One exact action-set signature | Reusable strategy patterns, control-testing tactics, known pitfalls, prompt experiment findings | Offline/profile review with promotion gates |
| `game` | One game identity (and scoped level/run data) | State, observed mechanics, contradictions, failed tests, successful plans/paths, exact evidence links | Existing v11 game-review/memory process |

### Profile global memory

A profile memory file is a **compact, learned method guide**. It can recommend, for instance, testing a second click on a similar target after a positive effect, or probing whether ACTION5 advances time before treating it as an interaction. It must not state that an arbitrary color means a switch, that a coordinate is universally correct, or that a named game's mechanic applies to all games in its profile.

Proposed logical storage:

```text
skills/
  SKILL.md                       # existing universal method; keep brief
  global/MEMORY.md               # existing shared general guidance, if retained
  profiles/
    a1234/{PROMPT.md,MEMORY.md}
    a12346/{PROMPT.md,MEMORY.md}
    a12345/{PROMPT.md,MEMORY.md}
    a6/{PROMPT.md,MEMORY.md}
    a56/{PROMPT.md,MEMORY.md}
    a123456/{PROMPT.md,MEMORY.md}
    a5/{PROMPT.md,MEMORY.md}
    a1235/{PROMPT.md,MEMORY.md}
    a1234567/{PROMPT.md,MEMORY.md}
    generic/{PROMPT.md,MEMORY.md}
  games/<game-id>/MEMORY.md      # existing; no migration required
```

Paths are illustrative; adapt them to v11's existing memory API and directory conventions. Avoid copying old universal global-memory advice verbatim into every profile. On startup and each review, the system loads only core + active profile + current game content. Full histories remain in durable storage, not the prompt.

**Profile memory admission policy:** A game review may propose a transferable lesson, but it enters a *candidate queue*, not `profiles/<id>/MEMORY.md`. Promotion requires: (a) a method-level claim, (b) linked transition evidence, (c) support from **at least two distinct games** with the same signature, (d) no unresolved counterevidence, and (e) a measurable improvement in held-out games or controlled ablation. A human approval path is acceptable initially. Contradictory results revise or retire the claim; source game IDs remain in provenance metadata and are not included in ordinary prompts.

**Prevent contamination:** In experiments, never feed a held-out game's results into profile memory before measuring its performance. Keep a separate baseline profile revision for A/B comparisons. Do not transfer game memory through a profile file, even when multiple games have nearly identical visuals.

### Context budgets

Limit each layer independently (configurable token caps); the current game state and legal-action specification take priority over recalled general guidance. Deduplicate overlapping core/profile advice. If the context exceeds the budget, compress/archive profile guidance before dropping current observations or critical game-specific contradictions. Instrument actual prompt token counts instead of assuming profile specialization saves tokens.

## A5. Profile selection and lifecycle

```text
Game start / level change / legal-action change
    -> read canonical current legal-action specs
    -> normalize, sort and hash action IDs (+ argument schema fingerprint)
    -> resolve exact profile ID or `generic`
    -> assemble core + profile + game + observation context
    -> make a bounded plan using the existing v11 action response contract
    -> validate and execute through shared v11 tools
    -> compare predicted vs observed effect
    -> update game memory via the existing review mechanism
    -> optionally record a transferable-method candidate for offline review
```

A profile signature has two parts: (1) the sorted legal action IDs, which determines the profile, and (2) a fingerprint of parameter requirements and bounds for reproducibility. Do not create a new profile just because the coordinate range changes with image dimensions. When profile switching occurs mid-run, log the old/new ID and avoid resetting game memory or experiment state automatically.

**Exploration safeguards:** Probe action meanings from observed evidence; never assume ACTION1–4 move a player or ACTION5 waits. Bound coordinate proposals and unique attempted `(state fingerprint, action ID, arguments)` tuples. For mixed profiles, explicitly test cross-action dependencies only after each action family's effect has been sampled enough to justify the test. For reversible controls, verify restoration rather than presuming it.

## A6. Experiment program: one profile at a time

Keep all nine profiles implemented as selectable configurations, but optimize and validate them sequentially. Use a fixed control harness and model settings when comparing prompt versions.

**Suggested sequence:**

1. `a1234` — freeze existing movement-only performance as the regression anchor; retain previously solved `ez01–ez04`.
2. `a6` — prioritize next because it isolates click-target discovery, coordinate accuracy, and remote-effects reasoning.
3. `a12346` — test click/movement dependencies and whether the same prompt components compose cleanly.
4. `a12345` and `a56` — develop unknown ACTION5 semantics and disambiguating probes.
5. `a123456` — test interactions among all three regular capability families.
6. `a5`, `a1235`, `a1234567` — evaluate low-frequency controls and fallback behavior.

For each profile, keep three prompt variants initially: (A) compact direct observation/action; (B) explicit hypothesis/test; (C) bounded look-ahead with verification. Change **one layer at a time**: prompt instructions first, then profile global memory, then exploration defaults. Never change game memory between paired comparisons unless resetting it identically for both arms.

**Metrics:** games won, levels cleared, total actions, distinct diagnostic actions, resets, invalid-action rate, invalid-coordinate rate, no-op rate, verified predictions, repeated equivalent attempts, model calls, prompt/completion tokens, latency, and replay success. Report *per-profile* aggregates and per-game outcomes. Compare to a common v11 prompt/no-profile control using identical seeds, starting states, budgets, and model parameters. Test on previously unseen games before updating profile memory.

## A7. Implementation plan

| Phase | Change | Acceptance criteria |
|---|---|---|
| 0. Freeze baseline | Record current v11 wins, prompt revisions, game memories, action metadata, fixed seeds | Reproducible baseline; no silent memory updates during trials |
| 1. Add profile resolver | Normalize exact legal IDs, implement nine registered profiles and safe `generic` fallback | All 249 catalog signatures map; unexpected runtime signature falls back; changes re-resolve |
| 2. Add prompt composition | Split existing instructions into shared core and profile-specific capability templates | No duplicated full agent prompts; output schema/actions unchanged; profile-only instructions are present |
| 3. Add profile-global memory | Resolve scoped memory alongside existing game memory, caps and provenance | No memory cross-talk; old game memories and replay files remain valid |
| 4. Add promotion workflow | Review suggestions become queued, evidence-linked candidates; controlled promotion/retirement | Single-game speculation cannot change shared global guidance |
| 5. Add experimental runner | Lock prompt/memory versions, run profile-level A/B/C evaluations, collect metrics | Comparative report by profile and game, no held-out contamination |
| 6. Roll out sequentially | Optimize `a6` first after freezing `a1234`, then mixed profiles | Movement regression maintained; each profile has verified wins and measured incremental benefit |

### Minimum automated tests

- Each of the nine documented action signatures maps to the correct profile. `ACTION7` does not disappear when present.
- Action order changes do not alter profile resolution; changing actual legal actions does.
- `ACTION6` coordinate argument bounds remain authoritative and visible in assembled context.
- Click-only prompts contain no movement/assumed-avatar instructions; ACTION5-only prompts do not invent coordinates.
- Profile switching preserves game memory, history and replay; it loads the appropriate profile-global memory.
- Game A evidence cannot alter Game B memory; an unapproved method suggestion cannot alter profile memory.
- Prompts stay under configured budgets; logging records profile ID and revisions for every decision.
- Existing v11 movement, solution replay, reset, and action-validation regression suites still pass.

## A8. Non-goals and open design decisions

**Non-goals:** multiple separate agent runtimes, hardcoded game-specific rules, a new memory engine, catalog-driven cheating, automatic transfer of solution paths, nine independent tool sets, or large semantic reclassification of games before acting.

**Decision to defer:** Should profile-level global memory become automatically writable after sufficient evidence? Start with **candidate generation + explicit approval**. Automatic promotion can be enabled later only if cross-game held-out gains and rollback mechanisms are demonstrated.

**Summary:** V11 remains one universal harness. Its action interface determines one of nine profiles, and each profile supplies a dedicated prompt and transferable method-memory layer. Game memory remains the same isolated game-level artifact. Evaluate profiles separately, beginning with click-only, while retaining the existing movement baseline.
