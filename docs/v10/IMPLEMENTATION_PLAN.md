# V10 Implementation Plan — ez01 Review and Memory

**25 October 2026 · Targets DESIGN.md v0.3**  
**Status:** Planned work; documentation changes do not implement this plan.  
**Baseline inspected:** b225637, including directional aliases on top of the d37a478 action-only runtime.

## 1. Delivery rules

Read each affected implementation, callers, and historical regression tests before editing. Search for existing equivalent helpers. Follow AGENTS.md: one owner per responsibility, explicit composition, no monkey patches or fixup modules, and regression tests for every bug fix.

Keep the live action response action-only. New structured reasoning and memory proposals belong exclusively to review sessions. Implement and test ez01 first. Do not introduce other-game requirements into its critical path.

Each phase is a reviewable commit with its validation evidence. Update docs/v10/README.md and command examples when the corresponding behavior exists. Do not label planned features implemented.

## 2. Phase 0 — Stabilize the inspected baseline

**Owners:** src/v10/analysis.py, live.py, tools/game.py, arcade_env.py, and their existing tests.

- Recheck current main before changing code; preserve the directional alias work already present.
- Fix the missing read_jsonl import used by prior-decision analysis. Reproduce the failure with an existing decisions log; verify all requested ablation arms report results rather than silently collected errors.
- Keep one authoritative directional mapping. Test all four alias-to-canonical round trips, legal-action filtering, prompts, and evidence logging. Distinguish command names from observed movement.
- Complete argument validation for supported schema types and constraints. Reject malformed schema definitions and non-finite/nested invalid values before dispatch. Reuse a full validator if already available; otherwise document the supported subset and reject unsupported constraints.
- Preserve native intermediate frames and remove the temporal-stack shape ambiguity. Test narrow palette grids separately from RGB representations.
- Add an append-only action intent before environment dispatch and outcome/error linkage afterward. Record attempted actions even if postprocessing fails.
- Make run evidence durable across restarts and cleanup; reject state/evidence directories nested inside disposable run directories.
- Make invalid-output random fallback explicit and off by default. Preserve an opt-in deterministic fallback regression test.

**Exit:** existing v10 regressions plus new reproductions pass; a short ez01 run logs aliases, canonical actions, frames, failures, and correct budgets. Deferred click support is listed explicitly.

## 3. Phase 1 — Add bounded review orchestration

**Owner:** src/v10/live.py for scheduling; a new src/v10/review.py for the new review responsibility; reuse History and StrataClient.

- Define versioned review request/response schemas and a distinct review prompt. Keep action SKILL.md concise; share game-independent review instructions without embedding ez01 mechanics.
- Add configuration for review enabled/disabled, interval 12, unchanged-frame threshold 8, batch size 32, image limit 8, and per-call limits from DESIGN.md.
- Freeze run/game/reset/level/evidence boundary and memory revision. Trigger at action boundaries; discard pending action tails when review starts.
- Coalesce triggers and prevent repeat review of the same boundary. Freeze the environment during review; review cannot issue actions or resets.
- Implement sequential evidence batches, deterministic representative frame selection, coverage records, review cursor, and bounded revisit of prior exceptions.
- Persist review requests, outputs, repair errors, skipped reasons, image IDs, summary coverage and timings in reviews.jsonl. One repair only; no memory mutation on repeated failure.
- Handle review at level transition before the next action, before reset, and at terminal/normal budget exit with remaining allowance.

**Tests:** simultaneous triggers; partial batch failure/resume; no future citations; no environment calls during review; no stale plan continuation; no duplicate review at one boundary.

**Exit:** review-only mode covers every transition through each completed boundary and leaves memory unchanged.

## 4. Phase 2 — Admit knowledge and render durable memory

**Owner:** src/v10/tools/memory.py; review.py submits validated proposals.

- Extend the existing ledger schema rather than adding a second store. Define migration behavior for legacy prediction entries and statuses; preserve old rows.
- Add scoped claims, conditions, alternatives, practical use, tests, support/contradiction references, versions, and review IDs.
- Validate patch count/field bounds and citation existence, game, chronology, level/reset metadata. Assign stable IDs in the harness.
- Implement candidate/repeated/contradicted/retired transitions and distinct-evidence counts. Supported status is unavailable until phase 4 provides a valid prospective audit.
- Merge duplicates without counting the same evidence twice. Preserve contradictions and history during revisions; do not reuse old support automatically after a claim changes.
- Serialize updates against the frozen revision and make each review/batch commit idempotent. Persist ledger changes before advancing the review cursor; recovery must reconcile interrupted writes.
- Regenerate MEMORY.md atomically. Enforce 50 active entries and the 12,000-character view limit without deleting the ledger.
- Add a review command to inspect a recorded prefix without advancing the game; default offline inspection to proposal-only, with explicit admission when intended.

**Tests:** invented/future/wrong-game citations; duplicate replay; partial-write recovery; contradictory claim updates; retirement; migration; cap handling; evidence resolution after a new run.

**Exit:** admitted review results produce readable, traceable memory with reliable recovery and bounded size.

## 5. Phase 3 — Use memory on later ez01 levels

**Owners:** live.py, analysis.py, tools/memory.py.

- Select at most five entries by current applicability, evidential status, and relevance, under the memory/prompt character caps.
- Include relevant contradictions with their claim. Treat previous-level support as untested in a new level. Exclude old transient positions from current-state facts.
- Supply current/previous actual frames, current-generation recent outcomes, and remaining budgets.
- Record the exact memory revision and selected/omitted entry IDs for each action decision.
- Keep no-memory and review-withheld modes available; isolate namespaces between evaluation arms.

**Tests:** level boundary transfer, reset isolation, stale coordinates, relevant contradiction retention, bounded long entries, deterministic selection, memory disabled.

**Exit:** a fixture demonstrates that reviewed knowledge from level one reaches level two without carrying false current-state facts. Run a retained ez01 smoke episode.

## 6. Phase 4 — Verify selected claims prospectively

**Owners:** review.py for nomination; tools/game.py for existing predicate helpers; memory.py for audit-based status updates; live.py for timing.

- Allow at most one optional pending test per review, separate from the action response.
- Record claim/version, applicability, proposed action, allowlisted predicate and boundary before the action.
- Evaluate only when the executed action and observable conditions match. A different action, uncertain identity, or missing measurement is unscorable/inconclusive.
- Promote only the exact claim/version linked to a valid later audit; preserve failures as contradictions. Past transitions cannot serve as future tests.
- Keep a predicate such as changed_pixels > 0 from proving an unrelated directional or goal claim. Require narrow claim/test alignment; semantic claims remain candidates when deterministic verification is unavailable.
- Support read-only replay of audits without multiplying evidence counts.

**Tests:** prospective timestamp/order; action mismatch; claim revision; unsupported predicates; missing data; repeated audit; contradicted support; retrospective-promotion rejection.

**Exit:** synthetic supported/failed/inconclusive cases work end to end. Real ez01 claims retain their actual evidential limits.

## 7. Phase 5 — Enforce budgets and evaluate

**Owners:** cli.py, strata_client.py, live.py, review.py, evaluation.py.

- Expose documented CLI settings for action/review caps, model attempts, cumulative reported tokens, deadline, fallback, review mode and state namespaces.
- Count transport fallback attempts and corrective completions in episode totals. Check remaining allowance before calls and between batches. Bound a call by remaining deadline where possible.
- Preserve unknown usage as unknown. Separate action/review usage, image/reasoning details when available, latency, skipped reviews, and memory counts.
- Add paired live A/B/C comparisons defined in DESIGN.md. Use independent initially empty namespaces or identical frozen snapshots; never share mutable memory between arms.
- Report actual compute and outcome differences; use offline matches only for wiring/debugging.
- Run at least three paired ez01 seeds as an initial feasibility report. Preserve evidence, MEMORY.md revisions, review logs, model/prompt versions, and configs. Report failures and overhead as well as wins.

**Exit:** the report can answer whether review-derived memory helps later ez01 levels and at what cost. Do not infer cross-game benefit.

## 8. Verification and completion

Run the existing suites:

```sh
PYTHONPATH=src pytest -q tests/test_v10_replay.py tests/test_v10_analysis.py tests/test_v10_memory_integrity.py tests/test_v10_live.py
```

Add review tests for the genuinely new responsibility and retain historical bug reproductions. Use deterministic fake environments/model responses for failure and persistence cases; validate Strata request wiring with a local mock endpoint before live inference. Live Qwen/Strata results are required for performance claims.

Completion requires aligned DESIGN.md/README.md, executable schema versions, passing relevant regressions, retained ez01 comparison artifacts, resolvable memory evidence, and no dependency on patch order.

## 9. Deferred work

After ez01 evidence supports the design: click-specific affordances, richer temporal identity, other games, calibrated experiment-utility ranking, and candidate cross-game templates. Cross-game transfer requires its own isolated evaluation and cannot promote another game's facts automatically.
