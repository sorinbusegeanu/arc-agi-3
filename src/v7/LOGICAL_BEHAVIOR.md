# v7 logical cognition behavior

This document defines the intended logical behavior of v7 independently of storage, multiprocessing, mmap, SQLite, or other implementation details. The v7 implementation must preserve the clean-break architecture while reproducing the useful cognition loop that existed in v6.

## 1. Memory has two time scales

v7 uses two cooperating forms of memory:

1. **Worker-local short-term learning** adapts during a single game/job. It learns immediately from the interaction that just occurred and may change the next action in the same game. It is never a canonical writer.
2. **Canonical long-term memory** stores validated M1-M6 knowledge across jobs, games, epochs, and restarts. Workers read an immutable published generation and return evidence; only the canonical writer mutates long-term memory.

A sampling worker must therefore not behave as a fixed policy for an entire 1000-step job. It must combine the immutable long-term generation with a local online overlay that is updated every step.

## 2. Decision context is hierarchical, not an exact-grid identity

The exact grid remains useful as evidence/provenance, but it must not be the only decision context.

For each decision v7 builds a bounded set of context signatures from general to specific:

- **C0 action context:** candidate action only.
- **C1 structural state context:** candidate action plus an abstract state signature that is insensitive to irrelevant exact cell identity where possible.
- **C2 temporal context:** structural state plus recent transformation/outcome families and recent actions.
- **C3 specific context:** temporal context plus the exact state signature as a final specialization/fallback discriminator.

Prediction and action selection prefer the most specific context with sufficient evidence and back off to more general contexts when support is weak. Prediction violations may justify more specific context; lack of exact-state reuse must not prevent generalization.

## 3. M1 — contextual contingencies

M1 represents supported relations of the form:

`context × action -> outcome/transformation`

For each context/action relation the system tracks bounded evidence including support, outcome distribution/confidence, positive and negative terminal evidence, failure evidence, prediction violations, and observed future-option change.

M1 must influence action choice in the same context. Global action statistics are only a weak prior/fallback and must never dominate well-supported contextual evidence.

## 4. M2 — transformation families

M2 groups M1 outcomes that are functionally similar rather than requiring exact transition identity. Transformation identity should emphasize normalized changed-cell geometry, before/after relation, direction/shape/color-change structure, and other bounded structural features while remaining insensitive to irrelevant absolute placement when appropriate.

M2 supplies outcome families used by prediction and temporal context.

## 5. M3 — functional roles

M3 represents recurring functional relations between a transformation family, action, carrier/state context, and observed effect. A role should become stronger when it recurs across distinct contexts and improves prediction/compression; it is not merely a renamed `(family, action)` pair.

Roles are indexed by context/action/family and directly contribute to prospective action scoring.

## 6. M4 — concepts and transfer

M4 binds compatible roles into reusable concepts. Concepts are useful only when they explain multiple lower-level memories and/or transfer beyond their source context/game.

Concept evidence must contribute to action choice when the current context matches its supported role structure. Transfer evidence is retrospective and must not be invented from the prospective transfer prior.

## 7. M5 — world model and future reachability

M5 represents learned action-conditioned transitions among abstract states/concepts. It is used prospectively, not only reported after the run.

Future-option value is based primarily on learned reachable future states/options over a bounded depth. Simple counts of currently indexed memories are not a substitute for reachability. When the learned transition graph lacks evidence, local action availability and contextual evidence may provide a bounded fallback estimate.

M5 contributes to action ranking through predicted reachability, progress/completion evidence, and failure/terminal risk.

## 8. M6 — successful strategies

M6 stores successful multi-step behavior, not only a representative action. A strategy contains a bounded action sequence/prefix, precondition/context signature, observed effect/success target, cost/length, reuse count, and failure count.

When a current context matches a strategy precondition, the next strategy action receives a prospective bonus. Reuse outcomes update strategy confidence; repeated failed reuse reduces or suppresses the strategy. Better/shorter successful trajectories strengthen efficient strategies.

## 9. Action decision loop

For every step:

1. Observe state and available actions.
2. Build multi-scale contexts using structural state plus local temporal history.
3. For every candidate action obtain contextual M1 evidence, matching M3/M4 abstractions, M5 reachability evidence, M6 strategy-prefix evidence, and worker-local evidence.
4. Score actions using positive prediction/completion/future-option/transfer/strategy evidence and negative failure/contradiction/stagnation evidence.
5. Use global action statistics only as a low-weight prior when contextual evidence is insufficient.
6. Preserve exploration: uncertainty, novelty, action balance, no-change avoidance, and small epsilon exploration prevent premature policy collapse.
7. Execute the action.
8. Immediately update the worker-local overlay, temporal context, transition graph, failure/no-change statistics, and trajectory state.
9. Emit evidence for canonical long-term learning.

Memory may override exploratory preference only when contextual evidence is materially stronger. Unsupported memory should not create deterministic exploitation.

## 10. Stagnation and reset behavior

The worker tracks recent no-change transitions, repeated outcome families, repeated states, and repeated failures. Persistent stagnation increases exploration and may trigger an environment reset when supported. A reset clears episode-local temporal context while retaining learned local statistics for the job.

## 11. Learning feedback loop

After each outcome:

`decision -> outcome -> prediction error -> local update -> next decision`

After each canonical wave:

`episode evidence -> M1 -> M2 -> M3 -> M4 -> M5 -> M6 -> published generation -> later decisions`

Higher memory levels must therefore feed back into behavior. It is invalid for M5/M6 to be created successfully while having no effect on action selection.

## 12. Generation semantics

All workers in one sampling wave may read the same immutable canonical generation. That does not remove within-game learning because each worker has its own short-term overlay. Canonical evidence is deterministically merged by the single writer and becomes visible to subsequent published generations/waves.

## 13. Required behavioral properties

The implementation is conformant only if all of the following are true:

- contextual evidence can change the selected action relative to global action statistics;
- a worker can learn within one game/job and change later actions before the next canonical generation;
- exact-grid mismatch does not prevent reuse of supported abstract context knowledge;
- prediction backs off from specific to general context when support is insufficient;
- future-option scoring uses learned transition reachability when available;
- M3/M4 evidence affects prospective action scores;
- M5 world-model evidence affects prospective action scores;
- M6 strategy prefixes affect prospective action scores and record reuse success/failure;
- negative/failure/contradiction evidence can suppress an otherwise globally popular action;
- exploration includes uncertainty/novelty/no-change/action-balance signals, not epsilon alone;
- all canonical mutation remains single-writer and deterministic;
- worker-local learning never mutates the shared canonical generation.
