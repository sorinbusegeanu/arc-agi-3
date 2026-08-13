# v7 logical behavior implementation review

Reviewed against `src/v7/LOGICAL_BEHAVIOR.md` after implementation.

| Design requirement | Implementation | Result |
|---|---|---|
| Worker-local learning plus immutable canonical long-term memory | `environment/cognition.py`, `environment/online_sampling.py`, existing single-writer runtime | CONFORMANT |
| General-to-specific C0-C3 context with backoff | `DecisionContext`, `LocalCognitionOverlay.build_context`, `ContextualActionScorer` | CONFORMANT |
| Exact-grid identity must not dominate reuse | structural and temporal contexts are always learned; C3 exact specialization is retained for contradiction or terminal evidence | CONFORMANT |
| Contextual M1 contingencies affect action choice | multi-scale M1 ingestion in `derivation/pipeline.py`; context-first scoring in `environment/cognition.py` | CONFORMANT |
| Global action statistics only weak fallback | contextual evidence dominates; global aggregate is low-weight prior | CONFORMANT |
| Functional M2 transformation abstraction | translation- and color-renaming-invariant transformation-family signature | CONFORMANT |
| M3 role identity includes context and affects behavior | canonical role key is family + action + context; role is prospectively indexed | CONFORMANT |
| M4 concepts affect behavior and empirical rejection is respected | concepts are prospective evidence; transfer-rejected concepts are filtered from read-side cognition | CONFORMANT |
| Future-option logic uses learned transitions rather than memory density | local learned transition graph supplies bounded reachability; canonical M5 retains cross-job transition evidence | CONFORMANT |
| M5 world models affect action choice | M5 memories are indexed by source context/action and scored prospectively | CONFORMANT |
| M6 represents and reuses successful multi-step behavior | bounded action/context sequences are persisted, derive M6, and index strategy steps prospectively | CONFORMANT |
| M6 reuse success/failure changes confidence | terminal decisions referencing M6 update strategy scores | CONFORMANT |
| Failure and contradiction evidence suppress actions | both are direct negative action-score terms | CONFORMANT |
| Rich exploration | action balance, no-change avoidance, uncertainty, novelty, epsilon and softmax exploration | CONFORMANT |
| Stagnation/reset handling | no-change, repeated-state and repeated-failure detection | CONFORMANT |
| Immediate decision-outcome-learning feedback | worker overlay updates after every step before the next decision | CONFORMANT |
| Higher memory levels feed back into behavior | M3/M4/M5/M6 are prospective action evidence | CONFORMANT |
| Single canonical writer | workers return evidence only; parent performs canonical mutation/commit | CONFORMANT |
| Immutable parallel read generations | shared view remains immutable; worker-local overlay never mutates it | CONFORMANT |

## Corrections found during review

1. Generic scientific M3 derivation still used a context-free key. It now includes `context_class`.
2. Generic scientific M1 significance treated terminal failure as ordinary non-terminal evidence. Terminal polarity now distinguishes positive, negative and neutral evidence.
3. Exact C3 context was initially persisted on every step. Context evidence now limits exact-state specialization to prediction violations or terminal evidence while retaining exact identity as provenance.
4. Transfer-rejected M4 concepts could remain in read-side cognition. `MemoryReadView.score_inputs` now suppresses them before action scoring.

No remaining logical-design differences were identified in the reviewed production path.
