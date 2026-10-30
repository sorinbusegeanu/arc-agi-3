# V11 implementation plan

The normative behavior and acceptance criteria are in [`DESIGN.md`](DESIGN.md). Complete each phase in `src/v11` and `tests/test_v11_*.py`; do not edit v10 source, tests, prompts, docs, or state as part of v11 work.

## Implementation status

- The v10.0.29 source fork, v11 CLI/runtime identity, and isolated `skills/v11` state defaults are in place. Version `11.1.0` adds exact action-set profiles and scoped profile learning.
- Version `11.1.1` adds a concise click-only decision contract, safe stop after invalid click decisions, bounded review context, failed-review backoff, and SIGTERM finalization.
- Version `11.1.2` selects a standalone skill for ACTION6-only games, includes exact click-field limits in the prompt, and measures board changes outside a versioned per-game UI mask. Raw full-frame changes remain telemetry and cannot establish board effect or progress.
- Version `11.1.3` starts click-only games from direct observation, attaches only the latest executed click batch with a compact outcome ledger, and normalizes the environment's `NOT_FINISHED` status to the prediction contract's `no_terminal` label.
- Version `11.1.4` explores up to ten independent click branches per current state, then advances hypotheses click by click as the live state changes. Click-only live batches carry per-click predictions and stop on a contradicted prediction; fresh batch contexts retain the latest screens and compact click history.
- Click alias validation, native-grid candidate proposals, local/remote click-effect measurements, capability-aware bootstrap, click-aware stagnation, and click metrics are implemented.
- Regression coverage includes click-only startup, click argument preservation, both click/movement route orders, remote effects, and the existing movement behavior suite.
- Profile implementation adds a focused suite in `tests/test_v11_profiles.py`; the full v11 suite passes **239 tests** with CUDA 13 Python.
- Frozen v10 suite: **232 passed, 4 failed** in this validation run. Failures are the missing `runs/v10/ez01_20` fixture (1), invalid historical route fixture data (1), and two pre-existing verified-route review expectations (2). V10 source/tests remain unchanged.
- Model-backed movement benchmarks and live click-game runs have not been performed; fixture correctness is separate from those runtime results.

1. **Fork baseline:** copy `src/v10` to `src/v11`, establish the v11 package/CLI/version, and isolate its skill, memory, route, run-discovery, and vLLM-log defaults. Record the baseline test results.
2. **Action contract:** add legal-gated `ACTION6`/`click` naming and canonicalization, keep coordinate arguments exact, and allow the required player assessment to be unknown.
3. **Visual support:** implement bounded deterministic click candidates and native-frame click-effect measurements; expose candidates only while click is legal.
4. **Bootstrap and fallback:** retain the movement-only routine, add capability-aware click/mixed probing, bound probes to four actions per unsolved level, and prefer grounded untried points for fallback.
5. **State-aware learning:** audit action comparisons for complete arguments, update v11 skill guidance, and add click/movement metrics and exact mixed-route replay coverage.
6. **Validation/release:** run v11 tests, run the frozen v10 suite independently, compare movement benchmarks, and synchronize v11 docs and the v11 patch/minor metadata. The automated suites have been run; live benchmarks remain outstanding.

The deterministic test fixtures must cover hotspot and empty-cell clicks, both mixed-action orders, remote effects, non-square frames, coordinate bounds, state-specific no-ops, reset/restoration, and exact route replay. Model-backed results are reported separately from fixture correctness.

7. **Action profiles and scoped learning (11.1):** resolve one of nine documented signatures or generic from runtime legal actions; compose capability prompts over the common core; isolate approved method memory by profile; queue evidence-linked candidates and require explicit cross-game, held-out promotion.
8. **Current profile implementation:** `src/v11/profiles.py`, modular prompts under `skills/v11/profile_capabilities/`, `v11 profiles` review commands, and one-time removal of legacy shared memory. Profile behavior has focused resolver, isolation, promotion, rejection, and migration tests.
