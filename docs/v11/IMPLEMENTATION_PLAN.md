# V11 implementation plan

The normative behavior and acceptance criteria are in [`DESIGN.md`](DESIGN.md). Complete each phase in `src/v11` and `tests/test_v11_*.py`; do not edit v10 source, tests, prompts, docs, or state as part of v11 work.

## Implementation status

- The v10.0.29 source fork, v11 CLI/runtime identity, and isolated `skills/v11` state defaults are in place.
- Click alias validation, native-grid candidate proposals, local/remote click-effect measurements, capability-aware bootstrap, click-aware stagnation, and click metrics are implemented.
- Regression coverage includes click-only startup, click argument preservation, both click/movement route orders, remote effects, and the existing movement behavior suite.
- Full v11 test suite: **242 passed** (CUDA 13 Python).
- Frozen v10 test suite: **232 passed, 4 failed**. The failures are the existing missing `runs/v10/ez01_20` fixture (2 tests) and two assertions expecting a completion review after verified route replay. V10 source/tests remain unchanged.
- Model-backed movement benchmarks and live click-game runs have not been performed; fixture correctness is separate from those runtime results.

1. **Fork baseline:** copy `src/v10` to `src/v11`, establish the v11 package/CLI/version, and isolate its skill, memory, route, run-discovery, and vLLM-log defaults. Record the baseline test results.
2. **Action contract:** add legal-gated `ACTION6`/`click` naming and canonicalization, keep coordinate arguments exact, and allow the required player assessment to be unknown.
3. **Visual support:** implement bounded deterministic click candidates and native-frame click-effect measurements; expose candidates only while click is legal.
4. **Bootstrap and fallback:** retain the movement-only routine, add capability-aware click/mixed probing, bound probes to four actions per unsolved level, and prefer grounded untried points for fallback.
5. **State-aware learning:** audit action comparisons for complete arguments, update v11 skill guidance, and add click/movement metrics and exact mixed-route replay coverage.
6. **Validation/release:** run v11 tests, run the frozen v10 suite independently, compare movement benchmarks, and synchronize v11 docs and `11.0.0` metadata. The automated suites have been run; live benchmarks remain outstanding.

The deterministic test fixtures must cover hotspot and empty-cell clicks, both mixed-action orders, remote effects, non-square frames, coordinate bounds, state-specific no-ops, reset/restoration, and exact route replay. Model-backed results are reported separately from fixture correctness.
