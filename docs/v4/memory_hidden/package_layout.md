Status: implemented and verified
Scope: package layout for `src/v4/memory_hidden` and `tests/v4/memory_hidden`
Source of truth: `src/v4/memory_hidden`, `tests/v4/memory_hidden`
Last verified against: repo state on 2026-03-29

# Package layout

Source package:
- `src/v4/memory_hidden/__init__.py`
- `src/v4/memory_hidden/typedState.py`
- `src/v4/memory_hidden/familyAdapters.py`
- `src/v4/memory_hidden/stateBuilder.py`
- `src/v4/memory_hidden/transitionModel.py`
- `src/v4/memory_hidden/heuristics.py`
- `src/v4/memory_hidden/search.py`
- `src/v4/memory_hidden/solverPolicy.py`

Test package:
- `tests/v4/memory_hidden/test_typed_state.py`
- `tests/v4/memory_hidden/test_family_adapters.py`
- `tests/v4/memory_hidden/test_transition_model_ms01.py`
- `tests/v4/memory_hidden/test_search.py`
- `tests/v4/memory_hidden/test_solver_policy_family_smoke.py`
- `tests/v4/memory_hidden/test_phase5_gate_ms01.py`
