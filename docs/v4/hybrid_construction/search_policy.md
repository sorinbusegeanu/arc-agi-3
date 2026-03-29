Status: implemented and verified
Scope: search and policy for `src/v4/hybrid_construction/search.py` and `src/v4/hybrid_construction/solverPolicy.py`
Source of truth: `src/v4/hybrid_construction/search.py`, `src/v4/hybrid_construction/solverPolicy.py`
Last verified against: repo state on 2026-03-29

# Search and policy

Implemented behavior:
- mixed exact search across movement and bridge-toggle actions
- Stage 2-compatible policy output
- family-explicit bridge logic kept outside generic movement or click packages
