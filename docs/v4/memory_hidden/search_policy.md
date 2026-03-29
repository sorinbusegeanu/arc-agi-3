Status: implemented and verified
Scope: search and solver policy for `src/v4/memory_hidden/search.py` and `src/v4/memory_hidden/solverPolicy.py`
Source of truth: `src/v4/memory_hidden/search.py`, `src/v4/memory_hidden/solverPolicy.py`
Last verified against: repo state on 2026-03-29

# Search and policy

Implemented behavior:
- exact BFS/A* over the currently known safe region
- prefer a proven safe path to the goal when the goal is already inside known-safe space
- otherwise choose a locally consistent frontier target that is not contradicted by current count constraints
- fail closed when no safe or locally consistent frontier move exists
