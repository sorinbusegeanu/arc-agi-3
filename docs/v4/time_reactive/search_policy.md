Status: implemented and verified
Scope: `sv01` search and policy
Source of truth: `/home/zodrak/zod/src/v4/time_reactive/search.py`, `/home/zodrak/zod/src/v4/time_reactive/solverPolicy.py`
Last verified against: unknown

# Search And Policy

Search is bounded exact search over the currently legal primitive actions for the remaining survival horizon used by the solver.

The policy:

- rebuilds typed state from the current parsed state every step
- searches for a survival-certifying bounded plan
- returns one Stage 2-compatible primitive action or short prefix
- falls back to a currently safe legal action when bounded certification fails
