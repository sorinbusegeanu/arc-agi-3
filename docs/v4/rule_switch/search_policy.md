Status: implemented and verified
Scope: `rs01` search and policy
Source of truth: `/home/zodrak/zod/src/v4/rule_switch/search.py`, `/home/zodrak/zod/src/v4/rule_switch/solverPolicy.py`
Last verified against: unknown

# Search And Policy

The current search filters out moves that would enter visible wrong-color targets and explores only bounded movement plans.

The policy:

- rebuilds typed state each step
- searches for a success-certifying safe-color plan
- returns one Stage 2-compatible primitive move or a short plan prefix
- falls back to a safe legal movement path when bounded search does not certify success
