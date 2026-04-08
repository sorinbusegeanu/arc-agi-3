Status: implemented; gate-covered, live regression still mixed
Scope: `tb01` mechanics as implemented in `src/v4/hybrid_construction`
Source of truth: `src/v4/hybrid_construction/familyAdapters.py`, `src/v4/hybrid_construction/solverPolicy.py`
Last verified against: repo state on 2026-03-30; targeted manual live regression for `tb01`

# tb01 mechanics

Implemented `tb01` treatment:
- hybrid movement plus bridge-toggle family
- geometry loaded from environment metadata
- built bridges read from the current observation
- mixed primitive search across move and `ACTION6`

Current boundary:

- the manual live regression runner now classifies the family as a zero-step `worker_timeout` with `tb01_timeout_before_first_action` instead of leaving the startup timeout blank
- `tb01` is still not live-verified end-to-end
