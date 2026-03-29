Status: implemented and verified
Scope: `tb01` mechanics as implemented in `src/v4/hybrid_construction`
Source of truth: `src/v4/hybrid_construction/familyAdapters.py`, `src/v4/hybrid_construction/solverPolicy.py`
Last verified against: repo state on 2026-03-29

# tb01 mechanics

Implemented `tb01` treatment:
- hybrid movement plus bridge-toggle family
- geometry loaded from environment metadata
- built bridges read from the current observation
- mixed primitive search across move and `ACTION6`
