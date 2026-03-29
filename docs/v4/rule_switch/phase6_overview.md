Status: implemented and verified
Scope: Phase 6 rule-switch overview
Source of truth: `/home/zodrak/zod/src/v4/rule_switch/*`, `/home/zodrak/zod/tests/v4/rule_switch/*`
Last verified against: unknown

# Phase 6 Overview

Phase 6 adds the exact movement-only rule-switch package for `rs01`.

Implemented scope:

- one explicit `rs01` typed-state package
- deterministic safe-color extraction from observation plus metadata
- deterministic movement-only transition modeling
- bounded exact search over legal movement actions
- Stage 2-compatible policy output
- dedicated package tests and a Phase 6 gate
