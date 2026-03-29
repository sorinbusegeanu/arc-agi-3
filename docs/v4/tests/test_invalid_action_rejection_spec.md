Status: implemented and verified
Scope: tests doc: test invalid action rejection spec
Source of truth: `/home/zodrak/zod/tests/v4/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# Invalid-Action Rejection Test Spec

The tests must cover:

- action id not exposed by the current observation `available_actions`
- malformed payload for a primitive action
- coordinate payload on an action that does not support it
- missing required payload when the real action type requires it
- illegal action submitted through the Stage 2 loop path
- explicit, source-attributed contract validator rejection

The tests must assert that:

- no transition record is created as if execution succeeded
- no step result claims success for a rejected action
- failure is localized to the action-selection or action-validation bucket
