Status: implemented and verified
Scope: agent contracts doc: terminal signal
Source of truth: `/home/zodrak/zod/src/v4/agentContract/*`, `/home/zodrak/zod/tests/v4/agentContract/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# v4 Terminal Signal

## Scope

`v4` terminal handling is derived only from the local engine `GameState` values returned on the observation.

Observed local `GameState` values are:

- `NOT_PLAYED`
- `NOT_FINISHED`
- `WIN`
- `GAME_OVER`

## Terminal Success Signal

Terminal success is signaled only when the returned observation has:

- `state == WIN`

## Terminal Failure Signal

Terminal failure is signaled only when the returned observation has:

- `state == GAME_OVER`

## Non-Terminal Intermediate Signal

Non-terminal intermediate state is signaled when the returned observation has:

- `state == NOT_FINISHED`

`NOT_PLAYED` is a valid observed state but is treated as pre-play or reset state, not as successful progress.

## Episode-End Versus Level-End

The local engine exposes a `full_reset` flag on the returned frame and internally distinguishes full reset from level reset when handling `RESET`.

The frame contract does not expose a separate terminal enum for “level complete but episode continues.” `v4` therefore treats only `WIN` and `GAME_OVER` as authoritative terminal outcomes, while preserving `full_reset` as reset-result metadata when the environment reports it.

## Truncation Or Time Limit

No separate truncation or time-limit signal is part of the observed local `FrameDataRaw` contract.

`v4` does not invent truncation flags unless a wrapper explicitly exposes one as direct environment output.

## Reset-Required Rule

After an observation with:

- `state == WIN`, or
- `state == GAME_OVER`

the environment requires reset before a new episode can meaningfully continue. Local wrapper examples follow this rule by checking terminal state and then resetting before continuing.

## Step-Result Recording Rule

Every step result must record terminality using the raw environment `state`, not a score heuristic.

At minimum it must capture:

- raw state before
- raw state after
- derived terminal signal
- whether reset is required

## What Is Not Allowed

- Inferring terminality from score changes alone
- Inferring terminality from planner beliefs
- Inferring terminality from POI completion
- Treating `NOT_PLAYED` as success or failure without a matching raw terminal state
- Inventing truncation or timeout terminal categories unless directly exposed by the environment
