# V10 Reasoning and Evidence Improvements

**Status:** Implemented in the current V10 source, with runtime evaluation pending.

This plan records the approved V10 changes. It supersedes earlier proposals in this file that called for action-only migration, numeric confidence, bounded review image selection, or stopping plans on a prediction mismatch.

## Approved behavior

### 1. Structured action decisions

The action model returns one `v10.reasoned_action/1` JSON object containing:

- A concise current-screen assessment: player location, target candidates, obstacles, and relevant changes.
- An optional falsifiable hypothesis with categorical uncertainty (`tentative` or `uncertain`).
- A short trajectory rationale.
- A categorical prediction for the first action, where testable.
- One canonical `actions` list, validated with the existing legal-action, reset, plan-horizon, and budget rules.

No numeric confidence or chain-of-thought is requested. Python executes only validated `actions`; reasoning fields cannot issue additional actions. There is no compatibility migration for action-only model responses: one repair is attempted. Live execution then uses its existing legal-action fallback; offline analysis reports the invalid response and does not score it.

The same contract is used by live action selection and offline analysis. The existing CLI maximum output-token setting remains in force.

### 2. Prediction evidence

For an uncertain decision with a testable prediction, Python joins the response to the first executed transition and appends one run-local `reasoning.jsonl` record. It includes the request/decision ID, action and screen references, prediction, measurable outcome, and categorical comparison (`supported`, `contradicted`, or `inconclusive`). Semantic hypotheses remain unresolved until review.

Prediction mismatches are diagnostic only. They do not stop or shorten an otherwise valid action plan and do not create an extra model call.

### 3. Review and durable memory

Reviews receive the existing game/global Markdown, new observations, full chronological action/screen history including intermediate frames, and verified route outcomes with level scope. One review request contains the complete sequence; there is no image cap, omission, or evidence batching.

Review instructions require concise, scoped memory that distinguishes:

- Established mechanics directly supported by transitions.
- Failed hypotheses contradicted by experiments.
- Unresolved hypotheses.
- The next feasible, informative experiment on an unsolved level.
- Verified successful strategies on completed levels.

Global memory is limited to reusable methods. A global mechanic requires direct evidence from at least two distinct games; repeated observations in one game do not count as independent support. Review output remains the exact two-field JSON contract, with one repair and atomic replacement preserving prior memory on failure.

### 4. Conservative visual similarity

Strict state fingerprints continue to govern route matching, replay, bootstrap restoration, and exact no-op safeguards. A separate perceptual comparison may identify sparse screen differences as a low-certainty exploration/stagnation signal. It cannot block actions, authorize replay, or trigger a reset. Candidate comparisons and later same-action divergences are logged in run-local `state_equivalence.jsonl`.

### 5. Validation and evaluation

Focused tests cover schema validation and repair, correct action mapping, prediction logging without plan interruption, screen/action provenance, review memory requirements, and the isolation of perceptual similarity from strict replay and action safety.

The approved model comparison is limited to **UL01, one seed, and a 300-action budget per arm**. Compare the existing V10 baseline, A+B, and A+B+C+D+E with matched model settings, starting memory, and isolated stores. Report levels completed, actions to completion, repeated ineffective actions, review cost, model calls/latency, and whether a saved strategy was reused. The comparison is diagnostic, not a claim of general solving improvement.
