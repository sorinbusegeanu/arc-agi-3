Discriminating_Test_Selector
0) Scope and non-goals

Scope: choose the next probe action that maximally distinguishes hypotheses using deterministic disagreement / elimination scoring, including coordinate proposals for ACTION6.
Non-goals: maintaining hypothesis confidence (engine does that), executing actions, learning/training.

1) Inputs and data contracts
1.1 Required inputs

hypotheses[] from Executable_Hypothesis_Engine

current_state_features from FP_Analyst (objects/masks/hotspots) 

02_Simple_Explorer

action_schema + available_actions (meta)

frontier snapshots (optional but supported) 

02_Simple_Explorer

1.2 Optional inputs

full_explorer coordinate heuristics (if already computed)

memory_evidence (future; not required for Gap 3)

2) Outputs (stable + machine-readable)

Return a single object:

selected_test:

action_sequence[] (length 1 by default; max 2)

for coord actions: explicit (x,y)

score_breakdown:

disagreement_score

elimination_score

loop_risk_penalty (if frontier/history provided)

alternatives_topM[] (deterministic ordering)

3) Candidate generation
3.1 Simple actions

All currently available non-coord actions in schema order. 

02_Simple_Explorer

3.2 Coordinate actions (ACTION6)

Generate a deterministic shortlist coords_topK (default K=8) using:

object centroids / bbox corners / boundary frontier cells (from FP_Analyst)

grid corners/edges

last-change bbox focus point (if available)

optional: Full_Explorer’s top coord suggestions if present (must be stable ordering)

No scanning loops beyond K.

4) Disagreement scoring (deterministic)

For each candidate action a (and each coord variant):

ask hypothesis engine to produce predicted_event_h(a) for top-K hypotheses

compute:

signature entropy (top-1 predicted signature)

noop split rate

delta-bin variance

meta-delta disagreement (available_actions/terminal/reward if predicted)

Disagreement score is the weighted sum.

5) Elimination scoring (deterministic)

Approximate version-space elimination without probabilistic inference:

count how many hypotheses would be “highly inconsistent” with at least one plausible observed signature outcome under this test (using each hypothesis’ hard constraints)

prefer higher elimination in ties.

6) Tie-breaking and safety

Stable sort by:

disagreement_score desc

elimination_score desc

loop_risk_penalty asc

action_key asc, then (y,x) asc

7) Interfaces and integration points

Expose:

select_test(hypotheses, fp_current, action_schema, cfg, ctx) -> TestSelectionReport

propose_coords(fp_current, cfg) -> coords[]

Consumers:

Orchestrator uses this when a disagreement is open or when probe budget remains. 

09_Swarm_Orchestrator_meta

Planner may request this in “info gain mode” early. 

00_agent_catalogue

8) Configuration (explicit defaults)

topK_hypotheses_used = 6

coord_topK = 8

max_action_sequence_len = 1

weights:

w_sig_entropy = 0.55

w_noop_split = 0.20

w_delta_var = 0.15

w_meta_disagree = 0.10

9) Logging and failure handling

If no hypotheses available: emit a generic probe (ACTION1, else first available simple action; else first coord proposal).

If only 1 hypothesis: prefer probes that maximize state novelty (use frontier if present).

10) Deliverables Codex should implement (files/classes)

Discriminating_Test_Selector implementation

dataclasses: TestSelectionReport, CandidateAction, CoordProposal

minimal CLI:

--agent test_selector --hypotheses <...> --fp <...> --outdir <...>
