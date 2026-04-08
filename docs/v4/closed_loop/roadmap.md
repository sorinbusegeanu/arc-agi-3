1. Certified planner
Status: implemented
2. Explicit subgoals
Status: implemented
3. Belief state
Status: implemented
4. Information-gain planning
Status: implemented
5. Hypothesis registry
Status: implemented
6. Experiment planner
Status: implemented
7. Temporal/resource model
Status: implemented
8. Compositional world model
Status: implemented
tracing exists to measure whether Step 6, Step 7, and Step 8 candidates are actually generated, accepted, and selected during real runs
subgoal activation tracing exists to measure whether Step 6, Step 7, and Step 8 subgoals are extracted, selected, and progressing during real runs
reference-population tracing exists to measure whether belief, hypothesis, temporal, and composition references are actually becoming live during real runs
builder diagnostics exist to measure whether family-specific typed-state builders and detectors are actually producing positive live signals during real runs
hypothesis-flow tracing exists to measure whether grounded builder signals are actually being converted into emitted hypotheses, retained in the registry, and exposed to subgoal extraction
9. Bounded branch-and-prune
10. Durable template library
