Background space classification in v1 is deterministic and advisory only.

- The adapter derives `traversable`, `blocking`, and `unknown` regions from the current frame.
- These outputs are stored only in `SceneSummary.raw_observation_payload`.
- Discovery Agent and Planner Agent may read them as advisory inputs.
- They are not authoritative runtime truth and do not change control authority.
