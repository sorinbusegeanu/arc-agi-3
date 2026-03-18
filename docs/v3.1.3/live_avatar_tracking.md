`info["avatar"]` is optional and is not assumed present.

Authoritative execution-time avatar location comes from live action-and-screen-change inference in [live_avatar_tracker.py](/home/zodrak/zod/src/v3_1/execution/live_avatar_tracker.py).

Current implementation:
- env worker owns one `LiveAvatarTracker` per active episode
- route execution consumes tracker state from env-worker telemetry
- static board scan is last-resort fallback only, with low confidence
- low-confidence avatar localization weakens route certainty and downstream mechanic evidence
