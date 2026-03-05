## Refactored A–E plan aligned to current codebase

You already have:

* **A (encoder):** `observation_encoder.py` (`ObservationEncoder.encode -> z_t`)
* **memory:** `recurrent_memory.py`
* **controller:** `hierarchical_controller.py` (mode sampling)
* **actor/value:** `policy_actor_value.py` / `policy_value_heads.py`
* **reward shaping:** `reward_shaper.py`
* **collector/trainer wiring:** `rl_agent.py`, `trainer.py`

So the plan should **extend**, not replace.

---

# A) Perception: define a canonical “frame_core” and expose it everywhere

### A1. Canonical view of observation (single source of truth)

* Add a helper in **`observation_encoder.py`** (or a small new utility module) that produces:

  * `frame_core_uint8`: HUD-masked 2D/3D representation (whatever you currently hash on)
  * optional `frame_core_downsampled` for speed (e.g., 64×64)
* Return it inside `ObservationEncoder.encode(...)` output dict (alongside `z_t` and `obs_norm`).

Reason: intrinsic modules must use the same masking/canonicalization as reward hashing; otherwise they learn HUD noise.

### A2. Rollout storage contract

* In the rollout step records (collector output), store *either*:

  * `frame_core_uint8` (compressed) OR
  * a small `frame_core_hash` + `z_t` + intrinsic stats
    Prefer storing **`z_t`** and `frame_core_hash` to avoid huge I/O.

Where: **collector codepath referenced by `RolloutCollector` used in `rl_agent.py`**.

---

# B) Dynamics/world model (optional) — defer until after intrinsic is stable

You already have a recurrent core (GRU) which can serve as the “dynamics memory”.
If you add a world model, make it **embedding-space**, not pixel-space.

### B1. Forward dynamics head

* New module file (recommended): `intrinsic_dynamics.py`
* Input: `h_t` (or `z_t`) + `action_key`
* Output: predicted next embedding `ẑ_{t+1}` (or `ĥ_{t+1}`)
* Loss computed in trainer using stored `(z_t, a_t, z_{t+1})`

Where to integrate:

* instantiate in **`rl_agent.py`** under `modules.intrinsic` gate
* train in **`trainer.py`** alongside PPO losses (separate optimizer group or shared optimizer, configurable)

This enables controllability reward later, but do not block on it.

---

# C) Meta-reward: replace “pixel change” with learned intrinsic + keep your current hash signals

You already moved toward hash/potential. Next step is to make novelty **learned and smooth**.

### C1. Start with RND (fastest to implement in your stack)

* New module file: `intrinsic_rnd.py`

  * `target_net` (frozen random)
  * `predictor_net` (trainable)
  * both consume `frame_core` OR `z_t` (choose one; see below)

Preferred input given your code:

* Use **`z_t`** from `ObservationEncoder` as RND input (cheap, consistent, already masked indirectly).
* If `z_t` still includes HUD leakage via `meta_vector`, then feed RND with `grid_embed` only.

### C2. Intrinsic scalar per step

* `rnd_err = ||pred(z) - target(z)||^2`
* normalize with running mean/var (like you do in `obs_norm_v1.py` style; create `intrinsic_norm_v1.py` or reuse pattern)

### C3. Reward composition (per-step)

* Keep terminal rewards as-is.
* Replace/attenuate current novelty/potential with:

  * `r_intr = clip(norm_rnd_err, 0, r_intr_clip)`
* Keep your **hash-change** reward small as a stabilizer (optional).

Where:

* Compute `r_intr` inside **`reward_shaper.py`** (so reward stays deterministic given stored intrinsic stats), OR compute in collector and store `r_intr` explicitly (preferred to avoid recomputation).

Recommendation with your current architecture:

* Collector computes intrinsic terms and stores them into step dict.
* `reward_shaper.py` only combines terms.

---

# D) Two-head RL: implement EXPLORE/EXPLOIT using your existing controller modes

You already have a controller producing `mode_id`. Use modes as policies over objectives, not separate networks.

### D1. Mode semantics

* Define two logical modes (can still keep 3 modes if you want):

  * `mode=EXPLORE`: optimize intrinsic-heavy reward
  * `mode=EXPLOIT`: optimize extrinsic-heavy reward (win + minimal intrinsic)

### D2. How to route without duplicating nets

* Keep **one** encoder/memory/actor.
* Only change reward weights by mode at training-time.
* In `trainer.py`, compute advantages using per-step reward that depends on `mode_id`:

  * explore steps get `r = r_ext + w_intr_explore * r_intr`
  * exploit steps get `r = r_ext + w_intr_exploit * r_intr` (smaller)

### D3. Mode scheduling hook (optional)

* Add a simple schedule in `hierarchical_controller.py` via `ctx`:

  * early steps of episode bias toward EXPLORE
  * later steps bias toward EXPLOIT
    No game-specific detection required.

---

# E) Dynamic shaping that doesn’t break the objective: potential-based intrinsic + caps

Your current shaping is “direct reward”. Convert intrinsic to **potential-shaped** to reduce endless wandering.

### E1. Potential-based intrinsic (drop-in)

* Maintain per-episode or global state:

  * `Φ_t = f(rnd_err_t)` (normalized)
* Use shaped intrinsic:

  * `r_intr_shaped = γ*Φ_{t+1} - Φ_t`
    This still gives dense signal but naturally decays once novelty saturates.

Where:

* Either compute in collector (needs `t+1`), or in `reward_shaper.py` when both prev and curr intrinsic terms are available.

### E2. Anti-farming

* Add config knobs:

  * `intrinsic_episode_cap`
  * `intrinsic_step_clip`
  * `intrinsic_disable_on_flash` (you already have flash logic)
* Enforce in `reward_shaper.py`.

### E3. “Milestones” without object detection (optional but important for ls20)

Instead of clustering pixels, cluster embeddings:

* Maintain a small episodic set of landmark embeddings (`z_landmarks`).
* When `min_dist(z_t, z_landmarks)` exceeds threshold, add a new landmark and give a **one-time** bonus (or potential bump).
  This approximates “trigger activated” as a region-change event.

Where:

* best placed in collector/episode context (cheap stateful logic)

---

# Implementation order (refactored to your modules)

1. **A1–A2**: add `frame_core` (or `grid_embed_only`) to encoder output + store minimal intrinsic inputs in rollouts
2. **C1–C3**: add RND module + intrinsic normalization + log intrinsic stats
3. **D1–D3**: switch to explore/exploit weighting by `mode_id` in trainer reward/advantage computation
4. **E1–E2**: convert intrinsic to potential-based + add caps (stop farming)
5. **E3**: add landmark milestones (embedding-based)
6. **B1**: optional dynamics head + controllability reward (only after wins start appearing)

---

## Key design choice you must lock (no assumptions)

**RND input:**

grid_embed` only  as the RND input 
