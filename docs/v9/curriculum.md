# Hydra v9 Training Curriculum

**Config:** `curriculum.yaml`  
**Version:** 2  
**CLI:** `PYTHONPATH=src python -m v9 continuous-run --games stepN ...`

## Purpose

The curriculum has 14 developmental steps and now includes the broader Gym family in addition to synthetic environments, MiniGrid/BabyAI, ARC and ALFRED.

Gym-derived training is intentionally spread across several stages rather than placed in one block:

```text
simple discrete dynamics
→ irreversible manipulation
→ structured planning/constraints
→ continuous or vector state
→ cross-family transfer
→ stochastic/high-dimensional visual control
```

This gives Hydra progressively different representations while keeping failures interpretable.

## Important implementation consequence

The current generic v8 Gym adapter accepts only:

```text
Discrete observation
+
Discrete action
```

while Chess and Sudoku use custom adapters.

v9 must therefore own several environment boundaries instead of routing all Gym-like environments through one old adapter:

```text
gym_discrete   FrozenLake, Taxi, CliffWalking
gym_structured CartPole, MountainCar, Acrobot, Blackjack, Box2D
gym_image      Atari/ALE
sokoban        grid manipulation
chess          legal-move structured action space
sudoku         structured placement action space
```

These are representation families, not separate cognitive systems. All must feed the same M0→M7 memory architecture.

## CLI resolution

`--games step1` through `--games step14` resolve from `curriculum.yaml`.

Also provide useful aliases:

```bash
--games gym_foundation
--games gym_broad
--games semantics
--games cross_family
```

The resolver returns structured environment specifications containing the adapter, native ID, options and condition.

Do not flatten heterogeneous entries into ARC game IDs.

---

# Curriculum

## Step 1 — Deterministic causal primitives

Synthetic micro-worlds:

```text
syn_move
syn_reversible
syn_irreversible
syn_enable
syn_block
syn_two_step
syn_history_context
syn_same_role_different_carrier
syn_alternative_strategy
syn_replan
syn_contextual_rule
syn_delayed_effect
```

Target: M0/M1 learning, prediction and basic causal structure.

---

## Step 2 — Gym discrete foundations

Use:

```text
FrozenLake 4×4 deterministic
FrozenLake 8×8 deterministic
Taxi
CliffWalking
```

This is the best early external environment family because state and actions are compact and causality is relatively easy to inspect.

Gymnasium currently documents FrozenLake as `Discrete` observation/action, and Taxi as `Discrete(500)` observation with `Discrete(6)` actions. The resolver should support installed-version candidates for Taxi and CliffWalking rather than assuming one package version forever.

Target:

```text
reachability
future options
navigation
pickup/drop-off sequence
hazard avoidance
alternative paths
```

---

## Step 3 — Sokoban and structural role invariance

Sokoban:

```text
Sokoban-small-v0
Sokoban-small-v1
Sokoban-v0
Sokoban-v1
```

plus synthetic surface/role variations.

Sokoban is particularly valuable because pushing creates irreversible state changes and dead ends.

Target:

```text
planning before acting
irreversible consequences
block/enabler structure
M2/M3 abstraction
same role across different appearances
```

Keep the external Sokoban package behind a native v9 adapter because installed Sokoban packages differ in observation formats.

---

## Step 4 — Structured planning without language

MiniGrid interaction-only:

```text
Empty
FourRooms
DoorKey
Unlock
KeyCorridor
```

plus:

```text
Chess
Sudoku
```

Chess configurations:

```text
chess_first_white
chess_random_white
```

Sudoku configurations:

```text
sudoku_clues_45
sudoku_clues_36
sudoku_clues_30
```

Target: legality, working memory, long dependencies and constraint reasoning without semantic text.

Chess and Sudoku should remain bespoke v9 adapters rather than pretending they are simple `Discrete` Gym environments.

---

## Step 5 — Arbitrary-symbol grounding

Synthetic tasks:

```text
sym_enable
sym_block
sym_navigate
sym_pick
```

Each under:

```text
C0 interaction only
C1 symbols only
C2 aligned interaction + symbols
C3 shuffled interaction + symbols
```

Target: prove grounding before natural language.

---

## Step 6 — BabyAI basic grounding

```text
GoToObj
GoTo
GoToObjMaze
```

Target: simple reference semantics such as action/object/color combinations.

---

## Step 7 — BabyAI action semantics

```text
Pickup
PickupLoc
Open
OpenDoor
UnlockLocal
Unlock
```

Target: distinguish action/consequence meaning using the same objects and colors.

---

## Step 8 — BabyAI relational composition

```text
PutNextLocal
PutNext
UnlockPickup
BlockedUnlockPickup
UnlockToUnlock
KeyCorridor
```

Target: relational and multi-stage grounded composition.

---

## Step 9 — Sequential semantics plus non-grid Gym

BabyAI sequential tasks plus:

```text
CartPole-v1
MountainCar-v0
Acrobot-v1
Blackjack-v1
```

Gymnasium's classic-control suite includes CartPole, MountainCar and Acrobot; these introduce vector/continuous-valued observations with discrete control. Blackjack adds a composite tuple observation.

Target:

```text
representation independence
temporal dynamics
control sequences
semantic composition
```

This step requires the general v9 structured-observation adapter.

---

## Step 10 — ARC simple transfer

```text
ez01 ez02 ez03 ez04
ul01 fs01 ic01 pb01 tp01 va01
```

Target: transfer previously learned structural ideas into novel ARC visual representation.

---

## Step 11 — ARC diverse structural challenge

```text
fs02 fs03 tp02 tp03
ic02 ic03 nw01 gr01 dt01 wk01
rf01 mo01 ex01 sq01 mm01 rs01
lo01 rp01 bn01 dl01
```

Target: context, altered rules, temporal dynamics, memory and structural generalization.

---

## Step 12 — Broad cross-family curriculum

Run a compact mixture of:

```text
Synthetic
FrozenLake
Taxi
CliffWalking
CartPole
MountainCar
Acrobot
Blackjack
Sokoban
Chess
Sudoku
MiniGrid
BabyAI
ARC
```

This is the main test of the claim that M2-M7 are representation/environment neutral.

Transfer experiments should be budgeted, not continuously exhaustive.

---

## Step 13 — Stochastic, continuous and high-dimensional visual control

Introduce:

### Stochastic discrete

```text
FrozenLake slippery
```

### Box2D / richer dynamics

```text
LunarLander-v3
BipedalWalker-v3
```

### Atari/ALE

```text
ALE/Pong-v5
ALE/Breakout-v5
ALE/Freeway-v5
ALE/Seaquest-v5
ALE/Qbert-v5
ALE/MontezumaRevenge-v5
```

### Synthetic uncertainty

```text
syn_stochastic_transition
syn_partial_observation
syn_action_failure
syn_ambiguous_symbol
syn_contextual_symbol
syn_regime_shift
```

### Dynamic ARC

```text
zq01 tt02 fw01 hz01 ox01 rb01
```

Atari belongs late because raw image observations create a qualitatively larger perception problem. It should test whether the v9 observation-normalization boundary generalizes, not be used to debug basic memory formation.

Do **not** add the full Atari catalog immediately. Start with these six representatives. After the image adapter and memory budgets are stable, `gym_broad` can be expanded to additional ALE environments.

---

## Step 14 — ALFRED

Task families:

```text
pick_and_place_simple
pick_clean_then_place_in_recep
pick_heat_then_place_in_recep
pick_cool_then_place_in_recep
look_at_obj_in_light
pick_two_obj_and_place
```

Target: long-horizon multimodal language grounding and planning.

---

# Why not put every Gym environment into training immediately?

Gymnasium spans very different observation/action contracts:

```text
ToyText           compact/discrete
Classic Control   vector state
Box2D             continuous/vector dynamics
Atari             image stream
MuJoCo            continuous high-dimensional control
```

Treating all of these as equivalent "Gym games" would hide adapter and cognition failures.

The curriculum should therefore use representative members first. Once a representation family passes its gate, the config can add more games from the same family without changing cognition.

Recommended later expansion:

```text
Classic Control:
    Pendulum
    MountainCarContinuous

Box2D:
    CarRacing

Atari:
    SpaceInvaders
    MsPacman
    Riverraid
    Asterix
    Enduro
    Frostbite
    Pitfall
    Adventure

MuJoCo:
    Reacher
    Hopper
    Walker2d
    HalfCheetah
    Ant
```

Continuous-action environments require a declared bounded v9 action encoding before admission. They should not be discretized ad hoc simply to fit the current memory code.

---

# Implementation requirements

1. Keep `curriculum.yaml` as the authoritative mapping.
2. Implement native v9 adapters for each representation family.
3. Resolve installed environment versions at startup.
4. Fail before a long run if a configured environment/package is unavailable.
5. Persist the fully resolved curriculum in the run root.
6. Preserve held-out seeds/configurations outside formation provenance.
7. Keep semantics suppressed for interaction-only conditions.
8. Keep C0/C1/C2/C3 distinct in evidence.
9. Do not add domain-semantic features to M2-M7.
10. Do not import v8 from curriculum or production v9 adapters.
11. Add `--games gym_foundation` and `--games gym_broad`.
12. `gym_broad` means broad representation coverage, not blindly every registered Gym environment.

# Progression principle

Advance by demonstrated capability, not by benchmark completion:

```text
Step 2  -> stable discrete contingencies/planning
Step 3  -> irreversible planning + role abstraction
Step 4  -> structured constraint reasoning
Step 5  -> grounded symbols beat controls
Step 9  -> semantics survive representation change
Step 12 -> cross-family transfer
Step 13 -> stochastic/image representations without memory failure
Step 14 -> long-horizon multimodal grounding
```


# Curriculum telemetry and evidence

Curriculum execution must produce explicit evidence that the intended developmental capability is emerging. Aggregate win rate alone is insufficient.

## Required telemetry

Record the following for every curriculum step and evaluation interval.

### Memory

```text
M0..M7 created / active / validated / retired
new canonical identities by M-level
deduplication / merge rate
persistent bytes by M-level
persistent-memory / cumulative-experience ratio
```

### Learning

```text
prediction error
prediction improvement over time
compression gain
replay selections / revisions / promotions
developmental Stage 0..7
ISF component distributions: PVI / OSI / PE / LV / TP / EP
```

### Abstraction

```text
M2 family reuse
M3 functional-role reuse
M4 candidates
M4 transfer-test eligible
M4 validated
context count and growth
lineage count and growth
canonical fork / reuse counts
StructuralEquivalenceSet count and resolution rate
```

### Transfer

```text
within-game transfer
within-environment transfer
within-family transfer
cross-family transfer
matched trials attempted / passed / failed
false-transfer rate
negative-transfer evidence
target/context trust changes
```

Transfer telemetry must distinguish structural correspondence from causally validated transfer.

### Semantic grounding

```text
G0..G5 counts
grounding promotions / suspensions
C0 / C1 / C2 / C3 performance
symbol-conditioned prediction delta
grounded action influence
held-out compositional grounding performance
```

### Behavior

```text
success / solved rate
trajectory length
best trajectory
strategy reuse
replanning attempts / success
outcome achievement reliability
trajectory efficiency
```

## Mandatory provenance dimensions

Relevant metrics must be attributable to:

```text
curriculum version
curriculum step
source environment family
source game/scenario
target environment family
target game/scenario
memory level
context
validation mode
ScientificConfigId
```

Cross-family metrics must report the source→target pair explicitly.

For example:

```text
Synthetic → MiniGrid
Sokoban → BabyAI
BabyAI → ARC
FrozenLake → ARC
ARC → ARC
```

A global transfer/reuse percentage without these dimensions is insufficient for curriculum evaluation.

## Curriculum evidence report

Generate a dedicated report after each curriculum run:

```text
curriculum_evidence.md
```

The report must contain one section for each executed step with:

```text
Step
Environments/games actually resolved
Expected capability
Observed evidence
Happy-path indicators
Warning indicators
Result
Primary blocker
Relevant telemetry
```

Result is one of:

```text
PASS
PARTIAL
FAIL
INSUFFICIENT_EVIDENCE
```

The report must distinguish absence of evidence from negative evidence.

## Step-specific evidence expectations

### Step 1

Happy path:

```text
stable M1 contingencies
falling prediction error
correct enable/block and delayed-effect attribution
```

Warnings:

```text
M0 growth without M1 consolidation
duplicate M1 identities
persistent prediction failure in deterministic scenarios
```

### Step 2

Happy path:

```text
M2 recurrence across states/maps
future-option structure
improving trajectory efficiency
```

Warnings:

```text
exact-state memorization
no reuse across maps/seeds
M2 identity explosion
```

### Step 3

Happy path:

```text
same functional role converges across appearances
irreversible/dead-end consequences influence behavior
M3 reuse increases
```

Warnings:

```text
appearance dominates role
repeated irreversible mistakes
context/lineage explosion
```

### Step 4

Happy path:

```text
longer dependency chains
legal-action/context structure
reusable planning structures
```

Warnings:

```text
state memorization
unbounded memory growth
planning does not improve over local action choice
```

### Step 5

Happy path:

```text
C2 aligned interaction+symbols outperforms C0/C1/C3
cross-modal predictive grounding emerges
```

Warnings:

```text
C1 or shuffled C3 performs like C2
symbol frequency creates behavioral authority
ungrounded symbols influence actions
```

### Step 6

Happy path:

```text
basic object/color/reference symbols become interaction-grounded
held-out combinations begin transferring
```

Warnings:

```text
whole-instruction memorization
token identity dominates structural identity
unseen combinations fail completely
```

### Step 7

Happy path:

```text
same object symbols support distinct go-to/pickup/open/unlock consequences
```

Warnings:

```text
object identity becomes tied to one action
action phrase changes do not change predicted consequence
```

### Step 8

Happy path:

```text
grounded primitives recombine into new relational/multi-stage tasks
M4/M5 reuse increases
```

Warnings:

```text
every combination is relearned
combinatorial memory growth
relational structure remains instruction-template specific
```

### Step 9

Happy path:

```text
semantic structures survive representation changes
sequential M5/M7 structures improve
non-grid Gym does not create a separate cognitive architecture
```

Warnings:

```text
BabyAI-only semantics
isolated Gym memory
no cross-representation structural correspondence
```

### Step 10

Happy path:

```text
existing movement/enabler/blocker/trajectory roles accelerate simple ARC learning
```

Warnings:

```text
ARC behaves as entirely cold learning
high-level memory becomes ARC-specific
visual identity overwhelms structural similarity
```

### Step 11

Happy path:

```text
context refinement handles related-but-different ARC mechanics
negative evidence remains target/context scoped
```

Warnings:

```text
one failed variant globally damages a concept
uncontrolled context growth
high false-correspondence rate
```

### Step 12

Happy path:

```text
M3/M4 structures are reused across environment families
cross-family transfer becomes measurable
execution remains target-local
```

Warnings:

```text
memory fragments by environment family
cross-family reuse is near zero
cross-family correspondences are mostly false positives
```

Step 12 is the principal environment-neutral abstraction gate.

### Step 13

Happy path:

```text
prediction uncertainty separates from model error
context/regime changes are distinguished from intrinsic stochasticity
higher-level memory remains stable under image/vector scale
```

Warnings:

```text
permanent high prediction error
stochasticity causes continuous context splitting
M0/M1 growth overwhelms consolidation
Atari produces large memory with negligible abstraction
```

### Step 14

Happy path:

```text
grounded language activates reusable M4/M5 structures
long-horizon M7 strategies reuse prior knowledge
novel instructions benefit from earlier experience
persistent-memory growth remains controlled
```

Warnings:

```text
sentence/template memorization
long trajectories overwhelm provenance/replay
no concept reuse
near-linear persistent-memory growth
```

## Progression interpretation

The curriculum runner should report evidence but should not silently advance, stop, alter thresholds, or modify scientific configuration based on the result unless an explicit curriculum-control mode is separately configured.

The expected developmental progression is:

```text
Steps 1-2   M0/M1 causal learning
Steps 3-4   M2/M3 abstraction and structured planning
Steps 5-8   cross-modal grounding and composition
Steps 9-11  reusable M4/M5 consequence structures
Step 12     cross-environment abstraction and transfer
Step 13     robustness to uncertainty/high-dimensional input
Step 14     long-horizon grounded M6/M7 behavior
```

The central curriculum success criterion is not that every environment is solved. It is that increasingly different M0/M1 experiences converge into reusable higher-level structure while preserving target-local execution and causal evidence.
