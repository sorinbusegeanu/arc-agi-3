The Codex comment is correct.

After `bigfix+01`, the system is likely **directionally better** at round flow and bookkeeping, but still **representation-poor** for the task you described. The weak points are still the same two foundations:

1. **action-to-motion inference is heuristic**
2. **avatar inference is heuristic**

So before cross-area matching becomes credible, the system needs richer internal state than “POI + bbox + diff magnitude + consequence label”.

## Critical view first

The current concept still assumes that these can be chained:

* target a POI
* move there
* observe change
* store consequence
* later compare another consequence elsewhere
* infer same mechanic

That chain only works if the system has **stable representations at each link**.

Right now it mostly has:

* object boxes
* candidate avatar boxes
* traversable points
* target progress values
* consequence records with coarse labels

That is not enough for robust multi-area mechanic induction.

What is missing is not one more heuristic. It is a **stack of representations** between perception and reasoning.

---

# Realistic additional representations needed

## 1) For true target-conditioned navigation

This needs more than a target centroid and a traversable point cloud.

### Needed representations

#### A. Action semantics model

A persistent per-game model of what each discrete action tends to do.

Store, for each action:

* expected displacement distribution `(dx, dy)`
* variance / confidence
* failure rate
* context dependence
* whether it is movement, interaction, menu-like, noop-like, teleport-like

Not one global average. It should be conditioned on:

* avatar state cluster
* local neighborhood type
* recent previous action
* whether contact occurred

Because in many games:

* the same action does not always cause the same motion
* blocked moves and interaction moves look similar unless conditioned on context

#### B. Local navigation state

A representation of the local neighborhood around the avatar.

For each live step, store:

* avatar cell / footprint
* nearby free cells
* nearby blocked cells
* uncertain cells
* one-step reachability frontier
* recent failed transitions

This is needed so target following is not “go toward centroid” but “choose a legal short-horizon move”.

#### C. Route hypothesis graph

Not just visited points. A graph with edge semantics.

For each edge:

* source cell
* destination cell
* action that produced it
* success frequency
* block frequency
* confidence
* one-way / two-way status
* transition type: normal move / forced transport / room transition / ambiguous

This matters because two adjacent observed positions are not always connected in the same way.

#### D. Target access geometry

A POI should not only have a bbox. It should have:

* approach boundary
* contact boundary
* legal access cells
* blocked sides
* likely interaction side
* stand-off distance if direct overlap is not required

Many POIs are not “touch the center”.
They are:

* stand next to lever
* align with door
* stop on trigger tile
* press action near object

### Why this is needed

Without these, the executor still only has:

* target coordinate
* maybe next subgoal
* heuristic action guess

That is enough for trivial open spaces, not for maze-like or interaction-heavy games.

---

## 2) For live avatar re-localization every step

This needs more than ranking current candidate objects by confidence.

### Needed representations

#### A. Persistent avatar track state

Maintain a real avatar track object with:

* track id
* last confirmed bbox / centroid
* velocity estimate
* action-conditioned predicted next position
* visibility confidence
* occlusion/missing counter
* appearance signature
* shape signature
* motion signature
* recent support evidence

This is not the same as “list of avatar candidates”.

#### B. Multi-hypothesis track set

In ambiguous scenes, keep multiple avatar hypotheses alive.

For each hypothesis:

* posterior score
* consistency with recent actions
* consistency with motion regions
* consistency with route graph
* exclusion against HUD/static objects

This is needed because the correct avatar may temporarily become ambiguous during:

* room changes
* similar sprites
* flicker
* moving enemies
* partial occlusion

#### C. Avatar appearance model

A lightweight, per-game representation of the avatar’s stable visual identity:

* color histogram
* shape mask template
* bbox aspect ratio range
* animation-state family
* common size range
* local texture/pattern signature

Not to identify perfectly, but to stop the tracker from switching identities every few steps.

#### D. Action-conditioned state transition model for the avatar

Given:

* previous avatar hypothesis
* action
* local neighborhood

predict:

* expected next avatar region
* expected uncertainty radius
* expected failure mode if blocked

This is essential for distinguishing:

* “avatar disappeared”
* “avatar moved”
* “candidate was wrong”
* “screen scrolled”
* “teleport/transition occurred”

### Why this is needed

Your current avatar layer is still closer to “rank candidates this frame”.
For directed exploration and consequence attribution, you need **tracking**, not just per-frame scoring.

---

## 3) For cumulative blackboard memory across rounds

This needs more than saving latest blackboard and appending round outputs.

### Needed representations

#### A. Persistent world model state

Per game, maintain structured persistent state:

* discovered regions / rooms
* known traversable graph
* known blocked graph
* uncertain transition zones
* stable objects
* dynamic objects
* avatar tracks
* POI histories
* event histories
* route histories

This is a **world hypothesis**, not just a round snapshot.

#### B. Evidence ledger

For every major hypothesis, maintain supporting and contradictory evidence separately.

Examples:

* POI exists
* POI is reachable
* POI causes local change
* POI causes delayed remote change
* action 3 usually moves left
* room transition exists at edge X

Each should store:

* evidence refs
* positive count
* negative count
* confidence
* recency
* context tags

This is needed so the system can degrade or reverse beliefs, not only accumulate them.

#### C. Entity lifecycle state

For every persistent object / POI:

* first seen
* last seen
* seen in which room contexts
* merged-from ids
* split-from ids
* stability score
* dormant / active status
* interaction status
* linked events

Without this, cross-round matching remains brittle.

#### D. Round-level decision memory

Store:

* which target was chosen
* why it was chosen
* route attempted
* whether progress occurred
* whether no-progress repeated
* what uncertainty remained afterward

This is needed so the controller can avoid cycling on targets that looked promising but repeatedly failed under instructed execution.

### Why this is needed

Without this layer, each round is still too close to a fresh analysis batch with weak carry-over.

---

## 4) For structured event extraction from screen changes

This is the biggest gap.

You do not just need “change magnitude”.
You need a representation of **what changed**.

### Needed representations

#### A. Change event object

For every meaningful change, represent:

* event id
* time / step span
* triggering context
* locality: local / remote / global / transition
* changed regions
* changed objects
* object births
* object deaths
* object moves
* object appearance changes
* object state changes
* terminal / reward flags
* confidence

This turns screen diff into something reason-able.

#### B. Object state delta representation

For each changed object:

* pre-state object record
* post-state object record
* delta type:

  * moved
  * appeared
  * disappeared
  * recolored
  * resized
  * activated
  * deactivated
  * opened
  * closed
  * destroyed
  * transformed
* magnitude / confidence

This is needed because “door opened” and “enemy moved” can have similar raw pixel diffs.

#### C. Region-change decomposition

Represent screen change by grouped regions:

* near target
* near avatar
* remote but same room
* edge/scroll transition
* full-scene replacement
* HUD-only

This is required to distinguish:

* contact event
* remote effect
* room transition
* noise
* HUD update

#### D. Temporal event envelope

Events should have:

* onset step
* peak step
* decay / end step
* delay from trigger
* persistence duration

A lot of effects are not instantaneous.
Without this, delayed consequences are mislabeled as unrelated.

### Why this is needed

Until changes become typed events, the system cannot really compare consequences across areas in a meaningful way.

---

## 5) For delayed causal linking between target contact and later remote effects

This needs more than “target_poi_id on step” plus consequence records.

### Needed representations

#### A. Intervention episode representation

When the system intentionally targets a POI, it should store a structured intervention record:

* intended target
* intended contact mode
* route taken
* whether target was reached
* whether contact happened
* precise contact time
* post-contact observation window
* post-contact event set

This is the basic causal unit.

#### B. Temporal causal window model

For each intervention:

* pre-contact baseline window
* immediate post-contact window
* delayed post-contact window
* timeout horizon

Events inside these windows should be scored differently.

This is needed because:

* some effects happen immediately
* some after 1–3 steps
* some after room transition
* some not at all

#### C. Counterfactual / contrast memory

You need stored comparisons like:

* same POI approached but not contacted
* same route without target contact
* same room explored with no event
* same action elsewhere with no event

Without contrast, the system will over-attribute random changes to the last target.

#### D. Cause-effect link record

A dedicated representation:

* cause candidate: POI/contact/intervention
* effect event id
* delay
* spatial separation
* confidence
* competing causes
* repeatability evidence
* falsification evidence

This should exist independently from the POI and independently from the event.

### Why this is needed

Remote effects are fundamentally **causal hypotheses**, not just POI properties.

---

## 6) For cross-area effect signatures that ignore absolute position

This is where current bbox-centric matching will fail hardest.

### Needed representations

#### A. Area / room identity representation

The system needs explicit room or area hypotheses:

* area id
* canonical state signature
* entry/exit boundaries
* local traversable graph
* stable objects in that area
* area-level palette / geometry descriptors

Without area identity, it cannot even say two events happened in different contexts in a stable way.

#### B. Canonical object/state signatures

For matching across areas, each relevant changed object/event needs a normalized signature:

* object class
* shape family
* relative size
* relative orientation
* local neighborhood structure
* state before / state after
* whether access path changed
* whether obstacle/free-space topology changed

This allows “door opened in room B” to match “door opened in room C” even if coordinates differ.

#### C. Relative geometry representation

Effects should be represented relative to:

* local room frame
* nearby landmarks
* doorway/chokepoint geometry
* avatar-independent coordinates

Absolute `(x, y)` is not enough.
Cross-area matching needs things like:

* “barrier removed at chokepoint”
* “passage became traversable on right edge of room”
* “object toggled near exit”

#### D. Topology delta signature

Represent the consequence in terms of graph change:

* new edge opened
* edge removed
* region connectivity changed
* chokepoint unlocked
* area transition enabled
* path length reduced to a known goal

This is much more useful than pixel diff for mechanic matching.

### Why this is needed

A switch in room A and a door in room B are usually linked through **topological consequence**, not visual similarity.

---

## 7) For relation-level matching, not bbox-level matching

This is the reasoning layer above event extraction.

### Needed representations

#### A. Interaction schema / relation graph

Represent the world as relations:

* avatar can-reach POI
* POI near chokepoint
* contact with POI precedes remote event
* remote event changes object state
* object-state change alters reachability
* effect repeats across episodes

This is much closer to the real mechanic.

#### B. Mechanic hypothesis objects

A first-class representation for candidate mechanics:

* trigger type
* trigger object class
* trigger context
* effect type
* effect target class
* delay distribution
* area relationship
* repeatability score
* falsification score

Examples:

* `contact(switch) -> remote_open(barrier)`
* `stand_on(tile) -> area_transition`
* `approach(enemy) -> terminal`

Without this, the system never graduates from observations to reusable rules.

#### C. Relational effect signature

An effect should be matchable by relation pattern, such as:

* trigger local, effect remote
* effect opens connectivity
* effect creates/removes obstacle
* effect repeats after same trigger in different areas
* effect is gated by approach side

This lets two different-looking consequences be recognized as the same mechanic family.

#### D. Hypothesis competition set

Store competing relational explanations:

* direct local effect
* delayed remote effect
* incidental movement
* room transition artifact
* HUD artifact
* unrelated enemy motion

The system should score and eliminate them over repeated evidence.

### Why this is needed

Without mechanic-level hypotheses, matching remains superficial and fragile.

---

# What this means for your specific scenario

For:

* move to POI in area A
* observe change
* later see similar change in area B
* decide whether they match

the system realistically needs this chain:

1. **stable avatar track**
2. **stable route/action semantics**
3. **intervention record**
4. **typed change event**
5. **cause-effect link**
6. **area identity**
7. **topology/state-delta signature**
8. **mechanic hypothesis**

Right now it mostly has:

* candidate target
* partial route heuristic
* coarse consequence record

That is several layers too shallow.

---

# About the Codex note on a second pass for ls20

Yes, that second pass is needed.

But it should not only improve “directed target acquisition quality”.
It should explicitly add these representations for the navigation side:

## Immediate next realistic pass

### Priority A

Strengthen the movement/control substrate:

* persistent action semantics model
* persistent avatar track state
* live multi-hypothesis relocalization
* route graph with edge success/failure semantics
* target access geometry

### Priority B

Strengthen the consequence substrate:

* typed event extraction
* intervention records
* causal windows
* cause-effect links

### Priority C

Only after that:

* area identity
* topology delta signatures
* mechanic hypotheses
* cross-area matching

If A is weak, B is noisy.
If B is noisy, C is mostly fiction.

---

# Final judgment

The concept is still valid, but the current implementation is too close to:

* **heuristic navigation**
* **heuristic avatar detection**
* **coarse diff logging**
* **weak memory aggregation**

To do real cross-area mechanic discovery, it needs a much richer representation stack.

The most important shift is this:

> Stop treating POIs and consequences as the main objects.
> Start treating **tracks, events, interventions, causal links, area states, and mechanic hypotheses** as the main objects.

That is the realistic path from “target a thing and note some pixels changed” to “this switch in one room opens a barrier in another room, and I can recognize the same mechanic elsewhere.”

