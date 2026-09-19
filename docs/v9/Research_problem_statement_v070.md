# **I Pushed This Before. It Moved.**

## **A Memory-Centric Theory of Emergent Intelligence**

**Version 0.7.0**

### **Abstract**

### **Version 0.7.0 Revision**

Version 0.7.0 extends the frozen v0.6.3.1 developmental theory with one
new falsifiable hypothesis: **learned relational reasoning can emerge as
a trainable operator over the explicit developmental memory substrate
without replacing that substrate as the authority for persistent
knowledge**.

The extension is motivated by Hydra System Design v9.7.2. It separates
two functions that earlier versions treated together:

-   **developmental memory formation** remains responsible for grounded
    M0-M7 knowledge, canonical identity, provenance, lineage, lifecycle,
    grounding, causal transfer validation, negative evidence,
    developmental stage, primary valence, ISF, and future-option
    structure;
-   **learned relational reasoning** operates over bounded, versioned
    subgraphs of that memory to estimate relevance, similarity,
    correspondence, consequences, strategy quality, candidate
    refinement, and useful reasoning operations.

The learned reasoner is therefore downstream of developmental grounding
and upstream of candidate selection and deliberation. Its outputs are
hypotheses, scores, latent representations, and refinements. They do not
by themselves create validated concepts, establish grounding, prove
transfer, or mutate canonical memory.

Version 0.7.0 also makes continual reasoner learning explicit. Published
model versions are immutable; candidate successor models learn
asynchronously from causally provenance-preserving interaction,
consolidation, transfer, grounding, replay, and deliberation evidence.
Publication requires improvement without unacceptable
historical-retention loss.

This extension creates a new experimental question: whether a learned
relational operator over emergent developmental memory improves
prediction, retrieval, structural correspondence, transfer, planning,
and sample efficiency beyond explicit-memory reasoning alone, while
preserving the developmental ordering and causal-validation requirements
of the theory.

### **Version 0.6.3.1 Revision**

Version 0.6.3.1 is a clarification release. It adds no new central
hypothesis or architectural mechanism. It makes explicit that persistent
structural ambiguity may represent a genuine equivalence class rather
than a failed search, and that unresolved structural symmetry should be
broken by future causal evidence rather than forced unique
identification.

Version 0.6.3 completes three targeted refinements identified during
critical review: dependency-scoped causal suspension during local
developmental regime re-entry, explicit structural ambiguity as the
reference criterion for multi-scale expansion, and bounded online
estimation of scale-normalization statistics. These changes do not
introduce new central hypotheses.

Version 0.6.2 refines the Version 0.6.1 mechanisms without expanding the
central hypothesis set. It adds local developmental regime re-entry for
non-stationary contexts, scale-normalized progressive structural
comparison with evidence-based stopping, and two explicit falsification
experiments for deep structural transfer and structural-prior
dependence.

Version 0.6.1 strengthens the theory in four areas exposed by critical
analysis: (1) pre-predictive cold-start development, (2) multi-scale
structural correspondence, (3) stability of concurrent developmental
processes, and (4) explicit dependence on declared structural priors. It
adds two falsifiable hypotheses, H17 and H18, and makes the boundary
conditions of semantic abstention explicit.

Version 0.6.0 extends the interaction-first theory to test the emergence
of grounded symbolic meaning from multimodal experience. The extension
does not introduce pretrained language models, embeddings, lexical
semantics, or a separate language architecture. Text is defined as an
ordered stream of observable symbols whose initial identities carry no
task-semantic meaning.

Interaction remains the source of causal grounding because agent
interventions permit consequences to be tested. Symbolic observations
may become meaningful only through empirically established temporal,
causal, predictive, and higher-order structural relationships with
interaction-derived memories. The central new question is whether
symbolic and interaction evidence can converge on shared functional
roles, concepts, consequence structures, and strategies.

Version 0.6.0 therefore adds a cross-modal grounding hypothesis and a
falsifiable experimental program. Successful grounding requires more
than symbol co-occurrence or sequence prediction: grounded symbols must
support transfer between symbolic evidence and previously unexperienced
interactions, and interaction-derived abstractions must constrain
interpretation of symbolic observations. Interaction-only, symbol-only,
aligned multimodal, and shuffled-symbol controls are required to
distinguish grounding from statistical symbol learning.

The macro-time/micro-time interaction model, Stage 0--7 developmental
model, memory-fitness framework, concurrent shared-memory architecture,
outcome equivalence, strategy efficiency, and conditional compression
proposition of Version 0.5.5 are retained.

-   **Intelligence is investigated as a developmental process of
    organizing remembered interactions under a minimal signed
    primary-valence signal rather than as a consequence of predefined
    task-semantic objects, goals, policies, or world models.**
-   **The primitive memory unit is an action-conditioned transformation
    defined over an explicitly declared observation structure.**
-   **Text and other symbolic channels may enter as ordered sensory
    streams of initially semantic-free symbols. Symbolic meaning is
    hypothesized to emerge only when those memories acquire predictive,
    explanatory, or transferable relationships with interaction-grounded
    higher-order structure.**
-   **One macro interaction may contain an ordered micro-time trace of
    passive intermediate observations produced by the environment after
    an action and before the settled next observation. These frames
    provide causal evidence without being counted as additional agent
    interventions.**
-   **The framework distinguishes semantic-free learning from
    structure-free learning: observation interfaces may provide
    domain-generic relations such as temporal order, equality,
    coordinates, or adjacency, while task-semantic entities remain
    emergent.**
-   **Primary valence and interaction significance are distinct. A
    minimal signed primary-valence signal provides motivational
    grounding, while the Interaction Significance Function allocates
    developmental resources using primary-valence impact, observed
    option-structure change, learned prediction error, learning value,
    transfer prior, explanatory potential, and bounded future-option
    reachability.**
-   **Concept candidates arise through compression and relational reuse;
    validated transferable concepts additionally require empirical reuse
    in previously unseen environments.**
-   **Outcome-equivalence abstractions emerge as persistent
    representations of distinct states or trajectories with sufficiently
    similar learned consequence structure, separating what is reached
    from how it is reached and enabling alternative-strategy reuse and
    replanning.**
-   **Stable target-like outcomes are treated as later learned
    preference relations over outcome-equivalence abstractions, grounded
    in realized primary-valence evidence rather than supplied as
    primitive goal semantics.**
-   **M7 strategy memory separates outcome-achievement reliability,
    signed primary-valence evidence, and realized trajectory cost. Among
    strategies that reach the same learned M6 outcome, lower expected
    action/interaction cost constitutes higher efficiency and may
    receive stronger reuse and planning weight.**
-   **Transfer is hypothesized to depend primarily on functional-role
    and graph-structural correspondence rather than perceptual
    similarity.**
-   **In recurrently compressible environments, persistent consolidated
    memory is predicted to grow sublinearly relative to accumulated
    experience while predictive, explanatory, transfer, and planning
    quality are preserved or improved.**

# **1. Introduction**

**Core Thesis**

This paper proposes that intelligence emerges from the acquisition,
organization, compression, restructuring, and transfer of remembered
interactions.

The primitive unit of knowledge is a remembered action-conditioned
transformation:

Observation(t) → Action(t) → \[O(t,1), ..., O(t,m_t)\] →
Observation(t+1), with V(t) ∈ {-1,0,+1}

From repeated interactions, the agent discovers contingencies. From
contingencies, it constructs increasingly compressed structures
including transformation families, functional roles, concepts, and
eventually world models.

The significance of an interaction is determined not by its perceptual
appearance but by its effects on future possibilities, learning,
transfer, and explanation. Interactions compete for persistence in
memory. Structures are promoted through increasing explanatory reach,
producing a developmental hierarchy of abstractions.

Objects are not primitive representations. They emerge as explanatory
compressions of recurring transformation patterns. Transfer emerges
primarily through reusable functional roles and graph structure rather
than perceptual similarity.

Intelligence is therefore defined not as optimization toward a
predefined semantic objective, but as the continuous restructuring of
memory under interaction, primary-valence evidence, and future-option
structure. Late in development, the agent may form persistent
**outcome-equivalence abstractions** that group distinct states or
trajectories by sufficiently similar learned consequence structure. This
separates the representation of what is reached from the strategies by
which it is reached. Efficiency becomes meaningful only after such
outcome comparability exists. M7 strategy memory therefore maintains
outcome-achievement reliability separately from realized trajectory
cost. The minimal cost is action count from strategy initiation to the
represented outcome; richer declared costs may additionally include
time, loops, blocked interactions, uncertainty, risk, or compute. When a
trajectory terminates in non-zero primary valence, the number of
interactions to that valence-bearing episode boundary is also retained
as efficiency evidence. Lower-cost trajectories may then receive
stronger memory fitness, replay priority, and planning preference within
the same learned outcome class or an explicitly no-worse class. Stable
goal-like behavior, when it appears, is treated as a later learned
preference over outcome abstractions rather than as a primitive
objective.

# **2. Research Problem Statement**

## **The Problem**

Most contemporary AI systems assume that intelligence emerges primarily
from one or more of the following:

-   Optimization toward an objective.\
-   Reward maximization.\
-   Prediction accuracy.\
-   Construction of world models.\
-   Object-centric representations.\
-   Large-scale statistical pretraining.

These approaches differ in implementation but typically assume that
useful abstractions depend on explicit representations of objects,
states, rewards, goals, policies, or predictive models.

This work investigates an alternative possibility.

The central claim is that reusable abstractions can emerge before object
representations, before explicit world models, and before sophisticated
prediction mechanisms. Intelligence may instead originate from
remembered interactions and the continuous restructuring of memory.

------------------------------------------------------------------------

## **Motivation**

Interactive environments such as ARC-style tasks provide an unusually
clean setting for studying abstraction formation.

The agent receives:

-   Observations.\
-   Primitive actions.\
-   State transitions.\
-   Temporal order.\
-   Explicitly declared non-semantic observation structure such as grid
    coordinates or adjacency when supplied by the interface.\
-   A minimal signed primary-valence signal derived at the environment
    boundary from terminal outcome polarity: positive, neutral, or
    negative. Environment label names remain available for external
    evaluation but are not used as semantic memory identities.

The learner is not provided with:

-   Object identities.\
-   Avatars.\
-   Keys.\
-   Doors.\
-   Enemies.\
-   Tools.\
-   Task-semantic reward interpretation.\
-   Goal semantics.\
-   Task-semantic labels.

Such structures exist only in the interpretation of a human observer.

The scientific challenge is to determine whether an agent can discover
these abstractions through experience alone.

## **Research Question**

Can an agent discover, compress, transfer, and reuse causal interaction
structures across previously unseen environments from remembered
interactions plus explicitly declared non-semantic observation-level
structural priors, without predefined object representations,
task-semantic concepts, predefined goal semantics, or an initial
integrated world model, while receiving only minimal signed primary
valence as motivational grounding?

## **Cross-Modal Grounding Research Question**

The Version 0.6.0 extension asks:

> **Can initially meaningless symbolic observations acquire grounded
> functional meaning solely through temporal, causal, predictive, and
> structural relationships with interaction-derived memories, without
> pretrained language models, embeddings, predefined lexical semantics,
> or an explicit symbol-grounding mechanism?**

A stronger consequence is also tested:

> **After grounding, can symbolic observations alter predictions,
> planning, or action selection in novel interaction configurations that
> the learner has not directly experienced?**

The reverse direction is required as well: interaction-derived
abstractions must constrain the interpretation and reuse of symbolic
observations. The scientific target is therefore **bidirectional
cross-modal transfer**, not symbol prediction accuracy alone.

## **Alternative Hypothesis**

Generalizable abstractions emerge from:

-   Remembered interactions.\
-   Contingency discovery.\
-   Prediction violations.\
-   Memory competition.\
-   Future-option discovery.\
-   Explanatory compression.\
-   Functional-role formation.

Under this hypothesis, concepts are not primitive structures.

Concepts emerge as compressed explanations of recurring interaction
patterns that support transfer to previously unseen environments.

## **Null Hypothesis**

Compression and restructuring of interaction histories are insufficient
to produce reusable abstractions.

Transferable concepts require explicit object representations,
predictive world models, task-semantic reward structures, or other
predefined semantic mechanisms beyond minimal signed primary valence.

If abstraction and transfer consistently fail to emerge from
interaction-centric memory alone, the theory is falsified.

## **Scientific Objective**

The objective of this research is not merely to improve benchmark
performance.

The objective is to determine whether a developmental pathway exists in
which:

Interactions → Contingencies → Functional Roles → Concepts → World
Models / Consequence Structure → Outcome-Equivalence Abstractions →
Alternative Strategies

emerge naturally from experience without requiring those structures to
be predefined.

If successful, the work would support a memory-centric account of
intelligence in which objects, concepts, goals, planning, and world
models are emergent consequences of remembered interaction rather than
foundational assumptions.

# **3. Interaction-First Ontology**

## **Minimal Commitments**

The theory begins from a deliberately restricted but not structure-free
interaction interface. At developmental time (t), the learner receives:

-   an observation (O_t),
-   an available action set (`\mathcal{A}`{=tex}\_t),
-   a selected action (A_t),
-   an optional ordered sequence of passive intermediate observations
    (T_t=(O\_{t,1},...,O\_{t,m_t})) emitted while that action unfolds,
-   a settled subsequent observation (O\_{t+1}),
-   temporal order at both macro and available micro resolution,
-   a minimal signed primary-valence signal (V_t) when the environment
    emits motivationally relevant terminal evidence.

The observation interface may also expose a declared set of
**domain-generic structural relations** (`\mathcal{R}`{=tex}\_O), such
as equality, coordinates, adjacency, neighborhood, ordering, or sensor
topology. These relations are inductive biases of the observation
representation. They are not task-semantic entities.

The theory therefore distinguishes two claims:

1.  **Semantic abstention:** object identities, roles, goals, tools,
    task-semantic reward interpretations, and task concepts are not
    provided as primitive representations. A signed primary-valence
    channel is permitted because it supplies motivational polarity
    without supplying task semantics.
2.  **Structural transparency:** any non-semantic structure supplied by
    the observation interface must be stated explicitly and treated as
    an inductive prior.

The interaction-first hypothesis is a claim about the emergence of
functional and semantic organization from interaction history under
declared structural priors. It is not a claim that learning can proceed
without any representational structure.

## **Multimodal Observation Streams**

Let the learner receive a set of ordered observation streams

\[ `\mathcal{S}`{=tex}={S^{(1)},S^{(2)},`\ldots`{=tex},S\^{(m)}}, \]

where each stream supplies observations indexed by causal availability
time. A stream may contain environmental observations, passive
micro-time frames, or discrete symbolic observations. A symbolic stream
is represented as

\[ S\^{(`\mathrm{sym}`{=tex})}=(s_1,s_2,`\ldots`{=tex},s_n), \]

where each (s_i) is an identity-bearing symbol with ordering and
provenance but **no supplied task-semantic interpretation**.

The admissible primitive relations remain domain-generic: equality,
inequality, temporal order, adjacency, containment in an observation
interval, and other explicitly declared structural relations. A symbol
identifier is not a semantic category.

Streams are not required to be synchronized one-to-one. Their evidence
may be linked only when temporal availability, repeated predictive
association, or later explanatory structure supports such linkage.

### **Interaction Privilege**

Multimodality does not make all streams epistemically equivalent. Agent
actions retain a privileged role because interventions permit candidate
causal relationships to be tested against consequences. Symbolic
observations can supply predictive evidence, but causal grounding
requires consistency with interaction-derived evidence.

### **Grounding Criterion**

A symbolic structure is considered **grounded** only if its learned
memory relations contribute to at least one interaction-relevant
capability beyond within-symbol statistics, such as:

1.  prediction of an interaction consequence;
2.  selection or rejection of an action;
3.  transfer to a held-out interaction configuration;
4.  explanation of a previously learned interaction regularity; or
5.  activation of a functional abstraction that was independently
    supported by interaction evidence.

Symbol co-occurrence, next-symbol prediction, or repeated temporal
proximity alone is insufficient evidence of grounded meaning.

## **Primitive Memory**

Let (D_O) be an observation-domain transformation operator. Let
(T_t=(O\_{t,1},...,O\_{t,m_t})) be the ordered sequence of passive
observations emitted after action (A_t) and before the environment
settles; (m_t) may be zero. The primitive macro-interaction record is:

\[ I_t = (O_t, A_t, T_t, `\Delta`{=tex}\_t, V_t), `\qquad`{=tex}
`\Delta`{=tex}*t = D_O(O_t,O*{t+1}), `\qquad`{=tex} V_t `\in`{=tex}
{-1,0,+1}. \]

When (T_t) is available, the learner may also retain the ordered
micro-transition sequence

\[ `\delta`{=tex}*{t,j}=D_O(O*{t,j-1},O\_{t,j}), \]

with (O\_{t,0}=O_t) and the final element connected to the settled
(O\_{t+1}). These micro-transitions are observational evidence about the
causal mechanics of one intervention; they are not additional actions.
If the environment exposes only the settled frame, (T_t) is empty and
the formulation reduces to the original single-transition record.

(V_t) is **primary valence**, not a task-semantic label. At the
environment boundary, positive terminal polarity maps to (+1), negative
terminal polarity to (-1), and absence of primary-valence evidence to
(0). The symbolic names of environment terminal states do not enter
memory identity. The primitive memory therefore records an intervention,
its observed transformation, and minimal motivational polarity. All
task-semantic interpretation remains developmental.

The primitive memory therefore records an intervention together with an
observed transformation. All higher-order structures are constructed
from collections of such records and their relations.

## **No Object Primitive**

Persistent entities are not supplied to the learner. A carrier is
introduced only when a latent persistence hypothesis explains recurring
transformations more compactly or predictively than independent
transition records.

The proposed developmental order is:

\[ `\text{Transformation}`{=tex} `\rightarrow`{=tex}
`\text{Transformation Family}`{=tex} `\rightarrow`{=tex}
`\text{Carrier Hypothesis}`{=tex} `\rightarrow`{=tex}
`\text{Functional Role}`{=tex} `\rightarrow`{=tex}
`\text{Concept}`{=tex}. \]

Objects, when they emerge, are therefore latent carrier hypotheses
supported by temporal and relational evidence rather than primitive
perceptual labels.

## **Minimal Primary-Valence Primitive**

The framework distinguishes **primitive motivational polarity** from
**task-semantic goal or reward representations**. A minimal signed
primary-valence signal is permitted as an innate drive channel:

\[ V_t `\in`{=tex} {-1,0,+1}. \]

Positive and negative terminal outcome polarity therefore provide
primitive motivational evidence. Their environment-specific names do not
become memory semantics, do not define objects, concepts, outcome
identities, or strategies, and do not specify which observations or
transformations will produce them.

Primary valence answers only whether an experienced consequence is
motivationally positive, neutral, or negative. The learner must still
discover from interaction which contingencies, transformation families,
carriers, roles, concepts, consequence structures, outcome-equivalence
classes, and strategies predict or mediate that valence.

Structural consequences such as continuation, action availability,
reversibility, and learned graph reachability remain separate evidence.
Future options may be useful instrumentally, but they are not used as a
substitute for primary valence.

### **Emergence of Goal-Like Structure**

The framework distinguishes three late-emerging structures that
conventional accounts often collapse into the word *goal*:

1.  **Outcome-equivalence abstraction:** a persistent representation
    grouping distinct states or trajectory endpoints whose learned
    consequence descriptors are sufficiently similar.
2.  **Strategy-to-outcome relation:** an empirically supported relation
    stating that a trajectory or procedure tends to reach a particular
    outcome class.
3.  **Target-like preference:** a stable learned tendency to select one
    reachable outcome class over alternatives across repeated comparable
    decisions.

Outcome equivalence does not imply desirability. A target-like
preference is an additional empirical relation learned from realized
primary-valence evidence under comparable reachable alternatives and
subsequent interaction history. A conventional goal is therefore
interpreted, when it emerges, as a sufficiently stable target-like
outcome used as a planning reference. The theory does not assume such a
preference at initialization.

This distinction separates **what is represented as an equivalent future
condition** from **how that condition was previously reached**, allowing
alternative strategies to be substituted without requiring a primitive
goal symbol.

## **No World-Model Primitive**

The learner is not initialized with an integrated predictive world
model. Local expectations arise only after recurring contingencies have
accumulated sufficient evidence. Larger predictive and explanatory
models are later compressions of already-discovered interaction
structure.

The proposed dependency is therefore:

\[ `\text{Interaction}`{=tex} `\rightarrow`{=tex} `\text{Memory}`{=tex}
`\rightarrow`{=tex} `\text{Contingency}`{=tex} `\rightarrow`{=tex}
`\text{Local Prediction}`{=tex} `\rightarrow`{=tex}
`\text{Role/Concept Structure}`{=tex} `\rightarrow`{=tex}
`\text{World Model}`{=tex}. \]

## **Emergence Criterion**

A higher-level structure is retained only if it provides measurable
benefit relative to the lower-level structures from which it was
derived. Relevant evidence may include:

-   compression,
-   prediction,
-   explanatory reach,
-   contextual resolution,
-   held-out transfer,
-   future-option estimation,
-   trajectory efficiency when outcomes are comparable.

Frequency alone is insufficient.

## **Ontological Commitment**

The minimal semantic ontology contains no predefined objects, goals,
roles, concepts, policies, or world models. The operational commitments
are instead:

1.  temporally ordered observations,
2.  available actions,
3.  an explicitly declared observation-level structural prior,
4.  remembered action-conditioned transformations.

All higher-level functional and semantic structures are empirical
hypotheses over this interaction history.

# **4. Developmental Hierarchy**

The theory proposes that intelligence develops through a sequence of
increasingly compressed and transferable structures.

Each level emerges because it explains, predicts, compresses, or
transfers information more effectively than the level beneath it. No
level is assumed to exist in advance. Each level is constructed from
experience.

## **Level 0: Observations**

The agent initially experiences only observations. These may be pixel
grids, sensory inputs, or other raw environmental signals.

At this stage the learner has no supplied task-semantic representations
of objects, goals, concepts, or functional meaning. It possesses only
the observation/action interface, declared structural priors, and
accumulated interaction evidence.

## **Level 1: Observation Transformations**

Repeated observations reveal change. The first useful abstraction is not
a thing but a transformation between observations. Because observation
spaces need not be vector spaces, the theory does not assume that change
is representable by arithmetic subtraction. Instead, define an
observation-space-specific transformation operator:

Δ(t) = D_O(O(t), O(t+1))

where D_O returns a representation of observed change appropriate to the
observation domain. For a pixel grid this may encode changed cells,
spatial displacement, connected change regions, or other non-semantic
structural differences. For other sensory spaces it may use a different
transformation representation. The operator is required to preserve
observable change without requiring object labels or semantic
interpretation.

The world is first experienced as transformations rather than entities.
This is the first step toward structure.

## **Level 2: Action-Conditioned Deltas**

The agent discovers that some transformations depend on actions. Instead
of merely observing change, the agent begins associating change with
intervention. The primitive memory becomes:

Observation(t) → Action → Δ

Knowledge becomes action-conditioned. The agent begins learning:

When I do X, Y tends to happen.

This is the earliest form of causal structure available to the system.

## **Level 3: Contingencies**

Repeated action-conditioned transformations become contingencies. A
contingency is a recurring action-consequence relationship.

Examples:

-   Move right → position changes.\
-   Touch red → termination.\
-   Click object → transformation occurs.

Contingencies are the first reusable units of experience. They support
primitive expectation.

## **Level 4: Prediction Violations**

Once contingencies exist, expectations become possible. When observed
outcomes differ from expected outcomes: Observed Δ ≠ Expected Δ a
prediction violation occurs.

Prediction violations are informative because they may reveal:

-   incomplete knowledge,\
-   hidden variables,\
-   missing context,\
-   novel structures.

Prediction violation becomes a major driver of corrective processing.
However, prediction error alone does not determine significance;
unexpected events may be structurally irrelevant.

## **Level 5: Transformation Families**

As contingencies accumulate, the agent begins compressing them. Multiple
contingencies that produce similar effects become grouped into
transformation families.

Examples:

-   pushing,\
-   moving,\
-   opening,\
-   blocking,\
-   enabling.

The agent begins recognizing recurring patterns independent of specific
situations. This is the first significant step toward abstraction.

## **Level 6: Transformation Carriers**

Transformation families reveal persistent regularities. The agent
notices that many transformations appear connected. To explain these
recurring patterns, the agent infers stable carriers. A carrier is a
hypothesized persistent structure responsible for recurring
transformations.

Examples that may eventually emerge:

-   objects,\
-   agents,\
-   tools,\
-   obstacles.

Importantly, carriers are inferred. They are not directly observed.
Objects emerge as explanations.

## **Level 7: Functional Roles**

Once carriers exist, the agent discovers that different carriers often
participate in similar structures. A key, button, switch, password, and
lever may all occupy the same functional position. The transferable
structure is not the carrier itself. The transferable structure is the
role.

Examples include:

-   Enabler\
-   Blocker\
-   Access Provider\
-   Trigger\
-   Transporter\
-   Protector

Roles provide substantially greater transfer than appearance-based
categories.

## **Level 8: Concepts**

A concept is not merely a recurring pattern. A concept is a transferable
compression of interaction history. A structure qualifies as a concept
only if it improves behavior in previously unseen environments. Transfer
is therefore part of the definition. Without transfer there is pattern
recognition. With transfer there is concept formation.

## **Level 9: World Models**

World models emerge as large-scale explanatory structures.

They integrate:

-   concepts,\
-   roles,\
-   carriers,\
-   contingencies,\
-   future-option structures.

World models are therefore not prerequisites for intelligence. They are
late-stage products of intelligence. The agent first experiences the
world. Only later does it construct explanations of the world.

## **Level 10: Outcome-Equivalence Abstractions**

Once the learner has accumulated sufficiently rich consequence
structure, it can discover that perceptually different states or
distinct trajectory endpoints are functionally equivalent under a
declared learned consequence representation. The resulting
**outcome-equivalence abstraction** is a persistent memory structure
that represents *what was reached* independently of the particular
sequence that reached it.

Evidence contributing to an outcome descriptor may include:

-   bounded future-option structure,
-   continuation and reversibility structure,
-   reachable roles or concepts,
-   predictive consequence profiles,
-   contextual conditions,
-   externally observed terminal state retained as scientific evidence
    but not assigned intrinsic internal utility.

Outcome equivalence is not the same as a goal. Two outcomes can be
equivalent without being preferred. A **target-like outcome** appears
only if later evidence establishes a stable learned preference for
reaching that outcome class over available alternatives.

This layer creates the missing separation between an outcome
representation and the strategies that can produce it.

## **Level 11: Alternative Strategies, Replanning, and Efficiency Selection**

Distinct trajectories may become linked to the same learned
outcome-equivalence abstraction:

\[ `\pi`{=tex}\_A `\rightarrow`{=tex}`\Omega`{=tex},`\qquad`{=tex}
`\pi`{=tex}\_B `\rightarrow`{=tex}`\Omega`{=tex},`\qquad`{=tex}
`\pi`{=tex}\_C `\rightarrow`{=tex}`\Omega`{=tex}. \]

If one trajectory becomes unavailable or unreliable, planning can select
another empirically supported trajectory associated with the same
outcome class. Replanning therefore follows from the separation of
**outcome identity** from **strategy identity** rather than from a
primitive goal representation.

Efficiency does not precede outcome abstraction. It becomes meaningful
only when the learner can recognize that multiple trajectories lead to
the same learned outcome class or to outcomes that are explicitly
comparable. For every supported M7 strategy-to-M6 relation, the learner
maintains three independent empirical quantities: probability of
reaching the represented outcome, expected signed primary valence
associated with realized trajectories, and expected trajectory cost.
Efficient strategy selection favors lower cost only within admissible
comparison classes and does not substitute for either reliability or
valence. The minimal trajectory cost is the number of
actions/interactions from strategy initiation until the represented
outcome is reached. A non-zero primary-valence episode boundary
additionally supplies an observed actions-to-valence cost. Richer
declared cost models may include time, risk, uncertainty, blocked
actions, repeated states, or cognitive load.

A complete integrated world model may support outcome-equivalence
discovery but is not a strict prerequisite: sufficiently supported local
consequence structure may be enough to establish an outcome class. The
hierarchy therefore represents a typical developmental ordering rather
than an unconditional dependency on complete world-model formation.

The developmental sequence therefore extends as follows:

Observation → Transformation → Contingency → Prediction Violation →
Transformation Family → Carrier → Functional Role → Concept → World
Model / Consequence Structure → Outcome-Equivalence Abstraction →
Alternative Strategy Reuse → Planning/Replanning → Efficiency Selection

## **Promotion Principle**

Movement between levels is not automatic. Promotion occurs when a
structure provides greater explanatory reach than the structures beneath
it. Observation Deltas explain observations. Contingencies explain
deltas. Transformation Families explain contingencies. Roles explain
families. Concepts explain roles. World Models explain concepts. The
hierarchy is therefore driven by explanation rather than frequency.

## **Central Claim**

The developmental sequence proposed by this theory is:

Observation → Transformation → Contingency → Prediction Violation →
Transformation Family → Carrier

→ Functional Role → Concept → World Model / Consequence Structure →
Outcome-Equivalence Abstraction → Alternative Strategy Reuse

This ordering directly opposes object-first theories. The theory
predicts that transferable interaction structures emerge before object
categories and that role-based abstractions emerge before semantic
concepts.

# **5. The Interaction Significance Function**

The developmental hierarchy requires a resource-allocation mechanism
that determines which interactions receive additional storage, analysis,
replay, and abstraction effort. This paper calls that mechanism the
**Interaction Significance Function (ISF)**.

The ISF is not a value function or behavioral objective. Primary valence
is a separate motivational signal; the ISF estimates the expected
developmental importance of an interaction using evidence available at
the time of evaluation.

## **Why Significance Is Necessary**

Interaction streams contain far more detail than can receive equal
computational treatment. Repetition, noise, prediction violations, local
structural changes, and potentially transferable events must therefore
compete for limited memory and analysis resources.

The ISF addresses the operational question:

**Which interactions should receive additional developmental
processing?**

## **Core Principle**

Significance is estimated from motivational, structural, and epistemic
evidence without importing task-semantic labels. The framework
explicitly distinguishes **developmental significance** from **signed
primary valence**. A positive and a negative primary-valence event may
both be highly significant, while carrying opposite motivational
direction. Likewise, a large contraction or expansion of available
future interactions may be structurally significant independent of its
valence.

## **Proposed Components**

For interaction (i) at time (t):

\[ ISF(i,t)=F\_{ISF}!`\left`{=tex}( N\_{PVI}, N\_{OSI}, N\_{PE},
N\_{`\widehat{LV}`{=tex}}, N\_{TP\_{prior}}, N\_{`\widehat{EP}`{=tex}};
W_t `\right`{=tex}). \]

The components are:

  Component                Meaning
  ------------------------ -----------------------------------
  (PVI)                    Primary-Valence Impact
  (OSI)                    Option-Structure Impact
  (PE)                     Empirical Prediction Error
  (`\widehat{LV}`{=tex})   Prospective Learning Value
  (TP\_{prior})            Prospective Transfer Prior
  (`\widehat{EP}`{=tex})   Prospective Explanatory Potential

(N\_\*) denotes bounded normalization within an appropriate
developmental and memory comparison class. (W_t) contains
stage-dependent weights.

The exact aggregation function is an empirical design choice. The theory
requires causal availability, bounded normalization, and preservation of
the unaggregated evidence vector; it does not require universal fixed
coefficients.

### **Normalization and Evidence Integrity**

Heterogeneous evidence channels must not be combined on arbitrary raw
scales. Implementations should use bounded robust normalization such as
quantile normalization, rank normalization, or clipped robust z-scores.
Every score should preserve:

-   raw component values,
-   normalized values,
-   evidence availability time,
-   score time,
-   developmental-stage snapshot,
-   next inferred developmental stage,
-   immutable score-schema/version identity.

No quantity observed only after a decision may be used retroactively as
if it had been available before that decision.

## **Component 1: Primary-Valence Impact**

Primary-Valence Impact captures the bounded magnitude of motivational
evidence available at the interaction while preserving its sign
separately:

\[ PVI(i,t)=g_V(\|V_t\|), `\qquad`{=tex} DirV(i,t)=sign(V_t). \]

For delayed consequences, the same channel may receive causally
attributed trajectory credit, for example:

\[ Credit_V(d)=V_t `\gamma`{=tex}\^d, `\qquad`{=tex}
0\<`\gamma`{=tex}\<1. \]

The sign is never collapsed into significance: negative primary valence
can be highly significant. Attribution must follow the experienced
causal trajectory and must not alter structural memory identity.

## **Component 2: Option-Structure Impact**

At the earliest stage, the learner does not know whether a transition is
intrinsically good or bad. It can, however, observe whether the local
interaction structure changed. Primitive evidence may include:

-   interaction continuation or cessation,
-   changes in immediately available actions,
-   appearance or disappearance of accessible transformations,
-   demonstrated reversibility,
-   entry into an empirically absorbing interaction pattern.

Let (OE_0(i)) denote this model-free option evidence. A Stage-0
option-structure impact can be defined as a bounded distance:

\[
OSI_0(i)=d\_{OE}!`\left`{=tex}(OE_0^{before},OE_0^{after}`\right`{=tex}).
\]

After a learned memory graph exists, the estimate can incorporate
bounded future-option change:

\[
OSI_t(i)=g!`\left`{=tex}(`\left`{=tex}\|`\Delta`{=tex}`\widehat{FO}`{=tex}\_t\^k(i)`\right`{=tex}\|`\right`{=tex}),
\]

where (g) is bounded and the sign of
(`\Delta`{=tex}`\widehat{FO}`{=tex}) is retained separately.

Primary valence does not determine (OSI). A terminal transition can
therefore have zero, positive, or negative future-option change
independently of its motivational sign.

## **Component 3: Prediction Error**

Prediction error is defined only after a supported contingency produces
an expectation:

\[ PE(i,t)=d\_{`\Delta`{=tex}} `\left`{=tex}( `\Delta`{=tex}\_t,
`\widehat{\Delta}`{=tex}\_t `\right`{=tex}). \]

If no sufficiently supported expectation exists, prediction error is
inactive rather than maximal. This avoids circularly using prediction
violations to create the very contingencies required to generate
predictions.

Prediction error allocates resources toward:

-   unexpected transformations,
-   unresolved conditional structure,
-   candidate context variables,
-   obsolete or overly broad abstractions.

## **Component 3: Prospective Learning Value**

Prospective learning value estimates expected information or explanatory
gain from further processing:

\[ `\widehat{LV}`{=tex}(i,t) = `\mathbb{E}`{=tex} \[
`\Delta`{=tex}Knowledge `\mid`{=tex}
`\mathcal{E}`{=tex}\_{`\le`{=tex}t}\]. \]

The implementation may operationalize knowledge gain through entropy
reduction, predictive improvement, contradiction resolution, graph
expansion, or another explicitly declared measure.

Realized learning value may be measured later, but only later decisions
may use it.

## **Component 4: Transfer Prior**

(TP\_{prior}) estimates whether a structure is worth testing outside its
formation context. It may use evidence such as:

-   recurrence across separated contexts,
-   structural invariance,
-   graph-neighborhood similarity,
-   compression across independent episodes.

It cannot validate transfer. Validation requires held-out empirical
reuse.

## **Component 5: Explanatory Potential**

(`\widehat{EP}`{=tex}) estimates the future explanatory reach of an
interaction or derived structure using only currently available
evidence. Realized explanatory reach is measured retrospectively after
descendants or supported structures exist.

## **Developmental Reweighting**

The relative importance of components changes with established
capabilities.

### **Stage 0: Interaction Seeding**

Dominant evidence:

-   recurrence and novelty outside the ISF aggregation,
-   immediate option-structure impact,
-   primary-valence impact when present.

Prediction error is inactive until supported contingencies exist.

### **Stage 1: Contingency Learning**

Dominant evidence:

-   prediction error,
-   learning value,
-   immediate option-structure impact.

### **Stage 2: Structural Abstraction**

Dominant evidence:

-   learning value,
-   explanatory potential,
-   transfer prior.

### **Stage 3: Transfer Development**

Dominant evidence:

-   transfer prior,
-   explanatory potential,
-   empirical transfer evidence for later retention decisions.

### **Stage 4: Consequence Integration and Initial Planning**

Dominant evidence increasingly includes learned future-option structure,
consequence relations, explanatory reach, and validated transfer.

### **Stage 5: Outcome-Equivalence and Preference Development**

Persistent outcome classes and learned preference relations over
reachable outcomes become available. Outcome identity remains separate
from strategy identity, reliability, primary valence, and cost.

### **Stage 6: Alternative-Strategy Linkage**

Multiple grounded strategies may be linked to the same learned M6
outcome or to an explicitly admissible outcome-comparison class. The
system can represent substitutability, but demonstrated online
replanning is not yet required.

### **Stage 7: Replanning and Strategy Efficiency**

The system demonstrates substitution among alternative strategies and
may prefer lower-cost trajectories when they preserve the selected
learned outcome or an explicitly no-worse outcome. Efficiency evidence
is meaningful only inside such outcome-comparison classes.

Stage transitions are inferred from already-established capability
evidence. Evidence generated during an interval may affect the next
stage but cannot change the weights that generated that same evidence.

## **Significance and Memory Selection**

Higher ISF increases the probability that an interaction receives:

-   retention,
-   replay,
-   deeper context analysis,
-   candidate abstraction construction,
-   transfer testing.

Low-significance records may be summarized or removed once higher-level
structures preserve their unique explanatory contribution.

## **Relationship to Future-Option Structure**

Future-option structure is not logically prior to memory. Stage-0 memory
is seeded by directly observed (OE_0). Learned multi-step future-option
estimates become available only after contingencies and graph structure
exist.

The developmental dependency is:

\[ Interaction `\rightarrow`{=tex} OE_0 `\rightarrow`{=tex}
Initial Memory `\rightarrow`{=tex} Contingencies `\rightarrow`{=tex}
Memory Graph `\rightarrow`{=tex} `\widehat{FO}`{=tex}\^{,k}
`\rightarrow`{=tex} Richer Significance. \]

Future-option structure therefore becomes increasingly important without
being smuggled in as an innate objective.

## **Central Claim**

The ISF is a resource-allocation hypothesis: interactions differ in the
amount of developmental processing they receive. The scientific claim is
that a bounded combination of structural impact, prediction error,
prospective learning value, transfer prior, and explanatory potential
should produce more transferable and compressive memory organization
than frequency- or novelty-only allocation.

# **6. Memory-Centric Intelligence and Multi-Scale Memory**

The framework assigns memory organization a primary explanatory role. It
investigates whether many capabilities usually modeled as separate
objectives can arise from decisions about:

-   what to remember,\
-   what to forget,\
-   what to compress,\
-   what to promote,\
-   what to reuse,\
-   what to explain.

Behavior emerges from memory organization. The hypothesis assigns memory
organization a primary explanatory role rather than treating memory as a
passive support system.

## **Memory-Centric Intelligence**

Many architectures treat memory primarily as storage serving prediction,
planning, or policy optimization. The present framework instead treats
the evolving memory substrate as the common state through which
learning, prediction, transfer, contextual refinement, planning, and
abstraction interact. The scientific question is whether this
organization is sufficient to generate the predicted developmental
sequence.

## **Continuous Restructuring**

Memory is not a static archive. It is a continuously evolving structure.
New experiences can:

-   reinforce existing structures,\
-   weaken existing structures,\
-   create new structures,\
-   merge structures,\
-   split structures,\
-   reorganize relationships,\
-   reveal missing context.

Learning therefore never truly stops. Intelligence is not convergence
toward a final representation. Intelligence is continuous memory
reorganization.

## **Multi-Scale Memory**

The theory proposes that memory exists simultaneously at multiple levels
of abstraction. Different memory scales support different cognitive
functions.

### **M0: Episodic Memory**

Stores individual experiences.

Examples:

-   specific observations,\
-   specific actions,\
-   specific outcomes,\
-   specific interaction histories.

Characteristics:

-   high detail,\
-   low compression,\
-   low transfer.

Purpose:

Preserve raw experience.

### **M1: Contingency Memory**

Stores recurring action-consequence relationships.

Examples:

-   move right → position changes,\
-   touch red → termination,\
-   click switch → state changes.

Characteristics:

-   recurring structure,\
-   expectation formation,\
-   prediction support.

Purpose:

Capture stable regularities.

### **M2: Interaction Families**

Stores compressed collections of contingencies.

Examples:

-   movement,\
-   activation,\
-   blocking,\
-   transformation,\
-   transport.

Characteristics:

-   reduced specificity,\
-   increased reuse,\
-   greater transfer potential.

Purpose:

Support abstraction across situations.

### **M3: Functional Roles**

Stores reusable structural positions within interaction graphs.

Examples:

-   Enabler\
-   Blocker\
-   Trigger\
-   Access Provider\
-   Connector\
-   Transport Mechanism

Characteristics:

-   highly transferable,\
-   appearance-independent,\
-   environment-independent.

Purpose:

Enable transfer across domains.

### **M4: Concepts**

Stores durable transferable abstractions.

A concept is not merely a pattern.

A concept is a structure that improves performance in previously unseen
environments.

Characteristics:

-   high compression,\
-   high explanatory power,\
-   high transfer value.

Purpose:

Generalization.

### **M5: World-Model and Consequence Structures**

Stores higher-order explanatory and predictive relations among concepts,
roles, contingencies, and learned future-option structure. A complete
integrated world model is not required before all later abstractions;
local consequence structure may mature earlier.

Purpose:

Support large-scale relational organization and learned consequence
description.

### **M6: Outcome-Equivalence and Strategy Structures**

Stores two explicitly distinct kinds of late memory:

-   **Outcome-equivalence abstractions** representing recurrent classes
    of consequentially similar future states or trajectory endpoints.
-   **Strategy structures** representing trajectories or procedures
    empirically associated with reaching those outcome classes.

The separation is essential: an outcome identity can remain stable when
a particular strategy fails, allowing replanning through another
strategy linked to the same outcome abstraction.

A target-like outcome is not a primitive memory type. It is a learned
preference relation over outcome-equivalence abstractions that becomes
stable enough to act as a planning reference.

## **Cross-Modal Memory Integration**

The memory hierarchy is not duplicated by modality. M0 may retain
modality-tagged episodes and M1 may retain modality-specific or
cross-modal contingencies, but promotion to M2--M6 remains governed by
the same criteria of compression, explanatory reach, empirical reuse,
transfer, and consequence structure.

The theory therefore predicts convergence rather than parallel semantic
hierarchies:

\[ `\text{interaction evidence}`{=tex} ;`\cup`{=tex};
`\text{symbolic evidence}`{=tex} `\rightarrow`{=tex}
`\text{shared families}`{=tex} `\rightarrow`{=tex}
`\text{shared functional roles}`{=tex} `\rightarrow`{=tex}
`\text{shared concepts}`{=tex}. \]

A higher-order memory may retain mixed provenance. Its identity is
determined by learned relational and functional structure, not by the
sensory modality from which supporting evidence originated.

### **No Semantic Shortcut**

No pretrained embedding, language model, dictionary, ontology, synonym
table, semantic parser, or manually supplied symbol-to-world mapping may
provide the cross-modal relation under the core experiment. Tokenization
may establish stable symbol identity and order, but not meaning.

## **Memory Promotion**

Movement between memory scales is not automatic. Promotion requires
evidence. A structure must justify its existence. The theory proposes
that promotion occurs when explanatory reach increases.

### **Episode → Contingency**

Multiple experiences reveal a recurring regularity.

### **Contingency → Interaction Family**

Multiple contingencies reveal a common transformation pattern.

### **Interaction Family → Functional Role**

Multiple transformation patterns reveal a common structural function.

### **Functional Role → Concept**

Multiple roles reveal a transferable abstraction.

## **Promotion Through Explanatory Reach**

Frequency alone is insufficient. A frequently observed event may explain
very little. A rarely observed event may explain an entire class of
experiences. The theory proposes that structures are promoted because
they explain lower-level structures. The value of a memory is therefore
linked to explanatory power rather than observation count.

## **Selective Forgetting**

Forgetting is not a limitation. Forgetting is a mechanism of
abstraction. If every experience remains equally important:

-   memory grows indefinitely,\
-   abstraction becomes difficult,\
-   transfer becomes inefficient.

Selective forgetting removes:

-   redundancy,\
-   noise,\
-   superseded structures,\
-   unnecessary detail.

Compression requires forgetting. Abstraction requires forgetting.
Concept formation requires forgetting.

## **Memory Competition**

Memories compete for persistence. Competition is influenced by:

\* Interaction Significance,

\* recurrence,

\* future-option impact,

\* explanatory reach,

\* transfer value,

\* contextual relevance,

\* interaction efficiency when outcomes are comparable.

Efficiency enters memory competition only after the system can compare
trajectories that produce equivalent or comparable outcomes. If two
trajectories produce the same future-option structure, the lower-cost
trajectory receives higher memory fitness. If a longer trajectory opens
new possibilities, reveals new contingencies, or reduces risk, it may
remain preferable despite requiring more actions.

Not all memories survive. Only a small subset become candidates for
higher-level abstraction. This competition creates developmental
pressure toward increasingly useful structures.

------------------------------------------------------------------------

## **Shared Memory as Cognitive Infrastructure**

All cognitive processes operate on the same evolving memory substrate.
Exploration writes to memory. Prediction reads from memory. Attention
modifies memory. Transfer reuses memory. Planning depends on memory.
Concept formation reorganizes memory. The shared memory substrate
becomes the mechanism through which otherwise independent processes
remain coherent. Coherence is hypothesized to emerge from shared memory
together with common evidence, consistency, publication, and resource
constraints rather than from a single global behavioral objective.

**Central Claim**

The framework tests whether continuous restructuring of memory across
multiple abstraction scales can serve as a developmental substrate from
which prediction, transfer, planning, and control emerge without
assuming one fixed global behavioral objective. Episodes become
contingencies. Contingencies become interaction families. Interaction
families become roles. Roles become concepts. Concepts become components
of world models. The emergence of intelligence is therefore the
emergence of increasingly powerful memory structures.

# **7. Concurrent Shared-Memory Update Architecture**

The theory requires several distinct update processes over a common
developmental memory substrate. The scientific object is a set of
**concurrent specialized update operators over shared memory**.

## **Why Multiple Update Processes Are Required**

Development imposes partially conflicting computational pressures:

-   exploration increases coverage,
-   prediction estimates regularity,
-   compression removes redundancy,
-   context refinement adds conditional structure,
-   transfer testing evaluates reuse,
-   pruning removes low-utility representation,
-   planning evaluates known paths,
-   efficiency estimation compares equivalent outcomes.

The theory does not assume that these processes optimize one global
behavioral objective.

However, unconstrained asynchronous modification of a shared graph could
produce race conditions, oscillation, or representational thrashing.
Shared memory alone is therefore not sufficient for coherence.

## **Developmental State**

Let the developmental state be:

\[ D_t=(M_t,G_t,W_t,`\mathcal{E}`{=tex}\_t), \]

where:

-   (M_t) is the multi-scale memory state,
-   (G_t) is the typed interaction graph,
-   (W_t) is the current developmental weighting state,
-   (`\mathcal{E}`{=tex}\_t) is the append-only scientific
    evidence/provenance ledger.

A specialized process (j) proposes an update through:

\[ U_j(D_t,`\xi`{=tex}*t)`\rightarrow`{=tex}`\widetilde{D}`{=tex}*{t+1},
\]

where (`\xi`{=tex}\_t) is the evidence visible to that process.

The update is accepted only if it satisfies the shared admissibility
constraints (`\mathcal{C}`{=tex}):

\[ D\_{t+1} = `\Pi`{=tex}*{`\mathcal{C}`{=tex}} `\left`{=tex}(
`\widetilde{D}`{=tex}*{t+1} `\right`{=tex}). \]

(`\Pi`{=tex}\_{`\mathcal{C}`{=tex}}) denotes validation, conflict
resolution, and atomic publication of an admissible next generation. It
is not a global reward optimizer.

## **Shared Admissibility Constraints**

At minimum, implementations should enforce:

### **C1 --- Provenance Preservation**

Scientific evidence and derivation ancestry cannot be silently rewritten
or destroyed by later abstraction.

### **C2 --- Candidate/Validation Separation**

Structural recurrence may create a candidate, but scientific validation
requires the evidence contract appropriate to that abstraction level.

### **C3 --- Causal Availability**

An update may use only evidence available at the time of that update.

### **C4 --- Resource Bounds**

Candidate generation, graph expansion, replay, and comparison must
operate under explicit computational budgets.

### **C5 --- Hysteresis**

Promotion, demotion, and retirement require sustained evidence to
prevent oscillation from short-lived fluctuations.

### **C6 --- Atomic Generations**

Concurrent processes operate on immutable published memory generations
and publish validated updates atomically. Partial cross-process state is
not exposed as a coherent generation.

### **C7 --- Context Before Destructive Replacement**

Contradictory evidence first triggers contextual refinement when an
explanatory split is available. Destructive replacement requires
stronger evidence.

These constraints provide system-level coherence without postulating one
shared objective.

## **Specialized Update Processes**

The reference architecture contains the following process classes.

### **Exploration Process**

Allocates interaction effort toward insufficiently sampled actions,
states, or graph regions.

### **Prediction-Violation Estimator**

Compares supported local expectations with observed transformations and
emits prediction-error evidence.

### **Memory Formation Process**

Converts selected interaction evidence into persistent low-level memory.

### **Compression Process**

Searches bounded candidate neighborhoods for recurring structure and
proposes higher-level representations.

Minimum Description Length may be used locally here as an implementation
criterion, but MDL is not assumed to be the objective of the entire
cognitive system.

### **Context-Refinement Process**

Searches for evidence-supported partitions that reduce contradiction or
predictive error.

# 

## **Local Developmental Regime Re-Entry**

The transition from pre-predictive to predictive significance is not
assumed to be irreversible.

A mature system may encounter a context in which previously supported
expectations cease to explain observations because environmental
dynamics, contextual conditions, or relevant structural relationships
have changed.

Regime state is therefore scoped to a memory/context pair rather than
imposed globally.

For an established contingency (C) in context (X), persistent
unexplained prediction violation may trigger:

\[ Predictive(X,C) `\rightarrow`{=tex} Uncertain(X,C)
`\rightarrow`{=tex} PrePredictive(X,C). \]

A candidate re-entry criterion is:

\[ PE(X,C) `\ge`{=tex}`\theta`{=tex}\_{`\mathrm{surprise}`{=tex}} \]

for at least (n\_{`\mathrm{shift}`{=tex}}) sufficiently independent
observations, subject to contradiction and context-refinement checks.

Re-entry suspends the authority of the obsolete expectation in the
affected context and reactivates pre-predictive significance:

\[ ISF\_{`\mathrm{cold}`{=tex}}(I_t `\mid`{=tex}X). \]

It does **not** erase the previous memory. The prior contingency remains
available as historical evidence, a candidate context-specific rule, and
a possible explanation if the earlier dynamics recur.

The intended sequence is:

``` text
supported expectation
        ↓
persistent unexplained violation
        ↓
local uncertainty
        ↓
context refinement fails to explain change
        ↓
local pre-predictive re-entry
        ↓
new recurrence discovery
        ↓
new supported expectation
```

This mechanism permits adaptation to non-stationary environments without
requiring global developmental reset or catastrophic forgetting.

Required measurements include:

-   false re-entry rate under ordinary stochastic noise;
-   detection delay after genuine structural change;
-   time to formation of a replacement contingency;
-   retention of the superseded contingency;
-   recovery when earlier dynamics recur.

### **Dependency-Scoped Causal Suspension**

Local regime re-entry must not permit an uncertain causal parent to
continue supporting descendants with unchanged authority.

When a contingency (C) enters the `Uncertain` or local `PrePredictive`
regime, outgoing causal/developmental dependencies whose validity relies
on (C) enter a suspended state:

\[ ACTIVE(C `\rightarrow`{=tex}D) `\rightarrow`{=tex} SUSPENDED(C
`\rightarrow`{=tex}D). \]

Suspension is dependency-scoped rather than descendant-wide.

A descendant (D) may remain active if it retains sufficient independent
support from other admissible evidence:

\[ Support(D `\setminus`{=tex}C)
`\ge`{=tex}`\theta`{=tex}\_{`\mathrm{support}`{=tex}}. \]

Otherwise the descendant enters a probationary state until the affected
dependency is revalidated, replaced, or rejected.

``` text
active parent
     ↓ persistent unexplained violation
uncertain parent
     ↓
dependent edge suspended
     ↓
descendant has independent support?
     ├── yes → descendant remains active
     └── no  → descendant enters probation
```

New contingencies formed during local re-entry establish a distinct
causal lineage. They must not silently inherit authority, transfer
evidence, or validated descendants from the suspended lineage.

If the previous environmental dynamics recur and the old contingency is
empirically revalidated, suspended dependencies may reactivate without
reconstructing the historical lineage from scratch.

This mechanism preserves historical knowledge while preventing
incompatible old and newly developing causal structures from
simultaneously exercising unqualified authority.

## **Progressive Multi-Scale Structural Correspondence**

Bounded candidate generation is necessary for computational
tractability, but a fixed local neighborhood is not assumed sufficient
for functional correspondence.

For memory (X), define structural descriptors over progressively larger
neighborhood scales:

\[ D_r(X), `\qquad`{=tex}r
`\in`{=tex}{1,2,4,8,`\ldots`{=tex},r\_{`\max`{=tex}}}. \]

Comparison proceeds adaptively:

``` text
cheap local retrieval
        ↓
candidate structural match
        ↓
progressively expanded comparison
        ↓
formal correspondence
        ↓
held-out intervention
        ↓
causal transfer evidence
```

Computation allocated to a candidate increases only when smaller-scale
evidence indicates explanatory or transfer potential. Candidate
retrieval therefore remains bounded while deep structural
correspondences remain discoverable.

This mechanism makes structural search an empirical resource-allocation
process rather than a fixed-radius assumption.

A required evaluation varies the dependency depth separating
functionally equivalent structures across environments and measures
correspondence recall, computational cost, and causal transfer success
as a function of structural depth.

### **Scale-Normalized Comparison**

Descriptors computed at different structural radii are not directly
assumed to be comparable. Larger neighborhoods contain more nodes,
paths, relation types, and dependency alternatives and would otherwise
dominate similarity scores through scale alone.

For scale (r), define:

\[ Sim_R\^{(r)}(X,Y) = `\sum`{=tex}\_k w_k(r),
`\widehat{S}`{=tex}\_k\^{(r)}(X,Y), \]

where (`\widehat{S}`{=tex}\_k\^{(r)}) is component (k) normalized
relative to the structural scale and (w_k(r)) is an explicitly declared
scale-dependent weight.

Normalization may account for quantities such as:

-   neighborhood size;
-   typed-edge opportunity count;
-   path count;
-   dependency depth;
-   relation-frequency baseline;
-   descriptor entropy.

The theory does not require one universal normalization function. It
requires the normalization rule to be declared, fixed within an
experimental condition, and independently ablated.

### **Evidence-Based Expansion and Stopping**

Structural radius does not automatically expand through all available
scales.

Let:

\[ `\Delta`{=tex}I_r(X,Y) \]

denote the incremental discriminatory or explanatory information
obtained by expanding from the preceding scale to (r).

Expansion continues only while at least one of the following holds:

\[ `\Delta`{=tex}I_r(X,Y) \> `\theta`{=tex}\_{`\mathrm{expand}`{=tex}},
\]

the correspondence remains materially ambiguous, or the expected
transfer value justifies additional computation.

Search stops when:

-   correspondence is sufficiently discriminated;
-   incremental structural information falls below threshold;
-   the candidate becomes inadmissible; or
-   the declared computation budget is exhausted.

Thus:

``` text
small-scale candidate
        ↓
normalize at current scale
        ↓
measure incremental information
        ↓
useful / ambiguous? ── yes ──> expand
        │
        no
        ↓
       stop
```

This prevents both fixed-radius blindness and unconditional exponential
neighborhood expansion.

### **Structural Ambiguity and Candidate Entropy**

Incremental structural information is operationally grounded by the
ambiguity of the current candidate correspondence set.

For source memory (X) and candidate set (C_r(X)) at structural scale
(r), derive a normalized candidate plausibility distribution:

\[ p_r(Y `\mid`{=tex}X), `\qquad`{=tex}Y `\in`{=tex}C_r(X). \]

Define structural candidate entropy:

\[ H_r(X) = -`\sum`{=tex}\_{Y `\in`{=tex}C_r(X)} p_r(Y
`\mid`{=tex}X)`\log`{=tex}p_r(Y `\mid`{=tex}X). \]

High entropy indicates that several candidates remain structurally
plausible. Low entropy indicates that available evidence discriminates a
small number of candidates, ideally one.

The reference operational definition of incremental structural
information is entropy reduction:

\[ `\Delta`{=tex}I_r(X) = H\_{r/2}(X)-H_r(X). \]

Structural expansion is justified when additional scale continues to
reduce ambiguity:

\[ `\Delta`{=tex}I_r(X) \> `\theta`{=tex}\_{`\mathrm{expand}`{=tex}}, \]

or when unresolved ambiguity remains:

\[ H_r(X) \> `\theta`{=tex}\_{`\mathrm{ambiguity}`{=tex}}. \]

Therefore:

``` text
candidate set
     ↓
candidate plausibility distribution
     ↓
structural entropy
     ↓
low ambiguity? ── yes ──> stop
     │
     no
     ↓
expand structural scale
     ↓
measure entropy reduction
```

Topological symmetry is thereby represented explicitly: if two or more
candidate structures remain indistinguishable at the current scale,
entropy remains elevated and additional structural evidence may be
requested.

Candidate entropy is the reference operationalization, not a mandatory
universal information metric. Alternative declared measures such as
description-length reduction or other statistical discrimination
criteria may be experimentally compared, provided their stopping
behavior and computational cost are reported.

### **Structural Symmetry and Equivalence-Class Stopping**

Persistent candidate ambiguity is not necessarily evidence that
structural search has failed.

Two or more candidates may be indistinguishable under all structural
evidence available at the current developmental state. In that case,
increasing the comparison radius may cease to provide discriminatory
information even while candidate entropy remains above the ordinary
ambiguity threshold.

A symmetry condition is therefore:

\[ H_r(X) \> `\theta`{=tex}\_{`\mathrm{ambiguity}`{=tex}} \]

while:

\[ `\Delta`{=tex}I_r(X)
`\le`{=tex}`\theta`{=tex}\_{`\mathrm{symmetry}`{=tex}} \]

for a declared number of successive admissible scale expansions, or
until the declared structural budget is exhausted.

Under this condition the system may retain:

\[ E_r(X)={Y_1,Y_2,`\ldots`{=tex},Y_n} \]

as a structural equivalence class rather than forcing selection of a
unique candidate.

``` text
persistent structural ambiguity
        ↓
additional scale adds no discrimination
        ↓
retain equivalence class
        ↓
future intervention / consequence evidence
        ↓
equivalence preserved or broken
```

Equivalence-class retention is not semantic identity. It records only
that the current evidence does not justify distinguishing the candidates
for the relevant structural purpose.

When possible, future intervention should provide the decisive evidence.
If members of the equivalence class produce distinguishable consequences
under admissible interventions, the class is refined. If they remain
consequence- equivalent across relevant held-out interventions,
retaining the equivalence may itself be the correct abstraction.

This rule prevents topological symmetry from causing unconditional
expansion to (r\_{`\max`{=tex}}) and reinforces the interaction-first
principle: when passive structural evidence cannot discriminate
hypotheses, causal interaction may be required.

### **Online Scale-Normalization Statistics**

Scale normalization must not require exact recomputation of global
statistics over the continuously evolving memory graph.

For structural component (k) at scale (r), the system may maintain
bounded online estimates such as:

\[ `\hat{\mu}`{=tex}*{k,r}, `\qquad`{=tex} `\hat{\sigma}`{=tex}*{k,r},
`\qquad`{=tex} `\hat{H}`{=tex}\_{k,r}, \]

representing estimated central tendency, dispersion, and background
descriptor entropy derived from previously observed neighborhoods.

A normalized component may therefore take the form:

\[ `\widehat{S}`{=tex}*k\^{(r)} = Normalize `\left`{=tex}( S_k\^{(r)};
`\hat{\mu}`{=tex}*{k,r}, `\hat{\sigma}`{=tex}*{k,r},
`\hat{H}`{=tex}*{k,r} `\right`{=tex}). \]

These estimates may be updated incrementally as admissible structural
neighborhoods are encountered. Exact global graph recomputation is not
required.

The normalization mechanism must satisfy four constraints:

1.  its update rule is explicitly declared;
2.  memory and update cost are bounded;
3.  statistics are indexed by the structural scale and relevant
    descriptor type;
4.  approximation error or estimator drift can be measured
    experimentally.

This preserves continuous-memory runtime constraints while allowing the
statistical background against which structural similarity is judged to
develop from accumulated experience.

## **Transfer-Test Process**

Schedules held-out structural and behavioral tests for candidate roles,
concepts, world-model components, and strategies.

### **Functional-Role Estimator**

Identifies recurring typed relational positions across distinct carriers
or contexts.

### **Concept-Validation Process**

Evaluates whether structurally supported abstractions produce empirical
held-out reuse.

### **Future-Option Estimator**

Estimates bounded reachability over the currently discovered interaction
graph.

### **Outcome-Equivalence Estimator**

Constructs and updates persistent outcome classes from recurring learned
consequence descriptors. It keeps outcome identity separate from the
trajectories or procedures that produced it.

### **Target-Preference Estimator**

Measures whether the learner develops a stable comparative preference
for one reachable outcome class over alternatives. This process does not
assign intrinsic desirability; it records preference only when supported
by repeated behavior and downstream evidence.

### **Planning Process**

Uses validated memory to compare known or inferred interaction
trajectories and may select among alternative strategies linked to the
same target-like or otherwise selected outcome abstraction.

### **Trajectory-Efficiency Estimator**

Compares cost only within learned outcome-equivalent or explicitly
comparable trajectory classes.

### **Retention and Pruning Process**

Applies fitness-weighted retention, quarantine, compression, and
retirement while preserving scientific provenance.

## **Asynchrony Without a Global Objective**

Different processes may update at different frequencies and may
temporarily favor opposing representational changes. For example,
exploration may add detail while compression removes redundancy.

The framework does **not** claim guaranteed convergence. Instead it
makes a testable stability claim:

> Under versioned publication, bounded candidate search, evidence
> contracts, context-first contradiction handling, hysteresis, and
> provenance-preserving pruning, concurrent update processes should
> avoid persistent destructive oscillation while continuing to
> reorganize memory.

A failure to achieve such bounded developmental stability would count
against this architectural hypothesis.

## **Shared Memory as Communication Substrate**

Processes communicate primarily through the published memory graph and
evidence ledger rather than through a privileged executive controller. A
prediction violation can therefore trigger context refinement because
both processes refer to the same evidence-linked memory identities.

The stronger statement is:

> Coherence is hypothesized to emerge from a shared evolving memory
> substrate **together with common evidence, consistency, publication,
> and resource constraints**; a single global behavioral objective is
> not required.

## **Developmental Dominance**

The relative computational budgets of these processes may change with
developmental stage.

Early development emphasizes exploration, recurrence, memory formation,
and local prediction.

Intermediate development emphasizes compression, context refinement,
carrier persistence, and bounded future-option estimation.

Later development emphasizes transfer testing, concept validation,
world-model organization, planning, and trajectory efficiency.

These shifts are resource-allocation changes rather than changes to
scientific validity criteria.

## **Central Claim**

The architecture is a constrained asynchronous dynamical system over
shared memory. Multiple specialized estimators may reorganize different
aspects of the same memory graph, but all accepted changes obey common
evidence and consistency constraints. This provides a falsifiable
alternative to both a monolithic controller and an unconstrained
"society" metaphor.

# **8. Transformation Primacy, Carrier Emergence, and Role-Before-Object**

A central claim of this theory is that stable entities are not the first
abstractions learned by an intelligent system. The agent first learns
transformations. Only later does it infer the existence of entities that
participate in those transformations. This reverses the conventional
object-first view of cognition.

## **Transformation Primacy**

The agent does not initially experience objects. The agent experiences
change. Its earliest memories consist of action-conditioned
transformations:

Observation(t) → Action(t) → \[O(t,1), ..., O(t,m_t)\] →
Observation(t+1), with V(t) ∈ {-1,0,+1}

The first regularities discovered are therefore transformation
regularities.

Examples:

-   movement,\
-   appearance,\
-   disappearance,\
-   activation,\
-   blockage,\
-   termination.

At this stage the agent has no reason to assume the existence of
persistent entities. It only observes recurring patterns of change. The
theory therefore proposes: What changes is learned before what exists.

## **Why Transformations Come First**

Transformations are directly observable. Objects are not. The agent can
directly observe:

-   visual changes,\
-   spatial changes,\
-   accessibility changes,\
-   interaction changes.

The existence of a stable object must be inferred. This creates an
asymmetry: Transformations are data. Objects are explanations. The
theory predicts that systems built from interaction histories will
discover recurring transformations before discovering recurring
entities.

## **Carrier Emergence**

As experience accumulates, certain transformations repeatedly appear
connected. Consider a simplified example. The agent repeatedly observes:

RIGHT → Δ₁

LEFT → inverse(Δ₁)

UP → Δ₂

DOWN → inverse(Δ₂)

The agent repeatedly encounters the same transformation structure.
Eventually a simpler explanation becomes available: A persistent carrier
exists that participates in these transformations. The carrier
hypothesis compresses many observations into a single explanatory
structure. The inferred carrier is what humans would later describe as
an object or agent.

## **Objects as Explanatory Compression**

In this theory, objects are not observations. Objects are compression
mechanisms. A persistent object is introduced when it explains recurring
transformation patterns more efficiently than independent memories.
Object formation therefore becomes an explanatory process rather than a
perceptual process. The developmental sequence becomes:

Transformations → Transformation Families → Carrier Hypotheses →
Object-Like Structures

Objects emerge because they simplify memory organization.

## **Carrier Stability**

Not all inferred carriers survive. A carrier must justify its existence.
A proposed carrier survives only if it improves:

-   prediction,\
-   compression,\
-   transfer,\
-   explanation.

Otherwise the carrier is discarded. This creates competitive pressure
among candidate representations. The ontology remains grounded in
interaction rather than appearance.

## **Role-Before-Object Principle**

Even after carriers emerge, another level of abstraction becomes more
important. Different carriers often participate in identical interaction
structures.

Examples:

-   key,\
-   switch,\
-   lever,\
-   button,\
-   password,\
-   access card.

Their appearances may be entirely different. Their physical
implementations may be unrelated. Yet they frequently occupy the same
functional position. The transferable abstraction is not the carrier.
The transferable abstraction is the role.

## **Functional Roles**

A role is a recurring structural position within an interaction graph.

Examples include:

### **Enabler** Expands future possibilities. Allows access to previously inaccessible interactions.

### **Blocker** Restricts future possibilities. Prevents progression.

### **Trigger** Initiates state transitions. Creates new interaction opportunities.

### **Connector** Links otherwise disconnected regions of experience.

### **Transporter** Moves the agent or other carriers between regions of the graph.

### **Continuation-Collapse Role** Is associated with transitions into empirically low-reachability or non-continuing regions.

## **Why Roles Transfer Better Than Objects**

Suppose an agent learns about keys. A traditional system may learn: This
object is a key. A role-based system instead learns: This structure
functions as an Enabler. Later environments may contain:

-   switches,\
-   passwords,\
-   buttons,\
-   codes,\
-   levers.

Although visually different, they occupy the same role. The role
transfers. The appearance does not. The theory predicts that successful
transfer will correlate more strongly with role similarity than object
similarity.

## **Graph Context and Role Identity**

Roles cannot be defined in isolation. Their identity emerges from graph
structure. A key is not a key because of its appearance. A key is not
even a key because it opens a door. A key is a key because it occupies a
position within a larger interaction structure such as:

Acquire → Unlock → Access → Progress

Meaning therefore emerges from graph context rather than intrinsic
properties.

## **Emergence of Concepts**

Role discovery provides the bridge from interaction patterns to
concepts. Once roles become reusable across environments, increasingly
general abstractions become possible. The progression becomes:

Carrier → Role → Concept

Rather than:

Object → Concept

This distinction is critical. The theory predicts that transferable
functional roles emerge before transferable object categories.

## **Predictions**

This framework produces several testable predictions:

1.  Stable transformation structures emerge before stable object
    representations.\
2.  Carrier hypotheses emerge after transformation families.\
3.  Functional-role similarity predicts transfer better than perceptual
    similarity.\
4.  Role transfer emerges before object-category transfer.\
5.  Similar graph positions predict transfer better than similar
    appearances.\
6.  Object representations emerge as explanatory compressions rather
    than perceptual primitives.

------------------------------------------------------------------------

## **Central Claim**

The theory rejects the assumption that intelligence begins with objects.
Intelligence begins with transformations. Objects emerge as inferred
carriers of recurring transformations. Roles emerge as transferable
structures connecting carriers. Concepts emerge from reusable roles. The
path to abstraction therefore proceeds through:

Transformation → Carrier → Role → Concept

rather than through object recognition alone.

# **9. Context Expansion, Graph Context, and Meaning**

A recurring problem in learning systems is the appearance of
contradiction. An agent learns one rule. Later it encounters evidence
that appears to violate that rule. Traditional approaches often respond
by replacing the original rule. This theory proposes a different
mechanism. Contradictions are treated primarily as evidence of missing
context. Learning proceeds through context expansion rather than concept
replacement.

## **Context Expansion Hypothesis**

Suppose the agent learns: Blue causes termination. After many successful
observations, this becomes a useful abstraction. Later the agent
encounters: Blue enables progress. The system now faces an apparent
contradiction. Two responses are possible.

### **Concept Replacement**

Discard the original abstraction and replace it with a new one.

### **Context Expansion**

Search for additional structure that explains both observations.

Examples:

-   location,\
-   state,\
-   inventory,\
-   prior interactions,\
-   surrounding entities,\
-   graph position.

The theory proposes that intelligent systems should prefer context
expansion before concept replacement.

## **Missing Context Resolution**

Contradictions often indicate incomplete understanding rather than
incorrect understanding. The first response to conflict should therefore
be: What contextual variable am I missing?

Examples:

Blue kills when touched directly.

Blue helps after activating a switch.

Blue kills in one region.

Blue helps in another region.

Blue kills during one phase.

Blue helps during another phase.

The original abstraction may remain locally correct. The problem is
incomplete context rather than faulty knowledge.

## **Why Context Matters**

Experience rarely consists of isolated interactions. The significance of
an interaction depends on surrounding structure.

The same action may:

-   help,\
-   harm,\
-   block,\
-   enable,

depending on context. Functional interpretation therefore cannot be
assigned solely from isolated events; it is inferred from relational
evidence.

## **Graph Context Hypothesis**

The theory proposes that context is represented primarily through graph
structure. Interactions are not interpreted independently. They are
interpreted through their position within a network of contingencies.
Functional interpretation is hypothesized to emerge primarily from local
relational neighborhoods.

## **Example: The Key Problem**

Consider a key. A purely appearance-based system may identify a visual
object. The present theory asks a different question: What role does
this structure play inside the interaction graph? The key may
participate in:

Acquire → Unlock → Access → Progress

The functional interpretation of the key is inferred from this
structure. If another carrier occupies a sufficiently similar relational
position, the theory predicts similar role evidence despite perceptual
differences.

## **Relational Functional Identity**

The theory proposes that functional identity is strongly constrained by
relational position in the learned interaction graph. A structure is
characterized by:

-   what precedes it,
-   what follows it,
-   what it enables,
-   what it blocks,
-   how it changes bounded future-option structure.

The empirical claim is not that appearance is irrelevant, but that
relational structure should provide greater predictive value for role
identity and transfer than appearance alone.

## **Context Discovery as a Cognitive Process**

Context discovery becomes a major cognitive function. When
contradictions arise, the system searches for:

-   hidden variables,\
-   latent states,\
-   conditional dependencies,\
-   graph partitions,\
-   missing interactions.

Learning therefore becomes an active search for explanatory structure.

## **Expansion Rather Than Replacement**

The theory predicts that intelligent systems should preserve useful
abstractions whenever possible.

Instead of:

Old Concept → Delete → New Concept

the preferred sequence becomes:

Old Concept → Discover Context → Expand Concept

This preserves accumulated knowledge while increasing explanatory power.

## **Relationship to Memory**

Context expansion naturally fits the memory-centric framework. A
contradiction triggers:

1.  Prediction violation.\
2.  Attention allocation.\
3.  Context search.\
4.  Memory restructuring.\
5.  Graph reorganization.

The result is not merely new information. The result is a more refined
memory structure.

## **Context and Transfer**

Context expansion also supports transfer. Without context: A learned
abstraction may fail in a new environment. With context, The system can
determine:

-   when an abstraction applies,\
-   when it does not apply,\
-   what conditions govern its use.

Transfer therefore depends not only on abstraction formation but also on
context discovery.

## **Context as Progressive Differentiation**

Development can be viewed as repeated refinement of context. Early
abstractions are broad. As experience grows:

-   exceptions appear,\
-   contradictions appear,\
-   hidden variables appear.

The system responds by increasing contextual resolution. The result is
progressively richer understanding. Knowledge becomes more precise
without discarding earlier discoveries.

## **Relationship to Concepts**

Concepts are not fixed definitions. Concepts are evolving structures
embedded within a growing context network. As context expands:

-   concepts become more accurate,\
-   concepts become more transferable,\
-   concepts become more explanatory.

Concept growth therefore occurs through contextualization rather than
repeated replacement.

## **Falsification**

This framework would be weakened if:

-   concept replacement consistently outperformed context expansion,\
-   contradictions rarely corresponded to missing contextual variables,\
-   graph context failed to improve transfer,\
-   local graph structure failed to predict meaning.

## **Central Claim**

The theory proposes that learning is primarily the discovery of missing
context. Contradictions are usually not evidence that an abstraction is
wrong. They are evidence that an abstraction is incomplete. Functional
identity is inferred from graph position together with available
perceptual evidence, and the theory predicts a stronger contribution
from relational structure than appearance alone. Intelligence therefore
progresses through continual expansion of contextual structure,
producing increasingly coherent and transferable explanations of
experience.

# **10. Future-Option Structure and Developmental Reachability**

The previous sections define how interaction memories are formed and
reorganized. This section introduces a graph-topological quantity used
to characterize how an interaction changes the set of futures currently
represented by the learner.

Future-option structure is **not** the source of primary valence and is
not a hidden task reward. It is a learned description of reachable
interaction structure that may acquire instrumental value through
experience.

## **Developmental Principle**

The learner initially possesses no multi-step reachability model. Early
development therefore records only directly observable option evidence
(OE_0), such as:

-   which actions are immediately available,
-   whether interaction continues,
-   which observed actions become unavailable,
-   whether a prior interaction situation can be revisited,
-   whether a previously unseen transition becomes accessible.

Only after contingencies and graph structure exist can the learner
estimate multi-step reachable structure.

## **Bounded Learned Reachability**

Let the currently discovered directed typed graph at developmental time
(t) be:

\[ G_t=(V_t,E_t). \]

For state or graph situation (s), define bounded reachability:

\[ R_t\^k(s) = `\left`{=tex}{ v`\in`{=tex}V_t:
d\_{G_t}(s,v)`\leq`{=tex}k `\right`{=tex}}, \]

where (k\<`\infty`{=tex}) is the learner's current effective planning or
reachability horizon.

For a finite discrete graph, the minimal estimate is:

\[ `\widehat{FO}`{=tex}*t\^k(s) = `\sum`{=tex}*{v`\in`{=tex}V_t}
`\mathbf{1}`{=tex} `\left[ d_{G_t}(s,v)\leq k \right]`{=tex}. \]

For very large or continuous state spaces, raw cardinality can be
replaced by a finite measure (`\mu`{=tex}\_t):

\[ `\widehat{FO}`{=tex}\_{t,`\mu`{=tex}}\^k(s) =
`\mu`{=tex}\_t(R_t\^k(s)). \]

A weighted discrete form is:

\[ `\widehat{FO}`{=tex}*{t,w}\^k(s) =
`\sum`{=tex}*{v`\in`{=tex}R_t\^k(s)} w_t(v), \]

where weights may represent uncertainty, reversibility, novelty,
abstraction level, or empirically learned accessibility. Any weighting
scheme must be declared separately and must not introduce hidden task
reward.

These quantities describe **discovered** reachability, not omniscient
environmental reachability.

## **Future-Option Change**

For interaction (i):

\[ `\Delta`{=tex}`\widehat{FO}`{=tex}\_t\^k(i) =
`\widehat{FO}`{=tex}*t\^k(s*{after}) -
`\widehat{FO}`{=tex}*t\^k(s*{before}). \]

The sign is descriptive:

-   (`\Delta`{=tex}`\widehat{FO}`{=tex}\>0): discovered reachable
    structure expanded,
-   (`\Delta`{=tex}`\widehat{FO}`{=tex}\<0): discovered reachable
    structure contracted,
-   (`\Delta`{=tex}`\widehat{FO}`{=tex}`\approx0`{=tex}): discovered
    reachability was approximately preserved.

The ISF may use the magnitude as a structural-impact signal while
preserving the sign as separate evidence. Expansion is not assigned
intrinsic positive utility, and contraction is not assigned intrinsic
negative utility merely by definition.

## **Primitive Structural Motifs**

Repeated future-option changes support motif formation.

### **Preservation / Reversibility**

Reachability remains approximately unchanged and an empirically
supported return path exists.

### **Expansion**

The transition exposes previously unreachable or previously undiscovered
interaction structure.

### **Contraction / Blocking**

The transition removes or restricts previously reachable interaction
structure.

### **Absorbing or Continuation-Collapse Pattern**

Repeated experience indicates that after a transition the reachable
interaction set contracts toward a stable, low-variation, or
non-continuing region.

This structural motif is independent of primary valence. A terminal
transition may carry positive or negative primary valence even when both
transitions produce similar continuation-collapse structure; learned
memory must preserve these as separate evidence dimensions.

## **Why Future-Option Structure Matters**

The framework hypothesizes that changes in discovered reachability are
useful predictors of:

-   attention allocation,
-   contextual investigation,
-   memory retention,
-   transfer opportunities,
-   later planning structure.

This is a scientific hypothesis, not an axiom that the agent should
maximize option count.

A system may later learn that some expansions are unhelpful, some
contractions are useful, or some smaller reachable sets are more
reliable. Those distinctions arise from subsequent evidence, not from
the definition of (`\widehat{FO}`{=tex}).

## **Trajectory Cost and Efficiency**

Once the learner has observed multiple trajectories with comparable
outcomes, cost becomes meaningful.

Let:

\[ Cost(`\pi`{=tex}) = `\sum`{=tex}\_{i`\in`{=tex}`\pi`{=tex}}c(i), \]

where (c(i)) may include action count, time, uncertainty, loop behavior,
blocked actions, repeated states, or another explicitly declared
resource measure.

For a trajectory:

\[ FOE(`\pi`{=tex}) = `\frac{\Delta \widehat{FO}^{\,k}(\pi)}`{=tex}
{Cost(`\pi`{=tex})}. \]

This ratio is used only after defining an outcome-equivalence relation.
It is not valid to compare a short trajectory and a longer trajectory
merely because one has lower cost if they produce materially different
outcomes.

## **Outcome-Equivalence Abstraction**

Outcome comparability is not only an experimental grouping rule. The
theory predicts that it can itself become a persistent learned
abstraction.

Let (Q_t(s)) denote the learner's current **consequence descriptor** for
a state or trajectory endpoint. Its declared dimensions may include
learned bounded future-option structure, continuation/reversibility
patterns, reachable roles or concepts, predictive consequence profiles,
and contextual conditions.

Pairwise consequence similarity may be proposed using a declared metric
(d_Q), but a threshold relation (d_Q`\le`{=tex}`\epsilon`{=tex}\_Q) is
not assumed to be transitive. Persistent outcome identity is therefore
defined by a learned partition or clustering operator (C_t) over
consequence descriptors:

\[ `\Omega`{=tex}\_t(s)=C_t(Q_t(s)). \]

Two states are outcome-equivalent at developmental time (t) when they
are assigned to the same persistent class:

\[ s_a `\equiv`{=tex}\_{`\Omega`{=tex}\_t} s_b `\iff`{=tex}
C_t(Q_t(s_a))=C_t(Q_t(s_b)). \]

The implementation must declare the class-formation criterion---for
example bounded within-class diameter, predictive interchangeability, or
another preregistered structural test. An **outcome-equivalence
abstraction** becomes persistent only when class support, stability, and
contextual consistency exceed the required thresholds. Classes may later
split or merge as consequence representation improves, while preserving
provenance to prior class identities.

Trajectories are linked to, but are not identical with, their outcome
abstractions:

\[ LeadsTo_t(`\pi`{=tex},`\Omega`{=tex}`\mid`{=tex}`\Gamma`{=tex}). \]

This separation permits multiple strategies to support the same outcome
identity under their learned applicability contexts. If one strategy
fails or becomes unavailable, another strategy with sufficient evidence
for the same outcome class can be selected without redefining the
outcome itself.

Outcome equivalence does not imply preference. A **target-like outcome**
is an outcome abstraction for which the learner later exhibits a stable
learned comparative preference when multiple reachable outcome classes
are available. Conventional goal-like behavior is therefore treated as a
late property of the relation between outcome abstractions, learned
preference, and planning---not as a primitive symbol.

For scientific evaluation, external benchmark labels may be used to test
whether learned outcome classes correspond to externally meaningful
outcome groups, but those labels do not define internal utility.

## **Developmental Sequence**

The proposed sequence is:

\[ Interaction `\rightarrow`{=tex} OE_0 `\rightarrow`{=tex}
Contingencies `\rightarrow`{=tex} G_t `\rightarrow`{=tex}
`\widehat{FO}`{=tex}\^{,k} `\rightarrow`{=tex}
Future`\text{-}`{=tex}Option Motifs `\rightarrow`{=tex} Initial Planning
`\rightarrow`{=tex} Outcome`\text{-}`{=tex}Equivalence Abstraction
`\rightarrow`{=tex} Alternative Strategy Reuse `\rightarrow`{=tex}
Replanning `\rightarrow`{=tex} Efficiency. \]

Thus future-option **observation** begins early, multi-step
future-option **estimation** is learned, initial planning may arise from
that structure, and persistent outcome identity later becomes separable
from the trajectories used to reach it. That separation specifically
enables alternative-strategy replanning.

## **Relation to Transfer**

Two environments may support transfer when structurally corresponding
roles induce similar changes in bounded reachability despite perceptual
differences. This motivates testing whether future-option and relational
similarity predict held-out reuse better than appearance.

## **Falsification**

The future-option component is weakened if, under controlled
comparisons:

-   bounded future-option change does not improve prediction of
    attention or retention,
-   future-option/role similarity does not improve transfer prediction
    beyond appearance,
-   future-option motifs appear only after explicit planning mechanisms
    create them,
-   alternative structural quantities consistently explain the same
    phenomena better.

## **Central Claim**

Future-option structure is a bounded, epistemic graph quantity learned
from interaction. Its scientific role is to characterize how discovered
reachable structure changes and to test whether those changes organize
later attention, abstraction, transfer, and planning. It is not a hidden
reward or survival objective.

# **11. Attention, Prediction Violation, Memory Selection, and Explanatory Promotion**

The previous sections describe the structures that may emerge during
development. This section specifies how limited computational resources
are allocated among competing memories and candidate abstractions.

## **Pre-Predictive Development and Cold Start**

Prediction error cannot organize development before supported
expectations exist. The theory therefore distinguishes two significance
regimes.

Before a contingency satisfies the minimum support required for an
active expectation, interaction significance is pre-predictive:

\[ ISF\_{`\mathrm{cold}`{=tex}}(I_t) =
f(N_t,`\Delta`{=tex}\^{struct}\_t,OSI_t,PVI_t,R_t), \]

where (N_t) denotes novelty, (`\Delta`{=tex}\^{struct}\_t) observable
structural change, (OSI_t) option-structure impact, (PVI_t)
primary-valence impact, and (R_t) recurrence potential.

After a contingency satisfies the expectation-admission criterion,
empirical prediction error becomes available:

\[ ISF\_{`\mathrm{predictive}`{=tex}}(I_t) =
f(PVI_t,OSI_t,PE_t,LV_t,TP_t,EP_t). \]

Thus:

``` text
pre-predictive exploration
        ↓
observable recurrence
        ↓
supported contingency
        ↓
active expectation
        ↓
prediction violation becomes informative
```

Novelty is not a permanent optimization objective. Its role is to
prevent developmental inertia before predictive structure exists.

A required cold-start evaluation measures time or interactions required
to form the first stable predictive contingency as observation
dimensionality, action breadth, stochasticity, and recurrence frequency
are varied.

## **Prediction Violation**

A prediction violation occurs when a supported expectation differs from
the observed transformation:

\[ PE_t = d\_{`\Delta`{=tex}} ( `\Delta`{=tex}\_t,
`\widehat{\Delta}`{=tex}\_t ) \> 0. \]

Prediction violation is evidence that the current conditional structure
is incomplete, obsolete, or insufficiently contextualized.

It does not by itself imply that the event is important.

## **Prediction Error Is Not Significance**

A large prediction error may arise from noise or an idiosyncratic event.
Conversely, an interaction with modest prediction error may have high
explanatory or transfer value.

The framework therefore separates:

-   **prediction error** --- mismatch with an established expectation,
-   **interaction significance** --- priority for additional
    developmental processing,
-   **memory fitness** --- later retention/replay value of a stored
    structure,
-   **scientific validation** --- evidence that a higher-level claim
    actually holds.

These quantities may correlate but are not interchangeable.

## **Attention as Resource Allocation**

Attention denotes additional compute, memory, or testing resources
assigned to candidate evidence.

Higher-priority interactions may receive:

-   deeper transformation analysis,
-   additional replay,
-   contextual partition search,
-   abstraction candidate generation,
-   held-out transfer testing.

Attention is therefore operational resource allocation rather than a
metaphorical mental state.

## **Primary Attention Signals**

Three broad evidence classes are expected to attract additional
resources:

### **Prediction Violation**

An established local expectation failed.

### **Option-Structure Impact**

Observed or learned reachable interaction structure changed
substantially.

### **Explanatory Opportunity**

The interaction may support compression, contextual resolution, or a
higher-level reusable representation.

The paper predicts that combinations of these signals outperform novelty
or frequency alone.

## **Memory Selection and Retention Dynamics**

Stored structures compete under finite cognitive budgets. The
operational mechanisms may include:

-   fitness-weighted retention,
-   bounded replay allocation,
-   graph pruning,
-   compression and replacement,
-   quarantine and reactivation,
-   provenance-preserving retirement.

A memory remains cognitively resident when current evidence indicates
continuing utility. Scientific provenance may persist after cognitive
retirement.

## **Selective Forgetting**

Forgetting is a representation-management mechanism. A lower-level
structure may be removed from active cognition when a higher-level
representation preserves its unique predictive, explanatory, contextual,
or transfer contribution.

This yields a stronger criterion than low frequency:

> A memory is safely compressible only when its removal does not
> eliminate unique supported structure required by active descendants or
> current evidence contracts.

## **Explanatory Reach**

Let (Supports(X)) denote lower-level structures materially explained by
(X). Then:

\[ ER(X)=\|Supports(X)\| \]

or a weighted equivalent.

The paper predicts that explanatory reach should predict candidate
promotion better than raw recurrence count.

## **Promotion Versus Validation**

Promotion has two distinct meanings that must not be conflated.

### **Candidate Promotion**

A structure receives a higher-level representation or additional testing
budget because its prospective evidence is strong.

### **Scientific Validation**

A structure satisfies the empirical evidence contract associated with
its abstraction level.

For concept-level transfer, candidate promotion can use (TP\_{prior});
validation requires held-out (TP\_{emp}) or an explicit
intervention-based transfer effect.

This distinction prevents high-frequency or highly explanatory
structures from being labeled transferable before transfer has actually
been tested.

## **Context Refinement Before Replacement**

When a supported abstraction accumulates contradictory evidence, the
first test is whether an evidence-supported contextual partition
explains the apparent conflict.

The sequence is:

\[ Prediction Violation `\rightarrow`{=tex} Context Search
`\rightarrow`{=tex} Candidate Refinement `\rightarrow`{=tex} Retest
`\rightarrow`{=tex} Retain, Split, or Demote. \]

Replacement becomes appropriate only if contextual refinement fails or
if the broader abstraction loses empirical support.

## **Developmental Changes in Resource Allocation**

Early stages allocate more resources to recurrence, local structural
impact, and emerging prediction error.

Intermediate stages increasingly allocate resources to contextual
refinement, compression, carrier persistence, and role formation.

Later stages allocate more resources to transfer testing, explanatory
reach, world-model organization, planning, and outcome-conditioned
efficiency.

The scientific criteria defining valid evidence do not change with
stage; only resource allocation does.

## **Unified Developmental Cycle**

A compact cycle is:

\[ Experience `\rightarrow`{=tex} Structural Evidence
`\rightarrow`{=tex} Significance `\rightarrow`{=tex} Attention
`\rightarrow`{=tex} Memory `\rightarrow`{=tex} Prediction Violation
`\rightarrow`{=tex} Context/Compression `\rightarrow`{=tex}
Candidate Promotion `\rightarrow`{=tex} Held`\text{-}`{=tex}Out Testing
`\rightarrow`{=tex} Retention/Pruning. \]

The cycle repeats as new evidence enters the system.

## **Central Claim**

The framework predicts that abstraction depends on selective resource
allocation rather than uniform retention. Prediction violations identify
local inadequacy, option-structure impact identifies consequential
structural change, explanatory reach identifies compression opportunity,
and empirical transfer determines whether higher-level relational
structure generalizes beyond its formation scope.

# **12. Concept Formation, Transfer, and World-Model Emergence**

The previous sections describe how interactions become memories, how
memories compete, and how abstractions are promoted.

This section addresses three questions:

1.  What is a concept?\
2.  How does transfer occur?\
3.  How do world models emerge?

The theory proposes that all three arise from the same developmental
process.

## **What Is a Concept?**

Many definitions of concepts focus on categories, labels, or object
classes. This theory proposes a different definition.

### **Definition**

The theory distinguishes a **concept candidate** from a **validated
transferable concept**. A concept candidate is an internally supported
abstraction that compresses interaction history and has sufficient
explanatory reach to warrant transfer testing. A validated transferable
concept is a concept candidate that has subsequently demonstrated
admissible structural correspondence and positive empirical reuse in
previously unseen environments under the declared transfer evidence
contract.

Several implications follow immediately. A validated concept is not:

-   a label,\
-   a cluster,\
-   a recurring pattern,\
-   a frequently observed structure.

A structure may therefore become a concept candidate before external
transfer evidence exists, but it becomes a **validated transferable
concept** only when it supports empirical transfer. Without transfer
there may still be compression, abstraction, pattern recognition, and
categorization; what is absent is evidence that the abstraction is
transferable.

## **Why Transfer Is Part of the Definition**

Consider two structures. The first compresses experience extremely well
but works only in one environment. The second compresses slightly less
effectively but improves performance across many environments. The
theory argues that the second structure is more concept-like. Concepts
are valuable because they generalize. Transfer is therefore not a
secondary property. Transfer is a defining property.

## **From Roles to Concepts**

Functional roles provide the bridge between interaction history and
conceptual abstraction. The developmental sequence becomes:

Episodes → Contingencies → Interaction Families → Roles → Concepts

A role becomes a concept candidate when compression and explanatory
evidence justify higher-level abstraction. It becomes a validated
transferable concept when held-out reuse succeeds across environments.

For example: Enabler may appear as:

-   key,\
-   switch,\
-   password,\
-   button,\
-   lever,\
-   access card.

The shared abstraction is not any individual carrier. The shared
abstraction is the transferable role. Repeated successful transfer
promotes the role toward conceptual status.

## **Transfer as Structural Reuse**

The theory rejects the idea that transfer is primarily driven by
appearance. Instead, transfer emerges through reuse of interaction
structure. A system transfers successfully when it recognizes:

-   similar contingencies,\
-   similar motifs,\
-   similar graph neighborhoods,\
-   similar functional roles,\
-   similar future-option structures.

Transfer is therefore fundamentally relational.

## **Levels of Transfer**

Transfer itself develops progressively.

### **Level 0: Episodic Transfer**

Reuse of nearly identical experiences. Highly specific. Minimal
generalization.

### **Level 1: Contingency Transfer**

Reuse of recurring action-consequence relationships. Generalization
remains local.

### **Level 2: Interaction-Family Transfer**

Reuse of common transformation patterns. Broader applicability.

### **Level 3: Role Transfer**

Reuse of functional roles across environments. Strong generalization.
Appearance becomes increasingly irrelevant.

### **Level 4: Concept Transfer**

Reuse of highly compressed abstractions. Transfer extends across
substantially different environments. The theory predicts that
higher-level transfer emerges before robust object-category transfer.

## **Why Functional Similarity Dominates Appearance**

Two structures may look identical yet perform completely different
functions. Conversely, two structures may appear unrelated while
performing the same role. The theory predicts: Functional similarity is
a stronger predictor of transfer than perceptual similarity. This
prediction directly challenges appearance-centric approaches to concept
formation.

## **Graph-Based Transfer**

Transfer is hypothesized to depend heavily on graph structure. Two
environments may support transfer when they contain:

-   similar dependency structures,
-   similar bottlenecks,
-   similar enabling or blocking relationships,
-   similar future-option patterns.

Let candidate (X) have typed neighborhood (N_A(X)) in formation
environment (A). Structural reuse in unseen environment (B) is first
treated as a candidate mapping:

\[ `\phi`{=tex}:N_A(X)`\rightarrow`{=tex}G_B \]

with bounded typed-edge preservation error. This approximate graph
correspondence determines whether a transfer test is structurally
admissible.

The mapping does not itself validate the concept. Validation
additionally requires positive held-out empirical effect when the
abstraction is enabled relative to an appropriate ablated or matched
baseline. Thus graph correspondence supplies structural alignment;
intervention supplies transfer evidence.

## **Concept Stability**

Concepts are not permanent. Concepts compete just as memories compete. A
concept survives when it continues to:

-   explain experience,\
-   support transfer,\
-   compress memory,\
-   predict useful structure.

Concepts may be refined, merged, divided, or discarded as development
continues. Concept formation remains an ongoing process.

## **Emergence of World Models**

The theory rejects world-model-first learning. Instead, world models
emerge gradually from accumulated abstractions. The sequence becomes:

Interaction → Memory → Contingency → Role → Concept → World Model

A world model is therefore not a primitive structure. It is a
large-scale compression of previously discovered concepts and
relationships.

## **What Is a World Model?**

A world model is an explanatory network linking:

-   concepts,\
-   roles,\
-   carriers,\
-   contingencies,\
-   future-option structures.

Its purpose is not merely prediction. Its purpose is explanation. A
world model organizes large regions of experience into a coherent
structure.

## **Prediction as an Emergent Capability**

Prediction is often treated as a prerequisite for intelligence. This
theory reverses that relationship. The agent first experiences. Then
remembers. Then compresses. Then abstracts. Only later does large-scale
prediction become possible. Prediction emerges from accumulated
structure. It is not the starting point.

## **Planning as a Consequence**

The same argument applies to planning. Planning depends on:

-   memory,\
-   abstractions,\
-   roles,\
-   concepts,\
-   future-option understanding.

The agent first discovers structure. Only later can it plan effectively
within that structure. Planning therefore emerges from development
rather than driving development.

## **Outcome Identity, Alternative Strategies, and Replanning**

Planning requires an additional distinction that is easy to hide inside
conventional goal language: the representation of an outcome must be
separable from a remembered procedure for reaching it.

The framework therefore predicts the emergence of persistent
outcome-equivalence abstractions. Different trajectories may acquire
evidence-supported links to the same outcome class. Once this separation
exists, the system can preserve an outcome representation while
replacing, suppressing, or bypassing a failed strategy.

Replanning is then defined as selecting a different strategy associated
with the same currently selected outcome abstraction, or with another
outcome abstraction judged equivalent under the learner's current
consequence representation.

A stable preference for a particular outcome class is a later learned
phenomenon. If such a preference persists across contexts and becomes a
planning reference, the resulting structure is **goal-like**, but the
goal is an emergent relation over learned outcome classes rather than a
primitive input.

## **Concepts, World Models, and Continuous Intelligence**

Even world models are not final products. As new experiences arrive:

-   concepts evolve,\
-   roles evolve,\
-   contexts expand,\
-   world models reorganize.

The theory therefore rejects convergence as the endpoint of
intelligence. World models themselves remain subject to continuous
restructuring.

## **Falsification**

The theory would be weakened if:

-   object categories consistently emerge before transferable roles,\
-   appearance predicts transfer better than functional structure,\
-   graph organization fails to predict transfer,\
-   world models must exist before abstraction emerges,\
-   transfer fails to improve with increasing abstraction level.

## **Central Claim**

Concepts are transferable compressions of interaction history. Transfer
emerges through reuse of roles, motifs, graph structure, and
future-option patterns rather than through appearance alone. World
models are not starting assumptions. They are late-stage explanatory
structures emerging from memory, abstraction, and transfer. Intelligence
therefore progresses from remembered interactions toward increasingly
powerful explanatory organizations of experience, culminating in
continuously evolving world models.

# **13. Formal Foundations**

This section states the minimum mathematical commitments of the theory.
The aim is to make the framework implementable and falsifiable without
implying stronger mathematical results than have been established.

## **13.0 Minimal Assumptions**

### **A1 --- Two-Scale Temporally Ordered Interaction**

The learner receives a macro-time sequence of externally selected
interventions:

\[ (O_t,`\mathcal{A}`{=tex}*t,A_t,T_t,O*{t+1}), \]

where (O_t) is the current observation, (`\mathcal{A}`{=tex}*t) is the
currently available action set, (A_t`\in`{=tex}`\mathcal{A}`{=tex}*t),
and (T_t=(O*{t,1},...,O*{t,m_t})) is an optional ordered micro-time
trace emitted by the environment while the selected action unfolds. The
settled observation (O\_{t+1}) closes the macro interaction.

Micro-time observations do not create additional agent interventions.
They may be used to infer within-action causal order, collision
sequences, propagation, activation, or other structural dynamics only
through relations admitted by the observation contract. If no
intermediate observations are exposed, (T_t) is empty and the model
reduces to the ordinary action/next-observation sequence.

### **A2 --- Declared Observation Structure**

Observations belong to a domain:

\[ `\mathcal{S}`{=tex}\_O=(`\mathcal{O}`{=tex},`\mathcal{R}`{=tex}\_O),
\]

where (`\mathcal{R}`{=tex}\_O) contains explicitly declared primitive
structural relations available to the learner.

Examples may include:

-   equality,
-   temporal order,
-   coordinate identity,
-   adjacency,
-   neighborhood,
-   sensor-channel topology.

The theory does not require
(`\mathcal{R}`{=tex}\_O=`\varnothing`{=tex}). It requires that
(`\mathcal{R}`{=tex}\_O) contain no task-semantic categories such as
object, key, door, agent, tool, goal, enemy, or reward meaning.

### **A3 --- Finite Computational Resources**

At each developmental interval, storage, replay, candidate generation,
structural matching, and transfer testing operate under finite budgets.

### **A4 --- Causal Availability**

A decision made at time (t) may use only evidence available no later
than (t).

### **A5 --- Minimal Signed Primary Valence**

A minimal signed primary-valence signal is a permitted primitive
motivational channel. Environment-specific terminal names are converted
at the boundary to (V_t `\in`{=tex} {-1,0,+1}) and are not used as
task-semantic memory identities. Primary valence is distinct from
significance, structural consequence, future-option change, outcome
identity, and strategy identity.

These assumptions define the intended ontology more precisely: the
learner is **semantically uncommitted**, not **representation-free**.

# **13.1 Observation and Transformation**

A macro transformation is:

\[ `\Delta`{=tex}*t=D_O(O_t,O*{t+1};`\mathcal{R}`{=tex}\_O). \]

When a within-action trace (T_t) is available, its ordered
micro-transformations are:

\[ `\delta`{=tex}*{t,j}=D_O(O*{t,j-1},O\_{t,j};`\mathcal{R}`{=tex}\_O),
\]

with the final micro-transition connected to the settled (O\_{t+1}). The
ordered sequence may be compressed into a temporal trace signature or
family only if the compression preserves the evidence required by the
relevant scientific test.

(D_O) may use only relations admitted by the declared
observation-structure contract.

For an ARC-style grid, one admissible reference representation is:

\[ D_O\^{ARC}(O_t,O\_{t+1}) = { (r,c,O_t\[r,c\],O\_{t+1}\[r,c\]) :
O_t\[r,c\]`\neq`{=tex}O\_{t+1}\[r,c\] }. \]

Derived statistics may include change count, coordinate extent, equality
patterns, connectedness under declared grid adjacency, and
displacement-like relations.

This operator **does encode a spatial prior**: row/column identity and
adjacency are supplied by the observation interface. The
interaction-first claim is therefore not that spatial structure is
learned from nothing. The claim is that persistent carrier identity,
functional role, concept status, and task meaning are not supplied by
(D_O).

# **13.2 Interaction**

The primitive macro-memory record is:

\[ I_t=(O_t,A_t,T_t,`\Delta`{=tex}\_t,V_t), `\qquad`{=tex} V_t
`\in`{=tex} {-1,0,+1}. \]

(T_t) is optional but ordered when present. It belongs to the evidence
generated by action (A_t); it is not expanded into fictitious additional
actions.

Primary valence is stored as evidence attached to the experienced
interaction. Its environment-specific source label is not part of the
canonical identity of the interaction-derived abstraction.

An implementation may store a compressed observation signature rather
than the full observation, provided the scientific evidence ledger can
reconstruct the information required by the relevant tests.

# **13.2.1 Macro-Time and Micro-Time Causal Traces**

Macro-time indexes agent interventions. Micro-time indexes passive
environment evolution observed between one action and the settled next
macro observation. For macro step (t), define:

\[ T_t=(O\_{t,1},...,O\_{t,m_t}), `\qquad`{=tex} m_t`\ge`{=tex}0. \]

A temporal trace descriptor may summarize the ordered sequence using
bounded structural statistics, transition-family signatures, or
carrier-lineage signatures. Any such descriptor must preserve ordering
information required by the claim being tested and must remain linked to
the originating macro action.

The theory therefore permits learning causal mechanics that are
invisible in the settled frame alone while preserving the intervention
semantics of the original interaction-first ontology.

# **13.3 Episode**

An episode is an ordered finite sequence:

\[ E=(I_1,I_2,`\ldots`{=tex},I_n). \]

Episodes correspond to detailed low-level evidence. Episode ordering is
defined over macro interactions; each macro interaction may contain its
own ordered micro-time trace. Episodes are not required to remain
permanently active once higher-level structures preserve their unique
explanatory contribution.

# **13.4 Contingency**

A contingency is an empirically supported conditional transition
regularity:

\[ C: P(`\Delta`{=tex}`\mid`{=tex}A,`\Gamma`{=tex}), \]

where (`\Gamma`{=tex}) is the currently represented context.

Let (N(C)) be support and (Stability(C)) an empirical concentration or
predictive-stability measure. An expectation becomes active only if:

\[ N(C)`\ge`{=tex}n\_{min} `\quad`{=tex}`\land`{=tex}`\quad`{=tex}
Stability(C)`\ge`{=tex}`\theta`{=tex}\_{stable}. \]

Before this condition is met, recurrence may seed memory but cannot
generate prediction-error evidence.

# **13.5 Empirical Prediction Error**

Once a supported local expectation exists:

\[ PE_t = d\_{`\Delta`{=tex}} `\left`{=tex}( `\Delta`{=tex}\_t,
`\widehat{\Delta}`{=tex}\_t `\right`{=tex}). \]

The distance (d\_{`\Delta`{=tex}}) must be declared for the
transformation representation. Prediction error is inactive when no
admissible expectation existed before the observation.

# **13.6 Typed Interaction Graph**

At developmental time (t), let:

\[ G_t=(V_t,E_t,`\tau`{=tex}\_V,`\tau`{=tex}\_E) \]

be a directed typed multi-scale graph, where:

-   (V_t) contains remembered structures,
-   (E_t) contains evidence-supported relations,
-   (`\tau`{=tex}\_V) maps nodes to memory types or levels,
-   (`\tau`{=tex}\_E) maps edges to relation types.

Possible edge types include temporal succession, dependency, enablement,
blocking, structural similarity, provenance, and explanation.

The graph is sparse by construction; absence of an edge is not
equivalent to evidence of no relation.

# **13.7 Primitive Observed Option Evidence**

Before a learned graph exists, define model-free option evidence (OE_0)
from quantities directly observable at the interaction interface.

One representation is:

\[ OE_0(t) = `\left`{=tex}( \|`\mathcal{A}`{=tex}\_t\|, Continue_t,
NovelTransition_t, ReversibleObserved_t `\right`{=tex}). \]

`Continue` denotes whether another interaction is observed to be
available after the transition. It is not a reward.

Stage-0 option-structure impact may be:

\[ OSI_0(i) = d\_{OE} `\left`{=tex}( OE_0\^{before}, OE_0\^{after}
`\right`{=tex}). \]

# **13.8 Bounded Future-Option Reachability**

After (G_t) exists, define the bounded reachable set:

\[ R_t\^k(s) = { v`\in`{=tex}V_t: d\_{G_t}(s,v)`\le`{=tex}k },
`\qquad`{=tex} k\<`\infty`{=tex}. \]

For finite graphs:

\[ `\widehat{FO}`{=tex}*t\^k(s) = `\sum`{=tex}*{v`\in`{=tex}V_t}
`\mathbf{1}`{=tex} \[ d\_{G_t}(s,v)`\le`{=tex}k\]. \]

For large or continuous representations:

\[ `\widehat{FO}`{=tex}\_{t,`\mu`{=tex}}\^k(s) =
`\mu`{=tex}\_t(R_t\^k(s)), \]

where (`\mu`{=tex}\_t) is an explicitly declared finite measure.

For weighted graphs:

\[ `\widehat{FO}`{=tex}*{t,w}\^k(s) =
`\sum`{=tex}*{v`\in`{=tex}R_t\^k(s)} w_t(v). \]

The online learner uses only the graph discovered by time (t). An oracle
reachability quantity may be computed offline for evaluation but may not
enter online learning.

# **13.9 Future-Option Change and Structural Impact**

For interaction (i):

\[ `\Delta`{=tex}`\widehat{FO}`{=tex}\_t\^k(i) =
`\widehat{FO}`{=tex}*t\^k(s*{after}) -
`\widehat{FO}`{=tex}*t\^k(s*{before}). \]

The sign is retained as structural evidence. The magnitude may
contribute to significance:

\[ OSI_t(i) = g( \|`\Delta`{=tex}`\widehat{FO}`{=tex}\_t\^k(i)\| ), \]

for a bounded (g).

This separates **importance of structural change** from **behavioral
preference for a direction of change**.

# **13.10 Structural Motifs**

A motif is a recurrent typed local graph pattern.

Examples include:

### **Expansion**

\[ `\Delta`{=tex}`\widehat{FO}`{=tex}\>0. \]

### **Contraction**

\[ `\Delta`{=tex}`\widehat{FO}`{=tex}\<0. \]

### **Reachability Preservation**

\[ \|`\Delta`{=tex}`\widehat{FO}`{=tex}\|\<`\epsilon`{=tex}. \]

### **Reversibility**

Reachability is approximately preserved and an empirically supported
return path exists.

### **Absorbing/Continuation-Collapse Pattern**

Repeated evidence indicates transition into a low-variation or
non-continuing reachable region.

Motif labels summarize learned structural regularities; they do not
carry intrinsic utility.

# **13.11 Interaction Cost and Realized Trajectory Length**

For trajectory (`\pi`{=tex}=(i_1,`\ldots`{=tex},i_n)):

\[ Cost(`\pi`{=tex}) = `\sum`{=tex}\_{j=1}\^{n} c(i_j). \]

The minimal implementation uses (c(i)=1), making cost equal to realized
action/interaction count. Richer declared cost models may include time,
repeated states, loops, blocked actions, uncertainty, risk, or compute.

For a strategy (`\pi`{=tex}) initiated under context (`\Gamma`{=tex})
and a represented outcome (`\Omega`{=tex}), define the realized
outcome-conditioned cost:

\[
C_t(`\pi`{=tex}`\rightarrow`{=tex}`\Omega`{=tex}`\mid`{=tex}`\Gamma`{=tex})
=
E_t\[Cost(`\pi`{=tex});\|;Outcome_t(`\pi`{=tex})=`\Omega`{=tex},`\Gamma`{=tex}\].
\]

The measured interval begins when the strategy is taken and ends when
its represented outcome is first reached. Unsuccessful runs may be
retained separately as censored or boundary-terminated trials rather
than being silently interpreted as successful low-cost runs.

When an episode reaches non-zero primary valence
(`V`{=tex}`\in`{=tex}{-1,+1}), the learner may additionally retain:

\[ C\^V_t(`\pi`{=tex}`\mid`{=tex}`\Gamma`{=tex}) =
E_t\[`\text{interactions from strategy initiation to the observed primary-valence boundary}`{=tex}\].
\]

This second statistic does not redefine outcome identity. It records how
quickly a realized strategy trajectory reaches motivationally
significant episode termination.

# **13.12 Consequence Descriptors and Outcome-Equivalence Memory**

Let (Q_t(s)) be the learner's causally available consequence descriptor
for a state or trajectory endpoint. Its dimensions must be declared by
the implementation and may include:

-   bounded learned future-option structure,
-   continuation and reversibility evidence,
-   reachable roles or concepts,
-   predictive consequence profiles,
-   contextual conditions,
-   other non-semantic learned structural quantities.

External benchmark outcome labels may appear only in the scientific
evaluation descriptor unless the experiment explicitly studies their
learned internal reconstruction.

A pairwise threshold
(d_Q(Q_t(s_a),Q_t(s_b))`\le`{=tex}`\epsilon`{=tex}\_Q) may generate
candidate similarity, but need not be transitive. Persistent outcome
identity is therefore defined through an explicitly declared partition
or clustering operator:

\[ C_t:`\mathcal{Q}`{=tex}\_t`\rightarrow`{=tex}`\mathcal{O}`{=tex}\_t,
\]

where (`\mathcal{Q}`{=tex}\_t) is the current consequence-descriptor
space and (`\mathcal{O}`{=tex}\_t) is the set of learned outcome
classes.

Define learned outcome equivalence by:

\[ s_a`\equiv`{=tex}\_{`\Omega`{=tex}\_t}s_b `\iff`{=tex}
C_t(Q_t(s_a))=C_t(Q_t(s_b)). \]

For a trajectory (`\pi`{=tex}) ending in state (s\_{end}(`\pi`{=tex})):

\[ Outcome_t(`\pi`{=tex})=C_t(Q_t(s\_{end}(`\pi`{=tex}))). \]

The implementation must declare its class-formation criterion, such as
bounded within-class diameter, predictive interchangeability, or another
preregistered structural condition. A class becomes a persistent
abstraction only after support, stability, and contextual consistency
exceed preregistered thresholds. Outcome classes are versioned: later
evidence may split or merge them, but provenance to prior identities
must be retained. This converts outcome comparability from an evaluator
convenience into a learned memory structure.

## **13.12.1 Strategy-to-Outcome Relations**

A strategy (`\pi`{=tex}) and an outcome abstraction (`\Omega`{=tex}) are
connected by an empirically supported, context-conditioned relation:

\[
LeadsTo_t(`\pi`{=tex},`\Omega`{=tex}`\mid`{=tex}`\Gamma`{=tex})`\in[0,1]`{=tex},
\]

representing the observed reliability with which the strategy reaches
that outcome class under learned applicability context (`\Gamma`{=tex}).

Multiple strategies may therefore support the same outcome identity
under different or overlapping contexts:

\[
{`\pi`{=tex}\_1,`\ldots`{=tex},`\pi`{=tex}\_m}`\rightarrow`{=tex}`\Omega`{=tex}.
\]

Strategy identity and outcome identity are distinct memory structures.
For each supported relation the learner therefore keeps
outcome-achievement reliability, signed primary-valence evidence, and
realized trajectory cost as separate sufficient statistics. A strategy
is not considered efficient merely because it is positively valenced,
and a low-cost strategy is not considered successful unless the
represented outcome was actually reached.

## **13.12.2 Learned Outcome Preference and Target-Like Outcomes**

Outcome equivalence does not imply desirability. Let the learner's
revealed comparative preference under a context (`\Gamma`{=tex}) be
estimated from repeated choices where both outcome classes are
represented as reachable:

# \[ Pref_t(`\Omega`{=tex}\_a,`\Omega`{=tex}\_b`\mid`{=tex}`\Gamma`{=tex})

P_t(choose path to `\Omega`{=tex}\_a`\mid`{=tex}`\Omega`{=tex}\_a,`\Omega`{=tex}\_b reachable,`\Gamma`{=tex}) -
P_t(choose path to `\Omega`{=tex}\_b`\mid`{=tex}`\Omega`{=tex}\_a,`\Omega`{=tex}\_b reachable,`\Gamma`{=tex}).
\]

A **target-like outcome** is an outcome-equivalence abstraction whose
learned preference is sufficiently stable and general across an
explicitly declared set of contexts. This is a descriptive property of
learned behavior, not an intrinsic utility assigned to the outcome.

A conventional goal may be operationally identified only when such a
target-like outcome is persistently used as a planning reference.

## **13.12.3 Replanning**

Let (`\Omega`{=tex}\^\*) be the currently selected outcome abstraction
and let (`\Pi`{=tex}\_t(`\Omega`{=tex}\^*,`\Gamma`{=tex})) be the set of
admissible strategies with supported
(LeadsTo_t(`\pi`{=tex},`\Omega`{=tex}\^*`\mid`{=tex}`\Gamma`{=tex}))
relations. Replanning occurs when the currently preferred strategy
becomes unavailable, contradicted, or dominated and the system selects:

\[
`\pi`{=tex}'`\in`{=tex}`\Pi`{=tex}\_t(`\Omega`{=tex}\^\*,`\Gamma`{=tex}),`\qquad`{=tex}`\pi`{=tex}'`\neq`{=tex}`\pi`{=tex}\_{current},
\]

without requiring (`\Omega`{=tex}\^\*) itself to be replaced.

This is the formal consequence of separating **what is to be reached**
from **how it is reached**.

## **13.12.4 Outcome Comparability and Relative Strategy Efficiency**

Two trajectories are admissible for direct efficiency comparison when
their learned outcome abstractions are the same or fall within an
explicitly declared no-worse relation:

\[ Outcome_t(`\pi`{=tex}\_1)=Outcome_t(`\pi`{=tex}\_2) \]

or an experimentally specified dominance criterion holds over the
relevant outcome dimensions.

For a comparison class (`\Pi`{=tex}\_t(`\Omega`{=tex},`\Gamma`{=tex}))
containing at least two empirically supported strategies, define:

\[ C\^\**t(`\Omega`{=tex},`\Gamma`{=tex}) =
`\min`{=tex}*{`\pi`{=tex}`\in`{=tex}`\Pi`{=tex}\_t}
C_t(`\pi`{=tex}`\rightarrow`{=tex}`\Omega`{=tex}`\mid`{=tex}`\Gamma`{=tex}),
\]

and normalized relative efficiency:

\[ Eff_t(`\pi`{=tex},`\Omega`{=tex}`\mid`{=tex}`\Gamma`{=tex}) =
``\frac{C^*_t(`\Omega`{=tex},`\Gamma`{=tex})}``{=tex}
{C_t(`\pi`{=tex}`\rightarrow`{=tex}`\Omega`{=tex}`\mid`{=tex}`\Gamma`{=tex})}.
\]

Thus the best known supported strategy in the comparison class has
(Eff=1), while less efficient strategies have values in ((0,1\]). No
efficiency bonus is defined from this comparison when only one strategy
is represented, because no empirical alternative exists.

If multiple strategies terminate in the same signed primary-valence
boundary, an analogous actions-to-valence comparison may be reported,
but it must not override differences in represented outcome identity or
outcome quality. Reliability, valence, and efficiency remain distinct
selection dimensions.

For scientific benchmarking, an external evaluator may additionally
define an oracle equivalence relation (`\sim`{=tex}*{Q}). The online
learner must not use (Q\_*) unless the experiment explicitly exposes
that information.

# **13.13 Outcome-Conditioned Efficiency and Future-Option Efficiency**

Outcome-conditioned relative efficiency (Eff) is the primary late-stage
strategy-efficiency statistic because it directly compares realized cost
for reaching the same represented M6 outcome.

Future-option efficiency remains an auxiliary structural statistic. For
an admissible trajectory:

\[ FOE(`\pi`{=tex}) =
`\frac{ \Delta\widehat{FO}^{\,k}(\pi) }{ Cost(\pi) }`{=tex}. \]

FOE must not replace primary valence or outcome-conditioned efficiency.
A positive-valence terminal trajectory may legitimately have zero
remaining future options, while still being both motivationally positive
and highly efficient if it reaches its outcome with low realized cost.

Neither Eff nor FOE is an initial learning objective. They become
late-stage comparison statistics after learned outcome-equivalence
structure or another explicitly admissible outcome-comparison relation
has been established.

# **13.14 Explanatory Reach and Compression**

Let (Supports(X)) be the set of lower-level structures whose prediction,
reconstruction, or relational organization is materially improved by
representation (X).

Raw explanatory reach is:

\[ ER(X)=\|Supports(X)\|. \]

A weighted form is:

\[ ER_w(X) = `\sum`{=tex}\_{Y`\in`{=tex}Supports(X)} `\omega`{=tex}(Y).
\]

A compression measure may be based on description length:

\[ CB(X) = L(Supports(X)) - L(X,Residuals`\mid`{=tex}X). \]

MDL is one admissible formalization of compression. It is not assumed to
be the global objective of all developmental processes.

# **13.15 Transfer Prior**

Before held-out testing:

\[ TP\_{prior}(X) = `\widehat{P}`{=tex} (
`\text{reuse outside formation scope}`{=tex} `\mid`{=tex}
`\mathcal{E}`{=tex}\_{`\le`{=tex}t} ). \]

The estimate may use structural recurrence, invariance, graph
correspondence, or compression evidence available at formation time.

(TP\_{prior}) allocates testing resources. It cannot establish transfer
validity.

# **13.16 Structural Correspondence Across Environments**

Let candidate (X) have a typed neighborhood (N_A(X)) in formation graph
(G_A). In an unseen environment graph (G_B), a candidate correspondence
is a mapping:

\[ `\phi`{=tex}: V(N_A(X)) `\rightarrow`{=tex} V(G_B). \]

For typed directed edges, define structural error:

\[ `\epsilon`{=tex}\_{struct}(`\phi`{=tex}) =
`\frac{ \sum_{(u,r,v)\in E(N_A(X))} w_r \mathbf{1} [ (\phi(u),r,\phi(v)) \notin E(G_B) ] }{ \sum_{(u,r,v)\in E(N_A(X))}w_r }`{=tex}.
\]

A structural transfer match exists when:

\[ `\min`{=tex}*{`\phi`{=tex}`\in`{=tex}`\Phi`{=tex}*B(X)}
`\epsilon`{=tex}*{struct}(`\phi`{=tex}) \< `\theta`{=tex}*{struct}. \]

This is an **approximate typed graph-homomorphism criterion**. It
provides structural evidence that an abstraction has a counterpart in
the unseen environment.

Structural correspondence alone does not validate a concept.

# **13.17 Empirical Transfer Evidence**

Let (X) be enabled in one intervention condition and ablated in a
matched comparison condition. Let (Q_B) denote a declared behavioral,
predictive, planning, or future-option evaluation metric in unseen
environment (B).

Define:

\[ TE(X,B) = Q_B\^{on} - Q_B\^{off}. \]

For metrics where lower is better, the sign is reversed by definition so
that positive (TE) always denotes improvement.

The strongest transfer evidence uses matched initial conditions or
otherwise controlled paired interventions. When exact paired rollouts
are impossible, the experimental report must identify the weaker
attribution design explicitly.

Empirical transfer is summarized only over targets outside the frozen
formation scope:

\[ TP\_{emp}(X) =
`\frac{ \#\{B:TE(X,B)>\theta_{effect}\} }{ \#\{B:\text{admissible held-out test}\} }`{=tex}.
\]

External benchmark outcome metrics may be used by the scientist for
evaluation without becoming internal reward primitives.

# **13.18 Concept Candidate and Validated Transferable Concept**

A **concept candidate** satisfies:

\[ CandidateConcept(X) `\iff`{=tex} CB(X)\>`\theta`{=tex}\_C
`\land`{=tex} ER(X)\>`\theta`{=tex}*E `\land`{=tex}
TP*{prior}(X)\>`\theta`{=tex}\_P. \]

A **validated transferable concept** additionally requires structural
correspondence and held-out empirical effect:

\[ ValidatedConcept(X) `\iff`{=tex} CandidateConcept(X) `\land`{=tex}
`\exists`{=tex}B`\notin`{=tex}Scope\_{form}(X):
`\epsilon`{=tex}*{struct}(`\phi`{=tex}*B)\<`\theta`{=tex}*{struct}
`\land`{=tex} TE(X,B)\>`\theta`{=tex}*{effect}, \]

with the required number of independent targets specified by the
experimental protocol.

This preserves the paper's central claim that transfer is part of
validated concept status while distinguishing candidate formation from
retrospective validation.

# **13.19 Promotion Candidate Score**

A simple test-allocation score is:

\[ PromotionScore\_{candidate}(X) = N\_{ISF}(X) `\cdot`{=tex} N\_{ER}(X)
`\cdot`{=tex} N\_{TP\_{prior}}(X). \]

The score controls candidate priority and access to expensive validation
tests. It cannot itself produce scientific validation.

# **13.20 Memory Fitness**

Let memory fitness be:

\[ MF(X,t) = F_M( N\_{ISF}, N\_{ER}, N\_{TP\_{prior}}, N\_{TP\_{emp}},
N\_{FO}, N\_{Context}, N_E; W_t ). \]

Components are active only when causally available. (TP\_{emp}) affects
later retention after actual transfer evidence exists. (N_E) is active
only for outcome-comparable trajectory memories.

Memory fitness affects cognitive residency, replay, and pruning; it must
remain logically distinct from scientific validation state.

# **13.21 Bounded Candidate Search**

For memory (X), expensive comparison is restricted to a candidate
neighborhood (C_t(X)) generated by cheap indices such as:

-   temporal locality,
-   action compatibility,
-   transformation signature,
-   memory level,
-   context signature,
-   typed neighborhood signatures,
-   previously validated correspondences.

The framework does not require complete-graph construction or
unrestricted subgraph isomorphism.

# **13.22 Conditional Compression Proposition**

The paper does not claim an unconditional theorem of sublinear memory
growth.

It makes the following empirical proposition:

> If the interaction distribution contains recurrent compressible
> structure, if higher-level representations can replace redundant
> lower-level representations without materially degrading predictive,
> explanatory, transfer, or planning quality, and if the rate of
> genuinely novel irreducible structure is bounded, then persistent
> consolidated memory should grow sublinearly relative to cumulative
> experience.

For cumulative experience count (E) and persistent consolidated memory
(M(E)), the predicted asymptotic signature is:

\[ `\frac{M(E)}{E}`{=tex}`\rightarrow 0`{=tex} \]

over sufficiently long regimes satisfying those conditions.

Failure under environments that satisfy the stated conditions counts
against the compression hypothesis. Linear growth in environments with
continually novel irreducible structure does not by itself falsify it.

# **13.23 Formal Scope of the Theory**

The formal objects above separate four claims that should not be
conflated:

1.  **Ontology:** task-semantic entities are not primitive.
2.  **Development:** increasingly abstract structures arise from
    remembered interaction.
3.  **Validation:** transfer and explanatory claims require explicit
    empirical evidence.
4.  **Resource dynamics:** cognition retains only a bounded,
    competitively selected subset of available representation.

The theory is therefore a developmental and empirical framework rather
than a proof that any particular implementation must converge or become
generally intelligent.

# **14. Formal Definition of Interaction Significance**

The Interaction Significance Function allocates developmental resources.
It does not define the behavioral objective. Primary valence supplies
signed motivational evidence as a separate primitive channel, while ISF
controls how much processing an interaction receives.

# **14.1 Definition**

For interaction (i) at developmental time (t):

\[ ISF(i,t) = F\_{ISF} `\left`{=tex}( N\_{PVI}(i,t), N\_{OSI}(i,t),
N\_{PE}(i,t), N\_{`\widehat{LV}`{=tex}}(i,t), N\_{TP\_{prior}}(i,t),
N\_{`\widehat{EP}`{=tex}}(i,t); W_t `\right`{=tex}). \]

  Symbol                   Meaning
  ------------------------ -----------------------------------
  (PVI)                    Primary-Valence Impact
  (OSI)                    Option-Structure Impact
  (PE)                     Empirical Prediction Error
  (`\widehat{LV}`{=tex})   Prospective Learning Value
  (TP\_{prior})            Prospective Transfer Prior
  (`\widehat{EP}`{=tex})   Prospective Explanatory Potential
  (W_t)                    Developmental weighting state

Inactive components are omitted or renormalized rather than assigned
artificial extreme values.

The framework does not prescribe universal coefficients. It requires
that (F\_{ISF}) be bounded, auditable, and monotone in active evidence
channels unless an implementation explicitly tests a different
hypothesis.

# **14.2 Primary-Valence and Option-Structure Impact**

Primary valence and option structure are orthogonal evidence dimensions.
The ISF may use bounded magnitude of primary valence for developmental
priority, while the sign remains separately available for memory
valuation and planning.

## **Option-Structure Impact**

At Stage 0:

\[ OSI_0(i) = d\_{OE} ( OE_0\^{before}, OE_0\^{after} ). \]

After graph-based reachability exists:

\[ OSI_t(i) = g( \|`\Delta`{=tex}`\widehat{FO}`{=tex}\_t\^k(i)\| ), \]

where (g) is bounded.

The direction of future-option change is stored separately:

\[ DirFO_t(i) = sign( `\Delta`{=tex}`\widehat{FO}`{=tex}\_t\^k(i) ). \]

This distinction is essential. A large structural contraction may be
highly significant without the theory declaring contraction
intrinsically bad; a large expansion may be significant without being
intrinsically good.

# **14.3 Prediction Error**

If an expectation was active before interaction (i):

\[ PE(i,t) = d\_`\Delta`{=tex} ( `\Delta`{=tex}*i,
`\widehat{\Delta}`{=tex}*{i,t} ). \]

Otherwise (PE) is inactive.

Prediction error identifies inadequacy in a supported predictive
relation. It may trigger deeper analysis or context refinement, but it
is not synonymous with significance.

# **14.4 Prospective Learning Value**

At decision time:

\[ `\widehat{LV}`{=tex}(i,t) = `\mathbb{E}`{=tex} \[ `\Delta`{=tex}K
`\mid`{=tex} `\mathcal{E}`{=tex}\_{`\le`{=tex}t}\], \]

where (`\Delta`{=tex}K) is an implementation-declared knowledge-gain
measure.

A later empirical quantity may be recorded:

\[ LV\_{emp}(i,t+k), \]

but it can influence only subsequent decisions.

# **14.5 Transfer Prior and Empirical Transfer**

At candidate time:

\[ TP\_{prior}(X,t) = `\widehat{P}`{=tex} (
`\text{held-out structural reuse}`{=tex} `\mid`{=tex}
`\mathcal{E}`{=tex}\_{`\le`{=tex}t} ). \]

After admissible tests:

\[ TP\_{emp}(X) =
`\frac{ SuccessfulHeldOutTests(X) }{ AdmissibleHeldOutTests(X) }`{=tex}.
\]

(TP\_{prior}) allocates tests. (TP\_{emp}) contributes to later
retention and validated concept status.

# **14.6 Explanatory Potential and Realized Reach**

At decision time:

\[ `\widehat{EP}`{=tex}(X,t) = `\mathbb{E}`{=tex} \[ ER\_{future}(X)
`\mid`{=tex} `\mathcal{E}`{=tex}\_{`\le`{=tex}t}\]. \]

After lower-level structures have actually been explained or replaced:

\[ ER\_{emp}(X,t+k) = \|Supports\_{t+k}(X)\|. \]

The prospective and realized quantities must remain separate in
experimental records.

# **14.7 Causal Availability and Score Snapshots**

Every significance or fitness decision should record at least:

-   `score_step`,
-   `evidence_available_step`,
-   `evidence_kind`,
-   raw component vector,
-   normalized component vector,
-   `developmental_stage_snapshot`,
-   `next_developmental_stage`,
-   `score_schema_version` (or equivalent immutable scoring-definition
    identity).

Admissibility requires:

\[ evidence_available_step `\le`{=tex} score_step. \]

Later evidence may revise future decisions but may not overwrite the
historical causal record.

# **14.8 Developmental Weighting**

Let:

\[ W_t = \[w\_{PVI}, w\_{OSI}, w\_{PE}, w\_{LV}, w\_{TP}, w\_{EP}\]\_t.
\]

The canonical stage model is Stage 0 through Stage 7. The current stage
is fixed for one evaluation interval; evidence created during that
interval may influence only the next inferred stage.

### **Stage 0 --- Interaction Seeding**

Active emphasis:

-   recurrence and novelty outside the ISF aggregation,
-   primary-valence impact when present,
-   immediate option-structure impact (OSI_0).

(PE) is inactive.

### **Stage 1 --- Contingency Development**

Higher weight on:

-   (PE),
-   (`\widehat{LV}`{=tex}),
-   (OSI).

### **Stage 2 --- Abstraction Development**

Higher weight on:

-   (`\widehat{LV}`{=tex}),
-   (TP\_{prior}),
-   (`\widehat{EP}`{=tex}).

### **Stage 3 --- Transfer Development**

Higher weight on:

-   (TP\_{prior}),
-   (`\widehat{EP}`{=tex}).

Actual (TP\_{emp}) influences later memory fitness but not the original
candidate score.

### **Stage 4 --- Consequence Integration and Initial Planning**

Learned future-option, consequence, and relational evidence receive
greater fitness and planning weight.

### **Stage 5 --- Outcome-Equivalence and Preference Development**

Persistent M6 outcome classes and learned preference relations over
reachable outcomes become available. Preference remains distinct from
equivalence and from primary valence.

### **Stage 6 --- Alternative-Strategy Linkage**

Multiple grounded M7 strategies may be associated with the same learned
M6 outcome or another explicitly admissible comparison class. This
establishes substitutability evidence without yet requiring demonstrated
replanning.

### **Stage 7 --- Replanning and Strategy Efficiency**

Demonstrated strategy substitution and outcome-conditioned efficiency
become active. M7 retains realized interaction cost to the represented
M6 outcome, and non-zero primary-valence boundaries may provide
additional actions-to-valence evidence. Efficiency is compared only
inside the corresponding outcome-comparison class.

Weights used during interval (t) are fixed before evidence from that
interval is incorporated into (Stage\_{t+1}).

# **14.9 Attention Allocation**

An implementation may allocate attention through a bounded softmax:

\[ P(Attend=i) =
`\frac{ \exp(\tau ISF_i) }{ \sum_j\exp(\tau ISF_j) }`{=tex}, \]

subject to explicit exploration floors and compute budgets.

This is preferable to treating (ISF/`\sum`{=tex}ISF) as mandatory
because some components may be inactive or tied.

# **14.10 Replay Allocation**

Replay can similarly use memory fitness:

\[ P(Replay=X) =
`\frac{ \exp(\tau_R MF(X)) }{ \sum_Y\exp(\tau_R MF(Y)) }`{=tex}. \]

Replay remains a resource-allocation decision, not scientific validation
evidence by itself.

# **14.11 Retention and Pruning**

A bounded retention model may be:

\[ P(Retain=X) = `\sigma`{=tex}( a,MF(X)-b ). \]

Operational systems should use hysteresis, minimum evidence volumes,
dependency protection, and provenance preservation before irreversible
retirement.

# **14.12 Promotion Candidate Score**

For an abstraction candidate:

\[ PromotionScore\_{candidate}(X) = N\_{ISF}(X) `\cdot`{=tex} N\_{ER}(X)
`\cdot`{=tex} N\_{TP\_{prior}}(X). \]

This score controls allocation of expensive contextual or transfer
testing.

It does **not** validate a concept.

# **14.13 Validated Concept Status**

Validated concept status requires the separate conditions defined in
Section 13:

\[ ValidatedConcept(X) `\Rightarrow`{=tex} StructuralCorrespondence(X)
`\land`{=tex} HeldOutEmpiricalTransfer(X). \]

No frequency, ISF, explanatory score, replay count, or transfer prior
may substitute for required held-out evidence.

# **14.14 Significance Across Memory Levels**

The same evidence categories may be computed for episodes,
contingencies, families, roles, concepts, and higher-level structures,
but their operational definitions may differ by level. An implementation
must therefore declare:

-   the comparison class,
-   the normalization method,
-   the active evidence channels,
-   the developmental stage,
-   the validation contract.

A common notation does not imply that evidence is interchangeable across
memory levels.

# **14.15 Significance Principle**

The testable claim is:

> Developmental resource allocation based on causally available
> structural impact, prediction error, prospective learning value,
> transfer prior, and explanatory potential should produce better
> compression and held-out reuse than allocation based on frequency,
> recency, or prediction error alone.

This statement replaces the stronger and less defensible claim that the
ISF constitutes a fundamental objective of intelligence.

# **15. Formalization of Memory Graphs, Roles, Concepts, and Explanatory Reach**

The previous sections introduced interactions, contingencies, memory
competition, future-option structure, and the Interaction Significance
Function. This section formalizes the graph structures that serve as the
primary substrate of cognition. The central claim is:

Intelligence operates on memory graphs rather than isolated memories.

# **15.1 The Memory Graph**

The cognitive state of the agent is represented as a dynamic graph: G =
(V, E)

Where: V = memory structures. E = relationships between memory
structures

The graph evolves continuously throughout development. New nodes appear.
New edges appear. Existing structures may be merged, split, promoted, or
removed.

# **15.2 Memory Nodes**

Nodes may exist at multiple abstraction levels.

### **M0 Nodes**

Episodes. Individual remembered interactions.

### **M1 Nodes**

Contingencies. Recurring action-consequence structures.

### **M2 Nodes**

Interaction Families. Compressed groups of contingencies.

### **M3 Nodes**

Functional Roles. Reusable structural positions.

### **M4 Nodes**

Concepts.Transferable explanatory abstractions.

### **M5 Nodes**

World-Model and consequence components. Large-scale explanatory and
predictive structures.

### **M6 Nodes**

Outcome-equivalence abstractions and strategy structures. Outcome nodes
represent recurrent classes of consequentially similar future states or
trajectory endpoints; strategy nodes represent procedures or
trajectories linked to those outcomes.

# **15.3 Edge Types**

The graph contains multiple edge classes.

## **Temporal Edges**

A → B Interaction B occurred after interaction A. These support sequence
learning.

## **Co-Occurrence Edges**

A ↔ B Structures frequently appear together. These support association.

## **Dependency Edges**

A ⇒ B B depends on A. These support causal discovery.

## **Enablement Edges**

A ⟶+ B A increases accessibility of B. These support future-option
estimation.

## **Blocking Edges**

A ⟶− B A restricts accessibility of B. These support obstacle discovery.

## **Explanatory Edges**

A ⟶E B A helps explain B. These become critical for promotion.

## **Similarity Edges**

A ≈ B A and B share structural properties. These support transfer.

## **Outcome-Equivalence Edges**

A ≡Q B A and B belong to the same learned outcome-equivalence class
under the current consequence descriptor.

## **Strategy-to-Outcome Edges**

π ⟶Ω Ω Trajectory or strategy π has empirical evidence of leading to
outcome abstraction Ω under its applicability context.

## **Preference Edges**

Ωa ≻Γ Ωb The learner exhibits a stable learned comparative preference
for outcome class Ωa over Ωb under context Γ. Such edges require
realized primary-valence evidence under comparable reachable
alternatives; they are not created merely because a planner previously
selected Ωa.

# **15.4 Relational Neighborhood**

Let (N(X)) denote the typed local neighborhood of node (X).

The theory proposes that the **functional interpretation** of (X) is
strongly constrained by the structure of (N(X)):

\[ FunctionalIdentity(X) `\approx`{=tex} f( N(X) ). \]

This is a relational hypothesis rather than a claim that intrinsic
perceptual properties are irrelevant. The empirical question is whether
neighborhood structure contributes more to role and transfer prediction
than appearance alone.

# **15.5 Functional Roles**

A functional role is defined as a recurring graph position. The role is
not determined by appearance. The role is determined by connectivity
patterns.

A role is represented by a recurrent typed neighborhood-equivalence
class:

\[ Role(X) = Class( N(X) ), \]

where the equivalence or similarity criterion is explicitly defined by
the implementation and evaluated across distinct carriers or contexts.

## **Example: Enabler**

An Enabler tends to satisfy:

Input → Enabler → Expanded Reachability

The specific carrier is irrelevant. A key, switch, password, or lever
may instantiate the same role.

## **Example: Blocker**

Input → Blocker → Reduced Reachability

Different appearances may still occupy the same graph position.

# **15.6 Role Similarity**

Role similarity is defined structurally. For two nodes (A) and (B):

\[ RoleSimilarity(A,B) = Sim_R( N(A), N(B) ). \]

For the present theory, (Sim_R) is operationalized as a **bounded typed
graph-neighborhood similarity** rather than an unrestricted
graph-isomorphism test or a learned embedding requirement. Its purpose
is to identify structurally plausible correspondence candidates cheaply
enough to participate in continuous developmental memory.

Let (D_R(X)) denote a bounded descriptor of the typed local neighborhood
of memory (X). The descriptor may contain only information already
causally available to the learner, including:

-   typed incoming and outgoing relation counts,
-   neighboring memory-level and memory-type distributions,
-   bounded local dependency structure,
-   enablement and blocking patterns,
-   bounded future-option profile,
-   consequence-structure profile,
-   context-partition information,
-   previously validated structural correspondences.

The implementation defines a finite set of normalized similarity
components (S_i(A,B)`\in[0,1]`{=tex}). A baseline role-similarity score
is:

\[ Sim_R(A,B) = `\frac{ \sum_i w_i S_i(A,B) }{ \sum_i w_i }`{=tex},
`\qquad`{=tex} w_i `\ge 0`{=tex}. \]

A reference decomposition is:

\[ Sim_R(A,B) = NormWeightedMean( S\_{rel}, S\_{level}, S\_{dep},
S\_{enable}, S\_{FO}, S\_{consequence}, S\_{context} ), \]

where:

-   (S\_{rel}) compares typed relation patterns;
-   (S\_{level}) compares the distribution of neighboring memory levels
    and types;
-   (S\_{dep}) compares bounded dependency structure;
-   (S\_{enable}) compares enabling and blocking structure;
-   (S\_{FO}) compares bounded future-option effects;
-   (S\_{consequence}) compares learned consequence profiles when
    causally available;
-   (S\_{context}) compares learned contextual applicability.

The exact component functions, weights, neighborhood radius, thresholds,
and candidate budgets are implementation parameters and must be reported
in experiments. They are not task-semantic labels.

## **15.6.1 Bounded Candidate Generation**

Role similarity must not require all-pairs comparison over persistent
memory.

For a new or changed memory (X), a bounded candidate set (C_t(X)) is
generated using cheap structural indices already permitted by Section
13.21, including memory level/type, relation signatures, neighborhood
signatures, context partitions, future-option buckets, consequence
buckets, and previously validated correspondences.

Exact (Sim_R) evaluation is restricted to (Y`\in`{=tex}C_t(X)), with:

\[ \|C_t(X)\| `\le`{=tex}K \]

for a declared implementation budget (K).

This makes graph-neighborhood similarity an incremental local operation
rather than a global graph scan.

## **15.6.2 Similarity Is Candidate Evidence, Not Validation**

A high (Sim_R(A,B)) does **not** establish that (A) and (B) are the same
concept, role, or transferable abstraction.

It establishes only that the pair is structurally plausible enough to
justify further evidence gathering.

The developmental sequence is:

\[ High Sim_R `\rightarrow`{=tex} Structural Correspondence Candidate
`\rightarrow`{=tex} Held`\text{-}`{=tex}out Probe `\rightarrow`{=tex}
Empirical Transfer Evidence. \]

Similarity edges (A`\approx`{=tex}B) may therefore be provisional.
Promotion into a validated cross-context role or concept requires the
existing transfer and held-out evidence contracts. Failed probes,
contradictions, or consequence divergence count against the
correspondence and may trigger context refinement rather than forced
merging.

Canonical memory identity remains distinct from similarity. Two memories
may retain different `MemoryUid` identities while carrying a similarity
or transfer-correspondence relation.

## **15.6.3 Runtime Requirement**

The similarity mechanism is required to preserve the continuous-memory
execution model:

-   actors must not block on similarity analysis;
-   M0-M7 derivation must not wait for global similarity computation;
-   candidate generation and exact comparison operate only on bounded
    affected neighborhoods;
-   similarity work may run asynchronously from dirty-memory
    notifications;
-   canonical graph mutation remains deterministic and
    reducer-controlled;
-   overload must reduce or defer similarity work rather than create
    unbounded queues or global barriers.

The empirical prediction remains:

\[ Perf( Transfer`\mid`{=tex}RoleSimilarity ) \> Perf(
Transfer`\mid`{=tex}AppearanceSimilarity ) \]

under held-out comparison.

A stronger falsifiable prediction is that bounded typed
graph-neighborhood similarity should improve cross-context
transfer-candidate precision and/or recall over exact canonical-key
matching alone, without materially degrading continuous-runtime
throughput at the declared similarity budget.

# **15.7 Concepts**

A concept is not merely a cluster. A concept is a graph structure
satisfying three properties:

### **Compression** The concept explains multiple lower-level structures.

### **Transfer** The concept succeeds across environments.

### **Explanatory Reach** The concept organizes substantial graph regions.

Formally, candidate formation and validation are separate:

\[ CandidateConcept(X) `\iff`{=tex} Compression(X) `\land`{=tex}
ER(X)\>`\theta`{=tex}*E `\land`{=tex} TP*{prior}(X)\>`\theta`{=tex}\_P.
\]

\[ ValidatedConcept(X) `\iff`{=tex} CandidateConcept(X) `\land`{=tex}
StructuralCorrespondence(X) `\land`{=tex} HeldOutEmpiricalTransfer(X).
\]

# **15.8 Explanatory Reach**

Explanatory reach is one of the most important quantities in the theory.

Let: Explains(X) represent the set of nodes explained by X.

Then: ER(X) = \|Explains(X)\|

A structure with larger explanatory reach explains more of memory.

# **15.9 Weighted Explanatory Reach**

Raw counts may be insufficient. A stronger formulation is:

ER(X) = Σ Importance(Y) for all Y ∈ Explains(X)

This rewards explanations of important structures rather than merely
numerous structures.

# **15.10 Explanatory Compression Ratio**

Compression can be measured as:

ECR(X) = Information Explained / Information Required

High ECR structures are desirable because they explain much while
requiring little representation. This provides a formal basis for
abstraction.

# **15.11 Promotion Through Explanatory Reach**

Promotion can now be formalized. A structure is promoted when:

ER(X) increases significantly and ECR(X) exceeds a threshold.

Explanatory reach and compression are necessary candidate-generation
evidence, but candidate allocation follows the common prospective score:

\[ PromotionScore\_{candidate}(X) = N\_{ISF}(X) `\cdot`{=tex} N\_{ER}(X)
`\cdot`{=tex} N\_{TP\_{prior}}(X). \]

This score allocates testing and cannot establish validated concept
status.

# **15.12 Graph Motifs**

Repeated graph structures become motifs. Examples:

### **Reversible** A ↔ B

### **Enable** A → Expanded Region

### **Block** A → Restricted Region

### **Continuation-Collapse** A → Empirically Low-Reachability Region

Motifs are higher-order structures discovered through repeated graph
analysis.

# **15.13 Graph Expansion**

Future-option discovery can be expressed graphically.

Let: Reach(X) represent reachable nodes from X.

Graph expansion is: GE(X) = \|ReachAfter(X)\| − \|ReachBefore(X)\|

Positive values indicate future-option expansion. Negative values
indicate restriction. This provides a formal approximation of
developmental future-option discovery.

# **15.14 Structural Transfer Graphs**

Let (X) have local typed neighborhood (N_A(X)) in environment graph
(G_A). A structurally plausible reuse in unseen graph (G_B) requires an
approximate typed mapping:

\[ `\phi`{=tex}: N_A(X)`\rightarrow`{=tex}G_B \]

whose edge-type preservation error satisfies:

\[ `\epsilon`{=tex}*{struct}(`\phi`{=tex})\<`\theta`{=tex}*{struct}. \]

The theory predicts that low structural error at the role, motif, and
graph-neighborhood levels is more predictive of held-out transfer than
perceptual similarity.

Structural correspondence is necessary evidence for relational transfer
but is not sufficient for validated concept status; empirical memory-on
versus memory-off effect remains required.

# **15.15 Outcome-Equivalence and Strategy Graphs**

Outcome-equivalence abstractions introduce a persistent graph layer
between consequence representation and strategy reuse. Let
(`\Omega`{=tex}\_j) denote an outcome node and
(`\Pi`{=tex}(`\Omega`{=tex}\_j)) the strategies with supported edges to
it.

The graph therefore distinguishes:

\[
`\Omega`{=tex}\_j`\quad`{=tex}`\text{from}`{=tex}`\quad`{=tex}{`\pi`{=tex}:`\pi`{=tex}`\rightarrow`{=tex}`\Omega`{=tex}\_j}.
\]

This representation supports three experimentally distinct capabilities:

1.  **Outcome recognition:** distinct endpoints are assigned to the same
    consequence class.
2.  **Alternative-strategy representation:** multiple trajectories are
    retained as routes to the same outcome.
3.  **Replanning:** failure or removal of one strategy triggers
    selection of another strategy without replacing the outcome
    abstraction.

A target-like outcome adds a learned preference relation over outcome
nodes. The graph therefore permits goal-like structure to emerge without
inserting a primitive goal node at initialization.

# **15.16 World Models**

A world model is defined as a high-level graph whose nodes are concepts
and whose edges explain large regions of experience.

Formally: WM = (Vconcepts, Eexplanatory) The purpose of a world model is
not merely prediction. Its purpose is large-scale organization of
memory. Prediction emerges as a consequence of that organization.

# **15.17 Unified Graph Interpretation**

The theory can now be summarized graphically. Interactions create nodes.
Nodes form contingencies. Contingencies form graph structures. Graph
structures form roles. Roles form concepts. Concepts form world models.
Promotion occurs through explanatory reach. Transfer candidates arise
through structural correspondence and are validated empirically.
Functional identity is inferred from relational neighborhoods.
Future-option estimates arise from bounded discovered reachability.
Outcome-equivalence nodes separate consequential future-state identity
from the strategy edges that reach it, enabling alternative-strategy
reuse and replanning. Goal-like structure, if it emerges, is represented
as a learned preference relation over outcome abstractions. Intelligence
becomes the continuous restructuring of a multi-scale memory graph under
developmental pressure.

# **16. Formal Developmental Dynamics**

Previous sections defined the structures of the theory. This section
defines how those structures evolve through time. The objective is to
formalize development as a continuous process of memory formation,
competition, promotion, graph growth, and abstraction. The central claim
is:

Intelligence is not a static architecture. Intelligence is a
developmental dynamical system.

# **16.1 Developmental State**

At time (t), define:

\[ D_t=(M_t,G_t,W_t,`\mathcal{E}`{=tex}\_t), \]

where (M_t) is multi-scale memory, (G_t) is the typed interaction graph,
(W_t) is the developmental weighting state, and (`\mathcal{E}`{=tex}\_t)
is the scientific evidence/provenance ledger.

The state evolves through specialized update operators rather than one
assumed global optimizer.

## **16.1.1 Constrained Concurrent Updates**

For update process (j):

\[ U_j(D_t,`\xi`{=tex}*t)`\rightarrow`{=tex}`\widetilde{D}`{=tex}*{t+1}.
\]

The proposed update is published only after satisfying the common
admissibility constraints (`\mathcal{C}`{=tex}):

\[ D\_{t+1} = `\Pi`{=tex}*{`\mathcal{C}`{=tex}} (
`\widetilde{D}`{=tex}*{t+1} ). \]

The constraint set includes causal evidence availability, provenance
preservation, candidate/validation separation, bounded resource use,
hysteresis, and atomic generation publication. This formulation
addresses representational thrashing without converting the architecture
into a single Variational-Free-Energy or MDL optimizer.

# **16.2 Experience Cycle**

Every interaction produces an update cycle:

Observe → Act → Transform → Evaluate Significance → Allocate Attention →
Update Memory → Update Graph → Competition → Promotion → Forgetting →
Continue

This cycle repeats indefinitely.

# **16.3 Attention Dynamics**

Attention resources are limited.

Let: A(i,t) represent attention allocated to interaction i.

One admissible bounded allocation is:

\[ P(Attend=i) = `\frac{\exp(\tau ISF(i,t))}`{=tex} {`\sum`{=tex}\_j
`\exp`{=tex}(`\tau`{=tex}ISF(j,t))}, \]

subject to an exploration floor and finite compute budget.

Attention is therefore a resource-allocation distribution over candidate
evidence.

# **16.4 Memory Formation Dynamics**

The probability that an interaction enters memory:

P(Store(i)) = σ(ISF(i)) where: σ = bounded activation function.

Higher significance interactions are more likely to enter memory.

# **16.5 Replay Dynamics**

Memory replay is essential because many abstractions emerge long after
the original experience. Replay probability may be allocated from memory
fitness:

\[ P(Replay=X) = `\frac{\exp(\tau_R MF(X,t))}`{=tex} {`\sum`{=tex}\_Y
`\exp`{=tex}(`\tau`{=tex}\_R MF(Y,t))}. \]

Replay therefore depends on current cognitive utility. Replay frequency
is not itself scientific validation evidence.

# **16.6 Memory Fitness**

Let: MF(X) represent memory fitness.

A first approximation:

MF(X,t) = F(N_ISF(X,t), N_ER(X,t), N_TP(X,t), N_R(X,t), N_E(X,t); W_t)

The inputs are normalized within comparable memory classes before
aggregation. N_TP uses TP_prior before external transfer evidence exists
and is updated by TP_emp afterward. N_E is inactive until
outcome-equivalent trajectories have been observed. Normalization must
be bounded and robust to transient outliers; raw component magnitudes
must not be directly compared across heterogeneous evidence channels.
Implementations should retain both raw and normalized component vectors,
apply explicit clipping or equivalent robust bounds, and constrain
developmental weights so that no single accidental spike can dominate
memory fitness. This formulation deliberately leaves the exact monotone
aggregation function F and developmental weights W_t to empirical
comparison rather than assigning unsupported universal coefficients.

Higher fitness increases:

-   retention,\
-   replay,\
-   promotion opportunities.

# **16.7 Forgetting Dynamics**

Memory persistence should not be permanent. The probability of
forgetting:

P(Forget(X)) = 1 / (1 + MF(X))

Structures with low fitness gradually disappear.

# **16.8 Compression Dynamics**

Compression occurs when multiple structures can be represented by a
smaller explanatory structure.

Suppose: X₁, X₂, ... Xₙ are replaced by abstraction A.

Compression benefit: CB(A) = Σ Cost(Xi) − Cost(A)

Compression becomes attractive when: CB(A) \> 0

# **16.9 Promotion Dynamics**

Promotion moves structures upward through the hierarchy.

The prospective allocation score is:

\[ PromotionScore\_{candidate}(X) = N\_{ISF}(X) `\cdot`{=tex} N\_{ER}(X)
`\cdot`{=tex} N\_{TP\_{prior}}(X). \]

Before unseen-environment validation, this score can create or
prioritize only a higher-level **candidate** representation.

Candidate promotion occurs when:

\[ PromotionScore\_{candidate}(X)\>`\theta`{=tex}\_{promotion}. \]

Validated concept status is determined separately by the held-out
evidence contract and cannot be produced by the promotion score.

# **16.10 Demotion Dynamics**

Not all abstractions survive. A promoted structure may later lose
utility, but demotion must be slower and more evidence-demanding than
promotion reversal caused by a short anomalous period.

The system therefore uses **hysteresis and probation**. In general:

theta_promotion \> theta_demotion

A structure enters probation when its relevant candidate or validated
promotion score falls below theta_demotion. Demotion is permitted only
after sustained contradictory evidence rather than a single low-scoring
interval. A minimal operational rule is:

DemotionAllowed(X) iff EvidenceVolume(X) \>= n_demotion and
FailureWindows(X) \>= k and ContextConfidence(X) \>= theta_context

If context discovery is still unstable or the current environment
appears anomalous, demotion pressure is reduced and the structure
remains in probation while additional evidence is collected. Empirical
transfer failures may contribute strongly to demotion, but only when
they are sufficiently numerous and contextually resolved.

Demotion should also preserve historical identity and provenance. A
demoted abstraction becomes inactive or candidate-only rather than being
erased immediately, allowing later reactivation if new evidence restores
its explanatory or transfer value. This prevents repeated destruction
and rediscovery of the same long-lived structure.

This prevents accumulation of obsolete abstractions while reducing
cognitive volatility under noisy or atypical experience.

# **16.11 Graph Growth**

The memory graph evolves continuously. Graph growth can be approximated
as:

GG(t) = \|V(t+1)\| − \|V(t)\| where: V = graph nodes.

Growth occurs through:

-   new experiences,\
-   new contingencies,\
-   new roles,\
-   new concepts.

# **16.12 Graph Compression**

Growth alone is insufficient. Compression must accompany growth. Graph
compression:

GC(t) = RemovedStructure(t)

Compression prevents uncontrolled expansion. Development therefore
becomes a balance between: Graph Growth and Graph Compression.

The theory makes a stronger developmental prediction: as abstractions
mature, persistent memory should grow more slowly than cumulative
experience. Let cumulative experience be E(t) and persistent
consolidated memory size be M(t). Define the **Memory Growth Ratio**
over an interval as:

MGR(t) = ΔM(t) / ΔE(t)

This prediction is conditional. In environments with recurrent
compressible structure and a bounded rate of irreducible novelty, the
theory predicts that MGR(t) declines with developmental maturity while
predictive accuracy, explanatory reach, transfer success, and planning
effectiveness remain stable or improve. A mature system should therefore
explain increasing amounts of recurring experience with proportionally
less persistent cognitive memory. Raw audit logs are excluded from
(M(t)). Linear growth in a deliberately non-recurrent stream of
irreducibly novel structure is not by itself a falsification.

# **16.13 Graph Expansion Evidence**

Future-option discovery provides an estimate of potentially unexplored
graph structure:

\[ GE_t = `\mathbb{E}`{=tex} \[
`\text{new discovered reachability}`{=tex} `\mid`{=tex}
`\mathcal{E}`{=tex}\_{`\le`{=tex}t}\]. \]

This quantity may allocate exploration effort. It is not treated as
intrinsic utility; its usefulness is an empirical hypothesis tested
through later learning, explanatory, and transfer outcomes.

# **16.14 Context Expansion Dynamics**

Contradictions trigger context discovery.

Suppose: PredictionError(X) θcontext

Then: Search(Context) is activated.

The system attempts to discover variables that reconcile conflicting
observations. Context growth therefore emerges naturally from prediction
failure.

# **16.15 Role Formation Dynamics**

Role candidates emerge when graph neighborhoods repeatedly recur.

Let: NS(X) represent neighborhood structure.

If: NS(A) ≈ NS(B) ≈ NS(C)

Then: RoleCandidate(R) is formed.

Repeated successful reuse increases promotion pressure.

------------------------------------------------------------------------

# **16.16 Concept Formation Dynamics**

Concept candidates emerge from role structures that have accumulated
sufficient compression and explanatory evidence to warrant held-out
transfer testing. A role or higher-level candidate becomes a validated
concept only after empirical transfer evidence is available. A concept
candidate becomes validated only when:

\[ ER(X)\>`\theta`{=tex}\_{explanation} \]

and the required number of held-out structural correspondences and
empirical transfer interventions satisfy their preregistered thresholds.

Concept formation is therefore driven jointly by compression and
explanation, while **validated transferable concept status** is
established retrospectively through held-out transfer evidence.

# **16.17 Developmental Stage Variables**

The theory proposes that development itself is measurable.

Let: Stage(t) represent developmental maturity. Stage is inferred from
already-established capabilities rather than imposed externally. The
canonical stages are:

### **Stage 0 --- Interaction Seeding**

Immediate recurrence, primary-valence evidence, and observable
option-structure change can seed memory without requiring prediction.

### **Stage 1 --- Contingency Capability**

Stable action-conditioned contingencies support causally valid
expectations and prediction error.

### **Stage 2 --- Structural Abstraction Capability**

Transformation families and carrier/role structure can be formed from
recurring lower-level evidence.

### **Stage 3 --- Transfer Capability**

Structural correspondence can generate held-out transfer candidates, and
empirical interventions can validate or reject target-scoped reuse.

### **Stage 4 --- Consequence Integration and Initial Planning**

Validated higher-order structure contributes to consequence
representations, learned future-option structure, and initial planning.

### **Stage 5 --- Outcome-Equivalence and Preference Capability**

The system can represent persistent M6 outcome classes independently of
the trajectories that reach them. Stable learned preference may emerge
as a separate relation over reachable outcome classes.

### **Stage 6 --- Alternative-Strategy Linkage Capability**

The system can associate multiple grounded M7 strategies with the same
learned outcome or another explicitly admissible outcome-comparison
class. Strategy identity, reliability, primary valence, and cost remain
separate.

### **Stage 7 --- Replanning and Efficient Strategy Selection Capability**

The system demonstrates substitution among alternative strategies and
can prefer lower-cost trajectories when they preserve the selected
learned outcome or an explicitly declared no-worse relation. Stage 7
therefore requires behavioral evidence beyond the mere existence of M7
records.

The stage is inferred only from causally prior evidence. Evidence
generated while Stage_t is active may update only Stage\_{t+1}.

# **16.18 Developmental Weight Evolution**

The ISF weights evolve over time.

The ISF weighting vector is:

\[ W_t= \[ w\_{PVI}(t), w\_{OSI}(t), w\_{PE}(t), w\_{LV}(t), w\_{TP}(t),
w\_{EP}(t)\]. \]

Weight updates depend on developmental stage. Early development can
emphasize primary-valence impact, immediate option-structure impact, and
recurrence-supported learning; later development can increase transfer
and explanatory emphasis. Primary valence remains motivational evidence
rather than a replacement for the other developmental channels.

To avoid simultaneous circular determination, stage and weight updates
are temporally ordered. Stage_t is inferred only from capabilities
established using evidence available before the current update interval.
W_t is then fixed for learning during that interval. Evidence produced
during the interval may update Stage\_(t+1), which determines W\_(t+1).
A capability may therefore change future weighting, but cannot alter the
weights used to generate the evidence that established that same
capability. Implementations should persist both the active
developmental-stage snapshot and the next inferred stage. The next
inferred stage cannot affect scoring until the following evaluation
interval.

# **16.19 Developmental Activity Index**

For descriptive reporting, an implementation may define a normalized
activity index:

\[ DAI(t) = h( Attention_t, Replay_t, Promotion_t, GraphExpansion_t ),
\]

where (h) is explicitly specified. This is a diagnostic summary rather
than a new theoretical primitive.

# **16.20 Developmental Equilibrium**

The theory rejects convergence toward a final fixed state. Instead:

Developmental Equilibrium = temporary balance between:

-   exploration,\
-   compression,\
-   forgetting,\
-   promotion,\
-   graph growth.

New experiences can always disrupt equilibrium. Intelligence therefore
remains adaptive rather than convergent.

# **16.21 Unified Developmental Equation**

The complete developmental cycle can be summarized as:

Experience → Significance → Attention → Memory → Competition → Replay →
Compression → Context Expansion → Promotion → Transfer → Concept
Formation → Consequence Integration → Outcome-Equivalence Formation →
Alternative Strategy Linking → Planning/Replanning → Efficiency
Selection → World-Model Reorganization → New Experience

This loop continuously restructures the memory graph.

# **16.22 Dynamic Definition of Intelligence**

The static definition from earlier sections can now be strengthened.

**Intelligence is the continuous developmental reorganization of a
multi-scale memory graph through significance-driven attention,
competition, compression, context expansion, transfer, and explanatory
promotion.**

This definition treats intelligence not as a representation, model, or
policy, but as an ongoing dynamical process.

# **17. Experimental Operationalization and Measurement**

A scientific theory must specify not only what exists, but how its
claims can be tested. The purpose of this section is to convert the
theory into measurable quantities, observable developmental milestones,
and falsifiable experiments. The central question becomes:

What observations would indicate that the proposed developmental pathway
is actually emerging?

# **17.1 General Experimental Principle**

The theory predicts a developmental sequence:

Interactions → Contingencies → Prediction Violations → Transformation
Families → Carriers → Functional Roles → Concepts →
Consequence/World-Model Structure → Outcome-Equivalence Abstractions →
Alternative Strategies → Replanning/Efficiency

Experiments should determine whether this sequence emerges naturally.
The theory is weakened if later stages emerge before earlier stages.

# **17.2 Measurement Levels**

The theory proposes measurements at five levels:

  Level   Measurement Target
  ------- -----------------------------------------------------------------
  M0      Interaction and episode statistics
  M1      Contingency emergence
  M2      Transformation-family emergence
  M3      Role emergence
  M4      Concept and transfer emergence
  M5      World-model and consequence-structure emergence
  M6      Outcome-equivalence, strategy linkage, and replanning emergence

------------------------------------------------------------------------

# **17.3 Measuring Contingency Emergence**

A contingency is considered discovered when: Prediction Accuracy(A,
Context) exceeds threshold θc.

Operational metric: Contingency Discovery Rate (CDR) = Discovered
Contingencies / Total Interactions

The theory predicts: CDR increases before role discovery begins.

# **17.4 Measuring Prediction Structure**

Prediction development can be measured using:

Prediction Error Rate: PER(t) = Average PE(t)

The theory predicts:

1.  High initial PER.\
2.  Rapid decline after contingency formation.\
3.  Temporary increases during context discovery.

These spikes become indicators of developmental restructuring.

# **17.5 Measuring Transformation Families**

Transformation-family emergence occurs when multiple contingencies
become compressed.

Define:

Compression Ratio: CR = Original Structures / Compressed Structures

The theory predicts:

CR increases before carrier emergence.

# **17.6 Measuring Carrier Emergence**

Carrier emergence can be detected through explanatory gain.

Let: PredictionWithoutCarrier

And PredictionWithCarrier represent predictive performance.

Carrier Utility: CU = PredictionWithCarrier − PredictionWithoutCarrier

A carrier is considered discovered when: CU \> θcarrier

This operationalizes the claim that objects emerge as explanatory
structures.

# **17.7 Measuring Role Emergence**

Role emergence is one of the most important milestones. Role discovery
occurs when structurally similar graph positions are grouped despite
perceptual differences.

Define:

Role Consistency Score: RCS = Average Similarity( GraphPositionA,
GraphPositionB ) across perceptually distinct carriers.

The theory predicts: RCS rises before concept formation.

# **17.8 Measuring Role Transfer**

Present multiple environments containing:

-   different appearances,\
-   similar structural functions.

Examples:

-   key,\
-   switch,\
-   password,\
-   button.

Role Transfer Rate:

RTR = Successful Role Reuses / Transfer Opportunities

The theory predicts: RTR exceeds object-category transfer during early
development.

# **17.9 Measuring Concept Emergence**

Concept evaluation should report candidate formation and validated
transfer separately.

A concept candidate requires measurable compression and explanatory
reach:

\[ CB(X)\>`\theta`{=tex}\_C, `\qquad`{=tex} ER(X)\>`\theta`{=tex}\_E. \]

Structural transfer testing then measures the best approximate typed
graph correspondence into unseen environments:

\[ `\epsilon`{=tex}\_{struct}\^{\*}(X,B) =
`\min`{=tex}*{`\phi`{=tex}}`\epsilon`{=tex}*{struct}(`\phi`{=tex}). \]

Validated transferable concept status additionally requires repeated
held-out empirical effect:

\[ TE(X,B) = Q_B^{on}-Q_B^{off} \> `\theta`{=tex}\_{effect} \]

for the required number of independent unseen targets.

A multiplicative summary score may be reported descriptively, but no
scalar summary may substitute for a missing required validation
component.

# **17.10 Measuring Future-Option Discovery**

Future-option theory requires distinguishing the agent's discovered
reachability from objective environmental reachability.

Online, for finite horizon (k):

\[ `\widehat{FO}`{=tex}*t\^k(s) = `\sum`{=tex}*{v`\in`{=tex}V_t}
`\mathbf{1}`{=tex} \[ d\_{G_t}(s,v)`\le`{=tex}k\]. \]

A finite measure or weighted form is used when raw cardinality is
inappropriate.

For evaluation environments in which true bounded reachability can be
computed offline, define an oracle quantity (FO\_\*\^k(s)).
Future-option discovery can then be measured as:

\[ FOD(t) = Agreement( `\Delta`{=tex}`\widehat{FO}`{=tex}*t\^k,
`\Delta`{=tex}FO*\*\^k ). \]

If exhaustive oracle reachability is unavailable, matched held-out
interaction probes can estimate the same quantity. The theory predicts
that useful bounded future-option estimates emerge after contingency
structure but before effective planning based on those estimates.

# **17.10.1 Measuring Outcome-Equivalence, Target-Like Preference, and Replanning**

Outcome-equivalence emergence must be measured separately from
trajectory efficiency.

Let (`\Omega`{=tex}) be a learned outcome class and (Q_t) the learner's
consequence descriptor. Useful measurements include:

### **Outcome-Equivalence Consistency (OEC)**

Compare held-out within-class and between-class consequence distances:

\[ OEC = 1- `\frac{\mathbb{E}[d_Q\mid same\ learned\ class]}`{=tex}
{`\mathbb{E}`{=tex}\[d_Q`\mid`{=tex}different learned class\]+`\epsilon`{=tex}}.
\]

A stronger evaluation compares learned classes against externally
defined outcome equivalence without exposing those labels online.

### **Strategy Multiplicity per Outcome (SMO)**

\[
SMO(`\Omega`{=tex})=\|{`\pi`{=tex}:LeadsTo(`\pi`{=tex},`\Omega`{=tex})`\ge`{=tex}`\theta`{=tex}\_L}\|.
\]

The theory predicts that mature outcome abstractions can support more
than one strategy when the environment affords alternatives.

### **Replanning Recovery Rate (RRR)**

Ablate or invalidate the currently preferred strategy while preserving
the represented outcome class:

\[ RRR = `\frac{SuccessfulAlternativeStrategySelections}`{=tex}
{PrimaryStrategyAblationTrials}. \]

This directly tests whether outcome identity is represented
independently of a specific trajectory.

### **Target-Like Outcome Stability (TOS)**

When several learned outcome classes are simultaneously reachable,
measure whether a comparative preference for one class persists across
repeated matched contexts. A target-like outcome requires stable
preference above a preregistered threshold; outcome equivalence alone
does not.

These measurements distinguish four phenomena that conventional goal
language can obscure: outcome recognition, outcome preference, strategy
association, and replanning.

# **17.11 Measuring Interaction Efficiency**

Efficiency can be measured only after the system has discovered learned
outcome-equivalence classes or otherwise established admissible outcome
comparability. The relevant question is not whether a trajectory is
short in isolation, but whether it achieves the same learned outcome
class or an explicitly no-worse outcome at lower cost.

Define:

Path Cost: PC(π) = number of interactions in trajectory π

Future-Option Efficiency: FOE(π) = ΔFO(π) / PC(π)

For solved episodes:

Benchmark Outcome Efficiency: BOE(π) = BenchmarkOutcomeReached(π) /
PC(π)

A normalized practical metric is:

NormalizedBenchmarkEfficiency = BestKnownSolutionLength /
ObservedSolutionLength

where 1.0 indicates the best known solution length and lower values
indicate less efficient solutions.

Additional operational metrics include:

\* steps_to_success

\* best_known_solution_length

\* wasted_action_ratio

\* blocked_action_ratio

\* loop_ratio

\* repeated_state_ratio

\* future_option_gain_per_action

\* equivalent_outcome_cost_gap

The theory predicts that efficiency improves after contingency discovery
and future-option discovery, not before them.

# **17.12 Measuring Graph Expansion**

Graph expansion is a developmental milestone.

Graph Expansion Rate: GER(t) = New Reachable Nodes / Total Nodes

The theory predicts: Periods of rapid learning correlate with elevated
GER.

# **17.13 Measuring Context Discovery**

Context discovery occurs when contradictions are resolved through
additional variables.

Define: Contradiction Resolution Rate:

CRR = Resolved Contradictions / Observed Contradictions

The primary comparison is held-out predictive or transfer error after
two matched interventions:

\[ Gain\_{context} = Error\_{replacement} - Error\_{refinement}. \]

The theory predicts positive (Gain\_{context}) on contradiction sets
where an admissible separating contextual variable exists. It does not
predict that every contradiction should cause a split.

# **17.14 Measuring Explanatory Reach**

Explanatory Reach: ER(X) = \|Explains(X)\|

Operationally: ER can be estimated as the number of lower-level
structures whose behavior can be predicted or compressed using X.

The theory predicts: ER is a stronger predictor of promotion than
frequency.

# **17.15 Measuring Promotion**

Promotion events can be counted directly. Promotion Rate:

PR(t) = Promotions / Unit Time

The theory predicts: Promotion frequency increases during periods of
abstraction formation.

# **17.16 Measuring Memory Competition**

Memory selection can be evaluated by tracking cognitive retention.
Memory Retention Ratio:

MRR = Retained Memories / Stored Memories

The theory predicts: High-ISF memories remain cognitively resident
significantly longer than low-ISF memories.

# **17.17 Measuring Transfer**

Transfer is a central outcome variable. The strongest measurement is a
matched intervention:

\[ TE(X,B) = Q_B\^{on} - Q_B\^{off}, \]

where (B) is outside the frozen formation scope of (X), the two
conditions are matched as closely as possible, and the direction of
(Q_B) is defined so that positive (TE) denotes improvement.

When exact paired rollouts are not possible, the report must identify
the weaker attribution design rather than treating observational reuse
as equivalent evidence.

The theory predicts: Transfer correlates more strongly with:

-   role similarity,\
-   graph similarity,\
-   future-option similarity,

than with appearance similarity.

Transfer evaluation must report prospective and retrospective quantities
separately. TP_prior measures which structures the system expected to
transfer before testing. TP_emp measures actual held-out reuse.
Calibration between TP_prior and TP_emp is itself an experimental
result; only TP_emp validates the transfer component of concept status.
\# **17.17.1 Measuring Developmental Memory Compression**

Long-horizon experiments should measure whether memory growth becomes
sublinear relative to accumulated experience. Let:

MGR(t) = Δ Persistent Consolidated Memory / Δ Experience

Additional measures should include:

-   bytes or records retained per interaction,
-   lower-level structures summarized or superseded by higher-level
    structures,
-   explanatory reach per unit of persistent memory,
-   transfer success per unit of persistent memory,
-   predictive performance per unit of persistent memory.

The prediction applies to regimes with recurrent compressible structure
and bounded rates of irreducible novelty. Under those conditions, MGR
should decline as development progresses while predictive accuracy,
explanatory reach, transfer, and planning quality remain stable or
improve. Approximately linear persistent-memory growth under such
conditions weakens the compression hypothesis. Linear growth in a
deliberately non-recurrent stream of irreducibly novel structure does
not by itself falsify it.

# **17.18 Measuring World-Model Emergence**

World models emerge when concepts become integrated. World-Model
Coherence:

WMC = Explained Experiences / Total Experiences

The theory predicts: World-model quality increases gradually rather than
appearing suddenly.

# **17.19 Developmental Milestones**

The theory predicts observable developmental milestones.

### **Milestone 1** Stable contingencies.

### **Milestone 2** Prediction violations drive attention.

### **Milestone 3** Transformation-family formation.

### **Milestone 4** Carrier emergence.

### **Milestone 5** Role emergence.

### **Milestone 6** Role transfer.

### **Milestone 7** Concept emergence.

### **Milestone 8** World-model or consequence-structure emergence.

### **Milestone 9** Outcome-equivalence abstraction.

### **Milestone 10** Multiple strategies linked to a shared outcome.

### **Milestone 11** Replanning after strategy failure and outcome-conditioned efficiency selection.

The ordering itself is a testable claim. A complete integrated world
model is not required to precede every local outcome-equivalence
abstraction; that dependency should be reported empirically.

# **17.20 Benchmark Environments**

Experiments should begin with environments exposing only:

-   observations,\
-   actions,\
-   terminal states.

No semantic labels. No object annotations. No predefined concepts.
ARC-style interactive environments are particularly suitable because
they minimize prior assumptions.

# **17.21 Primary Evaluation Question**

The primary evaluation question is not: How many tasks are solved?

The primary evaluation question is: Do transferable abstractions emerge
through the proposed developmental pathway?

Benchmark performance remains important but is secondary to
developmental evidence.

# **17.22 Falsifiability Through Measurement**

The theory becomes scientifically meaningful because each major claim
now corresponds to measurable quantities:

-   contingency discovery,\
-   role discovery,\
-   concept emergence,\
-   transfer,\
-   graph expansion,\
-   future-option discovery,\
-   explanatory reach,\
-   world-model emergence.

Failure of these measurements to appear in the predicted order weakens
the theory.

# **17.23 Experimental Definition of Success**

The theory is supported if experiments demonstrate that:

1.  Contingencies emerge before concepts.\
2.  Transformations emerge before objects.\
3.  Roles emerge before object categories.\
4.  Future-option motifs emerge before planning.\
5.  Transfer correlates more strongly with role similarity than
    appearance similarity.\
6.  Explanatory reach predicts promotion better than frequency.\
7.  Concepts emerge from transferable interaction structures.\
8.  World-model or local consequence structures emerge from previously
    discovered abstractions.\
9.  Outcome-equivalence abstractions emerge as persistent
    representations distinct from individual trajectories.\
10. Multiple strategies can become linked to the same learned outcome
    class when alternatives exist.\
11. Replanning can substitute an alternative strategy while preserving
    the selected outcome representation.\
12. Stable target-like preference, when present, emerges over learned
    outcome classes rather than being supplied as a primitive goal.\
13. Efficient strategies emerge only after learned outcome comparability
    exists.\
14. Among trajectories with the same learned outcome or an explicitly
    no-worse outcome, lower-cost trajectories receive stronger memory
    fitness and planning preference.

Demonstrating this sequence would provide evidence for the
interaction-first, memory-centric account of intelligence proposed in
this paper.

------------------------------------------------------------------------

# **Boundary Conditions and Structural Prior Budget**

The theory does not claim that intelligence emerges from an observation
stream containing no usable structure.

Semantic abstention is distinct from structural abstention.

Let

\[ `\mathcal{R}`{=tex}\_O \]

denote the explicitly declared set of domain-generic structural
relations available at the observation boundary. Examples may include
temporal order, equality, coordinate identity, adjacency, or other
representation-specific relations.

The theory's narrower claim is:

> **Given an observation interface with an explicitly declared
> structural prior budget, can semantic, functional, causal, conceptual,
> and symbolic organization emerge without those organizations
> themselves being predefined?**

The contribution of (`\mathcal{R}`{=tex}\_O) must therefore be
experimentally measured rather than treated as negligible.

A structural-prior ablation series should compare progressively richer
interfaces, for example:

\[ S_0={`\text{identity, temporal order}`{=tex}}, \]

\[ S_1=S_0`\cup`{=tex}{`\text{equality}`{=tex}}, \]

\[ S_2=S_1`\cup`{=tex}{`\text{coordinates}`{=tex}}, \]

\[ S_3=S_2`\cup`{=tex}{`\text{adjacency}`{=tex}}, \]

with richer topology supplied only in explicitly declared additional
conditions.

For spatial environments, required controls include fixed coordinate
permutation, changing coordinate permutation where experimentally
meaningful, and representations in which spatial adjacency is withheld.

The purpose is not to demonstrate intelligence from structure-free
input. It is to identify which structural priors are necessary, which
are merely sample-efficient, and which can be reconstructed through
interaction.

# **17.24 Learned Relational Reasoning Over Developmental Memory**

The preceding theory explains how grounded, auditable relational
structure can develop from interaction. Version 0.7.0 adds a second
question: once such structure exists, can a learned relational operator
acquire reusable reasoning regularities over it?

Let the explicit developmental memory at time (t) be:

\[ G_t = (V_t, E_t, P_t) \]

where (V_t) contains typed M0-M7 memories, (E_t) contains typed
evidence-supported relations, and (P_t) contains provenance, lifecycle,
context, authority, grounding, and validation state.

For a decision, the learner retrieves a bounded relevant subgraph:

\[ G_t\^{(b)} `\subseteq `{=tex}G_t \]

and applies a learned relational operator:

\[ Z\_{k+1} = R\_{`\theta`{=tex}\_n}(G_t\^{(b)}, X_t, Y_k, Z_k) \]

where:

-   (X_t) is the current grounded state;
-   (Y_k) is the current candidate;
-   (Z_k) is an ephemeral reasoning workspace;
-   (R\_{`\theta`{=tex}\_n}) is the currently published learned
    relational model;
-   \(k\) indexes recursive deliberation cycles.

The candidate may then be refined:

\[ Y\_{k+1} = Refine(X_t, Y_k, Z\_{k+1}) \]

This introduces learned computation over memory without changing the
ontology of persistent knowledge.

## **17.24.1 Authority Boundary**

The explicit developmental memory remains authoritative for persistent
knowledge. A learned reasoner may estimate:

-   memory relevance;
-   structural similarity;
-   correspondence likelihood;
-   likely consequences;
-   strategy quality;
-   candidate refinements;
-   useful next reasoning operations.

A learned-model output is not itself sufficient evidence for:

-   M4 concept validation;
-   causal transfer success;
-   symbol grounding;
-   canonical identity;
-   lifecycle promotion;
-   primary valence;
-   persistent causal authority.

Any persistent change proposed using learned reasoning must pass through
the same evidence, provenance, developmental, and causal-validation
mechanisms as other Hydra proposals.

This boundary is theoretically important. Otherwise the learned model
could manufacture the high-level structures whose emergence the theory
is intended to test.

## **17.24.2 Developmental Dependence**

The learned reasoner may consume multiple memory levels simultaneously,
but its training targets must preserve the causal order of evidence.

Later consolidation may supervise which earlier structures became
useful, but the implementation must preserve the temporal relation
between:

1.  evidence available when a prediction or decision was made;
2.  later evidence used to evaluate that prediction or decision;
3.  model version used during the decision;
4.  memory-graph version used during the decision.

Training on future consolidation is permitted as retrospective learning.
Scientific evaluation must not reinterpret that later evidence as
information that was available to the earlier behavior.

## **17.24.3 Learned Relational Representation**

The reasoner is expected to learn representations that are sensitive to
typed relations and functional structure rather than
environment-specific semantic labels.

The central prediction is not that a particular neural architecture is
necessary. HGT-style relational attention is the reference
implementation in Hydra v9.7.2. The theoretical claim concerns a broader
class of learned relational operators operating over bounded
developmental-memory subgraphs.

A successful learned representation should improve discrimination of
structurally useful memory while retaining the theory's existing
requirements for held-out causal validation.

## **17.24.4 Recursive Deliberation**

Reasoning depth may arise through repeated bounded applications of the
learned operator rather than through one monolithic inference pass.

A deliberation cycle may:

1.  retrieve or expand relevant memory;
2.  score correspondences or consequences;
3.  refine a candidate;
4.  invoke another reasoning operation;
5.  stop when the declared budget or stopping criterion is reached.

Recursive deliberation remains epistemically subordinate to
environmental evidence. Additional internal cycles can improve a
hypothesis but cannot convert an unvalidated hypothesis into causal
fact.

## **17.24.5 Continual Relational Learning**

Let (`\theta`{=tex}*n) denote immutable published model version (n). New
evidence may train candidate parameters (`\theta`{=tex}*{n+1}\^{\*})
asynchronously while (`\theta`{=tex}\_n) remains the inference
authority.

Training evidence may include:

-   observed transitions and action effects;
-   later Hydra consolidation outcomes;
-   matched causal transfer trials;
-   grounding controls;
-   successful and failed deliberation traces;
-   replay outcomes;
-   structure-preserving transformations and hard negatives.

A candidate model is published only after declared evaluation gates are
met:

\[ `\theta`{=tex}*{n+1} = Publish(`\theta`{=tex}*{n+1}\^{\*}) \]

subject to improvement and historical-retention criteria.

This creates two interacting forms of continual development:

\[ Memory\_{t+1} = Develop(Memory_t, Experience_t) \]

\[ `\theta`{=tex}\_{n+1} = Learn(`\theta`{=tex}*n, Evidence*{0:t}) \]

The memory substrate can continue to grow, revise, suspend, promote,
demote, and forget independently of the fixed parameter count of a
configured learned reasoner.

## **17.24.6 Scientific Separation of Memory and Model Evidence**

Every scientific report involving learned reasoning should distinguish:

-   explicit-memory evidence;
-   learned-model estimates;
-   deliberation-derived candidate changes;
-   subsequent environmental outcomes;
-   causal validation results.

This separation is required to determine whether an observed gain came
from better memory formation, better retrieval, better learned
inference, better deliberation, or target-local execution.

# **18. Predictions and Falsification Analysis**

A scientific theory must expose its principal claims to tests that can
fail. This section states the main developmental predictions and
specifies quantitative comparison criteria wherever practical.

The thresholds used in a particular experiment should be fixed before
evaluation or estimated on a development split and then held constant on
the final test split.

# **18.1 Developmental Ordering Predictions**

### **P1 --- Contingency Before Concept**

Stable local action-conditioned contingencies should emerge before
validated transferable concepts.

### **P2 --- Transformation Family Before Carrier**

Reusable transformation-family structure should emerge before persistent
carrier hypotheses become predictively useful.

### **P3 --- Carrier Before Functional Role**

Persistent carrier hypotheses should precede stable cross-context
functional-role abstractions.

### **P4 --- Role Before Validated Concept**

Cross-context or cross-environment role reuse should precede validated
concept status.

### **P5 --- Future-Option Motifs Before Planning**

Recurring bounded future-option motifs should be measurable before
planning based on those motifs becomes behaviorally effective.

### **P6 --- World Models Are Late**

Integrated world-model components should arise after lower-level
contingencies, roles, and concept candidates rather than being required
to seed them.

A systematic reversal of these dependencies challenges the proposed
developmental hierarchy.

# **18.2 Structural-Prior Prediction**

### **P7**

The theory predicts that task-semantic abstractions can emerge while
using only explicitly declared observation-level structural priors.

The theory does **not** predict invariance to removal of all spatial or
sensor topology.

For each experiment, the observation-structure contract
(`\mathcal{R}`{=tex}\_O) must therefore be reported. If semantic labels
or object identities must be inserted into (D_O) to obtain the claimed
developmental sequence, the interaction-first ontology is weakened.

# **18.3 Prediction-Violation Predictions**

### **P8**

Once supported contingencies exist, prediction-error events should
receive more replay or contextual analysis than matched low-error
events.

A simple intervention compares:

\[ Lift\_{PE} =
`\frac{ P(Resource\ Allocation\mid PE\ge q_{high}) }{ P(Resource\ Allocation\mid PE\le q_{low}) }`{=tex}.
\]

The prediction is (Lift\_{PE}\>1) after expectation activation.

### **P9**

Before expectation activation, prediction-error evidence should be
absent rather than spuriously maximal.

# **18.4 Explanatory Reach Prediction**

### **P10**

Explanatory reach should predict candidate promotion better than raw
frequency.

Let (Y\_{promote}) be a future candidate-promotion indicator. Compare
held-out predictive models:

\[ Perf(Y\_{promote}`\mid`{=tex}ER)
`\quad`{=tex}`\text{and}`{=tex}`\quad`{=tex}
Perf(Y\_{promote}`\mid`{=tex}Frequency). \]

The prediction is:

\[ Perf(Y\_{promote}`\mid`{=tex}ER) \>
Perf(Y\_{promote}`\mid`{=tex}Frequency). \]

The choice of predictive score---cross-validated log loss, AUROC, rank
correlation, or another predeclared statistic---must be fixed in the
experimental protocol.

# **18.5 Role Versus Appearance Transfer**

### **P11**

Functional-role similarity should predict held-out transfer better than
perceptual similarity.

Let (T) be empirical held-out transfer success. Then the primary
comparison is:

\[ Perf(T`\mid`{=tex}RoleSimilarity) \>
Perf(T`\mid`{=tex}AppearanceSimilarity). \]

A stronger multivariate test asks whether role similarity retains
incremental predictive value after conditioning on appearance:

\[ I(T;RoleSimilarity`\mid`{=tex}AppearanceSimilarity)\>0 \]

or an equivalent regression/ablation criterion.

### **P12**

Graph-neighborhood correspondence should predict transfer beyond
immediate local appearance.

### **P13**

Future-option structural similarity should add predictive information
beyond appearance when the relevant role changes reachable structure.

If appearance consistently matches or exceeds relational predictors
under controlled held-out tests, the role-centric account is weakened.

# **18.6 Concept-Validation Predictions**

### **P14**

Structural recurrence alone should not be sufficient for validated
concept status.

### **P15**

A concept candidate with an admissible approximate typed mapping into an
unseen environment should still fail validation when its matched
intervention does not improve the declared transfer metric.

### **P16**

Validated concepts should show positive empirical effect under held-out
memory-on versus memory-off comparison more often than structurally
matched but unvalidated candidates.

# **18.7 Context-Refinement Predictions**

### **P17**

For recurrent contradictions caused by hidden conditional structure,
evidence-driven context refinement should improve held-out prediction
relative to immediate global concept replacement.

Define:

\[ Gain\_{context} = Error\_{replacement} - Error\_{refinement}. \]

The prediction is positive mean (Gain\_{context}) on contradiction sets
where an admissible separating context exists.

### **P18**

When no supported contextual partition exists, refinement should not be
forced. The theory therefore predicts selective context expansion rather
than universal splitting.

# **18.8 Future-Option Predictions**

### **P19**

Bounded learned future-option change should predict allocation of
developmental resources beyond prediction error alone.

One comparison is:

\[
Perf(Attention`\mid`{=tex}PE,`\Delta`{=tex}`\widehat{FO}`{=tex}\^{,k})
\> Perf(Attention`\mid`{=tex}PE). \]

### **P20**

The online estimate (`\Delta`{=tex}`\widehat{FO}`{=tex}\^{,k}) should
increasingly agree with oracle or held-out probe estimates as the
interaction graph matures.

For environments with computable oracle reachability:

\[ FOD(t) = Agreement( `\Delta`{=tex}`\widehat{FO}`{=tex}*t\^{,k},
`\Delta`{=tex}FO*\*\^{,k} ). \]

The prediction is increasing (FOD(t)) after contingency formation.

### **P21**

No intrinsic preference for positive
(`\Delta`{=tex}`\widehat{FO}`{=tex}) is assumed. Directional preference
should therefore be demonstrable only when supported by later learned
consequences or explicit outcome-comparison evidence.

This prediction distinguishes structural significance from hidden
utility maximization.

# **18.9 Outcome-Equivalence, Goal-Like Structure, and Replanning Predictions**

### **P22**

Distinct states or trajectory endpoints with sufficiently similar
learned consequence descriptors should form persistent
outcome-equivalence abstractions before efficient strategy ranking
becomes stable.

### **P23**

When an environment affords multiple routes to the same learned outcome
class, the system should be able to retain or discover more than one
supported strategy-to-outcome relation.

### **P24**

After ablating or invalidating a preferred strategy, replanning should
select an alternative strategy linked to the same selected outcome
abstraction more often than a control system that stores successful
trajectories without a separate outcome representation.

### **P25**

Outcome equivalence alone should not imply directional preference.
Stable target-like preference should appear only after additional
learned evidence and should be measurable as a separate developmental
phenomenon.

# **18.10 Efficiency Predictions**

### **P26**

Trajectory efficiency should become measurable only after learned
outcome equivalence or another admissible outcome-comparison relation
has been established.

### **P27**

Within outcome-equivalent classes, lower interaction cost should predict
stronger strategy retention or reuse.

### **P28**

Cost-only shortest-path pressure should not dominate when shorter
trajectories produce worse declared outcome vectors.

### **P29**

Efficiency pressure should reduce repeated-state, loop, blocked-action,
and unnecessary-action ratios without suppressing early exploratory
coverage.

# **18.11 Conditional Memory-Compression Prediction**

### **P30**

In environments with recurrent compressible structure and bounded rates
of irreducible novelty, persistent consolidated memory should grow
sublinearly with cumulative experience while predictive, explanatory,
transfer, and planning quality are preserved.

Let:

\[ MGR(t) = `\frac{\Delta M(t)}{\Delta E(t)}`{=tex}. \]

The prediction is that (MGR(t)) declines with developmental maturity
under the stated conditions.

A stronger long-horizon signature is:

\[ `\frac{M(E)}{E}`{=tex}`\rightarrow0`{=tex}. \]

This is a conditional empirical prediction, not a general theorem.
Linear growth in a deliberately non-recurrent stream of irreducibly
novel experience does not falsify it.

# **18.12 Concurrent Shared-Memory Architecture Predictions**

### **P31**

Under versioned publication, bounded search, evidence contracts,
hysteresis, and provenance-preserving pruning, concurrent shared-memory
update processes should exhibit bounded representational churn rather
than persistent destructive oscillation.

Operational measures may include:

-   candidate creation/reversal rate,
-   repeated promotion-demotion cycles,
-   graph rewrite rate per unit evidence,
-   proportion of retired structures later recreated with equivalent
    identity,
-   memory growth under stable evidence.

### **P32**

Removing the shared consistency constraints while keeping the same
update processes should increase destructive churn or reduce
transfer/predictive stability.

The theory does not predict mathematical convergence to a fixed
representation.

# **18.13 Grounded Symbolic Meaning Hypothesis**

### **H16 --- Cross-Modal Grounded Meaning**

Initially semantic-free symbolic observations will acquire functional
meaning when their memories repeatedly participate in higher-order
structures that are also independently supported by grounded interaction
evidence.

H16 requires **bidirectional transfer**. Evidence is stronger when:

-   aligned symbolic experience improves held-out interaction prediction
    or action selection;
-   interaction-derived functional structure improves interpretation or
    reuse of held-out symbolic combinations;
-   learned cross-modal structure generalizes to combinations not
    observed during training; and
-   the effect survives removal of the symbolic stream during an
    interaction-only evaluation phase.

H16 is not supported by improved symbol prediction alone.

### **Required Controls**

At minimum, experiments must compare:

-   **C0 --- Interaction only:** environmental observations, actions,
    and consequences;
-   **C1 --- Symbols only:** the symbolic stream without aligned
    interaction evidence;
-   **C2 --- Aligned interaction + symbols:** both streams with their
    true temporal and structural relationships;
-   **C3 --- Shuffled interaction + symbols:** identical marginal symbol
    exposure with cross-modal alignment destroyed.

The central prediction is not merely (C2\>C0). C2 must also outperform
C1 and C3 on cross-modal transfer measures, demonstrating that the
benefit depends on grounded alignment rather than additional
observations or symbol frequency.

### **Developmental Evaluation**

A staged protocol is proposed:

1.  **G0 --- Interaction grounding:** learn action--consequence
    regularities without symbols.
2.  **G1 --- Concurrent symbols:** expose ordered symbols aligned with
    experienced interactions.
3.  **G2 --- Descriptive symbols:** test whether symbolic structures
    become linked to interaction-derived families and roles.
4.  **G3 --- Prospective symbols:** present symbolic evidence before
    interaction and test consequence prediction.
5.  **G4 --- Novel composition:** test unseen combinations of grounded
    symbolic and interaction structures.
6.  **G5 --- Symbol-mediated learning:** test whether symbolic evidence
    supports correct predictions about interactions not directly
    experienced during training.

Progression is empirical rather than assumed. Failure at an earlier
stage precludes claims about later symbolic competence.

# **H17 --- Developmental Stability Without a Global Behavioral Objective**

Independent asynchronous developmental processes governed by provenance,
hysteresis, evidence thresholds, dependency constraints, bounded
mutation, and causal admissibility can reach empirically stable but
plastic memory regimes without a single global behavioral objective.

Define a reversal rate:

\[ R\_{`\mathrm{rev}`{=tex}} =
`\frac{N_{\mathrm{promotion\rightarrow demotion}}       +N_{\mathrm{demotion\rightarrow reformation}}}`{=tex}
{N\_{`\mathrm{memory\ state\ transitions}`{=tex}}}. \]

Define memory churn:

\[ R\_{`\mathrm{churn}`{=tex}} =
`\frac{N_{\mathrm{created}}+N_{\mathrm{retired}}}`{=tex}
{`\max`{=tex}(1,N\_{`\mathrm{active}`{=tex}})}. \]

For memory (X), define structural persistence over interval
(`\Delta`{=tex}t):

\[ P\_{`\Delta`{=tex}t}(X) =
Pr(X `\text{remains admissible over}`{=tex} `\Delta`{=tex}t). \]

A stable developmental regime requires neither zero churn nor permanent
memory. It requires bounded reversal/churn together with continued
formation of useful novel structure and non-degrading
predictive/transfer capability.

H17 is tested by systematically varying promotion/demotion hysteresis,
retention thresholds, peer update rates, and evidence requirements.

Evidence against H17 occurs when no parameter region supports both
structural stability and continued developmental plasticity, or when
stability requires an undeclared centralized objective.

# **H18 --- Structural Prior Dependence**

The abstractions that emerge depend systematically on the structural
relations available in (`\mathcal{R}`{=tex}\_O), while interaction may
compensate for some, but not necessarily all, removed structural priors.

H18 predicts measurable differences in:

-   time to stable contingency formation;
-   compression;
-   carrier/role formation;
-   concept formation;
-   held-out transfer;
-   planning;
-   cross-modal grounding;

as structural priors are removed or scrambled.

The hypothesis is not that all priors are reconstructible.

The scientific objective is to estimate:

\[ Capability = F(`\mathcal{R}`{=tex}\_O, Experience, Computation) \]

and identify which elements of (`\mathcal{R}`{=tex}\_O) are necessary,
replaceable by experience, or primarily sample-efficiency aids.

For spatial environments, a particularly strong test compares ordinary
topology against fixed-permutation, topology-withheld, and other
structure-reduced observation conditions while preserving underlying
environment dynamics.

# **18.14 Learned Relational Reasoning Hypothesis**

### **H19 --- Learned Relational Reasoning Over Emergent Memory**

After sufficient grounded developmental structure exists, a learned
relational operator trained over bounded M0-M7 memory subgraphs will
improve one or more of:

-   relevant-memory retrieval;
-   structural correspondence;
-   consequence prediction;
-   candidate refinement;
-   strategy selection;
-   cross-family transfer;
-   sample efficiency;

relative to an otherwise matched explicit-memory-only system.

H19 additionally predicts that recursive learned deliberation can
outperform a single learned relational pass when tasks require
multi-step refinement, while incurring measurable additional reasoning
cost.

H19 is supported only if gains survive held-out evaluation and cannot be
explained solely by additional environment interaction, privileged
semantic labels, future-information leakage, or unmatched starting
states.

Required primary ablations are:

1.  **Hydra only** --- explicit developmental memory and existing
    explicit reasoning mechanisms;
2.  **Hydra + learned relational single-pass** --- one bounded learned
    reasoning pass;
3.  **Hydra + learned relational recursive deliberation** --- repeated
    bounded reasoning/refinement cycles.

Additional controls should include:

-   random/untrained relational representations;
-   frozen early learned model;
-   continually trained current model;
-   explicit similarity versus learned similarity;
-   model-only learned policy where experimentally feasible.

Measure at minimum:

-   sample efficiency;
-   prediction quality;
-   held-out task success;
-   cross-family transfer;
-   memory growth;
-   reasoning cost;
-   historical-stage retention/catastrophic forgetting;
-   trajectory efficiency.

### **H19 Developmental Constraint**

H19 is rejected as an explanation of developmental emergence if the
learned reasoner obtains its gains by directly creating or validating
high-level semantics that bypass the M0-M7 formation and
causal-validation pathway.

A positive behavioral result under such a bypass may demonstrate a
useful hybrid engineering architecture, but it would not support the
developmental claim of this paper.

### **H19 Continual-Learning Prediction**

Continual relational training should improve current-stage reasoning
while retaining useful relational competence from earlier environment
families better when combined with explicit Hydra memory and stratified
historical replay than when trained only on the newest curriculum stage.

A candidate successor model should not be treated as developmental
progress if new-stage gains are accompanied by unacceptable loss on
declared historical retention tests.

# **18.15 Strong Comparative Predictions**

The following comparisons are especially diagnostic.

### **SP1 --- Transformation Primacy**

Transformation-family evidence appears before stable carrier identity.

### **SP2 --- Relational Transfer**

\[ Perf(T`\mid`{=tex}Role/Graph) \> Perf(T`\mid`{=tex}Appearance). \]

### **SP3 --- Explanatory Promotion**

\[ Perf(Promotion`\mid`{=tex}ER) \>
Perf(Promotion`\mid`{=tex}Frequency). \]

### **SP4 --- Future-Option Contribution**

\[ Perf(ResourceAllocation`\mid`{=tex}PE,`\Delta`{=tex}FO) \>
Perf(ResourceAllocation`\mid`{=tex}PE). \]

### **SP5 --- Empirical Concept Validation**

Structurally plausible candidates that fail held-out interventions
remain unvalidated.

### **SP6 --- Outcome/Strategy Separation**

A persistent learned outcome representation should support
alternative-strategy substitution and replanning better than storing
successful trajectories without an explicit outcome-equivalence
abstraction.

### **SP7 --- Late World-Model Organization**

World-model components improve after concepts and relational structures
exist rather than serving as necessary primitives for their formation.

### **SP8 --- Conditional Developmental Compression**

Persistent memory becomes increasingly compact relative to accumulated
experience in recurrently compressible environments.

# **18.16 Core Falsification Criteria**

The framework is substantially weakened by repeated, well-powered
observations of the following.

## **F1 --- Semantic-Prior Necessity**

The predicted hierarchy requires predefined object identity, semantic
categories, task goals, or reward meaning in the transformation
representation.

## **F2 --- Object-First Emergence**

Stable object representations consistently become predictively or
transfer-useful before transformation families and relational roles.

## **F3 --- Appearance-Dominated Transfer**

Appearance similarity consistently predicts held-out transfer at least
as well as role, graph, and future-option similarity after controlling
for sample size and evaluation opportunity.

## **F4 --- World-Model Necessity**

An integrated world model must be learned before stable contingencies,
roles, or concept candidates can emerge.

## **F5 --- Failure of Prediction-Violation Allocation**

Once expectations exist, prediction error does not improve allocation of
replay, context search, or corrective processing relative to matched
non-violating events.

## **F6 --- Failure of Future-Option Contribution**

Bounded future-option structure adds no reproducible predictive or
causal value for attention, retention, transfer, or planning beyond
simpler baselines.

## **F7 --- Failure of Explanatory Reach**

Frequency predicts promotion at least as well as explanatory reach
across controlled held-out comparisons.

## **F8 --- Failure of Context Refinement**

When an admissible hidden contextual partition exists, immediate concept
replacement consistently outperforms context refinement on held-out
prediction and transfer.

## **F9 --- Failure of Empirical Transfer Validation**

Structures declared to be validated concepts do not produce reproducible
held-out causal reuse effects, or structural recurrence alone performs
equivalently to the proposed validation process.

## **F10 --- Failure of Developmental Ordering**

The predicted dependency sequence is systematically reversed---for
example, transferable concepts appear before the contingencies or roles
from which the theory claims they are derived.

## **F11 --- Failure of Outcome/Strategy Separation**

A separate learned outcome-equivalence abstraction does not improve
outcome recognition, alternative-strategy reuse, or replanning over
otherwise matched systems that store trajectories without persistent
outcome identity, or learned outcome classes fail to remain stable under
held-out consequence comparisons.

## **F12 --- Failure of Emergent Target-Like Structure**

Goal-like behavior requires an externally supplied goal or reward
representation, and no stable learned preference relation over
outcome-equivalence classes emerges from interaction-derived evidence.

## **F13 --- Failure of Efficiency Emergence**

After learned comparable outcomes exist, lower-cost trajectories do not
receive greater reuse or retention, or indiscriminate shortest-path
pressure performs better without sacrificing declared future-option or
outcome quality.

## **F14 --- Failure of Conditional Compression**

In recurrently compressible environments with bounded novelty,
persistent memory remains approximately linear in experience despite
mature abstractions, or compression gains require sacrificing
predictive, explanatory, transfer, or planning quality.

## **F15 --- Persistent Architectural Thrashing**

The constrained concurrent-update architecture exhibits repeated
destructive oscillation that is not reduced by the proposed consistency,
hysteresis, and publication constraints.

## **F16 --- Failure of Grounded Symbolic Emergence**

The cross-modal extension is falsified if symbolic observations fail to
acquire interaction-relevant functional significance under environments
containing recurrent, learnable symbol--interaction structure.

Strong negative evidence includes any of the following:

-   aligned symbols provide no reproducible advantage over
    shuffled-symbol controls;
-   apparent language competence is explained by within-symbol sequence
    statistics without cross-modal transfer;
-   successful transfer requires manually supplied semantic mappings,
    pretrained embeddings, or pretrained language representations;
-   grounded symbols cannot alter predictions or behavior in held-out
    interaction configurations; or
-   interaction-derived abstractions do not constrain interpretation of
    novel symbolic combinations.

This falsifies the Version 0.6.0 grounding extension without, by itself,
falsifying the interaction-only developmental theory.

## **F17 --- Failure of Developmental Stability**

H17 is rejected if concurrent local developmental processes exhibit
persistent oscillation, destructive churn, or collapse across the
admissible parameter range, or if stable operation requires an
undeclared global behavioral objective.

## **F18 --- Failure or Mischaracterization of Structural-Prior Dependence**

H18 is rejected or revised if structural-prior ablations do not produce
the predicted systematic capability differences, or if claimed emergent
abstractions are shown to depend on undeclared domain-semantic structure
at the observation boundary.

H19 is rejected or revised if a causally controlled, held-out comparison
shows no reproducible reasoning or behavioral advantage from the learned
relational operator over Hydra-only reasoning across the declared
evaluation regime; if recursive deliberation provides no advantage over
a matched single pass where multi-step refinement is required; or if
apparent gains depend on semantic shortcuts, future-information leakage,
or bypassing developmental validation.

# **18.17 Levels of Negative Evidence**

### **Local Failure**

Challenges an operational mechanism or estimator while leaving the
interaction-first thesis intact.

Examples:

-   a particular ISF aggregation,
-   a promotion threshold,
-   a future-option weighting scheme.

### **Structural Failure**

Challenges a major subsystem.

Examples:

-   relational transfer,
-   context refinement,
-   future-option estimation,
-   shared-memory concurrent updates.

### **Foundational Failure**

Challenges the central developmental ontology.

Examples:

-   semantic object priors are required,
-   object-first emergence dominates,
-   transferable abstractions do not arise from interaction-derived
    relational structure.

# **18.18 Interpretation of Negative Results**

Negative results should be classified by the claim actually tested. The
framework should not treat every failed implementation as a
falsification of the ontology, nor should it protect the ontology by
reclassifying every negative result as an implementation failure.

Experiments should therefore preregister:

-   the target theoretical claim,
-   the operational estimator,
-   required evidence,
-   alternative explanations,
-   the criterion for local, structural, or foundational failure.

# **18.19 Central Falsifiable Proposition**

The most important proposition is:

> Under explicitly declared non-semantic structural priors, transferable
> abstractions can emerge from remembered action-conditioned
> transformations through contingency learning, relational compression,
> context refinement, and held-out transfer validation before stable
> object-centric representations become the primary organizing
> structure.

The empirical program succeeds only if this proposition survives direct
comparison with object-first, appearance-based, world-model-first, and
simpler memory baselines.

# **19. Relationship to Existing Theories and Prior Work**

The purpose of this section is not to argue that existing approaches are
incorrect. Rather, it is to clarify how the present theory differs in
its assumptions, developmental ordering, and explanatory focus. Many
ideas within this framework overlap partially with existing work. The
primary contribution lies in their integration into a unified
interaction-first, memory-centric developmental theory.

# **19.1 Reinforcement Learning**

Classical Reinforcement Learning (RL) defines learning as optimization
of expected cumulative reward.

The central quantities are:

-   states,\
-   actions,\
-   rewards,\
-   policies,\
-   value functions.

Learning occurs through improvement of action selection.

## **Agreement**

The present theory agrees that experience matters. It also agrees that
future consequences influence learning.

## **Difference**

The present theory distinguishes minimal **primary valence** from an
engineered **task reward function**. Signed primary valence is
fundamental motivational evidence, but it does not provide task-semantic
states, goals, concepts, policies, or an externally specified value
function. The learner must still discover:

-   contingencies,\
-   future possibilities,\
-   functional roles,\
-   consequence and outcome structure,\
-   strategies and stable preferences.

Future-option structure and primary valence are separate: future-option
change is structural evidence, while valence supplies motivational
direction.

## **Central Distinction**

RL: Engineered task reward → value/policy optimization

This theory: Interaction + primary valence → developmental memory →
learned outcomes/strategies/preferences

# **19.2 Predictive Processing**

Predictive Processing proposes that cognition is organized around
prediction and prediction-error minimization. Prediction errors drive
updating of internal models.

## **Agreement**

The present theory agrees that prediction violations are important.
Prediction error is a major component of the Interaction Significance
Function. Prediction violations drive attention and memory formation.

## **Difference**

Prediction is not considered primary. Prediction emerges after
contingency formation. The developmental ordering is:

Interaction → Contingency → Prediction

rather than:

Prediction → Intelligence

Prediction serves memory organization rather than defining intelligence
itself.

## **Central Distinction**

Predictive Processing: Prediction is fundamental.

This theory: Remembered interaction is fundamental.

Prediction is emergent.

# **19.3 Active Inference**

Active Inference extends predictive processing by framing behavior as
minimization of expected surprise and uncertainty. Agents actively seek
observations that improve their internal models.

## **Agreement**

The present theory agrees that:

-   uncertainty reduction matters,\
-   exploration matters,\
-   prediction violations matter.

## **Difference**

The primary developmental pressure is not uncertainty reduction. The
primary developmental pressure is future-option discovery and memory
restructuring. An interaction may be important even when it does not
substantially reduce uncertainty if it:

-   expands future possibilities,\
-   reveals transferable roles,\
-   increases explanatory reach.

## **Central Distinction**

Active Inference: Reduce uncertainty.

This theory: Discover, preserve, expand, and reorganize future
possibilities.

# **19.4 World-Model Approaches**

Many contemporary systems attempt to learn explicit world models early
in development. Prediction, planning, and simulation are often treated
as foundational capabilities.

## **Agreement**

The present theory accepts that world models eventually emerge. World
models can become powerful explanatory structures.

## **Difference**

The theory rejects world-model primacy. World models are late-stage
products of development. The proposed sequence is:

Experience → Memory → Roles → Concepts → World Models rather than:

World Models → Concepts → Behavior

## **Central Distinction**

World-model approaches: World models enable intelligence.

This theory: Intelligence produces world models.

# **19.5 Symbolic AI**

Symbolic AI begins with explicit symbols and rules. Reasoning operates
through manipulation of predefined structures.

## **Agreement**

The theory agrees that high-level abstractions eventually emerge.
Concepts can become increasingly symbolic.

## **Difference**

No symbols are assumed initially. Symbols must emerge from interaction
history. The theory is therefore developmental rather than symbolic from
the outset.

## **Central Distinction**

Symbolic AI: Symbols are primitive.

This theory: Symbols are emergent.

# **19.6 Representation Learning**

Modern representation learning focuses on discovering latent structures
that support downstream tasks. Representations often emerge through
compression objectives.

## **Agreement**

The present theory strongly agrees that compression is important.
Abstraction depends on compression.

## **Difference**

Compression alone is insufficient. The theory proposes that successful
abstraction requires:

-   transfer,\
-   explanatory reach,\
-   future-option relevance.

A compressed structure that never transfers is not considered a concept.

## **Central Distinction**

Representation Learning: Compression creates representations.

This theory: Transferable explanatory compression creates concepts.

# **19.7 Foundation Models and Large-Scale Pretraining**

Large language models and foundation models derive capabilities from
exposure to massive datasets. Many abstractions emerge through
statistical learning.

## **Agreement**

The theory agrees that abstraction can emerge from large amounts of
experience. It also agrees that many concepts are learned rather than
predefined.

## **Difference**

The present theory focuses on embodied interaction rather than passive
observation. The primitive unit is:

Observation → Action → Transformation

rather than:

Observation alone.

The framework therefore emphasizes contingency discovery and
intervention.

## **Central Distinction**

Foundation Models: Learn from observation.

This theory: Learn from interaction.

# **19.8 Developmental Psychology**

Several aspects of the theory are broadly compatible with developmental
perspectives in psychology.

In particular:

-   abstraction emerges gradually,\
-   understanding develops in stages,\
-   context becomes increasingly important,\
-   concepts evolve through experience.

------------------------------------------------------------------------

## **Difference**

The present theory attempts to formalize these developmental processes
using:

-   interaction significance,\
-   future-option structure,\
-   explanatory reach,\
-   memory competition,\
-   graph organization.

------------------------------------------------------------------------

# **19.9 Symbol Grounding and Grounded Language Learning**

The present extension is related to symbol-grounding and
grounded-language research in treating linguistic or symbolic competence
as dependent on relationships beyond symbol sequences.

## **Agreement**

Meaning should be evaluated through relationships between symbols,
perception, action, and consequences rather than through symbol
statistics alone.

## **Difference**

The present theory does not introduce a dedicated semantic
representation space. Symbolic observations enter the same developmental
memory substrate as other observations. Their meaning is an empirical
property of participation in transferable higher-order interaction
structure.

The theory therefore predicts that semantic grounding can emerge from
ordinary memory promotion, restructuring, explanatory reach, and
transfer mechanisms, rather than requiring a separately specified
language-learning objective.

# **19.10 Graph-Based Cognitive Theories**

A number of cognitive theories represent knowledge as networks, graphs,
or relational structures. The present framework is most closely aligned
with this family of approaches.

## **Agreement**

Knowledge is relational. Meaning emerges from structure. Graph
organization is important.

## **Difference**

The present theory introduces a specific developmental mechanism:

Interaction → Graph → Role → Concept

along with:

-   Interaction Significance Function,\
-   Future-Option Theory,\
-   Promotion Through Explanatory Reach,\
-   Role-Before-Object Emergence.

These mechanisms provide a proposed path by which graph structure itself
emerges.

# **19.11 Unique Contributions of the Present Theory**

The framework is characterized by several claims that are unusual when
taken together.

### **Interaction-First Ontology**

Knowledge begins with remembered interactions rather than objects.

### **Memory-Centric Intelligence**

Memory organization is primary.

Concepts and persistent world knowledge emerge through developmental
memory organization. Prediction and planning may use both explicit
memory structure and a learned relational reasoner operating over that
structure.

### **Learned Reasoning Over Explicit Developmental Memory**

The theory distinguishes persistent knowledge formation from learned
relational inference. A trainable reasoner may improve retrieval,
correspondence, prediction, strategy ranking, and recursive deliberation
while explicit developmental memory retains authority for grounding,
provenance, validation, and persistent knowledge.

### **Transformation Primacy**

Transformations are learned before entities.

### **Role-Before-Object Emergence**

Transferable roles emerge before object categories.

### **Future-Option Structure**

Changes in bounded discovered reachability are investigated as a central
structural signal for developmental resource allocation.

### **Emergent Outcome and Goal-Like Structure**

Outcome identity is not assumed as a primitive goal representation. The
learner is predicted to form persistent outcome-equivalence abstractions
from recurring consequence structure, link multiple strategies to the
same outcome, and only later exhibit stable target-like preferences over
such outcome classes. This separates what is reached from how it is
reached and provides a developmental basis for replanning.

### **Emergent Efficiency**

Efficiency is not treated as a primitive optimization objective. It
emerges after the agent discovers learned outcome-equivalence classes or
otherwise admissible comparable trajectories and begins favoring
lower-cost paths that preserve the same outcome or an explicitly
no-worse future-option structure.

### **Promotion Through Explanatory Reach**

Abstractions emerge because they explain lower-level structures.

### **Concurrent Shared-Memory Update Architecture**

Coherence is hypothesized to emerge through a shared memory substrate
plus common evidence and consistency constraints without requiring a
single global behavioral objective.

### **Developmental Intelligence**

Intelligence is continuous restructuring rather than convergence.

### **Grounded Symbolic Emergence**

Initially semantic-free symbols are predicted to acquire functional
meaning through integration with interaction-grounded higher-order
memory. Grounding is evaluated by bidirectional cross-modal transfer
rather than linguistic similarity or symbol prediction alone.

# **19.12 Positioning Statement**

The theory should not be viewed as a replacement for reinforcement
learning, predictive processing, world models, representation learning,
or foundation models. Instead, it proposes a different developmental
foundation beneath them. The central claim is that remembered
interaction plus minimal primary valence, memory organization,
future-option discovery, role emergence, and explanatory compression may
constitute a more primitive layer from which many familiar cognitive
capabilities eventually arise.

# **19.13 Summary**

Existing approaches often organize intelligence around:

-   engineered task reward,\
-   prediction,\
-   symbols,\
-   world models,\
-   representations.

The present theory organizes intelligence around:

-   remembered interactions,\
-   minimal signed primary valence,\
-   significance-driven memory,\
-   future-option discovery,\
-   graph formation,\
-   role emergence,\
-   explanatory promotion.

The theory therefore shifts the focus of intelligence from optimization
and representation toward developmental memory organization.

# **20. Conclusion and Future Research Agenda**

## **Version 0.7.0 Research Extension**

The next empirical phase must test the boundary between **memory
formation** and **learned reasoning over memory**. Hydra v9.7.2 provides
the reference implementation: explicit M0-M7 developmental memory
supplies the grounded, auditable substrate; an HGT-style learned
relational operator performs bounded inference over retrieved subgraphs;
recursive deliberation refines candidates; and environmental
consequences return causal evidence to memory and future training.

The decisive question is whether this hybrid produces reproducible gains
over Hydra-only reasoning without weakening the developmental claims
that make those gains scientifically interpretable.

## **Conclusion**

This paper proposed an interaction-first, memory-centric theory of
emergent intelligence. The central claim is that intelligence does not
begin with predefined task-semantic objects, goals, concepts, symbols,
policies, or world models. Intelligence begins with remembered
interaction plus minimal signed primary valence. The primitive unit of
knowledge is an action-conditioned transformation carrying optional
signed primary-valence evidence:

Observation(t) → Action(t) → \[O(t,1), ..., O(t,m_t)\] →
Observation(t+1), with V(t) ∈ {-1,0,+1}

From repeated interactions, the agent discovers contingencies. From
contingencies emerge prediction violations, transformation families,
carriers, functional roles, concepts, consequence structures,
outcome-equivalence abstractions, alternative strategies, and eventually
increasingly integrated world models and replanning structures. The
theory proposes that memory is not a supporting component of
intelligence. Memory is the primary substrate of intelligence. Learning,
prediction, planning, transfer, abstraction, and concept formation
emerge through continuous restructuring of memory.

## **The Developmental Pathway**

The proposed developmental sequence is:

Observations → Transformations → Contingencies → Prediction Violations →
Transformation Families → Carriers → Functional Roles → Concepts →
Consequence / World-Model Structure → Outcome-Equivalence Abstractions →
Alternative Strategies → Replanning and Efficiency

Each stage emerges because it compresses, explains, transfers, or
organizes structures beneath it. No stage is assumed in advance. All
higher-level structures are earned through experience.

## **Interaction Significance**

A central contribution of the theory is the Interaction Significance
Function. The theory proposes that experiences are not treated equally.
Interactions compete for:

-   attention,\
-   memory,\
-   replay,\
-   promotion.

Significance emerges from multiple developmental pressures including:

-   option-structure impact,\
-   prediction error,\
-   learning value,\
-   transfer potential,\
-   explanatory potential.

The Interaction Significance Function provides the mechanism through
which experience becomes structured memory.

## **Future-Option Discovery**

The theory further proposes bounded future-option structure as a central
structural quantity for empirical investigation. Large changes in
discovered reachability are predicted to influence developmental
resource allocation. The agent initially discovers:

-   preservation,\
-   restriction,\
-   expansion,\
-   collapse

of future-option structure.

Only later does it learn to predict these changes. Future-option
discovery therefore precedes planning and may precede sophisticated
prediction. A further late abstraction groups distinct future states or
trajectory endpoints into persistent outcome-equivalence classes based
on learned consequence structure. Strategies are then linked to, but
remain distinct from, those outcome classes. This separation allows
replanning when one strategy fails. Stable target-like preference, when
it emerges, is a learned relation over outcome classes rather than a
primitive goal. Efficiency emerges still later and compares cost only
within the same learned outcome class or an explicitly no-worse
comparison class.

## **Transformation Before Object**

The theory rejects object-first learning. The agent first discovers
transformations. Objects emerge later as explanatory compressions of
recurring transformation patterns. The proposed developmental sequence
is:

Transformation → Carrier → Role → Concept

rather than:

Object → Concept

Transferable functional roles are predicted to emerge before
transferable object categories.

## **Relational Functional Identity and Context**

Functional interpretation is proposed to depend strongly on learned
graph structure rather than intrinsic appearance alone. Context is not
externally supplied. Context emerges from the neighborhood structure of
remembered interactions. Learning therefore proceeds primarily through
context expansion rather than concept replacement. Contradictions become
opportunities for discovering missing contextual structure.

## **Explanatory Reach**

The theory proposes that abstraction is driven primarily by explanatory
reach. Structures are promoted because they explain lower-level
structures. Frequency alone is insufficient. The developmental hierarchy
emerges through successive increases in explanatory reach and
compression.

## **Intelligence as Continuous Reorganization**

The framework does not require convergence to a fixed final
representation. It models intelligence as continuing reorganization of
memory under changing developmental evidence and resource pressure. New
experiences continually reshape:

-   contingencies,\
-   roles,\
-   concepts,\
-   future-option structures,\
-   outcome-equivalence classes,
-   strategy-to-outcome relations,
-   target-like preferences,
-   world models.

Learning therefore remains open-ended.

# **Future Research Agenda**

The present work establishes a theoretical foundation. Many components
remain to be refined, formalized, and experimentally validated. The most
important future directions are outlined below.

## **Formalization of Future-Option Structure**

Future-option theory currently relies on graph reachability
approximations.

Future work should develop richer measures including:

-   weighted reachability,\
-   uncertainty-aware reachability,\
-   developmental reachability,\
-   abstraction reachability,\
-   transfer reachability.

A rigorous mathematical treatment of future-option structure remains one
of the most important open problems.

## **Formalization of Explanatory Reach**

Explanatory reach is central to promotion and abstraction. Future work
should investigate:

-   information-theoretic formulations,\
-   compression-based formulations,\
-   causal formulations,\
-   graph-theoretic formulations.

The relationship between explanatory reach and concept formation
requires deeper analysis.

## **Developmental Weight Learning**

The current theory assumes developmental evolution of ISF weights.
Future work should determine:

-   how weights emerge,\
-   how weights adapt,\
-   whether developmental stages emerge naturally,\
-   whether stage transitions can be measured objectively.

## **Computational Architecture**

The theory is implementation-agnostic with respect to whether individual
mechanisms use symbolic, statistical, neural, or hybrid methods, but it
requires a minimum reference architecture so that its claims can be
implemented without introducing object or world-model primitives.

A reference execution path is:

Observation/Action Stream → Transformation Encoder D_O → M0 Episodic
Store → Recurrence and Contingency Discovery → Multi-Scale Memory Graph
→ Candidate Compression and Context Expansion → Role/Concept Candidate
Generation → Transfer Testing → Consequence Integration and
Outcome-Equivalence Formation → Strategy-to-Outcome Linking and
Target-Preference Estimation → Promotion, Replay, Demotion, and
Forgetting → Optional Late Planning, Replanning, and Efficiency
Selection

All concurrent update processes read and write through the shared memory
substrate. Prediction reads established contingencies rather than
seeding them: recurrence first establishes a sufficiently supported
local contingency, and only subsequent interactions can generate
prediction error from that expectation. Future-option estimation
operates over the currently discovered memory graph. Concept candidates
may exist before external validation, while validated transferable
concept status requires empirical unseen-environment transfer evidence.
Demotion uses hysteresis, probation, evidence-volume thresholds, and
context-confidence gating rather than single-window score drops.

To prevent combinatorial explosion, an implementation must constrain
graph construction and abstraction search. At minimum:

-   graph edges are created only for observed or evidence-supported
    relation types rather than all node pairs;
-   candidate comparisons are restricted by memory level, action
    compatibility, temporal/context locality, signatures, or indexed
    structural features;
-   motif and role discovery use bounded neighborhoods rather than
    unrestricted subgraph isomorphism over the full history;
-   promotion is staged so only a limited candidate frontier receives
    expensive transfer or explanatory testing;
-   low-value raw episodes may be compressed, summarized, or forgotten
    after their supported higher-level structures become stable;
-   exact global reachability is not required online; future-option
    estimates operate on bounded discovered graphs.

These constraints are part of the executable interpretation of the
theory, not claims that a particular database, clustering method, graph
engine, or neural architecture is required. Comparative implementations
may therefore test whether the developmental principles survive
different computational substrates.

## **Cross-Modal Grounding Experiments**

The first grounding experiments should use procedurally generated
environments with controlled symbolic instructions or descriptions and
discrete, verifiable actions. The objective is not linguistic breadth
but causal identifiability.

The experimental design should vary independently:

-   interaction diversity;
-   symbolic vocabulary size;
-   compositional depth;
-   temporal alignment quality;
-   proportion of symbolically described versus directly experienced
    events;
-   held-out combinations of symbols, roles, objects, actions, and
    outcomes; and
-   environmental novelty.

Primary measures should include cross-modal consequence prediction,
interaction success after symbolic instruction, transfer to held-out
combinations, interaction-only performance after multimodal training,
and memory-level evidence that symbolic and interaction provenance
converge on shared M2--M6 structures.

Synthetic or controlled language is appropriate for initial
falsification because it permits exact manipulation of alignment and
semantics. Richer natural-language environments should be introduced
only after cross-modal grounding is demonstrated under controlled
conditions.

## **Deep-Transfer Probe**

The progressive multi-scale correspondence claim requires a controlled
test in which functional equivalence is preserved while dependency depth
changes.

Construct at least two procedurally generated environments.

Environment A contains a short dependency:

\[ X `\rightarrow`{=tex}Outcome. \]

Environment B contains a functionally corresponding but deeper
dependency:

\[ X' `\rightarrow`{=tex}R_1 `\rightarrow`{=tex}R_2
`\rightarrow`{=tex}`\cdots`{=tex} `\rightarrow`{=tex}R_k
`\rightarrow`{=tex}Outcome'. \]

The underlying functional role and consequence relation are preserved
while the structural path length is manipulated.

Train the system until the relevant role structure is stable in
Environment A, then evaluate candidate discovery and causal transfer in
Environment B under different maximum structural radii.

Measure:

-   correspondence recall;
-   false-correspondence rate;
-   minimum radius required for discovery;
-   number of candidate expansions;
-   computational cost;
-   causal transfer success;
-   prediction improvement after transfer.

A successful result should demonstrate cases in which shallow search
fails, progressive expansion discovers the correspondence, and the
resulting correspondence survives held-out causal validation.

The experiment also tests computational boundedness. If required
comparison cost grows beyond the declared resource regime faster than
useful structural discrimination is obtained, the proposed multi-scale
mechanism is not supported.

## **Structural-Prior Permutation Series**

Structural-prior experiments must distinguish representation permutation
from actual removal of relational information.

For a spatial interactive environment, compare at least:

### **P0 --- Native topology**

Normal coordinates and declared topology are available.

### **P1 --- Fixed representation permutation with topology preserved**

Observation positions are permuted by a fixed mapping, but the
corresponding topological relations remain explicitly available.

This tests invariance to representational arrangement.

### **P2 --- Fixed representation permutation with topology withheld**

The same stable permutation is used, but adjacency/topological relations
are not supplied.

Persistent identity remains available across observations.

This tests whether interaction can reconstruct useful relational
organization from stable but non-spatial identities.

### **P3 --- Changing permutation with topology withheld**

Observation addresses are permuted between steps and explicit topology
is withheld.

Only the structural relations still declared by the experimental
condition remain available.

This is a substantially stronger information-removal condition and must
not be interpreted as equivalent to simple coordinate scrambling.

Measure:

-   contingency discovery rate;
-   time to first stable contingency;
-   transformation-family formation;
-   carrier/role formation;
-   concept validation;
-   outcome-equivalence formation;
-   held-out transfer;
-   interactions required per developmental milestone;
-   computation and memory cost.

The experiment estimates:

\[ Capability = F(`\mathcal{R}`{=tex}\_O, Experience, Computation) \]

while separating the effects of representational permutation, stable
identity, and explicit topology.

A result in which P1 approaches P0 would show robustness to superficial
representation arrangement. A substantial P2 cost would quantify the
sample-efficiency contribution of explicit topology. Performance under
P3 would test the stronger question of how much relational organization
can be reconstructed when both topology and stable spatial addressing
are absent.

## **Experimental Validation**

The theory makes strong developmental predictions. Future experiments
should test:

-   contingency emergence,\
-   role emergence,\
-   concept emergence,\
-   transfer emergence,\
-   future-option discovery,\
-   graph expansion,\
-   explanatory promotion,
-   outcome-equivalence emergence,
-   alternative-strategy linkage,
-   replanning after strategy invalidation,
-   target-like preference emergence.

Particular attention should be given to developmental ordering
predictions.

## **ARC and Interactive Benchmark Evaluation**

ARC-style interactive environments provide a useful testbed because they
minimize prior assumptions.

Future work should evaluate whether:

-   roles emerge before object categories,\
-   future-option motifs emerge before planning,\
-   transfer emerges through graph structure,\
-   concepts emerge through interaction compression,
-   learned outcome-equivalence classes separate consequential endpoints
    from the trajectories that reach them,
-   alternative strategies can be substituted after strategy failure
    without redefining the selected outcome.

Success on benchmarks is valuable, but developmental evidence is the
primary goal.

## **Comparison Against Alternative Theories**

Future research should directly compare the framework against:

-   reinforcement learning systems,\
-   predictive-processing systems,\
-   active-inference systems,\
-   world-model architectures,\
-   foundation-model approaches.

The goal is to determine which developmental assumptions best explain
abstraction and transfer.

## **Toward a Theory of Emergent Intelligence**

The long-term objective is not merely to solve tasks. The objective is
to understand how intelligence itself emerges. The theory proposes that
intelligence arises from remembered interaction, significance-driven
memory organization, future-option discovery, graph formation, role
emergence, explanatory compression, and continuous restructuring of
memory. Whether this developmental pathway is sufficient remains an
empirical question. The purpose of the research program is to answer
that question through formalization, implementation, and experiment.

## **Final Statement**

The central proposition of this work can be summarized as follows:

Intelligence begins not with predefined task-semantic objects, goals,
concepts, or world models, but with remembered interaction plus minimal
signed primary valence. Through valence-grounded and significance-driven
memory organization, future-option discovery, explanatory compression,
outcome-equivalence formation, and continuous restructuring,
increasingly powerful abstractions emerge. Objects, roles, concepts,
outcome identities, target-like preferences, planning, replanning, and
world models are not assumed as primitive semantic structures. They are
proposed products of development.

### **Version 0.6.0 Grounding Proposition**

If the memory-centric theory is general beyond a single observation
modality, then initially meaningless ordered symbols that reliably
co-occur with interaction structure should become integrated into the
same higher-order functional abstractions as grounded experience. Such
integration must support novel cross-modal prediction and transfer. If
it does not, the claim that the same developmental memory mechanisms can
produce grounded symbolic meaning is rejected.

------------------------------------------------------------------------

## **Version 0.6.1 Research Addendum**

Version 0.6.1 introduces four constraints on interpretation of the
theory:

1.  prediction error is not a cold-start mechanism; pre-predictive
    significance must seed recurrence before expectations become active;
2.  bounded structural retrieval must permit progressive multi-scale
    expansion rather than assume that functional equivalence is locally
    visible;
3.  stability of concurrent developmental processes is an empirical
    hypothesis, measured through reversal, churn, persistence,
    plasticity, and retained capability rather than assumed from
    architecture; and
4.  semantic abstention does not imply absence of inductive bias. The
    declared structural prior budget (`\mathcal{R}`{=tex}\_O) is an
    experimental variable whose contribution must be measured.

These changes preserve the central interaction-first and memory-centric
hypothesis while making its initialization conditions, computational
limits, stability assumptions, and observation-interface dependencies
explicitly falsifiable.

------------------------------------------------------------------------

## **Version 0.6.2 Research Addendum**

Version 0.6.2 makes four targeted refinements.

1.  **Developmental regime state is local and reversible.** Mature
    predictive memories may re-enter a pre-predictive regime when
    persistent unexplained violations indicate structural change.
    Previous memories are retained as historical/contextual evidence
    rather than erased.

2.  **Structural comparison is scale-normalized.** Similarity components
    at different graph radii must be normalized against the structural
    opportunities introduced by that scale.

3.  **Structural expansion is evidence-driven.** Multi-scale comparison
    expands only while additional structural context provides
    discriminatory, explanatory, or expected transfer value.

4.  **H17/H18 receive explicit controlled probes.** The Deep-Transfer
    Probe tests structural-depth sensitivity and computational
    boundedness. The Structural-Prior Permutation Series separates
    superficial representation permutation from removal of topology and
    stable identity.

No new central hypothesis is introduced in Version 0.6.2. These
additions tighten the mechanisms and experimental contracts supporting
the existing cold-start, developmental-stability, structural-transfer,
and structural-prior claims.

------------------------------------------------------------------------

## **Version 0.6.3 Research Addendum**

Version 0.6.3 introduces three final targeted refinements before the
theory moves primarily into implementation and empirical evaluation.

1.  **Dependency-scoped causal suspension.** When a predictive
    contingency becomes locally uncertain, dependencies relying on that
    contingency are suspended rather than either remaining fully
    authoritative or causing wholesale deletion of descendants.
    Independently supported descendants may remain active, while
    unsupported descendants enter probation. Newly learned alternatives
    form distinct causal lineages.

2.  **Explicit structural ambiguity.** Candidate entropy is introduced
    as the reference operational definition of ambiguity during
    progressive multi-scale structural comparison. Entropy reduction
    provides a concrete interpretation of incremental structural
    information, while persistent entropy identifies unresolved
    structural symmetry.

3.  **Bounded online normalization.** Scale-dependent structural
    normalization may rely on incrementally maintained local/background
    statistics rather than exact global recomputation over the evolving
    memory graph.

These refinements do not add a new central hypothesis. They make regime
re-entry, structural-search stopping, and scale normalization
operationally testable while preserving the interaction-first,
memory-centric theory.

Further architectural mechanisms should be introduced only when
experimental evidence exposes a specific failure not explained by the
existing hypotheses.

------------------------------------------------------------------------

## **Version 0.6.3.1 Clarification Addendum**

Version 0.6.3.1 freezes the conceptual framework while clarifying one
edge case in progressive structural comparison.

Persistent ambiguity combined with negligible incremental structural
information may represent genuine structural symmetry. The system is
therefore permitted to retain an explicit equivalence class of
candidates rather than forcing unique identification or repeatedly
expanding to the maximum graph radius.

Where admissible, later causal intervention or consequence evidence
should resolve whether the equivalence class must be refined or may
remain a valid outcome-equivalence abstraction.

No new central hypothesis is introduced. Remaining choices concerning
lineage storage, plausibility calibration, online-statistics data
structures, memory retirement, provenance representation, and concrete
graph implementation are system-design or experimental decisions rather
than additions to the theory.

------------------------------------------------------------------------

## **Version 0.7.0 Research Addendum**

Version 0.7.0 introduces **H19: Learned Relational Reasoning Over
Emergent Memory** and aligns the research contract with Hydra Memory
System Design v9.7.2.

The reference implementation uses an HGT-style heterogeneous graph
transformer, but H19 does not require HGT specifically. The scientific
claim is that a learned relational operator can exploit the explicit
developmental memory graph while remaining subordinate to its grounding,
provenance, lifecycle, and causal validation rules.

The principal new causal comparison is:

``` text
Hydra only
vs.
Hydra + learned relational single-pass
vs.
Hydra + learned relational recursive deliberation
```

with matched interaction budgets, held-out evaluation, model-version and
memory-version provenance, and explicit measurement of reasoning cost
and historical retention.

This extension changes the theory from:

``` text
developmental memory -> cognition
```

to the more precise:

``` text
interaction
    -> developmental memory formation
    -> explicit grounded relational substrate
    -> learned relational inference
    -> recursive candidate refinement
    -> action
    -> causal consequence
    -> memory update + continual reasoner learning
```

Persistent knowledge remains developmental and evidence-governed.
Learned reasoning becomes an adaptive computational operator over that
knowledge.
