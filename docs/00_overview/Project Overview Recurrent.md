# Project Overview: Recurrent RL Agent for Novel Game Environments

## Objective

Build a single end-to-end reinforcement learning agent that:

* Receives an observation at each step (grid(s) + metadata)
* Selects an action (discrete or coordinate-based)
* Learns through trial and error
* Reaches terminal win states efficiently
* Generalizes across unseen game mechanics without relying on a fixed rule catalog

This project replaces explicit mechanic inference with learned action utility through interaction.

---

# System Architecture

The system consists of seven core components.

---

## 1. Observation Encoder (Neural Network)

### Purpose

Transform raw environment observations into a compact embedding usable by the policy.

### Inputs

* Grid(s): typically 64×64 categorical values
* Metadata:

  * available actions
  * terminal flag (if present)
  * reward (if present)
  * any numeric fields

### Architecture

* CNN over grid(s)
* MLP over metadata
* Concatenation → final embedding `z_t`

Output:

```
z_t ∈ R^D
```

---

## 2. Memory Module (Neural Network)

### Purpose

Maintain state over time for partially observable or mode-dependent games.

### Input

* Current embedding `z_t`
* Previous action `a_{t-1}`
* Previous reward `r_{t-1}`
* Previous hidden state `h_{t-1}`

### Architecture

* GRU (default) or LSTM

Output:

```
h_t ∈ R^H
```

This enables adaptation to hidden mechanics and state transitions.

---

## 3. Policy Network (Neural Network)

### Purpose

Produce a probability distribution over valid actions.

### Outputs

#### A. Discrete Action Head

Logits over:

```
{RESET, ACTION1–5, ACTION7}
```

Masked by available actions.

#### B. Coordinate Action Head (ACTION6)

Two supported modes:

**Mode 1 (recommended): Proposal + Scorer**

* Deterministic coordinate proposer generates K candidate points
* MLP scorer ranks K candidates conditioned on `h_t`
* Softmax over K candidates

**Mode 2: Spatial Map**

* CNN/deconv head produces 64×64 probability map
* Mask invalid regions if needed

---

## 4. Value Network (Neural Network)

### Purpose

Estimate expected future return from current state.

Input:

```
h_t
```

Output:

```
V(h_t)
```

Used for actor-critic training.

---

## 5. Reward Shaping Module (Deterministic Logic)

### Purpose

Convert sparse terminal rewards into learnable signal.

### Reward components

* `+1` if terminal win
* `+α` if new state hash (novelty bonus)
* `+β` if non-noop effect (state changed)
* `-δ` if repeated state (loop penalty)
* Environment reward if provided

Reward shaping is deterministic and reproducible.

---

## 6. Rollout Collector

### Purpose

Run episodes and store trajectories.

Stored per step:

```
(o_t, a_t, r_t, o_{t+1}, done, available_actions_mask)
```

If using coordinate proposals:

* store proposal set and selected index

---

## 7. Trainer

### Learning Algorithm

Recurrent Actor-Critic (PPO-style recommended)

### Training Loop

1. Collect episodes
2. Compute returns and advantages
3. Update:

   * policy network
   * value network
   * recurrent memory

On-policy training recommended for stability.

---

# Inference Loop

At runtime:

1. Encode observation → `z_t`
2. Update memory → `h_t`
3. Compute action distribution
4. Select action (argmax or stochastic)
5. Execute action
6. Observe next state
7. Repeat until win or budget exhausted

---

# Why This Architecture

* No fixed mechanic catalog
* No explicit rule synthesis
* Learns action utility directly from experience
* Handles hidden modes via recurrence
* Handles coordinate actions via structured proposal
* Scales across heterogeneous games

---

# Generalization Strategy

Generalization is achieved through:

* Shared encoder across many game types
* Recurrent memory for within-episode adaptation
* Exploration bonuses (novelty reward)
* Training across diverse game distributions

The model learns:

* Which actions cause meaningful state transitions
* Where coordinate actions are effective
* How to avoid loops
* How to reach terminal win states

---

# Neural Components Summary

| Component            | Type                 |
| -------------------- | -------------------- |
| Grid encoder         | CNN                  |
| Metadata encoder     | MLP                  |
| Memory module        | GRU/LSTM             |
| Discrete action head | MLP                  |
| Coord scorer head    | MLP (or spatial CNN) |
| Value head           | MLP                  |

---

# Project Success Criteria

The project is successful when:

* The agent improves win rate over time
* Performance generalizes to unseen games
* No hard-coded rule catalog is required
* Behavior adapts within episode to unknown mechanics
* Training is stable and reproducible


