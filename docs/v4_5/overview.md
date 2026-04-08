# v4.5 Overview

v4.5 is a control-plane layer above the existing deterministic v4 execution kernel.

It introduces:

- 5 live agents
- 2 offline agents
- planner plugins
- optional LLM advisory support
- one control authority
- no direct environment access outside the Orchestrator

The existing v4 runtime, parser, planning, family logic, and policy modules remain the source of truth for deterministic execution behavior. v4.5 does not rewrite solver families. It wraps them behind explicit contracts, stage control, and offline optimization outputs.

## Architecture

- `src/v4/*`
  - deterministic execution kernel and family logic
- `src/v4_5/orchestrator/*`
  - control authority and stage progression
- `src/v4_5/agents/*`
  - live and offline agents
- `src/v4_5/plugins/*`
  - thin wrappers over existing v4 family domains
- `src/v4_5/advisory/*`
  - optional advisory-only interface
- `src/v4_5/perception/*`
  - shared advisory perception and board-support modules
- `src/v4_5/contracts/*`
  - explicit typed inputs and outputs
- `src/v4_5/adapters/*`
  - thin bridges from v4.5 contracts to v4 structures

## Control Authority

Only the Orchestrator may commit live execution. No other agent may call the environment directly. Planner outputs are advisory to the Orchestrator until one certified action or prefix is committed.

## Live Agents

- Orchestrator Agent
- Discovery Agent
- Hypothesis Agent
- Planner Agent
- Outcome Agent

## Offline Agents

- Post-Level Optimizer Agent
- Post-Game Optimizer Agent

## Advisory Support

Advisory support is optional. The default backend is deterministic and no-op. Advisory responses are advisory-only and cannot mutate authoritative state or bypass planner verification.
