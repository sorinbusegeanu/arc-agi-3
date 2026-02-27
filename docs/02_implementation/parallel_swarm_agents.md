Parallel Swarm Agents (Deterministic, CPU-only) — Implementation Doc
0) Purpose

Scale swarm execution horizontally by running many independent swarm runs in parallel (different game_id, seed, or both), while preserving:

determinism (reproducible per run)

strict artifact isolation (no trace contamination / append bugs)

identical single-run semantics (same orchestrator logic, just replicated)

This is parallelism across environments/runs, not within a single environment step.

1). In the module docs, Swarm_Orchestrator is a meta-agent: it schedules and arbitrates via a shared blackboard. It is not “one of the swarm agents”; it is the coordinator that invokes them. : orchestrator receives an agents registry containing all modules as concrete instances (even if deterministic/rule-based), then it calls them in order.

2) Parallelism model (what to implement)
2.1 Unit of parallel work

A Run is:

one env = Arcade.make(game_id, seed=seed, ...)

one SwarmOrchestratorConfig

one output directory

one decision trace stream + blackboard snapshots

Runs are independent. No shared blackboard. No shared env.

2.2 Concurrency strategy

Use multi-process parallelism (not threads) to avoid GIL contention and to keep each run’s state isolated.

One OS process = one run at a time (or small batch sequentially inside a worker process if you want lower startup overhead).

Coordinator schedules a list of (game_id, seed) jobs across worker processes.

3) Determinism contract (must hold under parallelism)

For each run:

Seed must be fully specified and recorded:

env seed

any RNG used by orchestrator/modules (even if “deterministic today”, future changes should not break reproducibility)

No shared mutable singletons:

caches must be per-process or read-only

avoid module-level global state that accumulates across runs

Artifact isolation:

output directory is unique per run (e.g., runs/swarm_parallel/<game_id>/<seed>/<run_id>).

decision_trace.jsonl must be opened in write mode for a new run, never appended to a previous run unless explicitly “resume”.

Resume must be explicit:

if resuming, load a blackboard snapshot and continue the same run id

otherwise treat as new run and overwrite outputs

4) Data/Artifact layout (recommended)

Per run directory:

decision_trace.jsonl

blackboard/blackboard_step_<k>.json (or sparse snapshots)

lessons.json (post-run summarizer output)

run_meta.json (run_id, game_id, seed, config hash, start/end timestamps, exit reason)

Per batch (coordinator output directory):

batch_meta.json (global config, job list, worker count)

aggregate_lessons.jsonl (one line per run)

aggregate_stats.json (summary metrics)

5) Execution flow
5.1 Coordinator (parent process)

Inputs:

list of jobs: (game_id, seed)

operation mode (offline default)

worker count N

common config template

Responsibilities:

create unique run ids / output dirs

dispatch jobs to workers

collect per-run result objects (success/failure + path + key metrics)

write batch aggregates

5.2 Worker (child process)

Given one job:

Create Arcade(operation_mode=...) and env = arcade.make(game_id, seed, ...)

Construct full agents registry (target architecture) or the current minimal one (FP_Analyst only) if you’re not done yet, but record what is actually present.

Run run_game(...)

Call summarizer on the produced trace(s) to generate lessons.json

Return a compact RunResult to coordinator.

6) Agent registry expectation (align with docs)

The docs describe the orchestrator as coordinating a registry including:

FP_Analyst

Simple_Explorer

Full_Explorer

Rule_Proposer

Mechanic_Classifier

Goal_Detector

Planner

Trajectory_Summarizer (post-run)

Implementation requirement for parallelism: each worker must build its own instances, and the orchestrator must never depend on a global singleton instance.

7) Common failure modes in parallel runs (and what to enforce)

Trace contamination / wrong step counts

Caused by appending to an existing decision_trace.jsonl.

Fix by: unique per-run directory + write-new unless resume is explicitly requested.

Mismatched run_id/game_id/seed in summaries

Caused by summarizer being called without run context or reading the wrong file.

Fix by: always write run_meta.json, pass context explicitly to summarizer, and ensure summarizer reads only the run’s trace path.

Hidden shared state

Caused by module-level caches or static counters.

Fix by: instantiate modules inside worker; if caching needed, scope it to the instance.

8) Minimal “parallel swarm” milestone (no refactor required)

Even if you keep the current “FP_Analyst only + orchestrator-internals” approach, parallelism is still valid if:

each worker executes exactly the same single-run code path you already use

per-run output dirs are unique

trace files are never reused unless resume is explicit

This gives immediate horizontal scale for data generation and debugging.

9) What “done” looks like (acceptance)

Running a batch of (game_id, seed) produces:

one isolated run directory per job

a batch aggregate file listing each run’s key metrics

Re-running the same batch produces byte-identical outputs per run (or diffs only in timestamps), assuming same code + same inputs.

No run ever reports mismatched steps due to reading stale traces.

Registry shape (recommended, minimal)

A single dict, plus a fixed call order list:

agents = {

"fp_analyst": FPAnalyst()

"simple_explorer": SimpleExplorer()

"full_explorer": FullExplorer()

"mechanic_classifier": MechanicClassifier()

"rule_proposer": RuleProposer()

"goal_detector": GoalDetector()

"planner": Planner()

}

call_order = [

fp_analyst

mechanic_classifier

rule_proposer

simple_explorer (probe)

full_explorer (probe)

planner (exploit)

goal_detector

]

Selection policy:

probe phase reads explorer proposals

exploit phase reads planner proposal

orchestrator applies one chosen action


## 11) Where it should live

* Exact new/modified files:

   `src/arc_agi_agent/swarm_orchestrator.py` (modify)
  
  ## 2) Public API to implement

Specify one of:

* a CLI: `python -m arc_agi_agent.swarm_parallel --jobs <...> --workers N ...`

## 3) Job specification format

--games ls20,bt11 
--start_seed 0 then automatically incremented

## 4) Resume/overwrite policy (must be explicit)

* default: **overwrite** if run dir exists (or fail-fast)


## 5) Artifact contract (exact filenames)

`decision_trace.jsonl` must be opened with `"w"` unless resume
blackboards go under `blackboard/`

## 6) Registry refactor scope (pick one)

refactor orchestrator to use agent instances and a call order.



