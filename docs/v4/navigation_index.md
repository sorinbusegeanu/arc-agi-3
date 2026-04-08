Status: implemented and verified
Scope: Compact navigation index for docs/v4
Source of truth: `/home/zodrak/zod/docs/v4/*`
Last verified against: current repo state on 2026-03-29; targeted movement tests for `pb02`, `pb03`, `fs02`, and `fs03`

# Navigation Index

## Overview

- [README.md](README.md): top-level v4 overview and package/test map.
- [current_status.md](current_status.md): precise implementation and verification status.
- [implemented_vs_planned.md](implemented_vs_planned.md): implemented vs documented-only split by track.

## Contracts

- [agent_contracts/README.md](agent_contracts/README.md): Phase 1 contract overview.
- [agent_contracts/observation_format.md](agent_contracts/observation_format.md): authoritative observation model.
- [agent_contracts/action_format.md](agent_contracts/action_format.md): authoritative action model and legality rules.
- [agent_contracts/environment_metadata.md](agent_contracts/environment_metadata.md): static environment metadata rules.
- [agent_contracts/terminal_signal.md](agent_contracts/terminal_signal.md): terminal mapping rules.

## Closed Loop

- [closed_loop/stage2_overview.md](closed_loop/stage2_overview.md): Stage 2 closed-loop overview.
- [closed_loop/package_layout.md](closed_loop/package_layout.md): Stage 2 package layout.
- [closed_loop/parsed_state.md](closed_loop/parsed_state.md): parsed-state surface and derived control state.
- [closed_loop/local_memory.md](closed_loop/local_memory.md): bounded session-local memory.
- [closed_loop/policy.md](closed_loop/policy.md): shared Stage 2 policy surface.

## Movement

- [movement/phase3_overview.md](movement/phase3_overview.md): Phase 3 movement overview.
- [movement/package_layout.md](movement/package_layout.md): movement package layout.
- [movement/typed_state.md](movement/typed_state.md): movement typed-state contract.
- [movement/transition_model.md](movement/transition_model.md): movement transition semantics.
- [movement/search_policy.md](movement/search_policy.md): movement search and policy rules.
- [movement/family_coverage.md](movement/family_coverage.md): movement family coverage summary.
- [movement/fs02_mechanics.md](movement/fs02_mechanics.md): `fs02` family mechanics.
- [movement/fs02_gate.md](movement/fs02_gate.md): `fs02` gate.
- [movement/fs03_mechanics.md](movement/fs03_mechanics.md): `fs03` family mechanics.
- [movement/fs03_gate.md](movement/fs03_gate.md): `fs03` gate.
- [movement/pb02_mechanics.md](movement/pb02_mechanics.md): `pb02` family mechanics and current boundary.
- [movement/pb02_gate.md](movement/pb02_gate.md): `pb02` gate.
- [movement/pb03_mechanics.md](movement/pb03_mechanics.md): `pb03` family mechanics.
- [movement/pb03_gate.md](movement/pb03_gate.md): `pb03` gate.

## Click/Perception

- [click_perception/phase4_overview.md](click_perception/phase4_overview.md): Phase 4 click/perception overview.
- [click_perception/package_layout.md](click_perception/package_layout.md): click package layout.
- [click_perception/typed_state.md](click_perception/typed_state.md): click typed-state contract.
- [click_perception/transition_model.md](click_perception/transition_model.md): click transition semantics.
- [click_perception/search_policy.md](click_perception/search_policy.md): click search and policy rules.
- [click_perception/family_coverage.md](click_perception/family_coverage.md): click family coverage summary.

## Memory Hidden

- [memory_hidden/phase5_overview.md](memory_hidden/phase5_overview.md): Phase 5 memory-hidden overview.
- [memory_hidden/package_layout.md](memory_hidden/package_layout.md): memory-hidden package layout.
- [memory_hidden/typed_state.md](memory_hidden/typed_state.md): memory-hidden typed-state contract.
- [memory_hidden/transition_model.md](memory_hidden/transition_model.md): memory-hidden transition semantics.
- [memory_hidden/search_policy.md](memory_hidden/search_policy.md): memory-hidden search and policy rules.
- [memory_hidden/family_coverage.md](memory_hidden/family_coverage.md): memory-hidden family coverage summary.
- [memory_hidden/ms01_mechanics.md](memory_hidden/ms01_mechanics.md): `ms01` mechanics.
- [memory_hidden/ms01_gate.md](memory_hidden/ms01_gate.md): `ms01` gate.

## Rule Switch

- [rule_switch/phase6_overview.md](rule_switch/phase6_overview.md): Phase 6 rule-switch overview.
- [rule_switch/package_layout.md](rule_switch/package_layout.md): rule-switch package layout.
- [rule_switch/typed_state.md](rule_switch/typed_state.md): rule-switch typed-state contract.
- [rule_switch/transition_model.md](rule_switch/transition_model.md): rule-switch transition semantics.
- [rule_switch/search_policy.md](rule_switch/search_policy.md): rule-switch search and policy rules.
- [rule_switch/family_coverage.md](rule_switch/family_coverage.md): rule-switch family coverage summary.
- [rule_switch/rs01_mechanics.md](rule_switch/rs01_mechanics.md): `rs01` mechanics.
- [rule_switch/rs01_gate.md](rule_switch/rs01_gate.md): `rs01` gate.

## Time/Reactive

- [time_reactive/phase7_overview.md](time_reactive/phase7_overview.md): Phase 7 time/reactive overview.
- [time_reactive/package_layout.md](time_reactive/package_layout.md): time/reactive package layout.
- [time_reactive/typed_state.md](time_reactive/typed_state.md): time/reactive typed-state contract.
- [time_reactive/transition_model.md](time_reactive/transition_model.md): time/reactive transition semantics.
- [time_reactive/search_policy.md](time_reactive/search_policy.md): time/reactive search and policy rules.
- [time_reactive/family_coverage.md](time_reactive/family_coverage.md): time/reactive family coverage summary.
- [time_reactive/sv01_mechanics.md](time_reactive/sv01_mechanics.md): `sv01` mechanics.
- [time_reactive/sv01_gate.md](time_reactive/sv01_gate.md): `sv01` gate.

## Hybrid Construction

- [hybrid_construction/phase8_overview.md](hybrid_construction/phase8_overview.md): Phase 8 hybrid-construction overview.
- [hybrid_construction/package_layout.md](hybrid_construction/package_layout.md): hybrid-construction package layout.
- [hybrid_construction/typed_state.md](hybrid_construction/typed_state.md): hybrid-construction typed-state contract.
- [hybrid_construction/transition_model.md](hybrid_construction/transition_model.md): hybrid-construction transition semantics.
- [hybrid_construction/search_policy.md](hybrid_construction/search_policy.md): hybrid-construction search and policy rules.
- [hybrid_construction/family_coverage.md](hybrid_construction/family_coverage.md): hybrid-construction family coverage summary.
- [hybrid_construction/tb01_mechanics.md](hybrid_construction/tb01_mechanics.md): `tb01` mechanics.
- [hybrid_construction/tb01_gate.md](hybrid_construction/tb01_gate.md): `tb01` gate.

## Status And Audits

- [source_to_doc_map.md](source_to_doc_map.md): source-package documentation coverage map.
- [test_to_doc_map.md](test_to_doc_map.md): test-package documentation coverage map.
- [documentation_gap_checklist.md](documentation_gap_checklist.md): missing and outdated documentation audit.
- [tests/README_stage2_next_gate.md](tests/README_stage2_next_gate.md): historical Stage 2 note plus the separate live regression runner note.
