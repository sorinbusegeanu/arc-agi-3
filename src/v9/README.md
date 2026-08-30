# Hydra v9.3

`src/v9` is the compatibility-first implementation of the Hydra v9.3 design and research contract.

Implemented phases:

- v9.00 immutable scientific configuration, runtime-stack and arena-layout baseline;
- v9.01 modality, symbol, environment-instance, and episode provenance;
- v9.02 bounded passive multimodal timeline with no synthetic action IDs;
- v9.03 deterministic synthetic symbolic grounding environment and controls;
- v9.04 event-grounded multimodal M0;
- v9.05 structural multimodal M1G relations;
- v9.06 bounded normalized M1N facts;
- v9.07 object-version/read-set state mutation authority;
- v9.08 mixed-modality M2 formation;
- v9.09 mixed-modality M3 role convergence;
- v9.10 sparse lineage/context overlays and evidence-aware probation;
- v9.11 progressive multi-scale structural similarity with bounded normalization;
- v9.12 structural cross-modal correspondence candidates without semantic authority;
- v9.13 empirical grounding maturity G0-G5;
- v9.14 shadow symbol-conditioned prediction;
- v9.15 G4 local/G5 cross-environment gated symbol influence;
- v9.16 matched H16 C0-C3 scientific controls;
- v9.17 raw-symbol BabyAI adapter boundary;
- v9.18 environment-neutral target-local transfer gate;
- v9.19 bounded payload residency with provenance retained after payload retirement;
- v9.20 explicit v8-to-v9 auxiliary snapshot migration and completeness audit;
- v9.21 backend-neutral raw-symbol ALFRED adapter boundary;
- v9.22 authority-area consolidation audit and empirical removal gate;
- v9.23 compatibility `python -m v9` entrypoint preserving the current CLI contract.

## Authority status

The existing v8 runtime remains authoritative for production sampling and action selection. v9.22 intentionally refuses removal of historical runtime layers until all required empirical gates are satisfied: interpretable H16 controls, validated BabyAI grounding, bounded memory growth, and stable snapshot/restart behavior.

This is a scientific gate, not an unfinished code path. Destructive runtime-stack consolidation must not be enabled from unit tests alone.
