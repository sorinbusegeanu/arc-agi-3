# `rl_v1` Implementation Notes

The V1 stack is implemented in isolation under [src/rl_v1](/home/zodrak/zod/src/rl_v1).

Current implementation scope:

- thin ARC adapter over `Arcade` and `EnvironmentWrapper`
- normalized observation and action contracts
- recurrent sequence rollout collector
- deterministic frame preprocessing with padded canvas and valid-region mask
- CNN encoder with token and spatial outputs
- Slot Attention plus relation transformer plus pooling
- GRU recurrent core and planner latent projection
- discrete policy head, separate click branch, and scalar value head
- latent dynamics model for next latent, reward, and done prediction
- latent beam-search planner
- custom PPO-style training loop with optional Fabric shim
- evaluation artifacts and core unit/integration tests

Default config is [src/rl_v1/configs/default_v1.json](/home/zodrak/zod/src/rl_v1/configs/default_v1.json). Small debug config is [src/rl_v1/configs/debug_v1.json](/home/zodrak/zod/src/rl_v1/configs/debug_v1.json).
