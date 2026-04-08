# `rl_v1` V1

`rl_v1` is a new isolated RL stack under [src/rl_v1](/home/zodrak/zod/src/rl_v1).

It is built directly on the current ARC toolkit contract:

- environment creation via `Arcade.make(...)`
- environment interaction via `EnvironmentWrapper.reset()` and `step(action, data=None, reasoning=None)`
- offline mode as the default development path
- explicit support for `ACTION6` with `x,y` payload

This package does not depend on or alias any existing deterministic or prior RL version tree in this repository.

Top-level package layout:

- `env`
- `data`
- `model`
- `planner`
- `training`
- `eval`
- `configs`
- `utils`

Primary entrypoint: [src/rl_v1/cli.py](/home/zodrak/zod/src/rl_v1/cli.py)
