Status: implemented and verified
Scope: `rs01` typed state
Source of truth: `/home/zodrak/zod/src/v4/rule_switch/typedState.py`, `/home/zodrak/zod/src/v4/rule_switch/familyAdapters.py`
Last verified against: unknown

# Typed State

`RuleSwitchTypedStateV4` includes:

- avatar position
- walkable cells and wall cells
- currently visible target cells
- currently active safe color
- configured safe-color cycle
- collected and remaining targets grouped by color
- visible cycle metadata exposed by observation and level data
