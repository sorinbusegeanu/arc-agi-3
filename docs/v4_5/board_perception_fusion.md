# Board Perception Fusion

The board perception fusion design has 3 decoupled layers:

- deterministic geometry layer
- deterministic board builder layer
- learned supplement layer

v1 defines interfaces for all three layers, but only the first two are active.

The learned layer is stubbed in v1 with passthrough/no-op fusion.

## Future Learned Design

- one shared CNN backbone
- additional heads
- optional family adapters later
- no separate family models by default

Source-facing design note:

- future training may use environment-provided truth such as avatar position and sprite/object metadata

