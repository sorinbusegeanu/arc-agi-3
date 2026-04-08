from __future__ import annotations

from dataclasses import replace

from v4.state.parsedState import ParsedStateV4

from .composedTransitionModel import ComposedTransitionModelV4
from .domainState import ComposedDomainStateV4


class CompositionUpdaterV4:
    def __init__(self) -> None:
        self.transition_model = ComposedTransitionModelV4()

    def initialize_from_parsed_state(self, parsed_state: ParsedStateV4) -> ComposedDomainStateV4:
        built = self.transition_model.build(parsed_state)
        return replace(built, revision=0)

    def update_from_parsed_state(self, previous_state: ComposedDomainStateV4 | None, parsed_state: ParsedStateV4) -> ComposedDomainStateV4:
        built = self.transition_model.build(parsed_state)
        return replace(built, revision=0 if previous_state is None else previous_state.revision + 1)
