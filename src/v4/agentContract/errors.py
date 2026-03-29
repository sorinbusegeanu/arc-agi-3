from __future__ import annotations

from typing import Any


class V4ContractError(Exception):
    """Base v4 contract error."""

    def __init__(self, message: str, *, source_field: str | None = None, context: dict[str, Any] | None = None) -> None:
        self.message = message
        self.source_field = source_field
        self.context = dict(context or {})
        details = message
        if source_field:
            details = f"{source_field}: {details}"
        if self.context:
            details = f"{details} | context={self.context}"
        super().__init__(details)


class V4ValidationError(V4ContractError):
    """Raised when authoritative contract validation fails."""


class V4AdapterError(V4ContractError):
    """Raised when a source object is incompatible with the adapter boundary."""


class V4MissingFieldError(V4ValidationError):
    """Raised when a required authoritative field is missing."""


class V4UnknownFieldError(V4ValidationError):
    """Raised when a strict authoritative payload contains unknown fields."""


class V4InvalidActionError(V4ValidationError):
    """Raised when an action id or action name is invalid."""


class V4IllegalActionError(V4ValidationError):
    """Raised when an action is illegal for the current observation."""


class V4InvalidPayloadError(V4ValidationError):
    """Raised when an action payload violates the contract."""


class V4InvalidTerminalSignalError(V4ValidationError):
    """Raised when terminal derivation does not match raw state."""


class V4InvalidTransitionError(V4ValidationError):
    """Raised when a transition record violates invariants."""


class V4MetadataMismatchError(V4ValidationError):
    """Raised when static metadata conflicts with observed authoritative state."""

