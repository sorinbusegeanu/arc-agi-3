from __future__ import annotations


class V31Error(Exception):
    """Base v3.1 error."""


class VersionMismatchError(V31Error):
    """Raised when helper output does not match the accepted planning context."""


class SnapshotNotFoundError(V31Error):
    """Raised when a snapshot handle cannot be resolved."""


class InvalidConfigurationError(V31Error):
    """Raised when runtime configuration is incomplete or invalid."""

