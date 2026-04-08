class V45Error(Exception):
    pass


class ContractValidationError(V45Error):
    pass


class StageTransitionError(V45Error):
    pass


class PluginRegistrationError(V45Error):
    pass


class ExecutionAuthorityError(V45Error):
    pass


class AvatarNotUniquelyIdentifiedError(V45Error):
    pass


class BootstrapPngExportError(V45Error):
    pass


class BootstrapVideoExportError(V45Error):
    pass


class DeterministicHudAnalysisError(V45Error):
    pass


class ToonTextCallError(V45Error):
    pass


class ToonVideoCallError(V45Error):
    pass


class BootstrapInvalidActionError(V45Error):
    pass
