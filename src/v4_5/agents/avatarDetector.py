from __future__ import annotations

from v4_5.contracts import AvatarDetectionResult, AvatarNotUniquelyIdentifiedError
from v4_5.perception.board_builder.avatarExtractor import extract_avatar_from_transition_records


class AvatarDetector:
    def __init__(self) -> None:
        self.last_result: AvatarDetectionResult | None = None

    def detect(
        self,
        *,
        primary_sequence,
        fallback_sequence=None,
    ) -> AvatarDetectionResult:
        primary = extract_avatar_from_transition_records(tuple(primary_sequence or ()), used_fallback=False)
        self.last_result = primary
        if primary.avatar_bbox is not None and primary.failure_reason is None:
            return primary
        if fallback_sequence is not None:
            fallback = extract_avatar_from_transition_records(tuple(fallback_sequence or ()), used_fallback=True)
            self.last_result = fallback
            if fallback.avatar_bbox is not None and fallback.failure_reason is None:
                return fallback
        failure_reason = None if self.last_result is None else self.last_result.failure_reason
        raise AvatarNotUniquelyIdentifiedError(failure_reason or "avatar not uniquely identified")
