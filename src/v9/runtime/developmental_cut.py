from __future__ import annotations

import contextvars
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Mapping

from .canonical_store import CanonicalStateHandle, PinnedCanonicalHandle
from .scientific_modes import ScientificVisibilityMode, coerce_visibility_mode


class DevelopmentalWorkStatus(str, Enum):
    APPLIED = "APPLIED"
    NO_CHANGE = "NO_CHANGE"
    REJECTED_BY_CAUSAL_RULE = "REJECTED_BY_CAUSAL_RULE"
    FAILED_DETERMINISTICALLY = "FAILED_DETERMINISTICALLY"
    QUARANTINED = "QUARANTINED"


REQUIRED_DEVELOPMENTAL_OPERATORS = (
    "m1_maturation",
    "m2_family_formation",
    "carrier_proposal_validation",
    "role_proposal_validation",
    "context_refinement",
    "grounding_maturation",
    "transfer_validation",
    "m4_validation",
    "future_option_updates",
    "m5_consequences",
    "m6_outcome_equivalence",
    "m7_strategy_replanning",
    "lifecycle",
    "replay_allocation_metadata",
)


@dataclass(frozen=True, slots=True)
class DevelopmentalOperator:
    name: str
    version: int
    work_budget: int
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or min(self.version, self.work_budget) <= 0:
            raise ValueError("developmental operators require a name, version and positive budget")


@dataclass(frozen=True, slots=True)
class DevelopmentalPipelineVersion:
    version: str
    operators: tuple[DevelopmentalOperator, ...]

    def __post_init__(self) -> None:
        names = tuple(operator.name for operator in self.operators)
        if names != REQUIRED_DEVELOPMENTAL_OPERATORS:
            raise ValueError("developmental pipeline must cover every actor-visible operator in order")


@dataclass(frozen=True, slots=True)
class DevelopmentalCandidate:
    stable_key: str
    priority: tuple[int, ...]
    evidence_version: int
    payload_checksum: str = ""


@dataclass(frozen=True, slots=True)
class DevelopmentalWorkResult:
    operator: str
    stable_key: str
    target_evidence_version: int
    status: DevelopmentalWorkStatus
    detail: str = ""


@dataclass(frozen=True, slots=True)
class DevelopmentalCutResult:
    evidence_cut_id: str
    base_handle_checksum: str
    pipeline_version: str
    results: tuple[DevelopmentalWorkResult, ...]
    manifest_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_cut_id": self.evidence_cut_id,
            "base_handle_checksum": self.base_handle_checksum,
            "pipeline_version": self.pipeline_version,
            "manifest_id": self.manifest_id,
            "results": [
                {
                    "operator": row.operator,
                    "stable_key": row.stable_key,
                    "target_evidence_version": row.target_evidence_version,
                    "status": row.status.value,
                    "detail": row.detail,
                }
                for row in self.results
            ],
        }


def default_developmental_pipeline(*, work_budget: int = 256) -> DevelopmentalPipelineVersion:
    return DevelopmentalPipelineVersion(
        "developmental-pipeline-v1",
        tuple(DevelopmentalOperator(name, 1, work_budget) for name in REQUIRED_DEVELOPMENTAL_OPERATORS),
    )


class DevelopmentalCut:
    def __init__(self, pipeline: DevelopmentalPipelineVersion | None = None) -> None:
        self.pipeline = pipeline or default_developmental_pipeline()

    def run(
        self,
        *,
        evidence_cut_id: str,
        base_handle: CanonicalStateHandle,
        candidates: Mapping[str, Iterable[DevelopmentalCandidate]],
        apply: Callable[[DevelopmentalOperator, DevelopmentalCandidate], DevelopmentalWorkStatus | tuple[DevelopmentalWorkStatus, str]],
    ) -> DevelopmentalCutResult:
        rows: list[DevelopmentalWorkResult] = []
        for operator in self.pipeline.operators:
            eligible = sorted(
                tuple(candidates.get(operator.name, ())),
                key=lambda candidate: (candidate.priority, candidate.stable_key, candidate.evidence_version),
            )[: operator.work_budget]
            for candidate in eligible:
                try:
                    outcome = apply(operator, candidate)
                    status, detail = outcome if isinstance(outcome, tuple) else (outcome, "")
                    if not isinstance(status, DevelopmentalWorkStatus):
                        raise TypeError("operator returned an invalid status")
                except Exception as exc:
                    status, detail = DevelopmentalWorkStatus.FAILED_DETERMINISTICALLY, type(exc).__name__
                rows.append(
                    DevelopmentalWorkResult(
                        operator.name, candidate.stable_key, candidate.evidence_version, status, detail
                    )
                )
        return self.result_from_rows(
            evidence_cut_id=evidence_cut_id,
            base_handle=base_handle,
            rows=rows,
        )

    def result_from_rows(
        self,
        *,
        evidence_cut_id: str,
        base_handle: CanonicalStateHandle,
        rows: Iterable[DevelopmentalWorkResult],
    ) -> DevelopmentalCutResult:
        ordered_rows = tuple(
            sorted(
                rows,
                key=lambda row: (
                    REQUIRED_DEVELOPMENTAL_OPERATORS.index(row.operator),
                    row.stable_key,
                    row.target_evidence_version,
                ),
            )
        )
        payload = {
            "evidence_cut_id": evidence_cut_id,
            "base_handle_checksum": base_handle.checksum,
            "pipeline_version": self.pipeline.version,
            "operators": [
                {"name": row.name, "version": row.version, "budget": row.work_budget, "dependencies": row.dependencies}
                for row in self.pipeline.operators
            ],
            "results": [
                [row.operator, row.stable_key, row.target_evidence_version, row.status.value, row.detail]
                for row in ordered_rows
            ],
        }
        manifest_id = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return DevelopmentalCutResult(
            evidence_cut_id, base_handle.checksum, self.pipeline.version, ordered_rows, manifest_id
        )


_ACTIVE_CUT: contextvars.ContextVar[str | None] = contextvars.ContextVar("v9_active_developmental_cut", default=None)


class DevelopmentalMutationGate:
    def __init__(self, mode: ScientificVisibilityMode | str) -> None:
        self.mode = coerce_visibility_mode(mode)

    def allow_cut(self, cut_id: str):
        if not cut_id:
            raise ValueError("cut_id is required")
        gate = self

        class _Scope:
            def __enter__(self):
                self.token = _ACTIVE_CUT.set(cut_id)
                return cut_id

            def __exit__(self, *_args):
                _ACTIVE_CUT.reset(self.token)

        return _Scope()

    def assert_publication_allowed(self, *, async_origin: bool = False) -> None:
        if self.mode is ScientificVisibilityMode.ASYNC_DEVELOPMENT:
            if not async_origin and _ACTIVE_CUT.get() is None:
                raise RuntimeError("asynchronous developmental publication requires a declared origin")
            return
        if _ACTIVE_CUT.get() is None:
            raise RuntimeError("MATCHED_REASONING developmental mutation is outside DevelopmentalCut")


class DevelopmentalCutSession:
    """One pinned matched-mode cut spanning explicitly composed runtime work.

    Runtime operations execute in their existing causal order.  Their outcomes
    are canonicalized into pipeline order when the immutable manifest is
    finalized, so worker or caller completion timing cannot change its identity.
    """

    def __init__(
        self,
        *,
        cut: DevelopmentalCut,
        gate: DevelopmentalMutationGate,
        evidence_cut_id: str,
        pinned_handle: PinnedCanonicalHandle,
    ) -> None:
        if not evidence_cut_id:
            pinned_handle.release()
            raise ValueError("evidence_cut_id is required")
        self.cut = cut
        self.gate = gate
        self.evidence_cut_id = evidence_cut_id
        self._pinned_handle = pinned_handle
        self.base_handle = pinned_handle.handle
        self._rows: list[DevelopmentalWorkResult] = []
        self._counts: dict[str, int] = {}
        self._closed = False

    def run(
        self,
        operator: str,
        operation: Callable[[], Any],
        *,
        stable_key: str | None = None,
        evidence_version: int = 1,
        status: DevelopmentalWorkStatus | Callable[[Any], DevelopmentalWorkStatus] = DevelopmentalWorkStatus.APPLIED,
    ) -> Any:
        if self._closed:
            raise RuntimeError("developmental cut session is closed")
        definitions = {row.name: row for row in self.cut.pipeline.operators}
        if operator not in definitions:
            raise KeyError(f"unknown developmental operator {operator!r}")
        count = self._counts.get(operator, 0)
        if count >= definitions[operator].work_budget:
            raise RuntimeError(f"developmental operator {operator!r} exceeded its fixed work budget")
        key = stable_key or f"{operator}:{count:08d}"
        try:
            with self.gate.allow_cut(self.evidence_cut_id):
                value = operation()
            resolved = status(value) if callable(status) else status
            if not isinstance(resolved, DevelopmentalWorkStatus):
                raise TypeError("developmental operation returned an invalid status")
        except BaseException as exc:
            self._rows.append(
                DevelopmentalWorkResult(
                    operator,
                    key,
                    int(evidence_version),
                    DevelopmentalWorkStatus.FAILED_DETERMINISTICALLY,
                    type(exc).__name__,
                )
            )
            self._counts[operator] = count + 1
            self.abort()
            raise
        self._rows.append(
            DevelopmentalWorkResult(operator, key, int(evidence_version), resolved)
        )
        self._counts[operator] = count + 1
        return value

    def finish(self) -> DevelopmentalCutResult:
        if self._closed:
            raise RuntimeError("developmental cut session is closed")
        try:
            recorded = set(self._counts)
            rows = list(self._rows)
            for operator in self.cut.pipeline.operators:
                if operator.name not in recorded:
                    rows.append(
                        DevelopmentalWorkResult(
                            operator.name,
                            f"{operator.name}:none",
                            0,
                            DevelopmentalWorkStatus.NO_CHANGE,
                        )
                    )
            return self.cut.result_from_rows(
                evidence_cut_id=self.evidence_cut_id,
                base_handle=self.base_handle,
                rows=rows,
            )
        finally:
            self._closed = True
            self._pinned_handle.release()

    def abort(self) -> None:
        if not self._closed:
            self._closed = True
            self._pinned_handle.release()


__all__ = [
    "DevelopmentalCandidate",
    "DevelopmentalCut",
    "DevelopmentalCutResult",
    "DevelopmentalCutSession",
    "DevelopmentalMutationGate",
    "DevelopmentalOperator",
    "DevelopmentalPipelineVersion",
    "DevelopmentalWorkResult",
    "DevelopmentalWorkStatus",
    "REQUIRED_DEVELOPMENTAL_OPERATORS",
    "default_developmental_pipeline",
]
