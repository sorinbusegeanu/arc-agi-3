"""ARC-AGI3 runtime patches loaded automatically through PYTHONPATH=src."""

try:
    from v6.higher_order_scope_index import apply_patch as _apply_scope_index_patch
except Exception:
    _apply_scope_index_patch = None
if _apply_scope_index_patch is not None:
    _apply_scope_index_patch()

try:
    from v6.hypothesis_pipeline_repairs import apply_patch as _apply_hypothesis_repairs
except Exception:
    _apply_hypothesis_repairs = None
if _apply_hypothesis_repairs is not None:
    _apply_hypothesis_repairs()

try:
    from v6.report_consolidation import apply_patch as _apply_report_consolidation
except Exception:
    _apply_report_consolidation = None
if _apply_report_consolidation is not None:
    _apply_report_consolidation()

try:
    from v6.h07_h09_next_repairs import apply_patch as _apply_h07_h09_next_repairs
except Exception:
    _apply_h07_h09_next_repairs = None
if _apply_h07_h09_next_repairs is not None:
    _apply_h07_h09_next_repairs()

try:
    from v6.h07_h08_evidence_repairs import apply_patch as _apply_h07_h08_evidence_repairs
except Exception:
    _apply_h07_h08_evidence_repairs = None
if _apply_h07_h08_evidence_repairs is not None:
    _apply_h07_h08_evidence_repairs()

from v6.remaining_report_repairs import apply_patch as _apply_remaining_report_repairs

_apply_remaining_report_repairs()
