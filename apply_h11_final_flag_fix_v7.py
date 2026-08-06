#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

path = Path("src/v6/hypothesis_h11_report.py")
text = path.read_text(encoding="utf-8")

anchor = '''        "motif_transfer_chain_provenance_truncated":
            len(provenance_sample) < len(links),
'''
replacement = anchor + '''        "motif_transfer_chain_provenance_is_sample":
            len(provenance_sample) < len(links),
'''

if '"motif_transfer_chain_provenance_is_sample":' not in text:
    if anchor not in text:
        raise RuntimeError("Could not find provenance truncation metric")
    text = text.replace(anchor, replacement, 1)

# Keep the key stable in early-return payloads.
for indent in ("            ", "                "):
    marker = indent + '"motif_transfer_chain_provenance_truncated": False,'
    replacement = (
        marker
        + "\n"
        + indent
        + '"motif_transfer_chain_provenance_is_sample": False,'
    )
    if marker in text and replacement not in text:
        text = text.replace(marker, replacement)

# Preserve it when the report is compacted.
compact_anchor = (
    '        compact["motif_transfer_chain_provenance_truncated"] = True\n'
)
compact_replacement = (
    compact_anchor
    + '        compact["motif_transfer_chain_provenance_is_sample"] = True\n'
)
if (
    'compact["motif_transfer_chain_provenance_is_sample"]' not in text
    and compact_anchor in text
):
    text = text.replace(compact_anchor, compact_replacement, 1)

compile(text, str(path), "exec")

backup = path.with_suffix(path.suffix + ".before_h11_fix_v7")
if not backup.exists():
    shutil.copy2(path, backup)

path.write_text(text, encoding="utf-8")
print(f"applied: {path}")
