#!/usr/bin/env python3
from __future__ import annotations

import re
import shutil
from pathlib import Path

path = Path("src/v6/hypothesis_h11_report.py")
text = path.read_text(encoding="utf-8")

# Create a compact embedded sample before the metrics dictionary.
marker = '''    metrics: dict[str, object] = {
'''
block = '''    provenance_report_sample = [
        {
            "motif_signature": row.get("motif_signature"),
            "role_signature": row.get("role_signature"),
            "concept_signature": row.get("concept_signature"),
            "transfer_pair_id": row.get("transfer_pair_id"),
            "fully_verified": row.get("fully_verified"),
        }
        for row in provenance_sample
    ]

    metrics: dict[str, object] = {
'''
if "provenance_report_sample = [" not in text:
    if marker not in text:
        raise RuntimeError("Could not locate H11 metrics dictionary")
    text = text.replace(marker, block, 1)

# Both required embedded aliases use compact summary rows.
text = re.sub(
    r'"motif_transfer_chain_provenance"\s*:\s*provenance_sample\s*,',
    '"motif_transfer_chain_provenance": provenance_report_sample,',
    text,
    count=1,
)
text = re.sub(
    r'"motif_transfer_chain_provenance_sample"\s*:\s*provenance_sample\s*,',
    '"motif_transfer_chain_provenance_sample": provenance_report_sample,',
    text,
    count=1,
)

required = [
    "provenance_report_sample = [",
    '"motif_transfer_chain_provenance": provenance_report_sample',
    '"motif_transfer_chain_provenance_sample": provenance_report_sample',
]
for item in required:
    if item not in text:
        raise RuntimeError(f"Missing required patch: {item}")

compile(text, str(path), "exec")

backup = path.with_suffix(path.suffix + ".before_h11_fix_v10")
if not backup.exists():
    shutil.copy2(path, backup)

path.write_text(text, encoding="utf-8")
print(f"applied: {path}")
