#!/usr/bin/env python3
from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path.cwd()
H11 = ROOT / "src/v6/hypothesis_h11_report.py"
TEST = ROOT / "src/v6/tests/test_v6_higher_order.py"


def backup(path: Path) -> None:
    target = path.with_suffix(path.suffix + ".before_h11_fix_v5")
    if not target.exists():
        shutil.copy2(path, target)


def patch_h11(text: str) -> str:
    # Exact missing-role diagnostic, including split literals.
    text = re.sub(
        r'"Future-option motifs exist, but motifs are not linked to roles\."',
        '"No future-option transfer links were produced because motifs lack role links."',
        text,
    )
    text = re.sub(
        r'"Future-option motifs exist, but motifs "\s*"are not linked to roles\."',
        '"No future-option transfer links were produced because motifs lack role links."',
        text,
    )

    # Main result provenance sample fields.
    if '"motif_transfer_chain_provenance_sample": provenance_sample' not in text:
        pattern = re.compile(
            r'(?P<i>[ \t]*)"motif_transfer_chain_provenance_sample_count":\s*\n'
            r'(?P=i)[ \t]*len\(provenance_sample\),'
        )
        match = pattern.search(text)
        if not match:
            raise RuntimeError("Cannot find provenance sample count metric")
        indent = match.group("i")
        replacement = (
            f'{indent}"motif_transfer_chain_provenance_sample": provenance_sample,\n'
            f'{indent}"motif_transfer_chain_provenance_sample_count":\n'
            f'{indent}    len(provenance_sample),\n'
            f'{indent}"motif_transfer_chain_provenance_truncated":\n'
            f'{indent}    len(provenance_sample) < len(links),'
        )
        text = text[:match.start()] + replacement + text[match.end():]

    # Emergent promoted-concept alias.
    if '"emergent_motifs_with_promoted_concept_count":' not in text:
        pattern = re.compile(
            r'(?P<i>[ \t]*)"verified_emergent_motifs_with_promoted_concept_count":\s*\n'
            r'(?P=i)[ \t]*len\(emergent_motifs_with_promoted\),'
        )
        match = pattern.search(text)
        if not match:
            raise RuntimeError("Cannot find verified emergent promoted metric")
        indent = match.group("i")
        replacement = (
            match.group(0)
            + f'\n{indent}"emergent_motifs_with_promoted_concept_count":\n'
            + f'{indent}    len(emergent_motifs_with_promoted),'
        )
        text = text[:match.start()] + replacement + text[match.end():]

    # Preserve public fields when compacting the report.
    if 'compact["motif_transfer_chain_provenance_sample"]' not in text:
        anchor = (
            '        compact["provenance_sample"] = []\n'
            '        compact["provenance_sample_truncated"] = True\n'
        )
        if anchor not in text:
            raise RuntimeError("Cannot find report compaction block")
        text = text.replace(
            anchor,
            anchor
            + '        compact["motif_transfer_chain_provenance_sample"] = []\n'
            + '        compact["motif_transfer_chain_provenance_sample_count"] = 0\n'
            + '        compact["motif_transfer_chain_provenance_truncated"] = True\n',
            1,
        )

    # Add missing diagnostics after whichever decision block currently exists.
    diagnostic_marker = "    motif_count = len(\n"
    diagnostic_block = '''    if not emergent_links and not any(
        "emergent future-option motifs" in item
        for item in missing_evidence
    ):
        missing_evidence.append(
            "No emergent future-option motifs with transfer evidence."
        )
    if (
        not links
        and int(
            derivation_summary.get("motifs_skipped_no_role_links") or 0
        )
        > 0
        and (
            "No future-option transfer links were produced because motifs lack role links."
            not in missing_evidence
        )
    ):
        missing_evidence.append(
            "No future-option transfer links were produced because motifs lack role links."
        )

'''
    if "No emergent future-option motifs with transfer evidence." not in text:
        if diagnostic_marker not in text:
            raise RuntimeError("Cannot find H11 post-decision insertion point")
        text = text.replace(
            diagnostic_marker,
            diagnostic_block + diagnostic_marker,
            1,
        )

    # Stable keys in early-return payloads.
    for indent in ("            ", "                "):
        marker = indent + '"core_metrics": {},'
        replacement = (
            indent + '"motif_transfer_chain_provenance_sample": [],\n'
            + indent + '"motif_transfer_chain_provenance_sample_count": 0,\n'
            + indent + '"motif_transfer_chain_provenance_truncated": False,\n'
            + indent + '"emergent_motifs_with_promoted_concept_count": 0,\n'
            + marker
        )
        if marker in text and replacement not in text:
            text = text.replace(marker, replacement)

    required = [
        '"motif_transfer_chain_provenance_sample": provenance_sample',
        '"motif_transfer_chain_provenance_truncated":',
        '"emergent_motifs_with_promoted_concept_count":',
        "No future-option transfer links were produced because motifs lack role links.",
        "No emergent future-option motifs with transfer evidence.",
    ]
    for item in required:
        if item not in text:
            raise RuntimeError(f"Missing applied H11 change: {item}")

    return text


def patch_test(text: str) -> str:
    old = '''    assert result["decision"] in {"INCONCLUSIVE", "PARTIALLY_VALID"}
    assert result["decision"] != "VALID"
'''
    new = '''    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
'''
    if old in text:
        text = text.replace(old, new, 1)
    elif new not in text:
        raise RuntimeError("Cannot find test_h11_no_valid_without_h09 expectation")
    return text


def apply(path: Path, patcher) -> None:
    original = path.read_text(encoding="utf-8")
    patched = patcher(original)
    compile(patched, str(path), "exec")
    backup(path)
    path.write_text(patched, encoding="utf-8")
    print(f"applied: {path}")


apply(H11, patch_h11)
apply(TEST, patch_test)
print("done")
