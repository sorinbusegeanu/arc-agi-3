#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def patch_h09(text: str) -> str:
    required = "Future-option derivation produced zero events despite available substrate."
    if required in text:
        return text

    patterns = [
        r'"Stable substrate evidence exists but no "\s*"future-option events were derived\."',
        r'"Stable substrate evidence exists but no future-option events were derived\."',
        r"'Stable substrate evidence exists but no '\s*'future-option events were derived\.'",
        r"'Stable substrate evidence exists but no future-option events were derived\.'",
    ]
    for pattern in patterns:
        new_text, count = re.subn(pattern, repr(required), text, count=1)
        if count == 1:
            return new_text

    raise RuntimeError("Could not locate H09 zero-event diagnostic")


def patch_h11(text: str) -> str:
    text = re.sub(
        r'"Future-option motifs exist, but motifs "\s*"are not linked to roles\."',
        repr("No future-option transfer links were produced because motifs lack role links."),
        text,
        count=1,
    )
    text = text.replace(
        "Future-option motifs exist, but motifs are not linked to roles.",
        "No future-option transfer links were produced because motifs lack role links.",
    )

    if '"motif_transfer_chain_provenance_sample":\n            provenance_sample,' not in text:
        anchor = '''        "motif_transfer_chain_provenance_sample_count":
            len(provenance_sample),
'''
        replacement = '''        "motif_transfer_chain_provenance_sample":
            provenance_sample,
        "motif_transfer_chain_provenance_sample_count":
            len(provenance_sample),
        "motif_transfer_chain_provenance_truncated":
            len(provenance_sample) < len(links),
'''
        if anchor not in text:
            raise RuntimeError("Could not locate H11 provenance sample metric anchor")
        text = text.replace(anchor, replacement, 1)

    if '"emergent_motifs_with_promoted_concept_count":' not in text:
        anchor = '''        "verified_emergent_motifs_with_promoted_concept_count":
            len(emergent_motifs_with_promoted),
'''
        replacement = anchor + '''        "emergent_motifs_with_promoted_concept_count":
            len(emergent_motifs_with_promoted),
'''
        if anchor not in text:
            raise RuntimeError("Could not locate H11 promoted-concept metric anchor")
        text = text.replace(anchor, replacement, 1)

    if 'compact["motif_transfer_chain_provenance_sample"] = []' not in text:
        anchor = '''        compact["provenance_sample"] = []
        compact["provenance_sample_truncated"] = True
'''
        replacement = anchor + '''        compact["motif_transfer_chain_provenance_sample"] = []
        compact["motif_transfer_chain_provenance_sample_count"] = 0
        compact["motif_transfer_chain_provenance_truncated"] = True
'''
        if anchor not in text:
            raise RuntimeError("Could not locate H11 report compaction anchor")
        text = text.replace(anchor, replacement, 1)

    text = re.sub(
        r'(?P<indent>\s*)"core_metrics": \{\},',
        lambda m: (
            f'{m.group("indent")}"motif_transfer_chain_provenance_sample": [],\n'
            f'{m.group("indent")}"motif_transfer_chain_provenance_sample_count": 0,\n'
            f'{m.group("indent")}"motif_transfer_chain_provenance_truncated": False,\n'
            f'{m.group("indent")}"emergent_motifs_with_promoted_concept_count": 0,\n'
            f'{m.group("indent")}"core_metrics": {{}},'
        ),
        text,
    )

    old = '''    if h11_blocked_by_h09:
        decision = "INSUFFICIENT_EVIDENCE"
        missing_evidence = [
            "H11 requires VALID H09 future-option motif evidence."
        ]
'''
    new = '''    if h11_blocked_by_h09:
        decision = "INSUFFICIENT_EVIDENCE"
        missing_evidence = [
            "H11 requires VALID H09 future-option motif evidence."
        ]
        if not emergent_links:
            missing_evidence.append(
                "No emergent future-option motifs with transfer evidence."
            )
        if (
            not links
            and int(
                derivation_summary.get("motifs_skipped_no_role_links") or 0
            )
            > 0
        ):
            missing_evidence.append(
                "No future-option transfer links were produced because motifs lack role links."
            )
'''
    if old in text:
        text = text.replace(old, new, 1)
    elif "No emergent future-option motifs with transfer evidence." not in text:
        raise RuntimeError("Could not locate H11 H09 dependency gate")

    return text


def patch_test_higher_order(text: str) -> str:
    old = '''    assert result["decision"] in {"INCONCLUSIVE", "PARTIALLY_VALID"}
    assert result["decision"] != "VALID"
'''
    new = '''    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
'''
    if old in text:
        return text.replace(old, new, 1)
    if 'assert result["decision"] == "INSUFFICIENT_EVIDENCE"' in text:
        return text
    raise RuntimeError("Could not locate stale H11 dependency test expectation")


def apply_file(repo: Path, relative: str, patcher, output_root: Path) -> None:
    source = repo / relative
    original = source.read_text(encoding="utf-8")
    patched = patcher(original)
    compile(patched, str(source), "exec")

    output = output_root / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(patched, encoding="utf-8")
    print(f"wrote complete replacement: {output}")

    if patched != original:
        backup = source.with_suffix(source.suffix + ".before_h09_h11_fix_v3")
        if not backup.exists():
            shutil.copy2(source, backup)
        source.write_text(patched, encoding="utf-8")
        print(f"applied: {source}")
    else:
        print(f"unchanged: {source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("complete_h09_h11_fixed_files_v3"),
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    output = args.output.resolve()

    apply_file(repo, "src/v6/hypothesis_h09_report.py", patch_h09, output)
    apply_file(repo, "src/v6/hypothesis_h11_report.py", patch_h11, output)
    apply_file(repo, "src/v6/tests/test_v6_higher_order.py", patch_test_higher_order, output)


if __name__ == "__main__":
    main()
