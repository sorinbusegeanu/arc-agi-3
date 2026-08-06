from __future__ import annotations

import re
from pathlib import Path

TARGET = Path("src/v6/future_options.py")


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")

    # Fix the future_option_events INSERT mismatch by regenerating the
    # placeholder list from the actual number of inserted columns.
    pattern = re.compile(
        r"(INSERT INTO future_option_events\s*\((?P<columns>.*?)\)\s*"
        r"VALUES\s*\()(?P<values>\?(?:\s*,\s*\?)*)(\))",
        re.DOTALL,
    )

    def fix_placeholders(match: re.Match[str]) -> str:
        columns = [
            item.strip()
            for item in match.group("columns").split(",")
            if item.strip()
        ]
        placeholders = ", ".join("?" for _ in columns)
        return (
            match.group(1)
            + placeholders
            + match.group(4)
        )

    text, insert_count = pattern.subn(
        fix_placeholders,
        text,
        count=1,
    )
    if insert_count != 1:
        raise RuntimeError(
            "Could not locate the future_option_events INSERT."
        )

    # The current code reads durable promotion fields from concept_candidates.
    # Rewrite those references to candidate-compatible aliases. Durable state
    # continues to be resolved through concept_promotion_state elsewhere.
    replacements = {
        "candidate.historically_promoted":
            "candidate.is_promoted",
        "candidate.currently_promoted":
            "candidate.is_promoted",
        "concept.historically_promoted":
            "concept.is_promoted",
        "concept.currently_promoted":
            "concept.is_promoted",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Handle unqualified SELECT-list references without touching schema or
    # concept_promotion_state columns.
    text = re.sub(
        r"(?<![\w.])historically_promoted(?!\s*=)",
        "is_promoted",
        text,
    )
    text = re.sub(
        r"(?<![\w.])currently_promoted(?!\s*=)",
        "is_promoted",
        text,
    )

    TARGET.write_text(text, encoding="utf-8")
    compile(text, str(TARGET), "exec")


if __name__ == "__main__":
    main()
