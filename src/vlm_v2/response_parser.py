from __future__ import annotations

from typing import Any

from .action_schema import parse_action_letters
from .models import ObjectActionProposal, ObjectDescriptor, StartLevelAnalysis


def parse_start_level_analysis(text: str) -> StartLevelAnalysis:
    raw = str(text or "").strip()
    layout = ""
    player = ""
    reasoning = ""
    hud = ""
    objects: list[ObjectDescriptor] = []
    in_objects = False
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("layout:"):
            layout = _after_colon(line)
            in_objects = False
            continue
        if lower.startswith("player:"):
            player = _after_colon(line)
            in_objects = False
            continue
        if lower.startswith("reasoning:"):
            reasoning = _after_colon(line)
            in_objects = False
            continue
        if lower.startswith("hud:"):
            hud = _after_colon(line)
            in_objects = False
            continue
        if lower.startswith("objects{"):
            in_objects = True
            continue
        if in_objects:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                objects.append(
                    ObjectDescriptor(
                        name=parts[0],
                        color=parts[1],
                        position=",".join(parts[2:]).strip(),
                    )
                )
    return StartLevelAnalysis(
        layout=layout,
        player=player,
        reasoning=reasoning,
        objects=objects,
        hud=hud,
        raw_text=raw,
    )


def parse_object_action_proposals(text: str) -> list[ObjectActionProposal]:
    raw = str(text or "").strip()
    proposals: list[ObjectActionProposal] = []
    in_actions = False
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("actions{"):
            in_actions = True
            continue
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2:
            if not in_actions:
                continue
            continue
        object_name, seq_letters = parts
        letters = "".join(ch for ch in seq_letters.upper() if ch in {"L", "R", "U", "D"})
        if not object_name or not letters:
            continue
        in_actions = True
        proposals.append(
            ObjectActionProposal(
                object_name=object_name,
                sequence_letters=letters,
                actions=parse_action_letters(letters),
            )
        )
    return proposals


def _after_colon(line: str) -> str:
    _, _, tail = line.partition(":")
    return tail.strip()
