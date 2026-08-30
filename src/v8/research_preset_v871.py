from __future__ import annotations

"""v8.71 research_1 packet-scope integration."""


_INSTALLED = False
_BASE_MIX_REQUESTED = None
_BASE_BUILD_PACKET = None


def _research_1_requested(argv) -> bool:
    values = tuple(str(value) for value in argv)
    for index, value in enumerate(values[:-1]):
        if value == "--games" and values[index + 1].strip().lower() == "research_1":
            return True
    return False


def _mixed_research_requested_v871(argv) -> bool:
    return bool(_BASE_MIX_REQUESTED(argv) or _research_1_requested(argv))


def _build_packet_v871(summary, *, revision, argv, h_report, reporting_cut, evidence_digest, log_tail):
    packet = _BASE_BUILD_PACKET(
        summary,
        revision=revision,
        argv=argv,
        h_report=h_report,
        reporting_cut=reporting_cut,
        evidence_digest=evidence_digest,
        log_tail=log_tail,
    )
    if _research_1_requested(argv):
        packet = packet.replace("ARC-only subset of mix", "ARC-only subset of research_1")
    return packet


def install_research_preset_v871() -> None:
    global _INSTALLED, _BASE_MIX_REQUESTED, _BASE_BUILD_PACKET
    if _INSTALLED:
        return
    from v8.research import researcher_packet

    _BASE_MIX_REQUESTED = researcher_packet._mix_requested
    researcher_packet._mix_requested = _mixed_research_requested_v871
    _BASE_BUILD_PACKET = researcher_packet.build_packet
    researcher_packet.build_packet = _build_packet_v871
    _INSTALLED = True
