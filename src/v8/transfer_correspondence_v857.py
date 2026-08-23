from __future__ import annotations

"""v8.57 formal structural transfer-correspondence generation.

The live peer generated scored SIMILAR_TO edges, while TransferValidator correctly
required a distinct TRANSFER_CORRESPONDENCE edge before any held-out transfer trial.
No live producer created that second edge, so automatic transfer experiments could
finish with attempted=0 even after structurally related worlds had been learned.

This layer closes only that missing structural-admissibility step:

    high-confidence cross-game SIMILAR_TO
        -> TRANSFER_CORRESPONDENCE
        -> existing TransferValidator candidate
        -> existing held-out memory-on/off experiment

The correspondence is structural evidence only.  It does not validate transfer,
promote M4, or bypass any empirical evidence gate.
"""

from dataclasses import replace


_INSTALLED = False
_BASE_PEER_SUBMIT = None
_BASE_RELATION_PACKET = None
_BASE_SIMILARITY_STATE_DICT = None
_BASE_SIMILARITY_LOAD_STATE = None

_CORRESPONDENCE_THRESHOLD = 0.80
_STATE_VERSION = 2


def _correspondence_allowed(peer, proposal) -> bool:
    from v8.model import MemoryLevel, RelationType

    if int(proposal.relation_type) != int(RelationType.SIMILAR_TO):
        return False
    if proposal.parent_uid.is_zero:
        return False
    if int(proposal.level) not in {int(MemoryLevel.M3), int(MemoryLevel.M4)}:
        return False
    if float(proposal.transfer_prior_sum) < _CORRESPONDENCE_THRESHOLD:
        return False

    source_games = frozenset(int(value) for value in peer.read_view.source_games(proposal.uid))
    target_games = frozenset(
        int(value) for value in peer.read_view.source_games(proposal.parent_uid)
    )
    if not source_games or not target_games:
        return False

    # Formal correspondence needs evidence from distinct formation scopes.  The
    # held-out target world is still selected and tested later by TransferValidator.
    return source_games != target_games


def _peer_submit_v857(self, proposal) -> None:
    from v8.model import RelationType, ValidationState

    _BASE_PEER_SUBMIT(self, proposal)
    if not _correspondence_allowed(self, proposal):
        return

    correspondence = replace(
        proposal,
        event_id=self._event_id(),
        watermark=int(self.current_watermark()),
        support_delta=0,
        relation_type=RelationType.TRANSFER_CORRESPONDENCE,
        validation_state=int(ValidationState.STRUCTURAL),
    )
    # Call the prior authority directly so this structural edge cannot recursively
    # manufacture another correspondence.
    _BASE_PEER_SUBMIT(self, correspondence)


def _relation_packet_v857(proposal):
    """Encode TRANSFER_CORRESPONDENCE as an edge-only canonical proposal."""
    from v8 import model

    packet = _BASE_RELATION_PACKET(proposal)
    if packet is not None:
        return packet
    if (
        proposal.relation_type != model.RelationType.TRANSFER_CORRESPONDENCE
        or proposal.parent_uid.is_zero
    ):
        return None

    source, target = sorted((proposal.uid, proposal.parent_uid))
    relation = model.RelationProposal(
        source_uid=source,
        target_uid=target,
        relation_type=model.RelationType.TRANSFER_CORRESPONDENCE,
        event_id=proposal.event_id,
        watermark=proposal.watermark,
        support_delta=max(1, int(proposal.support_delta)),
        score_sum=float(proposal.transfer_prior_sum),
        score_weight=1.0 if abs(float(proposal.transfer_prior_sum)) > 0.0 else 0.0,
        source_version=proposal.watermark,
        target_version=proposal.watermark,
    )
    return model.encode_relation_proposal(relation)


def _similarity_state_dict_v857(self) -> dict[str, object]:
    state = dict(_BASE_SIMILARITY_STATE_DICT(self))
    state["version"] = _STATE_VERSION
    return state


def _similarity_load_state_v857(self, state: dict[str, object] | None) -> None:
    if not state:
        _BASE_SIMILARITY_LOAD_STATE(self, state)
        return

    raw = dict(state)
    if int(raw.get("version", 1)) < _STATE_VERSION:
        # Existing snapshots may say every descriptor was already processed, but
        # those passes predate correspondence generation. Replay similarity once so
        # historical gp01/gp02-style evidence can produce the missing structural
        # edges without requiring new environment experience.
        self._processed_versions.clear()
        raw["processed_versions"] = []
    _BASE_SIMILARITY_LOAD_STATE(self, raw)


def install_transfer_correspondence_v857() -> None:
    global _INSTALLED, _BASE_PEER_SUBMIT, _BASE_RELATION_PACKET
    global _BASE_SIMILARITY_STATE_DICT, _BASE_SIMILARITY_LOAD_STATE
    if _INSTALLED:
        return

    from v8 import model
    from v8.peers import DevelopmentalPeerSupervisor
    from v8.similarity import BoundedNeighborhoodSimilarity

    _BASE_PEER_SUBMIT = DevelopmentalPeerSupervisor._submit
    DevelopmentalPeerSupervisor._submit = _peer_submit_v857

    _BASE_RELATION_PACKET = model._similarity_relation_packet
    model._similarity_relation_packet = _relation_packet_v857

    _BASE_SIMILARITY_STATE_DICT = BoundedNeighborhoodSimilarity.state_dict
    _BASE_SIMILARITY_LOAD_STATE = BoundedNeighborhoodSimilarity.load_state
    BoundedNeighborhoodSimilarity.state_dict = _similarity_state_dict_v857
    BoundedNeighborhoodSimilarity.load_state = _similarity_load_state_v857
    _INSTALLED = True
