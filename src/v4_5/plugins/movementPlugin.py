from __future__ import annotations

from v4.movement.heuristics import admissible_target_distance_heuristic
from v4.movement.typedState import MovementCommonFieldsV4, MovementTypedStateV4
from v4_5.contracts import PlanCandidate, PlanCandidateSet, PlannerContext, SCHEMA_VERSION
from v4_5.plugins.base import PlannerPlugin


class MovementPlugin(PlannerPlugin):
    plugin_name = "movement"
    reused_modules = ("src/v4/movement/*",)

    def build_candidates(self, context: PlannerContext) -> PlanCandidateSet:
        scene = context.discovery_report.scene_summary if context.discovery_report is not None else None
        profile = context.game_control_profile
        if scene is None:
            return self._empty(context)
        raw = dict(scene.raw_observation_payload or {})
        avatar_cells = tuple(raw.get("avatar_cells", ()))
        if not avatar_cells and scene.avatar_position is not None:
            avatar_cells = ((int(round(scene.avatar_position[0])), int(round(scene.avatar_position[1]))),)
        poi_cells = tuple(raw.get("poi_cells", ()))
        frame_size = raw.get("frame_size") or (1, 1)
        available = tuple(getattr(profile, "movement_actions", ()))
        if not avatar_cells:
            return self._empty(context)
        avatar = avatar_cells[0]
        candidates = []
        for token, delta in (("UP", (0, -1)), ("DOWN", (0, 1)), ("LEFT", (-1, 0)), ("RIGHT", (1, 0))):
            if token not in available:
                continue
            next_pos = (avatar[0] + delta[0], avatar[1] + delta[1])
            score = 0.0
            verified = False
            if poi_cells:
                current_state = MovementTypedStateV4(
                    common=MovementCommonFieldsV4(
                        game_family=context.env_id.split("-", 1)[0],
                        game_id=context.env_id,
                        level_index=int(str(context.level_id).lstrip("L") or 0),
                        avatar_position=avatar,
                        traversable_cells=(avatar, next_pos),
                        current_legal_actions=tuple(int(item) for item in raw.get("available_action_ids", ())),
                        terminal_status="non_terminal",
                        step_depth=0,
                        static_bounds=(int(frame_size[0]), int(frame_size[1])),
                        target_cells=tuple(tuple(cell) for cell in poi_cells),
                    ),
                    layout_evidence_source="direct_observation",
                )
                next_state = MovementTypedStateV4(
                    common=MovementCommonFieldsV4(
                        game_family=context.env_id.split("-", 1)[0],
                        game_id=context.env_id,
                        level_index=int(str(context.level_id).lstrip("L") or 0),
                        avatar_position=next_pos,
                        traversable_cells=(avatar, next_pos),
                        current_legal_actions=tuple(int(item) for item in raw.get("available_action_ids", ())),
                        terminal_status="non_terminal",
                        step_depth=1,
                        static_bounds=(int(frame_size[0]), int(frame_size[1])),
                        target_cells=tuple(tuple(cell) for cell in poi_cells),
                    ),
                    layout_evidence_source="direct_observation",
                )
                before = admissible_target_distance_heuristic(current_state)
                after = admissible_target_distance_heuristic(next_state)
                score = float(before - after)
                verified = after <= before
            candidates.append(
                PlanCandidate(
                    schema_version=SCHEMA_VERSION,
                    agent_name="MovementPlugin",
                    round_id=context.round_id,
                    plugin_name=self.plugin_name,
                    candidate_id=token.lower(),
                    action_prefix=(token,),
                    score=score,
                    verified=verified,
                    rationale_codes=("LIVE_MOVEMENT_CANDIDATE",),
                )
            )
        return PlanCandidateSet(
            schema_version=SCHEMA_VERSION,
            agent_name="MovementPlugin",
            round_id=context.round_id,
            plugin_name=self.plugin_name,
            candidates=tuple(candidates),
            rationale_codes=("THIN_WRAPPER", "REUSES_V4_MOVEMENT"),
        )

    def _empty(self, context: PlannerContext) -> PlanCandidateSet:
        return PlanCandidateSet(
            schema_version=SCHEMA_VERSION,
            agent_name="MovementPlugin",
            round_id=context.round_id,
            plugin_name=self.plugin_name,
            candidates=(),
            rationale_codes=("THIN_WRAPPER", "REUSES_V4_MOVEMENT"),
        )
