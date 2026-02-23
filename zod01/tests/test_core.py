from __future__ import annotations

import unittest

from zod01.src.controller import Controller
from zod01.src.critic import Critic
from zod01.src.explorer import Explorer
from zod01.src.mechanic_inference import MechanicInference
from zod01.src.memory_episodic import EpisodicMemory
from zod01.src.planner import Planner
from zod01.src.safety import SafetyGuard
from zod01.src.state_abstract import canonicalize_state
from zod01.src.transition_diff import compute_delta
from zod01.src.types import CanonicalState, ControllerContext, NormalizedAction, ParsedObservation, TransitionDelta
from zod01.src.world_model import EmpiricalWorldModel


class CoreTests(unittest.TestCase):
    def _obs(self, v: int = 0) -> ParsedObservation:
        return ParsedObservation(
            game_id="test",
            state="NOT_FINISHED",
            levels_completed=0,
            win_levels=1,
            guid="g",
            full_reset=False,
            available_actions=("ACTION1", "ACTION2", "ACTION5"),
            frame_layers=(((v, v), (v, v)),),
        )

    def test_hash_deterministic(self) -> None:
        a = canonicalize_state(self._obs(1))
        b = canonicalize_state(self._obs(1))
        self.assertEqual(a.state_hash, b.state_hash)
        self.assertEqual(a.payload, b.payload)

    def test_diff_noop(self) -> None:
        s0 = canonicalize_state(self._obs(0))
        s1 = canonicalize_state(self._obs(0))
        d = compute_delta(s0, s1)
        self.assertTrue(d.no_op)
        self.assertEqual(d.changed_cells, 0)

    def test_planner_bfs_path(self) -> None:
        mem = EpisodicMemory()
        d = TransitionDelta(1, 4, 0.25, False, True, ())
        mem.add_transition("a", NormalizedAction("ACTION1"), "b", d, False)
        mem.add_transition("b", NormalizedAction("ACTION1"), "c", d, False)
        p = Planner(mem)
        self.assertEqual(p.plan_to_any_goal("a", {"c"}), ["a", "b", "c"])

    def test_controller_returns_available_action(self) -> None:
        mem = EpisodicMemory()
        model = EmpiricalWorldModel(mem)
        explorer = Explorer(mem, model)
        safety = SafetyGuard()
        critic = Critic()
        inference = MechanicInference()
        controller = Controller(safety, critic, inference)

        proposals = explorer.propose("s0", ("ACTION1", "ACTION5"))
        ctx = ControllerContext(step_idx=0, recent_hashes=(), available_actions=("ACTION1", "ACTION5"))
        out, _debug = controller.choose(ctx, None, proposals)
        self.assertIn(out.action.name, ctx.available_actions)


if __name__ == "__main__":
    unittest.main()
