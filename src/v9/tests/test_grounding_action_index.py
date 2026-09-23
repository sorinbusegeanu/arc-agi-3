from __future__ import annotations

from v9 import ContinuousMemoryRuntime
from v9.cognition.grounding import GroundingEvidence
from v9.environments.schemas import EnvironmentIdentity
from v9.memory.model import CanonicalNode, MemoryLevel, MemoryType
from v9.runtime.config import RuntimeConfig


class _NoItemsDict(dict):
    def items(self):
        raise AssertionError("grounded policy scoring attempted a full graph payload scan")


def test_grounded_policy_uses_incremental_action_index(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(
            tmp_path / "indexed",
            restore=False,
            enable_snapshots=False,
            enable_peers=False,
        )
    )
    try:
        environment = runtime.environments.register(EnvironmentIdentity("synthetic", "indexed-test", "cfg", "instance"))
        interaction = CanonicalNode.build(
            MemoryLevel.M1,
            MemoryType.GROUNDED_CONTINGENCY,
            (991, 17),
            1,
        )
        runtime._defer_base_group((
            (
                interaction,
                {
                    "environment_instance_id": int(environment.value),
                    "episode_id": 1,
                    "executable_action_token": 3,
                    "grounded_context_signature": 77,
                },
                (),
            ),
        ))
        runtime.grounding.observe(
            GroundingEvidence(
                123,
                int(interaction.uid.lo),
                int(environment.value),
                0,
                0,
                1,
                recurrent_symbol=True,
                cross_modal_association=True,
                prospective_prediction=True,
            )
        )
        runtime.graph.payloads = _NoItemsDict(runtime.graph.payloads)
        by_type, by_context = runtime._grounded_policy_scores()
        assert by_type["indexed-test"][3] > 0.0
        assert by_context["indexed-test"][77][3] > 0.0
        diagnostics = runtime.unified_telemetry.diagnostic_metrics()
        assert int(diagnostics["grounding_policy_full_graph_scan"]) == 0
        assert int(diagnostics["grounding_policy_states_scanned"]) == 1
    finally:
        runtime.close(normal=False)
