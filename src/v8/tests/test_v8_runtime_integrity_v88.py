from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import v8
from v8 import development
from v8 import normalized_memory_v086 as normalized
from v8 import within_action_temporal_v88 as temporal
from v8 import within_action_temporal_v88_integrity_fix as integrity
from v8.model import EventId, MemoryLevel, MemoryType, MemoryUid, RelationType
from v8.structural_events import NormalizedPrimitive, StructuralFact


class FreshProcessRuntimeStackTests(unittest.TestCase):
    def test_clean_import_installs_current_runtime_stack(self) -> None:
        src = Path(__file__).resolve().parents[2]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(src) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        code = """
import json
import v8
from v8 import adaptive_allocator_occupancy_v840 as v840
from v8 import restart_causal_progress_v844 as v844
from v8 import runtime_stack_v88 as stack
from v8 import within_action_temporal_v88 as temporal
from v8 import within_action_temporal_v88_integrity_fix as integrity
from v8.runtime_v82 import V82ContinuousMemoryRuntime
print(json.dumps({
    'stack': stack._INSTALLED,
    'v840': v840._INSTALLED,
    'v844': v844._INSTALLED,
    'temporal': temporal._INSTALLED,
    'integrity': integrity._INSTALLED,
    'version': getattr(V82ContinuousMemoryRuntime, 'within_action_temporal_version', None),
}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            text=True,
            capture_output=True,
            check=True,
            timeout=60,
        )
        row = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(
            row,
            {
                "stack": True,
                "v840": True,
                "v844": True,
                "temporal": True,
                "integrity": True,
                "version": "v8.8",
            },
        )


class TemporalEvidenceIntegrityTests(unittest.TestCase):
    @staticmethod
    def _macro_tokens(count: int) -> tuple[int, ...]:
        return tuple(
            StructuralFact(
                NormalizedPrimitive.COMPONENT_GEOMETRY_CHANGED,
                1000 + index,
                2000 + index,
                0,
                1,
            ).token
            for index in range(int(count))
        )

    @staticmethod
    def _event(*, family: int, facts: tuple[int, ...]):
        event = temporal.V88ExperienceEvent(
            EventId(1, 2),
            3,
            4,
            5,
            6,
            7,
            8,
            3,
            9,
            10,
            11,
            0.0,
            2,
            0,
            12,
            13,
            0.0,
            temporal_trace_signature=101,
            temporal_transition_count=3,
            temporal_family_signature=int(family),
            carrier_lineage_signature=303,
            temporal_prediction_error=0.0,
        )
        return normalized.V86PipelineEvent(
            event,
            MemoryUid.zero(),
            0,
            1,
            0,
            0,
            0,
            tuple(facts),
        )

    def test_temporal_merge_never_evicts_macro_facts(self) -> None:
        macro = self._macro_tokens(8)
        temporal_token = integrity.temporal_family_token_v88(777)
        merged = temporal._merge_temporal_facts(macro, (temporal_token,))
        self.assertEqual(merged, macro)

        partial = self._macro_tokens(6)
        merged = temporal._merge_temporal_facts(partial, (temporal_token,))
        self.assertEqual(merged[:6], partial)
        self.assertIn(temporal_token, merged)

    def test_temporal_m1_is_published_even_when_macro_fact_slots_are_full(self) -> None:
        macro = self._macro_tokens(8)
        pipeline = self._event(family=777, facts=macro)
        grounded = development.derive_proposal(MemoryLevel.M1, pipeline)
        rows = normalized.derive_normalized_proposals(pipeline, grounded)
        temporal_uid = MemoryUid.from_key(
            MemoryLevel.M1,
            MemoryType.CONTINGENCY,
            (integrity.temporal_family_token_v88(777),),
        )
        self.assertEqual(len(macro), 8)
        self.assertIn(temporal_uid, {row.uid for row in rows})
        self.assertGreaterEqual(len(rows), 9)

    def test_distinct_temporal_mechanisms_form_distinct_m2_families(self) -> None:
        def node(family: int):
            token = integrity.temporal_family_token_v88(family)
            return SimpleNamespace(
                uid=MemoryUid.from_key(
                    MemoryLevel.M1, MemoryType.CONTINGENCY, (token,)
                ),
                level=int(MemoryLevel.M1),
                memory_type=int(MemoryType.CONTINGENCY),
                key_parts=(token,),
                support_count=3,
                future_option_delta=0.0,
            )

        first, second = node(1001), node(2002)

        class Engine:
            min_contingency_support = 2
            min_family_members = 2
            min_family_compression = 0.0

            @staticmethod
            def _admissible(_row):
                return True

        rows = normalized._normalized_m2_candidates(Engine(), (first, second), limit=8)
        temporal_rows = [
            row
            for row in rows
            if (int(row.key_parts[0]) & 0xFF)
            == int(NormalizedPrimitive.AUTONOMOUS_CHANGE)
        ]
        self.assertEqual(len(temporal_rows), 2)
        self.assertNotEqual(temporal_rows[0].uid, temporal_rows[1].uid)


class SharedTemporalPredictionTests(unittest.TestCase):
    @staticmethod
    def _node(uid, key_parts, support=3):
        return SimpleNamespace(
            uid=uid,
            level=int(MemoryLevel.M1),
            memory_type=int(MemoryType.CONTINGENCY),
            key_parts=tuple(key_parts),
            support_count=int(support),
        )

    def test_prediction_reads_canonical_lineage_and_provenance(self) -> None:
        game = 9001
        context = 42
        action = 3
        family_expected = 123456
        family_other = 654321

        grounded_uid = MemoryUid.from_key(
            MemoryLevel.M1, MemoryType.CONTINGENCY, (context, action, 7, 8)
        )
        grounded = self._node(grounded_uid, (context, action, 7, 8), support=4)

        expected_token = integrity.temporal_family_token_v88(family_expected)
        other_token = integrity.temporal_family_token_v88(family_other)
        expected_uid = MemoryUid.from_key(
            MemoryLevel.M1, MemoryType.CONTINGENCY, (expected_token,)
        )
        other_uid = MemoryUid.from_key(
            MemoryLevel.M1, MemoryType.CONTINGENCY, (other_token,)
        )
        expected = self._node(expected_uid, (expected_token,), support=3)
        other = self._node(other_uid, (other_token,), support=1)

        def edge(source, relation, target, support):
            return SimpleNamespace(
                source_uid=source,
                relation_type=int(relation),
                target_uid=target,
                support_count=int(support),
            )

        edges = (
            edge(expected_uid, RelationType.EXPLAINS, grounded_uid, 3),
            edge(other_uid, RelationType.EXPLAINS, grounded_uid, 1),
            edge(grounded_uid, RelationType.GAME_PROVENANCE, MemoryUid(0, game), 4),
        )

        class View:
            _strategy_version = (10, 20)

            def _refresh_strategy_cache(self):
                return None

            def node_records(self, *, level=None):
                return (grounded, expected, other)

            def edge_records(self):
                return edges

        view = View()
        self.assertEqual(
            integrity.temporal_prediction_error_from_view_v88(
                view, game, context, action, family_expected
            ),
            0.0,
        )
        self.assertEqual(
            integrity.temporal_prediction_error_from_view_v88(
                view, game, context, action, family_other
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
