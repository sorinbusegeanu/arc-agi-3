from __future__ import annotations

from typing import Dict, List, Optional

from codex_baseline_v2.learning.ranking_inference import rank_mechanics, rank_options
from codex_baseline_v2.shared.learning_records import MechanicRankingRecordV1, OptionRankingRecordV1


class RankerActor:
    def __init__(self) -> None:
        self.option_state: Optional[OptionRankingRecordV1] = None
        self.mechanic_state: Optional[MechanicRankingRecordV1] = None

    def rank_options(self, *args, **kwargs) -> Dict[str, object]:
        record, score_map = rank_options(*args, **kwargs)
        self.option_state = record
        return {"ranking_record": record.to_dict() if record is not None else None, "score_map": score_map}

    def rank_mechanics(self, *args, **kwargs) -> Dict[str, object]:
        record, score_map = rank_mechanics(*args, **kwargs)
        self.mechanic_state = record
        return {"ranking_record": record.to_dict() if record is not None else None, "score_map": score_map}
