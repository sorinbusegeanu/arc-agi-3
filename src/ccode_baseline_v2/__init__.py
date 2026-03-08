"""ccode_baseline_v2 — perception + hypothesis-driven exploration for ARC-AGI-3."""
from .structs import POIRecord, EpisodeRecord, HypothesisStoreState, ConsequenceResult
from .hypothesis_store import HypothesisStore
from .consequence_analyser import ConsequenceAnalyser
from .poi_detector import POIDetector, SpriteDetector
from .random_explorer import RandomExplorer
from .focused_explorer import FocusedExplorer, FrontierQueue
from .analysis_loop import AnalysisLoop
from .config import default_cfg

__all__ = [
    "POIRecord", "EpisodeRecord", "HypothesisStoreState", "ConsequenceResult",
    "HypothesisStore", "ConsequenceAnalyser",
    "POIDetector", "SpriteDetector",
    "RandomExplorer", "FocusedExplorer", "FrontierQueue",
    "AnalysisLoop", "default_cfg",
]
