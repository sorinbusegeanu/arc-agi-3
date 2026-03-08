"""run.py — CLI entry point for ccode_baseline_v2.

Usage:
    python -m ccode_baseline_v2.run --game <game_id> [options]

Or via uv:
    uv run src/ccode_baseline_v2/run.py --game <game_id>
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure src/ is on the path when run as a script
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Add other_repos paths if they exist (same pattern as run_rl.py)
_REPO_ROOT = os.path.dirname(_SRC_DIR)
for _extra in ("other_repos/arc-agi", "other_repos/ARCEngine"):
    _p = os.path.join(_REPO_ROOT, _extra)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from .config import default_cfg
from .analysis_loop import AnalysisLoop


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_env_factory(game_id: str, seed: int, op_mode: str):
    from arc_agi import Arcade, OperationMode

    arcade = Arcade(operation_mode=OperationMode(op_mode))

    def factory(ep_idx: int):
        env_seed = seed + ep_idx
        env = arcade.make(game_id, seed=env_seed)
        if env is None:
            raise RuntimeError(f"arcade.make failed for game_id={game_id}")
        return env, game_id, env_seed

    return factory


def main():
    parser = argparse.ArgumentParser(
        description="ccode_baseline_v2: perception + hypothesis-driven exploration for ARC-AGI-3"
    )
    parser.add_argument("--game",       required=True,  help="Game ID to run")
    parser.add_argument("--seed",       type=int,    default=0,       help="Base random seed")
    parser.add_argument("--op_mode",    default="offline",            help="Arcade operation mode")
    parser.add_argument("--store",      default=None,                 help="Path to existing HypothesisStore JSON to resume")
    parser.add_argument("--out_dir",    default="runs/ccode_v2",      help="Output directory")
    parser.add_argument("--no-clean",   action="store_true",          help="Keep previous run files (default: clean on start)")
    parser.add_argument("--n_random",   type=int,    default=None,    help="Override N_RANDOM_EPISODES")
    parser.add_argument("--m_focused",  type=int,    default=None,    help="Override M_FOCUSED_EPISODES")
    parser.add_argument("--max_steps",    type=int,  default=None,  help="Override MAX_STEPS_PER_EP")
    parser.add_argument("--max_versions", type=int,  default=None,  help="Override MAX_VERSIONS (loop budget)")
    args = parser.parse_args()

    cfg = default_cfg()
    if args.n_random    is not None: cfg["n_random_episodes"]  = args.n_random
    if args.m_focused   is not None: cfg["m_focused_episodes"] = args.m_focused
    if args.max_steps   is not None: cfg["max_steps_per_ep"]   = args.max_steps
    if args.max_versions is not None: cfg["max_versions"]       = args.max_versions

    import shutil
    out_dir = args.out_dir
    if not args.no_clean and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
        logger.info("cleaned out_dir=%s", out_dir)

    env_factory = _build_env_factory(args.game, args.seed, args.op_mode)

    loop = AnalysisLoop(
        env_factory=env_factory,
        cfg=cfg,
        seed=args.seed,
        store_path=args.store,
        out_dir=out_dir,
    )

    logger.info("Starting analysis loop: game=%s seed=%d out_dir=%s", args.game, args.seed, out_dir)
    exit_reason = loop.run()

    import json as _json
    summary_path = os.path.join(out_dir, "run_summary.json")
    print(f"\nExit reason : {exit_reason}")
    print(f"Out dir     : {out_dir}")
    if os.path.isfile(summary_path):
        s = _json.load(open(summary_path))
        f = s.get("final", {})
        print(f"Versions    : {s['versions_run']}  elapsed: {s['elapsed_sec']}s")
        print(f"POIs        : total={f.get('total_pois',0)}  reachable={f.get('reachable',0)}  visited={f.get('visited',0)}  deprioritised={f.get('deprioritised',0)}")
        print(f"Tags        : {f.get('tags',{})}")
        print(f"Consequences: {f.get('consequences',{})}")
        print(f"Confidence  : mean={f.get('conf_mean',0)}  max={f.get('conf_max',0)}")
        print(f"Summary     : {summary_path}")


if __name__ == "__main__":
    main()
