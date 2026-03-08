"""config.py — single source of truth for all tunable constants."""
from __future__ import annotations

N_RANDOM_EPISODES     = 100    # Phase 1 episode count
M_FOCUSED_EPISODES    = 50     # Phase 3 episode count
MAX_STEPS_PER_EP      = 150    # episode step budget

K_PROXIMITY_PX        = 16     # pixels: "close enough" to trigger consequence (was 8)
K_PROXIMITY_REACHABLE = 24     # pixels: was a POI ever approached? (reachability filter)
MAX_SPRITE_AREA       = 50     # grid cells: sprites are smaller than walls/floors
BG_THRESHOLD          = 0.05   # max color fraction to count as background (unused, bg derived from modal)
PIXEL_DIFF_THRESHOLD  = 0.05   # fraction of changed pixels = significant
HISTOGRAM_SHIFT_THR   = 0.30   # cosine distance = room change

MIN_BBOX_AREA         = 4      # ignore noise components smaller than this (pixels)
CLUSTER_CENTROID_DIST = 16.0   # max centroid distance (px) to merge bboxes into same cluster
HUD_FREQ_THRESH       = 0.90   # fraction of frames: bbox at same position → HUD
STATIC_VAR_THRESH     = 4.0    # centroid std (px): below = static POI
SECONDARY_BG_THRESH   = 0.15   # fraction of pixels: second color counts as bg
SPRITE_CORR_THRESH         = 0.3   # Pearson correlation threshold: motion → SELF (lowered)
MIN_MOTION_FRAMES          = 5     # min frames of motion data required to correlate
SELF_CORRELATION_THRESHOLD = 0.3   # alias used in SpriteDetector
MIN_MOVEMENT_STEPS         = 4     # minimum movement-action steps required for correlation

MIN_MOVEMENT_DIST     = 0.5    # minimum displacement (cells) for fallback centroid to count as moved

ALPHA_REWARD          = 0.5    # shaped reward weight toward POI
STUCK_STEPS           = 10     # steps with no progress before skipping POI target
MAX_VERSIONS          = 20     # max analysis cycles before BUDGET_EXHAUSTED

CONFIDENCE_NEW        = 0.5
CONFIDENCE_BIG        = 1.0
CONFIDENCE_NONE_DELTA = -0.3
STALE_VERSIONS          = 4    # visited POIs unseen for N versions → deprioritised (was 2)
STALE_VERSIONS_UNVISITED= 8   # unvisited POIs get more time before deprioritisation


def default_cfg() -> dict:
    """Return all config params as a flat dict for passing through the system."""
    return {
        "n_random_episodes":    N_RANDOM_EPISODES,
        "m_focused_episodes":   M_FOCUSED_EPISODES,
        "max_steps_per_ep":     MAX_STEPS_PER_EP,
        "k_proximity_px":           K_PROXIMITY_PX,
        "k_proximity_reachable":    K_PROXIMITY_REACHABLE,
        "max_sprite_area":          MAX_SPRITE_AREA,
        "pixel_diff_threshold": PIXEL_DIFF_THRESHOLD,
        "histogram_shift_thr":  HISTOGRAM_SHIFT_THR,
        "min_bbox_area":        MIN_BBOX_AREA,
        "cluster_centroid_dist":CLUSTER_CENTROID_DIST,
        "hud_freq_thresh":      HUD_FREQ_THRESH,
        "static_var_thresh":    STATIC_VAR_THRESH,
        "sprite_corr_thresh":          SPRITE_CORR_THRESH,
        "min_motion_frames":           MIN_MOTION_FRAMES,
        "self_correlation_threshold":  SELF_CORRELATION_THRESHOLD,
        "min_movement_steps":          MIN_MOVEMENT_STEPS,
        "min_movement_dist":    MIN_MOVEMENT_DIST,
        "alpha_reward":         ALPHA_REWARD,
        "stuck_steps":          STUCK_STEPS,
        "max_versions":         MAX_VERSIONS,
        "confidence_new":       CONFIDENCE_NEW,
        "confidence_big":       CONFIDENCE_BIG,
        "confidence_none_delta":CONFIDENCE_NONE_DELTA,
        "stale_versions":           STALE_VERSIONS,
        "stale_versions_unvisited": STALE_VERSIONS_UNVISITED,
    }
