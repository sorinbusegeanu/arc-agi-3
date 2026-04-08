from __future__ import annotations

from torch.nn import functional as F


def select_click_coordinates(click_head_module, planner_latent, spatial_features, valid_region_mask):
    logits = click_head_module(planner_latent, spatial_features, valid_region_mask)
    flat = logits.view(logits.shape[0], -1)
    index = int(flat.argmax(dim=-1).item())
    width = logits.shape[-1]
    y = index // width
    x = index % width
    return {
        "x": x,
        "y": y,
        "logprob": F.log_softmax(flat, dim=-1)[:, index],
        "logits": logits,
    }
