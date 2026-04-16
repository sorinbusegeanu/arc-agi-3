from __future__ import annotations

from v5_0.contact.policy_builder import build_contact_policies_for_poi


def _trajectory_features(actions: tuple[str, ...]) -> tuple[int, int, str | None]:
    dx = 0
    dy = 0
    first = actions[0] if actions else None
    for action in actions:
        token = str(action)
        if token == "LEFT":
            dx -= 1
        elif token == "RIGHT":
            dx += 1
        elif token == "UP":
            dy -= 1
        elif token == "DOWN":
            dy += 1
    return dx, dy, first


def dedupe_contact_trajectories(policies) -> tuple:
    ordered = tuple(policies or ())
    survivors: list = []
    for policy in ordered:
        planned = tuple(str(item) for item in tuple(getattr(policy, "planned_actions", ())))
        keep = True
        replace_index = None
        for index, existing in enumerate(tuple(survivors)):
            e_actions = tuple(str(item) for item in tuple(getattr(existing, "planned_actions", ())))
            same_seq = planned == e_actions
            p_dx, p_dy, p_first = _trajectory_features(planned)
            e_dx, e_dy, e_first = _trajectory_features(e_actions)
            same_disp_first_and_len = (p_dx, p_dy, p_first, len(planned)) == (e_dx, e_dy, e_first, len(e_actions))
            similar_move = same_disp_first_and_len
            if not (same_seq or similar_move):
                continue
            keep = False
            if len(planned) < len(e_actions):
                replace_index = index
            elif len(planned) == len(e_actions):
                if str(getattr(policy, "policy_id", "")) < str(getattr(existing, "policy_id", "")):
                    replace_index = index
            break
        if replace_index is not None:
            survivors[replace_index] = policy
            keep = False
        if keep:
            survivors.append(policy)
    survivors.sort(key=lambda item: (len(tuple(getattr(item, "planned_actions", ()))), str(getattr(item, "policy_id", ""))))
    return tuple(survivors)


def build_candidate_contact_trajectories_for_poi(
    *,
    selected_avatar,
    poi_candidate,
    transitions,
    episode_index: int,
):
    policies = build_contact_policies_for_poi(selected_avatar, poi_candidate, transitions, episode_index)
    if not policies:
        return tuple()
    deduped = dedupe_contact_trajectories(tuple(policies))
    return tuple(deduped)
