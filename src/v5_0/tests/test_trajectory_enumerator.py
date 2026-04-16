from __future__ import annotations

from v5_0.route.trajectory_enumerator import enumerate_routes_between_points


def test_all_shortest_manhattan_interleavings_generated():
    routes = enumerate_routes_between_points((0.0, 0.0), (2.0, 1.0), max_extra_steps=0, max_routes=32)
    shortest = [r.actions for r in routes if r.length == 3]
    assert len(shortest) == 3
    assert ("RIGHT", "RIGHT", "DOWN") in shortest
    assert ("RIGHT", "DOWN", "RIGHT") in shortest
    assert ("DOWN", "RIGHT", "RIGHT") in shortest


def test_routes_sorted_shortest_to_longest():
    routes = enumerate_routes_between_points((0.0, 0.0), (2.0, 1.0), max_extra_steps=2, max_routes=16)
    lengths = [r.length for r in routes]
    assert lengths == sorted(lengths)


def test_detours_appear_after_all_shortest_routes():
    routes = enumerate_routes_between_points((0.0, 0.0), (2.0, 1.0), max_extra_steps=2, max_routes=16)
    shortest_len = min(r.length for r in routes)
    shortest_count = sum(1 for r in routes if r.length == shortest_len)
    assert shortest_count >= 3
    assert all(r.length == shortest_len for r in routes[:shortest_count])
    assert all(r.length <= shortest_len + 2 for r in routes)


def test_duplicate_sequences_removed():
    routes = enumerate_routes_between_points((0.0, 0.0), (1.0, 1.0), max_extra_steps=2, max_routes=32)
    seqs = [r.actions for r in routes]
    assert len(seqs) == len(set(seqs))


def test_same_start_and_target_returns_one_empty_route():
    routes = enumerate_routes_between_points((5.0, 5.0), (5.0, 5.0), max_extra_steps=0, max_routes=32)
    assert len(routes) == 1
    assert routes[0].actions == tuple()
    assert routes[0].length == 0


def test_max_routes_limit_respected():
    routes = enumerate_routes_between_points((0.0, 0.0), (3.0, 3.0), max_extra_steps=2, max_routes=5)
    assert len(routes) == 5


def test_no_anti_target_first_step_without_hint():
    routes = enumerate_routes_between_points((0.0, 0.0), (3.0, 0.0), max_extra_steps=2, max_routes=32)
    assert all((not r.actions) or r.actions[0] != "LEFT" for r in routes)


def test_no_early_oscillation_routes():
    routes = enumerate_routes_between_points((0.0, 0.0), (2.0, 1.0), max_extra_steps=2, max_routes=32)
    for r in routes:
        if len(r.actions) >= 2:
            assert (r.actions[0], r.actions[1]) not in {("LEFT", "RIGHT"), ("RIGHT", "LEFT"), ("UP", "DOWN"), ("DOWN", "UP")}


def test_route_count_tightly_bounded():
    routes = enumerate_routes_between_points((0.0, 0.0), (6.0, 6.0), max_extra_steps=6, max_routes=64)
    assert len(routes) <= 32
