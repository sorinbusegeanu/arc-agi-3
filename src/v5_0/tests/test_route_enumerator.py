from __future__ import annotations

from v5_0.route.trajectory_enumerator import enumerate_routes_between_points


def test_all_shortest_manhattan_interleavings_are_generated():
    routes = enumerate_routes_between_points((0.0, 0.0), (2.0, 1.0), max_extra_steps=0, max_routes=64)
    shortest = [r.actions for r in routes if r.length == 3]
    assert ("RIGHT", "RIGHT", "DOWN") in shortest
    assert ("RIGHT", "DOWN", "RIGHT") in shortest
    assert ("DOWN", "RIGHT", "RIGHT") in shortest


def test_bounded_enumerator_does_not_require_detours():
    routes = enumerate_routes_between_points((0.0, 0.0), (2.0, 1.0), max_extra_steps=2, max_routes=128)
    base_len = min(r.length for r in routes)
    assert all(r.length == base_len for r in routes)


def test_routes_sorted_shortest_to_longest():
    routes = enumerate_routes_between_points((0.0, 0.0), (3.0, 2.0), max_extra_steps=6, max_routes=64)
    lengths = [r.length for r in routes]
    assert lengths == sorted(lengths)


def test_duplicates_removed():
    routes = enumerate_routes_between_points((0.0, 0.0), (1.0, 1.0), max_extra_steps=6, max_routes=128)
    seqs = [r.actions for r in routes]
    assert len(seqs) == len(set(seqs))


def test_zero_distance_returns_one_empty_route():
    routes = enumerate_routes_between_points((5.0, 5.0), (5.0, 5.0), max_extra_steps=6, max_routes=64)
    assert len(routes) == 1
    assert routes[0].actions == tuple()
    assert routes[0].length == 0


def test_max_routes_respected():
    routes = enumerate_routes_between_points((0.0, 0.0), (4.0, 3.0), max_extra_steps=6, max_routes=5)
    assert len(routes) == 5
