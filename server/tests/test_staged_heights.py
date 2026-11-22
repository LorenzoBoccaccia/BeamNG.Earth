"""Stage 5 of the staged solver (server/staged_heights.py): the banded solve, the heights of small systems against the kernel's, the
contradiction that has none, the switch that routes the shapes' heights through it."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import fast  # noqa: E402
import gradeshapes as gs  # noqa: E402
import staged  # noqa: E402
import staged_feasible as s4  # noqa: E402
import staged_heights as h5  # noqa: E402
import staged_places as places  # noqa: E402

Member = gs.Member


def line(x0, y0, x1, y1, n=20):
    return np.stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)], axis=1)


def road(key, ln, kind="ground", cls="secondary", level=0.0, half=4.0, kmh=60):
    return Member(key, ln, kind, cls, kmh, level, half, None)


def overpass():
    return [road(("mway", "ground", "m"), line(-400, 0, 400, 0, 200), cls="motorway", kmh=120, half=12.0),
            road(("deck", "deck", 0, 0), line(0, -60, 0, 60, 31), "deck", "secondary", 1.0, half=5.0),
            road(("south", "ground", "s"), line(0, -600, 0, -60, 100), half=5.0),
            road(("north", "ground", "n"), line(0, 60, 0, 600, 100), half=5.0)]


def placed(members, ground=None):
    pl = places.synthetic(members, ground=ground)
    return staged.run(pl.members, pl.meta, None, None, pl.ground, feasibility=False)


def test_the_sparse_ldl_solves_the_dense_system_and_again_after_new_values():
    rng = np.random.default_rng(5)
    n = 60
    M = np.zeros((n, n))
    for i in range(n):
        M[i, i] = rng.uniform(6.0, 9.0)
        for j in (i + 1, i + 2, (i * 7 + 3) % n):         # (bands, and a few entries off them: the rows between pieces)
            if i < j < n:
                M[i, j] = M[j, i] = rng.uniform(-1.0, 1.0)
    up = np.triu(M)
    ptr = np.r_[0, np.cumsum([int(np.count_nonzero(up[:, j])) for j in range(n)])]
    idx = np.concatenate([np.flatnonzero(up[:, j]) for j in range(n)])
    val = np.concatenate([up[np.flatnonzero(up[:, j]), j] for j in range(n)])
    f = fast.sparse_ldl(n, ptr, idx, val)
    b = rng.normal(size=n)
    x = f.solve(b)
    assert np.abs(M @ x - b).max() < 1e-12
    assert f.solve(b).tobytes() == x.tobytes()
    val2 = val * 1.5
    val2[ptr[1:] - 1] += 2.0                       # (the diagonal: the last entry of each column)
    f.refactor(val2)
    M2 = M * 1.5 + np.diag(np.full(n, 2.0))
    assert np.abs(M2 @ f.solve(b) - b).max() < 1e-12
    with pytest.raises(ValueError):                 # (an indefinite matrix is an error, never a regularised answer)
        f.refactor(-val)


def test_the_strongly_connected_pieces_of_a_graph():
    comp, n = h5._classes(6, np.array([0, 1, 2, 2, 3, 4]), np.array([1, 2, 0, 3, 4, 3]))
    assert n == 3 and comp[0] == comp[1] == comp[2] and comp[3] == comp[4] and len({comp[0], comp[3], comp[5]}) == 3


def _reduced(cons):
    import interchange_audit as ia
    dist, cyc = ia._spfa(cons.g)
    assert cyc is None
    z0 = np.array([dist[v] - dist[cons.g.zero] for v in range(cons.g.n - 1)])
    P = h5.Problem(cons)
    return P, z0, h5.Reduced(P, z0)


def test_the_interior_points_matrix_is_h_plus_g_transposed_d_g():
    S = placed(overpass())
    sysm = S.G.systems[0]
    pcs = gs.pieces(S.net, sysm)
    smp = gs.sample(S.net, pcs)
    gl = s4.ground_lists(S.ground, S.net, smp)
    cons = h5.constraints_of(S.net, sysm, pcs, smp, gl, S.ground.fixed, S.ground.at)
    P, z0, R = _reduced(cons)
    K = h5.Normal(R)
    n = R.n
    H = np.array([R.Hmul(e) for e in np.eye(n)]).T
    G = np.array([R.Gmul(e) for e in np.eye(n)]).T
    assert np.abs(H - H.T).max() < 1e-12 and np.abs(G.T @ np.ones(R.m) - R.Gt(np.ones(R.m))).max() < 1e-9
    d = np.random.default_rng(2).uniform(0.5, 20.0, R.m)
    M = H + G.T @ (d[:, None] * G)
    K.factor(d)
    b = np.random.default_rng(3).normal(size=n)
    assert np.abs(M @ K.solve(b) - b).max() < 1e-8 * max(1.0, np.abs(M).max())


def test_the_reduction_keeps_every_height_the_constraints_allow():
    """x = xbase + T y at y0 is the potentials."""
    S = placed(overpass())
    sysm = S.G.systems[0]
    pcs = gs.pieces(S.net, sysm)
    smp = gs.sample(S.net, pcs)
    gl = s4.ground_lists(S.ground, S.net, smp)
    cons = h5.constraints_of(S.net, sysm, pcs, smp, gl, S.ground.fixed, S.ground.at)
    P, z0, R = _reduced(cons)
    assert np.abs(R.full(R.y0, P.N) - z0).max() < 1e-9
    assert R.n <= P.N and R.m > 0


def test_an_overpass_gets_heights_that_hold_every_law_and_match_the_kernels():
    S = placed(overpass())
    sysm = S.G.systems[0]
    H = h5.solve_system(S.net, sysm, S.ground)
    st = H.stats
    assert st["feasible"] and st["converged"] and st["iterations"] < 60
    assert st["violation_m"] <= h5.VIOLATION_M and st["grade_excess"] <= 1e-3 and st["clearance_shortfall_m"] <= 1e-3
    pcs = gs.pieces(S.net, sysm)
    smp = gs.sample(S.net, pcs)
    gl = s4.ground_lists(S.ground, S.net, smp)
    zk = gs.solve_shape(S.net, sysm, pcs, smp, gl, S.ground.fixed, {})
    assert zk is not None
    assert max(float(np.abs(a - b).max()) for a, b in zip(H.zs, zk)) < 0.05      # (the same problem, the same answer)
    # the deck stands over the motorway by the clearance the law asks
    need = gs.clearance_need(S.net, sysm["crossings"][0]["u"], sysm["crossings"][0]["d"], sysm["crossings"][0]["q"])
    assert H.zs[0][len(H.zs[0]) // 2] >= need - 1e-3


def test_the_answer_is_the_same_bits_twice():
    a = h5.solve_system(placed(overpass()).net, placed(overpass()).G.systems[0], placed(overpass()).ground)
    b = h5.solve_system(placed(overpass()).net, placed(overpass()).G.systems[0], placed(overpass()).ground)
    assert all(x.tobytes() == y.tobytes() for x, y in zip(a.zs, b.zs)) and a.stats["iterations"] == b.stats["iterations"]


def test_a_system_with_a_contradiction_has_no_heights_and_says_so():
    def twisted(p1_level, q_level):
        return [Member(("P", "deck", 1), line(-100, 0, 0, 0, 30), "deck", "trunk_link", None, p1_level, 4.0, None),
                Member(("P", "deck", 2), line(0, 0, 100, 0, 30), "deck", "trunk_link", None, 3.0, 4.0, None),
                Member(("Q1", "deck", 1), np.array([(-19, -60), (-19, 0), (-19, 60)], np.float64), "deck", "trunk_link", None, q_level, 4.0, None),
                Member(("Q2", "deck", 1), np.array([(-19, 60), (15, 60), (15, 0), (15, -60)], np.float64), "deck", "trunk_link", None, q_level, 4.0, None)]
    S = placed(twisted(1.0, 2.0))
    H = h5.solve_system(S.net, S.G.systems[0], S.ground)
    assert H.zs is None and not H.stats["feasible"] and H.stats["short_m"] > 1.0
    S2 = placed(twisted(3.0, 2.0))
    H2 = h5.solve_system(S2.net, S2.G.systems[0], S2.ground)
    assert H2.zs is not None and H2.stats["feasible"]


def test_the_switch_routes_the_shapes_heights_through_stage_5(monkeypatch):
    S = placed(overpass())
    sysm = S.G.systems[0]
    pcs = gs.pieces(S.net, sysm)
    smp = gs.sample(S.net, pcs)
    gl = s4.ground_lists(S.ground, S.net, smp)
    monkeypatch.setenv("BNE_STAGED", "1")
    diag = {}
    zs = gs.solve_shape(S.net, sysm, pcs, smp, gl, S.ground.fixed, diag)
    assert zs is not None and diag.get("staged") and diag["converged"] and diag["iterations"] > 0
    monkeypatch.delenv("BNE_STAGED")
    diag2 = {}
    zk = gs.solve_shape(S.net, sysm, pcs, smp, gl, S.ground.fixed, diag2)
    assert not diag2.get("staged") and max(float(np.abs(a - b).max()) for a, b in zip(zs, zk)) < 0.05


def test_a_hilltop_ground_at_the_far_end_of_the_ramp_does_not_hold_the_deck_up():
    # the ramp road climbs a hill 85 m over its 600 m: the deck's end is held within what its embankment reaches of the ground
    # *somewhere* along the road, not of the ground 600 m up (which held it 47 m over its own ground)
    def hill(xy):
        xy = np.atleast_2d(xy)
        return np.clip(xy[:, 1] - 60.0, 0.0, None) * 0.14
    S = placed(overpass(), ground=hill)
    H = h5.solve_system(S.net, S.G.systems[0], S.ground)
    assert H.stats["feasible"] and H.stats["converged"]
    z = H.zs[0]
    assert abs(float(z[-1]) - float(hill(S.net.lines[S.G.systems[0]["seeds"][0]][-1:])[0])) < 12.0


def test_a_deck_with_no_weight_of_its_own_is_set_by_the_kernels_floor_on_every_weight():
    """Brunau's viaducts: a deck's middle weighs nothing and its line is kept by the stiffness alone, so what sets its height is the
    kernel's floor on every weight (carriageway_solver.rs W_MIN, 1e-3). With a tenth of it (1e-4 on top of the weights) the decks
    stood 8-10 m over the kernel's, whose answer is 1-2 m from the survey's and ours 4-8 m; with it the two agree to the tolerance."""
    try:
        pl = places.truth("zurich_brunau")
        S = staged.run(pl.members, pl.meta, pl.ll, pl.edge, pl.ground, name="brunau")
        sysm = S.G.systems[0]
        pcs = gs.pieces(S.net, sysm)
        smp = gs.sample(S.net, pcs)
        gl = s4.ground_lists(S.ground, S.net, smp)
        zk = gs.solve_shape(S.net, sysm, pcs, smp, gl, S.ground.fixed, {})
    except (OSError, ImportError):
        pytest.skip("no Brunau survey fixture here")
    if zk is None:
        pytest.skip("no kernel here")
    H = h5.solve_system(S.net, sysm, S.ground)
    assert H.stats["converged"]
    assert max(float(np.abs(a - b).max()) for a, b in zip(H.zs, zk)) < 0.1
    assert h5.W_MIN == 1e-3


def test_popolis_tubes_agree_with_the_kernel_curve_bound_and_all():
    """Popoli's five systems: the vertical-curve rows are the kernel's own (a tube there bends against them: 0.35 m over the bound
    without them, 15 m from the kernel's answer on two systems that a dual stop relative to the forces let go), and the stop on the
    dual residual in forces: with both, the heights agree to 1 cm."""
    try:
        pl = places.replay("popoli_ss17_underpass")
        S = staged.run(pl.members, pl.meta, pl.ll, pl.edge, pl.ground, name="popoli")
    except (OSError, ImportError, FileNotFoundError):
        pytest.skip("no Popoli replay here")
    worst = 0.0
    for n, sysm in enumerate(S.G.systems):
        pcs = gs.pieces(S.net, sysm)
        smp = gs.sample(S.net, pcs)
        gl = s4.ground_lists(S.ground, S.net, smp)
        zk = gs.solve_shape(S.net, sysm, pcs, smp, gl, S.ground.fixed, {})
        if zk is None:
            pytest.skip("no kernel here")
        H = h5.solve_system(S.net, sysm, S.ground)
        assert H.stats["converged"]
        worst = max(worst, max(float(np.abs(a - b).max()) for a, b in zip(H.zs, zk)))
        assert H.stats["curve_excess_m"] <= 2e-3
    assert worst < 0.01
