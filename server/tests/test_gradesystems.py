"""gradesystems: grade-separation systems solved as a whole."""
import numpy as np

import gradesystems as gs
from gradesystems import Member


def _line(x0, y0, x1, y1, n=50):
    return np.stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)], axis=1)


def _run(members, ground_fn):
    out = {}
    for sysm in gs.find_systems(members):
        pieces = gs.sample(members, sysm)
        ground = [ground_fn(p[2]) for p in pieces]
        edges = []
        for i, r, xy, sv in pieces:
            L = gs._arclen(np.asarray(members[i].line, float))[-1]
            edges.append((sv[0] > 1e-6, sv[-1] < L - 1e-6))
        zs = gs.solve(members, sysm, pieces, ground, edges)
        for (i, r, xy, sv), z in zip(pieces, zs):
            out.setdefault(members[i].key, []).append((xy, sv, z))
    return out


def _z_at(prof, q):
    best = None
    for xy, sv, z in prof:
        d = np.hypot(*(xy - q).T)
        k = int(np.argmin(d))
        if best is None or d[k] < best[0]:
            best = (d[k], z[k])
    return best[1]


def test_a_road_goes_under_a_railway_rather_than_the_railway_rising():
    flat = lambda xy: np.zeros(len(xy))
    m = [Member("rail", _line(-600, 0, 600, 0, 200), "rail", "rail", None, 0, 2.0, None),
         Member("west", _line(0, -500, 0, -20), "ground", "secondary", 50, 0, 4.0, None),
         Member("under", _line(0, -20, 0, 20, 10), "tunnel", "secondary", 50, -1, 4.0, None),
         Member("east", _line(0, 20, 0, 500), "ground", "secondary", 50, 0, 4.0, None)]
    out = _run(m, flat)
    assert abs(_z_at(out["rail"], np.array([0.0, 0.0]))) < 0.3          # the track stays
    z_under = _z_at(out["under"], np.array([0.0, 0.0]))
    assert z_under <= -(gs.CLEAR_ROAD_M + gs.COVER_M) + 0.2             # the road dips under it
    for k in ("west", "east"):
        for xy, sv, z in out[k]:
            assert (np.abs(np.diff(z) / np.diff(sv)) <= 0.09 + 0.01).all()  # within its grade law
    # far from the crossing, the road is on its ground again
    assert abs(_z_at(out["west"], np.array([0.0, -400.0]))) < 0.2


def test_an_overpass_rises_over_a_motorway_and_its_approaches_are_embankments_at_their_grade():
    flat = lambda xy: np.zeros(len(xy))
    m = [Member("mway", _line(-800, 0, 800, 0, 300), "ground", "motorway", 120, 0, 12.0, None),
         Member("deck", _line(0, -30, 0, 30, 12), "deck", "secondary", 60, 1, 5.0, None),
         Member("south", _line(0, -500, 0, -30), "ground", "secondary", 60, 0, 5.0, None),
         Member("north", _line(0, 30, 0, 500), "ground", "secondary", 60, 0, 5.0, None)]
    out = _run(m, flat)
    assert abs(_z_at(out["mway"], np.array([0.0, 0.0]))) < 0.3          # the motorway keeps its ground
    zd = _z_at(out["deck"], np.array([0.0, 0.0]))
    assert zd >= gs.CLEAR_ROAD_M + gs.DECK_THICK_M - 0.2
    for k in ("south", "north"):
        for xy, sv, z in out[k]:
            g = np.abs(np.diff(z) / np.diff(sv))
            assert (g <= 0.08 + 0.01).all()
    # the embankment reaches the ground within the lead-up length
    assert abs(_z_at(out["south"], np.array([0.0, -30 - gs.lead_m(m[2]) + 5]))) < 0.3


def test_systems_are_found_the_same_from_any_order_of_members():
    flat = lambda xy: np.zeros(len(xy))
    m = [Member("mway", _line(-800, 0, 800, 0, 300), "ground", "motorway", 120, 0, 12.0, None),
         Member("deck", _line(0, -30, 0, 30, 12), "deck", "secondary", 60, 1, 5.0, None),
         Member("south", _line(0, -500, 0, -30), "ground", "secondary", 60, 0, 5.0, None)]
    a = gs.find_systems(m)
    b = gs.find_systems(m[::-1])
    ka = gs.system_key(m, a[0])
    kb = gs.system_key(m[::-1], b[0])
    assert ka == kb


def test_a_road_under_a_rail_bridge_dips_and_the_bridge_stays():
    flat = lambda xy: np.zeros(len(xy))
    m = [Member("rail_w", _line(-600, 0, -15, 0, 150), "rail", "rail", None, 0, 2.0, None),
         Member("rail_br", _line(-15, 0, 15, 0, 8), "rail_deck", "rail", None, 1, 2.0, None),
         Member("rail_e", _line(15, 0, 600, 0, 150), "rail", "rail", None, 0, 2.0, None),
         Member("road", _line(0, -500, 0, 500, 200), "ground", "tertiary", 50, 0, 4.0, None)]
    out = _run(m, flat)
    z = _z_at(out["road"], np.array([0.0, 0.0]))
    assert z <= -(gs.CLEAR_ROAD_M + gs.DECK_THICK_M) + 0.3
    for xy, sv, zz in out["road"]:
        assert (np.abs(np.diff(zz) / np.diff(sv)) <= 0.09 + 0.01).all()
