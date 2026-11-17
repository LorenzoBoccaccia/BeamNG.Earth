"""Grade-separation systems solved as a whole (BNE_SYSTEMS=1, off by default).

A system is every road and rail part that one set of grade separations ties together: the decks (road and rail
bridges) and short tunnels (underpasses), the parts that cross over or under them, and the parts that lead up to their
ends. Its heights are found together, in one problem, from the laws every part obeys:

- the ground: a part on the ground stays on it wherever it can (an L1 pull, so it sits exactly on the ground and leaves
  it only where it must); a deck has no ground; an underpass is cheap to dig down, dear to raise;
- its grade: no steeper than its class and speed allow (motorway 4-5 %, a street 10-12 %, rail 2.5 %), and as smooth
  as its speed asks; a track is stiff, so a road goes under a railway rather than the railway rising over it;
- its clearances: a deck over a road or track by the headroom that road or track needs (a signed limit where there is
  one), a road over an underpass by its cover; the parts meeting at a joint at one height.

A part that leads up to the system (an approach, a road crossing under a deck) is in it only for the length it needs
to reach the ground again at its own grade (LEAD_*): past that it is graded as before, so the system's edge is where
the earth embankment or cut meets the ground. Every limit is a penalty, not a bound, so a solve always has an answer:
where the laws cannot all hold, the cheapest one gives (a road dug a little deeper than its grade would like) instead
of the whole system falling back to the per-road rules.

Pure: members in, profiles out. grading.py gathers the members from its cells and applies the profiles.
"""
import hashlib
import math
from collections import namedtuple

import numpy as np

import dnp

STEP_M = 4.0                # m between a member's samples
JOIN_M = 2.5                # m: two members' ends (or an end and another's line) this close meet there
CROSS_END_M = 2.0           # m: a crossing this near either part's end is a joint, not a separation
DECK_THICK_M = 1.2          # m from a deck's road surface to its underside
COVER_M = 1.0               # m of cover over an underpass's roof
TUNNEL_MAX_M = 300.0        # m: a longer road tunnel is a tunnel network of its own (tunnels.py), not an underpass
LEAD_H_M = 7.5              # m: the rise a lead-up road may need (headroom + deck) ...
LEAD_MIN_M = 60.0           # ... over its own grade, between these lengths (m)
LEAD_MAX_M = 400.0
EDGE_BLEND_M = 30.0         # m over which a profile hands back to the plan at a system's edge

# headroom (m) a deck must leave over what runs under it
CLEAR_ROAD_M = 4.8
CLEAR_MINOR_M = 4.0
CLEAR_PATH_M = 2.7
CLEAR_RAIL_M = 6.0

W_GROUND = 1.0              # the ground pull (per sample, L1 by reweighting)
W_DIG = 0.03                # an underpass's pull up toward the ground (it may lie under it)
W_RAIL = 60.0               # a track's pull to its bed: stiff
W_KEEP = 40.0               # a road or track mapped on the ground under a deck, at the crossing: the map's evidence it is
                            # at ground level there (the deck rises; an underpass, mapped as such, is what digs)
W_EDGE = 200.0              # a lead-up's far end on its ground (the plan takes over there)
W_JOIN = 200.0              # members meeting at one height
W_GRADE = 400.0             # a grade over the law
W_CLEAR = 600.0             # a clearance short
ITERS = 30

Member = namedtuple("Member", "key line kind cls kmh level half limit")
# key: hashable, unique; line (k, 2) frame ENU; kind: "ground", "deck", "tunnel", "rail", "rail_deck";
# cls: road class (rail: "rail"); kmh: speed limit or None; level: OSM layer (decks); half: half width (m);
# limit: a signed height limit (m) under decks, or None

PATHS = {"footway", "path", "cycleway", "pedestrian", "steps", "bridleway", "track"}
MINOR = {"residential", "service", "unclassified", "living_street", "track", "road"}
DEFAULT_KMH = {"motorway": 110, "motorway_link": 60, "trunk": 90, "trunk_link": 50, "primary": 70, "primary_link": 40,
               "secondary": 60, "secondary_link": 40, "tertiary": 50, "tertiary_link": 30, "residential": 40,
               "unclassified": 50, "service": 20, "living_street": 15, "road": 40, "track": 20}


def grade_law(m):
    """(max grade, smoothness weight) of member m by its class and speed: the steeper the slower (design grades for a
    speed, AASHTO's ranges); rail 2.5 %, tram 6 %, paths 12 %."""
    if m.kind in ("rail", "rail_deck"):
        return (0.06, 20.0) if m.cls == "tram" else (0.025, 80.0)
    if m.cls in PATHS:
        return 0.12, 1.0
    kmh = m.kmh or DEFAULT_KMH.get(m.cls, 40)
    g = float(np.interp(kmh, [20, 30, 40, 50, 60, 80, 100, 120], [0.15, 0.12, 0.10, 0.09, 0.08, 0.06, 0.05, 0.04]))
    if m.cls and m.cls.endswith("_link"):
        g = max(g, 0.06)
    smooth = float(np.interp(kmh, [20, 60, 120], [1.0, 10.0, 60.0]))
    return g, smooth


RANK_W = {"motorway": 12.0, "trunk": 8.0, "primary": 4.0, "secondary": 2.5, "tertiary": 1.8}


def ground_weight(m):
    """How dear moving member m off its ground is, by its class: a motorway keeps its ground and the street over it
    rises (its link half that); a path gives way to everything."""
    if m.cls in PATHS:
        return 0.5
    c = m.cls or ""
    if c.endswith("_link"):
        return RANK_W.get(c[:-5], 1.0) / 2
    return RANK_W.get(c, 1.0)


def clearance(m):
    """Headroom (m) member m needs under a deck."""
    if m.limit is not None and m.limit > 0:
        return float(m.limit)
    if m.kind in ("rail", "rail_deck"):
        return CLEAR_RAIL_M
    if m.cls in PATHS:
        return CLEAR_PATH_M
    if m.cls in MINOR:
        return CLEAR_MINOR_M
    return CLEAR_ROAD_M


def lead_m(m):
    """How far member m is part of a system from where it meets it: the length to come back to the ground from
    LEAD_H_M at its own grade."""
    return float(np.clip(LEAD_H_M / grade_law(m)[0], LEAD_MIN_M, LEAD_MAX_M))


def _arclen(line):
    return np.r_[0.0, np.cumsum(dnp.hypot(*np.diff(line, axis=0).T))]


def _seg_hits(a, b):
    """Plan crossings of polylines a and b: [(i, j, ta, tb, q)] (segment i of a at fraction ta, j of b at tb)."""
    p, r = a[:-1], a[1:] - a[:-1]
    q, sv = b[:-1], b[1:] - b[:-1]
    lo_a, hi_a = np.minimum(a[:-1], a[1:]), np.maximum(a[:-1], a[1:])
    lo_b, hi_b = np.minimum(b[:-1], b[1:]), np.maximum(b[:-1], b[1:])
    ii, jj = np.nonzero(((lo_a[:, None, :] <= hi_b[None, :, :]) & (hi_a[:, None, :] >= lo_b[None, :, :])).all(2))
    if not len(ii):
        return []
    R, S, D = r[ii], sv[jj], q[jj] - p[ii]
    den = R[:, 0] * S[:, 1] - R[:, 1] * S[:, 0]
    ok = np.abs(den) > 1e-12
    den = np.where(ok, den, 1.0)
    ta = (D[:, 0] * S[:, 1] - D[:, 1] * S[:, 0]) / den
    tb = (D[:, 0] * R[:, 1] - D[:, 1] * R[:, 0]) / den
    hit = ok & (ta >= 0.0) & (ta <= 1.0) & (tb >= 0.0) & (tb <= 1.0)
    return [(int(ii[k]), int(jj[k]), float(ta[k]), float(tb[k]), p[ii[k]] + ta[k] * r[ii[k]]) for k in np.flatnonzero(hit)]


def _proj(line, s, q):
    """Arc length along line (its arc lengths s) of the point nearest q, and the distance."""
    a, b = line[:-1], line[1:]
    ab = b - a
    L2 = np.maximum((ab * ab).sum(1), 1e-12)
    t = np.clip(((q - a) * ab).sum(1) / L2, 0.0, 1.0)
    d = dnp.hypot(*(a + t[:, None] * ab - q).T)
    j = int(np.argmin(d))
    return float(s[j] + t[j] * (s[j + 1] - s[j])), float(d[j])


def _is_seed(m, length):
    return m.kind in ("deck", "rail_deck") or (m.kind == "tunnel" and length <= TUNNEL_MAX_M)


def _over_under(ma, mb):
    """Which of two crossing members is over: (upper, lower) as 'a'/'b', or None for an at-grade crossing."""
    da, db = ma.kind in ("deck", "rail_deck"), mb.kind in ("deck", "rail_deck")
    ta, tb = ma.kind == "tunnel", mb.kind == "tunnel"
    if da and db:
        if ma.level == mb.level:
            return None
        return ("a", "b") if ma.level > mb.level else ("b", "a")
    if da or tb:
        return "a", "b"
    if db or ta:
        return "b", "a"
    return None


def find_systems(members):
    """The systems among members: [{"members": {index: [(s0, s1), ...] the ranges in it}, "joins": [...],
    "seps": [...]}], each closed (every relation of its seeds included). Deterministic in the members' keys."""
    n = len(members)
    lines = [np.asarray(m.line, np.float64)[:, :2] for m in members]
    ss = [_arclen(l_) for l_ in lines]
    lens = [float(s[-1]) for s in ss]
    seeds = [i for i in range(n) if _is_seed(members[i], lens[i]) and len(lines[i]) >= 2]
    if not seeds:
        return []
    boxes = np.array([np.r_[l_.min(0), l_.max(0)] for l_ in lines])
    par = {i: i for i in seeds}

    def root(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    def unite(a, b):
        ra, rb = root(a), root(b)
        if ra != rb:
            par[max(ra, rb)] = min(ra, rb)
    rel = []                # (seed, other, kind, s on seed, s on other, q)
    pad = JOIN_M + 0.5
    for i in seeds:
        bi = boxes[i]
        near = np.flatnonzero((boxes[:, 0] <= bi[2] + pad) & (boxes[:, 2] >= bi[0] - pad) &
                              (boxes[:, 1] <= bi[3] + pad) & (boxes[:, 3] >= bi[1] - pad))
        for j in near:
            j = int(j)
            if j == i or len(lines[j]) < 2:
                continue
            # joints: an end of either on the other
            for e_line, e_s, o_line, o_s, swap in ((lines[i], ss[i], lines[j], ss[j], False),
                                                    (lines[j], ss[j], lines[i], ss[i], True)):
                for se in (0.0, float(e_s[-1])):
                    q = e_line[0] if se == 0.0 else e_line[-1]
                    so, d = _proj(o_line, o_s, q)
                    if d <= JOIN_M:
                        si, sj = (so, se) if swap else (se, so)
                        rel.append((i, j, "join", si, sj, q))
            for a_i, b_j, ta, tb, q in _seg_hits(lines[i], lines[j]):
                si = ss[i][a_i] + ta * (ss[i][a_i + 1] - ss[i][a_i])
                sj = ss[j][b_j] + tb * (ss[j][b_j + 1] - ss[j][b_j])
                if min(si, lens[i] - si, sj, lens[j] - sj) < CROSS_END_M:
                    continue
                rel.append((i, j, "cross", si, sj, q))
    # seeds related to one another, or through one member they both reach, are one system
    reach = {}
    for i, j, kind, si, sj, q in rel:
        if j in par:
            unite(i, j)
        reach.setdefault(j, []).append(i)
    for j, lst in reach.items():
        for i in lst[1:]:
            unite(lst[0], i)
    groups = {}
    for i in seeds:
        groups.setdefault(root(i), []).append(i)
    out = []
    for r, gseeds in sorted(groups.items(), key=lambda kv: str(members[kv[0]].key)):
        gs = set(gseeds)
        ranges = {i: [(0.0, lens[i])] for i in gseeds}
        joins, seps = [], []
        for i, j, kind, si, sj, q in rel:
            if i not in gs:
                continue
            if j not in par:            # a lead-up or a road under: the length it needs from where it meets
                L = lead_m(members[j])
                if kind == "join":
                    ranges.setdefault(j, []).append((0.0, min(L, lens[j])) if sj < lens[j] / 2
                                                    else (max(0.0, lens[j] - L), lens[j]))
                else:
                    ranges.setdefault(j, []).append((max(0.0, sj - L), min(lens[j], sj + L)))
            ou = _over_under(members[i], members[j]) if kind == "cross" else None
            if ou is None:                  # (a joint, or two parts on the ground crossing: a junction)
                joins.append((i, si, j, sj))
            else:                           # (upper index, its s, lower index, its s)
                seps.append((i, si, j, sj) if ou[0] == "a" else (j, sj, i, si))
        # members related to the system's lead-ups at their joints (a slip road off an approach) stay outside: the
        # junction snaps meet them to it afterwards
        merged = {}
        for k, rs in ranges.items():
            rs = sorted(rs)
            cur = [list(rs[0])]
            for a, b in rs[1:]:
                if a <= cur[-1][1] + STEP_M:
                    cur[-1][1] = max(cur[-1][1], b)
                else:
                    cur.append([a, b])
            merged[k] = [(float(a), float(b)) for a, b in cur]
        out.append({"members": merged, "joins": joins, "seps": seps})
    return out


def system_key(members, sysm):
    """A system's identity: its members' keys and ranges (the same from whichever cell found it)."""
    h = hashlib.sha1()
    for i in sorted(sysm["members"], key=lambda i: str(members[i].key)):
        h.update(repr((members[i].key, [(round(a, 1), round(b, 1)) for a, b in sysm["members"][i]])).encode())
    return h.hexdigest()[:20]


def sample(members, sysm):
    """The system's samples: list of (member index, range index, xy (m, 2), s (m,)) per range piece."""
    pieces = []
    for i in sorted(sysm["members"], key=lambda i: str(members[i].key)):
        line = np.asarray(members[i].line, np.float64)[:, :2]
        s = _arclen(line)
        for r, (a, b) in enumerate(sysm["members"][i]):
            k = max(2, int(math.ceil((b - a) / STEP_M)) + 1)
            sv = np.linspace(a, b, k)
            xy = np.stack([np.interp(sv, s, line[:, 0]), np.interp(sv, s, line[:, 1])], axis=1)
            pieces.append((i, r, xy, sv))
    return pieces


def solve(members, sysm, pieces, ground, edges):
    """Heights of the system's samples. pieces from sample(); ground: per piece, the ground height at its samples;
    edges: per piece, (start, end) True where that end is the system's edge (the member goes on outside it, graded by
    the plans there). Returns per piece z (m,). Least squares with every law a penalty (active-set and reweighting
    iterations): always an answer."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    off = np.cumsum([0] + [len(p[3]) for p in pieces])
    N = int(off[-1])
    if N == 0:
        return []
    where = {}
    for pi_, (i, r, xy, sv) in enumerate(pieces):
        where.setdefault(i, []).append(pi_)

    def at(i, s):
        """(global sample index, weight) pairs interpolating member i at arc length s, or None outside its pieces."""
        for pi_ in where.get(i, ()):
            sv = pieces[pi_][3]
            if sv[0] - 1e-6 <= s <= sv[-1] + 1e-6:
                k = int(np.clip(np.searchsorted(sv, s) - 1, 0, len(sv) - 2))
                t = float(np.clip((s - sv[k]) / max(sv[k + 1] - sv[k], 1e-9), 0.0, 1.0))
                return [(off[pi_] + k, 1.0 - t), (off[pi_] + k + 1, t)]
        return None
    t_all = np.concatenate([np.asarray(g, np.float64) for g in ground])
    # fixed rows: smoothness, joins
    rows, cols, vals, rhs = [], [], [], []

    def row(entries, b, w):
        rid = len(rhs)
        for c, v in entries:
            rows.append(rid)
            cols.append(c)
            vals.append(v * w)
        rhs.append(b * w)
    law = {}
    for pi_, (i, r, xy, sv) in enumerate(pieces):
        m = members[i]
        g, smooth = grade_law(m)
        law[pi_] = g
        o = off[pi_]
        ds = max(float(sv[1] - sv[0]), 1e-3) if len(sv) > 1 else STEP_M
        w2 = math.sqrt(smooth) * STEP_M / ds
        for k in range(1, len(sv) - 1):
            row([(o + k - 1, 1.0), (o + k, -2.0), (o + k + 1, 1.0)], 0.0, w2)
        st, en = edges[pi_]
        for k, flag in ((0, st), (len(sv) - 1, en)):
            if flag:
                row([(o + k, 1.0)], t_all[o + k], W_EDGE)
    for i, si, j, sj in sysm["joins"]:
        a, b = at(i, si), at(j, sj)
        if a and b:
            row(a + [(c, -v) for c, v in b], 0.0, W_JOIN)
    n_fixed = len(rhs)
    fixed = (list(rows), list(cols), list(vals), list(rhs))
    # ground pulls (reweighted) and the active laws
    kinds = np.empty(N, object)
    for pi_, (i, r, xy, sv) in enumerate(pieces):
        kinds[off[pi_]:off[pi_ + 1]] = members[i].kind
    z = t_all.copy()
    for pi_, (i, r, xy, sv) in enumerate(pieces):     # (decks start straight between their ends' ground)
        if members[i].kind == "rail_deck":       # (its height: the straight line between its ends' ground)
            o0, o1 = off[pi_], off[pi_ + 1]
            t_all[o0:o1] = np.linspace(t_all[o0], t_all[o1 - 1], o1 - o0)
        if members[i].kind in ("deck", "rail_deck"):
            o0, o1 = off[pi_], off[pi_ + 1]
            z[o0:o1] = np.linspace(t_all[o0], t_all[o1 - 1], o1 - o0)
    seps = []
    for u, su, l_, sl in sysm["seps"]:
        a, b = at(u, su), at(l_, sl)
        if a and b:
            mu, ml = members[u], members[l_]
            need = (clearance(ml) + DECK_THICK_M) if mu.kind in ("deck", "rail_deck") else (
                (clearance(ml) if ml.kind == "tunnel" else CLEAR_ROAD_M) + COVER_M)
            keep = mu.kind in ("deck", "rail_deck") and ml.kind in ("ground", "rail")
            seps.append((a, b, need, keep))
    is_deck = kinds == "deck"
    is_rail = np.isin(kinds, ["rail", "rail_deck"])         # (a rail deck: the railways' own, held where it is)
    is_tun = kinds == "tunnel"
    rank_w = np.empty(N)
    for pi_, (i, r, xy, sv) in enumerate(pieces):
        rank_w[off[pi_]:off[pi_ + 1]] = ground_weight(members[i])
    # per consecutive pair of samples within a piece: (first index, ds, its law)
    pa, pds, plaw = [], [], []
    for pi_ in range(len(pieces)):
        sv = pieces[pi_][3]
        pa.append(np.arange(off[pi_], off[pi_ + 1] - 1))
        pds.append(np.diff(sv))
        plaw.append(np.full(len(sv) - 1, law[pi_]))
    pa, pds, plaw = np.concatenate(pa), np.maximum(np.concatenate(pds), 1e-6), np.concatenate(plaw)
    Fr, Fc, Fv, Fb = (np.asarray(x, np.float64) for x in fixed)
    idx = np.arange(N)
    for it in range(ITERS):
        r_ = z - t_all
        w = np.where(is_deck, 1e-6, np.where(is_rail, W_RAIL ** 2, W_GROUND * rank_w / np.maximum(np.abs(r_), 0.05)))
        w = np.where(is_tun & (r_ < 0), w * W_DIG, w)
        for a_, b_, need, keep in seps:        # (the ground road under a deck, at the crossing: on its ground)
            if keep:
                for cc, v in b_:
                    w[cc] += v * W_KEEP * rank_w[cc] / max(abs(r_[cc]), 0.05)
        gr = (z[pa + 1] - z[pa]) / pds
        over = np.flatnonzero(np.abs(gr) > plaw)            # (the grade laws broken now: pulled back to the limit)
        sep_on = [c for c, (a_, b_, need, _k) in enumerate(seps)
                  if sum(z[cc] * v for cc, v in a_) - sum(z[cc] * v for cc, v in b_) < need]
        r0 = n_fixed
        R = [Fr, r0 + idx]
        C = [Fc, idx.astype(np.float64)]
        V = [Fv, np.sqrt(w)]
        B = [Fb, np.sqrt(w) * t_all]
        r0 += N
        if len(over):
            k = len(over)
            rr = r0 + np.arange(k)
            R += [rr, rr]
            C += [(pa[over] + 1).astype(np.float64), pa[over].astype(np.float64)]
            V += [np.full(k, W_GRADE), np.full(k, -W_GRADE)]
            B += [W_GRADE * np.sign(gr[over]) * plaw[over] * pds[over]]
            r0 += k
        for c in sep_on:
            a_, b_, need, _k = seps[c]
            for cc, v in a_:
                R.append(np.array([r0]))
                C.append(np.array([cc], np.float64))
                V.append(np.array([v * W_CLEAR]))
            for cc, v in b_:
                R.append(np.array([r0]))
                C.append(np.array([cc], np.float64))
                V.append(np.array([-v * W_CLEAR]))
            B.append(np.array([need * W_CLEAR]))
            r0 += 1
        A = sp.csr_matrix((np.concatenate(V), (np.concatenate(R).astype(np.int64), np.concatenate(C).astype(np.int64))),
                          shape=(r0, N))
        AtA = (A.T @ A).tocsc()
        z_new = spla.spsolve(AtA + sp.identity(N, format="csc") * 1e-9, A.T @ np.concatenate(B))
        moved = float(np.abs(z_new - z).max())
        z = z_new
        if moved < 0.005:
            break
    return [z[off[p]:off[p + 1]] for p in range(len(pieces))]
