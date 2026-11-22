"""A railway's heights under the grade law the roads' carriageways are solved with.

A track is a line of parts (ground, bridge, tunnel) joined end to end; its bed is one height every STEP_M along it, from
the terrain under its ground parts and nothing under its structures, by the road laws' own kernel (carriageways.solve):

    minimise   sum w_i (z_i - t_i)^2
    such that  |z_{i+1} - z_i| <= grade * ds                       (railways.GRADE, a track's one grade)
               |z_{i-1} - 2 z_i + z_{i+1}| <= ds^2 / R              (its vertical curves: railways.RAIL_VERTICAL_RADIUS_M)
               z_i >= lo_i                                         (the clearance a deck has to keep over what it crosses)
               the parts' ends at a node equal, a siding's end on a track's line equal to its height there

t: the terrain along the track; w: 1 on the ground rising from 0 at a structure's end (the slope of its abutment is
not the track's), 0 under a bridge or a tunnel, ANCHOR_W at a loose end of the line.

The bed is no average of the terrain (it was, twice over SMOOTH_M and again over a hundred metres): a dip the track
cannot follow at its grade and its curve is not the track's height, it crosses it, and a deck's ends come from the track
this solve gives, not from the terrain's own smear under them (Compiegne's embankment, Mapterhorn at 2 m: 36.6 m where
the LiDAR HD says 40.4, a sag of 11 % in 30 m under a 7 m bridge). Pure given its inputs."""
import logging
import math

import numpy as np

import carriageways
import dnp

log = logging.getLogger("earthd.railways")

GROUND_W = 1.0              # weight of the terrain on the ground (relative: the solve has no other term)
ANCHOR_W = 5.0              # weight at a loose end of the line (a dead end, the window's edge: the terrain's own height)
TRUST_M = 40.0              # m from a structure's end over which the terrain's weight rises from 0 to GROUND_W
STEP_M = 2.0                # m between a part's samples (railways.STEP_M), for the report
JOIN_M = 2.0                # m: a part's end this near another part's line (not at its ends) is a siding's switch on it
LENGTH_BIAS = 1.0e-3        # the shortest line where nothing else decides (a deck's or a tunnel's interior has no terrain: the
                            # straight line between where the terrain's weight begins); too little to tilt a grade on the ground
ROUNDS = 4                  # fits of the terrain reweighted by 1 / |its miss|: a robust (median-like) fit, not a mean: a cutting's
                            # slope, a smeared embankment, a dip the track crosses are misses the track does not chase
MISS_FLOOR_M = 0.25         # m: a miss under this weighs as this (the reweighting's own floor, not a limit on any miss)
REPORT_MISS_M = 3.0         # m the bed may miss the terrain by before it is reported: a cutting or a fill this deep is a line
                            # the law does not let follow its ground (the report only: the bed is the law's)
MIN_SAMPLES = 4             # a chain of fewer samples than this has no curve to solve: its terrain


def chains_of(edges):
    """([[(part, reversed)]], {node: [(chain, 0 its first sample | 1 its last)]}): the parts (edges: (start node, end node,
    arc lengths) each, roads._chain_graph's) joined end to end through the nodes exactly two part ends meet at, and where
    the chains end. A ring of parts is one chain whose two ends are one node."""
    n = len(edges)
    at = {}
    for i, (a, b, _s) in enumerate(edges):
        at.setdefault(a, []).append((i, 0))
        at.setdefault(b, []).append((i, 1))

    def other(node, j):
        """(part, its end at node) of the one other part end at node, None where it is not exactly one (a branch, a loose
        end, a part ending on itself)."""
        here = at[node]
        if len(here) != 2:
            return None
        k, e = here[0] if here[0][0] != j else here[1]
        return None if k == j else (k, e)
    used = [False] * n
    out = []
    for i in range(n):
        if used[i]:
            continue
        used[i] = True
        seq = [(i, False)]
        while True:                                         # (on from its end: entered at its start, not reversed)
            j, rev = seq[-1]
            nxt = other(edges[j][0] if rev else edges[j][1], j)
            if nxt is None or used[nxt[0]]:
                break
            used[nxt[0]] = True
            seq.append((nxt[0], nxt[1] == 1))
        while True:                                         # (back from its start: left at its end, not reversed)
            j, rev = seq[0]
            nxt = other(edges[j][1] if rev else edges[j][0], j)
            if nxt is None or used[nxt[0]]:
                break
            used[nxt[0]] = True
            seq.insert(0, (nxt[0], nxt[1] == 0))
        out.append(seq)
    ends = {}
    for c, seq in enumerate(out):
        j, rev = seq[0]
        ends.setdefault(edges[j][1] if rev else edges[j][0], []).append((c, 0))
        j, rev = seq[-1]
        ends.setdefault(edges[j][0] if rev else edges[j][1], []).append((c, 1))
    return out, ends


def _across(t, structure, s, left=None, right=None):
    """t with each run of structure samples (a bridge's, a tunnel's) laid straight along s between the ground on either
    side: what is under a deck (a valley's floor) or over a tube (the hill) is not the track's height, and the kernel keeps
    a small weight even where there is none. A run at the chain's start or end is laid to left or right, the height of the
    node there (None: the ground's own, level)."""
    t = np.asarray(t, np.float64).copy()
    n = len(t)
    i = 0
    while i < n:
        if not structure[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and structure[j + 1]:
            j += 1
        lo = (s[i - 1], t[i - 1]) if i > 0 else (s[0], left) if left is not None else None
        hi = (s[j + 1], t[j + 1]) if j + 1 < n else (s[-1], right) if right is not None else None
        if lo is None and hi is None:
            pass
        elif lo is None or hi is None or hi[0] <= lo[0]:
            t[i:j + 1] = (lo or hi)[1]
        else:
            t[i:j + 1] = lo[1] + (hi[1] - lo[1]) * (s[i:j + 1] - lo[0]) / (hi[0] - lo[0])
        i = j + 1
    return t


class Track:
    """The chains of a set of parts and their solve. parts: [(xy (k, 2), structure bool)], edges: roads._chain_graph's
    (start node, end node, arc lengths) of each, targets: per part the terrain (k,) at its points."""

    def __init__(self, parts, edges, targets, radius, grade):
        self.n = len(parts)
        self.radius, self.grade = float(radius), float(grade)
        seqs, ends = chains_of(edges)
        self.chains = []
        self.where = [None] * self.n               # part -> (chain, first sample, count, reversed, dropped first point)
        for c, seq in enumerate(seqs):
            xs, ts, st = [], [], []
            for p, rev in seq:
                xy = np.asarray(parts[p][0], np.float64)
                t = np.asarray(targets[p], np.float64)
                if rev:
                    xy, t = xy[::-1], t[::-1]
                drop = 1 if xs else 0
                self.where[p] = (c, sum(len(x) for x in xs), len(xy) - drop, rev, drop)
                xs.append(xy[drop:])
                ts.append(t[drop:])
                st += [bool(parts[p][1])] * (len(xy) - drop)
            xy = np.concatenate(xs)
            s = np.r_[0.0, np.cumsum(dnp.hypot(*np.diff(xy, axis=0).T))]
            st = np.array(st, bool)
            cut = np.flatnonzero(st[1:] != st[:-1])
            marks = np.r_[s[cut], s[cut + 1]] if len(cut) else np.zeros(0)
            from_end = np.min(np.abs(s[:, None] - marks[None, :]), axis=1) if len(marks) else np.full(len(s), np.inf)
            w = carriageways.weights(st, from_end, TRUST_M) * GROUND_W
            self.chains.append(dict(xy=xy, s=s, t=np.concatenate(ts), w=w, structure=st, ds=np.diff(s)))
        self._laid_across(ends)
        for node, hits in ends.items():             # (a loose end of the line is held to the terrain there, and so is an end on
            for c, e in hits:                       # the ground where chains meet: not one inside a structure, a switch on a viaduct)
                if len(hits) == 1 or not self.chains[c]["structure"][-1 if e else 0]:
                    self.chains[c]["w"][-1 if e else 0] = ANCHOR_W
        self.couplings = []
        for node, hits in ends.items():             # (the chains that end at one node are one height there)
            for (a, ea), (b, eb) in zip(hits[:-1], hits[1:]):
                if a == b:
                    continue                            # (a ring's two ends: one chain, no coupling between its samples)
                self.couplings.append((a, -1 if ea else 0, b, -1 if eb else 0, "eq", 0.0))
        for na, ia, nb, j in carriageways.branch_ties([c["xy"] for c in self.chains], [c["structure"] for c in self.chains],
                                                       JOIN_M):
            self.couplings.append((na, ia, nb, j, "eq", 0.0))

    def _laid_across(self, ends):
        """Each run of structure is laid straight between the heights at its two ends (_across). The nodes inside a structure
        (a viaduct's switches: every chain that meets there ends on its deck, and the terrain there is what the deck
        crosses) have no height of their own and take the one that leaves the lines straight, the mean of their neighbours'
        weighted by the length of line between (a string's): the neighbours the ground at the far end of each run, a
        node on the ground (the terrain's), or a loose end (the terrain's too). A ground chain's own bridge ends in such
        a node as the viaduct's lines do, so a viaduct's switch and a bridge's end are one law."""
        height, fixed, link = {}, {}, []                    # node -> z; node -> [(z, length)]; [(node, node, length)]
        nodes = {}
        for node, hits in ends.items():
            for c, e in hits:
                nodes[(c, e)] = node
        for node, hits in ends.items():
            on = [self.chains[c]["t"][-1 if e else 0] for c, e in hits if not self.chains[c]["structure"][-1 if e else 0]]
            if on:
                height[node] = float(np.mean(on))
            elif len(hits) == 1:                            # (a loose end inside a structure: the terrain there)
                c, e = hits[0]
                height[node] = float(self.chains[c]["t"][-1 if e else 0])
        for (c, e), node in nodes.items():
            ch = self.chains[c]
            st = ch["structure"]
            if not st[-1 if e else 0] or node in height:
                continue
            ground = np.flatnonzero(~st)
            if not len(ground):
                if e == 0:                                  # (a run of structure alone, to its other end: once)
                    other = nodes[(c, 1)]
                    if other != node:
                        link.append((node, other, float(ch["s"][-1])))
                continue
            k = int(ground[0]) if e == 0 else int(ground[-1])
            d = float(abs(ch["s"][k] - ch["s"][-1 if e else 0]))
            fixed.setdefault(node, []).append((float(ch["t"][k]), max(d, 1.0)))
        unknown = sorted({n for n in nodes.values() if n not in height}, key=str)
        reach = {n for n in unknown if n in fixed}
        grew = True
        while grew:
            grew = False
            for u, v, _d in link:
                if (u in reach) != (v in reach) and u not in height and v not in height:
                    reach |= {u, v}
                    grew = True
        reach |= {n for u, v, _d in link for n in (u, v) if n not in height and (u in height or v in height)}
        unknown = [n for n in unknown if n in reach]
        if unknown:
            row = {node: i for i, node in enumerate(unknown)}
            m = np.zeros((len(unknown), len(unknown)))
            b = np.zeros(len(unknown))
            for node, hits in fixed.items():
                if node in row:
                    for z, d in hits:
                        m[row[node], row[node]] += 1.0 / d
                        b[row[node]] += z / d
            for u, v, d in link:
                g = 1.0 / max(d, 1.0)
                for x, y in ((u, v), (v, u)):
                    if x not in row:
                        continue
                    m[row[x], row[x]] += g
                    if y in row:
                        m[row[x], row[y]] -= g
                    elif y in height:
                        b[row[x]] += g * height[y]
            for node, z in zip(unknown, dnp.solve(m, b)):
                height[node] = float(z)
        for c, ch in enumerate(self.chains):
            left = height.get(nodes[(c, 0)]) if ch["structure"][0] else None
            right = height.get(nodes[(c, 1)]) if ch["structure"][-1] else None
            ch["t"] = _across(ch["t"], ch["structure"], ch["s"], left, right)

    def solve(self, lo=None):
        """Per part, its heights (k,): lo {part: (k,) lower bounds, -inf where none} the clearances kept."""
        specs = [np.full(len(ch["t"]), -np.inf) for ch in self.chains]
        for p, bound in (lo or {}).items():
            c, first, count, rev, drop = self.where[p]
            b = np.asarray(bound, np.float64)[::-1] if rev else np.asarray(bound, np.float64)
            specs[c][first:first + count] = np.maximum(specs[c][first:first + count], b[drop:])
            if drop:                                # (the joint with the part before is its first point too)
                specs[c][first - 1] = max(specs[c][first - 1], float(b[0]))
        solvable = [c for c, ch in enumerate(self.chains) if len(ch["t"]) >= MIN_SAMPLES]
        slot = {c: i for i, c in enumerate(solvable)}
        weights = [self.chains[c]["w"] for c in solvable]
        ties = [(slot[a], i if i >= 0 else len(self.chains[a]["t"]) + i, slot[b], j if j >= 0 else len(self.chains[b]["t"]) + j,
                 kind, v) for a, i, b, j, kind, v in self.couplings if a in slot and b in slot]
        info, zs = {}, []
        for _round in range(ROUNDS if solvable else 0):
            problem = [dict(t=self.chains[c]["t"], w=w, lo=specs[c], grade=self.grade, K=self.radius / 100.0,
                            length=LENGTH_BIAS, lam=0.0, ds=self.chains[c]["ds"]) for c, w in zip(solvable, weights)]
            zs = carriageways.solve(problem, ties, diagnostics=info)
            if not info.get("converged") or not info.get("feasible"):
                break
            weights = [self.chains[c]["w"] / np.maximum(np.abs(np.asarray(z) - self.chains[c]["t"]), MISS_FLOOR_M)
                       for c, z in zip(solvable, zs)]
        if solvable and (not info.get("converged") or not info.get("feasible")):
            log.warning("rail profile: %d chains of %d parts (%d m): %s, grade excess %.4f, curve excess %.4f, bounds %.3f m "
                        "short, clearance %.3f m short: a mapped line the grade %.3f and the radius %.0f m do not hold",
                        len(solvable), self.n, sum(self.chains[c]["s"][-1] for c in solvable), info.get("status"),
                        info.get("grade_excess", 0.0), info.get("curve_excess", 0.0), info.get("lower_shortfall_m", 0.0),
                        info.get("clearance_shortfall_m", 0.0), self.grade, self.radius)
        z = [np.asarray(ch["t"], np.float64).copy() for ch in self.chains]
        for c in solvable:
            z[c] = np.asarray(zs[slot[c]], np.float64)
            ch = self.chains[c]
            miss = np.where(ch["w"] > 0.0, np.abs(z[c] - ch["t"]), 0.0)
            if miss.max() > REPORT_MISS_M:
                k = int(np.argmax(miss))
                log.warning("rail profile: a line of %d m, the grade %.3f and the radius %.0f m cannot follow its ground: the bed "
                            "misses it by %.1f m at (%.0f, %.0f), %.0f m of it by more than %.1f m", ch["s"][-1], self.grade,
                            self.radius, miss[k], ch["xy"][k][0], ch["xy"][k][1], float((miss > REPORT_MISS_M).sum()) * STEP_M,
                            REPORT_MISS_M)
        out = []
        for p in range(self.n):
            c, first, count, rev, drop = self.where[p]
            zz = z[c][first - drop:first + count] if drop else z[c][first:first + count]
            out.append(zz[::-1].copy() if rev else zz.copy())
        return out
