"""The room a way has to move: what else goes with it.

A way that goes down (a road dipped under a bridge) or up (an approach filled to a deck) does not do so alone. Its dip
or rise runs out at its grade, depth / grade m along it each side; where that run reaches a junction before it is
spent, the ways that meet there are at the same height there and have to go with it, each running out at its own grade,
and the junctions on those do the same in turn. A closed way is a roundabout's ring, level all round (grading's
RING_LEVEL_ON): whatever moves it moves it whole, and every arm on it goes as deep as the ring.
This is the cost findground's law weighs a move by besides the way's own run (findground.leaves, dip_extra and
fill_extra): every way that must move with it, by the earth it moves.

Measured where it matters: Compiegne's roundabout, a 7 m rail bridge over its ring and arms (tests/drivability.py): the
dip of one road of it, asked of a law that counted that road alone, scored 3.9 -> 41.4 (grades of 40 % on the arms).

Junctions are where two ways' vertices or segments come within JOIN_M of each other (the map's shared node; two ways
that cross with no node are one too, at grade). Pure."""
import numpy as np

import dnp

JOIN_M = 2.0                # m: ways closer than this to each other, at a vertex or between two segments, are one junction
MIN_DEPTH_M = 0.25          # m: a move smaller than this at a junction drags nothing (the roads' own surface is as flat)
MAX_LEVELS = 4              # junctions of junctions followed this deep (a dip of 6 m at 6 % runs 100 m: a few streets)
CELL_M = 25.0               # m: the grid the segments are indexed in


def _seg_pair(p, q, r, t):
    """(distance, point on pq, point on rt) of the closest approach of two segments (2-vectors): 0 where they cross."""
    d1, d2 = q - p, t - r
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) > 1e-12:
        w = r - p
        a = (w[0] * d2[1] - w[1] * d2[0]) / den
        b = (w[0] * d1[1] - w[1] * d1[0]) / den
        if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
            x = p + a * d1
            return 0.0, x, x
    best = (np.inf, p, r)
    for pt, a, b, other_is_k in ((p, r, t, True), (q, r, t, True), (r, p, q, False), (t, p, q, False)):
        d = b - a
        u = float(np.clip(np.dot(pt - a, d) / max(float(np.dot(d, d)), 1e-12), 0.0, 1.0))
        on = a + u * d
        dist = float(dnp.hypot(*(pt - on)))
        if dist < best[0]:
            best = (dist, pt, on) if other_is_k else (dist, on, pt)
    return best


class Ways:
    """Plain ways (lines (n, 2) in metres, each with the grade it dips and climbs at) and what meets what."""

    def __init__(self, lines, grades):
        self.lines = [np.asarray(ln, np.float64).reshape(-1, 2) for ln in lines]
        self.grades = [float(g) for g in grades]
        self.s = [np.r_[0.0, np.cumsum(dnp.hypot(*np.diff(ln, axis=0).T))] if len(ln) > 1 else np.zeros(1)
                  for ln in self.lines]
        self.L = [float(s[-1]) for s in self.s]
        self.closed = [len(ln) > 3 and float(dnp.hypot(*(ln[0] - ln[-1]))) < JOIN_M for ln in self.lines]
        self._grid = None
        self._met = {}

    def _index(self):
        if self._grid is None:
            grid = {}
            for k, ln in enumerate(self.lines):
                for j in range(len(ln) - 1):
                    lo = np.floor((np.minimum(ln[j], ln[j + 1]) - JOIN_M) / CELL_M).astype(int)
                    hi = np.floor((np.maximum(ln[j], ln[j + 1]) + JOIN_M) / CELL_M).astype(int)
                    for cx in range(lo[0], hi[0] + 1):
                        for cy in range(lo[1], hi[1] + 1):
                            grid.setdefault((cx, cy), []).append((k, j))
            self._grid = grid
        return self._grid

    def at_point(self, pt, tol=JOIN_M):
        """[(way, arc)] the ways that pass within tol of pt, where."""
        grid, out = self._index(), {}
        pt = np.asarray(pt, np.float64)
        for k, j in grid.get((int(np.floor(pt[0] / CELL_M)), int(np.floor(pt[1] / CELL_M))), ()):
            a, b = self.lines[k][j], self.lines[k][j + 1]
            d = b - a
            u = float(np.clip(np.dot(pt - a, d) / max(float(np.dot(d, d)), 1e-12), 0.0, 1.0))
            dist = float(dnp.hypot(*(pt - (a + u * d))))
            if dist <= tol and (k not in out or dist < out[k][0]):
                out[k] = (dist, float(self.s[k][j] + u * (self.s[k][j + 1] - self.s[k][j])))
        return [(k, v[1]) for k, v in out.items()]

    def _all(self, i):
        """[(arc on i, way, arc on it)] every other way that meets way i, along it, one per way and place (a junction's
        several vertices are one); made once per way."""
        got = self._met.get(i)
        if got is None:
            grid, ln, s = self._index(), self.lines[i], self.s[i]
            seen, found = set(), []
            for j in range(len(ln) - 1):
                p, q = ln[j], ln[j + 1]
                lo = np.floor((np.minimum(p, q) - JOIN_M) / CELL_M).astype(int)
                hi = np.floor((np.maximum(p, q) + JOIN_M) / CELL_M).astype(int)
                for cx in range(lo[0], hi[0] + 1):
                    for cy in range(lo[1], hi[1] + 1):
                        for k, m in grid.get((cx, cy), ()):
                            if k == i or (k, m, j) in seen:
                                continue
                            seen.add((k, m, j))
                            dist, on_i, on_k = _seg_pair(p, q, self.lines[k][m], self.lines[k][m + 1])
                            if dist > JOIN_M:
                                continue
                            found.append((float(s[j] + dnp.hypot(*(on_i - p))), k,
                                          float(self.s[k][m] + dnp.hypot(*(on_k - self.lines[k][m])))))
            found.sort()
            got = []
            for ai, k, ak in found:
                if not any(k == k2 and abs(ai - ai2) < 3.0 * JOIN_M for ai2, k2, _ in got):
                    got.append((ai, k, ak))
            self._met[i] = got
        return got

    def _junctions(self, i, a, b):
        """[(arc on i in [a, b], way, arc on it)] the other ways that meet way i between its arcs a and b."""
        return [t for t in self._all(i) if a - JOIN_M <= t[0] <= b + JOIN_M]

    def _room(self, k, s, d):
        """m of way k from arc s in direction d (+1, -1) to its end (a ring has no end: its length)."""
        if self.closed[k]:
            return self.L[k]
        return self.L[k] - s if d > 0 else s

    def _arms(self, i, lo, hi, s, depth, g, seen, level, flat=False):
        """m of the ways meeting way i between its arcs lo and hi that a move of depth m at arc s (running out at grade g,
        or not at all where flat: a ring) takes with it, each at its own grade, and what those take in turn. A way that
        meets i end to end with no third way there is i itself, cut at that node (the map's parts of one road): its run is
        the move's own, not an arm's, and the run goes on along it."""
        total = 0.0
        found = self._junctions(i, lo, hi)
        for ai, k, ak in found:
            dj = 0.0 if flat else abs(ai - s)
            if self.closed[i] and not flat:
                dj = min(dj, self.L[i] - dj)
            depth_j = depth - g * dj
            if depth_j < MIN_DEPTH_M:
                continue
            at_end = (ai < JOIN_M or ai > self.L[i] - JOIN_M) and not self.closed[i] and not self.closed[k]
            alone = sum(1 for aj, _k2, _a2 in found if abs(aj - ai) < 3.0 * JOIN_M) == 1
            if at_end and alone and (ak < JOIN_M or ak > self.L[k] - JOIN_M):
                dk = 1 if ak < JOIN_M else -1
                if (k, dk) not in seen:
                    seen.update(((k, dk), (k, -dk)))        # (its way back is the road's own)
                    if level < MAX_LEVELS:
                        total += self._walk(k, ak, depth_j, dk, self.grades[k], seen, level + 1)
                continue
            total += self._enter(k, ak, depth_j, seen, level)
        return total

    def _enter(self, k, ak, depth, seen, level):
        """m of way k (met at arc ak) and what meets it that a move of depth m there takes: a ring whole and its arms as
        deep as it is, any other way out to its own grade each side."""
        total = 0.0
        if self.closed[k]:
            if (k, 1) in seen:
                return 0.0
            seen.update(((k, 1), (k, -1)))
            total += self.L[k]
            if level < MAX_LEVELS:
                total += self._arms(k, 0.0, self.L[k], ak, depth, self.grades[k], seen, level + 1, True)
            return total
        gk = self.grades[k]
        for dk in (1, -1):
            if (k, dk) in seen:
                continue
            if (dk > 0 and ak > self.L[k] - JOIN_M) or (dk < 0 and ak < JOIN_M):
                continue                    # (the way ends here: nothing of it on that side)
            seen.add((k, dk))
            run = min(depth / gk, self._room(k, ak, dk))
            total += run
            if level < MAX_LEVELS:
                a, b = (ak, ak + run) if dk > 0 else (ak - run, ak)
                total += self._arms(k, a, b, ak, depth, gk, seen, level + 1)
        return total

    def _walk(self, i, s, depth, d, g, seen, level):
        run = min(depth / g, self._room(i, s, d))
        a, b = (s, s + run) if d > 0 else (s - run, s)
        spans = [(a, b)]
        if self.closed[i]:
            L = self.L[i]
            spans = [(max(a, 0.0), min(b, L))]
            if a < 0.0:
                spans.append((a + L, L))
            if b > L:
                spans.append((0.0, b - L))
        return sum(self._arms(i, lo, hi, s, depth, g, seen, level) for lo, hi in spans)

    def dragged(self, i, s, depth, directions=(-1, 1), grade=None):
        """m of the other ways that move with way i's move of depth m at its arc s, over the directions along it (+1 toward
        its end, -1 toward its start) it runs out in, at grade (its own where None): the ways it meets within the run
        (depth / grade) each go as deep as the move is there, and run out at their own grades, and so on (MAX_LEVELS).
        The way's own run is not counted, but a ring's whole length is: what its run does not reach of it goes too."""
        if depth < MIN_DEPTH_M:
            return 0.0
        g = float(grade or self.grades[i])
        seen = {(i, 1), (i, -1)}
        if self.closed[i]:                  # (a ring goes whole and level: its arms as deep as it, its run beyond its own)
            return max(0.0, self.L[i] - len(directions) * depth / g) + self._arms(i, 0.0, self.L[i], s, float(depth), g, seen, 0, True)
        return float(sum(self._walk(i, float(s), float(depth), d, g, seen, 0) for d in directions))
