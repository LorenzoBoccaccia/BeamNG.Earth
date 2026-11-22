"""Stage 5 of the staged grade solver: the heights of a system that has them (see staged_paths for the whole).

Input: a system stage 4 found feasible (its difference constraints: the grade law per step by the step's own run, clearances by
the order stage 2 gave, joints equal, the ground points stage 3 found held to the elevation, the floors and ceilings), and the
shapes' own objective (gradeshapes.shape_problem's chains: a target and weight per sample, the stiffness of a deck's line).
Output: a height per sample of every piece, the answer's residuals, and the iterations it took. Our own solver, deterministic
(plain floats, the same operations in the same order), no kernel, no time budget: an iteration cap, then it says what it reached.

The problem is the shapes' (carriageways.solve), the same one:

    minimise   sum_i max(w_i, W_MIN) (z_i - t_i)^2  +  sum_chains lam * sum (z_{i-1} - 2 z_i + z_{i+1})^2
    such that  lo_e <= z_b - z_a <= hi_e   for every edge e of the constraint graph   (grade, clearance, joint, overlap)
               lo_k <= z_k <= hi_k          for every node with a floor, a ceiling or a ground point held
               |r_i . z| <= hm^2 / (100 K)  for every window (i, i + 1, i + 2) of a piece   (the vertical curve, the kernel's row)

The curve rows are not difference constraints, which is what makes stage 4 exact: stage 4 says whether the grade law, the clearances
and the joints can all hold, not whether the curves can too (a system that has heights for them and none for the curve is reported
by stage 5 as not converged, with the curve rows' violation in stats["violation_by_kind"]). They are three entries wide and inside the
matrix's band, so they cost the factorisation no fill. W_MIN (the kernel's own floor on every weight,
native/earthfast carriageway_solver.rs) makes the objective strictly convex and is part of the shapes' answer, not a regulariser of
ours: a deck's middle weighs nothing and its line is kept by lam alone, so what sets its height is this floor's pull of every sample
to its natural line (a deck joined at both ends, no target). At Brunau a weight of 1e-4 here instead (what this solver first took) put
decks 8-10 m over the kernel's and the survey's. Measured against the kernel: Popoli's five systems agree to 4 mm and in objective to
1e-4, curve excess included; the kernel's answer breaks none of our rows (the same problem), ours none of its.

The method: the primal-dual interior-point method (Mehrotra's predictor-corrector, Nocedal and Wright chapter 16), the rows as
slack rows G y + s = b, s >= 0 with a multiplier lam >= 0 each, and the steps ours: a step is one exact factorisation of
H + G' diag(lam / s) G (a sparse LDL' by QDLDL on an AMD ordering: earthfast's SparseLdl, the pair Clarabel carries for the kernel,
deterministic: no threads, one order of operations) and two solves with it. It runs on the problem the zero-weight cycles of the
constraint graph leave (Reduced): the edges that are tight at stage 4's potentials and close a cycle (the strongly connected pieces
of the tight edges) fix their nodes' heights relative to each other whatever the objective (a joint's equality, a held point, a
box that has met its floor), so each such piece is one variable and an offset, the rows inside it are constants, and what remains
has an interior. Measured: Nanpu's system 0 (78 382 samples, 152 832 edges, 76 082 curve rows) reduces to 62 243 variables and
406 774 rows and takes 90 steps and 16 s (the kernel's 12 s). The first version of this stage was an ADMM, a first-order method whose
residuals fall by a fraction a sweep, so a tolerance a thousand times smaller costs a thousand times the sweeps whatever solves the
linear system: with conjugate gradients it took 658 s and stopped at its cap with 0.035 m of violation, and with this exact solve it
ran 50 000 sweeps in 1247 s and stopped at 0.015 m. The multipliers of the equalities, which the first-order iteration has to find
by sweeping, are what the Newton steps find at once. The start is Mehrotra's (section 16.6), infeasible; the primal residual
reaches rounding (1e-14 m).

Convergence: the objective is strictly convex (W_MIN) and the rows' set closed and convex with an interior once the pinned pieces
are contracted; stage 4 says it is not empty. What the code reports is the residuals it stopped on.

Stopping: the primal residual under PRIMAL_M, the dual one under DUAL_ABS (a force, not a share of the forces: the objective's flat
directions weigh W_MIN, so a dual residual r is a height error of up to r / (2 W_MIN) there; DUAL_ABS 1e-5 is 5 mm per sample, and the
rounding of the late steps keeps it from going under about 2e-6 at Nanpu) and the mean complementarity under GAP_ABS; or MAX_ITER
steps, or a factorisation that rounding took to a zero pivot. The constraint violation of the answer itself is measured, edge by
edge (stats["violation_m"]): where it is over VIOLATION_M the answer is not given as converged."""
import math

import numpy as np

import dnp
import fast
import gradeshapes as gs
import interchange_audit as ia
import staged_feasible as s4
import staged_ground as s3

VIOLATION_M = 2e-3      # m: an answer whose worst constraint is violated by more than this is not given as converged
MAX_ITER = 200          # interior-point steps at most (a count, never a clock)
PRIMAL_M = 1e-6         # m: no row is further than this from its bounds
DUAL_ABS = 1e-5         # the dual residual (a force)
GAP_ABS = 1e-8          # the mean complementarity
TIGHT_M = 1e-6          # m: an edge this near its potentials' difference is on a zero-weight cycle's candidates (stage 4's contradiction)
STEP_FRACTION = 0.99    # of the way to a slack or a multiplier reaching zero
W_MIN = 1e-3            # least weight of a sample on its target: the kernel's (carriageway_solver.rs W_MIN), the strictly convex part
CURVE_MIN_GAP_M = 0.5   # m: the least spacing a segment's own `ds` gives (carriageways.MIN_SEGMENT_M)
TRACE = None           # a list: ipm appends (step, primal residual, dual residual, mean complementarity) (a measurement's, nothing reads it)


class Problem:
    """A system's objective and constraints as arrays: chains (offsets), target, weight, stiffness; slabs per pair of nodes; a box per
    node. Built from interchange_audit.Constraints (the constraints stage 4 checked, the shapes' chains)."""

    def __init__(self, cons):
        g = cons.g
        self.cons = cons
        self.N = N = g.n - 1
        Z = g.zero
        self.off = list(cons.off)
        t = np.zeros(N)
        w = np.zeros(N)
        d = np.zeros(N)
        e = np.zeros(N)
        f = np.zeros(N)
        for c, ch in enumerate(cons.chains):
            a, b = self.off[c], self.off[c + 1]
            m = b - a
            t[a:b] = np.asarray(ch["t"], np.float64)
            w[a:b] = np.asarray(ch["w"], np.float64)
            lam = float(ch.get("lam", 0.0))
            if lam > 0.0 and m >= 3:
                # (the second differences' normal matrix: rows (1, -2, 1) at j, j+1, j+2)
                d[a:b - 2] += lam
                d[a + 1:b - 1] += 4.0 * lam
                d[a + 2:b] += lam
                e[a:b - 2] += -2.0 * lam
                e[a + 1:b - 1] += -2.0 * lam
                f[a:b - 2] += lam
        self.t = t
        self.wt = np.maximum(w, W_MIN)
        self.hd = 2.0 * (self.wt + d)       # (the Hessian's three bands: H = 2 (W + lam D2'D2))
        self.he = 2.0 * e
        self.hf = 2.0 * f
        self.h = 2.0 * self.wt * t
        lo = np.full(N, -np.inf)
        hi = np.full(N, np.inf)
        slab = {}
        for k in range(len(g.eu)):
            u, v, wk = g.eu[k], g.ev[k], g.ew[k]
            if u == Z:
                hi[v] = min(hi[v], wk)
            elif v == Z:
                lo[u] = max(lo[u], -wk)
            elif u < v:                     # z_v - z_u <= wk
                s = slab.setdefault((u, v), [-math.inf, math.inf])
                s[1] = min(s[1], wk)
            else:                           # z_v - z_u <= wk with v < u: z_u - z_v >= -wk
                s = slab.setdefault((v, u), [-math.inf, math.inf])
                s[0] = max(s[0], -wk)
        keys = sorted(slab)
        self.A = np.array([a for a, _b in keys], np.int64)
        self.B = np.array([b for _a, b in keys], np.int64)
        self.DLO = np.array([slab[k][0] for k in keys], np.float64)
        self.DHI = np.array([slab[k][1] for k in keys], np.float64)
        self.lo, self.hi = lo, hi
        # the vertical curve's bound, the kernel's own row of each window (carriageway_solver.rs curve_row): the change of grade across
        # (i, i + 1, i + 2) times the mean spacing, [hm / h0, -(hm / h0 + hm / h1), hm / h1] . z, within +- hm^2 / (100 K); the rows are
        # inside the pentadiagonal band, so they cost the banded factor nothing
        k0, ra, rc, cb = [], [], [], []
        for c, ch in enumerate(cons.chains):
            a, b = self.off[c], self.off[c + 1]
            if b - a < 3:
                continue
            ds = ch.get("ds")
            gap = np.maximum(np.asarray(ds, np.float64), CURVE_MIN_GAP_M) if ds is not None else np.full(b - a - 1, float(ch["step"]))
            h0, h1 = gap[:-1], gap[1:]
            hm = 0.5 * (h0 + h1)
            k0.append(np.arange(a, b - 2))
            ra.append(hm / h0)
            rc.append(hm / h1)
            cb.append(hm * hm / (float(ch["K"]) * 100.0))
        cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)
        self.K0 = cat(k0, np.int64)
        self.CA = cat(ra, np.float64)
        self.CC = cat(rc, np.float64)
        self.CB = cat(cb, np.float64)

    def objective(self, z):
        """The shapes' objective at z (the kernel's floor on the weights included): sum w (z - t)^2 + lam sum (second differences)^2."""
        val = float(np.sum(self.wt * dnp.sq(z - self.t)))
        for c, ch in enumerate(self.cons.chains):
            a, b = self.off[c], self.off[c + 1]
            lam = float(ch.get("lam", 0.0))
            if lam > 0.0 and b - a >= 3:
                zz = z[a:b]
                val += lam * float(np.sum(dnp.sq(zz[:-2] - 2.0 * zz[1:-1] + zz[2:])))
        return val

    def violation(self, z):
        """The most any edge or box is violated by z (m), and per kind of constraint: (worst, {kind: worst})."""
        g = self.cons.g
        Z = g.zero
        zz = np.append(z, 0.0)
        eu = np.asarray(g.eu, np.int64)
        ev = np.asarray(g.ev, np.int64)
        ew = np.asarray(g.ew, np.float64)
        over = zz[ev] - zz[eu] - ew
        kinds = {}
        for kind in sorted(set(g.kind)):
            sel = np.fromiter((k_ == kind for k_ in g.kind), bool, len(g.kind))
            kinds[kind] = float(max(over[sel].max(), 0.0)) if sel.any() else 0.0
        worst = float(max(over.max(), 0.0)) if len(over) else 0.0
        if len(self.K0):
            row = self.CA * z[self.K0] - (self.CA + self.CC) * z[self.K0 + 1] + self.CC * z[self.K0 + 2]
            kinds["curve"] = float(max((np.abs(row) - self.CB).max(), 0.0))
            worst = max(worst, kinds["curve"])
        return worst, kinds


class Heights:
    """Stage 5's answer for one system: zs (a height array per piece, as gradeshapes.solve_shape gives them), pieces, samples, stats."""

    def __init__(self):
        self.zs = None
        self.pcs = self.smp = None
        self.stats = {}


def _bandmul(d, e, f, v):
    """The symmetric pentadiagonal matrix (d, e, f) times v."""
    out = d * v
    out[:-1] += e[:-1] * v[1:]
    out[1:] += e[:-1] * v[:-1]
    out[:-2] += f[:-2] * v[2:]
    out[2:] += f[:-2] * v[:-2]
    return out


def _classes(n, eu, ev):
    """The strongly connected components of the directed graph on n nodes with the edges eu -> ev (Tarjan, iterative, in index
    order: the same labels everywhere): (label per node, number of components)."""
    adj = [[] for _ in range(n)]
    for u, v in zip(eu.tolist(), ev.tolist()):
        adj[u].append(v)
    index = [-1] * n
    low = [0] * n
    on = [False] * n
    comp = [-1] * n
    stack = []
    count = nc = 0
    for root in range(n):
        if index[root] != -1:
            continue
        index[root] = low[root] = count
        count += 1
        stack.append(root)
        on[root] = True
        work = [[root, 0]]
        while work:
            top = work[-1]
            v = top[0]
            if top[1] < len(adj[v]):
                w = adj[v][top[1]]
                top[1] += 1
                if index[w] == -1:
                    index[w] = low[w] = count
                    count += 1
                    stack.append(w)
                    on[w] = True
                    work.append([w, 0])
                elif on[w]:
                    low[v] = min(low[v], index[w])
            else:
                work.pop()
                if work:
                    u = work[-1][0]
                    low[u] = min(low[u], low[v])
                if low[v] == index[v]:
                    while True:
                        w = stack.pop()
                        on[w] = False
                        comp[w] = nc
                        if w == v:
                            break
                    nc += 1
    return np.array(comp, np.int64), nc


class Reduced:
    """The problem on what is free. A zero-weight cycle of the constraint graph (the edges that are tight at stage 4's potentials, and
    the strongly connected pieces they make) fixes the heights of its nodes relative to each other, whatever the objective: a
    joint's equality, a held point, a box that meets its floor. Those nodes are one variable and an offset (x = xbase + T y, T
    taking a node to its class's variable, the zero node's class no variable at all), the rows between a class's own nodes are
    constants and drop out, and what is left has an interior: every row a variable has can have slack. min 1/2 y'Hy + g'y such that
    G y <= b, the rows of G at most three entries wide (an edge's two, a box's one, a curve window's three)."""

    def __init__(self, P, z0):
        g = P.cons.g
        N = P.N
        assert g.zero == N
        eu = np.asarray(g.eu, np.int64)
        ev = np.asarray(g.ev, np.int64)
        ew = np.asarray(g.ew, np.float64)
        d = np.append(np.asarray(z0, np.float64), 0.0)
        tight = ew + d[eu] - d[ev] <= TIGHT_M
        comp, nc = _classes(N + 1, eu[tight], ev[tight])
        first = np.full(nc, N, np.int64)
        np.minimum.at(first, comp, np.arange(N + 1, dtype=np.int64))
        first[comp[N]] = N                      # (the zero node is its class's reference: xbase is its height there)
        self.xbase = d - d[first[comp]]
        free = np.ones(nc, bool)
        free[comp[N]] = False
        vid = np.cumsum(free) - 1
        vid[~free] = -1
        self.varof = vid[comp]                  # per node (the zero's last): its variable, -1 for a constant
        self.n = n = int(free.sum())
        self.y0 = d[first[free]]
        self.classes = nc
        xb, vo = self.xbase, self.varof
        self.hrow, self.hcol, self.hval = [], [], []
        idx = np.arange(N, dtype=np.int64)
        chain = np.searchsorted(np.array(P.off[1:], np.int64), idx, side="right")
        hv = np.zeros(n)

        def band(i, j, val):
            a, b = vo[i], vo[j]
            keep = (a >= 0) & (b >= 0)
            a, b, val = a[keep], b[keep], val[keep]
            same = a == b
            np.add.at(hv, a[same], 2.0 * val[same])
            lo = np.minimum(a[~same], b[~same])
            hi = np.maximum(a[~same], b[~same])
            self.hrow.append(lo)
            self.hcol.append(hi)
            self.hval.append(val[~same])
        keep = vo[idx] >= 0
        np.add.at(hv, vo[idx[keep]], P.hd[idx[keep]])
        i1 = idx[:-1][chain[:-1] == chain[1:]]
        band(i1, i1 + 1, P.he[i1])
        i2 = idx[:-2][chain[:-2] == chain[2:]]
        band(i2, i2 + 2, P.hf[i2])
        self.hdiag = hv
        self.hrow = np.concatenate(self.hrow) if self.hrow else np.zeros(0, np.int64)
        self.hcol = np.concatenate(self.hcol) if self.hcol else np.zeros(0, np.int64)
        self.hval = np.concatenate(self.hval) if self.hval else np.zeros(0)
        # the linear term of y: T'(H xbase - h)
        pull = _bandmul(P.hd, P.he, P.hf, xb[:N]) - P.h
        self.g = np.bincount(vo[idx[keep]], weights=pull[keep], minlength=n)
        # the rows
        cs, vs, lo, hi = [], [], [], []

        def rows(nodes, coefs, lower, upper):
            m = len(nodes[0])
            c = np.zeros((m, 3), np.int64)
            v = np.zeros((m, 3))
            shift = np.zeros(m)
            for j, nd in enumerate(nodes):
                cf = np.broadcast_to(np.asarray(coefs[j], np.float64), (m,))
                shift += cf * xb[nd]
                var = vo[nd]
                c[:, j] = np.where(var >= 0, var, 0)
                v[:, j] = np.where(var >= 0, cf, 0.0)
            live = v != 0.0
            ref = c[np.arange(m), np.argmax(live, axis=1)]
            same = np.all(~live | (c == ref[:, None]), axis=1)
            dead = ~live.any(axis=1) | (same & (np.abs(v.sum(axis=1)) < 1e-12))
            ok = ~dead
            cs.append(c[ok])
            vs.append(v[ok])
            lo.append((lower - shift)[ok])
            hi.append((upper - shift)[ok])
        rows([P.A, P.B], [-1.0, 1.0], P.DLO, P.DHI)
        boxed = np.flatnonzero(np.isfinite(P.lo) | np.isfinite(P.hi))
        rows([boxed], [1.0], P.lo[boxed], P.hi[boxed])
        if len(P.K0):
            rows([P.K0, P.K0 + 1, P.K0 + 2], [P.CA, -(P.CA + P.CC), P.CC], -P.CB, P.CB)
        c = np.concatenate(cs)
        v = np.concatenate(vs)
        lo = np.concatenate(lo)
        hi = np.concatenate(hi)
        up, dn = np.isfinite(hi), np.isfinite(lo)
        self.gc = np.concatenate([c[up], c[dn]])
        self.gv = np.concatenate([v[up], -v[dn]])
        self.gb = np.concatenate([hi[up], -lo[dn]])
        self.m = len(self.gb)
        # (the entries as contiguous arrays per position in a row, with the rows that have one: the products read these)
        self.ent = []
        for k in range(3):
            at = np.flatnonzero(self.gv[:, k] != 0.0)
            self.ent.append((at, np.ascontiguousarray(self.gc[at, k]), np.ascontiguousarray(self.gv[at, k])))

    def Hmul(self, y):
        out = self.hdiag * y
        out += np.bincount(self.hrow, weights=self.hval * y[self.hcol], minlength=self.n)
        out += np.bincount(self.hcol, weights=self.hval * y[self.hrow], minlength=self.n)
        return out

    def Gmul(self, y):
        out = np.zeros(self.m)
        for at, c, v in self.ent:
            out[at] += v * y[c]
        return out

    def Gt(self, w):
        out = np.zeros(self.n)
        for at, c, v in self.ent:
            out += np.bincount(c, weights=v * w[at], minlength=self.n)
        return out

    def full(self, y, N):
        """The heights of the N nodes for the variables y."""
        vo = self.varof[:N]
        return self.xbase[:N] + np.where(vo >= 0, y[np.where(vo >= 0, vo, 0)], 0.0)


class Normal:
    """H + G' diag(d) G as the upper triangle of a CSC matrix: its structure made once (the entries of H, then the pairs of a row of
    G, three entries wide) and its values for any d by one weighted sum, so the same bits for the same d; its factors:
    earthfast's SparseLdl (QDLDL on an AMD ordering, the kernel's own pair), made at the first d and redone on the same ordering
    at each new one."""

    def __init__(self, R):
        n, m = R.n, R.m
        rows, cols, coef, src = [], [], [], []

        def add(r, c, w, s):
            rows.append(np.asarray(r, np.int64))
            cols.append(np.asarray(c, np.int64))
            coef.append(np.asarray(w, np.float64))
            src.append(np.asarray(s, np.int64))
        idx = np.arange(n, dtype=np.int64)
        add(idx, idx, R.hdiag, np.full(n, m))
        add(R.hrow, R.hcol, R.hval, np.full(len(R.hrow), m))
        r = np.arange(m, dtype=np.int64)
        for k in range(3):
            for l in range(k, 3):
                w = R.gv[:, k] * R.gv[:, l] * np.where((k < l) & (R.gc[:, k] == R.gc[:, l]), 2.0, 1.0)
                live = w != 0.0
                a, b = R.gc[:, k][live], R.gc[:, l][live]
                add(np.minimum(a, b), np.maximum(a, b), w[live], r[live])
        key = np.concatenate(cols) * n + np.concatenate(rows)
        uniq, inv = np.unique(key, return_inverse=True)
        self.inv = inv.reshape(-1)
        self.coef = np.concatenate(coef)
        self.src = np.concatenate(src)
        self.indices = uniq % n
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(uniq // n, minlength=n))]).astype(np.int64)
        self.n = n
        self.ldl = None
        self.factorisations = 0

    def factor(self, dvec):
        vals = np.bincount(self.inv, weights=self.coef * np.append(dvec, 1.0)[self.src], minlength=len(self.indices))
        if self.ldl is None:
            self.ldl = fast.sparse_ldl(self.n, self.indptr, self.indices, vals)
        else:
            self.ldl.refactor(vals)
        self.factorisations += 1

    def solve(self, rhs):
        return self.ldl.solve(rhs)


def _step(s, ds, lam, dlam):
    """The longest step in (0, 1] that keeps s and lam positive."""
    a = 1.0
    neg = ds < 0.0
    if neg.any():
        a = min(a, float((-s[neg] / ds[neg]).min()))
    neg = dlam < 0.0
    if neg.any():
        a = min(a, float((-lam[neg] / dlam[neg]).min()))
    return a


def ipm(P, z0, max_iter=MAX_ITER):
    """The primal-dual interior-point method (Mehrotra's predictor-corrector, the textbook's: Nocedal and Wright, chapter 16, with
    the slacks of a quadratic program) on the problem Reduced makes of P: min 1/2 y'Hy + g'y such that Gy + s = b, s >= 0, with the
    multipliers lam >= 0 and s_i lam_i = sigma mu. A step is one factorisation of H + G' diag(lam / s) G and two solves with it,
    the affine direction and the corrected one, and the steps are ours. The answer to stats: the primal residual (the most a row is
    over), the dual one (a force), the mean complementarity. (z, stats)."""
    N = P.N
    R = Reduced(P, z0)
    n, m = R.n, R.m
    stats = dict(classes=R.classes, variables=n, rows=m)
    if n == 0:
        return R.full(np.zeros(0), N), dict(stats, iterations=0, factorisations=0, primal_residual_m=0.0, dual_residual=0.0,
                                            gap=0.0, converged=True)
    K = Normal(R)
    y = R.y0.copy()
    if m == 0:
        K.factor(np.zeros(0))
        y = K.solve(-R.g)
        return R.full(y, N), dict(stats, iterations=1, factorisations=1, primal_residual_m=0.0, dual_residual=0.0, gap=0.0,
                                  converged=True)
    # the starting point is Mehrotra's (Nocedal and Wright, section 16.6): the y with slack and multiplier equal (H + G'G y = -g - G'b),
    # both moved to be positive and then to be of the same size, so that no product s lam starts far from the mean
    K.factor(np.ones(m))
    y = K.solve(-R.g - R.Gt(R.gb))
    s = R.gb - R.Gmul(y)
    s = s + max(-1.5 * float(s.min()), 0.0)
    lam = s.copy()
    lift = 0.5 * float(np.dot(s, lam)) / float(s.sum())
    s = s + lift
    lam = lam + lift
    converged = False
    it = 0
    r_pri = r_dual = gap = math.inf
    while it < max_iter:
        rd = R.Hmul(y) + R.g + R.Gt(lam)
        rp = R.Gmul(y) + s - R.gb
        mu = float(np.dot(s, lam)) / m
        r_pri = float(np.abs(rp).max())
        r_dual = float(np.abs(rd).max())
        gap = mu
        if TRACE is not None:
            TRACE.append((it, r_pri, r_dual, gap))
        if r_pri <= PRIMAL_M and r_dual <= DUAL_ABS and gap <= GAP_ABS:
            converged = True
            break
        it += 1
        try:
            K.factor(lam / s)
        except ValueError as e:
            # (a pivot that rounding took to zero: the matrix's entries span 1e13 as the slacks close; what the iteration reached is
            # what it reports, not converged)
            stats["factor_error"] = str(e)
            break

        def direction(rc):
            dy = K.solve(-rd - R.Gt((lam * rp - rc) / s))
            ds = -rp - R.Gmul(dy)
            dl = (-rc - lam * ds) / s
            return dy, ds, dl
        dy, ds, dl = direction(s * lam)
        a = _step(s, ds, lam, dl)
        mu_aff = float(np.dot(s + a * ds, lam + a * dl)) / m
        r = mu_aff / mu
        sigma = r * r * r
        dy, ds, dl = direction(s * lam + ds * dl - sigma * mu)
        a = STEP_FRACTION * _step(s, ds, lam, dl)
        y = y + a * dy
        s = s + a * ds
        lam = lam + a * dl
    stats.update(iterations=it, factorisations=K.factorisations, primal_residual_m=r_pri, dual_residual=r_dual, gap=gap,
                 converged=converged, fill=int(K.ldl.nnz_l) if K.ldl is not None else 0)
    return R.full(y, N), stats


def constraints_of(net, sysm, pcs, smp, gl, fixed_z, at=None, spacing="true"):
    """The system's constraint graph and the shapes' chains: interchange_audit.Constraints with the points held on the ground
    (staged_feasible.holds_of). at(member, s) -> the elevation (None: the held points are left out)."""
    fixed_z = s3.anchored(net, fixed_z)         # (a street over a tunnel's footprint is not at the terrain there: staged_ground)
    if at is not None:
        at = s3.anchored(net, at)
    trace = {}
    problem = gs.shape_problem(net, sysm, pcs, smp, gl, fixed_z, trace) + (trace,)
    hs = []
    if at is not None:
        base = ia.Constraints(net, sysm, pcs, smp, gl, fixed_z, spacing, problem)
        ground = type("G", (), {"at": staticmethod(lambda m, s: at(m, s)), "fixed": staticmethod(fixed_z)})
        hs = s4.holds_of(net, sysm, pcs, smp, ground, base.node)
    return s4.Constraints(net, sysm, pcs, smp, gl, fixed_z, spacing, problem, hs)


def solve_constraints(cons):
    """Heights for constraints cons (staged_feasible.Constraints): Heights; stats["feasible"] False where the graph has a negative
    cycle (stage 4 names it): no heights then, zs None."""
    H = Heights()
    H.pcs, H.smp = cons.pcs, cons.smp
    g = cons.g
    dist, cyc = ia._spfa(g)
    if cyc is not None:
        w = math.fsum(g.ew[k] for k in cyc)
        if w < -ia.CONTRADICTION_M:
            H.stats = dict(feasible=False, converged=False, short_m=-w)
            return H
        dist = [0.0] * g.n
    z0 = np.array([dist[v] - dist[g.zero] for v in range(g.n - 1)], np.float64)
    P = Problem(cons)
    z, st = ipm(P, z0)
    viol, kinds = P.violation(z)
    H.zs = [z[P.off[c]:P.off[c + 1]].copy() for c in range(len(cons.chains))]
    st.update(feasible=True, nodes=P.N, edges=len(P.A), boxes=int(np.count_nonzero(np.isfinite(P.lo) | np.isfinite(P.hi))),
              objective=P.objective(z), objective_start=P.objective(np.clip(z0, P.lo, P.hi)), violation_m=viol, violation_by_kind=kinds)
    st["joint_gap_m"] = max([kinds.get("join", 0.0), kinds.get("overlap", 0.0)])
    st["converged"] = bool(st["converged"] and viol <= VIOLATION_M)
    st.update(measure(cons, H.zs))
    H.stats = st
    return H


def measure(cons, zs):
    """What the heights are, measured the kernel's way: the worst grade excess (rise over a step's run less the law, as a share of
    the law's rise, per cent), the worst vertical-curve excess (second difference over the kernel's bound, m), the worst clearance
    shortfall (m)."""
    grade = curve = 0.0
    for c, ch in enumerate(cons.chains):
        z = np.asarray(zs[c], np.float64)
        d = np.asarray(cons.smp[c]["d"], np.float64)
        if len(z) >= 2:
            dd = np.maximum(np.diff(d), gs_min_step())
            grade = max(grade, float((np.abs(np.diff(z)) / dd - ch["grade"]).max()))
        if len(z) >= 3:
            step = float(ch["step"])
            bound = step * step / (float(ch["K"]) * 100.0)
            curve = max(curve, float((np.abs(z[:-2] - 2.0 * z[1:-1] + z[2:]) - bound).max()))
    short = 0.0
    for cp in cons.cps:
        a, i, b, j, kind, val = cp[:6]
        if kind == "ge":
            short = max(short, float(val) - (float(zs[a][i]) - float(zs[b][j])))
    return dict(grade_excess=max(grade, 0.0), curve_excess_m=max(curve, 0.0), clearance_shortfall_m=max(short, 0.0))


def gs_min_step():
    return 0.5


def solve_system(net, sysm, ground, spacing="true"):
    """Stage 5 on one system of stage 3 (ground: staged_fixtures.Ground or alike: at, fixed, xy): Heights."""
    pcs = gs.pieces(net, sysm)
    if not pcs:
        H = Heights()
        H.zs, H.pcs, H.smp, H.stats = [], pcs, [], dict(feasible=True, converged=True, iterations=0)
        return H
    smp = gs.sample(net, pcs)
    gl = s4.ground_lists(ground, net, smp)
    cons = constraints_of(net, sysm, pcs, smp, gl, ground.fixed, ground.at, spacing)
    return solve_constraints(cons)


def solve_shape(net, sysm, pcs, smp, ground, fixed_z, diagnostics=None):
    """gradeshapes.solve_shape's signature and answer (a height array per piece, None where there are none), through stage 5: the
    switch BNE_STAGED routes the shapes' heights here for measurement. ground: the elevation under each piece's samples (lists);
    fixed_z(i, s): the height of what the shape does not move, which for a road on the ground is the elevation (the held points
    read it)."""
    def at(m, s):
        z = fixed_z(m, s)
        return np.full(np.shape(s), np.nan) if z is None else z
    cons = constraints_of(net, sysm, pcs, smp, ground, fixed_z, at)
    H = solve_constraints(cons)
    if diagnostics is not None:
        diagnostics.update({k: v for k, v in H.stats.items() if not isinstance(v, dict)})
        diagnostics["staged"] = True
        diagnostics["converged"] = bool(H.stats.get("converged"))
        diagnostics["feasible"] = bool(H.stats.get("feasible"))
    if H.zs is None or not H.stats.get("converged"):
        return None
    return H.zs
