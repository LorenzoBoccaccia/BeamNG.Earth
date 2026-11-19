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
matrix's band, so the ADMM below carries them at no cost in factoring. W_MIN (the kernel's own floor on every weight,
native/earthfast carriageway_solver.rs) makes the objective strictly convex and is part of the shapes' answer, not a regulariser of
ours: a deck's middle weighs nothing and its line is kept by lam alone, so what sets its height is this floor's pull of every sample
to its natural line (a deck joined at both ends, no target). At Brunau a weight of 1e-4 here instead (what this solver first took) put
decks 8-10 m over the kernel's and the survey's. Measured against the kernel: Popoli's five systems agree to 4 mm and in objective to
1e-4, curve excess included; the kernel's answer breaks none of our rows (the same problem), ours none of its.

The method: ADMM in OSQP's form (Stellato et al. 2020), l <= A x <= u the rows above, the objective 1/2 x'Hx - h'x:

    solve (H + sigma I + A' diag(rho) A) x~ = sigma x + h + A'(rho z - y)       z~ = A x~
    x <- a x~ + (1 - a) x      z <- clip(a z~ + (1 - a) z + y / rho, l, u)       y <- y + rho (a z~ + (1 - a) z - z)

Convergence is a theorem, not a hope: the objective is strictly convex (W_MIN) and the rows' set closed convex and, as the system is
feasible (stage 4: no negative cycle, and the potentials it found are the start), the two-block ADMM converges for every rho > 0
and relaxation a in (0, 2): the primal residual |Ax - z| and the dual one |Hx - h + A'y| go to zero and x to the one minimiser
(Boyd et al. 2011, section 3.2). What the code reports is the residuals it stopped on, never more than that. The linear system is
solved exactly: a sparse LDL' (QDLDL on an AMD ordering, earthfast's SparseLdl, the pair Clarabel carries for the kernel) of the
matrix, factored once for a rho and solved by two triangular sweeps an iteration, the way OSQP's direct solver does it. The steps
are ours; only the linear solve is a library's, and a deterministic one (no threads, one order of operations). The first version
solved it by conjugate gradients with the banded part of the matrix as the preconditioner: the rows between pieces (joints,
clearances, overlaps) left about 35 solves a sweep of ~10 ms each in Python, and Nanpu's 78 382 samples took 658 s and stopped at the
cap with 0.035 m of violation. A joint's equality is a row between two pieces' samples whose multiplier the iteration finds: the
residual at the joints (stats["joint_gap_m"]) is what is left of it. rho is rescaled by the residuals' balance (OSQP's rule) at
most MAX_RHO times, each a new factorisation on the same ordering; the work is MAX_ITER sweeps at most, never a clock.

The exact bounds: from stage 4's graph the tightest height each node can have (shortest paths from the zero node, forward for the
ceiling, backward for the floor: Bellman-Ford with a queue, the graph has no negative cycle) are the box rows' bounds (the set of
feasible heights is unchanged by tightening a box with what the constraints already imply), and the potentials stage 4 found,
heights that satisfy every constraint, are where the iteration starts.

Stopping: the primal residual under TOL_M and the dual one under DUAL_ABS (a force, not a share of the forces: the objective's flat
directions weigh W_MIN, so a dual residual r is a height error of up to r / (2 W_MIN) there; DUAL_ABS 1e-4 is 5 cm per sample, and a
share of the forces let Popoli's tubes stop 14 m from the kernel's), or MAX_ITER. The
constraint violation of the answer itself is measured, edge by edge (stats["violation_m"]): where it is over VIOLATION_M the answer
is not given as converged."""
import math
from collections import deque

import numpy as np

import dnp
import fast
import gradeshapes as gs
import interchange_audit as ia
import staged_feasible as s4
import staged_ground as s3

TOL_M = 5e-4            # m: the iteration stops when no row is further than this from its bounds (and the dual residual is small)
VIOLATION_M = 2e-3      # m: an answer whose worst constraint is violated by more than this is not given as converged
MAX_ITER = 50000        # sweeps at most (the work is bounded by this, never by a clock). The sweeps are first order: Perth's six-piece system needs
#                         10 330 of them to the kernel's answer (0.3 mm), an exact solve each, whatever rho (tried 25 to 2 500, rescaled every 25 to 100)
MAX_RHO = 12            # rescalings of rho at most
RHO0 = 10.0
RHO_EQ = 1.0
SIGMA = 1e-6
DUAL_ABS = 1e-4
ALPHA = 1.6             # over-relaxation (Boyd 3.4.3: any value in (0, 2) keeps the convergence)
W_MIN = 1e-3            # least weight of a sample on its target: the kernel's (carriageway_solver.rs W_MIN), the strictly convex part
CURVE_MIN_GAP_M = 0.5   # m: the least spacing a segment's own `ds` gives (carriageways.MIN_SEGMENT_M)
RHO_EVERY = 25         # sweeps between looks at the residuals' balance


def implied_bounds(g):
    """(floor, ceiling) per node of g (a Graph with no negative cycle): the tightest heights the difference constraints leave a node
    with, relative to the zero node: ceiling_v = shortest path zero -> v, floor_v = - shortest path v -> zero. Lists of g.n."""
    n, Z = g.n, g.zero
    rev = [[] for _ in range(n)]
    for k, v in enumerate(g.ev):
        rev[v].append(k)

    def spfa(source, edges_at, other, forward):
        dist = [math.inf] * n
        dist[source] = 0.0
        queue, inq = deque([source]), [False] * n
        inq[source] = True
        while queue:
            u = queue.popleft()
            inq[u] = False
            du = dist[u]
            for k in edges_at[u]:
                v = other[k]
                nd = du + g.ew[k]
                if nd < dist[v] - ia.EPS_M:
                    dist[v] = nd
                    if not inq[v]:
                        inq[v] = True
                        queue.append(v)
        return dist
    up = spfa(Z, g.adj, g.ev, True)
    down = spfa(Z, rev, g.eu, False)
    return [-d for d in down], up


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


class Kkt:
    """The sweeps' matrix H + sigma I + A' diag(rho) A as the upper triangle of a CSC matrix, its structure made once and its values
    for any rho by one weighted sum (the same entries in the same order, so the same bits), and its factors: earthfast's SparseLdl
    (QDLDL on an AMD ordering, the kernel's own pair), made at the first rho and redone on the same ordering at each new one. The
    rows of A: the edges (z_b - z_a), the boxes (z_k), the curve windows (three entries wide)."""

    def __init__(self, P, bn, c0, c1, c2):
        N = P.N
        m1, m2, m3 = len(P.A), len(bn), len(P.K0)
        self.N = N
        const = m1 + m2 + m3            # (the source row of the entries that do not scale with a row's rho)
        rows, cols, coef, src = [], [], [], []

        def add(r, c, w, s):
            r = np.asarray(r, np.int64)
            rows.append(r)
            cols.append(np.asarray(c, np.int64))
            coef.append(np.broadcast_to(np.asarray(w, np.float64), r.shape))
            src.append(np.broadcast_to(np.asarray(s, np.int64), r.shape))
        idx = np.arange(N, dtype=np.int64)
        chain = np.searchsorted(np.array(P.off[1:], np.int64), idx, side="right")
        add(idx, idx, P.hd + SIGMA, const)
        i1 = idx[:-1][chain[:-1] == chain[1:]]
        add(i1, i1 + 1, P.he[i1], const)
        i2 = idx[:-2][chain[:-2] == chain[2:]]
        add(i2, i2 + 2, P.hf[i2], const)
        re = np.arange(m1, dtype=np.int64)
        add(P.A, P.A, 1.0, re)
        add(P.B, P.B, 1.0, re)
        add(P.A, P.B, -1.0, re)             # (A < B: the upper triangle's row A of column B)
        add(bn, bn, 1.0, m1 + np.arange(m2, dtype=np.int64))
        if m3:
            K0 = P.K0
            rc = m1 + m2 + np.arange(m3, dtype=np.int64)
            add(K0, K0, c0 * c0, rc)
            add(K0 + 1, K0 + 1, c1 * c1, rc)
            add(K0 + 2, K0 + 2, c2 * c2, rc)
            add(K0, K0 + 1, c0 * c1, rc)
            add(K0 + 1, K0 + 2, c1 * c2, rc)
            add(K0, K0 + 2, c0 * c2, rc)
        key = np.concatenate(cols) * N + np.concatenate(rows)
        uniq, self.inv = np.unique(key, return_inverse=True)
        self.inv = self.inv.reshape(-1)
        self.coef = np.concatenate(coef)
        self.src = np.concatenate(src)
        self.indices = uniq % N
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(uniq // N, minlength=N))]).astype(np.int64)
        self.ldl = None
        self.factorisations = 0

    def factor(self, rv):
        """Factors for the rows' weights rv (rho per row of A)."""
        vals = np.bincount(self.inv, weights=self.coef * np.append(rv, 1.0)[self.src], minlength=len(self.indices))
        if self.ldl is None:
            self.ldl = fast.sparse_ldl(self.N, self.indptr, self.indices, vals)
        else:
            self.ldl.refactor(vals)
        self.factorisations += 1

    def solve(self, rhs):
        return self.ldl.solve(rhs)


def admm(P, z0, max_iter=MAX_ITER):
    """ADMM on the problem as OSQP has it (Stellato et al. 2020): minimise 1/2 x'Hx - h'x such that l <= Ax <= u, the rows of A the
    edges' differences, the boxes and the curve windows. Each sweep: solve (H + sigma I + A' diag(rho) A) x~ = sigma x + h + A'(rho z - y)
    exactly (a sparse LDL' on a fill-reducing ordering, factored once for a rho: Kkt), z~ = A x~, the relaxed (alpha) averages,
    z <- the relaxed z + y/rho clipped to [l, u], y <- y + rho (relaxed z - z). Convergence: for every rho > 0 the residuals go to
    zero and x to the minimiser (Boyd et al. 2011, section 3.2; the objective is strictly convex by W_MIN, the sets closed convex).
    (z, stats)."""
    N = P.N
    A, B = P.A, P.B
    floor, ceil = implied_bounds(P.cons.g)
    lo = np.maximum(P.lo, np.array(floor[:N], np.float64))
    hi = np.minimum(P.hi, np.array(ceil[:N], np.float64))
    bn = np.flatnonzero(np.isfinite(lo) | np.isfinite(hi))
    m1, m2, m3 = len(A), len(bn), len(P.K0)
    K0 = P.K0
    K1, K2 = K0 + 1, K0 + 2
    c0, c1, c2 = P.CA, -(P.CA + P.CC), P.CC

    def Cmul(v):
        return c0 * v[K0] + c1 * v[K1] + c2 * v[K2]
    Lw = np.r_[P.DLO, lo[bn], -P.CB]
    Uw = np.r_[P.DHI, hi[bn], P.CB]
    eq = (Uw - Lw) <= 1e-9
    x = np.clip(np.asarray(z0, np.float64), lo, hi)
    Ax = np.r_[x[B] - x[A], x[bn], Cmul(x)]
    z = np.minimum(np.maximum(Ax, Lw), Uw)
    y = np.zeros(m1 + m2 + m3)
    rho = RHO0
    kkt = Kkt(P, bn, c0, c1, c2)

    def factor(rho):
        rv = np.where(eq, RHO_EQ * rho, rho)
        kkt.factor(rv)
        return rv

    def Aty(w):
        out = np.bincount(B, weights=w[:m1], minlength=N) - np.bincount(A, weights=w[:m1], minlength=N) + np.bincount(bn, weights=w[m1:m1 + m2], minlength=N)
        if m3:
            wc = w[m1 + m2:]
            out = out + np.bincount(K0, weights=wc * c0, minlength=N) + np.bincount(K1, weights=wc * c1, minlength=N) + np.bincount(K2, weights=wc * c2, minlength=N)
        return out

    rv = factor(rho)
    rescalings = 0
    it = 0
    r_pri = r_dual = math.inf
    converged = False
    while it < max_iter:
        it += 1
        xt = kkt.solve(SIGMA * x + P.h + Aty(rv * z - y))
        zt = np.r_[xt[B] - xt[A], xt[bn], Cmul(xt)]
        xr = ALPHA * xt + (1.0 - ALPHA) * x
        zr = ALPHA * zt + (1.0 - ALPHA) * z
        z_new = np.minimum(np.maximum(zr + y / rv, Lw), Uw)
        y = y + rv * (zr - z_new)
        x, z = xr, z_new
        Ax = np.r_[x[B] - x[A], x[bn], Cmul(x)]
        Px = _bandmul(P.hd, P.he, P.hf, x)
        ATy = Aty(y)
        r_pri = float(np.abs(Ax - z).max(initial=0.0))
        r_dual = float(np.abs(Px - P.h + ATy).max(initial=0.0))
        if r_pri <= TOL_M and r_dual <= DUAL_ABS:
            converged = True
            break
        if it % RHO_EVERY == 0 and rescalings < MAX_RHO:
            # (OSQP's rule: rho by the square root of the ratio of the normalised residuals, when it is off by 5x)
            pn = r_pri / max(float(np.abs(Ax).max(initial=0.0)), float(np.abs(z).max(initial=0.0)), 1e-12)
            dn = r_dual / max(float(np.abs(Px).max(initial=0.0)), float(np.abs(ATy).max(initial=0.0)),
                              float(np.abs(P.h).max(initial=0.0)), 1e-12)
            ratio = math.sqrt(pn / max(dn, 1e-30))
            if ratio > 5.0 or ratio < 0.2:
                rho = min(max(rho * ratio, 1e-6), 1e8)
                rv = factor(rho)
                rescalings += 1
    stats = dict(iterations=it, factorisations=kkt.factorisations, rho=rho, rho_rescalings=rescalings, primal_residual_m=r_pri,
                 dual_residual=r_dual, converged=converged)
    return x, stats


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
    z, st = admm(P, z0)
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
