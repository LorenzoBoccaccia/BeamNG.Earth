"""Step 6 (SPEC.md): the pieces, their samples and the solve, and the interchange's embankments and profiles, as the
Heights and Region artifacts. A system is solved once for the run: the owner of its interchange does it (owner.py), under
the interchange's lock, so there is no cache or wait of its own here."""
import logging

import numpy as np

import dnp
import gradeshapes as _gsh

from . import artifacts, cache, s4_terrain

log = logging.getLogger("earthd.grading")

FORMAT = 1
LAWS = (("gradeshapes", ["STEP_M", "DECK_MAX_GRADE", "DECK_AIR_M", "SOFT_M", "DIP_MARGIN_M", "RAMP_MAX_M", "W_END",
                         "W_HOLD", "W_DIP", "W_SMOOTH", "W_STIFF", "W_JOIN"]),)
LAWS_DIGEST = cache.laws_digest(*LAWS)
STEP = 6


def problem(planner, net, sysm):
    """(pcs, smp): the system's pieces and their samples."""
    pcs = _gsh.pieces(net, sysm)
    return pcs, _gsh.sample(net, pcs)


def solve(planner, net, sysm, pcs, smp, fixed_z):
    """(ground, zs, diag) of one system: the ground along its samples and the solve."""
    diag = {}
    ground = s4_terrain.ground_of(planner, smp)
    zs = _gsh.solve_shape(net, sysm, pcs, smp, ground, fixed_z, diag)
    return ground, zs, diag


def _plain(diag):
    """The solver's diagnostics as plain values (np scalars unwrapped, anything else as its str). A wall-clock time is
    not solver state: dropped, or no two runs would give the same bytes."""
    out = {}
    for k, v in diag.items():
        if str(k) == "elapsed_s":
            continue
        if isinstance(v, np.generic):
            v = v.item()
        out[str(k)] = v if v is None or isinstance(v, (str, int, float, bool)) else str(v)
    return out


def run(planner, net, sysm, pcs, smp, zs, diag, system, scope):
    """The Heights artifact of one system's solve (solved False: no answer, the system left as it is)."""
    solved = zs is not None
    pieces = []
    for pc, x, z in zip(pcs, smp, zs if solved else [None] * len(pcs)):
        pieces.append(artifacts.Piece(pc["kind"], int(pc["main"]), np.asarray(x["xy"], np.float64),
                                      np.asarray(x["d"], np.float64), np.asarray(x["mem"], np.int64),
                                      np.asarray(x["ms"], np.float64),
                                      np.full(len(x["xy"]), np.nan) if z is None else np.asarray(z, np.float64)))
    return artifacts.Heights(artifacts.Meta(STEP, FORMAT, "", (), LAWS_DIGEST, scope), system, pieces, solved,
                             _plain(diag))


def region_of(planner, net, solved, shapes, hk, beside=()):
    """(prof, Region): the embankments round the solved systems and the roads' profiles, as data. beside: the pieces of
    other systems of net, solved elsewhere (gradeshapes.embankments)."""
    def ground_along(i, sv):
        xy = np.stack([dnp.interp(sv, net.s[i], net.lines[i][:, 0]), dnp.interp(sv, net.s[i], net.lines[i][:, 1])],
                      axis=1)
        return planner._ground_line(xy)

    emb = _gsh.embankments(net, solved, shapes, ground_along, beside)
    prof = {}
    for (pcs, smp, zs) in shapes:
        for pc, x, z in zip(pcs, smp, zs):
            if pc["kind"] == "deck":
                prof.setdefault(str(net.m[pc["main"]].key[0]), []).append((x["xy"], np.asarray(z, np.float64),
                                                                           "deck"))
    for j, lst in emb.items():
        m = net.m[j]
        if m.kind not in ("ground", "tunnel"):
            continue
        for sv, z, mode in lst:
            xy = np.stack([dnp.interp(sv, net.s[j], net.lines[j][:, 0]),
                           dnp.interp(sv, net.s[j], net.lines[j][:, 1])], axis=1)
            prof.setdefault(str(m.key[0]), []).append((xy, np.asarray(z, np.float64), mode))
    region = artifacts.Region(artifacts.Meta(STEP, FORMAT, "", (), LAWS_DIGEST, "region:%s" % hk), str(hk),
                              [artifacts.Profile(str(fid), np.asarray(xy, np.float64), np.asarray(z, np.float64),
                                                 str(mode))
                               for fid, lst in prof.items() for (xy, z, mode) in lst])
    return prof, region
