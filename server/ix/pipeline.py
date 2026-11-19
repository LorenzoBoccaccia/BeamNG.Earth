"""The interchange pipeline's orchestrator (SPEC.md): region(planner, home, hk) is what _Planner._shape_region_calc
returned, built by calling the s1..s6 steps in order. This PR writes the artifacts only -- it never reads them back to
skip generation -- so with BNE_IX_CACHE unset nothing changes at all."""
import dataclasses
import logging
import os
import time

import numpy as np

import gradeshapes as _gsh

from . import cache, s1_paths, s2_extent, s3_order, s4_terrain, s5_datum, s6_heights

log = logging.getLogger("earthd.grading")


def _put(artifact, parents, inputs):
    """The artifact under its key (its meta filled in), written to the cache when it is on: (artifact, key)."""
    key = cache.key_of(artifact.meta.step, artifact.meta.format, artifact.meta.laws, parents, inputs)
    out = dataclasses.replace(artifact, meta=dataclasses.replace(artifact.meta, key=key, parents=tuple(parents)))
    if cache.ROOT:
        cache.put(out)
    return out, key


def _members_bytes(members):
    """The members' bytes: what the region steps (1, 2) were made from."""
    return [(repr(m.key) + repr((m.kind, m.cls, m.kmh, m.level, m.half, m.held))).encode() +
            np.asarray(m.line, np.float64).tobytes() for m in members]


def _samples_bytes(net, sysm, smp):
    """The samples' xy bytes, as _shape_sys_key digests them: what a system's steps (3-6) were made from."""
    out = [_gsh.system_key(net, sysm).encode()]
    for x in smp:
        out.append(np.round(np.asarray(x["xy"], np.float64), 3).tobytes())
        for i in np.unique(x["mem"]).tolist():
            out.append(repr(net.m[i].key).encode() + repr((net.m[i].kind, getattr(net.m[i], "level", None))).encode())
    return out


def region(planner, home, hk):
    """What _shape_region_calc(home, hk) returned: {road feature id: [(xy (m, 2), z (m,), mode)]}, built by the s1..s6
    steps (the artifacts written, never read back)."""
    import grading      # (the planner's module: its laws and its solve lock; imported here, grading imports us first)
    t0 = time.perf_counter()
    lon0, lat0, lon1, lat1 = (float(v) for v in home.bounds())
    ce, cn, _ = planner.frame.geodetic_to_enu(np.array([lat0, lat0, lat1, lat1]), np.array([lon0, lon1, lon0, lon1]),
                                              0.0)
    h_lo = np.array([np.min(ce), np.min(cn)]) - grading.SHAPES_PAD_M
    h_hi = np.array([np.max(ce), np.max(cn)]) + grading.SHAPES_PAD_M
    box_lo, box_hi = h_lo - 200.0, h_hi + 200.0
    cap_lo, cap_hi = h_lo - grading.SHAPES_CAP_M, h_hi + grading.SHAPES_CAP_M

    extent, net, mine, loaded, members = s2_extent.run(planner, hk, h_lo, h_hi, box_lo, box_hi, cap_lo, cap_hi)
    region_scope = "region:%s" % hk
    inputs = _members_bytes(members)
    _paths, paths_key = _put(s1_paths.run(planner, net, region_scope), (), inputs)
    _extent, extent_key = _put(extent, (paths_key,), inputs)
    t1 = time.perf_counter()

    fixed_z = s2_extent.make_fixed(planner, net)

    shapes, failed, solved, keys = [], 0, [], []
    for sysm in mine:
        pcs, smp, ck = s6_heights.problem(planner, net, sysm)
        system = _gsh.system_key(net, sysm)
        scope = "system:%s" % system
        inputs = _samples_bytes(net, sysm, smp)
        ground, zs, diag = s6_heights.solve(planner, net, sysm, pcs, smp, ck, fixed_z)
        order, order_key = _put(s3_order.run(planner, net, sysm, system, scope), (extent_key,), inputs)
        terrain, _ground = s4_terrain.run(planner, smp, ground, system, scope)
        terrain, terrain_key = _put(terrain, (order_key,), inputs)
        datum, datum_key = _put(s5_datum.run(planner, net, sysm, system, scope), (terrain_key,), inputs)
        _heights, heights_key = _put(s6_heights.run(planner, net, sysm, pcs, smp, zs, diag, system, scope),
                                     (datum_key,), inputs)
        if zs is None:          # (no answer: the system as it is, and why in the log; no stand-in)
            failed += 1
            log.error("grading: SHAPES infeasible system in region %s: %d pieces, %d crossings, %d ends; %s; "
                      "members %s", home, len(pcs), len(sysm["crossings"]), len(sysm["ends"]),
                      diag.get("qp_error") or ", ".join("%s %s" % (k, diag[k]) for k in sorted(diag)
                                                        if k not in ("converged", "feasible")),
                      ", ".join(sorted({str(net.m[i].key[0])[:8] + ":" + net.m[i].kind for i in sysm["seeds"]}))[:600])
            continue
        solved.append(sysm)
        shapes.append((pcs, smp, zs))
        keys.append(heights_key)
        if os.environ.get("BNE_SHAPES_DEBUG"):
            planner._shape_debug(net, sysm, pcs, smp, zs, ground, fixed_z, diag)

    prof, region = s6_heights.region_of(planner, net, solved, shapes, hk)
    _put(region, (extent_key,) + tuple(keys), _members_bytes(members) + [k.encode() for k in keys])
    log.warning("grading: SHAPES pid %d region %s: %d cells, %d members, %d systems (%d infeasible, left as they are), "
                "%d roads graded, find %.2f s, solve %.2f s", os.getpid(), hk, len(loaded), len(members), len(mine),
                failed, len(prof), t1 - t0, time.perf_counter() - t1)
    return prof
