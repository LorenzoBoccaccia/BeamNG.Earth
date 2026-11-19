"""railsolve: a track's heights as one solve under its grade and vertical-curve law."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import railsolve  # noqa: E402
import roads  # noqa: E402

GRADE = 0.05
RADIUS = 1000.0


def line(x0, x1, y=0.0, step=2.0):
    x = np.arange(x0, x1 + 1e-9, step)
    return np.c_[x, np.full(len(x), y)]


def track(parts, targets):
    """Track of parts [(xy, structure)] with their terrain targets (z (k,) each)."""
    _nodes, edges, _adj = roads._chain_graph([p[0] for p in parts])
    return railsolve.Track(parts, edges, targets, RADIUS, GRADE)


class ChainsTest(unittest.TestCase):
    def test_parts_meeting_end_to_end_are_one_chain_and_a_branch_ends_three(self):
        seqs, ends = railsolve.chains_of([(0, 1, None), (2, 1, None), (2, 3, None)])
        self.assertEqual(seqs, [[(0, False), (1, True), (2, False)]], "a part drawn the other way is reversed")
        seqs, ends = railsolve.chains_of([(0, 1, None), (1, 2, None), (3, 2, None), (2, 4, None)])
        self.assertEqual(len(seqs), 3, "three part ends at node 2: its chains end there")
        self.assertEqual(len(ends[2]), 3)


class LawTest(unittest.TestCase):
    def test_a_dip_in_the_terrain_the_track_cannot_follow_is_not_its_height(self):
        """Compiegne's embankment, smeared to a sag 4 m deep and 60 m wide under a short bridge: the track holds its level
        through it, sagging by under half of it (the LiDAR HD's 40.4 m against the terrain's 36.6 m): the radius lets it
        bend by d^2 / 2R, 1.6 m in the fit's balance of the dip's samples against the shoulders'."""
        ground = line(-300.0, 300.0)
        x = ground[:, 0]
        t = 40.0 - 4.0 * np.exp(-(x / 25.0) ** 2)
        got = track([(ground, False)], [t]).solve()[0]
        self.assertGreater(float(got[np.abs(x) < 10.0].min()), 38.0, "level at the sag: the terrain's own is 36.0")
        self.assertLess(float(np.abs(got - 40.0)[np.abs(x) > 100.0].max()), 0.3, "on the terrain away from it")

    def test_the_grade_and_the_curve_hold_where_the_terrain_asks_for_more(self):
        ground = line(0.0, 800.0)
        x = ground[:, 0]
        t = np.where(x < 400.0, 0.0, 0.12 * (x - 400.0))                  # a flat, then a 12 % hill
        got = track([(ground, False)], [t]).solve()[0]
        self.assertLessEqual(float(np.abs(np.diff(got)).max() / 2.0), GRADE + 1e-4)
        bend = np.abs(np.diff(np.diff(got))) / 4.0
        self.assertLessEqual(float(bend.max()), 1.0 / RADIUS + 1e-4)

    def test_the_terrain_under_a_deck_or_over_a_tube_is_no_part_of_the_bed(self):
        a, b, c = line(-300.0, 0.0), line(0.0, 140.0), line(140.0, 400.0)
        flat = [np.full(len(p), 10.0) for p in (a, b, c)]
        flat[1] = np.full(len(b), -30.0)                                    # a valley under the bridge
        got = track([(a, False), (b, True), (c, False)], flat).solve()
        self.assertLess(float(np.abs(got[1] - 10.0).max()), 0.05, "the deck straight between its ends")
        hill = [np.full(len(p), 10.0) for p in (a, b, c)]
        hill[1] = np.full(len(b), 60.0)                                     # a hill over the tube
        got = track([(a, False), (b, True), (c, False)], hill).solve()
        self.assertLess(float(np.abs(got[1] - 10.0).max()), 0.05, "the tube straight between its portals")

    def test_a_lower_bound_is_kept_and_the_track_rises_to_it_by_the_law(self):
        a, b, c = line(-400.0, 0.0), line(0.0, 20.0), line(20.0, 420.0)
        flat = [np.full(len(p), 10.0) for p in (a, b, c)]
        t = track([(a, False), (b, True), (c, False)], flat)
        got = t.solve({1: np.full(len(b), 13.0)})
        self.assertGreaterEqual(float(got[1].min()), 13.0 - 1e-6)
        joined = np.concatenate([got[0], got[1][1:], got[2][1:]])
        self.assertLessEqual(float(np.abs(np.diff(joined)).max() / 2.0), GRADE + 1e-4)
        self.assertAlmostEqual(float(got[0][-1]), float(got[1][0]), places=6, msg="no step at the joint")

    def test_a_sidings_end_on_a_line_is_that_lines_height_there(self):
        main = line(0.0, 600.0)
        siding = np.c_[np.arange(100.0, 400.0, 2.0), np.arange(0.0, 150.0, 1.0) * 40.0 / 150.0]       # leaves the main at x = 100
        t = track([(main, False), (siding, False)], [np.full(len(main), 100.0), np.full(len(siding), 100.5)])
        got = t.solve()
        self.assertAlmostEqual(float(got[1][0]), float(got[0][50]), delta=0.02, msg="tied where it leaves")
        self.assertGreater(float(got[1][-1]), 100.4, "and on its own ground beyond")

    def test_a_chain_too_short_to_bend_is_its_terrain(self):
        short = line(0.0, 4.0)
        got = track([(short, False)], [np.array([1.0, 2.0, 1.0])]).solve()[0]
        np.testing.assert_allclose(got, [1.0, 2.0, 1.0])


if __name__ == "__main__":
    unittest.main()
