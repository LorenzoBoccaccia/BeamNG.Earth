"""wayroom: the room a move has, the ways it drags with it (and findground's law weighing it)."""
import unittest

import numpy as np

import findground
import wayroom


def line(*pts):
    return np.array(pts, float)


class WayRoomTest(unittest.TestCase):
    def test_a_dip_that_runs_out_before_a_junction_drags_nothing(self):
        # the street dips 2 m at 10 %: 20 m each side; its side street is 40 m away
        ways = wayroom.Ways([line((-100, 0), (100, 0)), line((40, 0), (40, 100))], [0.10, 0.10])
        self.assertEqual(ways.dragged(0, 100.0, 2.0), 0.0)

    def test_a_dip_that_reaches_a_junction_takes_the_side_street_down_with_it(self):
        # 30 m away the dip is 2 - 0.1 * 30 = ... use 4 m at 10 %: 40 m each side; the side street at 30 m is 1 m down, and
        # runs out at its 10 % in 10 m
        ways = wayroom.Ways([line((-100, 0), (100, 0)), line((30, 0), (30, 100))], [0.10, 0.10])
        self.assertAlmostEqual(ways.dragged(0, 100.0, 4.0), 10.0, delta=0.01)

    def test_a_street_crossing_through_goes_down_on_both_sides(self):
        ways = wayroom.Ways([line((-100, 0), (100, 0)), line((30, -50), (30, 50))], [0.10, 0.10])
        self.assertAlmostEqual(ways.dragged(0, 100.0, 4.0), 20.0, delta=0.01)

    def test_ways_that_cross_with_no_shared_vertex_meet_too(self):
        a = wayroom.Ways([line((-100, 0), (100, 0)), line((30, -50), (30, 50))], [0.1, 0.1])
        self.assertEqual(len(a._junctions(0, 90.0, 140.0)), 1)

    def test_the_junctions_of_a_junction_go_down_in_turn(self):
        # a second side street off the first one, 5 m along it, still inside the first's 10 m of run
        ways = wayroom.Ways([line((-100, 0), (100, 0)), line((30, 0), (30, 100)), line((30, 5), (130, 5))], [0.1, 0.1, 0.1])
        self.assertGreater(ways.dragged(0, 100.0, 4.0), 10.0 + 5.0)

    def test_a_road_cut_into_parts_at_a_node_is_one_road(self):
        # the dip of 4 m at 10 % runs 40 m: past the part's end 3 m away on into the next part, which is the road's own run
        # and no arm; a side street 20 m along the next part is still reached by it
        parts = wayroom.Ways([line((-100, 0), (0, 0)), line((0, 0), (100, 0))], [0.1, 0.1])
        self.assertEqual(parts.dragged(0, 97.0, 4.0), 0.0)
        side = wayroom.Ways([line((-100, 0), (0, 0)), line((0, 0), (100, 0)), line((20, 0), (20, 100))], [0.1, 0.1, 0.1])
        self.assertAlmostEqual(side.dragged(0, 97.0, 4.0), (4.0 - 0.1 * 23.0) / 0.1, delta=0.1)

    def test_a_ways_end_is_the_end_of_its_dip(self):
        ways = wayroom.Ways([line((0, 0), (10, 0))], [0.1])
        self.assertEqual(ways.dragged(0, 5.0, 5.0), 0.0)
        near = wayroom.Ways([line((0, 0), (200, 0)), line((3, 0), (3, 100))], [0.1, 0.1])
        # (the side street starts at the street's own end side: only toward its far end, 30 m of 3 m at 10 %)
        self.assertAlmostEqual(near.dragged(0, 0.0, 6.0, (1,)), (6.0 - 0.3) / 0.1, delta=0.01)

    def test_a_ring_goes_down_whole_and_level_with_its_arms_as_deep_as_it(self):
        ring = [(30 * np.cos(t), 30 * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 80)]
        arm = line((30, 0), (130, 0))
        other = line((-30, 0), (-130, 0))
        ways = wayroom.Ways([line(*ring), arm, other], [0.1, 0.1, 0.1])
        self.assertTrue(ways.closed[0])
        # the arm 10 m out meets the ring: 1 m down there, 10 m of the arm, and the ring's 188 m, and the other arm's 10 m
        got = ways.dragged(1, 10.0, 2.0, (-1,))
        self.assertAlmostEqual(got, ways.L[0] + 2.0 / 0.1 - 0.0, delta=2.0 * 0.1 + 20.0)
        # a dip of the ring itself: it all goes, and both arms as deep as it
        whole = ways.dragged(0, 0.0, 2.0)
        self.assertAlmostEqual(whole, ways.L[0] - 2 * 2.0 / 0.1 + 20.0 + 20.0, delta=1.0)


class LawWithRoomTest(unittest.TestCase):
    def test_the_room_a_dip_takes_can_turn_it_into_a_rise(self):
        # a 20 m bridge over a street with 3 m short: 10 m of own run each side, 3 / 0.05 = 60 m of ramp: 100 m of fill
        # against the street's 2 * 3 / 0.1 = 60 m of cut: the street goes down; with a side street's 45 m more it does not
        args = (findground.GROUND, findground.BRIDGE, 3.0, (10.0, 10.0), 0.05, 0.1)
        self.assertEqual(findground.leaves(*args), "lower")
        self.assertEqual(findground.leaves(*args, True, 45.0, 0.0), "upper")

    def test_the_room_a_fill_takes_can_turn_it_into_a_dip(self):
        args = (findground.GROUND, findground.BRIDGE, 3.0, (30.0, 30.0), 0.05, 0.1)
        self.assertEqual(findground.leaves(*args), "upper")
        self.assertEqual(findground.leaves(*args, True, 0.0, 30.0), "lower")


if __name__ == "__main__":
    unittest.main()
