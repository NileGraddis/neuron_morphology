import unittest
import sys

from shapely.geometry import LineString, Point

import neuron_morphology.snap_polygons.cortex_surfaces as cs


class TestStandalone(unittest.TestCase):

    def test_remove_duplicates(self):
        cases = [
            [
                [Point(0, 1), Point(0, 1), Point(1, 1), Point(0, 1)],
                [Point(0, 1), Point(1, 1), Point(0, 1)]
            ],
            [
                [Point(0, 1), Point(0, 0), Point(0, 0), Point(0, 0)],
                [Point(0, 1), Point(0, 0)]
            ],
            [
                [Point(0, 1), Point(0.0, 1.0), Point(1, 1)],
                [Point(0, 1), Point(1, 1)]
            ],
            [
                [Point(0, 1), Point(sys.float_info.epsilon, 1), Point(1, 1)],
                [Point(0, 1), Point(sys.float_info.epsilon, 1), Point(1, 1)]
            ]
        ]

        for inpt, expected in cases:
            with self.subTest():
                obtained = cs.remove_duplicates(inpt)
                self.assertEqual(LineString(obtained), LineString(expected))

    def test_find_transition(self):
        condition = lambda pt: Point(0, 0).distance(pt) <= 1
        cases = [
            [Point(0, -2), Point(0, 0), 0, Point(0, -1)],
            [Point(0, -4), Point(0, 0), 2, Point(0, -1)],
            [Point(0, -4), Point(0, -1), 10, Point(0, -1)],
            [Point(0, -1.5), Point(0, 0.5), 1, Point(0, -1)]
        ]

        for unmet, met, iterations, expected in cases:
            with self.subTest():
                obtained = cs.find_transition(
                    unmet, met, condition, iterations)
                print(obtained.xy, expected.xy)
                self.assertEqual(obtained, expected)