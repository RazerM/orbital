import math
import unittest
from math import fmod, radians, tau

from orbital import utilities
from orbital.utilities import mod


class TestUtilities(unittest.TestCase):
    def test_mod(self):
        # These test cases are taken and modified from Python's fmod test
        # cases. orbital.utilities.mod returns
        nan = float("NaN")
        inf = float("Inf")
        ninf = float("-Inf")
        self.assertRaises(TypeError, mod)
        self.assertTrue(math.isnan(mod(nan, 1.0)))
        self.assertTrue(math.isnan(mod(1.0, nan)))
        self.assertTrue(math.isnan(mod(nan, nan)))
        self.assertEqual(mod(1.0, 0.0), 1.0)
        self.assertRaises(ValueError, mod, inf, 1.0)
        self.assertRaises(ValueError, mod, ninf, 1.0)
        self.assertRaises(ValueError, mod, inf, 0.0)
        self.assertEqual(mod(3.0, inf), 3.0)
        self.assertEqual(mod(-3.0, inf), -3.0)
        self.assertEqual(mod(3.0, ninf), 3.0)
        self.assertEqual(mod(-3.0, ninf), -3.0)
        self.assertEqual(mod(0.0, 3.0), 0.0)
        self.assertEqual(mod(0.0, ninf), 0.0)

    def test_eccentric_anomaly_from_mean(self):
        # Pairs of (e, E) to test round-trips.
        CASES = [
            (0.0, 0.0),
            (0.0, radians(90)),
            (0.0, radians(270)),
            (0.75, 0.0),
            (0.75, radians(90)),
            (0.75, radians(270)),
            (0.75, radians(450)),
            (0.75, radians(-90)),
            (0.75, 0.000001),
            (1.0, radians(90)),
            (1.0, radians(270)),
        ]
        # These functions should round-trip, no matter what value is given (as
        # long as e <= 1.0).
        for i, (e, E) in enumerate(CASES):
            M = utilities.mean_anomaly_from_eccentric(e, E)
            self.assertAlmostEqual(
                utilities.eccentric_anomaly_from_mean(e, M),
                fmod(E, tau),
                msg=f"Case #{i:d}: ({e:f}, {E:f})",
            )
