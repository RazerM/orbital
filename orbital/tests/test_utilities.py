from orbital.utilities import mod
import math
import unittest


class TestUtilities(unittest.TestCase):
    def test_mod(self):
        # These test cases are taken and modified from Python's fmod test
        # cases. orbital.utilities.mod returns
        nan = float('NaN')
        inf = float('Inf')
        ninf = float('-Inf')
        self.assertRaises(TypeError, mod)
        self.assertTrue(math.isnan(mod(nan, 1.)))
        self.assertTrue(math.isnan(mod(1., nan)))
        self.assertTrue(math.isnan(mod(nan, nan)))
        self.assertEqual(mod(1., 0.), 1.)
        self.assertRaises(ValueError, mod, inf, 1.)
        self.assertRaises(ValueError, mod, ninf, 1.)
        self.assertRaises(ValueError, mod, inf, 0.)
        self.assertEqual(mod(3.0, inf), 3.0)
        self.assertEqual(mod(-3.0, inf), -3.0)
        self.assertEqual(mod(3.0, ninf), 3.0)
        self.assertEqual(mod(-3.0, ninf), -3.0)
        self.assertEqual(mod(0.0, 3.0), 0.0)
        self.assertEqual(mod(0.0, ninf), 0.0)
