from orbital.utilities import mod
from orbital.elements import KeplerianElements
from scipy.constants import pi
import unittest
import random
import orbital.bodies
import math


class TestOrbitalElements(unittest.TestCase):

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

    def test_anomaly_at_time(self):
        orbit = KeplerianElements.orbit_with_period(90 * 60, e=0, body=orbital.bodies.earth)
        self.assertAlmostEqual(orbit.M, 0)
        orbit.t += orbit.T
        self.assertAlmostEqual(orbit.M, 0)

    def test_apsides(self):
        orbit = KeplerianElements.orbit_with_period(90 * 60, body=orbital.bodies.earth)

        # Test that only general and specific apside attributes work.
        self.assertRaises(AttributeError, lambda: orbit.apohelion_radius)
        self.assertRaises(AttributeError, lambda: orbit.perihelion_radius)
        self.assertEqual(orbit.apocenter_radius, orbit.apogee_radius)
        self.assertEqual(orbit.pericenter_radius, orbit.perigee_radius)

        # Ensure earth apsides haven't been added to class definition dynamically.
        venusian_orbit = KeplerianElements.orbit_with_period(90 * 60, body=orbital.bodies.venus)
        self.assertRaises(AttributeError, lambda: venusian_orbit.apogee_radius)
        self.assertRaises(AttributeError, lambda: venusian_orbit.perigee_radius)

        # Test multiple apsis names
        self.assertEqual(venusian_orbit.apocenter_radius, venusian_orbit.apocytherion_radius)
        self.assertEqual(venusian_orbit.pericenter_radius, venusian_orbit.perikrition_radius)

if __name__ == '__main__':
    unittest.main()
