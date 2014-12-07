import math
import unittest

from numpy import radians
from orbital.bodies import earth, venus
from orbital.elements import KeplerianElements
from orbital.utilities import mod
from scipy.constants import kilo


class TestOrbitalElements(unittest.TestCase):

    def test_anomaly_at_time(self):
        orbit = KeplerianElements.with_period(90 * 60, e=0, body=earth)
        self.assertAlmostEqual(orbit.M, 0)
        orbit.t += orbit.T
        self.assertAlmostEqual(orbit.M, 0)

    def test_apsides(self):
        orbit = KeplerianElements.with_period(90 * 60, body=earth)

        # Test that only general and specific apside attributes work.
        self.assertRaises(AttributeError, lambda: orbit.apohelion_radius)
        self.assertRaises(AttributeError, lambda: orbit.perihelion_radius)
        self.assertEqual(orbit.apocenter_radius, orbit.apogee_radius)
        self.assertEqual(orbit.pericenter_radius, orbit.perigee_radius)

        # Ensure earth apsides haven't been added to class definition dynamically.
        venusian_orbit = KeplerianElements.with_period(90 * 60, body=venus)
        self.assertRaises(AttributeError, lambda: venusian_orbit.apogee_radius)
        self.assertRaises(AttributeError, lambda: venusian_orbit.perigee_radius)

        # Test multiple apsis names
        self.assertEqual(venusian_orbit.apocenter_radius, venusian_orbit.apocytherion_radius)
        self.assertEqual(venusian_orbit.pericenter_radius, venusian_orbit.perikrition_radius)

    def test_mean_motion(self):
        """Test setting the mean motion, and its associated effects on a and M0."""
        orbit = KeplerianElements.with_altitude(500 * kilo, M0=radians(35), body=earth)

        # Propagate to set t nonzero to test M0 fix when a is set.
        orbit.propagate_anomaly_by(M=radians(10))

        new_n = 0.5 * orbit.n
        orbit.n = new_n

        # While we're here, test n set correctly.
        self.assertAlmostEqual(orbit.n, new_n)

        # Test that fixed M0 allows correct propagation.
        orbit.propagate_anomaly_to(M=radians(40))
        self.assertAlmostEqual(orbit.M, radians(40))

    def test_period(self):
        orbit = KeplerianElements.with_period(2 * 60 * 60, M0=radians(35), body=earth)
        self.assertAlmostEqual(orbit.T, 2 * 60 * 60)

        # Propagate to set t nonzero to test M0 fix when a is set.
        orbit.propagate_anomaly_by(M=radians(10))

        new_T = 3 * 60 * 60
        orbit.T = new_T

        # While we're here, test n set correctly.
        self.assertAlmostEqual(orbit.T, new_T)

        # Test that fixed M0 allows correct propagation.
        orbit.propagate_anomaly_to(M=radians(40))
        self.assertAlmostEqual(orbit.M, radians(40))

if __name__ == '__main__':
    unittest.main()
