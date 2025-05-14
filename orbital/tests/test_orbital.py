from math import sqrt, tau
import unittest

from astropy import time
import numpy as np
from numpy import radians
import numpy.testing
from orbital import earth, KeplerianElements, venus
from orbital.utilities import mod, Position, Velocity
from scipy.constants import kilo

J2000 = time.Time('J2000', scale='utc')


class TestOrbitalElements(unittest.TestCase):

    def test_circular(self):
        RADIUS = 10000000.0
        orbit = KeplerianElements(a=RADIUS, e=0.0, i=0.0, raan=0.0,
                                  arg_pe=0.0, M0=0.0, body=earth)
        self.assertAlmostEqual(orbit.a, RADIUS)
        self.assertAlmostEqual(orbit.e, 0.0)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, 0.0)

        self.assertAlmostEqual(orbit.epoch, J2000)
        self.assertAlmostEqual(orbit.t, 0.0)
        self.assertAlmostEqual(orbit.M, 0.0)
        self.assertAlmostEqual(orbit.E, 0.0)
        self.assertAlmostEqual(orbit.f, 0.0)

        numpy.testing.assert_almost_equal(orbit.r, Position(RADIUS, 0, 0))
        numpy.testing.assert_almost_equal(orbit.v, Velocity(0, sqrt(earth.mu / RADIUS), 0))

        # Manually calculate angular velocity and period of a circular orbit.
        self.assertAlmostEqual(orbit.n, sqrt(earth.mu / RADIUS ** 3))
        self.assertAlmostEqual(orbit.T, tau * sqrt(RADIUS ** 3 / earth.mu))
        self.assertAlmostEqual(orbit.fpa, 0.0)

        self.assertAlmostEqual(orbit.apocenter_radius, RADIUS)
        self.assertAlmostEqual(orbit.pericenter_radius, RADIUS)
        self.assertAlmostEqual(orbit.apocenter_altitude, RADIUS - earth.mean_radius)
        self.assertAlmostEqual(orbit.pericenter_altitude, RADIUS - earth.mean_radius)

        numpy.testing.assert_almost_equal(orbit.U, np.array([1, 0, 0]))
        numpy.testing.assert_almost_equal(orbit.V, np.array([0, 1, 0]))
        numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
        self.assertUVWMatches(orbit)

    def test_circular_arg_pe(self):
        RADIUS = 10000000.0
        orbit = KeplerianElements(a=RADIUS, e=0.0, i=0.0, raan=0.0,
                                  arg_pe=radians(45), M0=0.0, body=earth)
        self.assertAlmostEqual(orbit.arg_pe, radians(45))
        # The arg_pe gives the angle at the epoch, so these vectors should be
        # rotated 45°.
        numpy.testing.assert_almost_equal(orbit.r,
            Position(RADIUS * 0.5 * sqrt(2), RADIUS * 0.5 * sqrt(2), 0))
        numpy.testing.assert_almost_equal(orbit.v,
            Velocity(-sqrt(earth.mu / RADIUS) * 0.5 * sqrt(2),
                     sqrt(earth.mu / RADIUS) * 0.5 * sqrt(2),
                     0))

        numpy.testing.assert_almost_equal(orbit.U, np.array([0.5 * sqrt(2), 0.5 * sqrt(2), 0]))
        numpy.testing.assert_almost_equal(orbit.V, np.array([-0.5 * sqrt(2), 0.5 * sqrt(2), 0]))
        numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
        self.assertUVWMatches(orbit)

    def test_anomaly_at_time(self):
        RADIUS = 10000000.0
        orbit = KeplerianElements(a=RADIUS, M0=radians(90), body=earth)
        # For a circular orbit, all three anomalies are the same.
        # Test all of the properties that change with time.
        self.assertAlmostEqual(orbit.t, 0.0)
        self.assertAlmostEqual(orbit.M, radians(90))
        self.assertAlmostEqual(orbit.E, radians(90))
        self.assertAlmostEqual(orbit.f, radians(90))
        numpy.testing.assert_almost_equal(orbit.r, Position(0, RADIUS, 0))
        numpy.testing.assert_almost_equal(orbit.v, Velocity(-sqrt(earth.mu / RADIUS), 0, 0))
        self.assertAlmostEqual(orbit.fpa, 0.0)
        numpy.testing.assert_almost_equal(orbit.U, np.array([0, 1, 0]))
        numpy.testing.assert_almost_equal(orbit.V, np.array([-1, 0, 0]))
        numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
        self.assertUVWMatches(orbit)

        # 1/4 of the way around.
        orbit.t = orbit.T * 0.25
        self.assertAlmostEqual(orbit.M, radians(180))
        self.assertAlmostEqual(orbit.E, radians(180))
        self.assertAlmostEqual(orbit.f, radians(180))
        numpy.testing.assert_almost_equal(orbit.r, Position(-RADIUS, 0, 0))
        numpy.testing.assert_almost_equal(orbit.v, Velocity(0, -sqrt(earth.mu / RADIUS), 0))
        self.assertAlmostEqual(orbit.fpa, 0.0)
        numpy.testing.assert_almost_equal(orbit.U, np.array([-1, 0, 0]))
        numpy.testing.assert_almost_equal(orbit.V, np.array([0, -1, 0]))
        numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
        self.assertUVWMatches(orbit)

        # 1/2 of the way around.
        orbit.t = orbit.T * 0.5
        self.assertAlmostEqual(orbit.M, radians(270))
        self.assertAlmostEqual(orbit.E, radians(270))
        self.assertAlmostEqual(orbit.f, radians(270))
        numpy.testing.assert_almost_equal(orbit.r, Position(0, -RADIUS, 0))
        numpy.testing.assert_almost_equal(orbit.v, Velocity(sqrt(earth.mu / RADIUS), 0, 0))
        self.assertAlmostEqual(orbit.fpa, 0.0)
        numpy.testing.assert_almost_equal(orbit.U, np.array([0, -1, 0]))
        numpy.testing.assert_almost_equal(orbit.V, np.array([1, 0, 0]))
        numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
        self.assertUVWMatches(orbit)

        # A full revolution around.
        orbit.t = orbit.T
        self.assertAlmostEqual(orbit.M, radians(90))
        self.assertAlmostEqual(orbit.E, radians(90))
        self.assertAlmostEqual(orbit.f, radians(90))
        numpy.testing.assert_almost_equal(orbit.r, Position(0, RADIUS, 0))
        numpy.testing.assert_almost_equal(orbit.v, Velocity(-sqrt(earth.mu / RADIUS), 0, 0))
        self.assertAlmostEqual(orbit.fpa, 0.0)
        numpy.testing.assert_almost_equal(orbit.U, np.array([0, 1, 0]))
        numpy.testing.assert_almost_equal(orbit.V, np.array([-1, 0, 0]))
        numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
        self.assertUVWMatches(orbit)

        # 5/4 of the way around.
        orbit.t = orbit.T * 1.25
        self.assertAlmostEqual(orbit.M, radians(180))
        self.assertAlmostEqual(orbit.E, radians(180))
        self.assertAlmostEqual(orbit.f, radians(180))
        numpy.testing.assert_almost_equal(orbit.r, Position(-RADIUS, 0, 0))
        numpy.testing.assert_almost_equal(orbit.v, Velocity(0, -sqrt(earth.mu / RADIUS), 0))
        self.assertAlmostEqual(orbit.fpa, 0.0)
        numpy.testing.assert_almost_equal(orbit.U, np.array([-1, 0, 0]))
        numpy.testing.assert_almost_equal(orbit.V, np.array([0, -1, 0]))
        numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
        self.assertUVWMatches(orbit)

    def test_elliptical(self):
        A = 10000000.0
        orbit = KeplerianElements(a=A, e=0.75, i=0.0, raan=0.0, arg_pe=0.0,
                                  M0=0.0, body=earth)
        self.assertAlmostEqual(orbit.a, A)
        self.assertAlmostEqual(orbit.e, 0.75)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, 0.0)

        self.assertAlmostEqual(orbit.epoch, J2000)
        self.assertAlmostEqual(orbit.t, 0.0)
        self.assertAlmostEqual(orbit.M, 0.0)
        self.assertAlmostEqual(orbit.E, 0.0)
        self.assertAlmostEqual(orbit.f, 0.0)

        numpy.testing.assert_almost_equal(orbit.r, Position(2500000, 0, 0))
        numpy.testing.assert_almost_equal(orbit.v, Velocity(0, 16703.901013, 0))

        # Angular velocity and period are the same as for a circular orbit.
        self.assertAlmostEqual(orbit.n, sqrt(earth.mu / A ** 3))
        self.assertAlmostEqual(orbit.T, tau * sqrt(A ** 3 / earth.mu))
        self.assertAlmostEqual(orbit.fpa, 0.0)

        self.assertAlmostEqual(orbit.apocenter_radius, 17500000.0)
        self.assertAlmostEqual(orbit.pericenter_radius, 2500000.0)
        self.assertAlmostEqual(orbit.apocenter_altitude, 17500000.0 - earth.mean_radius)
        self.assertAlmostEqual(orbit.pericenter_altitude, 2500000.0 - earth.mean_radius)

        numpy.testing.assert_almost_equal(orbit.U, np.array([1, 0, 0]))
        numpy.testing.assert_almost_equal(orbit.V, np.array([0, 1, 0]))
        numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
        self.assertUVWMatches(orbit)

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

    def assertUVWMatches(self, orbit):
        """Check that orbit's UVW matches U, V and W.

        This should always be true.
        """
        numpy.testing.assert_almost_equal(orbit.UVW[0], orbit.U)
        numpy.testing.assert_almost_equal(orbit.UVW[1], orbit.V)
        numpy.testing.assert_almost_equal(orbit.UVW[2], orbit.W)

if __name__ == '__main__':
    unittest.main()
