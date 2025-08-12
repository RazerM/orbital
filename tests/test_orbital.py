import unittest
import warnings
from math import sqrt, tau

import numpy as np
import numpy.testing
from astropy import time
from numpy import radians
from numpy.linalg import norm
from scipy.constants import kilo

from orbital import KeplerianElements, earth, venus
from orbital.utilities import ConvergenceError, OrbitalWarning, Position, Velocity

J2000 = time.Time("J2000", scale="utc")


class TestOrbitalElements(unittest.TestCase):
    def test_circular(self):
        RADIUS = 10000000.0
        orbit = KeplerianElements(
            a=RADIUS, e=0.0, i=0.0, raan=0.0, arg_pe=0.0, M0=0.0, body=earth
        )
        self.assertAlmostEqual(orbit.a, RADIUS)
        self.assertAlmostEqual(orbit.e, 0.0)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, 0.0)

        self.assertAlmostEqual(orbit.ref_epoch, J2000)
        self.assertEqual(orbit.body, earth)
        self.assertAlmostEqual(orbit.t, 0.0)

        self.assertAlmostEqual(orbit.epoch, J2000)
        self.assertAlmostEqual(orbit.t, 0.0)
        self.assertAlmostEqual(orbit.M, 0.0)
        self.assertAlmostEqual(orbit.E, 0.0)
        self.assertAlmostEqual(orbit.f, 0.0)

        numpy.testing.assert_almost_equal(orbit.r, Position(RADIUS, 0, 0))
        numpy.testing.assert_almost_equal(
            orbit.v, Velocity(0, sqrt(earth.mu / RADIUS), 0)
        )

        # Manually calculate angular velocity and period of a circular orbit.
        self.assertAlmostEqual(orbit.n, sqrt(earth.mu / RADIUS**3))
        self.assertAlmostEqual(orbit.T, tau * sqrt(RADIUS**3 / earth.mu))
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
        orbit = KeplerianElements(
            a=RADIUS, e=0.0, i=0.0, raan=0.0, arg_pe=radians(45), M0=0.0, body=earth
        )
        self.assertAlmostEqual(orbit.arg_pe, radians(45))
        # The arg_pe gives the angle at the epoch, so these vectors should be
        # rotated 45°.
        numpy.testing.assert_almost_equal(
            orbit.r, Position(RADIUS * 0.5 * sqrt(2), RADIUS * 0.5 * sqrt(2), 0)
        )
        numpy.testing.assert_almost_equal(
            orbit.v,
            Velocity(
                -sqrt(earth.mu / RADIUS) * 0.5 * sqrt(2),
                sqrt(earth.mu / RADIUS) * 0.5 * sqrt(2),
                0,
            ),
        )

        numpy.testing.assert_almost_equal(
            orbit.U, np.array([0.5 * sqrt(2), 0.5 * sqrt(2), 0])
        )
        numpy.testing.assert_almost_equal(
            orbit.V, np.array([-0.5 * sqrt(2), 0.5 * sqrt(2), 0])
        )
        numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
        self.assertUVWMatches(orbit)

    def test_anomaly_at_time(self):
        RADIUS = 10000000.0
        orbit = KeplerianElements(a=RADIUS, M0=radians(90), body=earth)
        # Test all of the properties that change with time.
        self.assertAlmostEqual(orbit.epoch, J2000)
        self.assertAlmostEqual(orbit.t, 0.0)
        # For a circular orbit, all three anomalies are the same.
        self.assertAlmostEqual(orbit.M, radians(90))
        self.assertAlmostEqual(orbit.E, radians(90))
        self.assertAlmostEqual(orbit.f, radians(90))
        numpy.testing.assert_almost_equal(orbit.r, Position(0, RADIUS, 0))
        numpy.testing.assert_almost_equal(
            orbit.v, Velocity(-sqrt(earth.mu / RADIUS), 0, 0)
        )
        self.assertAlmostEqual(orbit.fpa, 0.0)
        numpy.testing.assert_almost_equal(orbit.U, np.array([0, 1, 0]))
        numpy.testing.assert_almost_equal(orbit.V, np.array([-1, 0, 0]))
        numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
        self.assertUVWMatches(orbit)

        # 1/4 of the way around.
        orbit.t = orbit.T * 0.25
        self.assertAlmostEqual(
            orbit.epoch, J2000 + time.TimeDelta(orbit.T * 0.25, format="sec")
        )
        self.assertAlmostEqual(orbit.M, radians(180))
        self.assertAlmostEqual(orbit.E, radians(180))
        self.assertAlmostEqual(orbit.f, radians(180))
        numpy.testing.assert_almost_equal(orbit.r, Position(-RADIUS, 0, 0))
        numpy.testing.assert_almost_equal(
            orbit.v, Velocity(0, -sqrt(earth.mu / RADIUS), 0)
        )
        self.assertAlmostEqual(orbit.fpa, 0.0)
        numpy.testing.assert_almost_equal(orbit.U, np.array([-1, 0, 0]))
        numpy.testing.assert_almost_equal(orbit.V, np.array([0, -1, 0]))
        numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
        self.assertUVWMatches(orbit)

        # 1/2 of the way around.
        orbit.t = orbit.T * 0.5
        self.assertAlmostEqual(
            orbit.epoch, J2000 + time.TimeDelta(orbit.T * 0.5, format="sec")
        )
        self.assertAlmostEqual(orbit.M, radians(270))
        self.assertAlmostEqual(orbit.E, radians(270))
        self.assertAlmostEqual(orbit.f, radians(270))
        numpy.testing.assert_almost_equal(orbit.r, Position(0, -RADIUS, 0))
        numpy.testing.assert_almost_equal(
            orbit.v, Velocity(sqrt(earth.mu / RADIUS), 0, 0)
        )
        self.assertAlmostEqual(orbit.fpa, 0.0)
        numpy.testing.assert_almost_equal(orbit.U, np.array([0, -1, 0]))
        numpy.testing.assert_almost_equal(orbit.V, np.array([1, 0, 0]))
        numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
        self.assertUVWMatches(orbit)

        # A full revolution around.
        orbit.t = orbit.T
        self.assertAlmostEqual(
            orbit.epoch, J2000 + time.TimeDelta(orbit.T, format="sec")
        )
        self.assertAlmostEqual(orbit.M, radians(90))
        self.assertAlmostEqual(orbit.E, radians(90))
        self.assertAlmostEqual(orbit.f, radians(90))
        numpy.testing.assert_almost_equal(orbit.r, Position(0, RADIUS, 0))
        numpy.testing.assert_almost_equal(
            orbit.v, Velocity(-sqrt(earth.mu / RADIUS), 0, 0)
        )
        self.assertAlmostEqual(orbit.fpa, 0.0)
        numpy.testing.assert_almost_equal(orbit.U, np.array([0, 1, 0]))
        numpy.testing.assert_almost_equal(orbit.V, np.array([-1, 0, 0]))
        numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
        self.assertUVWMatches(orbit)

        # 5/4 of the way around.
        orbit.t = orbit.T * 1.25
        self.assertAlmostEqual(
            orbit.epoch, J2000 + time.TimeDelta(orbit.T * 1.25, format="sec")
        )
        self.assertAlmostEqual(orbit.M, radians(180))
        self.assertAlmostEqual(orbit.E, radians(180))
        self.assertAlmostEqual(orbit.f, radians(180))
        numpy.testing.assert_almost_equal(orbit.r, Position(-RADIUS, 0, 0))
        numpy.testing.assert_almost_equal(
            orbit.v, Velocity(0, -sqrt(earth.mu / RADIUS), 0)
        )
        self.assertAlmostEqual(orbit.fpa, 0.0)
        numpy.testing.assert_almost_equal(orbit.U, np.array([-1, 0, 0]))
        numpy.testing.assert_almost_equal(orbit.V, np.array([0, -1, 0]))
        numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
        self.assertUVWMatches(orbit)

    def test_zero(self):
        orbit = KeplerianElements(
            a=0.0, e=0.0, i=0.0, raan=0.0, arg_pe=0.0, M0=0.0, body=earth
        )
        self.assertAlmostEqual(orbit.a, 0.0)
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

        numpy.testing.assert_almost_equal(orbit.r, Position(0, 0, 0))
        # XXX Arguably, v should be the zero vector.
        self.assertRaises(ZeroDivisionError, lambda: orbit.v)

        # XXX Arguably, n should be infinity.
        self.assertRaises(ZeroDivisionError, lambda: orbit.n)
        # XXX If n is infinity, T should be 0.0.
        self.assertRaises(ZeroDivisionError, lambda: orbit.T)
        self.assertAlmostEqual(orbit.fpa, 0.0)

        self.assertAlmostEqual(orbit.apocenter_radius, 0.0)
        self.assertAlmostEqual(orbit.pericenter_radius, 0.0)
        self.assertAlmostEqual(orbit.apocenter_altitude, -earth.mean_radius)
        self.assertAlmostEqual(orbit.pericenter_altitude, -earth.mean_radius)

        numpy.testing.assert_almost_equal(orbit.U, np.array([1, 0, 0]))
        numpy.testing.assert_almost_equal(orbit.V, np.array([0, 1, 0]))
        numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
        self.assertUVWMatches(orbit)

    def test_inclined(self):
        RADIUS = 10000000.0
        orbit = KeplerianElements(
            a=RADIUS,
            e=0.0,
            i=radians(45),
            raan=radians(90),
            arg_pe=0.0,
            M0=0.0,
            body=earth,
        )
        self.assertAlmostEqual(orbit.a, RADIUS)
        self.assertAlmostEqual(orbit.e, 0.0)
        self.assertAlmostEqual(orbit.i, radians(45))
        self.assertAlmostEqual(orbit.raan, radians(90))
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, 0.0)

        self.assertAlmostEqual(orbit.epoch, J2000)
        self.assertAlmostEqual(orbit.t, 0.0)
        self.assertAlmostEqual(orbit.M, 0.0)
        self.assertAlmostEqual(orbit.E, 0.0)
        self.assertAlmostEqual(orbit.f, 0.0)

        numpy.testing.assert_almost_equal(orbit.r, Position(0, RADIUS, 0))
        numpy.testing.assert_almost_equal(
            orbit.v,
            Velocity(
                -sqrt(earth.mu / RADIUS) * 0.5 * sqrt(2),
                0,
                sqrt(earth.mu / RADIUS) * 0.5 * sqrt(2),
            ),
        )

        # n, T, apsides do not change when an inclination is introduced.
        self.assertAlmostEqual(orbit.n, sqrt(earth.mu / RADIUS**3))
        self.assertAlmostEqual(orbit.T, tau * sqrt(RADIUS**3 / earth.mu))
        self.assertAlmostEqual(orbit.fpa, 0.0)

        self.assertAlmostEqual(orbit.apocenter_radius, RADIUS)
        self.assertAlmostEqual(orbit.pericenter_radius, RADIUS)
        self.assertAlmostEqual(orbit.apocenter_altitude, RADIUS - earth.mean_radius)
        self.assertAlmostEqual(orbit.pericenter_altitude, RADIUS - earth.mean_radius)

        numpy.testing.assert_almost_equal(orbit.U, np.array([0, 1, 0]))
        numpy.testing.assert_almost_equal(
            orbit.V, np.array([-0.5 * sqrt(2), 0, 0.5 * sqrt(2)])
        )
        numpy.testing.assert_almost_equal(
            orbit.W, np.array([0.5 * sqrt(2), 0, 0.5 * sqrt(2)])
        )
        self.assertUVWMatches(orbit)

        # Advance time: 1/4 of the way around.
        orbit.t = orbit.T * 0.25
        self.assertAlmostEqual(orbit.M, radians(90))
        self.assertAlmostEqual(orbit.E, radians(90))
        self.assertAlmostEqual(orbit.f, radians(90))
        numpy.testing.assert_almost_equal(
            orbit.r, Position(-RADIUS * 0.5 * sqrt(2), 0, RADIUS * 0.5 * sqrt(2))
        )
        numpy.testing.assert_almost_equal(
            orbit.v, Velocity(0, -sqrt(earth.mu / RADIUS), 0)
        )
        self.assertAlmostEqual(orbit.fpa, 0.0)
        numpy.testing.assert_almost_equal(
            orbit.U, np.array([-0.5 * sqrt(2), 0, 0.5 * sqrt(2)])
        )
        numpy.testing.assert_almost_equal(orbit.V, np.array([0, -1, 0]))
        numpy.testing.assert_almost_equal(
            orbit.W, np.array([0.5 * sqrt(2), 0, 0.5 * sqrt(2)])
        )
        self.assertUVWMatches(orbit)

    def test_elliptical(self):
        A = 10000000.0
        orbit = KeplerianElements(
            a=A, e=0.75, i=0.0, raan=0.0, arg_pe=0.0, M0=0.0, body=earth
        )
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
        self.assertAlmostEqual(orbit.n, sqrt(earth.mu / A**3))
        self.assertAlmostEqual(orbit.T, tau * sqrt(A**3 / earth.mu))
        self.assertAlmostEqual(orbit.fpa, 0.0)

        self.assertAlmostEqual(orbit.apocenter_radius, 17500000.0)
        self.assertAlmostEqual(orbit.pericenter_radius, 2500000.0)
        self.assertAlmostEqual(orbit.apocenter_altitude, 17500000.0 - earth.mean_radius)
        self.assertAlmostEqual(orbit.pericenter_altitude, 2500000.0 - earth.mean_radius)

        numpy.testing.assert_almost_equal(orbit.U, np.array([1, 0, 0]))
        numpy.testing.assert_almost_equal(orbit.V, np.array([0, 1, 0]))
        numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
        self.assertUVWMatches(orbit)

    def test_elliptical_extreme_e(self):
        A = 500.0
        e = 0.99999999
        orbit = KeplerianElements(
            a=A, e=e, i=0.0, raan=0.0, arg_pe=0.0, M0=0.0, body=earth
        )
        self.assertAlmostEqual(orbit.a, A)
        self.assertAlmostEqual(orbit.e, e)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, 0.0)

        self.assertAlmostEqual(orbit.epoch, J2000)
        self.assertAlmostEqual(orbit.t, 0.0)
        self.assertAlmostEqual(orbit.M, 0.0)
        self.assertAlmostEqual(orbit.E, 0.0)
        self.assertAlmostEqual(orbit.f, 0.0)

        # Problem case: e is very close to 1.0 and M is very close to 360°.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=OrbitalWarning)
            orbit.M = radians(359.9)
        self.assertAlmostEqual(orbit.M, radians(359.9))
        # XXX This should produce the results below (it does if
        # utilities.MAX_ITERATIONS is set to 100000), but instead it fails to
        # converge in 100 iterations.
        self.assertRaises(ConvergenceError, lambda: orbit.E)
        # self.assertAlmostEqual(orbit.E, radians(347.454759))
        # self.assertAlmostEqual(orbit.f, radians(180.073718))

    def test_elliptical_times(self):
        A = 10000000.0
        orbit = KeplerianElements(a=A, e=0.75, body=earth)
        # Test all of the properties that change with time.
        # At periapsis.
        self.assertAlmostEqual(orbit.t, 0.0)
        self.assertAlmostEqual(orbit.M, 0.0)
        self.assertAlmostEqual(orbit.E, 0.0)
        self.assertAlmostEqual(orbit.f, 0.0)
        numpy.testing.assert_almost_equal(orbit.r, Position(2500000, 0, 0))
        numpy.testing.assert_almost_equal(orbit.v, Velocity(0, 16703.901013, 0))
        self.assertAlmostEqual(orbit.fpa, 0.0)
        numpy.testing.assert_almost_equal(orbit.U, np.array([1, 0, 0]))
        numpy.testing.assert_almost_equal(orbit.V, np.array([0, 1, 0]))
        numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
        self.assertUVWMatches(orbit)

        # 1/4 of the period has elapsed (more than a quarter of the way around).
        orbit.t = orbit.T * 0.25
        self.assertAlmostEqual(orbit.M, radians(90))
        self.assertAlmostEqual(orbit.E, radians(125.140095))
        self.assertAlmostEqual(orbit.f, radians(157.802569))
        numpy.testing.assert_almost_equal(
            orbit.r, Position(-13255776.4031414, 5408888.899183, 0)
        )
        numpy.testing.assert_almost_equal(
            orbit.v, Velocity(-3606.1267047, -1678.8615886, 0)
        )
        self.assertAlmostEqual(orbit.fpa, radians(42.837854))
        numpy.testing.assert_almost_equal(orbit.U, np.array([-0.9258875, 0.3777993, 0]))
        numpy.testing.assert_almost_equal(
            orbit.V, np.array([-0.3777993, -0.9258875, 0])
        )
        numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
        self.assertUVWMatches(orbit)

        # 1/2 of the period has elapsed, half way around (at apoapsis).
        orbit.t = orbit.T * 0.5
        self.assertAlmostEqual(orbit.M, radians(180))
        self.assertAlmostEqual(orbit.E, radians(180))
        self.assertAlmostEqual(orbit.f, radians(180))
        numpy.testing.assert_almost_equal(orbit.r, Position(-17500000, 0, 0))
        numpy.testing.assert_almost_equal(orbit.v, Velocity(0, -2386.2715733, 0))
        self.assertAlmostEqual(orbit.fpa, 0.0)
        numpy.testing.assert_almost_equal(orbit.U, np.array([-1, 0, 0]))
        numpy.testing.assert_almost_equal(orbit.V, np.array([0, -1, 0]))
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
        self.assertEqual(
            venusian_orbit.apocenter_radius, venusian_orbit.apocytherion_radius
        )
        self.assertEqual(
            venusian_orbit.pericenter_radius, venusian_orbit.perikrition_radius
        )

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

    def test_with_altitude(self):
        # TODO: This isn't the best example, as most of the cases below get a
        # negative periapsis altitude because it's so close to the earth.
        ALTITUDE = 10000.0

        # Circular orbit.
        orbit = KeplerianElements.with_altitude(ALTITUDE, body=earth)
        self.assertAlmostEqual(norm(orbit.r), ALTITUDE + earth.mean_radius)
        self.assertAlmostEqual(orbit.apocenter_radius, ALTITUDE + earth.mean_radius)
        self.assertAlmostEqual(orbit.pericenter_radius, ALTITUDE + earth.mean_radius)
        self.assertAlmostEqual(orbit.apocenter_altitude, ALTITUDE)
        self.assertAlmostEqual(orbit.pericenter_altitude, ALTITUDE)
        self.assertAlmostEqual(orbit.a, ALTITUDE + earth.mean_radius)
        self.assertAlmostEqual(orbit.e, 0.0)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, 0.0)

        self.assertAlmostEqual(orbit.ref_epoch, J2000)
        self.assertEqual(orbit.body, earth)
        self.assertAlmostEqual(orbit.t, 0.0)

        # Elliptical orbit.
        orbit = KeplerianElements.with_altitude(ALTITUDE, e=0.75, body=earth)
        self.assertAlmostEqual(norm(orbit.r), ALTITUDE + earth.mean_radius)
        self.assertAlmostEqual(orbit.apocenter_radius, 38296000.0 + earth.mean_radius)
        self.assertAlmostEqual(orbit.pericenter_radius, ALTITUDE + earth.mean_radius)
        self.assertAlmostEqual(orbit.apocenter_altitude, 38296000.0)
        self.assertAlmostEqual(orbit.pericenter_altitude, ALTITUDE)
        self.assertAlmostEqual(orbit.a, 25524000.0)
        self.assertAlmostEqual(orbit.e, 0.75)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, 0.0)

        # Elliptical orbit, nonzero M0.
        orbit = KeplerianElements.with_altitude(
            ALTITUDE, e=0.75, M0=radians(35), body=earth
        )
        self.assertAlmostEqual(orbit.M, radians(35))
        self.assertAlmostEqual(norm(orbit.r), ALTITUDE + earth.mean_radius)
        self.assertAlmostEqual(
            orbit.apocenter_radius, 7094311.533422537 + earth.mean_radius
        )
        self.assertAlmostEqual(
            orbit.pericenter_radius, -4447384.066653923 + earth.mean_radius
        )
        self.assertAlmostEqual(orbit.apocenter_altitude, 7094311.533422537)
        self.assertAlmostEqual(orbit.pericenter_altitude, -4447384.066653923)
        self.assertAlmostEqual(orbit.a, 7694463.733384307)
        self.assertAlmostEqual(orbit.e, 0.75)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, radians(35))
        self.assertAlmostEqual(orbit.t, 0.0)

    def test_with_period(self):
        orbit = KeplerianElements.with_period(2 * 60 * 60, M0=radians(35), body=earth)
        self.assertAlmostEqual(orbit.T, 2 * 60 * 60)

        self.assertAlmostEqual(orbit.a, 8058997.3045416)
        self.assertAlmostEqual(orbit.e, 0.0)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, radians(35))

        self.assertAlmostEqual(orbit.ref_epoch, J2000)
        self.assertEqual(orbit.body, earth)
        self.assertAlmostEqual(orbit.t, 0.0)

        # TODO: These should be part of separate tests, as they have nothing to
        # do with with_period.
        # Propagate to set t nonzero to test M0 fix when a is set.
        orbit.propagate_anomaly_by(M=radians(10))

        new_T = 3 * 60 * 60
        orbit.T = new_T

        # While we're here, test n set correctly.
        self.assertAlmostEqual(orbit.T, new_T)

        # Test that fixed M0 allows correct propagation.
        orbit.propagate_anomaly_to(M=radians(40))
        self.assertAlmostEqual(orbit.M, radians(40))

    def test_with_apside_altitudes(self):
        # Circular orbit.
        orbit = KeplerianElements.with_apside_altitudes(10000.0, 10000.0, body=earth)
        self.assertAlmostEqual(orbit.apocenter_radius, 10000.0 + earth.mean_radius)
        self.assertAlmostEqual(orbit.pericenter_radius, 10000.0 + earth.mean_radius)
        self.assertAlmostEqual(orbit.apocenter_altitude, 10000.0)
        self.assertAlmostEqual(orbit.pericenter_altitude, 10000.0)
        self.assertAlmostEqual(orbit.a, 10000.0 + earth.mean_radius)
        self.assertAlmostEqual(orbit.e, 0.0)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, 0.0)

        self.assertAlmostEqual(orbit.ref_epoch, J2000)
        self.assertEqual(orbit.body, earth)
        self.assertAlmostEqual(orbit.t, 0.0)

        # Elliptical orbit.
        orbit = KeplerianElements.with_apside_altitudes(10000.0, 38296000.0, body=earth)
        self.assertAlmostEqual(orbit.apocenter_radius, 38296000.0 + earth.mean_radius)
        self.assertAlmostEqual(orbit.pericenter_radius, 10000.0 + earth.mean_radius)
        self.assertAlmostEqual(orbit.apocenter_altitude, 38296000.0)
        self.assertAlmostEqual(orbit.pericenter_altitude, 10000.0)
        self.assertAlmostEqual(orbit.a, 25524000.0)
        self.assertAlmostEqual(orbit.e, 0.75)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, 0.0)

        # Elliptical orbit, arguments reversed (should make no difference).
        orbit = KeplerianElements.with_apside_altitudes(38296000.0, 10000.0, body=earth)
        self.assertAlmostEqual(orbit.apocenter_altitude, 38296000.0)
        self.assertAlmostEqual(orbit.pericenter_altitude, 10000.0)
        self.assertAlmostEqual(orbit.a, 25524000.0)
        self.assertAlmostEqual(orbit.e, 0.75)

    def test_with_apside_radii(self):
        # Circular orbit.
        orbit = KeplerianElements.with_apside_radii(10000000.0, 10000000.0, body=earth)
        self.assertAlmostEqual(orbit.apocenter_radius, 10000000.0)
        self.assertAlmostEqual(orbit.pericenter_radius, 10000000.0)
        self.assertAlmostEqual(orbit.apocenter_altitude, 10000000.0 - earth.mean_radius)
        self.assertAlmostEqual(
            orbit.pericenter_altitude, 10000000.0 - earth.mean_radius
        )
        self.assertAlmostEqual(orbit.a, 10000000.0)
        self.assertAlmostEqual(orbit.e, 0.0)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, 0.0)

        self.assertAlmostEqual(orbit.ref_epoch, J2000)
        self.assertEqual(orbit.body, earth)
        self.assertAlmostEqual(orbit.t, 0.0)

        # Elliptical orbit.
        orbit = KeplerianElements.with_apside_radii(10000000.0, 20000000.0, body=earth)
        self.assertAlmostEqual(orbit.apocenter_radius, 20000000.0)
        self.assertAlmostEqual(orbit.pericenter_radius, 10000000.0)
        self.assertAlmostEqual(orbit.apocenter_altitude, 20000000.0 - earth.mean_radius)
        self.assertAlmostEqual(
            orbit.pericenter_altitude, 10000000.0 - earth.mean_radius
        )
        self.assertAlmostEqual(orbit.a, 15000000.0)
        self.assertAlmostEqual(orbit.e, 1.0 / 3.0)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, 0.0)

        # Elliptical orbit, arguments reversed (should make no difference).
        orbit = KeplerianElements.with_apside_radii(20000000.0, 10000000.0, body=earth)
        self.assertAlmostEqual(orbit.apocenter_radius, 20000000.0)
        self.assertAlmostEqual(orbit.pericenter_radius, 10000000.0)
        self.assertAlmostEqual(orbit.a, 15000000.0)
        self.assertAlmostEqual(orbit.e, 1.0 / 3.0)

    def test_from_state_vector_circular(self):
        # Circular orbit.
        RADIUS = 10000000.0
        R = Position(RADIUS, 0, 0)
        V = Velocity(0, sqrt(earth.mu / RADIUS), 0)

        orbit = KeplerianElements.from_state_vector(R, V, body=earth)
        numpy.testing.assert_almost_equal(orbit.r, R)
        numpy.testing.assert_almost_equal(orbit.v, V)
        self.assertAlmostEqual(orbit.a, RADIUS)
        self.assertAlmostEqual(orbit.e, 0.0)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, 0.0)

        self.assertAlmostEqual(orbit.ref_epoch, J2000)
        self.assertEqual(orbit.body, earth)
        self.assertAlmostEqual(orbit.t, 0.0)

        # 1/4 of the way around.
        R = Position(0, RADIUS, 0)
        V = Velocity(-sqrt(earth.mu / RADIUS), 0, 0)
        orbit = KeplerianElements.from_state_vector(R, V, body=earth)
        numpy.testing.assert_almost_equal(orbit.r, R)
        numpy.testing.assert_almost_equal(orbit.v, V)
        self.assertAlmostEqual(orbit.a, RADIUS)
        self.assertAlmostEqual(orbit.e, 0.0)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        # t is always 0; instead it is expected to set M0.
        self.assertAlmostEqual(orbit.M0, radians(90))
        self.assertAlmostEqual(orbit.t, 0.0)
        self.assertAlmostEqual(orbit.M, radians(90))

    def test_from_state_vector_zero(self):
        # Degenerate orbit with r=0 and v=0.
        R = Position(0, 0, 0)
        V = Velocity(0, 0, 0)

        # This case currently violates an assertion due to internal values being
        # NaN.
        # XXX An AssertionError indicates an unexpected case. This should
        # probably be explicitly detected and raise another kind of error.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.assertRaises(
                AssertionError, KeplerianElements.from_state_vector, R, V, body=earth
            )

    def test_from_state_vector_circular_retrograde(self):
        # Circular orbit, retrograde.
        # Regression test for https://github.com/RazerM/orbital/issues/18.
        RADIUS = 10000000.0
        R = Position(RADIUS, 0, 0)
        V = Velocity(0, -sqrt(earth.mu / RADIUS), 0)

        # XXX This erroneously generates warnings and raises an assertion, due
        # to internal values being NaN.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.assertRaises(
                AssertionError, KeplerianElements.from_state_vector, R, V, body=earth
            )
        # The expected values, after this bug is fixed:
        # numpy.testing.assert_almost_equal(orbit.r, R)
        # numpy.testing.assert_almost_equal(orbit.v, V)
        # self.assertAlmostEqual(orbit.a, RADIUS)
        # self.assertAlmostEqual(orbit.e, 0.0)
        # self.assertAlmostEqual(orbit.i, radians(180))
        # self.assertAlmostEqual(orbit.raan, 0.0)
        # self.assertAlmostEqual(orbit.arg_pe, 0.0)
        # self.assertAlmostEqual(orbit.M0, 0.0)

        # self.assertAlmostEqual(orbit.ref_epoch, J2000)
        # self.assertEqual(orbit.body, earth)
        # self.assertAlmostEqual(orbit.t, 0.0)

    def test_from_state_vector_inclined(self):
        # Inclined circular orbit, 1/4 of the way around.
        # Regression test for https://github.com/RazerM/orbital/issues/38.
        RADIUS = 10000000.0
        R = Position(-RADIUS * 0.5 * sqrt(2), 0, RADIUS * 0.5 * sqrt(2))
        V = Velocity(0, -sqrt(earth.mu / RADIUS), 0)

        with warnings.catch_warnings():
            # XXX This has a warning for dividing by zero.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            orbit = KeplerianElements.from_state_vector(R, V, body=earth)
        # XXX: r, v and arg_pe are nan due to a bug in from_state_vector.
        # This happens for a perfect circle, but doesn't happen in
        # test_from_state_vector_circular for some reason.
        # numpy.testing.assert_almost_equal(orbit.r, R)
        # numpy.testing.assert_almost_equal(orbit.v, V)
        self.assertAlmostEqual(orbit.a, RADIUS)
        self.assertAlmostEqual(orbit.e, 0.0)
        self.assertAlmostEqual(orbit.i, radians(45))
        self.assertAlmostEqual(orbit.raan, radians(90))
        # self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, radians(90))

        self.assertAlmostEqual(orbit.ref_epoch, J2000)
        self.assertEqual(orbit.body, earth)
        self.assertAlmostEqual(orbit.t, 0.0)

    def test_from_state_vector_elliptical(self):
        # Elliptical orbit, at periapsis.
        R = Position(2500000, 0, 0)
        V = Velocity(0, 16703.9010129, 0)

        orbit = KeplerianElements.from_state_vector(R, V, body=earth)
        numpy.testing.assert_almost_equal(orbit.r, R)
        numpy.testing.assert_almost_equal(orbit.v, V)
        self.assertAlmostEqual(orbit.a, 10000000.0, places=3)
        self.assertAlmostEqual(orbit.e, 0.75)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, 0.0)

        self.assertAlmostEqual(orbit.ref_epoch, J2000)
        self.assertEqual(orbit.body, earth)
        self.assertAlmostEqual(orbit.t, 0.0)

        # Elliptical orbit, 1/4 of the period elapsed.
        R = Position(-13255776.4031414, 5408888.899183, 0)
        V = Velocity(-3606.1267047, -1678.8615886, 0)
        orbit = KeplerianElements.from_state_vector(R, V, body=earth)
        numpy.testing.assert_almost_equal(orbit.r, R, decimal=4)
        numpy.testing.assert_almost_equal(orbit.v, V)
        self.assertAlmostEqual(orbit.a, 10000000.0, places=3)
        self.assertAlmostEqual(orbit.e, 0.75)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, radians(90))
        self.assertAlmostEqual(orbit.t, 0.0)
        self.assertAlmostEqual(orbit.M, radians(90))

    def test_from_state_vector_elliptical_arg_pe_gt_180(self):
        # Elliptical orbit, at periapsis, with arg_pe > 180°.
        # Regression test for https://github.com/RazerM/orbital/issues/39.
        R = Position(0, -2500000, 0)
        V = Velocity(16703.9010129, 0, 0)

        orbit = KeplerianElements.from_state_vector(R, V, body=earth)
        # XXX These do not match (they are 180° out, due to arg_pe).
        # numpy.testing.assert_almost_equal(orbit.r, R)
        # numpy.testing.assert_almost_equal(orbit.v, V)
        self.assertAlmostEqual(orbit.a, 10000000.0, places=3)
        self.assertAlmostEqual(orbit.e, 0.75)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        # XXX This is incorrectly calculated as 90°.
        # self.assertAlmostEqual(orbit.arg_pe, radians(270.0))
        self.assertAlmostEqual(orbit.M0, 0.0)

        self.assertAlmostEqual(orbit.ref_epoch, J2000)
        self.assertEqual(orbit.body, earth)
        self.assertAlmostEqual(orbit.t, 0.0)

    def test_from_state_vector_f_at_periapsis(self):
        # Elliptical orbit, inclined, at periapsis.
        # Regression test for https://github.com/RazerM/orbital/issues/40.
        # In this particular case, the above bug would cause f to be nan due to
        # floating point rounding errors.
        R = Position(0, -1767766.952966369, -1767766.952966369)
        V = Velocity(16703.901013, 0, 0)

        # XXX Currently crashes due to the above bug.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.assertRaises(
                AssertionError, KeplerianElements.from_state_vector, R, V, body=earth
            )
        # numpy.testing.assert_almost_equal(orbit.r, R)
        # numpy.testing.assert_almost_equal(orbit.v, V)
        # self.assertAlmostEqual(orbit.a, 10000000.0, places=2)
        # self.assertAlmostEqual(orbit.e, 0.75)
        # self.assertAlmostEqual(orbit.i, radians(45))
        # self.assertAlmostEqual(orbit.raan, 0.0)
        # self.assertAlmostEqual(orbit.arg_pe, radians(270.0))
        # self.assertAlmostEqual(orbit.M0, 0.0)

        # self.assertAlmostEqual(orbit.ref_epoch, J2000)
        # self.assertEqual(orbit.body, earth)
        # self.assertAlmostEqual(orbit.t, 0.0)

    def test_from_state_vector_iss(self):
        # ISS (Zarya) from 2008-09-20 12:25:40
        # Source: SGP4 parsed TLE example from
        # https://en.wikipedia.org/wiki/Two-line_element_set
        # (This is the same state vector used by test_from_tle below.)
        # Used as a real-world smoke test, just ensure that the vectors
        # round-trip.
        R = Position(4083902.4635207, -993631.9996058, 5243603.6653708)
        V = Velocity(2512.8372952, 7259.888525, -583.7785365)
        orbit = KeplerianElements.from_state_vector(R, V, body=earth)
        numpy.testing.assert_almost_equal(orbit.r, R)
        numpy.testing.assert_almost_equal(orbit.v, V)

    def test_from_state_vector_roundtrips(self):
        # Simple round-trip tests for a lot of edge cases in
        # utilities.elements_from_state_vector. Just confirm that r and v come
        # back relatively unchanged for all of these inputs (without
        # hand-verifying all the elements).

        # List of (r, v) pairs.
        # XXX Commented-out cases are failing.
        CASES = [
            # Zero.
            # (Position(0, 0, 0), Velocity(0, 0, 0)),
            # Circular flat, prograde.
            (Position(10000000, 0, 0), Velocity(0, 6313.4811435530555, 0)),
            # Circular flat, f > 180°.
            (Position(0, -10000000, 0), Velocity(6313.4811435530555, 0, 0)),
            # Circular flat, retrograde.
            # (Position(10000000, 0, 0), Velocity(0, -6313.4811435530555, 0)),
            # Circular inclined, prograde.
            # (Position(10000000, 0, 0), Velocity(0, 4464.305329499764, 4464.305329499764)),
            # Circular inclined, prograde (raan 90°, M0 90°).
            # (Position(-7071067.811865476, 0, 7071067.811865476), Velocity(0, -6313.4811435530555, 0)),
            # Circular inclined, raan > 180°.
            # (Position(0, -10000000, 0), Velocity(4464.305329499764, 0, 4464.305329499764)),
            # Circular inclined, f > 180°.
            # (Position(0, -7071067.811865476, -7071067.811865476), Velocity(6313.4811435530555, 0, 0)),
            # Circular polar.
            # (Position(10000000, 0, 0), Velocity(0, 0, 6313.4811435530555)),
            # Elliptical flat (e=0.75), prograde.
            (Position(2500000, 0, 0), Velocity(0, 16703.901013, 0)),
            # Elliptical flat (e=0.75), arg_pe > 180°.
            # (Position(0, -2500000, 0), Velocity(16703.901013, 0, 0)),
            # Elliptical flat (e=0.75), retrograde.
            # (Position(2500000, 0, 0), Velocity(0, -16703.901013, 0)),
            # Elliptical flat (e=0.75), f > 180°.
            (
                Position(-13255776.4031414, -5408888.899183, 0),
                Velocity(3606.1267047, -1678.8615886, 0),
            ),
            # Elliptical inclined.
            (
                Position(2500000, 0, 0),
                Velocity(0, 11811.441678561141, 11811.441678561141),
            ),
            # Elliptical inclined, arg_pe > 180°.
            # (Position(0, -1767766.952966369, -1767766.952966369), Velocity(16703.901013, 0, 0)),
        ]

        for i, (r, v) in enumerate(CASES):
            orbit = KeplerianElements.from_state_vector(r, v, body=earth)
            numpy.testing.assert_almost_equal(
                orbit.r, r, decimal=4, err_msg=f"Case #{i:d}: {orbit}"
            )
            numpy.testing.assert_almost_equal(
                orbit.v, v, decimal=4, err_msg=f"Case #{i:d}: {orbit}"
            )

    def test_from_tle(self):
        # Sample TLE from Wikipedia:
        # https://en.wikipedia.org/wiki/Two-line_element_set
        # ISS (ZARYA)
        LINE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
        LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
        orbit = KeplerianElements.from_tle(LINE1, LINE2, body=earth)
        self.assertAlmostEqual(orbit.t, 0)
        self.assertTimeEqual(orbit.ref_epoch, time.Time("2008-09-20 12:25:40.0"))

        # These values for r and v were computed by SGP4. Internally, from_tle
        # uses these state vectors as an intermediate step, so verify that the
        # resulting r and v values match.
        R = Position(4083902.4635207, -993631.9996058, 5243603.6653708)
        V = Velocity(2512.8372952, 7259.888525, -583.7785365)
        numpy.testing.assert_almost_equal(orbit.r, R)
        numpy.testing.assert_almost_equal(orbit.v, V)

        # Regression test for https://github.com/RazerM/orbital/issues/43.

        # NOTE: These expected values have just been set to match the output
        # from this function. Note that the e, i, raan, arg_pe and M0 are
        # given directly in the TLE data, and a can be computed from the mean
        # motion in the TLE data as shown here.
        # EXPECTED_N_REV_PER_DAY = 15.72125391
        # EXPECTED_N = EXPECTED_N_REV_PER_DAY * tau / 86400  # [rad/s]
        # EXPECTED_A = (earth.mu / EXPECTED_N ** 2) ** (1 / 3)
        #
        # Yet the output values do not match. This is due to rounding errors
        # converting to state vectors and back. The largest discrepancy is a,
        # which is off by 5 km. See the note about arg_pe and M0.
        #
        # The comment after each line gives the actual expected value based on
        # the TLE data.
        self.assertAlmostEqual(orbit.a, 6725547.816501163)  # 6730960.68
        self.assertAlmostEqual(orbit.e, 0.0008330)  # 0.0006703
        self.assertAlmostEqual(orbit.i, radians(51.621653))  # 51.6416
        self.assertAlmostEqual(orbit.raan, radians(247.45773))  # 247.4627
        # Note: The following two values are off by a huge amount (about 18°)
        # but the errors cancel out. Given the eccentricity is so low, the sum
        # of these two angles is all that really matters.
        self.assertAlmostEqual(orbit.arg_pe, radians(112.50348))  # 130.5360
        self.assertAlmostEqual(orbit.M0, radians(343.056983))  # 325.0288

    def test_set_epoch(self):
        RADIUS = 10000000.0
        orbit = KeplerianElements(
            a=RADIUS, e=0.0, i=0.0, raan=0.0, arg_pe=0.0, M0=0.0, body=earth
        )
        self.assertTimeEqual(orbit.epoch, time.Time("2000-01-01 12:00:00"))
        orbit.epoch = time.Time("2000-01-01 12:03:00")
        self.assertTimeEqual(orbit.epoch, time.Time("2000-01-01 12:03:00"))
        self.assertAlmostEqual(orbit.t, 180.0)
        self.assertAlmostEqual(orbit.M, (180 * orbit.n) % tau)

    def test_set_M(self):
        # Circular trajectory.
        RADIUS = 10000000.0
        orbit = KeplerianElements(
            a=RADIUS, e=0.0, i=0.0, raan=0.0, arg_pe=0.0, M0=0.0, body=earth
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=OrbitalWarning)
            orbit.M = radians(495)
        self.assertAlmostEqual(orbit.M, radians(135))
        self.assertAlmostEqual(orbit.E, radians(135))
        self.assertAlmostEqual(orbit.f, radians(135))
        # t does not get set.
        self.assertAlmostEqual(orbit.t, 0.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=OrbitalWarning)
            orbit.M = radians(-45)
        self.assertAlmostEqual(orbit.M, radians(315))
        self.assertAlmostEqual(orbit.E, radians(315))
        self.assertAlmostEqual(orbit.f, radians(315))

    def test_set_E(self):
        # Elliptical trajectory.
        A = 10000000.0
        orbit = KeplerianElements(
            a=A, e=0.75, i=0.0, raan=0.0, arg_pe=0.0, M0=0.0, body=earth
        )
        # Set E such that M is 90 degrees.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=OrbitalWarning)
            orbit.E = radians(125.140095)
        self.assertAlmostEqual(orbit.M, radians(90))
        self.assertAlmostEqual(orbit.E, radians(125.140095))
        self.assertAlmostEqual(orbit.f, radians(157.802569))
        # t does not get set.
        self.assertAlmostEqual(orbit.t, 0.0)

        # Test values outside the range (0, tau].
        # Regression test for https://github.com/RazerM/orbital/issues/37.
        # XXX Unlike setting M or f, setting E does not mod by tau. Attempting
        # to read the value of E back results in non-convergence because the
        # value of M is outside of the expected range.
        # Commented-out asserts are failing.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=OrbitalWarning)
            orbit.E = radians(485.140095)
        # self.assertAlmostEqual(orbit.M, radians(90))
        # self.assertAlmostEqual(orbit.E, radians(125.140095))
        # self.assertAlmostEqual(orbit.f, radians(157.802569))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=OrbitalWarning)
            orbit.E = radians(-180)
        # self.assertAlmostEqual(orbit.M, radians(180))
        # self.assertAlmostEqual(orbit.E, radians(180))
        # self.assertAlmostEqual(orbit.f, radians(180))

    def test_set_f(self):
        # Elliptical trajectory.
        A = 10000000.0
        orbit = KeplerianElements(
            a=A, e=0.75, i=0.0, raan=0.0, arg_pe=0.0, M0=0.0, body=earth
        )
        # Set f such that M is 90 degrees.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=OrbitalWarning)
            orbit.f = radians(157.802569)
        self.assertAlmostEqual(orbit.M, radians(90))
        self.assertAlmostEqual(orbit.E, radians(125.140095))
        self.assertAlmostEqual(orbit.f, radians(157.802569))
        # t does not get set.
        self.assertAlmostEqual(orbit.t, 0.0)

        # mod tau test.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=OrbitalWarning)
            orbit.f = radians(517.802569)
        self.assertAlmostEqual(orbit.M, radians(90))
        self.assertAlmostEqual(orbit.E, radians(125.140095))
        self.assertAlmostEqual(orbit.f, radians(157.802569))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=OrbitalWarning)
            orbit.f = radians(-180)
        self.assertAlmostEqual(orbit.M, radians(180))
        self.assertAlmostEqual(orbit.E, radians(180))
        self.assertAlmostEqual(orbit.f, radians(180))

    def test_set_a(self):
        orbit = KeplerianElements(
            a=10000000.0, e=0.0, i=0.0, raan=0.0, arg_pe=0.0, M0=radians(45), body=earth
        )
        orbit.t = orbit.T * 0.125  # Another 45 degrees.
        self.assertAlmostEqual(orbit.M, radians(90))
        orbit.a = 5000000.0
        self.assertAlmostEqual(orbit.a, 5000000.0)
        # M should be preserved.
        self.assertAlmostEqual(orbit.M, radians(90))

    def test_set_v(self):
        # Start with a circular trajectory.
        RADIUS = 2500000.0
        R = Position(RADIUS, 0, 0)
        V = Velocity(0, sqrt(earth.mu / RADIUS), 0)
        orbit = KeplerianElements(
            a=RADIUS, e=0.0, i=0.0, raan=0.0, arg_pe=0.0, M0=0.0, body=earth
        )
        numpy.testing.assert_almost_equal(orbit.r, R)
        numpy.testing.assert_almost_equal(orbit.v, V)

        # Elliptical trajectory.
        # Increase the velocity at the periapsis, which should grow the apoapsis
        # and make an elliptical orbit. The r and periapsis should remain the
        # same.
        V = Velocity(0, 16703.901013, 0)
        orbit.v = V
        numpy.testing.assert_almost_equal(orbit.r, R)
        numpy.testing.assert_almost_equal(orbit.v, V)
        self.assertAlmostEqual(orbit.pericenter_radius, RADIUS)
        self.assertAlmostEqual(orbit.a, 10000000, places=2)
        self.assertAlmostEqual(orbit.e, 0.75)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, 0.0)

    def test_set_v_arg_pe_gt_180(self):
        # Test the same concept as
        # test_from_state_vector_elliptical_arg_pe_gt_180 but with the v setter
        # instead of from_state_vector.
        # Regression test for https://github.com/RazerM/orbital/issues/39.
        # Elliptical orbit, at periapsis, with arg_pe = 90°.
        orbit = KeplerianElements(
            a=10000000.0,
            e=0.75,
            i=0.0,
            raan=0.0,
            arg_pe=radians(90),
            M0=0.0,
            body=earth,
        )
        R = Position(0, 2500000, 0)
        numpy.testing.assert_almost_equal(orbit.r, R)
        numpy.testing.assert_almost_equal(orbit.v, Velocity(-16703.9010129, 0, 0))

        # Reduce velocity to switch around the apoapsis and periapsis, so
        # arg_pe = 270°.
        V = Velocity(-10000, 0, 0)

        def set_v(value):
            orbit.v = value

        # XXX The 'r and v changed' detection logic is triggered in this case,
        # causing a RuntimeError to be raised. If this was not raised, the
        # following asserts would be wildly off.
        self.assertRaises(RuntimeError, set_v, V)
        # numpy.testing.assert_almost_equal(orbit.r, R)
        # numpy.testing.assert_almost_equal(orbit.v, V)
        # arg_pe should have rotated around 180°, and M0 to match (so r is in
        # the same spot as it was before).
        # XXX This is incorrectly calculated as 90°.
        # self.assertAlmostEqual(orbit.arg_pe, radians(270.0))
        self.assertAlmostEqual(orbit.M0, radians(180.0))

    def test_set_n(self):
        # Circular trajectory.
        RADIUS = 10000000.0
        orbit = KeplerianElements(
            a=RADIUS, e=0.0, i=0.0, raan=0.0, arg_pe=0.0, M0=0.0, body=earth
        )
        NEW_N = sqrt(earth.mu / 5000000.0**3)
        orbit.n = NEW_N
        self.assertAlmostEqual(orbit.n, NEW_N)
        self.assertAlmostEqual(orbit.T, tau / NEW_N)
        self.assertAlmostEqual(orbit.a, 5000000.0)

    def test_set_T(self):
        # Circular trajectory.
        RADIUS = 10000000.0
        orbit = KeplerianElements(
            a=RADIUS, e=0.0, i=0.0, raan=0.0, arg_pe=0.0, M0=0.0, body=earth
        )
        NEW_T = tau * sqrt(5000000.0**3 / earth.mu)
        orbit.T = NEW_T
        self.assertAlmostEqual(orbit.n, tau / NEW_T)
        self.assertAlmostEqual(orbit.T, NEW_T)
        self.assertAlmostEqual(orbit.a, 5000000.0)

    def assertUVWMatches(self, orbit):
        """Check that orbit's UVW matches U, V and W.

        This should always be true.
        """
        numpy.testing.assert_almost_equal(orbit.UVW[0], orbit.U)
        numpy.testing.assert_almost_equal(orbit.UVW[1], orbit.V)
        numpy.testing.assert_almost_equal(orbit.UVW[2], orbit.W)

    def assertTimeEqual(self, t1, t2):
        """Assert that two Time values match to the nearest second."""
        self.assertEqual(
            t1.strftime("%Y-%b-%d %H:%M:%S"), t2.strftime("%Y-%b-%d %H:%M:%S")
        )


if __name__ == "__main__":
    unittest.main()
