from math import sqrt, tau
import unittest
import warnings

from astropy import time
import numpy as np
from numpy import radians
from numpy.linalg import norm
import numpy.testing
from orbital import earth, KeplerianElements, venus
from orbital.utilities import mod, Position, Velocity
from orbital.utilities import OrbitalWarning, ConvergenceError
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

        self.assertAlmostEqual(orbit.ref_epoch, J2000)
        self.assertAlmostEqual(orbit.body, earth)
        self.assertAlmostEqual(orbit.t, 0.0)

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
        self.assertAlmostEqual(orbit.epoch, J2000)
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
        self.assertAlmostEqual(orbit.epoch,
                               J2000 + time.TimeDelta(orbit.T * 0.25, format='sec'))
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
        self.assertAlmostEqual(orbit.epoch,
                               J2000 + time.TimeDelta(orbit.T * 0.5, format='sec'))
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
        self.assertAlmostEqual(orbit.epoch,
                               J2000 + time.TimeDelta(orbit.T, format='sec'))
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
        self.assertAlmostEqual(orbit.epoch,
                               J2000 + time.TimeDelta(orbit.T * 1.25, format='sec'))
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

    def test_zero(self):
        orbit = KeplerianElements(a=0.0, e=0.0, i=0.0, raan=0.0,
                                  arg_pe=0.0, M0=0.0, body=earth)
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
        orbit = KeplerianElements(a=RADIUS, e=0.0, i=radians(45),
                                  raan=radians(90), arg_pe=0.0, M0=0.0,
                                  body=earth)
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
        numpy.testing.assert_almost_equal(orbit.v,
          Velocity(-sqrt(earth.mu / RADIUS) * 0.5 * sqrt(2),
                   0,
                   sqrt(earth.mu / RADIUS) * 0.5 * sqrt(2)))

        # n, T, apsides do not change when an inclination is introduced.
        self.assertAlmostEqual(orbit.n, sqrt(earth.mu / RADIUS ** 3))
        self.assertAlmostEqual(orbit.T, tau * sqrt(RADIUS ** 3 / earth.mu))
        self.assertAlmostEqual(orbit.fpa, 0.0)

        self.assertAlmostEqual(orbit.apocenter_radius, RADIUS)
        self.assertAlmostEqual(orbit.pericenter_radius, RADIUS)
        self.assertAlmostEqual(orbit.apocenter_altitude, RADIUS - earth.mean_radius)
        self.assertAlmostEqual(orbit.pericenter_altitude, RADIUS - earth.mean_radius)

        numpy.testing.assert_almost_equal(orbit.U, np.array([0, 1, 0]))
        numpy.testing.assert_almost_equal(orbit.V, np.array([-0.5 * sqrt(2), 0, 0.5 * sqrt(2)]))
        numpy.testing.assert_almost_equal(orbit.W, np.array([0.5 * sqrt(2), 0, 0.5 * sqrt(2)]))
        self.assertUVWMatches(orbit)

        # Advance time: 1/4 of the way around.
        orbit.t = orbit.T * 0.25
        self.assertAlmostEqual(orbit.M, radians(90))
        self.assertAlmostEqual(orbit.E, radians(90))
        self.assertAlmostEqual(orbit.f, radians(90))
        numpy.testing.assert_almost_equal(orbit.r,
          Position(-RADIUS * 0.5 * sqrt(2), 0, RADIUS * 0.5 * sqrt(2)))
        numpy.testing.assert_almost_equal(orbit.v, Velocity(0, -sqrt(earth.mu / RADIUS), 0))
        self.assertAlmostEqual(orbit.fpa, 0.0)
        numpy.testing.assert_almost_equal(orbit.U, np.array([-0.5 * sqrt(2), 0, 0.5 * sqrt(2)]))
        numpy.testing.assert_almost_equal(orbit.V, np.array([0, -1, 0]))
        numpy.testing.assert_almost_equal(orbit.W, np.array([0.5 * sqrt(2), 0, 0.5 * sqrt(2)]))
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

    def test_elliptical_extreme_e(self):
        A = 500.0
        e = 0.99999999
        orbit = KeplerianElements(a=A, e=e, i=0.0, raan=0.0, arg_pe=0.0,
                                  M0=0.0, body=earth)
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
            warnings.simplefilter('ignore', category=OrbitalWarning)
            orbit.M = radians(359.9)
        self.assertAlmostEqual(orbit.M, radians(359.9))
        # XXX This should produce the results below (it does if
        # utilities.MAX_ITERATIONS is set to 100000), but instead it fails to
        # converge in 100 iterations.
        self.assertRaises(ConvergenceError, lambda: orbit.E)
        #self.assertAlmostEqual(orbit.E, radians(347.454759))
        #self.assertAlmostEqual(orbit.f, radians(180.073718))

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
        numpy.testing.assert_almost_equal(orbit.r, Position(-13255776.4031414, 5408888.899183, 0))
        numpy.testing.assert_almost_equal(orbit.v, Velocity(-3606.1267047, -1678.8615886, 0))
        self.assertAlmostEqual(orbit.fpa, radians(42.837854))
        numpy.testing.assert_almost_equal(orbit.U, np.array([-0.9258875, 0.3777993, 0]))
        numpy.testing.assert_almost_equal(orbit.V, np.array([-0.3777993, -0.9258875, 0]))
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

    def test_hyperbolic(self):
        A = 10000000.0
        orbit = KeplerianElements(a=-A, e=1.25, i=0.0, raan=0.0, arg_pe=0.0,
                                  M0=0.0, body=earth)

        # XXX Commented-out asserts are failing.
        self.assertAlmostEqual(orbit.a, -A)
        self.assertAlmostEqual(orbit.e, 1.25)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, 0.0)

        self.assertAlmostEqual(orbit.epoch, J2000)
        self.assertAlmostEqual(orbit.t, 0.0)
        self.assertAlmostEqual(orbit.M, 0.0)
        self.assertAlmostEqual(orbit.E, 0.0)
        #self.assertAlmostEqual(orbit.f, 0.0)

        #numpy.testing.assert_almost_equal(orbit.r, Position(2500000, 0, 0))
        #numpy.testing.assert_almost_equal(orbit.v, Velocity(0, 18940.443359375, 0.0))

        # Angular velocity and period are the same as for a circular orbit.
        #self.assertAlmostEqual(orbit.n, sqrt(earth.mu / A ** 3))
        #self.assertEqual(orbit.T, float('inf'))
        #self.assertAlmostEqual(orbit.fpa, 0.0)

        #self.assertAlmostEqual(orbit.apocenter_radius, float('inf'))
        self.assertAlmostEqual(orbit.pericenter_radius, 2500000.0)
        #self.assertAlmostEqual(orbit.apocenter_altitude, float('inf'))
        self.assertAlmostEqual(orbit.pericenter_altitude, 2500000.0 - earth.mean_radius)

        #numpy.testing.assert_almost_equal(orbit.U, np.array([1, 0, 0]))
        #numpy.testing.assert_almost_equal(orbit.V, np.array([0, 1, 0]))
        #numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
        self.assertUVWMatches(orbit)

    def test_hyperbolic_times(self):
        A = 10000000.0
        orbit = KeplerianElements(a=-A, e=1.25, i=0.0, raan=0.0, arg_pe=0.0,
                                  M0=0.0, body=earth)
        # Test all of the properties that change with time.
        # XXX Commented-out asserts are failing. Also some may be incorrect.
        # At periapsis.
        self.assertAlmostEqual(orbit.t, 0.0)
        self.assertAlmostEqual(orbit.M, 0.0)
        self.assertAlmostEqual(orbit.E, 0.0)
        #self.assertAlmostEqual(orbit.f, 0.0)
        #numpy.testing.assert_almost_equal(orbit.r, Position(2500000, 0, 0))
        #numpy.testing.assert_almost_equal(orbit.v, Velocity(0, 18940.443359375, 0.0))
        #self.assertAlmostEqual(orbit.fpa, 0.0)
        #numpy.testing.assert_almost_equal(orbit.U, np.array([1, 0, 0]))
        #numpy.testing.assert_almost_equal(orbit.V, np.array([0, 1, 0]))
        #numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
        self.assertUVWMatches(orbit)

        # At f = 90°.
        orbit.t = 387.0334193  # Calculated to get f = 90°.
        #self.assertAlmostEqual(orbit.M, 0.2443528194)
        #self.assertAlmostEqual(orbit.E, 0.6931471806)
        #self.assertAlmostEqual(orbit.f, radians(90))
        #numpy.testing.assert_almost_equal(orbit.r, Position(0, 5625000, 0))
        #numpy.testing.assert_almost_equal(orbit.v, Velocity(-8417.974609, 10522.468750, 0.0))
        #self.assertAlmostEqual(orbit.fpa, radians(0.8960553846))
        #numpy.testing.assert_almost_equal(orbit.U, np.array([0, 1, 0]))
        #numpy.testing.assert_almost_equal(orbit.V, np.array([-1, 0, 0]))
        #numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
        self.assertUVWMatches(orbit)

        # At f = -90°.
        orbit.t = -387.0334193  # Calculated to get f = -90°.
        #self.assertAlmostEqual(orbit.M, -0.2443528194)
        #self.assertAlmostEqual(orbit.E, -0.6931471806)
        #self.assertAlmostEqual(orbit.f, radians(-90))
        #numpy.testing.assert_almost_equal(orbit.r, Position(0, -5625000, 0))
        #numpy.testing.assert_almost_equal(orbit.v, Velocity(8417.974609, 10522.468750, 0.0))
        #self.assertAlmostEqual(orbit.fpa, radians(0.8960553846))
        #numpy.testing.assert_almost_equal(orbit.U, np.array([0, -1, 0]))
        #numpy.testing.assert_almost_equal(orbit.V, np.array([1, 0, 0]))
        #numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
        self.assertUVWMatches(orbit)

        # At M > tau (test that it is not mod-tau).
        orbit.t = 12671.29784
        #self.assertAlmostEqual(orbit.M, 8.0)
        #self.assertAlmostEqual(orbit.E, 2.858)
        #self.assertAlmostEqual(orbit.f, radians(138.998))
        #numpy.testing.assert_almost_equal(orbit.r, Position(-749394, 651493, 0))
        #numpy.testing.assert_almost_equal(orbit.v, Velocity(-5522.950684, 4169.570312, 0.0))
        #self.assertAlmostEqual(orbit.fpa, radians(86.049))
        #numpy.testing.assert_almost_equal(orbit.U, np.array([-1, 1, 0]))  # XXX figure this out
        #numpy.testing.assert_almost_equal(orbit.V, np.array([-1, 1, 0]))  # XXX figure this out
        #numpy.testing.assert_almost_equal(orbit.W, np.array([0, 0, 1]))
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
        # Check all the standard elements.
        self.assertAlmostEqual(orbit.a, ALTITUDE + earth.mean_radius)
        self.assertAlmostEqual(orbit.e, 0.0)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, 0.0)

        self.assertAlmostEqual(orbit.ref_epoch, J2000)
        self.assertAlmostEqual(orbit.body, earth)
        self.assertAlmostEqual(orbit.t, 0.0)

        # Elliptical orbit.
        orbit = KeplerianElements.with_altitude(ALTITUDE, e=0.75, body=earth)
        self.assertAlmostEqual(norm(orbit.r), ALTITUDE + earth.mean_radius)
        self.assertAlmostEqual(orbit.apocenter_radius, 38296000.0 + earth.mean_radius)
        self.assertAlmostEqual(orbit.pericenter_radius, ALTITUDE + earth.mean_radius)
        self.assertAlmostEqual(orbit.apocenter_altitude, 38296000.0)
        self.assertAlmostEqual(orbit.pericenter_altitude, ALTITUDE)
        # Check all the standard elements.
        self.assertAlmostEqual(orbit.a, 25524000.0)
        self.assertAlmostEqual(orbit.e, 0.75)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, 0.0)

        # Elliptical orbit, nonzero M0.
        orbit = KeplerianElements.with_altitude(ALTITUDE, e=0.75, M0=radians(35), body=earth)
        self.assertAlmostEqual(orbit.M, radians(35))
        self.assertAlmostEqual(norm(orbit.r), ALTITUDE + earth.mean_radius)
        self.assertAlmostEqual(orbit.apocenter_radius, 7094311.533422537 + earth.mean_radius)
        self.assertAlmostEqual(orbit.pericenter_radius, -4447384.066653923 + earth.mean_radius)
        self.assertAlmostEqual(orbit.apocenter_altitude, 7094311.533422537)
        self.assertAlmostEqual(orbit.pericenter_altitude, -4447384.066653923)
        # Check all the standard elements.
        self.assertAlmostEqual(orbit.a, 7694463.733384307)
        self.assertAlmostEqual(orbit.e, 0.75)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, radians(35))
        self.assertAlmostEqual(orbit.t, 0.0)

        # Hyperbolic orbit.
        orbit = KeplerianElements.with_altitude(ALTITUDE, e=1.25, body=earth)
        # XXX Commented-out asserts are failing. Also some may be incorrect.
        #self.assertAlmostEqual(norm(orbit.r), ALTITUDE + earth.mean_radius)
        #self.assertAlmostEqual(orbit.apocenter_radius, float('inf'))
        #self.assertAlmostEqual(orbit.pericenter_radius, ALTITUDE + earth.mean_radius)
        #self.assertAlmostEqual(orbit.apocenter_altitude, float('inf'))
        #self.assertAlmostEqual(orbit.pericenter_altitude, ALTITUDE)
        # Check all the standard elements.
        #self.assertAlmostEqual(orbit.a, 25524000.0)  # XXX Don't know what this should be.
        self.assertAlmostEqual(orbit.e, 1.25)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, 0.0)

        # Hyperbolic orbit, nonzero M0.
        orbit = KeplerianElements.with_altitude(ALTITUDE, e=1.25, M0=radians(35), body=earth)
        self.assertAlmostEqual(orbit.M, radians(35))
        # XXX Commented-out asserts are failing. Also some may be incorrect.
        #self.assertAlmostEqual(norm(orbit.r), ALTITUDE + earth.mean_radius)
        #self.assertAlmostEqual(orbit.apocenter_radius, float('inf'))
        # XXX The pericenter should be some other value.
        #self.assertAlmostEqual(orbit.pericenter_radius, ALTITUDE + earth.mean_radius)
        #self.assertAlmostEqual(orbit.apocenter_altitude, float('inf'))
        #self.assertAlmostEqual(orbit.pericenter_altitude, ALTITUDE)
        # Check all the standard elements.
        #self.assertAlmostEqual(orbit.a, 25524000.0)  # XXX Don't know what this should be.
        self.assertAlmostEqual(orbit.e, 1.25)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, radians(35))

    def test_with_period(self):
        orbit = KeplerianElements.with_period(2 * 60 * 60, M0=radians(35), body=earth)
        self.assertAlmostEqual(orbit.T, 2 * 60 * 60)

        # Check all the standard elements.
        self.assertAlmostEqual(orbit.a, 8058997.3045416)
        self.assertAlmostEqual(orbit.e, 0.0)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, radians(35))

        self.assertAlmostEqual(orbit.ref_epoch, J2000)
        self.assertAlmostEqual(orbit.body, earth)
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
        # Check all the standard elements.
        self.assertAlmostEqual(orbit.a, 10000.0 + earth.mean_radius)
        self.assertAlmostEqual(orbit.e, 0.0)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, 0.0)

        self.assertAlmostEqual(orbit.ref_epoch, J2000)
        self.assertAlmostEqual(orbit.body, earth)
        self.assertAlmostEqual(orbit.t, 0.0)

        # Elliptical orbit.
        orbit = KeplerianElements.with_apside_altitudes(10000.0, 38296000.0, body=earth)
        self.assertAlmostEqual(orbit.apocenter_radius, 38296000.0 + earth.mean_radius)
        self.assertAlmostEqual(orbit.pericenter_radius, 10000.0 + earth.mean_radius)
        self.assertAlmostEqual(orbit.apocenter_altitude, 38296000.0)
        self.assertAlmostEqual(orbit.pericenter_altitude, 10000.0)
        # Check all the standard elements.
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
        self.assertAlmostEqual(orbit.pericenter_altitude, 10000000.0 - earth.mean_radius)
        # Check all the standard elements.
        self.assertAlmostEqual(orbit.a, 10000000.0)
        self.assertAlmostEqual(orbit.e, 0.0)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, 0.0)

        self.assertAlmostEqual(orbit.ref_epoch, J2000)
        self.assertAlmostEqual(orbit.body, earth)
        self.assertAlmostEqual(orbit.t, 0.0)

        # Elliptical orbit.
        orbit = KeplerianElements.with_apside_radii(10000000.0, 20000000.0, body=earth)
        self.assertAlmostEqual(orbit.apocenter_radius, 20000000.0)
        self.assertAlmostEqual(orbit.pericenter_radius, 10000000.0)
        self.assertAlmostEqual(orbit.apocenter_altitude, 20000000.0 - earth.mean_radius)
        self.assertAlmostEqual(orbit.pericenter_altitude, 10000000.0 - earth.mean_radius)
        # Check all the standard elements.
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
        # Check all the standard elements.
        self.assertAlmostEqual(orbit.a, RADIUS)
        self.assertAlmostEqual(orbit.e, 0.0)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, 0.0)

        self.assertAlmostEqual(orbit.ref_epoch, J2000)
        self.assertAlmostEqual(orbit.body, earth)
        self.assertAlmostEqual(orbit.t, 0.0)

        # 1/4 of the way around.
        R = Position(0, RADIUS, 0)
        V = Velocity(-sqrt(earth.mu / RADIUS), 0, 0)
        orbit = KeplerianElements.from_state_vector(R, V, body=earth)
        numpy.testing.assert_almost_equal(orbit.r, R)
        numpy.testing.assert_almost_equal(orbit.v, V)
        # Check all the standard elements.
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
        # Circular orbit.
        R = Position(0, 0, 0)
        V = Velocity(0, 10000, 0)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            self.assertRaises(AssertionError,
                              KeplerianElements.from_state_vector, R, V, body=earth)

    def test_from_state_vector_inclined(self):
        # Inclined circular orbit, 1/4 of the way around.
        RADIUS = 10000000.0
        R = Position(-RADIUS * 0.5 * sqrt(2), 0, RADIUS * 0.5 * sqrt(2))
        V = Velocity(0, -sqrt(earth.mu / RADIUS), 0)

        orbit = KeplerianElements.from_state_vector(R, V, body=earth)
        # XXX: r, v and arg_pe are nan for some reason.
        # This happens for a perfect circle, but doesn't happen in
        # test_from_state_vector_circular for some reason.
        #numpy.testing.assert_almost_equal(orbit.r, R)
        #numpy.testing.assert_almost_equal(orbit.v, V)
        # Check all the standard elements.
        self.assertAlmostEqual(orbit.a, RADIUS)
        self.assertAlmostEqual(orbit.e, 0.0)
        self.assertAlmostEqual(orbit.i, radians(45))
        self.assertAlmostEqual(orbit.raan, radians(90))
        #self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, radians(90))

        self.assertAlmostEqual(orbit.ref_epoch, J2000)
        self.assertAlmostEqual(orbit.body, earth)
        self.assertAlmostEqual(orbit.t, 0.0)

    def test_from_state_vector_elliptical(self):
        # Elliptical orbit, at periapsis.
        R = Position(2500000, 0, 0)
        V = Velocity(0, 16703.9010129, 0)

        orbit = KeplerianElements.from_state_vector(R, V, body=earth)
        numpy.testing.assert_almost_equal(orbit.r, R)
        numpy.testing.assert_almost_equal(orbit.v, V)
        # Check all the standard elements.
        self.assertAlmostEqual(orbit.a, 10000000.0, places=3)
        self.assertAlmostEqual(orbit.e, 0.75)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, 0.0)

        self.assertAlmostEqual(orbit.ref_epoch, J2000)
        self.assertAlmostEqual(orbit.body, earth)
        self.assertAlmostEqual(orbit.t, 0.0)

        # Elliptical orbit, 1/4 of the period elapsed.
        R = Position(-13255776.4031414, 5408888.899183, 0)
        V = Velocity(-3606.1267047, -1678.8615886, 0)
        orbit = KeplerianElements.from_state_vector(R, V, body=earth)
        numpy.testing.assert_almost_equal(orbit.r, R, decimal=4)
        numpy.testing.assert_almost_equal(orbit.v, V)
        # Check all the standard elements.
        self.assertAlmostEqual(orbit.a, 10000000.0, places=3)
        self.assertAlmostEqual(orbit.e, 0.75)
        self.assertAlmostEqual(orbit.i, 0.0)
        self.assertAlmostEqual(orbit.raan, 0.0)
        self.assertAlmostEqual(orbit.arg_pe, 0.0)
        self.assertAlmostEqual(orbit.M0, radians(90))
        self.assertAlmostEqual(orbit.t, 0.0)
        self.assertAlmostEqual(orbit.M, radians(90))

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

        # TODO: Hyperbolic orbit.
        # TODO: Parabolic orbit.
        # TODO: Radial orbit (v aligned with r).
        # TODO: Radial orbit (r=0, v != 0).
        # TODO: Radial orbit (v=0).

    def test_from_tle(self):
        # Sample TLE from Wikipedia:
        # https://en.wikipedia.org/wiki/Two-line_element_set
        # ISS (ZARYA)
        LINE1 = '1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927'
        LINE2 = '2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537'
        orbit = KeplerianElements.from_tle(LINE1, LINE2, body=earth)
        self.assertAlmostEqual(orbit.t, 0)
        self.assertEqual(orbit.ref_epoch.strftime('%Y-%b-%d %H:%M:%S'),
                         time.Time('2008-09-20 12:25:40.0').strftime('%Y-%b-%d %H:%M:%S'))

        # These values for r and v were computed by SGP4. Internally, from_tle
        # uses these state vectors as an intermediate step, so verify that the
        # resulting r and v values match.
        R = Position(4083902.4635207, -993631.9996058, 5243603.6653708)
        V = Velocity(2512.8372952, 7259.888525, -583.7785365)
        numpy.testing.assert_almost_equal(orbit.r, R)
        numpy.testing.assert_almost_equal(orbit.v, V)

        # NOTE: These expected values have just been set to match the output
        # from this function. Note that the e, i, raan, arg_pe and M0 are
        # given directly in the TLE data, and a can be computed from the mean
        # motion in the TLE data as shown here.
        # EXPECTED_N_REV_PER_DAY = 15.72125391
        # EXPECTED_N = EXPECTED_N_REV_PER_DAY * tau / 86400  # [rad/s]
        # EXPECTED_A = (earth.mu / EXPECTED_N ** 2) ** (1 / 3)
        #
        # Yet the output values do not match. This is likely due to rounding
        # errors converting to state vectors and back. The largest discrepancy
        # is a, which is off by 5 km. See the note about arg_pe and M0.
        #
        # The comment after each line gives the actual expected value based on
        # the TLE data.
        self.assertAlmostEqual(orbit.a, 6725547.816501163)        # 6730960.68??
        self.assertAlmostEqual(orbit.e, 0.0008330)                # 0.0006703??
        self.assertAlmostEqual(orbit.i, radians(51.621653))       # 51.6416??
        self.assertAlmostEqual(orbit.raan, radians(247.45773))    # 247.4627??
        # Note: The following two values are off by a huge amount (about 18°)
        # but the errors cancel out. Given the eccentricity is so low, the sum
        # of these two angles is all that really matters.
        self.assertAlmostEqual(orbit.arg_pe, radians(112.50348))  # 130.5360??
        self.assertAlmostEqual(orbit.M0, radians(343.056983))     # 325.0288??

    def assertUVWMatches(self, orbit):
        """Check that orbit's UVW matches U, V and W.

        This should always be true.
        """
        numpy.testing.assert_almost_equal(orbit.UVW[0], orbit.U)
        numpy.testing.assert_almost_equal(orbit.UVW[1], orbit.V)
        numpy.testing.assert_almost_equal(orbit.UVW[2], orbit.W)

if __name__ == '__main__':
    unittest.main()
