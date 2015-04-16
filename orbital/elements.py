# encoding: utf-8
from __future__ import absolute_import, division, print_function

import warnings

import numpy as np
from astropy import time
from numpy import arccos as acos
from numpy import arctan, cos, degrees, dot, sin, sqrt
from numpy.linalg import norm
from represent import RepresentationMixin
from scipy.constants import kilo, pi

import orbital.maneuver
import orbital.utilities as ou
from orbital.utilities import *

J2000 = time.Time('J2000', scale='utc')


class KeplerianElements(RepresentationMixin, object):

    """Defines an orbit using keplerian elements.

    :param a: Semimajor axis [m]
    :param e: Eccentricity [-]
    :param i: Inclination [rad]
    :param raan: Right ascension of ascending node (:math:`\Omega`) [rad]
    :param arg_pe: Argument of periapsis (:math:`\omega`) [rad]
    :param M0: Mean anomaly at `ref_epoch` (:math:`M_{0}`) [rad]
    :param body: Reference body, e.g. earth
    :type body: :py:class:`orbital.bodies.Body`
    :param ref_epoch: Reference epoch
    :type ref_epoch: :py:class:`astropy.time.Time`
    """

    def __init__(self, a=None, e=0, i=0, raan=0, arg_pe=0, M0=0,
                 body=None, ref_epoch=J2000):
        self._a = a
        self.e = e
        self.i = i
        self.raan = raan
        self.arg_pe = arg_pe
        self.M0 = M0

        self._M = M0
        self.body = body
        self.ref_epoch = ref_epoch

        self._t = 0  # This is important because M := M0
        
        super(KeplerianElements, self).__init__()

    @classmethod
    def with_altitude(cls, altitude, body, e=0, i=0, raan=0, arg_pe=0, M0=0,
                      ref_epoch=J2000):
        """Initialise with orbit for a given altitude.

        For eccentric orbits, this is the altitude at the
        reference anomaly, M0
        """
        r = radius_from_altitude(altitude, body)
        a = r * (1 + e * cos(true_anomaly_from_mean(e, M0))) / (1 - e ** 2)

        return cls(a=a, e=e, i=i, raan=raan, arg_pe=arg_pe, M0=M0, body=body,
                   ref_epoch=ref_epoch)

    @classmethod
    def with_period(cls, period, body, e=0, i=0, raan=0, arg_pe=0, M0=0,
                    ref_epoch=J2000):
        """Initialise orbit with a given period."""

        ke = cls(e=e, i=i, raan=raan, arg_pe=arg_pe, M0=M0, body=body,
                 ref_epoch=ref_epoch)

        ke.T = period
        return ke

    @classmethod
    def with_apside_altitudes(cls, alt1, alt2, i=0, raan=0, arg_pe=0, M0=0,
                              body=None, ref_epoch=J2000):
        """Initialise orbit with given apside altitudes."""

        altitudes = [alt1, alt2]
        altitudes.sort()

        pericenter_altitude = altitudes[0]
        apocenter_altitude = altitudes[1]

        apocenter_radius = radius_from_altitude(apocenter_altitude, body)
        pericenter_radius = radius_from_altitude(pericenter_altitude, body)

        a, e = elements_for_apsides(apocenter_radius, pericenter_radius)

        return cls(a=a, e=e, i=i, raan=raan, arg_pe=arg_pe, M0=M0, body=body,
                   ref_epoch=ref_epoch)

    @classmethod
    def with_apside_radii(cls, radius1, radius2, i=0, raan=0, arg_pe=0, M0=0,
                          body=None, ref_epoch=J2000):
        """Initialise orbit with given apside radii."""

        radii = [radius1, radius2]
        radii.sort()

        pericenter_radius = radii[0]
        apocenter_radius = radii[1]

        a, e = elements_for_apsides(apocenter_radius, pericenter_radius)

        return cls(a=a, e=e, i=i, raan=raan, arg_pe=arg_pe, M0=M0, body=body,
                   ref_epoch=ref_epoch)

    @property
    def epoch(self):
        """Current epoch calculated from time since ref_epoch."""
        return self.ref_epoch + time.TimeDelta(self.t, format='sec')

    @epoch.setter
    def epoch(self, value):
        """Set epoch, adjusting current mean anomaly (from which
        other anomalies are calculated).
        """
        t = (value - self.ref_epoch).sec
        self._M = self.M0 + self.n * t
        self._M = ou.mod(self._M, 2 * pi)
        self._t = t

    @property
    def t(self):
        """Time since ref_epoch."""
        return self._t

    @t.setter
    def t(self, value):
        """Set time since ref_epoch, adjusting current mean anomaly (from which
        other anomalies are calculated).
        """
        self._M = self.M0 + self.n * value
        self._M = ou.mod(self._M, 2 * pi)
        self._t = value

    @property
    def M(self):
        """Mean anomaly [rad]."""
        return self._M

    @M.setter
    def M(self, value):
        warnings.warn('Setting anomaly does not set time, use KeplerianElements'
                      '.propagate_anomaly_to() instead.', OrbitalWarning)
        self._M = ou.mod(value, 2 * pi)

    @property
    def E(self):
        """Eccentric anomaly [rad]."""
        return eccentric_anomaly_from_mean(self.e, self._M)

    @E.setter
    def E(self, value):
        warnings.warn('Setting anomaly does not set time, use KeplerianElements'
                      '.propagate_anomaly_to() instead.', OrbitalWarning)
        self._M = mean_anomaly_from_eccentric(self.e, value)

    @property
    def f(self):
        """True anomaly [rad]."""
        return true_anomaly_from_mean(self.e, self._M)

    @f.setter
    def f(self, value):
        warnings.warn('Setting anomaly does not set time, use KeplerianElements'
                      '.propagate_anomaly_to() instead.', OrbitalWarning)
        self._M = mean_anomaly_from_true(self.e, value)

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        """Set semimajor axis and fix M0.

        To fix self.M0, self.n is called. self.n is a function of self.a
        This is safe, because the new value for self._a is set first, then
        self.M0 is fixed.
        """
        self._a = value
        self.M0 = ou.mod(self.M - self.n * self.t, 2 * pi)

    @property
    def r(self):
        """Position vector (:py:class:`orbital.utilities.Position`) [m]."""
        pos = orbit_radius(self.a, self.e, self.f) * self.U
        return Position(x=pos[0], y=pos[1], z=pos[2])

    @property
    def v(self):
        """Velocity vector (:py:class:`orbital.utilities.Velocity`) [m/s]."""
        r_dot = sqrt(self.body.mu / self.a) * (self.e * sin(self.f)) / sqrt(1 - self.e ** 2)
        rf_dot = sqrt(self.body.mu / self.a) * (1 + self.e * cos(self.f)) / sqrt(1 - self.e ** 2)
        vel = r_dot * self.U + rf_dot * self.V
        return Velocity(x=vel[0], y=vel[1], z=vel[2])

    @v.setter
    def v(self, value):
        """Set velocity by altering orbital elements.

        This method uses 3 position variables, and 3 velocity
        variables to set the 6 orbital elements.
        """
        r = self.r
        v = value
        h = angular_momentum(r, v)
        n = node_vector(h)

        mu = self.body.mu
        ev = eccentricity_vector(r, v, mu)

        E = specific_orbital_energy(r, v, mu)

        self.a = -mu / (2 * E)
        self.e = norm(ev)

        SMALL_NUMBER = 1e-15

        # Inclination is the angle between the angular
        # momentum vector and its z component.
        self.i = acos(h.z / norm(h))

        if abs(self.i - 0) < SMALL_NUMBER:
            # For non-inclined orbits, raan is undefined;
            # set to zero by convention
            self.raan = 0
            if abs(self.e - 0) < SMALL_NUMBER:
                # For circular orbits, place periapsis
                # at ascending node by convention
                self.arg_pe = 0
            else:
                # Argument of periapsis is the angle between
                # eccentricity vector and its x component.
                self.arg_pe = acos(ev.x / norm(ev))
        else:
            # Right ascension of ascending node is the angle
            # between the node vector and its x component.
            self.raan = acos(n.x / norm(n))
            if n.y < 0:
                self.raan = 2 * pi - self.raan

            # Argument of periapsis is angle between
            # node and eccentricity vectors.
            self.arg_pe = acos(dot(n, ev) / (norm(n) * norm(ev)))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if abs(self.e - 0) < SMALL_NUMBER:
                if abs(self.i - 0) < SMALL_NUMBER:
                    # True anomaly is angle between position
                    # vector and its x component.
                    self.f = acos(r.x / norm(r))
                    if v.x > 0:
                        self.f = 2 * pi - self.f
                else:
                    # True anomaly is angle between node
                    # vector and position vector.
                    self.f = acos(dot(n, r) / (norm(n) * norm(r)))
                    if dot(n, v) > 0:
                        self.f = 2 * pi - self.f
            else:
                if ev.z < 0:
                    self.arg_pe = 2 * pi - self.arg_pe

                # True anomaly is angle between eccentricity
                # vector and position vector.
                self.f = acos(dot(ev, r) / (norm(ev) * norm(r)))

                if dot(r, v) < 0:
                    self.f = 2 * pi - self.f

        # Fix mean anomaly at epoch for new orbit and position.
        self.M0 = ou.mod(self.M - self.n * self.t, 2 * pi)

        # Now check that the computed properties for position and velocity are
        # reasonably close to the inputs.
        # 1e-4 is a large uncertainty, but we don't want to throw an error
        # within small differences (e.g. 1e-4 m is 0.1 mm)
        if (abs(self.v - v) > 1e-4).any() or (abs(self.r - r) > 1e-4).any():
            raise RuntimeError(
                'Failed to set orbital elements for velocity. Please file a bug'
                ' report at https://github.com/RazerM/orbital/issues')

    @property
    def n(self):
        """Mean motion [rad/s]."""
        return sqrt(self.body.mu / self.a ** 3)

    @n.setter
    def n(self, value):
        """Set mean motion by adjusting semimajor axis."""
        self.a = (self.body.mu / value ** 2) ** (1 / 3)

    @property
    def T(self):
        """Period [s]."""
        return 2 * pi / self.n

    @T.setter
    def T(self, value):
        """Set period by adjusting semimajor axis."""
        self.a = (self.body.mu * value ** 2 / (4 * pi ** 2)) ** (1 / 3)

    @property
    def fpa(self):
        return arctan(self.e * sin(self.f) / (1 + self.e * cos(self.f)))

    def propagate_anomaly_to(self, **kwargs):
        """Propagate to time in future where anomaly is equal to value passed in.

        :param M: Mean anomaly [rad]
        :param E: Eccentricity anomaly [rad]
        :param f: True anomaly [rad]

        This will propagate to a maximum of 1 orbit ahead.

        .. note::

           Only one parameter should be passed in.
        """
        operation = orbital.maneuver.PropagateAnomalyTo(**kwargs)
        self.apply_maneuver(operation)

    def propagate_anomaly_by(self, **kwargs):
        """Propagate to time in future by an amount equal to the anomaly passed in.

        :param M: Mean anomaly [rad]
        :param E: Eccentricity anomaly [rad]
        :param f: True anomaly [rad]

        .. note::

           Only one parameter should be passed in.
        """
        operation = orbital.maneuver.PropagateAnomalyBy(**kwargs)
        self.apply_maneuver(operation)

    def __getattr__(self, attr):
        """Dynamically respond to correct apsis names for given body."""
        if not attr.startswith('__'):
            for apoapsis_name in self.body.apoapsis_names:
                if attr == '{}_radius'.format(apoapsis_name):
                    return self.apocenter_radius
            for periapsis_name in self.body.periapsis_names:
                if attr == '{}_radius'.format(periapsis_name):
                    return self.pericenter_radius
        raise AttributeError(
            "'{name}' object has no attribute '{attr}'"
            .format(name=type(self).__name__, attr=attr))

    def apply_maneuver(self, maneuver, iter=False, copy=False):
        """ Apply maneuver to orbit.

        :param maneuver: Maneuver
        :type maneuver: :py:class:`orbital.maneuver.Maneuver`
        :param bool iter: Return an iterator.
        :param bool copy: Each orbit yielded by the generator will be a copy.

        If :code:`iter=True`, the returned iterator is of each intermediate orbit
        and the next operation, as shown in this table:

        +-------------------------------------+------------------+
        |                Orbit                |    Operation     |
        +=====================================+==================+
        | Original orbit                      | First operation  |
        +-------------------------------------+------------------+
        | Orbit after first operation applied | Second operation |
        +-------------------------------------+------------------+

        The final orbit is not returned, as it is accessible after the method has completed.

        If each orbit returned must not be altered, use :code:`copy=True`
        """
        if isinstance(maneuver, orbital.maneuver.Operation):
            maneuver = orbital.maneuver.Maneuver(maneuver)

        if iter:
            return maneuver.__iapply__(self, copy)
        else:
            if copy:
                raise ValueError('copy can only be True if iter=True')
            maneuver.__apply__(self)

    @property
    def apocenter_radius(self):
        return (1 + self.e) * self.a

    @property
    def pericenter_radius(self):
        return (1 - self.e) * self.a

    @property
    def U(self):
        """Radial direction unit vector."""
        u = self.arg_pe + self.f

        sin_u = sin(u)
        cos_u = cos(u)
        sin_raan = sin(self.raan)
        cos_raan = cos(self.raan)
        cos_i = cos(self.i)

        return np.array(
            [cos_u * cos_raan - sin_u * sin_raan * cos_i,
             cos_u * sin_raan + sin_u * cos_raan * cos_i,
             sin_u * sin(self.i)]
        )

    @property
    def V(self):
        """Transversal in-flight direction unit vector."""
        u = self.arg_pe + self.f

        sin_u = sin(u)
        cos_u = cos(u)
        sin_raan = sin(self.raan)
        cos_raan = cos(self.raan)
        cos_i = cos(self.i)

        return np.array(
            [-sin_u * cos_raan - cos_u * sin_raan * cos_i,
             -sin_u * sin_raan + cos_u * cos_raan * cos_i,
             cos_u * sin(self.i)]
        )

    @property
    def W(self):
        """Out-of-plane direction unit vector."""
        sin_i = sin(self.i)
        return np.array(
            [sin(self.raan) * sin_i,
             -cos(self.raan) * sin_i,
             cos(self.i)]
        )

    @property
    def UVW(self):
        """Calculate U, V, and W vectors simultaneously.

        In situations where all are required, this function may be faster
        but it exists for convenience.
        """
        return uvw_from_elements(self.i, self.raan, self.arg_pe, self.f)

    def __str__(self):
        return ('{name}:\n'
                '    Semimajor axis (a)                           = {a:10.3f} km\n'
                '    Eccentricity (e)                             = {self.e:13.6f}\n'
                '    Inclination (i)                              = {i:8.1f} deg\n'
                '    Right ascension of the ascending node (raan) = {raan:8.1f} deg\n'
                '    Argument of perigee (arg_pe)                 = {arg_pe:8.1f} deg\n'
                '    Mean anomaly at reference epoch (M0)         = {M0:8.1f} deg\n'
                '    Reference epoch (ref_epoch)                  = {self.ref_epoch!s}\n'
                '        Mean anomaly (M)                         = {M:8.1f} deg\n'
                '        Time (t)                                 = {self.t:.1f} s\n'
                '        Epoch (epoch)                            = {self.epoch!s}'
                ).format(
                    name=self.__class__.__name__,
                    self=self,
                    a=self.a / kilo,
                    i=degrees(self.i),
                    raan=degrees(self.raan),
                    arg_pe=degrees(self.arg_pe),
                    M0=degrees(self.M0),
                    M=degrees(self.M))
