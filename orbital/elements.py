# encoding: utf-8
from __future__ import absolute_import, division, print_function

import warnings
from datetime import timedelta

import numpy as np
import sgp4.io
import sgp4.propagation
from astropy import time
from numpy import arctan, cos, degrees, sin, sqrt
from represent import ReprMixin
from scipy.constants import kilo, pi
from sgp4.earth_gravity import wgs72

from . import utilities as ou
from .maneuver import (
    Maneuver, Operation, PropagateAnomalyBy, PropagateAnomalyTo)
from .utilities import *

J2000 = time.Time('J2000', scale='utc')

__all__ = [
    'KeplerianElements',
]


class KeplerianElements(ReprMixin, object):

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

    @classmethod
    def from_state_vector(cls, r, v, body, ref_epoch=J2000):
        """Create orbit from given state vector."""
        elements = elements_from_state_vector(r, v, body.mu)

        self = cls(
            a=elements.a,
            e=elements.e,
            i=elements.i,
            raan=elements.raan,
            arg_pe=elements.arg_pe,
            M0=mean_anomaly_from_true(elements.e, elements.f),
            body=body,
            ref_epoch=ref_epoch)

        # Fix mean anomaly at epoch for new orbit and position.
        oldM0 = self.M0
        self.M0 = ou.mod(self.M - self.n * self.t, 2 * pi)
        assert self.M0 == oldM0

        return self

    @classmethod
    def from_tle(cls, line1, line2, body):
        """Create object by parsing TLE using SGP4."""

        # Get state vector at TLE epoch
        sat = sgp4.io.twoline2rv(line1, line2, wgs72)
        r, v = sgp4.propagation.sgp4(sat, 0)
        ref_epoch = time.Time(sat.epoch, scale='utc')

        # Convert km to m
        r, v = np.array(r) * kilo, np.array(v) * kilo

        return cls.from_state_vector(r, v, body=body, ref_epoch=ref_epoch)

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
        r, v = self.r, value
        elements = elements_from_state_vector(r, v, self.body.mu)
        self._a = elements.a
        self.e = elements.e
        self.i = elements.i
        self.raan = elements.raan
        self.arg_pe = elements.arg_pe
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=OrbitalWarning)
            self.f = elements.f

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
        operation = PropagateAnomalyTo(**kwargs)
        self.apply_maneuver(operation)

    def propagate_anomaly_by(self, **kwargs):
        """Propagate to time in future by an amount equal to the anomaly passed in.

        :param M: Mean anomaly [rad]
        :param E: Eccentricity anomaly [rad]
        :param f: True anomaly [rad]

        .. note::

           Only one parameter should be passed in.
        """
        operation = PropagateAnomalyBy(**kwargs)
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
        :type maneuver: :py:class:`maneuver.Maneuver`
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
        if isinstance(maneuver, Operation):
            maneuver = Maneuver(maneuver)

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
                '    Period (T)                                   = {T}\n'
                '    Reference epoch (ref_epoch)                  = {self.ref_epoch!s}\n'
                '        Mean anomaly (M)                         = {M:8.1f} deg\n'
                '        Time (t)                                 = {t}\n'
                '        Epoch (epoch)                            = {self.epoch!s}'
                ).format(
                    name=self.__class__.__name__,
                    self=self,
                    a=self.a / kilo,
                    i=degrees(self.i),
                    raan=degrees(self.raan),
                    arg_pe=degrees(self.arg_pe),
                    M0=degrees(self.M0),
                    M=degrees(self.M),
                    T=timedelta(seconds=self.T),
                    t=timedelta(seconds=self.t))
