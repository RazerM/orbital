from math import acos, cos, sin, sqrt, degrees

import numpy as np
from astropy import time
from numpy.linalg import norm
from scipy import cross, dot, sign
from scipy.constants import pi

import orbital.utilities as ou
from orbital.utilities import *


J2000 = time.Time('J2000', scale='utc')


class KeplerianElements():

    """Defines an orbit using keplerian elements.

    Keplerian Elements:
    a         -- Semimajor axis                        [m]
    e         -- Eccentricity                          [-]
    i         -- Inclination                           [rad]
    raan      -- Right ascension of the ascending node [rad]
    arg_pe    -- Argument of periapsis                 [rad]
    M0        -- Mean anomaly at ref_epoch             [rad]

    Reference frame:
    body      -- Instance of orbital.bodies.Body
    ref_epoch -- astropy.time.Time

    Time-dependent properties:
    t     -- Time since ref_epoch (s)
    epoch -- astropy.time.Time of time since ref_epoch
    M     -- Mean anomaly at time t
    f     -- True anomaly at time t
    E     -- Eccentric anomaly at time t
    """

    def __init__(self, a=None, e=0, i=0, raan=0, arg_pe=0, M0=0,
                 body=None, ref_epoch=J2000):
        self.a = a
        self.e = e
        self.i = i
        self.raan = raan
        self.arg_pe = arg_pe
        self.M0 = M0

        self._M = M0
        self.body = body
        self.ref_epoch = ref_epoch

        self._t = 0  # This is important because M := M0

    @classmethod
    def orbit_with_altitude(cls, altitude, body, e=0, i=0, raan=0,
                            arg_pe=0, M0=0, ref_epoch=J2000):
        """Initialise with orbit for a given altitude.

        For eccentric orbits, this is the altitude at the
        reference anomaly, M0
        """
        r = body.orbital_radius(altitude=altitude)
        a = r * (1 + e * cos(true_anomaly_from_mean(e, M0))) / (1 - e ** 2)
        return cls(a=a, e=e, i=i, raan=raan, arg_pe=arg_pe, M0=M0, body=body,
                   ref_epoch=ref_epoch)

    @classmethod
    def orbit_with_period(cls, period, body, e=0, i=0, raan=0, arg_pe=0,
                          M0=0, ref_epoch=J2000):
        """Initialise orbit with a given period."""
        ke = cls(e=e, i=i, raan=raan, arg_pe=arg_pe, M0=M0, body=body,
                 ref_epoch=ref_epoch)
        ke.T = period
        return ke

    def propagate_anomaly_to(self, M=None, E=None, f=None):
        """Propagate to time in future where anomaly is equal to value passed in.

        This will propagate to a maximum of 1 orbit ahead.
        """
        anomaly_error = ValueError('Only one anomaly parameter can be propagated.')

        mean_old = self.M
        time_old = self.t

        if M is not None:
            if E is not None or f is not None:
                raise anomaly_error

            self.M = M
        elif E is not None:
            if M is not None or f is not None:
                raise anomaly_error

            self.E = E
        elif f is not None:
            if M is not None or E is not None:
                raise anomaly_error

            self.f = f

        # Here, '<=' is used so that for M=0, propagate_anomaly_to(M=0) will
        # propagate by one full orbit
        if self.M <= mean_old:
            mean_new = self.M + 2 * pi
            time_difference = (mean_new - mean_old) / self.n
            self.t = time_old + time_difference

    def propagate_anomaly_by(self, M=None, E=None, f=None):
        anomaly_error = ValueError('Only one anomaly parameter can be propagated.')

        if M is not None:
            if E is not None or f is not None:
                raise anomaly_error

            self.t += M / self.n
        elif E is not None:
            if M is not None or f is not None:
                raise anomaly_error

            orbits, E = ou.divmod(E, 2 * pi)
            self.t += orbits * self.T
            self.t += mean_anomaly_from_eccentric(self.e, E) / self.n
        elif f is not None:
            if M is not None or E is not None:
                raise anomaly_error

            orbits, f = ou.divmod(f, 2 * pi)
            self.t += orbits * self.T
            self.t += mean_anomaly_from_true(self.e, f) / self.n

    def __getattr__(self, attr):
        """Dynamically respond to correct apsis names for given body."""
        for apoapsis_name in self.body.apoapsis_names:
            if attr == '{}_radius'.format(apoapsis_name):
                return self.apocenter_radius
        for periapsis_name in self.body.periapsis_names:
            if attr == '{}_radius'.format(periapsis_name):
                return self.pericenter_radius
        raise AttributeError("'{name}' object has no attribute '{attr}'".format(name=type(self).__name__, attr=attr))

    @property
    def r(self):
        """Position vector [x, y, z] [m]."""
        pos = orbit_radius(self.a, self.e, self.f) * self.U
        return Position(x=pos[0], y=pos[1], z=pos[2])

    @property
    def v(self):
        """Velocity vector [x, y, z] [m/s]."""
        r_dot = sqrt(self.body.mu / self.a) * (self.e * sin(self.f)) / sqrt(1 - self.e ** 2)
        rf_dot = sqrt(self.body.mu / self.a) * (1 + self.e * cos(self.f)) / sqrt(1 - self.e ** 2)
        vel = r_dot * self.U + rf_dot * self.V
        return Velocity(x=vel[0], y=vel[1], z=vel[2])

    @v.setter
    def v(self, value):
        v = np.array([value.x, value.y, value.z])
        h = cross(self.r, v)
        n = cross(np.array([0, 0, 1]), h)

        r = self.r
        r = np.array([r.x, r.y, r.z])
        mu = self.body.mu
        ev = 1 / mu * ((norm(v) ** 2 - mu / norm(r)) * r - dot(r, v) * v)

        E = norm(v) ** 2 / 2 - mu / norm(r)

        self.a = -mu / (2 * E)
        self.e = norm(ev)
        self.i = acos(h[2] / norm(h))

        if self.i == 0:
            self.raan = 0
            self.arg_pe = acos(ev[0] / norm(ev))
        else:
            self.raan = acos(ev[0] / norm(n))
            if n[1] < 0:
                self.raan = 2 * pi - self.raan
            self.arg_pe = acos(dot(n, ev) / (norm(n) * norm(ev)))

        if self.e == 0:
            if self.i == 0:
                self.f = acos(r[0] / norm(r))
                if v[0] > 0:
                    self.f = 2 * pi - self.f
            else:
                self.f = acos(dot(n, r) / (norm(n) * norm(r)))
                if dot(n, v) > 0:
                    self.f = 2 * pi - self.f
        else:
            if ev[2] < 0:
                self.arg_pe = 2 * pi - self.arg_pe
            d = dot(ev, r) / (norm(ev) * norm(r))
            if abs(d) - 1 < 1e-15:
                d = sign(d)
            self.f = acos(d)
            if dot(r, v) < 0:
                self.f = 2 * pi - self.f

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, value):
        self._M = ou.mod(value, 2 * pi)

    @property
    def epoch(self):
        """Current epoch."""
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
    def n(self):
        """Mean motion."""
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
    def apocenter_radius(self):
        return (1 + self. e) * self.a

    @property
    def pericenter_radius(self):
        return (1 - self. e) * self.a

    @property
    def E(self):
        """Eccentric anomaly [rad]."""
        return eccentric_anomaly_from_mean(self.e, self._M)

    @E.setter
    def E(self, value):
        self.M = mean_anomaly_from_eccentric(self.e, value)

    @property
    def f(self):
        """True anomaly [rad]."""
        return true_anomaly_from_mean(self.e, self._M)

    @f.setter
    def f(self, value):
        self.M = mean_anomaly_from_true(self.e, value)

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
            [cos_u * cos_raan - sin_u* sin_raan * cos_i,
             cos_u * sin_raan + sin_u* cos_raan * cos_i,
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

    def __repr__(self):
        return ('{name}(\n'
                '\ta={a!r},\n'
                '\te={e!r},\n'
                '\ti={i!r},\n'
                '\traan={raan!r},\n'
                '\targ_pe={arg_pe!r},\n'
                '\tM0={M0!r})'
               ).format(
                    name=self.__class__.__name__,
                    a=self.a,
                    e=self.e,
                    i=self.i,
                    raan=self.raan,
                    arg_pe=self.arg_pe,
                    M0=self.M0)

    def __str__(self):
        return ('{name}:\n'
                '\tSemimajor axis (a)                           = {a!r},\n'
                '\tEccentricity (e)                             = {e!r} deg,\n'
                '\tInclination (i)                              = {i!r} deg,\n'
                '\tRight ascension of the ascending node (raan) = {raan!r} deg,\n'
                '\tArgument of perigee (arg_pe)                 = {arg_pe!r} deg,\n'
                '\tMean anomaly at epoch (M0)                   = {M0!r} deg'
               ).format(
                    name=self.__class__.__name__,
                    a=degrees(self.a),
                    e=degrees(self.e),
                    i=degrees(self.i),
                    raan=degrees(self.raan),
                    arg_pe=degrees(self.arg_pe),
                    M0=degrees(self.M0)
               )

    def __getstate__(self):
        return {'a': self.a,
                'e': self.e,
                'i': self.i,
                'raan': self.raan,
                'arg_pe': self.arg_pe,
                'M0': self.M0,
                'body': self.body,
                'ref_epoch': self.ref_epoch,
                '_M': self._M,
                '_t': self._t}

    def __setstate__(self, state):
        self.a = state['a']
        self.e = state['e']
        self.i = state['i']
        self.raan = state['raan']
        self.arg_pe = state['arg_pe']
        self.M0 = state['M0']
        self.body = state['body']
        self.ref_epoch = state['ref_epoch']
        self._M = state['_M']
        self._t = state['_t']
