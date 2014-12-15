from collections import namedtuple
from contextlib import contextmanager
from math import atan2, floor, fmod, isinf, isnan

import numpy as np
from numpy import cross, cos, dot, sin, sqrt
from numpy.linalg import norm
from scipy.constants import pi

MAX_ITERATIONS = 100

__all__ = [
    'altitude_from_radius',
    'angular_momentum',
    'eccentric_anomaly_from_mean',
    'eccentric_anomaly_from_true',
    'eccentricity_vector',
    'elements_for_apsides',
    'mean_anomaly_from_eccentric',
    'mean_anomaly_from_true',
    'node_vector',
    'orbit_radius',
    'Position',
    'radius_from_altitude',
    'specific_orbital_energy',
    'StateVector',
    'true_anomaly_from_eccentric',
    'true_anomaly_from_mean',
    'uvw_from_elements',
    'Velocity',
]


# Exceptions

class ConvergenceError(Exception):
    pass

# Generators, context managers

@contextmanager
def saved_state(orbit):
    """Context manager to restore orbit upon leaving the block."""
    state = orbit.__getstate__()
    yield
    orbit.__setstate__(state)


def lookahead(collection, fillvalue=None):
    """Generates a series with lookahead to the next item."""
    first = True
    for next_item in collection:
        if first:
            first = False
        else:
            yield current_item, next_item
        current_item = next_item
    yield current_item, fillvalue


# Anomaly conversions

def eccentric_anomaly_from_mean(e, M, tolerance=1e-14):
    """Convert mean anomaly to eccentric anomaly.

    Implemented from [A Practical Method for Solving the Kepler Equation][1]
    by Marc A. Murison from the U.S. Naval Observatory

    [1]: http://murison.alpheratz.net/dynamics/twobody/KeplerIterations_summary.pdf
    """
    Mnorm = fmod(M, 2 * pi)
    E0 = M + (-1 / 2 * e ** 3 + e + (e ** 2 + 3 / 2 * cos(M) * e ** 3) * cos(M)) * sin(M)
    dE = tolerance + 1
    count = 0
    while dE > tolerance:
        t1 = cos(E0)
        t2 = -1 + e * t1
        t3 = sin(E0)
        t4 = e * t3
        t5 = -E0 + t4 + Mnorm
        t6 = t5 / (1 / 2 * t5 * t4 / t2 + t2)
        E = E0 - t5 / ((1 / 2 * t3 - 1 / 6 * t1 * t6) * e * t6 + t2)
        dE = abs(E - E0)
        E0 = E
        count += 1
        if count == MAX_ITERATIONS:
            raise ConvergenceError('Did not converge after {n} iterations. (e={e!r}, M={M!r})'.format(n=MAX_ITERATIONS, e=e, M=M))
    return E


def eccentric_anomaly_from_true(e, f):
    """Convert true anomaly to eccentric anomaly."""
    E = atan2(sqrt(1 - e ** 2) * sin(f), e + cos(f))
    E = mod(E, 2 * pi)
    return E


def mean_anomaly_from_eccentric(e, E):
    """Convert eccentric anomaly to mean anomaly."""
    return E - e * sin(E)


def mean_anomaly_from_true(e, f):
    """Convert true anomaly to mean anomaly."""
    E = eccentric_anomaly_from_true(e, f)
    return E - e * sin(E)


def true_anomaly_from_eccentric(e, E):
    """Convert eccentric anomaly to true anomaly."""
    return 2 * atan2(sqrt(1 + e) * sin(E / 2), sqrt(1 - e) * cos(E / 2))


def true_anomaly_from_mean(e, M, tolerance=1e-14):
    """Convert mean anomaly to true anomaly."""
    E = eccentric_anomaly_from_mean(e, M, tolerance)
    return true_anomaly_from_eccentric(e, E)


# Orbital element helper functions

def orbit_radius(a, e, f):
    """Calculate scalar orbital radius."""
    return (a * (1 - e ** 2)) / (1 + e * cos(f))


def elements_for_apsides(apocenter_radius, pericenter_radius):
    """Calculate planar orbital elements for given apside radii."""
    ra = apocenter_radius
    rp = pericenter_radius

    a = (ra + rp) / 2
    e = (ra - rp) / (ra + rp)
    return a, e


def uvw_from_elements(i, raan, arg_pe, f):
    """Return U, V, W unit vectors.

    :param float i: Inclination (:math:`i`) [rad]
    :param float raan:  Right ascension of ascending node (:math:`\Omega`) [rad]
    :param float arg_pe: Argument of periapsis (:math:`\omega`) [rad]
    :param float f: True anomaly (:math:`f`) [rad]
    :return: Radial direction unit vector (:math:`U`)
    :return: Transversal (in-flight) direction unit vector (:math:`V`)
    :return: Out-of-plane direction unit vector (:math:`W`)
    :rtype: :py:class:`numpy.ndarray`
    """
    u = arg_pe + f

    sin_u = sin(u)
    cos_u = cos(u)
    sin_raan = sin(raan)
    cos_raan = cos(raan)
    sin_i = sin(i)
    cos_i = cos(i)

    U = np.array(
        [cos_u * cos_raan - sin_u * sin_raan * cos_i,
         cos_u * sin_raan + sin_u * cos_raan * cos_i,
         sin_u * sin_i]
    )

    V = np.array(
        [-sin_u * cos_raan - cos_u * sin_raan * cos_i,
         -sin_u * sin_raan + cos_u * cos_raan * cos_i,
         cos_u * sin_i]
    )

    W = np.array(
        [sin_raan * sin_i,
         -cos_raan * sin_i,
         cos_i]
    )

    return U, V, W


def angular_momentum(position, velocity):
    """Return angular momentum.

    :param position: Position (r) [m]
    :type position: :py:class:`~orbital.utilities.Position`
    :param velocity: Velocity (v) [m/s]
    :type velocity: :py:class:`~orbital.utilities.Velocity`
    :return: Angular momentum (h) [N·m·s]
    :rtype: :py:class:`~orbital.utilities.XyzVector`
    """
    return XyzVector.from_array(np.cross(position, velocity))


def node_vector(angular_momentum):
    """Return node vector.

    :param angular_momentum: Angular momentum (h) [N·m·s]
    :type angular_momentum: :py:class:`numpy.ndarray`
    :return: Node vector (n) [N·m·s]
    :rtype: :py:class:`~orbital.utilities.XyzVector`
    """
    return XyzVector.from_array(np.cross([0, 0, 1], angular_momentum))


def eccentricity_vector(position, velocity, mu):
    """Return eccentricity vector.

    :param position: Position (r) [m]
    :type position: :py:class:`~orbital.utilities.Position`
    :param velocity: Velocity (v) [m/s]
    :type velocity: :py:class:`~orbital.utilities.Velocity`
    :param mu: Standard gravitational parameter (:math:`\mu`) [m\ :sup:`3`\ ·s\ :sup:`-2`]
    :type mu: float
    :return: Eccentricity vector (ev) [-]
    :rtype: :py:class:`~orbital.utilities.XyzVector`
    """

    # This isn't required, but get base arrays so that return value isn't an
    # instance of Position().
    r = position.__array__()
    v = velocity.__array__()
    ev = 1 / mu * ((norm(v) ** 2 - mu / norm(r)) * r - dot(r, v) * v)
    return XyzVector.from_array(ev)


def specific_orbital_energy(position, velocity, mu):
    """Return specific orbital energy.

    :param position: Position (r) [m]
    :type position: :py:class:`~orbital.utilities.Position`
    :param velocity: Velocity (v) [m/s]
    :type velocity: :py:class:`~orbital.utilities.Velocity`
    :param mu: Standard gravitational parameter (:math:`\mu`) [m\ :sup:`3`\ ·s\ :sup:`-2`]
    :type mu: float
    :return: Specific orbital energy (E) [J/kg]
    :rtype: float
    """
    r = position
    v = velocity
    return norm(v) ** 2 / 2 - mu / norm(r)


# User helper functions

def radius_from_altitude(altitude, body):
    return altitude + body.mean_radius


def altitude_from_radius(radius, body):
    return radius - body.mean_radius


# Math functions

def mod(x, y):
    """Return the modulus after division of x by y.

    Python's x % y is best suited for integers, and math.mod returns with the
    sign of x.

    This function is modelled after Matlab's mod() function.
    """

    if isnan(x) or isnan(y):
        return float('NaN')

    if isinf(x):
        raise ValueError('math domain error')

    if isinf(y):
        return x

    if y == 0:
        return x

    n = floor(x / y)
    return x - n * y


def divmod(x, y):
    """Return quotient and remainder from division of x by y."""
    return (floor(x / y), mod(x, y))


# Objects for package

class XyzVector(np.ndarray):
    """Subclass of numpy's ndarray with x/y/z initialiser and property syntax."""
    def __new__(cls, x, y, z):
        # Create ndarray and cast to our class type
        obj = np.asarray(np.hstack([x, y, z])).view(cls)

        # Finally, we must return the newly created object:
        return obj

    @classmethod
    def from_array(cls, array):
        return cls(array[0], array[1], array[2])


    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = value

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, value):
        self[1] = value

    @property
    def z(self):
        return self[2]

    @z.setter
    def z(self, value):
        self[2] = value

    def __str__(self):
        """Override superclass __str__"""
        return self.__repr__()

    def __repr__(self):
        return '{name}(x={x!r}, y={y!r}, z={z!r})'.format(name=self.__class__.__name__, x=self.x, y=self.y, z=self.z)


class Position(XyzVector):
    pass


class Velocity(XyzVector):
    pass


StateVector = namedtuple('StateVector', ['position', 'velocity'])

# Other

class Anomaly:
    """This package allows an anomaly to be represented and retrieved
    unambiguously.
    """
    def __init__(self, **kwargs):
        super().__init__()

        # The defaults
        valid_args = set(['M', 'E', 'f'])

        extra_args = set(kwargs.keys()) - valid_args

        # Check for invalid keywords
        if extra_args:
            raise TypeError('Invalid kwargs: ' + ', '.join(list(extra_args)))

        # Ensure a valid keyword was passed
        if not kwargs:
            raise TypeError('Required argument missing.')

        # Ensure only one keyword was passed, but allow other 2 anomaly
        # parameters to be None.
        if sum(1 for x in kwargs.values() if x is not None) > 1:
            raise ValueError('Only one anomaly parameter can be set.')

        # Now remove the superfluous None values.
        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        self.key, self.anomaly = kwargs.popitem()

    def M(self, e):
        if self.key == 'M':
            return self.anomaly
        elif self.key == 'E':
            return mean_anomaly_from_eccentric(e, self.anomaly)
        elif self.key == 'f':
            return true_anomaly_from_eccentric(e, self.anomaly)

    def E(self, e):
        if self.key == 'M':
            return eccentric_anomaly_from_mean(e, self.anomaly)
        elif self.key == 'E':
            return self.anomaly
        elif self.key == 'f':
            return eccentric_anomaly_from_true(e, self.anomaly)

    def f(self, e):
        if self.key == 'M':
            return true_anomaly_from_mean(e, self.anomaly)
        elif self.key == 'E':
            return true_anomaly_from_eccentric(e, self.anomaly)
        elif self.key == 'f':
            return self.anomaly
