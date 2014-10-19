from collections import namedtuple
from math import atan2, cos, floor, fmod, isinf, isnan, sin, sqrt

import numpy as np
from scipy.constants import pi

MAX_ITERATIONS = 100

__all__ = [
    'eccentric_anomaly_from_mean',
    'eccentric_anomaly_from_true',
    'mean_anomaly_from_eccentric',
    'mean_anomaly_from_true',
    'mod',
    'orbit_radius',
    'Position',
    'StateVector',
    'true_anomaly_from_eccentric',
    'true_anomaly_from_mean',
    'Velocity'
]


class ConvergenceError(Exception):
    pass


def eccentric_anomaly_from_mean(e, M, tolerance=1e-14):
    """
    Convert mean anomaly to eccentric anomaly

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
            raise ConvergenceError('Did not converge after {n} iterations. M={M!r}'.format(n=MAX_ITERATIONS, M=M))
    return E


def eccentric_anomaly_from_true(e, f):
    E = atan2(sqrt(1 - e ** 2) * sin(f), e + cos(f))
    E = mod(E, 2 * pi)
    return E


def mean_anomaly_from_eccentric(e, E):
    return E - e * sin(E)


def mean_anomaly_from_true(e, f):
    E = eccentric_anomaly_from_true(e, f)
    return E - e * sin(E)


def true_anomaly_from_eccentric(e, E):
    return 2 * atan2(sqrt(1 + e) * sin(E / 2), sqrt(1 - e) * cos(E / 2))


def true_anomaly_from_mean(e, M, tolerance=1e-14):
    E = eccentric_anomaly_from_mean(e, M, tolerance)
    return true_anomaly_from_eccentric(e, E)


def orbit_radius(a, e, f):
    return (a * (1 - e ** 2)) / (1 + e * cos(f))


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
    return (floor(x / y), mod(x, y))

class Position:
    def __init__(self, x, y, z):
        self.array = np.array([x, y, z])

    @property
    def x(self):
        return self.array[0]

    @x.setter
    def x(self, value):
        self.array[0] = value

    @property
    def y(self):
        return self.array[1]

    @y.setter
    def y(self, value):
        self.array[1] = value

    @property
    def z(self):
        return self.array[2]

    @z.setter
    def z(self, value):
        self.array[2] = value

    def __repr__(self):
        return '{name}(x={x!r}, y={y!r}, z={z!r})'.format(name=self.__class__.__name__, x=self.x, y=self.y, z=self.z)

    def __add__(self, other):
        if isinstance(other, numbers.Number):
            return Position(self.x + other, self.y + other, self.z + other)
        elif hasattr(other, 'x') and hasattr(other, 'y') and hasattr(other, 'z'):
            return Position(self.x + other.x, self.y + other.y, self.z + other.z)
        elif len(other) == 3:
            return Position(self.x + other[0], self.y + other[1], self.z + other[2])

    def __truediv__(self, other):
        return Position(self.x / other, self.y / other, self.z / other)

    def __mul__(self, other):
        return Position(self.x * other, self.y * other, self.z * other)

    def __rtruediv__(self, other):
        return Position(other / self.x, other / self.y, other / self.z)

    __rmul__ = __mul__
    __radd__ = __add__

Velocity = namedtuple('Velocity', ['x', 'y', 'z'])

StateVector = namedtuple('StateVector', ['position', 'velocity'])


class MeanAnomaly:

    def __init__(self, M):
        self.M = M
        self.e = None

    @property
    def f(self):
        return true_anomaly_from_mean(self.e, self.M)

    @f.setter
    def f(self, value):
        self.M = mean_anomaly_from_true(self.e, value)

    @property
    def E(self):
        return eccentric_anomaly_from_mean(self.e, self.M)

    @E.setter
    def E(self, value):
        self.M = mean_anomaly_from_eccentric(self.e, value)


class TrueAnomaly:

    def __init__(self, f):
        self.f = f
        self.e = None

    @property
    def M(self):
        return mean_anomaly_from_true(self.e, self.f)

    @M.setter
    def M(self, value):
        self.f = true_anomaly_from_mean(self.e, value)

    @property
    def E(self):
        return eccentric_anomaly_from_true(self.e, self.f)

    @E.setter
    def E(self, value):
        self.f = true_anomaly_from_eccentric(self.e, value)


class EccentricAnomaly:

    def __init__(self, E):
        self.E = E
        self.e = None

    @property
    def M(self):
        return mean_anomaly_from_eccentric(self.e, self.E)

    @M.setter
    def M(self, value):
        self.E = eccentric_anomaly_from_mean(self.e, value)

    @property
    def f(self):
        return true_anomaly_from_eccentric(self.e, self.E)

    @f.setter
    def E(self, value):
        self.E = eccentric_anomaly_from_true(self.e, value)
