from collections import namedtuple
from math import atan2, floor, fmod, isinf, isnan

import numpy as np
from numpy import sin, cos, sqrt
from scipy.constants import pi

MAX_ITERATIONS = 100

__all__ = [
    'eccentric_anomaly_from_mean',
    'eccentric_anomaly_from_true',
    'mean_anomaly_from_eccentric',
    'mean_anomaly_from_true',
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


class XyzVector(np.ndarray):
    """Subclass of numpy's ndarray with x/y/z initialiser and property syntax."""
    def __new__(cls, x, y, z):
        # Create ndarray and cast to our class type
        obj = np.asarray([x, y, z]).view(cls)

        # Finally, we must return the newly created object:
        return obj

    @property
    def x(self):
        if len(self.shape) == 1:
            return self[0]
        else:
            return self[:,0]

    @x.setter
    def x(self, value):
        if len(self.shape) == 1:
            self[0] = value
        else:
            self[:,0] = value

    @property
    def y(self):
        if len(self.shape) == 1:
            return self[1]
        else:
            return self[:,1]

    @y.setter
    def y(self, value):
        if len(self.shape) == 1:
            self[1] = value
        else:
            self[:,1] = value

    @property
    def z(self):
        if len(self.shape) == 1:
            return self[2]
        else:
            return self[:,2]

    @z.setter
    def z(self, value):
        if len(self.shape) == 1:
            self[2] = value
        else:
            self[:,2] = value

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


class MeanAnomaly:
    """Convenience class for representing an anomaly unambiguously.

    After initialisation, the f, E, or M property can be accessed
    for the anomaly value.
    """
    def __init__(self, M, e=None):
        self.M = M
        self.e = e

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
    """Convenience class for representing an anomaly unambiguously.

    After initialisation, the f, E, or M property can be accessed
    for the anomaly value.
    """
    def __init__(self, f, e=None):
        self.f = f
        self.e = e

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
    """Convenience class for representing an anomaly unambiguously.

    After initialisation, the f, E, or M property can be accessed
    for the anomaly value.
    """
    def __init__(self, E, e=None):
        self.E = E
        self.e = e

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
