from collections import namedtuple
from math import atan2, cos, floor, fmod, isinf, isnan, sin, sqrt
from scipy.constants import pi
import numpy as np

__all__ = [
    'eccentric_anomaly_from_mean',
    'eccentric_anomaly_from_true',
    'mean_anomaly_from_eccentric',
    'mean_anomaly_from_true',
    'true_anomaly_from_eccentric',
    'true_anomaly_from_mean',
    'orbit_radius',
    'PositionVector',
    'VelocityVector',
    'StateVector'
]


class ConvergenceError(Exception):
    pass


def eccentric_anomaly_from_mean(e, M, tolerance=1e-14):
    """
    Convert mean anomaly to eccentric anomaly

    Implemented from http://murison.alpheratz.net/dynamics/twobody/KeplerIterations_summary.pdf
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
        if count == 100:
            raise ConvergenceError('Did not converge after 100 iterations.')
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

PositionVector = namedtuple('Position', ['x', 'y', 'z'])

VelocityVector = namedtuple('Velocity', ['x', 'y', 'z'])

StateVector = namedtuple('StateVector', ['position', 'velocity'])
