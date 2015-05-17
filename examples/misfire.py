##############################################
# Allow this code to be run from the examples
# directory without orbital installed.
from pathlib import Path
import sys

examples_dir = Path(__file__).parent.resolve()
orbital_dir = examples_dir.parent
sys.path.append(str(orbital_dir))
##############################################
from copy import copy

from numpy import cos, degrees, radians, sin, sqrt
from orbital import earth, KeplerianElements
from scipy.constants import kilo
import orbital.utilities as util

try:
    from tabulate import tabulate
except ImportError:
    print("This example requires the 'tabulate' package, please run:\n"
          '$ pip install tabulate')
    exit()

"""
A 800 kg spacecraft is orbiting the Earth on an elliptical orbit with a
semi-major axis of 46000 km and an eccentricity of 0.65, an inclination of 35
degrees, a right ascension of the ascending node of 80 degrees and an argument
of the pericentre of 0.

At 11 hours from the last passage at the pericentre the engine is misfired due
to a malfunction. The thrust has a modulus of 600 N. The telemetry onboard says
that the thrust has an out of plane component and an in plane component
directed against the velocity. The out of plane component is 30 % of the total
thrust. The engine is on for 5 minutes.

Assuming the total variation of velocity is instantaneous, compute the
difference between the nominal position and velocity of the spacecraft 4 hours
after the misfire and its actual position and velocity. Then compute the
difference in orbital parameters.
"""

orbit = KeplerianElements(
    a=46000 * kilo,
    e=0.65,
    i=radians(35),
    raan=radians(80),
    body=earth)

orbit.t += 11 * 60 * 60
print('After 11 h,')
print(orbit)
print('\nOrbital state vector:')
print(orbit.r, orbit.v, '', sep='\n')

thrust_total = 300  # N
mass = 800  # kg

# 30 % of thrust is out of plane
thrust_W = 0.3 * thrust_total

# remaining thrust is directed against the velocity
thrust_in_plane = -sqrt(thrust_total ** 2 - thrust_W ** 2)

# Get in-plane components using flight path angle
thrust_U = thrust_in_plane * sin(orbit.fpa)
thrust_V = thrust_in_plane * cos(orbit.fpa)

thrust_duration = 5 * 60
dv_U = util.impulse_from_finite(thrust_U / mass, duration=thrust_duration)
dv_V = util.impulse_from_finite(thrust_V / mass, duration=thrust_duration)
dv_W = util.impulse_from_finite(thrust_W / mass, duration=thrust_duration)

v_U = dv_U * orbit.U
v_V = dv_V * orbit.V
v_W = dv_W * orbit.W

orbit2 = copy(orbit)

orbit2.v += v_U + v_V + v_W

orbit.t += 4 * 60 * 60
orbit2.t += 4 * 60 * 60

print('After 4 more hours:')
print(tabulate(
    [
        ['a', 'km', orbit.a / kilo, orbit2.a / kilo],
        ['e', '-', orbit.e, orbit2.e],
        ['i', 'deg', degrees(orbit.i), degrees(orbit2.i)],
        ['raan', 'deg', degrees(orbit.raan), degrees(orbit2.raan)],
        ['arg_pe', 'deg', degrees(orbit.arg_pe), degrees(orbit2.arg_pe)],
        ['f', 'deg', degrees(orbit.f), degrees(orbit2.f)]
    ],
    headers=['', 'Unit', 'Nominal', 'Actual'],
    floatfmt='.1f'))

print('\nNominal state vector:')
print(orbit.r, orbit.v, sep='\n')

print('\nActual state vector:')
print(orbit2.r, orbit2.v, sep='\n')
