##############################################
# Allow this code to be run from the examples
# directory without orbital installed.
from pathlib import Path
import sys

examples_dir = Path(__file__).parent.resolve()
orbital_dir = examples_dir.parent
sys.path.append(str(orbital_dir))
##############################################

from scipy.constants import kilo

from orbital import KeplerianElements
from orbital.bodies import earth
from orbital.maneuver import Maneuver
from orbital.plotting import plot

orbit1 = KeplerianElements.with_altitude(1000 * kilo, body=earth)

man1 = Maneuver.set_apocenter_altitude_to(10000 * kilo)
plot(orbit1, title='Maneuver 1', maneuver=man1)

man2 = Maneuver.set_apocenter_radius_to(22000 * kilo)
plot(orbit1, title='Maneuver 2', maneuver=man2)

man3 = Maneuver.set_pericenter_radius_to(6800 * kilo)
plot(orbit1, title='Maneuver 3', maneuver=man3)

man4 = Maneuver.set_pericenter_altitude_to(500 * kilo)
plot(orbit1, title='Maneuver 4', maneuver=man4)

man5 = Maneuver.change_apocenter_by(1000 * kilo)
plot(orbit1, title='Maneuver 5', maneuver=man5)

man6 = Maneuver.change_pericenter_by(-500 * kilo)
plot(orbit1, title='Maneuver 6', maneuver=man6)

man7 = Maneuver.hohmann_transfer_to_altitude(10000 * kilo)
plot(orbit1, title='Maneuver 7', maneuver=man7)

import matplotlib.pyplot as plt
plt.show()
