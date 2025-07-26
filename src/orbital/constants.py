from datetime import timedelta as _timedelta

from scipy.constants import kilo as _kilo

# IAU 2009 System of Astronomical Constants
# Updated as of 2014

constant_of_gravitation = 6.67428e-11  # m^3 kg^-1 s^-2
solar_mass_parameter = 1.32712440041e20  # m^3 s^-2

sun_mass = solar_mass_parameter / constant_of_gravitation
geocentric_gravitational_constant = earth_mu = 3.986004415e14  # m^3 s^-2

earth_mass = geocentric_gravitational_constant / constant_of_gravitation

mercury_mass = sun_mass / 6.0236e6
venus_mass = sun_mass / 4.08523710e5
mars_mass = sun_mass / 3.09870359e6
jupiter_mass = sun_mass / 1.047348644e3
saturn_mass = sun_mass / 3.4979018e3
uranus_mass = sun_mass / 2.290298e4
neptune_mass = sun_mass / 1.941226e4

sun_radius_equatorial = 696000 * _kilo
mercury_radius_equatorial = 2439.7 * _kilo
venus_radius_equatorial = 6051.8 * _kilo
earth_radius_equatorial = 6378.1366 * _kilo
mars_radius_equatorial = 3396.19 * _kilo
jupiter_radius_equatorial = 71492 * _kilo
saturn_radius_equatorial = 60268 * _kilo
uranus_radius_equatorial = 25559 * _kilo
neptune_radius_equatorial = 24764 * _kilo

mercury_mu = mercury_mass * constant_of_gravitation
venus_mu = venus_mass * constant_of_gravitation
mars_mu = mars_mass * constant_of_gravitation
jupiter_mu = jupiter_mass * constant_of_gravitation
saturn_mu = saturn_mass * constant_of_gravitation
uranus_mu = uranus_mass * constant_of_gravitation
neptune_mu = neptune_mass * constant_of_gravitation

mercury_radius_polar = mercury_radius_mean = mercury_radius_equatorial
venus_radius_polar = venus_radius_mean = venus_radius_equatorial

# The following constants are not from IAU
earth_radius_mean = 6371.0 * _kilo
earth_radius_polar = 6356.8 * _kilo

mars_radius_mean = 3389.5 * _kilo
mars_radius_polar = 3376.2 * _kilo

jupiter_radius_mean = 69911 * _kilo
jupiter_radius_polar = 66854 * _kilo

saturn_radius_mean = 58232 * _kilo
saturn_radius_polar = 54364 * _kilo

uranus_radius_mean = 25362 * _kilo
uranus_radius_polar = 24973 * _kilo

neptune_radius_mean = 24622 * _kilo
neptune_radius_polar = 24341 * _kilo

# 4.1 s, 56 minutes, 23 hours
earth_sidereal_day = _timedelta(hours=23, minutes=56, seconds=4.1).total_seconds()
