# encoding: utf-8
from __future__ import absolute_import, division, print_function

from represent import RepresentationMixin

import orbital.constants as oc


class Body(RepresentationMixin, object):
    r"""Reference body for a Keplerian orbit.

    :param float mass: Mass (:math:`m`) [kg]
    :param float mu: Standard gravitational parameter (:math:`\mu`) [m\ :sup:`3`\ ·s\ :sup:`-2`]
    :param float mean_radius: Mean radius (:math:`r_\text{mean}`) [m]
    :param float equatorial_radius: Equatorial radius (:math:`r_\text{equat}`) [m]
    :param float polar_radius: Polar radius (:math:`r_\text{polar}`) [m]
    :param apoapsis_names: Specific apoapsis name(s) for body. E.g. `apogee` for earth.
    :type apoapsis_names: String, or list of strings.
    :param periapsis_names: Specific periapsis name(s) for body.
    :type apoapsis_names: String, or list of strings.
    :param plot_color: Color understood by Matplotlib, e.g. '#FF0000' or 'r'
    """
    def __init__(self, mass, mu, mean_radius, equatorial_radius, polar_radius, apoapsis_names=None, periapsis_names=None, plot_color=None):
        self.mass = mass
        self.mu = mu
        self.mean_radius = mean_radius
        self.equatorial_radius = equatorial_radius
        self.polar_radius = polar_radius
        self.apoapsis_names = apoapsis_names
        self.periapsis_names = periapsis_names
        self.plot_color = plot_color

        super(Body, self).__init__()

    @property
    def apoapsis_names(self):
        return self._apoapsis_names

    @apoapsis_names.setter
    def apoapsis_names(self, value):
        if isinstance(value, str):
            self._apoapsis_names = [value]
        elif value is None:
            self._apoapsis_names = []
        else:
            self._apoapsis_names = value

    @property
    def periapsis_names(self):
        return self._periapsis_names

    @periapsis_names.setter
    def periapsis_names(self, value):
        if isinstance(value, str):
            self._periapsis_names = [value]
        elif value is None:
            self._periapsis_names = []
        else:
            self._periapsis_names = value

    def __repr__(self):
        # Intercept __repr__ from RepresentationMixin to
        # use orbital.bodies.<planet> for the defaults.
        if __name__ == 'orbital.bodies':
            for name, instance in _defaults.items():
                if self is instance:
                    return __name__ + '.' + name
        return super(Body, self).__repr__()

    def _repr_pretty_(self, p, cycle):
        # Intercept _repr_pretty_ from RepresentationMixin to
        # use orbital.bodies.<planet> for the defaults.
        if __name__ == 'orbital.bodies':
            for name, instance in _defaults.items():
                if self is instance:
                    p.text(__name__ + '.' + name)
                    return

        super(Body, self)._repr_pretty_(p, cycle)

mercury = Body(
    mass=oc.mercury_mass,
    mu=oc.mercury_mu,
    mean_radius=oc.mercury_radius_mean,
    equatorial_radius=oc.mercury_radius_equatorial,
    polar_radius=oc.mercury_radius_polar,
    apoapsis_names='aphermion',
    periapsis_names='perihermion',
    plot_color='#ffd8b0'
)

venus = Body(
    mass=oc.venus_mass,
    mu=oc.venus_mu,
    mean_radius=oc.venus_radius_mean,
    equatorial_radius=oc.venus_radius_equatorial,
    polar_radius=oc.venus_radius_polar,
    apoapsis_names=['apocytherion', 'apocytherean', 'apokrition'],
    periapsis_names=['pericytherion', 'pericytherean', 'perikrition'],
    plot_color='#d58f41'
)

earth = Body(
    mass=oc.earth_mass,
    mu=oc.earth_mu,
    mean_radius=oc.earth_radius_mean,
    equatorial_radius=oc.earth_radius_equatorial,
    polar_radius=oc.earth_radius_polar,
    apoapsis_names='apogee',
    periapsis_names='perigee',
    plot_color='#4e82ff'
)

mars = Body(
    mass=oc.mars_mass,
    mu=oc.mars_mu,
    mean_radius=oc.mars_radius_mean,
    equatorial_radius=oc.mars_radius_equatorial,
    polar_radius=oc.mars_radius_polar,
    apoapsis_names='apoareion',
    periapsis_names='periareion',
    plot_color='#ffc98a'
)

jupiter = Body(
    mass=oc.jupiter_mass,
    mu=oc.jupiter_mu,
    mean_radius=oc.jupiter_radius_mean,
    equatorial_radius=oc.jupiter_radius_equatorial,
    polar_radius=oc.jupiter_radius_polar,
    apoapsis_names=['apozene', 'apojove'],
    periapsis_names=['perizene', 'perijove'],
    plot_color='#ff7726'
)

saturn = Body(
    mass=oc.saturn_mass,
    mu=oc.saturn_mu,
    mean_radius=oc.saturn_radius_mean,
    equatorial_radius=oc.saturn_radius_equatorial,
    polar_radius=oc.saturn_radius_polar,
    apoapsis_names=['apokrone', 'aposaturnium'],
    periapsis_names=['perikrone', 'perisaturnium'],
    plot_color='#ffe296'
)

uranus = Body(
    mass=oc.uranus_mass,
    mu=oc.uranus_mu,
    mean_radius=oc.uranus_radius_mean,
    equatorial_radius=oc.uranus_radius_equatorial,
    polar_radius=oc.uranus_radius_polar,
    apoapsis_names='apouranion',
    periapsis_names='periuranion',
    plot_color='#becaff'
)

neptune = Body(
    mass=oc.neptune_mass,
    mu=oc.neptune_mu,
    mean_radius=oc.neptune_radius_mean,
    equatorial_radius=oc.neptune_radius_equatorial,
    polar_radius=oc.neptune_radius_polar,
    apoapsis_names='apoposeidion',
    periapsis_names='periposeidion',
    plot_color='#8da4ff'
)

_defaults = {
    'mercury': mercury,
    'venus': venus,
    'earth': earth,
    'mars': mars,
    'jupiter': jupiter,
    'saturn': saturn,
    'uranus': uranus,
    'neptune': neptune}
