import orbital.constants as oc


class Body:
    def __init__(self, mass, mu, mean_radius, equatorial_radius, polar_radius, apoapsis_names, periapsis_names):
        self.mass = mass
        self.mu = mu
        self.mean_radius = mean_radius
        self.equatorial_radius = equatorial_radius
        self.polar_radius = polar_radius
        self.apoapsis_names = apoapsis_names
        self.periapsis_names = periapsis_names

    def altitude(self, radius):
        return radius - self.mean_radius

    def orbital_radius(self, altitude):
        return altitude + self.mean_radius

    @property
    def apoapsis_names(self):
        return self._apoapsis_names

    @apoapsis_names.setter
    def apoapsis_names(self, value):
        if isinstance(value, str):
            self._apoapsis_names = [value]
        else:
            self._apoapsis_names = value

    @property
    def periapsis_names(self):
        return self._periapsis_names

    @periapsis_names.setter
    def periapsis_names(self, value):
        if isinstance(value, str):
            self._periapsis_names = [value]
        else:
            self._periapsis_names = value


earth = Body(
    mass=oc.earth_mass,
    mu=oc.earth_mu,
    mean_radius=oc.earth_radius_mean,
    equatorial_radius=oc.earth_radius_equatorial,
    polar_radius=oc.earth_radius_polar,
    apoapsis_names='apogee',
    periapsis_names='perigee'
)

mercury = Body(
    mass=oc.mercury_mass,
    mu=oc.mercury_mu,
    mean_radius=oc.mercury_radius_mean,
    equatorial_radius=oc.mercury_radius_equatorial,
    polar_radius=oc.mercury_radius_polar,
    apoapsis_names='aphermion',
    periapsis_names='perihermion'
)

venus = Body(
    mass=oc.venus_mass,
    mu=oc.venus_mu,
    mean_radius=oc.venus_radius_mean,
    equatorial_radius=oc.venus_radius_equatorial,
    polar_radius=oc.venus_radius_polar,
    apoapsis_names=['apocytherion', 'apocytherean', 'apokrition'],
    periapsis_names=['pericytherion', 'pericytherean', 'perikrition']
)

mars = Body(
    mass=oc.mars_mass,
    mu=oc.mars_mu,
    mean_radius=oc.mars_radius_mean,
    equatorial_radius=oc.mars_radius_equatorial,
    polar_radius=oc.mars_radius_polar,
    apoapsis_names='apoareion',
    periapsis_names='periareion'
)

jupiter = Body(
    mass=oc.jupiter_mass,
    mu=oc.jupiter_mu,
    mean_radius=oc.jupiter_radius_mean,
    equatorial_radius=oc.jupiter_radius_equatorial,
    polar_radius=oc.jupiter_radius_polar,
    apoapsis_names=['apozene', 'apojove'],
    periapsis_names=['perizene', 'perijove']
)

saturn = Body(
    mass=oc.saturn_mass,
    mu=oc.saturn_mu,
    mean_radius=oc.saturn_radius_mean,
    equatorial_radius=oc.saturn_radius_equatorial,
    polar_radius=oc.saturn_radius_polar,
    apoapsis_names=['apokrone', 'aposaturnium'],
    periapsis_names=['perikrone', 'perisaturnium']
)

uranus = Body(
    mass=oc.uranus_mass,
    mu=oc.uranus_mu,
    mean_radius=oc.uranus_radius_mean,
    equatorial_radius=oc.uranus_radius_equatorial,
    polar_radius=oc.uranus_radius_polar,
    apoapsis_names='apouranion',
    periapsis_names='periuranion'
)

neptune = Body(
    mass=oc.neptune_mass,
    mu=oc.neptune_mu,
    mean_radius=oc.neptune_radius_mean,
    equatorial_radius=oc.neptune_radius_equatorial,
    polar_radius=oc.neptune_radius_polar,
    apoapsis_names='periposeidion',
    periapsis_names='apoposeidion'
)
