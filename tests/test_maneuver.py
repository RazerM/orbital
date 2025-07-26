import unittest

from numpy import radians
from scipy.constants import kilo, mega, pi

from orbital import KeplerianElements, earth
from orbital.maneuver import (
    ChangeApocenterBy,
    ChangePericenterBy,
    SetApocenterAltitudeTo,
    SetApocenterRadiusTo,
    SetPericenterAltitudeTo,
    SetPericenterHere,
    SetPericenterRadiusTo,
)


class TestOperation(unittest.TestCase):
    def setUp(self):
        self.LEO = KeplerianElements.with_altitude(200 * kilo, body=earth)

    def test_set_apside_radii(self):
        set_apocenter = SetApocenterRadiusTo(7 * mega)

        self.LEO.propagate_anomaly_to(M=0)
        self.LEO.apply_maneuver(set_apocenter)
        self.assertAlmostEqual(
            self.LEO.apocenter_radius, set_apocenter.apocenter_radius
        )
        self.assertTrue(self.LEO.e > 0)

        set_pericenter = SetPericenterRadiusTo(6.8 * mega)
        self.LEO.propagate_anomaly_to(M=pi)
        self.LEO.apply_maneuver(set_pericenter)
        self.assertAlmostEqual(
            self.LEO.pericenter_radius, set_pericenter.pericenter_radius
        )
        self.assertAlmostEqual(
            self.LEO.apocenter_radius, set_apocenter.apocenter_radius
        )

    def test_set_apside_radii_velocity(self):
        set_apocenter = SetApocenterRadiusTo(7 * mega)

        self.LEO.propagate_anomaly_to(M=0)
        self.LEO.v += set_apocenter.velocity_delta(self.LEO)
        self.assertAlmostEqual(
            self.LEO.apocenter_radius, set_apocenter.apocenter_radius
        )
        self.assertTrue(self.LEO.e > 0)

        set_pericenter = SetPericenterRadiusTo(6.8 * mega)
        self.LEO.propagate_anomaly_to(M=pi)
        self.LEO.v += set_pericenter.velocity_delta(self.LEO)
        self.assertAlmostEqual(
            self.LEO.pericenter_radius, set_pericenter.pericenter_radius
        )
        self.assertAlmostEqual(
            self.LEO.apocenter_radius, set_apocenter.apocenter_radius
        )

    def test_set_apside_altitudes(self):
        set_apocenter = SetApocenterAltitudeTo(1000 * kilo)
        self.LEO.propagate_anomaly_to(M=0)
        self.LEO.apply_maneuver(set_apocenter)
        self.assertAlmostEqual(
            self.LEO.apocenter_radius,
            self.LEO.body.mean_radius + set_apocenter.apocenter_altitude,
        )
        self.assertTrue(self.LEO.e > 0)

        set_pericenter = SetPericenterAltitudeTo(150 * kilo)
        self.LEO.propagate_anomaly_to(M=pi)
        self.LEO.apply_maneuver(set_pericenter)
        self.assertAlmostEqual(
            self.LEO.pericenter_radius,
            self.LEO.body.mean_radius + set_pericenter.pericenter_altitude,
        )
        self.assertAlmostEqual(
            self.LEO.apocenter_radius,
            self.LEO.body.mean_radius + set_apocenter.apocenter_altitude,
        )

    def test_set_apside_altitudes_velocity(self):
        set_apocenter = SetApocenterAltitudeTo(1000 * kilo)
        self.LEO.propagate_anomaly_to(M=0)
        self.LEO.v += set_apocenter.velocity_delta(self.LEO)
        self.assertAlmostEqual(
            self.LEO.apocenter_radius,
            self.LEO.body.mean_radius + set_apocenter.apocenter_altitude,
        )
        self.assertTrue(self.LEO.e > 0)

        set_pericenter = SetPericenterAltitudeTo(150 * kilo)
        self.LEO.propagate_anomaly_to(M=pi)
        self.LEO.v += set_pericenter.velocity_delta(self.LEO)
        self.assertAlmostEqual(
            self.LEO.pericenter_radius,
            self.LEO.body.mean_radius + set_pericenter.pericenter_altitude,
        )
        self.assertAlmostEqual(
            self.LEO.apocenter_radius,
            self.LEO.body.mean_radius + set_apocenter.apocenter_altitude,
        )

    def test_change_apsides(self):
        set_apocenter = ChangeApocenterBy(20 * kilo)
        old_apocenter_radius = self.LEO.apocenter_radius
        self.LEO.propagate_anomaly_to(M=0)
        self.LEO.apply_maneuver(set_apocenter)
        self.assertAlmostEqual(
            old_apocenter_radius + set_apocenter.delta, self.LEO.apocenter_radius
        )

        set_pericenter = ChangePericenterBy(-20 * kilo)
        old_pericenter_radius = self.LEO.pericenter_radius
        self.LEO.propagate_anomaly_to(M=pi)
        self.LEO.apply_maneuver(set_pericenter)
        self.assertAlmostEqual(
            old_apocenter_radius + set_apocenter.delta, self.LEO.apocenter_radius
        )
        self.assertAlmostEqual(
            old_pericenter_radius + set_pericenter.delta, self.LEO.pericenter_radius
        )

    def test_change_apsides_velocity(self):
        set_apocenter = ChangeApocenterBy(20 * kilo)
        old_apocenter_radius = self.LEO.apocenter_radius
        self.LEO.propagate_anomaly_to(M=0)
        self.LEO.v += set_apocenter.velocity_delta(self.LEO)
        self.assertAlmostEqual(
            old_apocenter_radius + set_apocenter.delta, self.LEO.apocenter_radius
        )

        set_pericenter = ChangePericenterBy(-20 * kilo)
        old_pericenter_radius = self.LEO.pericenter_radius
        self.LEO.propagate_anomaly_to(M=pi)
        self.LEO.v += set_pericenter.velocity_delta(self.LEO)
        self.assertAlmostEqual(
            old_apocenter_radius + set_apocenter.delta, self.LEO.apocenter_radius
        )
        self.assertAlmostEqual(
            old_pericenter_radius + set_pericenter.delta, self.LEO.pericenter_radius
        )

    def test_propagation(self):
        self.LEO.propagate_anomaly_by(M=4 * pi)
        self.assertEqual(self.LEO.t, self.LEO.T * 2)
        self.LEO.propagate_anomaly_by(f=4 * pi)
        self.assertEqual(self.LEO.t, self.LEO.T * 4)
        self.LEO.propagate_anomaly_by(E=4 * pi)
        self.assertEqual(self.LEO.t, self.LEO.T * 6)

        self.LEO.propagate_anomaly_to(M=pi)
        self.assertAlmostEqual(self.LEO.t, self.LEO.T * 6.5)
        self.LEO.propagate_anomaly_to(f=2 * pi)
        self.assertAlmostEqual(self.LEO.t, self.LEO.T * 7)
        self.LEO.propagate_anomaly_to(E=pi)
        self.assertAlmostEqual(self.LEO.t, self.LEO.T * 7.5)

    def test_set_pericenter_here(self):
        self.LEO.propagate_anomaly_to(M=radians(135))
        old_position = self.LEO.r
        old_time = self.LEO.t
        set_pericenter = SetPericenterHere()
        self.LEO.apply_maneuver(set_pericenter)
        self.assertAlmostEqual(old_position.x, self.LEO.r.x)
        self.assertAlmostEqual(old_position.y, self.LEO.r.y)
        self.assertAlmostEqual(old_position.z, self.LEO.r.z)
        self.assertAlmostEqual(old_time, self.LEO.t)


class TestManeuver(unittest.TestCase):
    pass
