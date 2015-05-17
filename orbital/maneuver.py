# encoding: utf-8
from __future__ import absolute_import, division, print_function

import warnings
from copy import copy

from represent import ReprHelperMixin
from numpy import allclose as almost_equal
from scipy.constants import pi

import orbital.utilities as ou
from orbital.utilities import (
    elements_for_apsides, mean_anomaly_from_eccentric, mean_anomaly_from_true,
    radius_from_altitude, saved_state)

_copy = copy  # Used when a keyword argument is called 'copy'

__all__ = [
    'ChangeApocenterBy',
    'ChangeInclinationBy',
    'ChangePericenterBy',
    'Circularise',
    'Maneuver',
    'PropagateAnomalyBy',
    'PropagateAnomalyTo',
    'SetApocenterAltitudeTo',
    'SetApocenterRadiusTo',
    'SetInclinationTo',
    'SetPericenterAltitudeTo',
    'SetPericenterHere',
    'SetPericenterRadiusTo',
]


class Operation(object):
    """Base class for orbital operations.

    Maneuvers are used by the user to construct operations, because operations
    contain assumptions about when they can be used. For example, it would not
    make sense to apply an operation to raise the apocenter, if the anomaly was
    not currently at the pericenter.

    Subclasses can implement an __apply__ method as a shortcut method, rather
    than applying a velocity change directly, for example.
    """
    def plot(self, orbit, plotter, next_operation=None):
        """Convenience method to call __plot__ if defined by subclass.

        next_operation allows us to make decisions about what to plot,
        e.g. plotting a partial orbit during a transfer.
        """
        if hasattr(self, '__plot__') and callable(getattr(self, '__plot__')):
            self.__plot__(orbit, plotter, next_operation)

    def __add__(self, other):
        if isinstance(other, Operation):
            return Maneuver([copy(self), copy(other)])
        elif isinstance(other, Maneuver):
            return Maneuver([copy(self)] + other.operations)
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Maneuver):
            return Maneuver(other.operations + [copy(self)])
        else:
            return NotImplemented


class ImpulseOperation(Operation):
    def __init__(self):
        super(ImpulseOperation, self).__init__()

    def velocity_delta(self):
        """Return velocity delta of impulse."""
        raise NotImplementedError(
            'Subclasses of {}.{} must implement {}'
            .format(__name__, __class__.__name__, self.velocity_delta.__name__))


class TimeOperation(Operation):
    def __init__(self):
        super(TimeOperation, self).__init__()

    def time_delta(self, orbit):
        """Return the time delta to propagate the orbit by.

        :param orbit: Orbit
        :type orbit: :py:class:`orbital.elements.KeplerianElements`
        """
        raise NotImplementedError(
            'Subclasses of {}.{} must implement {}'
            .format(__name__, __class__.__name__, self.time_delta.__name__))


class SetApocenterRadiusTo(ReprHelperMixin, ImpulseOperation):
    """Operation for setting apocenter radius. At time of application, orbit
    position must be at pericenter.
    """
    def __init__(self, apocenter_radius):
        super(SetApocenterRadiusTo, self).__init__()
        self.apocenter_radius = apocenter_radius

    def __apply__(self, orbit):
        a, e = elements_for_apsides(self.apocenter_radius,
                                    orbit.pericenter_radius)
        orbit.a = a
        orbit.e = e

        # Ensure other elements are changed if necessary
        orbit.v = orbit.v

    def __plot__(self, orbit, plotter, next_operation=None):
        if orbit.apocenter_radius > self.apocenter_radius:
            label = 'Lowered apocenter'
        else:
            label = 'Raised apocenter'
        self.__apply__(orbit)

        with saved_state(orbit):
            if (next_operation is not None and
                    isinstance(next_operation, TimeOperation)):
                orbit.apply_maneuver(next_operation)
                f2 = orbit.f
                if f2 == 0:
                    f2 = 2 * pi
            else:
                f2 = 2 * pi

        plotter._plot_orbit(orbit, f1=0, f2=f2, label=label)

    def velocity_delta(self, orbit):
        with saved_state(orbit):
            # get velocity at pericenter
            orbit.propagate_anomaly_to(M=0)
            old_velocity = orbit.v

            a, e = elements_for_apsides(self.apocenter_radius,
                                        orbit.pericenter_radius)
            orbit.a = a
            orbit.e = e

            new_velocity = orbit.v

        return new_velocity - old_velocity

    def _repr_helper_(self, r):
        r.positional_from_attr('apocenter_radius')


class SetApocenterAltitudeTo(ReprHelperMixin, ImpulseOperation):
    """Operation for setting apocenter altitude. At time of application, orbit
    position must be at pericenter.
    """
    def __init__(self, apocenter_altitude):
        super(SetApocenterAltitudeTo, self).__init__()
        self.apocenter_altitude = apocenter_altitude

    def __apply__(self, orbit):
        apocenter_radius = orbit.body.mean_radius + self.apocenter_altitude
        a, e = elements_for_apsides(apocenter_radius,
                                    orbit.pericenter_radius)
        orbit.a = a
        orbit.e = e

        # Ensure other elements are changed if necessary
        orbit.v = orbit.v

    def __plot__(self, orbit, plotter, next_operation=None):
        radius = radius_from_altitude(self.apocenter_altitude, orbit.body)
        if orbit.apocenter_radius > radius:
            label = 'Lowered apocenter'
        else:
            label = 'Raised apocenter'
        self.__apply__(orbit)

        with saved_state(orbit):
            if (next_operation is not None and
                    isinstance(next_operation, TimeOperation)):
                orbit.apply_maneuver(next_operation)
                f2 = orbit.f
                if f2 == 0:
                    f2 = 2 * pi
            else:
                f2 = 2 * pi

        plotter._plot_orbit(orbit, f1=0, f2=f2, label=label)

    def velocity_delta(self, orbit):
        with saved_state(orbit):
            # get velocity at pericenter
            orbit.propagate_anomaly_to(M=0)
            old_velocity = orbit.v

            apocenter_radius = orbit.body.mean_radius + self.apocenter_altitude
            a, e = elements_for_apsides(apocenter_radius,
                                        orbit.pericenter_radius)
            orbit.a = a
            orbit.e = e

            new_velocity = orbit.v

        return new_velocity - old_velocity

    def _repr_helper_(self, r):
        r.positional_from_attr('apocenter_altitude')


class ChangeApocenterBy(ReprHelperMixin, ImpulseOperation):
    """Operation for changing apocenter radius. At time of application, orbit
    position must be at pericenter.
    """
    def __init__(self, delta):
        super(ChangeApocenterBy, self).__init__()
        self.delta = delta

    def __apply__(self, orbit):
        a, e = elements_for_apsides(orbit.apocenter_radius + self.delta,
                                    orbit.pericenter_radius)
        orbit.a = a
        orbit.e = e

        # Ensure other elements are changed if necessary
        orbit.v = orbit.v

    def __plot__(self, orbit, plotter, next_operation=None):
        if self.delta < 0:
            label = 'Lowered apocenter'
        else:
            label = 'Raised apocenter'
        self.__apply__(orbit)

        with saved_state(orbit):
            if (next_operation is not None and
                    isinstance(next_operation, TimeOperation)):
                orbit.apply_maneuver(next_operation)
                f2 = orbit.f
                if f2 == 0:
                    f2 = 2 * pi
            else:
                f2 = 2 * pi

        plotter._plot_orbit(orbit, f1=0, f2=f2, label=label)

    def velocity_delta(self, orbit):
        with saved_state(orbit):
            # get velocity at pericenter
            orbit.propagate_anomaly_to(M=0)
            old_velocity = orbit.v

            a, e = elements_for_apsides(orbit.apocenter_radius + self.delta,
                                        orbit.pericenter_radius)
            orbit.a = a
            orbit.e = e

            new_velocity = orbit.v

        return new_velocity - old_velocity

    def _repr_helper_(self, r):
        r.positional_from_attr('delta')


class SetPericenterRadiusTo(ReprHelperMixin, ImpulseOperation):
    """Operation for setting pericenter radius. At time of application, orbit
    position must be at apocenter.
    """
    def __init__(self, pericenter_radius):
        super(SetPericenterRadiusTo, self).__init__()
        self.pericenter_radius = pericenter_radius

    def __apply__(self, orbit):
        a, e = elements_for_apsides(orbit.apocenter_radius,
                                    self.pericenter_radius)
        orbit.a = a
        orbit.e = e

        # Ensure other elements are changed if necessary
        orbit.v = orbit.v

    def __plot__(self, orbit, plotter, next_operation=None):
        if orbit.pericenter_radius > self.pericenter_radius:
            label = 'Lowered pericenter'
        else:
            label = 'Raised pericenter'
        self.__apply__(orbit)

        with saved_state(orbit):
            if (next_operation is not None and
                    isinstance(next_operation, TimeOperation)):
                orbit.apply_maneuver(next_operation)
                f2 = orbit.f
                if almost_equal(f2, pi):
                    f2 = pi + 2 * pi
            else:
                f2 = pi + 2 * pi

        plotter._plot_orbit(orbit, f1=pi, f2=f2, label=label)

    def velocity_delta(self, orbit):
        with saved_state(orbit):
            # get velocity at apocenter
            orbit.propagate_anomaly_to(M=pi)
            old_velocity = orbit.v

            a, e = elements_for_apsides(orbit.apocenter_radius,
                                        self.pericenter_radius)
            orbit.a = a
            orbit.e = e

            new_velocity = orbit.v

        return new_velocity - old_velocity

    def _repr_helper_(self, r):
        r.positional_from_attr('pericenter_radius')


class SetPericenterAltitudeTo(ReprHelperMixin, ImpulseOperation):
    """Operation for setting pericenter altitude. At time of application, orbit
    position must be at apocenter.
    """
    def __init__(self, pericenter_altitude):
        super(SetPericenterAltitudeTo, self).__init__()
        self.pericenter_altitude = pericenter_altitude

    def __apply__(self, orbit):
        pericenter_radius = orbit.body.mean_radius + self.pericenter_altitude
        a, e = elements_for_apsides(orbit.apocenter_radius,
                                    pericenter_radius)
        orbit.a = a
        orbit.e = e

        # Ensure other elements are changed if necessary
        orbit.v = orbit.v

    def __plot__(self, orbit, plotter, next_operation=None):
        radius = radius_from_altitude(self.pericenter_altitude, orbit.body)
        if orbit.pericenter_radius > radius:
            label = 'Lowered pericenter'
        else:
            label = 'Raised pericenter'
        self.__apply__(orbit)

        with saved_state(orbit):
            if (next_operation is not None and
                    isinstance(next_operation, TimeOperation)):
                orbit.apply_maneuver(next_operation)
                f2 = orbit.f
                if almost_equal(f2, pi):
                    f2 = pi + 2 * pi
            else:
                f2 = pi + 2 * pi

        plotter._plot_orbit(orbit, f1=0, f2=f2, label=label)

    def velocity_delta(self, orbit):
        with saved_state(orbit):
            # get velocity at apocenter
            orbit.propagate_anomaly_to(M=pi)
            old_velocity = orbit.v

            pericenter_radius = orbit.body.mean_radius + self.pericenter_altitude
            a, e = elements_for_apsides(orbit.apocenter_radius,
                                        pericenter_radius)
            orbit.a = a
            orbit.e = e

            new_velocity = orbit.v

        return new_velocity - old_velocity

    def _repr_helper_(self, r):
        r.positional_from_attr('pericenter_altitude')


class ChangePericenterBy(ReprHelperMixin, ImpulseOperation):
    """Operation for changing pericenter. At time of application, orbit
    position must be at apocenter.
    """
    def __init__(self, delta):
        super(ChangePericenterBy, self).__init__()
        self.delta = delta

    def __apply__(self, orbit):
        a, e = elements_for_apsides(orbit.apocenter_radius,
                                    orbit.pericenter_radius + self.delta)
        orbit.a = a
        orbit.e = e

        # Ensure other elements are changed if necessary
        orbit.v = orbit.v

    def __plot__(self, orbit, plotter, next_operation=None):
        if self.delta < 0:
            label = 'Lowered pericenter'
        else:
            label = 'Raised pericenter'
        self.__apply__(orbit)

        with saved_state(orbit):
            if (next_operation is not None and
                    isinstance(next_operation, TimeOperation)):
                orbit.apply_maneuver(next_operation)
                f2 = orbit.f
                if almost_equal(f2, pi):
                    f2 = pi + 2 * pi
            else:
                f2 = pi + 2 * pi

        plotter._plot_orbit(orbit, f1=0, f2=f2, label=label)

    def velocity_delta(self, orbit):
        with saved_state(orbit):
            # get velocity at apocenter
            orbit.propagate_anomaly_to(M=pi)
            old_velocity = orbit.v

            a, e = elements_for_apsides(orbit.apocenter_radius,
                                        orbit.pericenter_radius + self.delta)
            orbit.a = a
            orbit.e = e

            new_velocity = orbit.v

        return new_velocity - old_velocity

    def _repr_helper_(self, r):
        r.positional_from_attr('delta')


class SetInclinationTo(ReprHelperMixin, ImpulseOperation):
    """Operation for setting inclination. At time of application, orbit
    position must be at the ascending or descending node.
    """
    def __init__(self, inclination):
        super(SetInclinationTo, self).__init__()
        self.inclination = inclination

    def __apply__(self, orbit):
        orbit.i = self.inclination

    def __plot__(self, orbit, plotter, next_operation=None):
        self.__apply__(orbit)
        plotter._plot_orbit(orbit, label='Changed inclination')

    def velocity_delta(self, orbit):
        with saved_state(orbit):
            orbit.f = 2 * pi - orbit.arg_pe
            old_velocity = orbit.v

            self.__apply__(orbit)
            new_velocity = orbit.v

        return new_velocity - old_velocity

    def _repr_helper_(self, r):
        r.positional_from_attr('inclination')


class ChangeInclinationBy(ReprHelperMixin, ImpulseOperation):
    """Operation for changing inclination. At time of application, orbit
    position must be at the ascending or descending node.
    """
    def __init__(self, delta):
        super(ChangeInclinationBy, self).__init__()
        self.delta = delta

    def __apply__(self, orbit):
        orbit.i += self.delta

    def __plot__(self, orbit, plotter, next_operation=None):
        self.__apply__(orbit)
        plotter._plot_orbit(orbit, label='Changed inclination')

    def velocity_delta(self, orbit):
        with saved_state(orbit):
            orbit.f = 2 * pi - orbit.arg_pe
            old_velocity = orbit.v

            self.__apply__(orbit)
            new_velocity = orbit.v

        return new_velocity - old_velocity

    def _repr_helper_(self, r):
        r.positional_from_attr('delta')


class PropagateAnomalyTo(ReprHelperMixin, TimeOperation):
    """Operation for propagating to time in future where anomaly is equal to
    value passed in.

    One (and only one) of these parameters must be passed in:

    :param float M: Mean anomaly
    :param float E: Eccentric anomaly
    :param float f: True anomaly
    """
    def __init__(self, **kwargs):
        super(PropagateAnomalyTo, self).__init__()

        # The defaults
        valid_args = {'M', 'E', 'f'}

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
            raise ValueError('Only one anomaly parameter can be propagated.')

        # Now remove the superfluous None values.
        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        self.key, self.anomaly = kwargs.popitem()

    def time_delta(self, orbit):
        if self.key == 'f':
            M = mean_anomaly_from_true(orbit.e, self.anomaly)
        elif self.key == 'E':
            M = mean_anomaly_from_eccentric(orbit.e, self.anomaly)
        elif self.key == 'M':
            M = self.anomaly

        # Propagate one orbit if destination is 'behind' current state.
        if M < orbit.M:
            M += 2 * pi

        return (M - orbit.M) / orbit.n

    def __plot__(self, orbit, plotter, next_operation=None):
        f1 = orbit.f
        orbit.t += self.time_delta(orbit)
        f2 = orbit.f

        plotter._plot_position(orbit, f2, propagated=True)

    def _repr_helper_(self, r):
        r.keyword_with_value(self.key, self.anomaly)


class PropagateAnomalyBy(ReprHelperMixin, TimeOperation):
    """Operation for propagating anomaly by a given amount.

    One (and only one) of these parameters must be passed in:

    :param float M: Mean anomaly
    :param float E: Eccentric anomaly
    :param float f: True anomaly
    """
    def __init__(self, **kwargs):
        super(PropagateAnomalyBy, self).__init__()

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
            raise ValueError('Only one anomaly parameter can be propagated.')

        # Now remove the superfluous None values.
        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        self.key, self.anomaly = kwargs.popitem()

    def time_delta(self, orbit):
        if self.key == 'f':
            orbits, f = ou.divmod(self.anomaly, 2 * pi)
            M = mean_anomaly_from_true(orbit.e, f)
            return orbits * orbit.T + M / orbit.n
        elif self.key == 'E':
            orbits, E = ou.divmod(self.anomaly, 2 * pi)
            M = mean_anomaly_from_eccentric(orbit.e, E)
            return orbits * orbit.T + M / orbit.n
        elif self.key == 'M':
            return self.anomaly / orbit.n

    def __plot__(self, orbit, plotter, next_operation=None):
        f1 = orbit.f
        orbit.t += self.time_delta(orbit)
        f2 = orbit.f

        plotter._plot_position(orbit, f2, propagated=True)

    def _repr_helper_(self, r):
        r.keyword_with_value(self.key, self.anomaly)


class Circularise(ReprHelperMixin, ImpulseOperation):
    """Operation for circularising an orbit."""
    def __init__(self, raise_pericenter=True):
        """Assumptions: anomaly is at the correct apside."""
        super(Circularise, self).__init__()
        self.raise_pericenter = raise_pericenter

    def __apply__(self, orbit):
        if self.raise_pericenter:
            radius = orbit.apocenter_radius
        else:
            radius = orbit.pericenter_radius

        a, e = elements_for_apsides(radius, radius)
        orbit.a = a
        orbit.e = e

        # Ensure other elements are changed if necessary
        orbit.v = orbit.v

    def __plot__(self, orbit, plotter, next_operation=None):
        self.__apply__(orbit)
        plotter._plot_orbit(orbit, label='Circularised')

    def velocity_delta(self, orbit):
        with saved_state(orbit):
            if self.raise_pericenter:
                orbit.propagate_anomaly_to(M=pi)
                radius = orbit.apocenter_radius
            else:
                orbit.propagate_anomaly_to(M=0)
                radius = orbit.pericenter_radius

            old_velocity = orbit.v

            a, e = elements_for_apsides(radius, radius)
            orbit.a = a
            orbit.e = e

            new_velocity = orbit.v

            return new_velocity - old_velocity

    def _repr_helper_(self, r):
        r.keyword_from_attr('raise_pericenter')


class SetPericenterHere(ReprHelperMixin, Operation):
    """Operation for setting pericenter to current location (in preparation
    for a maneuver to an elliptical orbit. Initial orbit must be circular.
    """
    def __init__(self):
        super(SetPericenterHere, self).__init__()

    def __apply__(self, orbit):
        """Assumptions: orbit is circular"""
        orbit.arg_pe = orbit.f
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            orbit.f = 0

    def __plot__(self, orbit, plotter, next_operation=None):
        self.__apply__(orbit)
        plotter._plot_position(orbit, label='Pericenter set here')

    def _repr_helper_(self, r):
        pass


class Maneuver(ReprHelperMixin, object):
    """A Maneuver is a collection of operations, and class methods are provided
    to easily create maneuvers which ensure orbits are propagated to the
    correct position between impulse operations.
    """
    def __init__(self, operations):
        if not isinstance(operations, list):
            operations = [operations]

        self.operations = operations

    @classmethod
    def set_apocenter_radius_to(cls, apocenter_radius):
        """Maneuver for setting apocenter radius."""
        operations = [
            PropagateAnomalyTo(M=0),
            SetApocenterRadiusTo(apocenter_radius)]
        return cls(operations)

    @classmethod
    def set_pericenter_radius_to(cls, pericenter_radius):
        """Maneuver for setting pericenter radius."""
        operations = [
            PropagateAnomalyTo(M=pi),
            SetPericenterRadiusTo(pericenter_radius)]
        return cls(operations)

    @classmethod
    def set_apocenter_altitude_to(cls, apocenter_altitude):
        """Maneuver for setting apocenter altitude."""
        operations = [
            PropagateAnomalyTo(M=0),
            SetApocenterAltitudeTo(apocenter_altitude)]
        return cls(operations)

    @classmethod
    def set_pericenter_altitude_to(cls, pericenter_altitude):
        """Maneuver for setting pericenter altitude."""
        operations = [
            PropagateAnomalyTo(M=pi),
            SetPericenterAltitudeTo(pericenter_altitude)]
        return cls(operations)

    @classmethod
    def change_apocenter_by(cls, delta):
        """Maneuver for changing apocenter."""
        operations = [
            PropagateAnomalyTo(M=0),
            ChangeApocenterBy(delta)]
        return cls(operations)

    @classmethod
    def change_pericenter_by(cls, delta):
        """Maneuver for changing pericenter."""
        operations = [
            PropagateAnomalyTo(M=pi),
            ChangePericenterBy(delta)]
        return cls(operations)

    @classmethod
    def hohmann_transfer_to_radius(cls, radius):
        """Maneuver for a hohmann transfer to given radius.

        Initial orbit must be circular. Will raise apocenter from position
        at time of application.
        """
        operations = [
            SetPericenterHere(),
            SetApocenterRadiusTo(radius),
            PropagateAnomalyTo(M=pi),
            Circularise()]
        return cls(operations)

    @classmethod
    def hohmann_transfer_to_altitude(cls, altitude):
        """Maneuver for a hohmann transfer to given altitude.

        Initial orbit must be circular. Will raise apocenter from position
        at time of application.
        """
        operations = [
            SetPericenterHere(),
            SetApocenterAltitudeTo(altitude),
            PropagateAnomalyTo(M=pi),
            Circularise()]
        return cls(operations)

    @classmethod
    def set_inclination_to(cls, inclination):
        """Maneuver for setting inclination."""
        operations = [
            lambda orbit: PropagateAnomalyTo(f=2 * pi - orbit.arg_pe),
            SetInclinationTo(inclination)]
        return cls(operations)

    @classmethod
    def change_inclination_by(cls, delta):
        """Maneuver for changing inclination."""
        operations = [
            lambda orbit: PropagateAnomalyTo(f=2 * pi - orbit.arg_pe),
            ChangeInclinationBy(delta)]
        return cls(operations)

    @classmethod
    def bielliptic_transfer(cls):
        raise NotImplementedError

    def __apply__(self, orbit):
        for operation in self.operations:
            if callable(operation):
                operation = operation(orbit)
            if hasattr(operation, '__apply__') and callable(getattr(operation, '__apply__')):
                operation.__apply__(orbit)
            elif isinstance(operation, ImpulseOperation):
                orbit.v += operation.velocity_delta(orbit)
            elif isinstance(operation, TimeOperation):
                orbit.t += operation.time_delta(orbit)

    def __iapply__(self, orbit, copy=False):
        for operation in self.operations:
            if callable(operation):
                operation = operation(orbit)

            if copy:
                yield _copy(orbit), operation
            else:
                yield orbit, operation

            if hasattr(operation, '__apply__') and callable(getattr(operation, '__apply__')):
                operation.__apply__(orbit)
            elif isinstance(operation, ImpulseOperation):
                orbit.v += operation.velocity_delta(orbit)
            elif isinstance(operation, TimeOperation):
                orbit.t += operation.time_delta(orbit)

    def _repr_helper_(self, r):
        r.positional_from_attr('operations')

    def __add__(self, other):
        if isinstance(other, Maneuver):
            return Maneuver(self.operations + other.operations)
        else:
            return NotImplemented
