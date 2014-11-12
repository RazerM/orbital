class ImpulseOperation:
    def velocity_delta(self):
        raise NotImplementedError(
            'Subclasses of {}.{} must implement {}'
            .format(__name__, __class__.__name__, self.velocity_delta.__name__))


class TimeOperation:
    def time_delta(self, orbit):
        """Return the time delta to propagate the orbit by.

        :param orbit: Orbit
        :type orbit: :py:class:`orbital.elements.KeplerianElements`
        """
        raise NotImplementedError(
            'Subclasses of {}.{} must implement {}'
            .format(__name__, __class__.__name__, self.time_delta.__name__))


class PropagateAnomalyTo(TimeOperation):
    """Operation for propagating to time in future where anomaly is equal to
    value passed in.

    One (and only one) of these parameters must be passed in:

    :param float M: Mean anomaly
    :param float E: Eccentric anomaly
    :param float f: True anomaly
    """
    def __init__(self, **kwargs):
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
            M = mean_anomaly_from_true(orbit.e, self.anomaly)
        elif self.key == 'E':
            M = mean_anomaly_from_eccentric(orbit.e, self.anomaly)
        elif self.key == 'M':
            M = self.M

        # Propagate one orbit if destination is 'behind' current state.
        if M < orbit.M:
            M += 2 * pi

        return (M - orbit.M) / self.n

    def __repr__(self):
        return '{}({key}={anomaly!r})'.format(__class__.__name__, key=self.key, anomaly=self.anomaly)


class Maneuver:
    """Todo: Each maneuver will contain a list of operations, which are therefore
    independent of orbit and calculated at the time the maneuver is applied.
    """
    def __init__(self):
        pass

    @classmethod
    def raise_apocenter_by(cls, delta, orbit):
        pass

    @classmethod
    def change_apocenter_to(cls, apocenter, orbit):
        pass

    @classmethod
    def lower_apocenter_by(cls, delta, orbit):
        pass

    @classmethod
    def raise_pericenter_by(cls, delta, orbit):
        pass

    @classmethod
    def change_pericenter_to(cls, pericenter, orbit):
        pass

    @classmethod
    def lower_pericenter_by(cls, delta, orbit):
        pass

    @classmethod
    def hohmann_transfer(cls):
        # how to specify new orbit?
        # - new semimajor axix/radius/altitude
        pass

    def bielliptic_transfer(cls):
        pass
