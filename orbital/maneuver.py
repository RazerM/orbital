class Maneuver:
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
