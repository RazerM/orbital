import numpy as np

class Position(np.ndarray):

    def __new__(cls, x, y, z):
        # Create ndarray and cast to our class type
        obj = np.asarray([x, y, z]).view(cls)

        # Finally, we must return the newly created object:
        return obj

    @property
    def x(self):
        if len(self.shape) == 1:
            return self[0]
        else:
            return self[:,0]

    @x.setter
    def x(self, value):
        if len(self.shape) == 1:
            self[0] = value
        else:
            self[:,0] = value

    @property
    def y(self):
        if len(self.shape) == 1:
            return self[1]
        else:
            return self[:,1]

    @y.setter
    def y(self, value):
        if len(self.shape) == 1:
            self[1] = value
        else:
            self[:,1] = value

    @property
    def z(self):
        if len(self.shape) == 1:
            return self[2]
        else:
            return self[:,2]

    @z.setter
    def z(self, value):
        if len(self.shape) == 1:
            self[2] = value
        else:
            self[:,2] = value

    def __str__(self):
        """Override superclass __str__"""
        return self.__repr__()

    def __repr__(self):
        return '{name}(x={x!r}, y={y!r}, z={z!r})'.format(name=self.__class__.__name__, x=self.x, y=self.y, z=self.z)

class boob(Position):
    pass

# a = Position([1,2,3])
a = boob(x=1, y=2, z=3)
print(a)
a = np.multiply(a, 3)
print(a + 3)

