"""plotting module for orbital

This implementation was inspired by poliastro (c) 2012 Juan Luis Cano (BSD License)
"""
from copy import copy

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy import sin, cos
from scipy.constants import kilo, pi

from orbital.utilities import uvw_from_elements, orbit_radius


def plot2d(orbit, animate=False, speedup=5000):
    """Convenience function to 2D plot orbit in a new figure."""
    plotter = Plotter2D()
    if animate:
        plotter.animate(orbit, speedup)
    else:
        plotter.plot(orbit)


def plot3d(orbit, animate=False, speedup=5000):
    """Convenience function to 3D plot orbit in a new figure."""
    plotter = Plotter3D()
    if animate:
        plotter.animate(orbit, speedup)
    else:
        plotter.plot(orbit)


# Alias plot2d as plot
plot = plot2d


class Plotter2D():
    """2D Plotter

    Handles still and animated plots of an orbit.

    TODO: Allow maneuver plots. I'm considering adding a maneuver parameter to
    plot() that would plot after each Operation.
    """
    def __init__(self, axes=None, num_points=100):
        if axes:
            self.fig = axes.get_figure()
        else:
            self.fig = plt.figure()
            axes = self.fig.add_subplot(111)
        self.axes = axes
        self.num_points = num_points

    def plot(self, orbit):
        f = np.linspace(0, 2 * pi, self.num_points)

        p = orbit.a * (1 - orbit.e * 2)
        pos = np.array([cos(f), sin(f), 0 * f]) * p / (1 + orbit.e * cos(f))
        pos /= kilo

        self.axes.add_patch(
            mpl.patches.Circle((0, 0), orbit.body.mean_radius / kilo,
                               lw=0, color='#EBEBEB'))

        self.axes.plot(pos[0, :], pos[1, :], '--', color='red')
        self.axes.set_aspect(1)

        f = orbit.f
        p = orbit.a * (1 - orbit.e * 2)
        pos = np.array([cos(f), sin(f), 0 * f]) * p / (1 + orbit.e * cos(f))
        pos /= kilo

        self.pos_dot, = self.axes.plot(pos[0], pos[1], 'o', mew=0)

        self.axes.set_xlabel("$p$ [km]")
        self.axes.set_ylabel("$q$ [km]")

    def animate(self, orbit, speedup=5000):
        # Copy orbit so we can change anomaly without restoring state
        orbit = copy(orbit)

        self.plot(orbit)

        p = orbit.a * (1 - orbit.e * 2)

        def fpos(f):
            pos = np.array([cos(f), sin(f), 0 * f]) * p / (1 + orbit.e * cos(f))
            pos /= kilo
            return pos

        time_per_orbit = orbit.T / speedup
        interval = 1000 / 30
        times = np.linspace(orbit.t, orbit.t + orbit.T, time_per_orbit * 30)

        def animate(i):
            orbit.t = times[i - 1]
            pos = fpos(orbit.f)
            self.pos_dot.set_data(pos[0], pos[1])

            return self.pos_dot

        # blit=True causes an error on OS X, disable for now.
        ani = animation.FuncAnimation(
            self.fig, animate, len(times), interval=interval, blit=False)


class Plotter3D():
    """3D Plotter

    Handles still and animated plots of an orbit.
    """
    def __init__(self, axes=None, num_points=100):
        if axes:
            self.fig = axes.get_figure()
        else:
            self.fig = plt.figure()
            axes = self.fig.add_subplot(111, projection='3d')
        self.axes = axes
        self.num_points = num_points

    def plot(self, orbit):
        # Plot orbit
        f = np.linspace(0, 2 * pi, self.num_points)
        U, _, _ = uvw_from_elements(orbit.i, orbit.raan, orbit.arg_pe, f)
        pos = orbit_radius(orbit.a, orbit.e, f) * U
        x, y, z = pos[0, :], pos[1, :], pos[2, :]
        x, y, z = x / kilo, y / kilo, z / kilo

        self.axes.plot(x, y, z, '--', color='red')

        # Plot body
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 50)
        cx = orbit.body.mean_radius * np.outer(np.cos(u), np.sin(v))
        cy = orbit.body.mean_radius * np.outer(np.sin(u), np.sin(v))
        cz = orbit.body.mean_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        cx, cy, cz = cx / kilo, cy / kilo, cz / kilo
        self.axes.plot_surface(cx, cy, cz, rstride=5, cstride=5, color='#EBEBEB', edgecolors='#ADADAD', shade=False)

        # Plot current position
        f = orbit.f
        U, _, _ = uvw_from_elements(orbit.i, orbit.raan, orbit.arg_pe, f)
        pos = orbit_radius(orbit.a, orbit.e, f) * U
        px, py, pz = pos[0], pos[1], pos[2]
        px, py, pz = px / kilo, py / kilo, pz / kilo

        self.pos_dot, = self.axes.plot([px], [py], [pz], 'o')

        # Thanks to the following SO answer, we can make sure axes are equal
        # http://stackoverflow.com/a/13701747/2093785

        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([x.max() - x.min(),
                              y.max() - y.min(),
                              z.max() - z.min()]).max()
        Xb = (0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() +
              0.5 * (x.max() + x.min()))
        Yb = (0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() +
              0.5 * (y.max() + y.min()))
        Zb = (0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() +
              0.5 * (z.max() + z.min()))

        for xb, yb, zb in zip(Xb, Yb, Zb):
            self.axes.plot([xb], [yb], [zb], 'w')

        self.axes.set_xlabel("$x$ [km]")
        self.axes.set_ylabel("$y$ [km]")
        self.axes.set_zlabel("$z$ [km]")

    def animate(self, orbit, speedup=5000):
        # Copy orbit so we can change anomaly without restoring state
        orbit = copy(orbit)

        self.plot(orbit)

        f = np.linspace(0, 2 * pi, self.num_points)

        def fpos(f):
            U, _, _ = uvw_from_elements(orbit.i, orbit.raan, orbit.arg_pe, f)
            pos = orbit_radius(orbit.a, orbit.e, f) * U
            pos /= kilo
            return pos[0], pos[1], pos[2]

        time_per_orbit = orbit.T / speedup
        interval = 1000 / 30
        times = np.linspace(orbit.t, orbit.t + orbit.T, time_per_orbit * 30)

        def animate(i):
            orbit.t = times[i - 1]
            x, y, z = fpos(orbit.f)
            self.pos_dot.set_data([x], [y])
            self.pos_dot.set_3d_properties([z])

            return self.pos_dot

        # blit=True causes an error on OS X, disable for now.
        ani = animation.FuncAnimation(
            self.fig, animate, len(times), interval=interval, blit=False)
