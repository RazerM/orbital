"""plotting module for orbital

This implementation was inspired by poliastro (c) 2012 Juan Luis Cano (BSD License)
"""
# encoding: utf-8
from __future__ import absolute_import, division, print_function
from copy import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy import cos, sin
from scipy.constants import kilo, pi

from .maneuver import TimeOperation
from .utilities import (
    lookahead, orbit_radius, saved_state, uvw_from_elements)

__all__ = [
    'plot2d',
    'plot3d',
    'plot',
    'Plotter2D',
    'Plotter3D',
]


def plot2d(orbit, title='', maneuver=None, animate=False, speedup=5000):
    """Convenience function to 2D plot orbit in a new figure."""
    plotter = Plotter2D()
    if animate:
        return plotter.animate(orbit, title=title, speedup=speedup)
    else:
        plotter.plot(orbit, title=title, maneuver=maneuver)


def plot3d(orbit, title='', maneuver=None, animate=False, speedup=5000):
    """Convenience function to 3D plot orbit in a new figure."""
    plotter = Plotter3D()
    if animate:
        return plotter.animate(orbit, title=title, speedup=speedup)
    else:
        plotter.plot(orbit, title=title, maneuver=maneuver)


plot = plot2d


class Plotter2D():
    """2D Plotter

    Handles still and animated plots of an orbit.
    """
    def __init__(self, axes=None, num_points=100):
        if axes:
            self.fig = axes.get_figure()
        else:
            self.fig = plt.figure()
            axes = self.fig.add_subplot(111)

        self.axes = axes
        self.axes.set_aspect(1)
        self.axes.set_xlabel("$p$ [km]")
        self.axes.set_ylabel("$q$ [km]")

        self.points_per_rad = num_points / (2 * pi)

    def plot(self, orbit, maneuver=None, title=''):
        self._plot_body(orbit)

        if maneuver is None:
            self._plot_orbit(orbit)
            self.pos_dot = self._plot_position(orbit)
        else:
            self._plot_orbit(orbit, label='Initial orbit')
            self.propagate_counter = 1

            states = lookahead(
                orbit.apply_maneuver(maneuver, iter=True, copy=True),
                fillvalue=(None, None))

            with saved_state(orbit):
                for (orbit, operation), (_, next_operation) in states:
                    with saved_state(orbit):
                        operation.plot(orbit, self, next_operation)
            self.axes.legend()
        self.axes.set_title(title)

    def animate(self, orbit, speedup=5000, title=''):
        # Copy orbit, because it will be modified in the animation callback.
        orbit = copy(orbit)

        self.plot(orbit)

        p = orbit.a * (1 - orbit.e ** 2)

        def fpos(f):
            pos = np.array([cos(f), sin(f), 0 * f]) * p / (1 + orbit.e * cos(f))
            pos /= kilo
            return pos

        time_per_orbit = orbit.T / speedup
        interval = 1000 / 30
        times = np.linspace(orbit.t, orbit.t + orbit.T, int(time_per_orbit * 30))

        def animate(i):
            orbit.t = times[i - 1]
            pos = fpos(orbit.f)
            self.pos_dot.set_data(pos[0], pos[1])

            return self.pos_dot

        self.axes.set_title(title)

        # blit=True causes an error on OS X, disable for now.
        ani = animation.FuncAnimation(
            self.fig, animate, len(times), interval=interval, blit=False)

        return ani

    @staticmethod
    def _perifocal_coords(orbit, f):
        p = orbit.a * (1 - orbit.e ** 2)
        pos = np.array([cos(f), sin(f), 0 * f]) * p / (1 + orbit.e * cos(f))
        pos /= kilo
        return pos

    def _plot_orbit(self, orbit, f1=0, f2=2 * pi, label=None):
        if f2 < f1:
            f2 += 2 * pi

        num_points = self.points_per_rad * (f2 - f1)
        f = np.linspace(f1, f2, int(num_points))

        pos = self._perifocal_coords(orbit, f)

        self.axes.plot(pos[0, :], pos[1, :], '--', linewidth=1, label=label)

    def _plot_position(self, orbit, f=None, propagated=False, label=None):
        if f is None:
            f = orbit.f

        pos = self._perifocal_coords(orbit, f)

        if propagated:
            if label is not None:
                raise TypeError('propagated flag sets label automatically')

            label = 'Propagated position {}'.format(self.propagate_counter)
            self.propagate_counter += 1

        pos_dot, = self.axes.plot(
            pos[0], pos[1], 'o', label=label)

        return pos_dot

    def _plot_body(self, orbit):
        color = '#EBEBEB'
        if orbit.body.plot_color is not None:
            color = orbit.body.plot_color
        self.axes.add_patch(Circle((0, 0), orbit.body.mean_radius / kilo,
                                   linewidth=0, color=color))


class Plotter3D(object):
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
        self.axes.set_xlabel("$x$ [km]")
        self.axes.set_ylabel("$y$ [km]")
        self.axes.set_zlabel("$z$ [km]")

        # These are used to fix aspect ratio of final plot.
        # See Plotter3D._force_aspect()
        self._coords_x = np.array(0)
        self._coords_y = np.array(0)
        self._coords_z = np.array(0)

        self.points_per_rad = num_points / (2 * pi)

    def plot(self, orbit, maneuver=None, title=''):
        self._plot_body(orbit)

        if maneuver is None:
            self._plot_orbit(orbit)
            self.pos_dot = self._plot_position(orbit)
        else:
            self._plot_orbit(orbit, label='Initial orbit')
            self.propagate_counter = 1

            states = lookahead(
                orbit.apply_maneuver(maneuver, iter=True, copy=True),
                fillvalue=(None, None))

            with saved_state(orbit):
                for (orbit, operation), (_, next_operation) in states:
                    with saved_state(orbit):
                        operation.plot(orbit, self, next_operation)
            self.axes.legend()
        self.axes.set_title(title)

        self._force_aspect()

    def animate(self, orbit, speedup=5000, title=''):
        # Copy orbit, because it will be modified in the animation callback.
        orbit = copy(orbit)

        self.plot(orbit)

        num_points = self.points_per_rad * 2 * pi
        f = np.linspace(0, 2 * pi, int(num_points))

        def fpos(f):
            U, _, _ = uvw_from_elements(orbit.i, orbit.raan, orbit.arg_pe, f)
            pos = orbit_radius(orbit.a, orbit.e, f) * U
            pos /= kilo
            return pos[0], pos[1], pos[2]

        time_per_orbit = orbit.T / speedup
        interval = 1000 / 30
        times = np.linspace(orbit.t, orbit.t + orbit.T, int(time_per_orbit * 30))

        def animate(i):
            orbit.t = times[i - 1]
            x, y, z = fpos(orbit.f)
            self.pos_dot.set_data([x], [y])
            self.pos_dot.set_3d_properties([z])

            return self.pos_dot

        self.axes.set_title(title)

        # blit=True causes an error on OS X, disable for now.
        ani = animation.FuncAnimation(
            self.fig, animate, len(times), interval=interval, blit=False)

        return ani

    @staticmethod
    def _xyz_coords(orbit, f):
        U, _, _ = uvw_from_elements(orbit.i, orbit.raan, orbit.arg_pe, f)
        pos = orbit_radius(orbit.a, orbit.e, f) * U
        pos /= kilo
        return pos

    def _plot_orbit(self, orbit, f1=0, f2=2 * pi, label=None):
        if f2 < f1:
            f2 += 2 * pi

        num_points = self.points_per_rad * (f2 - f1)
        f = np.linspace(f1, f2, int(num_points))

        pos = self._xyz_coords(orbit, f)
        x, y, z = pos[0, :], pos[1, :], pos[2, :]

        self.axes.plot(x, y, z, '--', linewidth=1, label=label)

        self._append_coords_for_aspect(x, y, z)

    def _plot_position(self, orbit, f=None, propagated=False, label=None):
        if f is None:
            f = orbit.f

        pos = self._xyz_coords(orbit, f)
        x, y, z = pos[0], pos[1], pos[2]

        if propagated:
            if label is not None:
                raise TypeError('propagated flag sets label automatically')

            label = 'Propagated position {}'.format(self.propagate_counter)
            self.propagate_counter += 1

        pos_dot, = self.axes.plot(
            [x], [y], [z], 'o', label=label)

        return pos_dot

    def _plot_body(self, orbit):
        color = '#EBEBEB'
        if orbit.body.plot_color is not None:
            color = orbit.body.plot_color

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 50)
        cx = orbit.body.mean_radius * np.outer(np.cos(u), np.sin(v))
        cy = orbit.body.mean_radius * np.outer(np.sin(u), np.sin(v))
        cz = orbit.body.mean_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        cx, cy, cz = cx / kilo, cy / kilo, cz / kilo
        self.axes.plot_surface(cx, cy, cz, rstride=5, cstride=5, color=color,
                               edgecolors='#ADADAD', shade=False)

    def _append_coords_for_aspect(self, x, y, z):
        self._coords_x = np.append(self._coords_x, x)
        self._coords_y = np.append(self._coords_y, y)
        self._coords_z = np.append(self._coords_z, z)

    def _force_aspect(self):
        # Thanks to the following SO answer, we can make sure axes are equal
        # http://stackoverflow.com/a/13701747/2093785

        # Create cubic bounding box to simulate equal aspect ratio

        x = self._coords_x
        y = self._coords_y
        z = self._coords_z

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
