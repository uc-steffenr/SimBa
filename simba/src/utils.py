"""Defines tools for SimBa simulation."""
import numpy as np
from shapely import Polygon
from time import perf_counter


def R(ang : float) -> np.ndarray:
    """2D rotation matrix.

    Parameters
    ----------
    ang : float
        Rotation angle.

    Returns
    -------
    np.ndarray
        Rotation matrix with given angle.
    """
    return np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])


def generate_verts(k : int,
                   separation : float,
                   obstacles : list[Polygon]=[],
                   border_bounds : list[tuple[float]]=[(-5., 5.), (-6., 6.)],
                   r_bounds : tuple[float]=(0.5, 1.5),
                   rot_bounds : tuple[float]=(0., 45.),
                   ) -> np.ndarray:
    """Generates vertices for polygon of k-sides.

    Parameters
    ----------
    k : int
        Number of sides for a regular polygon. Bounded between 1 and 8.
        If 1, the polygon is just a circle. 2 means a line, and the rest
        are self explanatory.
    separation : float
        Value which the distance between any two obstacles can't be
        within.
    obstacles : list[Polygon]
        List of current obstacles to avoid, by default [].
    border_bounds : list[tuple[float]], optional
        Bounds of the border the shape should appear in, by default
        [(-5., 5.), (-6., 6.)].
    r_bounds : tuple[float], optional
        Bounds of the radius of the regular polygon, by default
        (0.5, 1.5).
    rot_bounds : tuple[float], optional
        Bounds of the rotation that should occur, by default (0., 45.).

    Returns
    -------
    np.ndarray
        Vertices of the shape requested.
    """

    if k > 2:
        alpha = (2*np.pi/k)*np.arange(k)
        ang = np.random.uniform(low=rot_bounds[0], high=rot_bounds[1])
    else:
        alpha = np.linspace(0., 2*np.pi, 50)

    r = np.random.uniform(low=r_bounds[0], high=r_bounds[1])

    while True:
        xc = np.random.uniform(low=border_bounds[0][0],
                            high=border_bounds[0][1])
        yc = np.random.uniform(low=border_bounds[1][0],
                            high=border_bounds[1][1])
        verts = np.vstack((xc + r*np.cos(alpha), yc + r*np.sin(alpha))).T
        if k > 2:
            verts = verts @ R(ang)

        border_bound_check = np.all(verts[:, 0] > border_bounds[0][0]) and \
                             np.all(verts[:, 0] < border_bounds[0][1]) and \
                             np.all(verts[:, 1] > border_bounds[1][0]) and \
                             np.all(verts[:, 1] < border_bounds[1][1])

        poly = Polygon(verts)
        if len(obstacles) > 0:
            obstacle_too_close = any([poly.distance(obst) < separation \
                                  for obst in obstacles])
        else:
            obstacle_too_close = False

        if border_bound_check and not obstacle_too_close:
            break

    return verts


def timing(func):
    """Times specified function evaluation.

    Parameters
    ----------
    func : callable
        Wrapped function with timing.
    """
    def wrapper(*args, **kwargs):
        t0 = perf_counter()
        result = func(*args, **kwargs)
        dt = perf_counter() - t0
        print(f'{func.__name__} took {dt} seconds')
        return result, dt

    return wrapper
