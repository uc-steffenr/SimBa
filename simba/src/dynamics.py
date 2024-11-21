"""Defines plant dynamics of system."""
import numpy as np


def plant(t : float,
          X : np.ndarray,
          u : np.ndarray,
          m : float,
          r : float,
          kd : float,
          kn : float
          ) -> np.ndarray:
    """Scipy-compatible Roomba plant dynamics.

    Parameters
    ----------
    t : float
        Time value, in s.
    X : np.ndarray
        Current state.
    u : np.ndarray
        Control vector.
    m : float
        Mass of roomba, in kg.
    r : float
        Radius of roomba, in m.
    kd : float
        Coefficient of drag in wheels, in Ns^2/m.
    kn : float
        Coefficient of normal force in wheels, in Ns^2/m.

    Returns
    -------
    np.ndarray
        Time derivative values of the states.
    """
    xdot = X[1]
    ydot = X[3]
    theta = X[4]
    thetadot = X[5]

    vl = xdot*np.cos(theta) + ydot*np.sin(theta) - thetadot*r # left wheel velocity
    vr = xdot*np.cos(theta) + ydot*np.sin(theta) + thetadot*r # right wheel velocity
    vn = -xdot*np.sin(theta) + ydot*np.cos(theta) # normal velocity

    Fd = kd*(vr*np.abs(vr) + vl*np.abs(vl)) # tangential drag force on both wheels
    Fn = kn*vn*np.abs(vn) # force normal to robot
    Ft = np.sum(u) - Fd # tangential force to robot

    Xdot = np.array([xdot,
                     (Ft*np.cos(theta) + 2.*Fn*np.sin(theta))/m,
                     ydot,
                     (Ft*np.sin(theta) - 2.*Fn*np.cos(theta))/m,
                     thetadot,
                     (u[1] - u[0] - kd*vr*np.abs(vr) + \
                       kd*vl*np.abs(vl))/(0.5*m*r)])

    return Xdot
