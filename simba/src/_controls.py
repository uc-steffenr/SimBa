"""Defines temporary controls for system."""
import numpy as np


def controls(t : float,
             X : np.ndarray,
             target : np.ndarray,
             readings : np.ndarray
             ) -> np.ndarray:
    """Temporary time-based controls of roomba system.

    Parameters
    ----------
    t : float
        Time, in s.
    X : np.ndarray
        Current state of the system.

    Returns
    -------
    np.ndarray
        Control vector, in N (left, right).
    """
    u = np.zeros(2) # first is left, second is right
    if t < 2.4:
        u[0] = 2.7
        u[1] = 2.7
    elif t < 10.6665:
        u[0] = 1.4
        u[1] = 2.7
    elif t < 11.84589:
        u[0] = 2.7
        u[1] = 2.7

    return u
