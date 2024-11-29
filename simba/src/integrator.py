"""Defines integration methods."""
import numpy as np


def rk4(dynamics : callable,
        t_span : tuple[float],
        dt : float,
        y0 : np.ndarray,
        event : callable
        ) -> tuple[np.ndarray, np.ndarray, int]:
    """_summary_

    Parameters
    ----------
    dynamics : function
        _description_
    t_span : tuple[float]
        _description_
    dt : float
        _description_
    y0 : np.ndarray
        _description_
    event : function
        _description_

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        _description_
    """
    status = 0
    ts = np.arange(t_span[0], t_span[1], dt)
    ys = np.zeros((len(y0), len(ts)))
    ys[:, 0] = y0

    y = y0
    for i, t in enumerate(ts):
        try:
            k1 = dt * dynamics(t, y)
            k2 = dt * dynamics(t + 0.5*dt, y + 0.5*k1)
            k3 = dt * dynamics(t + 0.5*dt, y + 0.5*k2)
            k4 = dt * dynamics(t + dt, y + k3)
        except:
            raise Warning('Warning: integrator step failure!')
            status = -1

        y += (1./6.)*(k1 + 2.*k2 + 2.*k3 + k4)
        ys[:, i] = y

        if event(t, y) <= 0.:
            status = 1
            if i != len(ts) - 1:
                ys = ys[:i+1]
                ts = ts[:i+1]
            break

    return ts, ys, status
