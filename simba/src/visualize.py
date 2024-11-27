"""Defines visualization methods for runs."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import os

from .agent import Agent


def plot_states(t : np.ndarray,
                y : np.ndarray,
                save : bool=True,
                show : bool=False,
                save_dir : str=os.path.abspath('')
                ) -> None:
    """Plots states of simulation.

    Parameters
    ----------
    t : np.ndarray
        Time steps for simulation.
    y : np.ndarray
        States for simulation.
    save : bool, optional
        Flag to determine whether to save the plot, by default True.
    show : bool, optional
        Flag to determine whether to show the plot, by default False.
    save_dir : str, optional
        Directory to save results to, by default os.path.abspath('').
    """
    names = ['x (m)', 'xdot (m/s)', 'y (m)', 'ydot (m/s)', 'theta (deg)',
             'thetadot (deg/s)']
    fig, ax = plt.subplots(3, 2, sharex=True)
    ax = ax.ravel()

    [a.plot(t, y[i, :]) for i, a in enumerate(ax) if i < 4]
    [a.plot(t, np.rad2deg(y[i, :])) for i, a in enumerate(ax) if i >= 4]
    [a.set_ylabel(names[i]) for i, a in enumerate(ax)]
    [a.set_xlabel('time (s)') for a in ax]
    [a.grid() for a in ax]

    plt.tight_layout()

    if show:
        plt.show()
    if save:
        fig.savefig(os.path.join(save_dir, 'states.jpg'))


def plot_controls(agent : Agent,
                  save : bool=True,
                  show : bool=False,
                  save_dir : str=os.path.abspath('')
                  ) -> None:
    """Plots controls of simulation.

    Parameters
    ----------
    agent : Agent
        Agent that traversed an environment.
    save : bool, optional
        Flag to save results, by default True.
    show : bool, optional
        Flag to determine whether to show results, by default False.
    save_dir : str, optional
        Path to save results to, by default os.path.abspath('').
    """
    fig, ax = plt.subplots(1, 2)
    ax = ax.ravel()

    us = np.array(agent.control_actions)

    [a.plot(us[:, i]) for i, a in enumerate(ax)]
    [a.set_xlabel('time (s)') for a in ax]
    [a.set_ylabel('wheel force (N)') for a in ax]

    ax[0].set_title('Left Wheel Control')
    ax[1].set_title('Right Wheel Control')

    [a.grid() for a in ax]

    if show:
        plt.show()
    if save:
        fig.savefig(os.path.join(save_dir, 'controls.jpg'))
    

def animate(t : np.ndarray,
            X : np.ndarray,
            agent : Agent,
            target : np.ndarray,
            obst_verts : np.ndarray,
            bounds : list[tuple[float]],
            save : bool=True,
            show : bool=False,
            save_dir : str=os.path.abspath(''),
            **func_animate_kwargs
            ) -> None:
    """Animates simulation.

    Parameters
    ----------
    t : np.ndarray
        Time steps of simulation.
    X : np.ndarray
        States of simulation.
    agent : Agent
        Agent that traverses the environment.
    target : np.ndarray
        Target the agent tries to get to.
    obst_verts : np.ndarray
        Vertices of the obstacles.
    bounds : list[tuple[float]]
        Bounds of the simulation environment.
    save : bool, optional
        Flag to determine whether to save the results, by default True.
    show : bool, optional
        Flag to determine whether to show the results, by default False.
    save_dir : str, optional
        Directory to save results to, by default os.path.abspath('').
    """
    xrange = (bounds[0][0]-1, bounds[0][1]+1)
    yrange = (bounds[1][0]-1, bounds[0][1]+1)

    x = X[0, 0]
    y = X[2, 0]
    theta = X[4, 0]
    
    fig, ax = plt.subplots(1, 1, subplot_kw={'aspect': 'equal'})

    body = ax.fill(x+agent.verts[0, :],
                   y+agent.verts[1, :],
                   facecolor='silver',
                   edgecolor='black',
                   linewidth=1)[0]
    ln = ax.plot([x, x+agent.r*np.cos(theta)],
                 [y, y+agent.r*np.sin(theta)],
                 color='k',
                 linewidth=1)[0]
    trace = ax.plot([x], [y], '--')[0]

    angs = 2.*np.pi*(np.arange(agent.num_sensors)/agent.num_sensors) + \
           agent.offset + theta
    sensors = np.array([[(x+agent.r*np.cos(angs)), 
                         (x+(agent.r+agent.sensor_dist)*np.cos(angs))],
                        [(y+agent.r*np.sin(angs)), 
                         (y+(agent.r+agent.sensor_dist)*np.sin(angs))]]
                       ).transpose(2, 0, 1)
    sens = [ax.plot(sens[:, 0], sens[:, 1])[0] for sens in sensors]
    

    obsts = [ax.fill(v[:, 0], v[:, 1]) for v in obst_verts]
    boundary = Rectangle((bounds[0][0], bounds[1][0]),
                         bounds[0][1]-bounds[0][0],
                         bounds[1][1]-bounds[1][0],
                         facecolor='none',
                         edgecolor='r')
    trgt = ax.scatter(target[0], target[1], marker='*', c='r')

    ax.add_patch(boundary)
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)

    def update(i):
        x = X[0, i]
        y = X[2, i]
        theta = X[4, i]

        body.set_xy(np.array([x+agent.verts[0, :], y+agent.verts[1, :]]).T)
        ln.set_data([x, x+agent.r*np.cos(theta)], [y, y+agent.r*np.sin(theta)])
        trace.set_data(X[0, :i], X[2, :i])

        angs = 2.*np.pi*(np.arange(agent.num_sensors)/agent.num_sensors) + \
            agent.offset + theta
        sensors = np.array([[(x+agent.r*np.cos(angs)), 
                             (x+(agent.r+agent.sensor_dist)*np.cos(angs))],
                            [(y+agent.r*np.sin(angs)), 
                             (y+(agent.r+agent.sensor_dist)*np.sin(angs))]]
                          ).transpose(2, 0, 1)
    
        [sen.set_data(sensors[j, 0, :], sensors[j, 1, :]) \
                      for j, sen in enumerate(sens)]

    ani = FuncAnimation(fig, update, frames=len(t), blit=False,
                        **func_animate_kwargs)

    if show:
        plt.show()
    if save:
        ani.save(os.path.join(save_dir, 'animation.gif'))


def plot_sensor_readings(agent : Agent,
                         save : bool=True,
                         show : bool=False,
                         save_dir : str=os.path.abspath('')
                         ) -> None:
    """Plots sensor readings for every timestep.

    Parameters
    ----------
    agent : Agent
        Agent that traverses the environment.
    save : bool, optional
        Flag to determine whether to save the results, by default True.
    show : bool, optional
        Flag to determine whether to show the results, by default False.
    save_dir : str, optional
        Directory to save results to, by default os.path.abspath('').
    """
    # TODO: make this use closest multipliers for row and col
    fig, ax = plt.subplots(1, agent.num_sensors)
    ax = ax.ravel()

    read = np.array(agent.sensor_readings)
    [ax[i].plot(read[:, i]) for i in range(len(ax))]
    [ax[i].set_xlabel(f'Sensor {i+1}') for i in range(len(ax))]

    plt.tight_layout()

    if show:
        plt.show()
    if save:
        fig.savefig(os.path.join(save_dir, 'sensors.jpg'))
