"""Tests visualize methods."""
import numpy as np

from simba import (Agent,
                   Environment,
                   plot_states,
                   plot_sensor_readings,
                   animate)


def test_plot_states():
    agent = Agent()
    env = Environment(agent)
    _, t, y = env.evaluate(t_eval=np.linspace(0., 18., 100))

    plot_states(t, y, save=False, show=True)


def test_plot_sensor_readings():
    agent = Agent()
    env = Environment(agent)
    _, t, y = env.evaluate(t_eval=np.linspace(0., 18., 100))

    plot_sensor_readings(agent, save=False, show=True)


def test_animate():
    agent = Agent()
    env = Environment(agent)
    _, t, y = env.evaluate(t_eval=np.linspace(0., 18., 100))
    
    animate(t, y, agent, env.target, env.verts, env.bounds, save=False,
            show=True)


if __name__ == '__main__':
    test_plot_states()
    test_plot_sensor_readings()
    test_animate()
