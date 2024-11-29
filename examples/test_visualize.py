"""Tests visualize methods."""
import numpy as np

from simba import (Agent,
                   Environment,
                   plot_states,
                   plot_controls,
                   plot_sensor_readings,
                   animate)


def test_plot_states():
    agent = Agent()
    env = Environment(agent)
    _, t, y, _ = env.evaluate()

    plot_states(t, y, save=False, show=True)


def test_plot_controls():
    agent = Agent(track_controls=True)
    env = Environment(agent)
    _, _, _, _ = env.evaluate()

    plot_controls(agent, save=False, show=True)


def test_plot_sensor_readings():
    agent = Agent(track_readings=True)
    env = Environment(agent)
    _, t, y, _ = env.evaluate()

    plot_sensor_readings(agent, save=False, show=True)


def test_animate():
    agent = Agent()
    env = Environment(agent)
    _, t, y, _ = env.evaluate()

    # plot_states(t, y, save=True, show=False)
    # plot_sensor_readings(agent, save=True, show=False)
    animate(t, y, agent, env.target, env.verts, env.bounds, save=False,
            show=True)


if __name__ == '__main__':
    test_plot_states()
    test_plot_controls()
    test_plot_sensor_readings()
    test_animate()
