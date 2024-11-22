"""Test Environment methods."""
import numpy as np

from simba import Agent, Environment


def dummy_controller(t, X, target, reading):
    if t > 1.:
        u = np.array([1., -1.])
    else:
        u = np.zeros(2)

    return u


def test_instantiation():
    # Showing all variables that can be specified to instantiate an 
    # Environment
    agent = Agent()                 # agent to navigate environment
    t_dur = 18.                     # time duration for simulation
    bounds = [(-5., 5.), (-5., 5.)] # bounds for simulation environment
    num_obstacles = 3               # number of obstacles in environment
    obst_size_bounds = (0.5, 1.5)   # size range for radius of regular 
                                    # polygon obstacles
    rot_bounds = (0., 45.)          # rotation range for obstacles, in
                                    # degrees
    sep_factor = 1.5                # factor to be multiplied by agent's
                                    # radius to ensure obstacles aren't
                                    # too close
    epsilon = 0.01                  # collision coefficient to determine
                                    # agent's velocity after a collision
    seed = 12345                    # seed to randomly generate
                                    # environment
    target_thresh = 0.01            # threshold to determine if the
                                    # target is achieved

    env = Environment(agent,
                      t_dur,
                      bounds,
                      num_obstacles,
                      obst_size_bounds,
                      rot_bounds,
                      sep_factor,
                      epsilon,
                      seed,
                      target_thresh)
    
    assert env.agent == agent
    assert env.t_dur == t_dur
    assert env.bounds == bounds
    assert env.num_obstacles == num_obstacles
    assert env.obst_size_bounds == obst_size_bounds
    assert env.rot_bounds == rot_bounds
    assert env.sep_factor ==  sep_factor
    assert env.epsilon == epsilon
    assert env.seed == seed
    assert env.target_thresh == target_thresh
    print('Environment instantiated!')


def test_evaluate():
    agent = Agent()
    env = Environment(agent)
    metrics, t, y = env.evaluate(t_eval=np.linspace(0., 18., 100))

    assert len(t) == 100
    print('Evaluate method works!')


if __name__ == '__main__':
    test_instantiation()
    test_evaluate()
