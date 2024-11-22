"""Test Agent methods."""
import numpy as np
from shapely import LineString

from simba import Agent


def dummy_controller(t, X, target, reading):
    if t > 1.:
        u = np.array([1., -1.])
    else:
        u = np.zeros(2)

    return u

def test_instantiation():
    # Showing all the variables that can be set to instantiate an agent
    # All of these are the default values
    r = 0.165                   # radius of agent (m)
    m = 2.7                     # mass of agent (kg)
    num_sensors = 3             # number of sensors
    offset = np.deg2rad(60.)    # offset of sensors to 0 degrees
    sensor_dist = 2.            # distance the proximity sensor can read 
                                # to
    f_max = 2.7                 # maximum force each wheel can apply (N)
    kd = 3.4                    # drag coefficient
    kn = 1_000.                 # normal coefficient
    controls = dummy_controller # control method
    collision_thresh = 0.001    # threshold of distance to object to
                                # consider a collision
    heading_thresh = 0.01       # theshold of heading difference to
                                # target to consider off track

    # NOTE: every argument to Agent is a kwarg, so either instantiate
    # like this or make a dictionary for the arguments to be put in
    # agent and use ** to make them kwargs to agent.
    agent = Agent(r=r,
                  m=m,
                  num_sensors=num_sensors,
                  offset=offset,
                  sensor_dist=sensor_dist,
                  f_max=f_max,
                  kd=kd,
                  kn=kn,
                  controls=controls,
                  collision_thresh=collision_thresh,
                  heading_thresh=heading_thresh)

    assert agent.r == r
    assert agent.m == m
    assert agent.num_sensors == num_sensors
    assert agent.offset == offset
    assert agent.sensor_dist == sensor_dist
    assert agent.f_max == f_max
    assert agent.kd == kd
    assert agent.kn == kn
    assert agent._controls.__code__ == controls.__code__
    assert agent._collision_thresh == collision_thresh
    assert agent._heading_thresh == heading_thresh
    print('Values instantiated correctly!')


def test_sensor_reading():
    agent = Agent()

    # current state is at the origin with no velocity
    X = np.zeros(6)
    # placing wall right in front of agent
    wall = LineString(np.array([[1., 1.], [-2.5, 2.5]]))

    readings = agent.sensor_reading(X, [wall])
    assert any(readings < 2.)
    print('Sensor readings work!')


def test_controls():
    agent = Agent(controls=dummy_controller)
    X = np.zeros(6)
    target = np.array([1., 5.])

    t = 0.
    u = agent.controls(t, X, target, [])
    assert np.all(u == np.zeros(2))

    t = 2.
    u = agent.controls(t, X, target, [])
    assert np.all(u == np.array([1., -1.]))

    print('Controls work!')


if __name__ == '__main__':
    test_instantiation()
    test_sensor_reading()
    test_controls()
