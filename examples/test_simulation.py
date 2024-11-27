"""Test Simulation methods."""
from simba import Agent, Simulation, timing


def test_instantiation():
    # Showing all variables that can be specified
    agent = Agent() # agent to navigate environments
    n_conditions = 5 # number of environments to simulate
    n_obstacles = (2, 5) # range of obstacles that can occur in envs
    sim_seed = 12345 # seed for Simulation to instantiate envs with
    # random seeds
    n_proc = 12 # number of processors to use for calculation
    env_kwargs = dict(t_dur = 18.,
                      bounds = [(-5., 5.), (-5., 5.)],
                      num_obstacles = 3,
                      obst_size_bounds = (0.5, 1.5),
                      rot_bounds = (0., 45.),
                      sep_factor = 1.5,
                      epsilon = 0.01,
                      seed = 12345,
                      target_thresh = 0.01)

    sim = Simulation(agent,
                     n_conditions,
                     n_obstacles,
                     sim_seed,
                     n_proc,
                     **env_kwargs)

    assert sim.agent == agent
    assert sim.N == n_conditions
    assert sim.n_obstacles == n_obstacles
    assert sim.seed == sim_seed
    assert sim.n_proc == n_proc
    print('Simulation instantiated!')


def test_run_simulation():
    agent = Agent()
    n_conditions = 5
    sim = Simulation(agent, n_conditions)

    metrics = sim.run_simulation()

    assert 'collision_count' in metrics
    assert 'heading_count' in metrics
    assert 'total_time' in metrics
    assert 'status' in metrics
    assert len(metrics['collision_count']) == n_conditions
    print('Simulation ran successfully!')


def test_run_parallel_simulation():
    agent = Agent()
    n_conditions = 50
    sim = Simulation(agent, n_conditions, n_proc=5)

    metrics, _ = timing(sim.run_parallel_simulations)()

    assert 'collision_count' in metrics
    assert 'heading_count' in metrics
    assert 'total_time' in metrics
    assert 'status' in metrics
    assert len(metrics['collision_count']) == n_conditions
    print('Parallel simulation ran successfully!')


if __name__ == '__main__':
    test_instantiation()
    test_run_simulation()
    test_run_parallel_simulation()
