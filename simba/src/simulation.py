"""Defines simulation class to obtain metrics over multiple runs."""
import numpy as np
from multiprocessing import Pool

from .agent import Agent
from .environment import Environment


def evaluate_env(args):
    env, solve_ivp_kwargs = args
    met, _, _, us = env.evaluate(**solve_ivp_kwargs)
    return met, env.agent.control_times, env.agent.control_states, us


class Simulation:
    def __init__(self,
                 agent : Agent,
                 n_conditions : int,
                 n_obstacles : tuple[int]=(2,5),
                 sim_seed : int=12345,
                 n_proc : int=1,
                 track_times : bool=False,
                 track_states : bool=False,
                 **environment_kwargs
                 ) -> None:
        """Instantiates simulation class for multiple sim evaluation.

        Parameters
        ----------
        agent : Agent
            Agent to traverse environments.
        n_conditions : int
            Number of simulations to evalute.
        n_obstacles : tuple[int], optional
            Range of number of obstacles for each environment, by
            default (2,5).
        sim_seed : int, optional
            Seed for simulation to set environment seeds, by
            default 12345.
        n_proc : int, optional
            Number of processors to run tasks in parallel, by default 1.
        track_times : bool, optional
            Whether or not to return the times of each run as a metric, 
            by default False.
        track_states : bool, optional
            Whether or not to return states of each run as a metric, by
            default False.
        """
        self.agent = agent
        self.N = n_conditions
        self.n_obstacles = n_obstacles
        self.seed = sim_seed
        self.n_proc = n_proc
        self.track_states = track_states
        self.track_times = track_times

        self.rng = np.random.default_rng(sim_seed)
        env_seeds = self.rng.integers(low=10_000, high=60_000,
                                      size=n_conditions)
        env_obstacles = self.rng.integers(low=n_obstacles[0],
                                          high=n_obstacles[1],
                                          size=n_conditions)

        if 'seed' in environment_kwargs.keys():
            environment_kwargs.pop('seed')
        if 'num_obstacles' in environment_kwargs.keys():
            environment_kwargs.pop('num_obstacles')

        self.envs = [Environment(self.agent,
                                 seed=env_seeds[i],
                                 num_obstacles=env_obstacles[i],
                                 **environment_kwargs)
                     for i in range(n_conditions)]


    def run_simulation(self, **solve_ivp_kwargs) -> dict[np.ndarray]:
        """Runs specified number of simulations and returns metrics.

        Returns
        -------
        dict[np.ndarray]
            Dictionary of results. Keys are:
                - collision_steps \\
                    Steps where a collision was read by a sensor.
                - heading_steps \\
                    Steps where agent heading to target is outside of the
                    specified threshold.
                - total_time \\
                    Time spent for each environment evaluation.
                - status \\
                    Termination status of integrator (-1 for failed
                    integration, 0 for reaching final time, and 1 for
                    achieving the termination event, i.e. reaching the
                    target). See scipy.integrate.solve_ivp docs for more
                    information.
                - progress \\
                    Tracks difference between initial distance to target
                    and final distance to target
                - states \\
                    If track_states is True, then the states for the run
                    will be returned.
                - controls \\
                    If agent.track_controls is True, then the controls
                    for the run will be returned.
        """
        metrics = dict(collision_steps=[None]*self.N,
                       heading_steps=[None]*self.N,
                       total_time=np.zeros(self.N),
                       status=np.zeros(self.N, dtype=np.int8),
                       progress=np.zeros(self.N))

        if self.track_times:
            metrics['times'] = [None]*self.N
        if self.track_states:
            metrics['states'] = [None]*self.N
        if self.agent.track_controls:
            metrics['controls'] = [None]*self.N

        for i, env in enumerate(self.envs):
            met, _, _, u = env.evaluate(**solve_ivp_kwargs)
            metrics['collision_steps'][i] = met['collision_steps']
            metrics['heading_steps'][i] = met['heading_steps']
            metrics['total_time'][i] = met['total_time']
            metrics['status'][i] = met['status']
            metrics['progress'][i] = met['progress']
            if self.track_times:
                metrics['times'][i] = env.agent.control_times
            if self.track_states:
                metrics['states'][i] = env.agent.control_states
            if self.agent.track_controls:
                metrics['controls'][i] = u

        return metrics


    def run_parallel_simulations(self, **solve_ivp_kwargs) -> dict[np.ndarray]:
        """Parallel version of run_simulation.

        Returns
        -------
        dict[np.ndarray]
            Dictionary of results. Keys are:
                - collision_steps \\
                    Steps where a collision was read by a sensor.
                - heading_steps \\
                    Steps where agent heading to target is outside of the
                    specified threshold.
                - total_time \\
                    Time spent for each environment evaluation.
                - status \\
                    Termination status of integrator (-1 for failed
                    integration, 0 for reaching final time, and 1 for
                    achieving the termination event, i.e. reaching the
                    target). See scipy.integrate.solve_ivp docs for more
                    information.
                - progress \\
                    Tracks difference between initial distance to target
                    and final distance to target
                - states \\
                    If track_states is True, then the states for the run
                    will be returned.
                - controls \\
                    If agent.track_controls is True, then the controls
                    for the run will be returned.
        """
        metrics = dict(collision_steps=[None]*self.N,
                       heading_steps=[None]*self.N,
                       total_time=np.zeros(self.N),
                       status=np.zeros(self.N, dtype=np.int8),
                       progress=np.zeros(self.N))

        if self.track_times:
            metrics['times'] = [None]*self.N
        if self.track_states:
            metrics['states'] = [None]*self.N
        if self.agent.track_controls:
            metrics['controls'] = [None]*self.N

        args = [(env, solve_ivp_kwargs) for env in self.envs]

        with Pool(processes=self.n_proc) as pool:
            results = pool.map(evaluate_env, args)

        for i, met in enumerate(results):
            metrics['collision_steps'][i] = met[0]['collision_steps']
            metrics['heading_steps'][i] = met[0]['heading_steps']
            metrics['total_time'][i] = met[0]['total_time']
            metrics['status'][i] = met[0]['status']
            metrics['progress'][i] = met[0]['progress']
            if self.track_times:
                metrics['times'][i] = met[1]
            if self.track_states:
                metrics['states'][i] = met[2]
            if self.agent.track_controls:
                metrics['controls'][i] = met[3]

        return metrics


    def reset(self) -> None:
        """Resets RNG for Simulation and for environments.
        """
        self.rng = np.random.default_rng(self.seed)

        for env in self.envs:
            env.reset_rng()
