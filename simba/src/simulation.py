"""Defines simulation class to obtain metrics over multiple runs."""
import numpy as np
from multiprocessing import Pool

from .agent import Agent
from .environment import Environment


def evaluate_env(args):
    env, solve_ivp_kwargs = args
    met, ts, ys, us = env.evaluate(**solve_ivp_kwargs)
    return met, ts, ys, us


class Simulation:
    def __init__(self,
                 agent : Agent,
                 n_conditions : int,
                 n_obstacles : tuple[int]=(2,5),
                 sim_seed : int=12345,
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
        """
        self.agent = agent
        self.N = n_conditions
        self.n_obstacles = n_obstacles
        self.seed = sim_seed

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
                - collision_count
                    Number of collisions picked up by sensors.
                - heading_count
                    Number of timesteps where agent heading to target is
                    outside of the specified threshold.
                - total_time
                    Time spent for each environment evaluation.
                - status
                    Termination status of integrator (-1 for failed
                    integration, 0 for reaching final time, and 1 for
                    achieving the termination event, i.e. reaching the
                    target). See scipy.integrate.solve_ivp docs for more
                    information.
        """
        metrics = dict(collision_count=np.zeros(self.N),
                       heading_count=np.zeros(self.N),
                       total_time=np.zeros(self.N),
                       status=np.zeros(self.N),
                       states=[],
                       controls=[])

        for i, env in enumerate(self.envs):
            met, _, y, u = env.evaluate(**solve_ivp_kwargs)
            metrics['collision_count'][i] = met['collision_count']
            metrics['heading_count'][i] = met['heading_count']
            metrics['total_time'][i] = met['total_time']
            metrics['status'][i] = met['status']
            metrics['states'].append(y)
            metrics['controls'].append(u)

        return metrics


    def run_parallel_simulations(self, **solve_ivp_kwargs) -> dict[np.ndarray]:
        """Parallel version of run_simulation.

        Returns
        -------
        dict[np.ndarray]
            Dictionary of results. Keys are:
                - collision_count
                    Number of collisions picked up by sensors.
                - heading_count
                    Number of timesteps where agent heading to target is
                    outside of the specified threshold.
                - total_time
                    Time spent for each environment evaluation.
                - status
                    Termination status of integrator (-1 for failed
                    integration, 0 for reaching final time, and 1 for
                    achieving the termination event, i.e. reaching the
                    target). See scipy.integrate.solve_ivp docs for more
                    information.
        """
        metrics = dict(collision_count=np.zeros(self.N),
                       heading_count=np.zeros(self.N),
                       total_time=np.zeros(self.N),
                       status=np.zeros(self.N),
                       states=[],
                       controls=[])

        args = [(env, solve_ivp_kwargs) for env in self.envs]

        with Pool() as pool:
            results = pool.map(evaluate_env, args)

        for i, met in enumerate(results):
            metrics['collision_count'][i] = met[0]['collision_count']
            metrics['heading_count'][i] = met[0]['heading_count']
            metrics['total_time'][i] = met[0]['total_time']
            metrics['status'][i] = met[0]['status']
            metrics['states'].append(met[2])
            metrics['controls'].append(met[3])

        return metrics


    def reset(self) -> None:
        """Resets RNG for Simulation and for environments.
        """
        self.rng = np.random.integers(low=10_000, high=60_000, size=self.N)

        for env in self.envs:
            env.reset_rng()
