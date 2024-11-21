"""Defines the environment class for simulation."""
import numpy as np
from scipy.integrate import solve_ivp
from shapely import Polygon, LinearRing, Point

from agent import Agent
from dynamics import plant
from utils import generate_verts


class Environment:
    def __init__(self,
                 agent : Agent,
                 t_dur : float=18.,
                 bounds : list[tuple[float]]=[(-5., 5.), (-5., 5.)],
                 num_obstacles : int=3,
                 obst_size_bounds : tuple[float]=(0.5, 1.5),
                 rot_bounds : tuple[float]=(0., 45.),
                 sep_factor : float=1.5,
                 epsilon : float=0.01,
                 seed : int=12345,
                 target_thresh : float=0.01
                 ) -> None:
        """Instantiates environment for simulation.

        Parameters
        ----------
        agent : Agent
            Robotic agent to traverse environment.
        t_dur : float, optional
            Duration of the simulation, by default 18 seconds.
        bounds : list[tuple[float]], optional
            Bounds of the environment, by default
            [(-5., 5.), (-5., 5.)].
        num_obstacles : int, optional
            Number of obstacles in the environment, by default 3.
        obst_size_bounds : tuple[float], optional
            Bounds for the size of the obstacles, by default (0.5, 1.5).
        rot_bounds : tuple[float], optional
            Bounds for how the obstacles are rotated, by default
            (0., 45.).
        sep_factor : float, optional
            Factor to be multiplied with agent's radius to determine how
            separated obstacles should be, by default 1.5.
        epsilon : float, optional
            Collision coefficient for agent's velocity after colliding
            with an obstacle, by default 0.01.
        seed : int, optional
            Seeds the random number generator for the environment
            generation, by default 12345.
        target_thresh : float
            Threshold of distance to target to consider achieving the
            target.
        """
        self.agent = agent
        self.t_dur = t_dur
        self.bounds = bounds
        self.num_obstacles = num_obstacles
        self.obst_size_bounds = obst_size_bounds
        self.rot_bounds = rot_bounds
        self.sep_factor = sep_factor
        self.epsilon = epsilon
        self.seed = seed
        self.target_thresh = target_thresh

        self.rng = np.random.default_rng(seed)
        
        self.bound_verts = np.array([[bounds[0][0], bounds[1][0]],
                                     [bounds[0][1], bounds[1][0]],
                                     [bounds[0][1], bounds[1][1]],
                                     [bounds[0][0], bounds[1][1]]])

    # will generate roomba position and target
    def _generate_environment(self):
        self.obstacles = []
        self.verts = []

        for _ in range(self.num_obstacles):
            k = self.rng.integers(low=2, high=8)
            self.verts.append(generate_verts(k,
                                             self.sep_factor*self.agent.r,
                                             self.obstacles,
                                             self.bounds,
                                             self.obst_size_bounds,
                                             self.rot_bounds))
            self.obstacles.append(Polygon(self.verts[-1]))

        self.obstacles.append(LinearRing(self.bound_verts))

        while True:
            x0 = self.rng.uniform(low=self.bounds[0][0],
                                  high=self.bounds[0][1])
            y0 = self.rng.uniform(low=self.bounds[1][0],
                                  high=self.bounds[1][1])
            xt = self.rng.uniform(low=self.bounds[0][0],
                                   high=self.bounds[0][1])
            yt = self.rng.uniform(low=self.bounds[1][0],
                                  high=self.bounds[1][1])

            X0 = Point(x0, y0)
            Xt = Point(xt, yt)

            x0_within_check = any([X0.within(obst) for obst in \
                                   self.obstacles[:-1]])
            xt_within_check = any([Xt.within(obst) for obst in \
                                   self.obstacles[:-1]])

            # TODO: May want to put something in about initial state and
            # target state being a certain distance from each other

            if not x0_within_check and not xt_within_check:
                break

        # NOTE: We could make the initial velocities a random number too
        # xdot0 = self.rng.uniform(low=-1.5, high=1.5)
        # ydot0 = self.rng.uniform(low=-1.5, high=1.5)
        # thetadot0 = self.rng.uniform(low=-np.pi, high=np.pi)

        xdot0 = 0.
        ydot0 = 0.
        thetadot0 = 0.

        theta0 = self.rng.uniform(low=0., high=2.*np.pi)

        self.X0 = np.array([x0, xdot0, y0, ydot0, theta0, thetadot0])
        self.target = np.array([xt, yt])

    def dynamics(self, t, X):
        verts = self.agent.verts + np.array([X[0], X[2]])[:, np.newaxis]
        agent_poly = Polygon(verts.T)
        collision_check = any([agent_poly.distance(obst) <= \
                               self.agent._collision_thresh for obst in \
                               self.obstacles])

        if collision_check:
            X[1] *= -self.epsilon
            X[3] *= -self.epsilon

        u = self.agent.controls(t, X, self.target, self.obstacles)

        return plant(t, X, u, self.agent.m, self.agent.r, self.agent.kd,
                     self.agent.kn)

    def collision_event(self, t, X):
        verts = self.agent.verts + np.array([X[0], X[2]])[:, np.newaxis]
        agent_poly = Polygon(verts.T)
        dist = np.min(np.array([agent_poly.distance(obst) for obst in \
                                self.obstacles]))

        return dist - self.agent._collision_thresh

    def target_event(self, t, X):
        dist = np.sqrt((self.target[0] - X[0])**2 + (self.target[1] - X[2])**2)
        return dist - self.target_thresh

    def evaluate(self, **solve_ivp_kwargs) -> tuple[dict, np.ndarray]:
        """Performs single evaluation of agent in enrionment.

        Returns
        -------
        tuple[dict, np.ndarray]
            Metrics, time steps, and states.
        """
        self.agent.reset()
        self._generate_environment()

        metrics = dict(collision_count=None,
                       heading_count=None,
                       total_time=None,
                       status=None)

        sol = solve_ivp(self.dynamics, (0., self.t_dur), self.X0,
                        events=[self.collision_event, self.target_event],
                        **solve_ivp_kwargs)

        metrics['collision_count'] = self.agent.collision_count
        metrics['heading_count'] = self.agent.heading_count
        metrics['total_time'] = sol.t[-1]
        metrics['status'] = sol.status

        return metrics, sol.t, sol.y

    def reset_rng(self):
        """Resets the RNG for the environment.
        """
        self.rng = np.random.default_rng(self.seed)

Environment.target_event.terminal = True


# if __name__ == '__main__':
#     agent = Agent()
#     env = Environment(agent, num_obstacles=4)
#     metrics, t, y = env.evaluate(t_eval=np.linspace(0., env.t_dur, 100))
#     print(env.X0)
#     print(env.target)
#     print(metrics)

#     from visualize import plot_states, animate, plot_sensor_readings

#     plot_states(t, y)
#     animate(t, y, agent, env.target, env.verts, env.bounds, interval=100)
#     plot_sensor_readings(agent)
