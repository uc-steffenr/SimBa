"""Defines agent class for roomba."""
import numpy as np
from shapely import Polygon, LineString, intersection
from ._controls import controls


class Agent:
    def __init__(self, **properties) -> None:
        """Initializes agent.

        Parameters
        ----------
        r : float
            Radius of agent, by default 0.165 m.
        m : float
            Mass of agent, by default 2.7 kg.
        num_sensors : int
            Number of equally spaced sensors on agent, by default 3.
        offset : float
            Offset of starting sensor angle, by default 60 degrees.
        sensor_dist : float
            Distance that the proximity sensor can see to, by default 2
            meteres.
        f_max : float
            Maximum force the wheels can apply, by default 2.7.
        kd : float
            Drag coefficient of agent's wheels, by default 3.4.
        kn : float
            Normal coefficient of agent's wheels, by default 1,000.
        controls : callable
            Control function for agent. Must have arguments t
            (time: float), X (state: 1x6 np.ndarray), target (target
            state: 1x2 np.ndarray), and readings (sensor readings:
            1xnum_sensors np.ndarray). It returns a control vector for
            the left and right wheel (1x2 np.ndarray). By default, it's
            a time-based controller.
        collision_thresh : float
            Threshold value for sensors to consider a collision, by
            default 0.001.
        heading_thresh : float
            Threshold value for heading to differ from target heading
            to be considered as a cost, by default 0.01.
        """
        # agent characteristics
        self.r = properties.get('r', 0.165)
        self.m = properties.get('m', 2.7)
        self.num_sensors = properties.get('num_sensors', 3)
        self.offset = properties.get('offset', np.deg2rad(60.))
        self.sensor_dist = properties.get('sensor_dist', 2.)
        self.f_max = properties.get('f_max', 2.7)
        self.kd = properties.get('kd', 3.4)
        self.kn = properties.get('kn', 1_000.)

        # internal variables used for controls and metrics
        self._controls = properties.get('controls', controls)
        self._collision_thresh = properties.get('collision_thresh', 0.001)
        self._heading_thresh = properties.get('heading_thresh', 0.01)

        # set up metrics
        self.collision_count = 0.
        self.heading_count = 0.

        # TODO: this stuff is ugly... want to fix it later
        # calculate vertices for reuse in Polygons
        self.verts = np.array([
            self.r*np.cos(np.linspace(0., 2*np.pi, 50)),
            self.r*np.sin(np.linspace(0., 2.*np.pi, 50))
            ])

        self.angs = 2.*np.pi*(np.arange(self.num_sensors)/self.num_sensors) + \
                    self.offset
        
        self.sensor_readings = []
        self.control_actions = []

    def sensor_reading(self,
                       X : np.ndarray,
                       obstacles : list[Polygon]
                       ) -> np.ndarray:
        """Evaluates proximity sensor reading.

        Parameters
        ----------
        X : np.ndarray
            Current state.
        obstacles : list[Polygon]
            List of obstacles in environment.

        Returns
        -------
        np.ndarray
            Proximity values read by sensor.
        """
        # calculate sensor lines at current state
        angs = self.angs + X[4]
        sensor_verts = np.array([[X[0] + self.r*np.cos(angs),
                                  X[2] + self. r*np.sin(angs)],
                                 [X[0] + (self.r+self.sensor_dist)*np.cos(angs),
                                  X[2] + (self.r+self.sensor_dist)*np.sin(angs)
                                  ]]).transpose(2, 0, 1)

        sensors = [LineString(sv) for sv in sensor_verts]

        # find sensor intersections with obstacles
        intersections = np.array([[intersection(sens, obst) \
                                   for sens in sensors] for obst in obstacles])

        # find candidate sensor readings to obstacles
        readings = np.ones_like(intersections)*self.sensor_dist
        for i, obst in enumerate(intersections):
            for j, inter in enumerate(obst):
                if inter.length > 0. or inter.geom_type == 'Point':
                    x1 = inter.coords.xy[0][0]
                    y1 = inter.coords.xy[1][0]
                    readings[i, j] = np.sqrt((sensor_verts[j, 0, 0] - x1)**2 + \
                                             (sensor_verts[j, 0, 1] - y1)**2)

        # return minimum values of candidate sensor readings
        readings = np.min(readings, axis=0)
        self.sensor_readings.append(readings)
        return readings

    def controls(self,
                 t : float,
                 X : np.ndarray,
                 target : np.ndarray,
                 obstacles : list[Polygon]
                 ) -> np.ndarray:
        """Calculates controls of agent and tracks collision and heading
        metrics.

        Parameters
        ----------
        t : float
            Current time.
        X : np.ndarray
            Current state.
        target : np.ndarray
            Target state (only x and y).
        obstacles : list[Polygon]
            List of obstacles in environment.

        Returns
        -------
        np.ndarray
            Left and right wheel forces.
        """
        if len(obstacles) > 0:
            readings = self.sensor_reading(X, obstacles)
        else:
            readings = np.ones(self.num_sensors)*self.sensor_dist

        if np.any(readings < self._collision_thresh):
            self.collision_count += 1

        target_heading = np.arctan((target[1] - X[2])/(target[0] - X[0]))
        if np.abs(target_heading - X[4]) > self._heading_thresh:
            self.heading_count += 1

        u = self._controls(t, X, target, readings)

        # enforce f_max
        # TODO: come back and see if it's possible to just use one value
        u = np.min([u, np.ones_like(u)*self.f_max], axis=0)
        u = np.max([u, np.ones_like(u)*-self.f_max], axis=0)

        return u

    def reset_collision_count(self):
        """Resets collision metric count.
        """
        self.collision_count = 0

    def reset_heading_count(self):
        """Resets heading metric count.
        """
        self.heading_count = 0

    def reset_sensor_readings(self):
        """Resets sensor readings list.
        """
        self.sensor_readings = []

    def reset_control_actions(self):
        """Resets control actions list.
        """
        self.control_actions = []

    def reset(self):
        """Resets metrics.
        """
        self.reset_collision_count()
        self.reset_heading_count()
        self.reset_sensor_readings()
        self.reset_control_actions()
