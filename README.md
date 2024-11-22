# SimBa v0.0.1
## Roomba Simulator
### Author: Nate Steffen
### Contact: steffenr@mail.uc.edu


## How To Install
Change directory to where SimBa is located \
`cd path/to/SimBa/`

Create conda environment \
`conda create -f environment.yaml`

Activate environment \
`conda activate simba_env`

Install SimBa \
`pip install -e .`


## Overview
Simulates roomba-like robotic agent in randomized environments. The
states are:

$$\bf{X} = [x, \dot{x}, y, \dot{y}, \vartheta, \dot{\vartheta}]$$

The controls are:
$$\bf{u} = [F_{left}, F_{right}]$$

Robotic agent properties are specified in the `Agent` class.
Environments are generated and evaluated using the `Environment` class.
Multiple random environments are evaluated using the `Simulation` class.
Visualization methods for the states, sensor readings, and animations
are available using `plot_states`, `plot_sensor_readings`, and 
`animate`, respectively. Please see examples for specific implementation.
