# Install & Setup:
* Ensure you have Conda (Anaconda) installed
* Clone the project from GitHub
```bash
git clone https://github.com/samuroo/Ember_PDM.git
```
* Create the conda envriroment and activate it
```bash
cd Ember_PDM
conda env create -f environment.yml
conda activate PDM_EMBER_ENV
```

# Directory/File Overview
## `main_sim.py`
One of two main python files to run simulation. This script uses `controller_path.py` on user desired static enviroment. See below for changable parameters.

### Running `main_sim.py`
In-line command options:
* --env : Global environment: hallway or floor_plan (default: floor_plan)
* --duration : Simulation duration in seconds (default: 60)
* --vis_horizon : Visualize MPC prediction horizon
* --vis_global_solver : Visualize global path planner

### Examples of command line options
Run hallway environment for 30 seconds:
```bash
python main_sim.py --env hallway --duration 30
```

Visualize MPC horizon:
```bash
python main_sim.py --vis_horizon True
```

Visualize Global Solver solve:
```bash
python main_sim.py --vis_global_solver True
```

## `Dynamics`
This directory contains the quadcopters core dynamic models and system matrices.
### `quadcopter_linear.py`
This file defines a differential flatness based linearized quadcopter model with a full 12-state representation, along with the MPC weight matrices used by `controller_path.py`.

## `Controllers`  
This directory holds various MPC controllers.  
### `controller_path.py`
A 12-state MPC controller for tracking a path of length N. It is used in `main_sim.py` under a static environment. The terminal cost and terminal set are also included.
### `controller_path_dynamic.py`
A 12-state MPC controller for tracking a path of length N. It is used in `main_sim_dynamic.py` under a dynamic environment.

## `Environment`
This directory contains visualization and pybullet environment related utilities. 
### `path_vis.py`
Given a path, this module plots a 3D line within the environment and visualizes the MPC prediction horizon using spheres along the path. Note: building/envrioment models are not implemented here.

`Enviroment`:  
    Directory to hold any sort of visual or enviromental classes/functions/etc.  
    path_vis.py : provided a path, plots a line in 3D space within the envrioment, also plots spheres along the path for MPC horizon prediction  
    (We should add our building in this directory)  

`Global_solver`:  
    Directory to hold global path planner code and environment urdf's. 
     environment3d.py : defines 3D collision-checking environment using axis-aligned boxes
     urdf_to_boxes3d.py : loads urdf file into PyBullet and converts to a set of axis aligned 3D boxes
     solve_rrt_from_urdf.py : runs a selected 3D motion planner on a selected urdf world
     rrt3d_basic.py : standard RRT path finding algorithm
     rrt3d_star.py : RRT* path finding algorithm
     rrt3d_connect.py : RRT-Connect path finding algorithm
     bit_star.py : BIT* path finding algorithm
     assets/ : contains the urdf files of the tested environments
