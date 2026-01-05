Setup:
* Install pybullet, numpy, scipy and cxvpy
* Install gym-pybullet-drones and its dependecies with: https://github.com/utiasDSL/gym-pybullet-drones    
* Launch the main simulation after activating the ```drones``` environment.  

`main_sim.py`:  
    Main python file to run simulation. Calls upon a controller, dynamical quadcopter drone model, and uses enviroment to set up its enviroment.  

`Dynamics`:  
     Directory to hold the basic A and B dynamic matrices in class structure, also discretized for MPC.  
    quadcopter_linear.py : differential flatness matrices 12-state  

`Controllers`:  
    Directory to hold various controllers  
    controller_path.py : 12-state MPC controller for a trajctroy path of N values  

`Enviroment`:  
    Directory to hold any sort of visual or enviromental classes/functions/etc.  
    path_vis.py : provided a path, plots a line in 3D space within the envrioment, also plots spheres along the path for MPC horizon prediction  
    (We should add our building in this directory)  

`Global`:  
    Directory to hold global path planner code  
    (Empty now)  
