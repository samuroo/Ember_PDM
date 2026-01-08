# GENERAL IMPORTS
import time
import math
import argparse
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.Logger import Logger

# PYBULLET DEFAULT VARIABLES
DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 50
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

# AUTHOR CLASS & FUNCTION IMPORTS
from controllers.controller_path import MPCPathController
from dynamics.quadcopter_linear import QuadcopterLinearized
from enviroment.path_vis import draw_path, update_horizon_visualization
from global_solver.solve_rrt_3d_from_urdf import solve_rrt_from_urdf
from global_solver.urdf_to_boxes3d import load_env_and_extract_boxes3d


# main run simulation function
def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    
    # DEFINE START & END (depends on enviroment)
    if args.env == "floor_plan":
        start = (6.0, -10.0, 0.2)
        goal = (43.0, -21.0, 0.2)
    elif args.env == "hallway":
        start = (4.0, 0.0, 1.0)
        goal = (-4.0, 0.0, 1.0)

    # Init quadcopter postion (x,y,z) and orientation (roll,pitch,yaw)
    INIT_XYZS = np.array([start])
    INIT_RPYS = np.array([[0, 0, 0]])

    # solve path and interpolate it for MPC
    if args.env == "floor_plan":
        path = solve_rrt_from_urdf(urdf_path=ENVIROMENT_URDF, algo_name="bit_star", start=start, goal=goal, visualize=args.vis_global_solver)
        path = interpolate_path(path, points_per_segment=85)
    elif args.env == "hallway":
        path = solve_rrt_from_urdf(urdf_path=ENVIROMENT_URDF, algo_name="basic", start=start, goal=goal, visualize=args.vis_global_solver)
        path = interpolate_path(path, points_per_segment=10)

    # Create the environment
    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )

    # Obtain the PyBullet Client ID from the environment
    PYB_CLIENT = env.getPyBulletClient()

    # init the logger, camera, and time variables
    logger = Logger(logging_freq_hz=control_freq_hz, num_drones=num_drones, output_folder=output_folder, colab=colab)
    if args.env == "floor_plan":
        p.resetDebugVisualizerCamera(
            cameraDistance=40,
            cameraYaw=-107,
            cameraPitch=-2,
            cameraTargetPosition=[43.0, -21.0, 0.2],
            physicsClientId=PYB_CLIENT
        )
    elif args.env == "hallway":
        p.resetDebugVisualizerCamera(
            cameraDistance=10,
            cameraYaw=90,
            cameraPitch=-3,
            cameraTargetPosition=[-4.0, 0.0, 1.0],
            physicsClientId=PYB_CLIENT
        )
    dt = 1.0 / env.CTRL_FREQ
    t_wall_start = time.perf_counter()

    # draw the path in the pybullet enviroment and add the collision boxes
    draw_path(path)
    load_env_and_extract_boxes3d("global_solver/" + ENVIROMENT_URDF)

    # init MPC controller
    mpc = MPCPathController(quadcopter, HORIZON_N)

    # init input control array
    action = np.zeros((num_drones,4))

    # Main simulation Loop
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        # Step the simulation with the contol input provided
        obs, _, _, _, _ = env.step(action)
        
        # Main control
        for j in range(num_drones):
            # get current state
            state = obs_to_state(obs, j)
            x_init = state

            # get the current index of the trajctory path based off of the closest waypoint
            d = np.linalg.norm(path - state[0:3], axis=1)
            t_idx = np.argmin(d)
            
            # build a trajectory (12, N+1) from nearest way point
            x_ref_traj = build_ref_traj(path, t_idx, HORIZON_N)

            # visualize the MPC horzion (slows down sim)
            if args.vis_horizon:
                update_horizon_visualization(x_ref_traj)

            # compute control action
            u0 = mpc.solve(x_init, x_ref_traj)
            
            # add next control action
            action[j, :] = u0

            # Keep real time synced
            t_target = t_wall_start + (i + 1) * dt
            t_now = time.perf_counter()
            sleep_time = t_target - t_now
            if sleep_time > 0:
                time.sleep(sleep_time)

        # log actions taken by quadcopters
        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j])

    # Close the environment 
    env.close()

    # Plot the simulation results 
    if plot:
        plot_3d_from_logger(logger, path)
        logger.plot()

        print("Completion percentage: " + str(completion_percentage(logger, path)) + "%")
        print("Completion time: " + round(str(t_now - t_wall_start)) + " s")
        print("RMS Tracking Error: " + str(rms_tracking_error(logger, path)*1000) + " mm")        

########## HELPER FUNCTIONS ##########
# convert the observations from sim to our states
def obs_to_state(obs, j):
    pos = obs[j, 0:3]
    rpy = obs[j, 7:10]
    pos_dot = obs[j, 10:13]
    rpy_dot = obs[j, 13:16]
    return np.concatenate([pos, rpy, pos_dot, rpy_dot])

# build refernce path for MPC (lists of N values)
def build_ref_traj(path, t_idx, N):
    traj = np.zeros((12, N+1))
    steps = len(path)
    for k in range(N+1):
        idx = min(t_idx + k, steps - 1)
        x_ref, y_ref, z_ref = path[idx]
        traj[0, k] = x_ref
        traj[1, k] = y_ref
        traj[2, k] = z_ref
    return traj

def plot_3d_from_logger(logger, path):
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')

    for j in range(logger.NUM_DRONES):
        n = int(logger.counters[j])  # how many samples are valid for this drone

        # Grab x, y, z up to n (ignore preallocated zeros after that)
        xs = logger.states[j, 0, :n]
        ys = logger.states[j, 1, :n]
        zs = logger.states[j, 2, :n]

        # Plot path as a line
        ax.plot(xs, ys, zs, label="Quadcopter")

    ax.plot(path[:,0],path[:,1],path[:,2], label="Path")
    ax.scatter(path[0,0], path[0,1], path[0,2], s=40, marker='o', color="orange")       # start
    ax.scatter(path[-1,0], path[-1,1], path[-1,2], s=40, marker='o', color="orange")    # end

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Drone trajectory from logger")
    ax.legend()

    plt.show(block=False)

def interpolate_path(path, points_per_segment):
    fine_path = []

    for i in range(len(path) - 1):
        p0 = path[i]
        p1 = path[i + 1]

        for alpha in np.linspace(0, 1, points_per_segment, endpoint=False):
            p = (1 - alpha) * p0 + alpha * p1
            fine_path.append(p)

    fine_path.append(path[-1])  # include final point
    return np.array(fine_path)



def rms_tracking_error(logger, path):
    """
    logger: logger object
    path: array of shape (N_ref, 3)
    returns: RMS position error [m]
    """

    # Assume single drone
    j = 0
    n = int(logger.counters[j])

    # Drone positions
    drone_pos = np.vstack((
        logger.states[j, 0, :n],
        logger.states[j, 1, :n],
        logger.states[j, 2, :n],
    )).T  # shape (n, 3)

    # Euclidean error at each timestep
    errors = np.full(len(drone_pos), np.inf)

    for k in range(len(drone_pos)):
        for i in range(len(path)):
            d = math.dist(path[i], drone_pos[k])
            if errors[k] > d:
                errors[k] = d

    # RMS
    rms = np.sqrt(np.mean(errors**2))
    return rms

def completion_percentage(logger, path):
    drone_last_pos = [logger.states[0, 0, -1],
                      logger.states[0, 1, -1],
                      logger.states[0, 2, -1]]
    min_dst = 999
    min_point = np.zeros(np.shape(drone_last_pos))
    percentage = 0
    for i in range(len(path)):
        if min_dst > math.dist(path[i], drone_last_pos):
            min_dst = math.dist(path[i], drone_last_pos)
            min_point = path[i]
            percentage = round(i / float(len(path)) * 100)
    return percentage

#######################################
if __name__ == "__main__":
    # parse the envrioment argument
    parser = argparse.ArgumentParser(description="PDM Quadcopter simulation")
    parser.add_argument("--env", type=str, default="floor_plan", help="Set global enviroment: hallway or floor_plan")
    parser.add_argument("--vis_horizon", type=bool, default=False)
    parser.add_argument("--duration", type=float, default=60, help="Set duration of simualtion in seconds.")
    parser.add_argument("--vis_global_solver", type=bool, default=False)

    args = parser.parse_args()

    # define horizon for MPC controller
    HORIZON_N = 20 

    # define assest enviroment
    if args.env == "hallway":
        ENVIROMENT_URDF = "assets/hallway_env1.urdf"
    elif args.env == "floor_plan":
        ENVIROMENT_URDF = "assets/DunderMifflin_Scranton.urdf"
    else:
        print("No enviroment for " + args.env)
        exit()

    quadcopter = QuadcopterLinearized()
    run(duration_sec=args.duration)


