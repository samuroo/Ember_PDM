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

vis_sphere_id = None

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
        path = resample_path_n_points(path, n_points=1000)
    elif args.env == "hallway":
        path = solve_rrt_from_urdf(urdf_path=ENVIROMENT_URDF, algo_name="basic", start=start, goal=goal, visualize=args.vis_global_solver)
        path = resample_path_n_points(path, n_points=250)

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
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

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

    vis_sphere_id = None
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
            
            # visualtize the drone with red dot on it
            if args.vis_drone_location:
                if vis_sphere_id is None:
                    visual_id = p.createVisualShape(
                        shapeType=p.GEOM_SPHERE,
                        radius=0.15,
                        rgbaColor=[0, 1, 0, 1]
                    )
                    vis_sphere_id = p.createMultiBody(
                        baseMass=0,  # mass=0 => static, and since no collision shape => visual-only
                        baseVisualShapeIndex=visual_id,
                        basePosition=list([x_init[0], x_init[1], x_init[2]])
                    )
                else:
                    p.resetBasePositionAndOrientation(
                        vis_sphere_id,
                        [float(x_init[0]), float(x_init[1]), float(x_init[2])],
                        [0, 1, 0, 1]
                    )

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

    # Get performance
    comp_perc = completion_percentage(logger, path)
    path_length = logger_length(logger)
    comp_time = t_now - t_wall_start
    avg_vel = path_length / comp_time # m/s
    rms_err = rms_tracking_error(logger, path)*1000
    # Print
    print("||=====================================||")
    if args.env == "floor_plan":
        print("            Floor Plan Results")
    elif args.env == "hallway": 
        print("             Hallway results")
    print("||=====================================||")
    print("  Completion percentage: " + str(round(comp_perc, 2)) + "%")
    print("  Average velocity: " + str(round(avg_vel, 2)) + " m/s")
    print("  RMS Tracking Error: " + str(round(rms_err, 2)) + " mm")
    print("||=====================================||")
    # Save 
    if args.env == "floor_plan":
        filename = "results/results_floor.npz"
    elif args.env == "hallway": 
        filename = "results/results_hallway.npz"
    result_array = np.vstack((comp_perc, avg_vel, rms_err)).T
    append_npz(filename=filename, new_data=result_array, plot=plot)

    

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

### old interploate_path ###
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

def resample_path_n_points(path: np.ndarray, n_points: int) -> np.ndarray:
    """
    Re sample with an even distbuted number of points
    """
    path = np.asarray(path, dtype=float)
    if len(path) < 2 or n_points <= 1:
        return path[:1].copy()

    # cumulative distance s along the path
    seg = path[1:] - path[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg_len)))

    # if all the points are the same
    if s[-1] == 0:
        return np.repeat(path[:1], n_points, axis=0)

    # evenly spaced distances along total length
    s_new = np.linspace(0.0, s[-1], n_points)

    # finally interpolate each dimension independently
    out = np.vstack([np.interp(s_new, s, path[:, d]) for d in range(path.shape[1])]).T
    out[-1] = path[-1]  # exact final point
    return out


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

def logger_length(logger):
    
    # Assume single drone
    j = 0
    n = int(logger.counters[j])

    drone_pos = np.vstack((
        logger.states[0, 0, :],
        logger.states[0, 1, :],
        logger.states[0, 2, :],
    )).T  # shape (n, 3)

    # Length
    d = 0
    for i in range(1, len(drone_pos)):
            d += math.dist(drone_pos[i-1], drone_pos[i])
    return d

def append_npz(filename, new_data, plot):
    try:
        data = np.load(filename)['arr_0']
        data = np.vstack((data, new_data))
    except FileNotFoundError:
        data = new_data
    np.savez(filename, data)

    print("\tMean values over all trials:")
    print("\t----------------------------")
    print("  Completion percentage: " + str(round(np.mean(data[:, 0]), 2)) + "%")
    print("  Average velocity: " + str(round(np.mean(data[:, 1]), 2)) + " m/s")
    print("  RMSE: " + str(round(np.mean(data[:, 2]), 2)) + " mm")   
    print("||=====================================||")

    if plot:
        visualize_results(data)

def visualize_results(data):
    fig, axs = plt.subplots(3, 1, figsize=(7, 15))
    n = range(0, len(data))
    if args.env == "floor_plan":
        axs[0].set_title('Floor Plan Results')
    elif args.env == "hallway": 
        axs[0].set_title('Hallway results')
    axs[0].bar(n, data[:, 0], color='tab:green')
    axs[0].set_ylabel("Completion ratio [%]")
    axs[1].bar(n, data[:, 1], color='tab:cyan')
    axs[1].set_ylabel("Completion velocity [m/s]")
    axs[2].bar(n, data[:, 2], color='tab:orange')
    axs[2].set_xlabel("Measurement [-]")
    axs[2].set_ylabel("RMS Error [mm]")

    plt.show()

#######################################
if __name__ == "__main__":
    # parse the envrioment argument
    parser = argparse.ArgumentParser(description="PDM Quadcopter simulation")
    parser.add_argument("--env", type=str, default="floor_plan", help="Set global enviroment: hallway or floor_plan")
    parser.add_argument("--vis_horizon", type=bool, default=False)
    parser.add_argument("--duration", type=float, default=60, help="Set duration of simualtion in seconds.")
    parser.add_argument("--vis_global_solver", type=bool, default=False)
    parser.add_argument("--vis_drone_location", type=bool, default=False)

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