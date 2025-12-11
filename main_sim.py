import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.Logger import Logger

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
DEFAULT_DURATION_SEC = 15
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

from controllers.controller_path import mpc_control_path
from dynamics.quadcopter_linear import QuadcopterLinearized
from enviroment.path_vis import draw_path, update_horizon_visualization

# define horizon for MPC controller
HORIZON_N = 20

# Main run simulation function
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

    # Init postion and orientation (Roll, Pitch Yaw)
    INIT_XYZS = np.array([[-4, 0, 3]])
    INIT_RPYS = np.array([[0, 0, 0]])

    # Init a target (right now only used for linear path)
    TARGET_XYZS = np.array([3,1,2.5])
    
    # amount of way points on the path
    steps = 120

    # FOR TEST PURPOSES - LINEAR PATH
    """
    alphas = np.linspace(0.0, 1.0, steps)
    path = (1 - alphas)[:, None] * INIT_XYZS + alphas[:, None] * TARGET_XYZS
    """

    # FOR TEST PURPOSES - CIRCULAR PATH
    """
    waypoints = np.linspace(0, 7*np.pi/4, steps)
    R = 1.0
    xc, yc, zc = INIT_XYZS[0]
    path = np.zeros((steps, 3))
    path[:,0] = xc + np.sqrt(R**2/2) + R * np.cos(waypoints + 5*np.pi/4)
    path[:,1] = yc + np.sqrt(R**2/2) + R * np.sin(waypoints + 5*np.pi/4)
    path[:,2] = zc
    """

    path = np.array([
        [-4.0,          0.0,          1.0        ],
        [-3.87657877,  -0.46472457,   1.13710679],
        [-3.49441376,  -0.71876212,   0.93857509],
        [-2.99671409,  -0.67102954,   0.94265429],
        [-2.52376163,  -0.82472532,   0.8907546 ],
        [-2.17027851,  -0.63738638,   0.5908316 ],
        [-1.67400351,  -0.58612145,   0.62374098],
        [-1.41475046,  -0.16222689,   0.56805198],
        [-0.91655595,  -0.1473009,    0.60779419],
        [-0.88334149,   0.06837946,   0.15792889],
        [-0.55094397,  -0.21225517,  -0.08855861],
        [-0.06516102,  -0.18959835,   0.02763776],
        [ 0.34052381,  -0.45688493,   0.14586509],
        [ 0.82387744,  -0.39653831,   0.25868156],
        [ 1.11029244,  -0.66760395,   0.56607361],
        [ 1.48708611,  -0.68840557,   0.89408858],
        [ 1.96892046,  -0.55640843,   0.91439639],
        [ 2.13786524,  -0.61260374,   1.38162197],
        [ 2.60407165,  -0.45923149,   1.28607861],
        [ 3.07027807,  -0.30585924,   1.19053525],
        [ 3.23262635,   0.09084943,   1.44795565],
        [ 3.66219765,   0.03999245,   1.19719269],
        [ 4.0,          0.0,          1.0        ]
    ])
    path[:,2] += 2
    path = interpolate_path(path)


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

    # Init the logger
    logger = Logger(logging_freq_hz=control_freq_hz, num_drones=num_drones,
                    output_folder=output_folder, colab=colab)

    # Init input control array
    action = np.zeros((num_drones,4))

    # Draw the path visualy 
    draw_path(path)

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
            update_horizon_visualization(x_ref_traj)

            # compute control action
            u0 = mpc_control_path(quadcopter, HORIZON_N, x_init, x_ref_traj)
            
            # add next control action
            action[j, :] = u0

        # log actions taken by quadcopters
        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j])
        
        # Printout simulation observation (obs)
        #env.render()

    # Close the environment 
    env.close()

    # Plot the simulation results 
    if plot:
        plot_3d_from_logger(logger, path)
        logger.plot()
        


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


def interpolate_path(path, points_per_segment=20):
    fine_path = []

    for i in range(len(path) - 1):
        p0 = path[i]
        p1 = path[i + 1]

        for alpha in np.linspace(0, 1, points_per_segment, endpoint=False):
            p = (1 - alpha) * p0 + alpha * p1
            fine_path.append(p)

    fine_path.append(path[-1])  # include final point
    return np.array(fine_path)
        


#######################################

if __name__ == "__main__":
   quadcopter = QuadcopterLinearized()
   run()
   print("End of Sim")


