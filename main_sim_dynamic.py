import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import pybullet_data


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
DEFAULT_DURATION_SEC = 60
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

from controllers.controller_path_dynamic import mpc_control_path
from dynamics.quadcopter_linear import QuadcopterLinearized
from enviroment.path_vis import draw_path, update_horizon_visualization
from global_solver.solve_rrt_3d_from_urdf import solve_rrt_from_urdf
from global_solver.urdf_to_boxes3d import load_env_and_extract_boxes3d

# define horizon for MPC controller
HORIZON_N = 20

# define assest enviroment
ENVIROMENT_URDF = "assets/hallway_env1.urdf"

from pathlib import Path
print("URDF path:", ENVIROMENT_URDF)
print("Exists?", Path(ENVIROMENT_URDF).exists())

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
    
    # DEFINE START & END (depends on enviroment)
    start = (4.0, 0.0, 1.0)
    goal = (-4.0, 0.0, 1.0)

    # SOLVE PATH
    # path = solve_rrt_from_urdf(urdf_path=ENVIROMENT_URDF, start=start, goal=goal, visualize=True)
    path = np.array([
                    [ 4.,          0.,          1.        ],
                    [ 3.52369249,  0.15197371,  0.9940712 ],
                    [ 3.08575409,  0.20210888,  1.23007209],
                    [ 2.73771617,  0.42012299,  0.94487137],
                    [ 2.41997896,  0.57268674,  0.59023441],
                    [ 2.00355015,  0.71128252,  0.35069683],
                    [ 1.50986206,  0.65279185,  0.40409078],
                    [ 1.01617398,  0.59430117,  0.45748472],
                    [ 0.52248589,  0.5358105,   0.51087867],
                    [ 0.0287978,   0.47731982,  0.56427262],
                    [-0.22290227,  0.16549656,  0.86329323],
                    [-0.72209677,  0.14362395,  0.88136088],
                    [-1.09113835, -0.1553079,   0.72500211],
                    [-1.06640259, -0.61824148,  0.91230038],
                    [-0.99211523, -0.84435156,  1.00202259],
                    [-1.32652733, -0.82940177,  1.04343213],
                    [-1.82117994, -0.82064274,  1.09583432],
                    [-2.21032456, -0.52350688,  1.09720863],
                    [-2.64276063, -0.39701285,  1.38041772],
                    [-3.07519671, -0.27051882,  1.4636268 ],
                    [-3.36988517, -0.63868207,  1.2974506 ],
                    [-3.7032025,  -0.30083285,  1.14010556],
                    [-4.,          0.,          1.        ]
                ])



    path = interpolate_path(path)

    # Init postion (x,y,z) and orientation (roll,pitch,yaw)
    INIT_XYZS = np.array([start])
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

    # add collsiion boxes
    load_env_and_extract_boxes3d("global_solver/"+ ENVIROMENT_URDF)

    radiis = np.array([[0.2, 0.1, 0.8],
                       [0.2, 0.1, 0.8]])

    position1 = (0.0, 0.0, -0.2)

    position2 = (-11.4, -1.0, -0.2)

    #Creating moving ellipsoid 1
    ellipsoid_id1 = create_moving_ellipsoid(
    client_id=PYB_CLIENT,
    radii=radiis[0, :],
    position=position1
    )

    ellipsoid_id2 = create_moving_ellipsoid(
    client_id=PYB_CLIENT,
    radii=radiis[1, :],
    position=position2
    )

    # Defining movement

    dt = 1/env.CTRL_FREQ

    times = np.arange(0, duration_sec, dt)

    
    ellipsoid_pos_horizon1 = np.vstack([
        position1[0] + 0.4 * times,
        position1[1] + 0.5 * np.ones_like(times),
        position1[2] + 1.0 * np.ones_like(times)
        ])
    
    ellipsoid_pos_horizon2 = np.vstack([
        position2[0] + 0.4 * times,
        position2[1] + 0.5 * np.ones_like(times),
        position2[2] + 1.0 * np.ones_like(times)
        ])
    
    ellipsoids_horizon = np.vstack((ellipsoid_pos_horizon1, ellipsoid_pos_horizon2))

    
    #Sampling environment boxes

    ENV_BODY_ID = 2  # hallway body id in urdf

    # sample points from all static environment boxes
    env_points = sample_env_boxes_flat(
        client_id=PYB_CLIENT,
        body_id=ENV_BODY_ID,
        links=[0,1,2,3,4,5],  # all links in the hallway (should be updated if environment is changed)
        points_per_link=2000 # number of points sampled per object
    )


    # Main simulation Loop
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        # Step the simulation with the contol input provided
        obs, _, _, _, _ = env.step(action)

        # can define motion of ellipsoid here
        ellipsoid_pos1 = ellipsoid_pos_horizon1[:, i]
        ellipsoid_pos2 = ellipsoid_pos_horizon2[:, i]

        # Ellipsoid 1
        p.resetBasePositionAndOrientation(
            ellipsoid_id1,
            ellipsoid_pos1.tolist(),
            [0, 0, 0, 1],
            physicsClientId=PYB_CLIENT
        )

        ellipsoid_center1, _ = p.getBasePositionAndOrientation(
            ellipsoid_id1,
            physicsClientId=PYB_CLIENT
        )
        # sampling points on the ellipsoid surface
        ellipsoid_points1 = sample_ellipsoid_surface(
            center=np.array(ellipsoid_center1),
            radii=radiis[0, :],
            num_points=2000
        )

        # Ellipsoid 2
        p.resetBasePositionAndOrientation(
            ellipsoid_id2,
            ellipsoid_pos2.tolist(),
            [0, 0, 0, 1],
            physicsClientId=PYB_CLIENT
        )

        ellipsoid_center2, _ = p.getBasePositionAndOrientation(
            ellipsoid_id2,
            physicsClientId=PYB_CLIENT
        )
        # sampling points on the ellipsoid surface
        ellipsoid_points2 = sample_ellipsoid_surface(
            center=np.array(ellipsoid_center2),
            radii=radiis[1, :],
            num_points=2000
        )

        ellipsoids_points = [
            ellipsoid_points1,  # (N,3)
            ellipsoid_points2
        ]

        
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
            # update_horizon_visualization(x_ref_traj)

            # compute control action
            u0 = mpc_control_path(quadcopter, HORIZON_N, x_init, 
                                  x_ref_traj, 
                                  env_points, 
                                  ellipsoids_points, 
                                  ellipsoids_horizon,
                                  i)
            
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

def create_moving_ellipsoid(
    client_id,
    radii=(0.4, 0.25, 0.4),
    position=(0, 0, 0.5),
    color=(1, 0, 0, 0.6)
):
    # Load PyBullet mesh assets
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Use a unit sphere mesh
    sphere_mesh = "sphere_smooth.obj"

    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=sphere_mesh,
        meshScale=radii,
        physicsClientId=client_id
    )

    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=sphere_mesh,
        meshScale=radii,
        rgbaColor=color,
        physicsClientId=client_id
    )

    ellipsoid_id = p.createMultiBody(
        baseMass=0.0,  # kinematic obstacle
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        baseOrientation=[0, 0, 0, 1],
        physicsClientId=client_id
    )

    return ellipsoid_id

def sample_ellipsoid_surface(center, radii, num_points):
    """
    Sample approximately uniform points on the surface of an ellipsoid.

    center: (3,) array
    radii: (a, b, c)
    num_points: int
    """
    a, b, c = radii
    points = np.zeros((num_points, 3))

    for i in range(num_points):
        v = np.random.normal(size=3)
        v /= np.linalg.norm(v)  # unit sphere
        points[i] = center + np.array([a * v[0], b * v[1], c * v[2]])

    return points



def sample_env_boxes_flat(client_id, body_id, links=None, points_per_link=200):
    """
    Sample points from all box-shaped links of a given environment body.
    Returns flat array: [body_id, link_id, x, y, z]
    """
    all_points = []

    if links is None:
        links = list(range(p.getNumJoints(body_id, physicsClientId=client_id)))

    # include base link
    links = [-1] + links

    for link_id in links:
        shape_data = p.getCollisionShapeData(
            body_id, link_id, physicsClientId=client_id
        )
        if not shape_data:
            continue

        geom_type = shape_data[0][2]

        if geom_type != p.GEOM_BOX:
            continue

        aabb_min, aabb_max = p.getAABB(
            body_id, link_id, physicsClientId=client_id
        )

        points = sample_box_surface(np.array(aabb_min), np.array(aabb_max), points_per_link)

        for pt in points:
            all_points.append([body_id, link_id, pt[0], pt[1], pt[2]])

    if len(all_points) == 0:
        return np.empty((0,5))

    return np.array(all_points)

def sample_box_surface(aabb_min, aabb_max, num_points):
    """
    Uniformly sample points on the surface of an axis-aligned box
    """
    points = []
    for _ in range(num_points):
        face = np.random.randint(6)
        x = np.random.uniform(aabb_min[0], aabb_max[0])
        y = np.random.uniform(aabb_min[1], aabb_max[1])
        z = np.random.uniform(aabb_min[2], aabb_max[2])

        if face == 0: x = aabb_min[0]
        elif face == 1: x = aabb_max[0]
        elif face == 2: y = aabb_min[1]
        elif face == 3: y = aabb_max[1]
        elif face == 4: z = aabb_min[2]
        elif face == 5: z = aabb_max[2]

        points.append([x,y,z])

    return np.array(points)






#######################################
if __name__ == "__main__":
   quadcopter = QuadcopterLinearized()
   run()


