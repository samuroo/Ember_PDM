import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

from urdf_to_boxes3d import load_env_and_extract_boxes3d
from environment3d import Environment3D, BoxObstacle3D
from rrt3d_basic import RRT3DBasic, RRT3DConfig


def draw_box3d(ax, box, color="orange", linewidth=1.2):
    """
    Draw a wireframe box on a 3D axis.
    box: (xmin, xmax, ymin, ymax, zmin, zmax)
    """
    xmin, xmax, ymin, ymax, zmin, zmax = box

    xs = [xmin, xmax]
    ys = [ymin, ymax]
    zs = [zmin, zmax]

    # List all 8 vertices
    corners = [
        (xs[0], ys[0], zs[0]),
        (xs[1], ys[0], zs[0]),
        (xs[1], ys[1], zs[0]),
        (xs[0], ys[1], zs[0]),
        (xs[0], ys[0], zs[1]),
        (xs[1], ys[0], zs[1]),
        (xs[1], ys[1], zs[1]),
        (xs[0], ys[1], zs[1]),
    ]

    # Edges: pairs of indices in corners
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom rectangle
        (4, 5), (5, 6), (6, 7), (7, 4),  # top rectangle
        (0, 4), (1, 5), (2, 6), (3, 7)   # verticals
    ]

    for i, j in edges:
        x = [corners[i][0], corners[j][0]]
        y = [corners[i][1], corners[j][1]]
        z = [corners[i][2], corners[j][2]]
        ax.plot(x, y, z, color=color, linewidth=linewidth)


def main():
    urdf_path = "assets/hallway_env1.urdf"

    # -------------------------------------------------
    # 1. URDF -> 3D boxes
    # -------------------------------------------------
    boxes_raw, bounds = load_env_and_extract_boxes3d(urdf_path)

    # Wrap in BoxObstacle3D
    box_obstacles = [
        BoxObstacle3D(*b) for b in boxes_raw
    ]

    # -------------------------------------------------
    # 2. Build Environment3D
    # -------------------------------------------------
    robot_radius = 0.1
    env = Environment3D(box_obstacles, bounds=bounds,
                        robot_radius=robot_radius)

    # -------------------------------------------------
    # 3. Define start/goal in 3D (inside hallway)
    #    Example: same x/y as before, fixed z = 1.0
    # -------------------------------------------------
    start = np.array([-4.0, 0.0, 1.0])
    goal  = np.array([ 4.0, 0.0, 1.0])

    # -------------------------------------------------
    # 4. Setup Matplotlib 3D figure
    # -------------------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D RRT Planner (from URDF)")
    mn, mx = bounds
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.set_zlim(mn, mx)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Draw obstacles
    for b in boxes_raw:
        draw_box3d(ax, b)

    # Draw start & goal
    ax.scatter(start[0], start[1], start[2], c="g", s=50)
    ax.scatter(goal[0], goal[1], goal[2], c="r", s=50)

    plt.ion()
    plt.show()
    plt.pause(0.001)

    # -------------------------------------------------
    # 5. Define draw_callback for RRT (live plotting)
    # -------------------------------------------------
    def draw_edge(p1, p2):
        x = [p1[0], p2[0]]
        y = [p1[1], p2[1]]
        z = [p1[2], p2[2]]
        ax.plot(x, y, z, "b-", linewidth=0.5)
        ax.scatter(p2[0], p2[1], p2[2], c="b", s=5)
        plt.pause(0.001)

    # -------------------------------------------------
    # 6. Choose RRT algorithm (switchable)
    # -------------------------------------------------
    cfg = RRT3DConfig(
        step_size=0.5,
        max_iter=3000,
        goal_sample_rate=0.1,
        goal_threshold=0.5,
        n_collision_samples=10,
    )

    # here is the hook to switch algorithms later
    ALGOS = {
        "basic": RRT3DBasic,
        # "rrt_star": RRT3DStar,  # e.g. add later
    }
    algo_name = "basic"
    PlannerClass = ALGOS[algo_name]

    planner = PlannerClass(start, goal, env, cfg, draw_callback=draw_edge)

    # -------------------------------------------------
    # 7. Run planning
    # -------------------------------------------------
    path = planner.plan()

    if path is not None:
        print("Path found with", path.shape[0], "points.")

        # Plot final path in red
        ax.plot(path[:, 0], path[:, 1], path[:, 2], "r-", linewidth=2.5)
    else:
        print("No path found.")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
