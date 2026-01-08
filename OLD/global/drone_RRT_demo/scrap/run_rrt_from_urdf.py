import matplotlib.pyplot as plt
from urdf_to_rectangles import load_env_and_extract_rects
from rrt_rect import RRT, show_animation


def main():
    # ------------------------------------------------------
    # 1. Load rectangles from URDF through PyBullet
    # ------------------------------------------------------
    rects, rand_area = load_env_and_extract_rects("assets/hallway_env1.urdf")
    print("Obstacles:", rects)
    print("Random Area:", rand_area)

    # ------------------------------------------------------
    # 2. Define start and goal positions (inside hallway)
    # ------------------------------------------------------
    start = [-4.0, 0.0]
    goal = [4.0, 0.0]

    # ------------------------------------------------------
    # 3. Create and run the RRT planner
    # ------------------------------------------------------
    rrt = RRT(
        start=start,
        goal=goal,
        obstacle_list=rects,
        rand_area=rand_area,
        expand_dis=0.5,
        path_resolution=0.1,
        goal_sample_rate=10,
        max_iter=1000,
        robot_radius=0.1,
    )

    path = rrt.planning(animation=show_animation)

    if path is None:
        print("No path found!")
        return

    print("Path found with", len(path), "points.")

    # ------------------------------------------------------
    # 4. Visualize final path
    # ------------------------------------------------------
    if show_animation:
        rrt.draw_graph()
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        plt.plot(xs, ys, "-r")
        plt.show()


if __name__ == "__main__":
    main()
