import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# === Functions ===

def sample_free(bounds):
    """
    Sample a random point uniformly within the given bounds.

    bounds: np.array of shape (2, 2)
        bounds[0] = [xmin, xmax]
        bounds[1] = [ymin, ymax]
    """
    x = np.random.rand() * (bounds[0, 1] - bounds[0, 0]) + bounds[0, 0]
    y = np.random.rand() * (bounds[1, 1] - bounds[1, 0]) + bounds[1, 0]
    return np.array([x, y])

def steer(x_nearest, x_rand, step_size):
    """
    Steer from x_nearest toward x_rand by at most step_size.

    Returns the new point x_new.
    If x_rand is closer than step_size, returns x_rand.
    """
    direction = x_rand - x_nearest
    dist = np.linalg.norm(direction)
    if dist == 0:
        return x_nearest.copy()
    if dist <= step_size:
        return x_rand.copy()
    # Move only step_size along the direction vector
    return x_nearest + (direction / dist) * step_size

def collision_free(p1, p2, obstacles, rob_r, n_points=20):
    """
    Check if the straight-line segment from p1 to p2 is collsion-free.

    We treat obstacles as circles and the robot as a circle with radius rob_r
    To simplify, we inflate each obstacle radius by rob_r and then require that
    every sampled point along the segment is outside all inflated obstacles.

    Parameters
    ----------
    p1, p2 : np.array shape (2,)
        Segment endpoints.
    obstacles : np.array shape (Nobs, 3)
        Each row: [cx, cy, r]  for circular obstacles.
    rob_r : float
        Robot radius.
    n_points : int
        Number of sample points along the segment (including endpoints).
    """
    # Linearly interpolate between p1 and p2
    t_values = np.linspace(0, 1, n_points)
    segment_points = p1 + (p2 -p1) * t_values[:, None]  # shape (n_points, 2)

    centers = obstacles[:, :2]          # (N_obs, 2)
    radii = obstacles[:, 2] + rob_r     # inflated radii (clearance)

    # For each segment point, check distance to each obstacle center
    # Result shape: (n_points, N_obs)
    diff = segment_points[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=2)

    # Collision occurs if any point is inside any inflated obstacle radius
    # So collision-free means all distances are > radii
    return np.all(dists > radii)


# === Main RRT* function ===

def main():
    # --- Environment Declaration ---
    bounds = np.array([[0, 100],        # x-axis limits [xmin, xmax]
                       [0, 100]])       # y-axis limits [ymin, ymax]
    
    # Obstacles : [x_center, y_center, radius]
    obstacles = np.array([
        [20, 20, 8],
        [46, 32, 12],
        [72, 60, 9]
    ])

    startp  = np.array([5, 5])
    endp    = np.array([95, 95])

    # Visualisation setup
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_title("RRT* Planner")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Draw obstacles as circles
    for (cx, cy, r) in obstacles:
        ax.add_patch(Circle((cx, cy), r, fill=False))

    # Draw start (green) and goal (red)
    ax.plot(startp[0], startp[1], 'go', markersize=10, linewidth=2, label='Start')
    ax.plot(endp[0],   endp[1],   'ro', markersize=10, linewidth=2, label='Goal')

    ax.legend(loc="upper left")

    plt.ion()
    plt.show()
    plt.pause(0.001)


    # --- RRT* Parameters ---
    it_max     = 5000        # Maximum number of iterations
    step_size  = 3.0         # Maximum distance to move toward a sample
    rob_r      = 5.0         # Robot radius for collision checking
    neigh_rad  = 15.0        # Neighborhood radius for RRT* (choose parent & rewire)

    # RRT* data structures
    V = [startp]             # List of nodes (2D points)
    parent = [-1]            # parent[i] = index of parent of node i (start has no parent)
    cost = [0.0]             # cost[i] = cost from start to node i

    goal_idx = None          # Index of goal node in V once connected

    # --- Main RRT* loop ---
    for it in range(it_max):

        # 1) Sample a random point in free space
        x_rand = sample_free(bounds)

        # 2) Find nearest existing node in the tree
        V_array = np.array(V)  # (N, 2)
        dists = np.linalg.norm(V_array - x_rand, axis=1)
        nearest_idx = np.argmin(dists)
        x_nearest = V[nearest_idx]

        # 3) Steer from nearest node toward the random sample
        x_new = steer(x_nearest, x_rand, step_size)

        # 4) Check collision for edge (nearest -> new)
        if not collision_free(x_nearest, x_new, obstacles, rob_r):
            continue  # Sample rejected

        # 5) RRT* choose best parent among neighbors
        #    - Find all existing nodes within neigh_rad of x_new
        dists_new = np.linalg.norm(V_array - x_new, axis=1)
        neighbor_indices = np.where(dists_new < neigh_rad)[0]

        # If no neighbors found (should at least get nearest), fall back to nearest
        if len(neighbor_indices) == 0:
            neighbor_indices = np.array([nearest_idx])

        # Initialize best parent as the nearest node
        best_parent = nearest_idx
        best_cost = cost[nearest_idx] + np.linalg.norm(x_new - V[nearest_idx])

        # Try all neighbors as potential parents (if collision-free)
        for idx in neighbor_indices:
            if idx == nearest_idx:
                continue  # We'll handle nearest anyway
            if not collision_free(V[idx], x_new, obstacles, rob_r):
                continue  # Can't connect collision-free, skip
            c = cost[idx] + np.linalg.norm(x_new - V[idx])
            if c < best_cost:
                best_cost = c
                best_parent = idx

        # 6) Actually add x_new to the tree with the best parent
        V.append(x_new)
        new_idx = len(V) - 1
        parent.append(best_parent)
        cost.append(best_cost)

        # Draw the new edge (best_parent -> new_idx)
        x_p = V[best_parent]
        ax.plot([x_p[0], x_new[0]],
                [x_p[1], x_new[1]], 'b-', linewidth=0.8)
        ax.plot(x_new[0], x_new[1], 'b.', markersize=3)
        plt.pause(0.001)

        # 7) Rewire neighbors: see if going through x_new improves their cost
        for idx in neighbor_indices:
            if idx == new_idx:
                continue  # Skip the new node itself
            # Cost if we go new_idx -> idx
            if not collision_free(V[new_idx], V[idx], obstacles, rob_r):
                continue  # Can't connect collision-free

            new_cost_via_new = cost[new_idx] + np.linalg.norm(V[new_idx] - V[idx])

            # If going via new node is cheaper, update parent and cost
            if new_cost_via_new < cost[idx]:
                parent[idx] = new_idx
                cost[idx] = new_cost_via_new
                # NOTE: We don't erase old drawn edges; the final path will still be correct,
                #       but the blue "tree" drawing may show some outdated edges.

        # 8) Check if we can connect the new node to the actual goal
        #    If close enough and collision-free, add the goal as a node and stop.
        if np.linalg.norm(x_new - endp) <= step_size:
            if collision_free(x_new, endp, obstacles, rob_r):
                V.append(endp)
                goal_idx = len(V) - 1
                parent.append(new_idx)
                cost.append(cost[new_idx] + np.linalg.norm(endp - x_new))

                # Draw final edge to goal
                ax.plot([x_new[0], endp[0]],
                        [x_new[1], endp[1]], 'g-', linewidth=1.5)
                plt.pause(0.001)

                print(f"Goal reached at iteration {it}, cost = {cost[goal_idx]:.2f}")
                break


    # === Backtrack and plot final result ===
    if goal_idx is not None:
        # Build path from goal back to start using the parent array
        path_indices = []
        curr = goal_idx
        while curr != -1:
            path_indices.append(curr)
            curr = parent[curr]
        path_indices = path_indices[::-1]  # reverse to go start -> goal

        # Plot the final RRT* path in thick red
        for i in range(len(path_indices) - 1):
            p1 = V[path_indices[i]]
            p2 = V[path_indices[i + 1]]
            ax.plot([p1[0], p2[0]],
                    [p1[1], p2[1]], 'r-', linewidth=3)

        plt.ioff()
        plt.show()
    else:
        print("Goal not reached within iteration limit.")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
