import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# === Define environment ===

bounds = np.array([[0, 100],        # x-axis field limits
                   [0, 100]])       # y-axis field limits

# Obstacles: [x_center, y_center, radius]
obstacles = np.array([
    [20, 20, 8],
    [46, 32, 12],
    [72, 60, 9]
])

startp  = np.array([5, 5])
endp    = np.array([95, 95])

# === Figure setup ===
fig, ax = plt.subplots()
ax.set_aspect('equal', 'box')       # maintains x/y ration in graph
ax.set_xlim(bounds[0])
ax.set_ylim(bounds[1])
ax.set_title("RRT with heuristic")
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Plot obstacles as circles
for (cx, cy, r) in obstacles:
    ax.add_patch(Circle((cx, cy), r, fill=False))

# Plot start and goal
ax.plot(startp[0], startp[1], 'go', markersize=10, linewidth=2)
ax.plot(endp[0],   endp[1],   'ro', markersize=10, linewidth=2)

plt.ion()
plt.show()
plt.pause(0.001)

# === RRT parameters ===

it_max = 10000   # Max iterations
rob_r = 5        # Robot radius
increment = 3    # Step size

heuristic_slack = 1             # allowable deviation from goal

# Vertices and edges
V = np.array([startp])  # shape (N, 2)
E = []                  # list of (parent_idx, child_idx)

goal_idx = None         # will store goal index if reached

# === Main RRT loop ===

for i in range(it_max):

    # Uniform random sampling (no direct goal sampling)
    randp = np.array([
        np.random.rand() * bounds[0, 1],
        np.random.rand() * bounds[1, 1]
    ])

    # Distances from vertices to sample
    distances = np.linalg.norm(V - randp, axis=1)
    nearest_idx = np.argmin(distances)

    if distances[nearest_idx] == 0:
        continue

    # Step toward sampled point
    direction = (randp - V[nearest_idx]) / distances[nearest_idx]
    new_p = V[nearest_idx] + direction * increment

    # --- Heuristic: prefer moves that don't go much further from the goal ---
    h_parent = np.linalg.norm(V[nearest_idx] - endp)
    h_new = np.linalg.norm(new_p - endp)

    # Strict version (only closer): if h_new >= h_parent: continue
    # Softer version with slack:
    if h_new > h_parent + heuristic_slack:
        continue

    # === Collision checking ===
    n_points = 10
    dx = np.linspace(0.0, increment, n_points)
    line_points = V[nearest_idx] + dx[:, None] * direction

    centers = obstacles[:, :2]
    radii = obstacles[:, 2]
    clearance = radii + rob_r

    dist_new = np.linalg.norm(new_p - centers, axis=1)
    dist_line = np.linalg.norm(line_points[:, None, :] - centers[None, :, :], axis=2)

    if np.all(dist_new > clearance) and np.all(dist_line > clearance):

        V = np.vstack([V, new_p])
        new_idx = V.shape[0] - 1
        E.append((nearest_idx, new_idx))

        ax.plot([V[nearest_idx, 0], new_p[0]],
                [V[nearest_idx, 1], new_p[1]], 'b-')
        ax.plot(new_p[0], new_p[1], 'b.', markersize=4)
        plt.pause(0.001)

        if np.linalg.norm(new_p - endp) < increment:
            V = np.vstack([V, endp])
            goal_idx = V.shape[0] - 1
            E.append((new_idx, goal_idx))
            print("Goal reached!")
            break

# === Backtrack ===
if goal_idx is not None:
    E_arr = np.array(E, dtype=int)
    path = [goal_idx]
    curr = goal_idx
    while curr != 0:
        parent = E_arr[E_arr[:, 1] == curr, 0][0]
        curr = parent
        path.insert(0, curr)

    for k in range(len(path) - 1):
        p1 = V[path[k]]
        p2 = V[path[k+1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=3)

    plt.ioff()
    plt.show()
else:
    print("Goal not reached within iteration limit.")
    plt.ioff()
    plt.show()