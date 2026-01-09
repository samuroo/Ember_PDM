import numpy as np
from dataclasses import dataclass

@dataclass
# Configuration for RRT* 3D
class RRT3DStarConfig:
    step_size: float = 0.3
    max_iter: int = 3000
    goal_sample_rate: float = 0.2
    goal_threshold: float = 0.5
    n_collision_samples: int = 10
    neighbor_radius: float = 1.0
    max_neighbors: int | None = None

class RRT3DStar:
    """
    3D RRT*
    - env: Environment3D
    - draw_callback: function(parent_point, new_point) -> None for live plotting (can be None)
    """

    def __init__(self, start, goal, env, cfg: RRT3DStarConfig,
                 draw_callback=None):
        """
        Stores the cfg or returns to defaults
        Converts starta nd goal psoitions into numpy arrays
        Saves env collision checker and  and draw callback
        """
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.env = env
        self.cfg = cfg
        self.draw_callback = draw_callback

        # Nodes, parents, and cost-to-come
        self.V = np.array([self.start])     # shape (N, 3), array of node coordinates
        self.parents = [-1]                 # parent index for each node
        self.costs = [0.0]                  # cost from start to node i
        self.goal_idx = None                # stores index of goal node once one is found

    def _sample_point(self):
        """
        With probability of goal_sample_rate returns the goal as the "sampled" point
        Otherwise samples a random point uniformly inside the provided area
        """
        if np.random.rand() < self.cfg.goal_sample_rate:
            return self.goal.copy()
        if hasattr(self.env, "bounds_xyz"):
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.env.bounds_xyz
            return np.array([
                np.random.uniform(xmin, xmax),
                np.random.uniform(ymin, ymax),
                np.random.uniform(zmin, zmax)
            ])
        mn, mx = self.env.bounds
        return np.random.uniform(mn, mx, size=3)

    def _nearest_index(self, point):
        """
        Given a sampled point find the closest tree node eexisiting in V
        """
        diff = self.V - point
        dists = np.linalg.norm(diff, axis=1)
        return int(np.argmin(dists))

    def _steer(self, from_point, to_point):
        """
        Compute the direction from nearest tree node to the sampled point
        Move from nearest tree node to the sampled node by at most step_size
        """
        direction = to_point - from_point
        dist = np.linalg.norm(direction)
        if dist == 0.0:
            return from_point.copy()
        step = min(self.cfg.step_size, dist)
        return from_point + direction / dist * step

    def _edge_cost(self, p, q):
        """
        Euclidean distance between points q and p
        """
        return float(np.linalg.norm(q - p))

    def _near_indices(self, point):
        """
        Returns indices of nodes within neighbor_radius of point
        If none found goes back to the nearest node
        """
        if self.V.shape[0] == 1:
            return [0]
        diff = self.V - point
        dists = np.linalg.norm(diff, axis=1)

        near = np.where(dists <= self.cfg.neighbor_radius)[0].tolist()
        if len(near) == 0:
            return [int(np.argmin(dists))]      # Fallback: at least include the nearest

        if self.cfg.max_neighbors is not None and len(near) > self.cfg.max_neighbors:
            near_sorted = sorted(near, key=lambda i: dists[i])  # Take the closest max_neighbors among the near set
            near = near_sorted[: self.cfg.max_neighbors]
        return near

    def _backtrack_path(self):
        """
        Reconstruct the path by walking backwards through the list until start is reached
        """
        if self.goal_idx is None:
            return None

        path_indices = []
        curr = self.goal_idx
        while curr != -1:
            path_indices.append(curr)
            curr = self.parents[curr]
        path_indices.reverse()
        return self.V[path_indices]

    def plan(self):
        for i in range(self.cfg.max_iter):
            # Pick random target and nearest nod
            randp = self._sample_point()
            nearest_idx = self._nearest_index(randp)
            nearest_p = self.V[nearest_idx]

            # Compute candidate new node
            new_p = self._steer(nearest_p, randp)

            # Reject node if collision detected
            if not self.env.is_segment_free(nearest_p, new_p,
                                            n_samples=self.cfg.n_collision_samples):
                continue

            # Select parenmt node according to lowest cost-to-come
            near_ids = self._near_indices(new_p)

            best_parent = nearest_idx
            best_cost = self.costs[nearest_idx] + self._edge_cost(nearest_p, new_p)

            for j in near_ids:
                pj = self.V[j]
                if not self.env.is_segment_free(pj, new_p,
                                                n_samples=self.cfg.n_collision_samples):
                    continue
                cand_cost = self.costs[j] + self._edge_cost(pj, new_p)
                if cand_cost < best_cost:
                    best_cost = cand_cost
                    best_parent = j

            # Add new node to existing tree + update cost list
            self.V = np.vstack([self.V, new_p])
            new_idx = self.V.shape[0] - 1
            self.parents.append(best_parent)
            self.costs.append(best_cost)

            # Draw tree expansion (optional)
            if self.draw_callback is not None:
                self.draw_callback(self.V[best_parent], new_p)

            # Rewire: try to improve neighbors by going through new node
            for j in near_ids:
                if j == best_parent:
                    continue
                pj = self.V[j]

                # cost if rewired through new node
                new_cost_to_j = self.costs[new_idx] + self._edge_cost(new_p, pj)
                if new_cost_to_j + 1e-12 >= self.costs[j]:
                    continue

                # must be collision-free edge new -> j
                if not self.env.is_segment_free(new_p, pj,
                                                n_samples=self.cfg.n_collision_samples):
                    continue

                # rewire
                old_parent = self.parents[j]
                self.parents[j] = new_idx
                self.costs[j] = new_cost_to_j

                # draw rewired edge (optional)
                if self.draw_callback is not None:
                    self.draw_callback(new_p, pj)

            # Check if goal is reached
            if np.linalg.norm(new_p - self.goal) < self.cfg.goal_threshold:
                if self.env.is_segment_free(new_p, self.goal,
                                            n_samples=self.cfg.n_collision_samples):
                    self.V = np.vstack([self.V, self.goal])
                    goal_idx = self.V.shape[0] - 1
                    self.parents.append(new_idx)
                    self.costs.append(self.costs[new_idx] + self._edge_cost(new_p, self.goal))
                    self.goal_idx = goal_idx
                    print(f"Goal reached in {i+1} iterations (RRT*).")
                    return self._backtrack_path()

        print("Goal not reached within iteration limit (RRT*).")
        return None
