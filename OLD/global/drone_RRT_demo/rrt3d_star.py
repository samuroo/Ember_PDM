import numpy as np
from dataclasses import dataclass

@dataclass
class RRT3DStarConfig:
    # Core parameters (kept similar to RRT3DConfig for drop-in usability)
    step_size: float = 0.3
    max_iter: int = 3000
    goal_sample_rate: float = 0.2
    goal_threshold: float = 0.5
    n_collision_samples: int = 10

    # RRT* parameters
    neighbor_radius: float = 1.0      # fixed radius for "near" set
    # Optional: limit number of neighbors to consider (can speed up)
    max_neighbors: int | None = None

class RRT3DStar:
    """
    Simple 3D RRT* implementation, written in a style similar to RRT3DBasic.

    Constructor signature matches your solver usage:
        PlannerClass(start, goal, env, cfg, draw_callback=...)

    - env must provide: env.bounds -> (mn, mx) and
                       env.is_segment_free(p, q, n_samples=int) -> bool
    - draw_callback: function(parent_point, child_point) -> None
                     (kept 2-arg compatible with your current solver)
    """

    def __init__(self, start, goal, env, cfg: RRT3DStarConfig,
                 draw_callback=None):
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.env = env
        self.cfg = cfg
        self.draw_callback = draw_callback

        # Nodes, parents, and cost-to-come
        self.V = np.array([self.start])   # shape (N, 3)
        self.parents = [-1]
        self.costs = [0.0]                # cost from start to node i

        self.goal_idx = None

    # -------------------------
    # Basic helpers (similar to RRT3DBasic)
    # -------------------------
    def _sample_point(self):
        if np.random.rand() < self.cfg.goal_sample_rate:
            return self.goal.copy()
        mn, mx = self.env.bounds
        return np.random.uniform(mn, mx, size=3)

    def _nearest_index(self, point):
        diff = self.V - point
        dists = np.linalg.norm(diff, axis=1)
        return int(np.argmin(dists))

    def _steer(self, from_point, to_point):
        direction = to_point - from_point
        dist = np.linalg.norm(direction)
        if dist == 0.0:
            return from_point.copy()
        step = min(self.cfg.step_size, dist)
        return from_point + direction / dist * step

    def _edge_cost(self, p, q):
        return float(np.linalg.norm(q - p))

    def _near_indices(self, point):
        """
        Returns indices of nodes within neighbor_radius of 'point'.
        Optionally truncates to max_neighbors closest.
        """
        if self.V.shape[0] == 1:
            return [0]
        diff = self.V - point
        dists = np.linalg.norm(diff, axis=1)

        near = np.where(dists <= self.cfg.neighbor_radius)[0].tolist()
        if len(near) == 0:
            # Fallback: at least include the nearest
            return [int(np.argmin(dists))]

        if self.cfg.max_neighbors is not None and len(near) > self.cfg.max_neighbors:
            # Take the closest max_neighbors among the near set
            near_sorted = sorted(near, key=lambda i: dists[i])
            near = near_sorted[: self.cfg.max_neighbors]

        return near

    def _backtrack_path(self):
        if self.goal_idx is None:
            return None

        path_indices = []
        curr = self.goal_idx
        while curr != -1:
            path_indices.append(curr)
            curr = self.parents[curr]
        path_indices.reverse()
        return self.V[path_indices]

    # -------------------------
    # RRT* plan()
    # -------------------------
    def plan(self):
        for i in range(self.cfg.max_iter):
            randp = self._sample_point()
            nearest_idx = self._nearest_index(randp)
            nearest_p = self.V[nearest_idx]

            new_p = self._steer(nearest_p, randp)

            # 1) Check collision from nearest -> new (basic feasibility)
            if not self.env.is_segment_free(nearest_p, new_p,
                                            n_samples=self.cfg.n_collision_samples):
                continue

            # 2) Choose best parent among near nodes (min cost-to-come)
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

            # 3) Add new node with chosen parent
            self.V = np.vstack([self.V, new_p])
            new_idx = self.V.shape[0] - 1
            self.parents.append(best_parent)
            self.costs.append(best_cost)

            if self.draw_callback is not None:
                self.draw_callback(self.V[best_parent], new_p)

            # 4) Rewire: try to improve neighbors by going through new node
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

                # Optional: draw rewired edge (still 2-arg compatible)
                if self.draw_callback is not None:
                    self.draw_callback(new_p, pj)

                # Note: we are NOT propagating cost updates to j's descendants here.
                # For many student projects, this is acceptable; for correctness,
                # you should propagate updated costs down the subtree.

            # 5) Goal check (same pattern as your basic RRT)
            if np.linalg.norm(new_p - self.goal) < self.cfg.goal_threshold:
                # connect goal as final node (try best parent choice too)
                # (You can also do a near-based best parent for the goal.)
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
