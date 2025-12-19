import numpy as np
import math
from dataclasses import dataclass

@dataclass
class RRT3DConfig:
    step_size: float = 0.5
    max_iter: int =  2000
    goal_sample_rate: float = 0.1
    goal_threshold: float = 0.5
    n_collision_samples: int = 10

class RRT3DBasic:
    """
    Simple 3D RRT

    - env: Environment3D
    - draw_callback: function(parent_point, new_point) -> None for live plotting (can be None)
    """

    def __init__(self, start, goal, env, cfg: RRT3DConfig,
                 draw_callback=None):
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.env = env
        self.cfg = cfg
        self.draw_callback = draw_callback

        # Vertices and edges
        self.V = np.array([self.start])  # shape (N, 3)
        self.parents = [-1]              # parent index for each node
        self.goal_idx = None

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
        """
        Move from 'from_point' toward 'to_point' by at most step_size.
        """
        direction = to_point - from_point
        dist = np.linalg.norm(direction)
        if dist == 0.0:
            return from_point.copy()
        step = min(self.cfg.step_size, dist)
        return from_point + direction / dist * step
    
    def plan(self):
        for i in range(self.cfg.max_iter):
            randp = self._sample_point()
            nearest_idx = self._nearest_index(randp)
            nearest_p = self.V[nearest_idx]

            new_p = self._steer(nearest_p, randp)

            # Collision checking for segment nearest_p -> new_p
            if not self.env.is_segment_free(nearest_p, new_p,
                                            n_samples=self.cfg.n_collision_samples):
                continue

            # Add new node
            self.V = np.vstack([self.V, new_p])
            new_idx = self.V.shape[0] - 1
            self.parents.append(nearest_idx)

            # Draw tree expansion, if requested
            if self.draw_callback is not None:
                self.draw_callback(nearest_p, new_p)

            # Check goal
            if np.linalg.norm(new_p - self.goal) < self.cfg.goal_threshold:
                # Add goal as final node and connect
                self.V = np.vstack([self.V, self.goal])
                goal_idx = self.V.shape[0] - 1
                self.parents.append(new_idx)
                self.goal_idx = goal_idx
                print(f"Goal reached in {i+1} iterations.")
                return self._backtrack_path()

        print("Goal not reached within iteration limit.")
        return None

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