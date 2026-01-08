import numpy as np
import math
from dataclasses import dataclass

@dataclass
# Configuration for RRT3D Basic
class RRT3DConfig:
    step_size: float = 0.3
    max_iter: int =  3000
    goal_sample_rate: float = 0.2
    goal_threshold: float = 0.5
    n_collision_samples: int = 10

class RRT3DBasic:
    """
    Simple 3D RRT
    - env: Environment3D
    - draw_callback: function(parent_point, new_point) -> None for live plotting (can be None)
    """

    def __init__(self, start, goal, env, cfg: None, draw_callback=None):
        """
        Stores the cfg or returns to defaults
        Converts starta nd goal psoitions into numpy arrays
        Saves env collision checker and  and draw callback
        """
        self.cfg = cfg if cfg is not None else RRT3DConfig()
        
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.env = env
        self.draw_callback = draw_callback

        # Vertices and edges
        self.V = np.array([self.start])     # shape (N, 3), array of node coordinates
        self.parents = [-1]                 # parent index for each node
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

            # Add new node to existing tree
            self.V = np.vstack([self.V, new_p])
            new_idx = self.V.shape[0] - 1
            self.parents.append(nearest_idx)

            # Draw tree expansion (optional)
            if self.draw_callback is not None:
                self.draw_callback(nearest_p, new_p)

            # Check if goal is reached
            if np.linalg.norm(new_p - self.goal) < self.cfg.goal_threshold:     # Add goal as final node and connect
                self.V = np.vstack([self.V, self.goal])
                goal_idx = self.V.shape[0] - 1
                self.parents.append(new_idx)
                self.goal_idx = goal_idx
                print(f"Goal reached in {i+1} iterations.")
                return self._backtrack_path()

        print("Goal not reached within iteration limit.")
        return None

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