import numpy as np
from dataclasses import dataclass

@dataclass
class RRT3DConnectConfig:
    step_size: float = 0.3
    max_iter: int = 3000
    goal_threshold: float = 0.5
    n_collision_samples: int = 10

class RRT3DConnect:
    """
    Bidirectional RRT-Connect in 3D.

    Drop-in compatible with your solver:
        PlannerClass(start, goal, env, cfg, draw_callback=...)

    env must provide:
      - env.bounds -> (mn, mx)
      - env.is_segment_free(p, q, n_samples=int) -> bool

    draw_callback(parent_point, new_point) is optional (2-arg compatible).
    """

    def __init__(self, start, goal, env, cfg: RRT3DConnectConfig, draw_callback=None):
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.env = env
        self.cfg = cfg
        self.draw_callback = draw_callback

        # Tree A: from start
        self.Va = np.array([self.start])   # (Na,3)
        self.Pa = [-1]                     # parents indices
        # Tree B: from goal
        self.Vb = np.array([self.goal])    # (Nb,3)
        self.Pb = [-1]

    # -------------------------
    # Helpers
    # -------------------------
    def _sample_point(self):
        mn, mx = self.env.bounds
        if hasattr(self.env, "bounds_xyz"):
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.env.bounds_xyz
            return np.array([
                np.random.uniform(xmin, xmax),
                np.random.uniform(ymin, ymax),
                np.random.uniform(zmin, zmax),
            ], dtype=float)

        return np.random.uniform(mn, mx, size=3)

    @staticmethod
    def _edge_cost(p, q):
        return float(np.linalg.norm(q - p))

    def _nearest_index(self, V, point):
        diff = V - point
        dists = np.linalg.norm(diff, axis=1)
        return int(np.argmin(dists))

    def _steer(self, from_point, to_point):
        direction = to_point - from_point
        dist = np.linalg.norm(direction)
        if dist == 0.0:
            return from_point.copy()
        step = min(self.cfg.step_size, dist)
        return from_point + (direction / dist) * step

    def _add_node(self, V, P, parent_idx, new_p):
        V = np.vstack([V, new_p])
        P.append(parent_idx)
        new_idx = V.shape[0] - 1
        if self.draw_callback is not None:
            self.draw_callback(V[parent_idx], new_p)
        return V, P, new_idx

    # -------------------------
    # Core RRT-Connect operations
    # -------------------------
    def _extend(self, V, P, target):
        """
        Try to extend tree (V,P) one step toward target.
        Returns: (status, V, P, new_idx)
          status in {"trapped", "advanced", "reached"}
        """
        nearest_idx = self._nearest_index(V, target)
        nearest_p = V[nearest_idx]
        new_p = self._steer(nearest_p, target)

        if not self.env.is_segment_free(nearest_p, new_p, n_samples=self.cfg.n_collision_samples):
            return "trapped", V, P, None

        V, P, new_idx = self._add_node(V, P, nearest_idx, new_p)

        if np.linalg.norm(new_p - target) <= self.cfg.goal_threshold:
            return "reached", V, P, new_idx
        return "advanced", V, P, new_idx

    def _connect(self, V, P, target):
        """
        Repeatedly extend toward target until trapped or reached.
        Returns: (status, V, P, last_idx)
          status in {"trapped", "reached"}
        """
        last_idx = None
        while True:
            status, V, P, last_idx = self._extend(V, P, target)
            if status == "trapped":
                return "trapped", V, P, last_idx
            if status == "reached":
                return "reached", V, P, last_idx

    # -------------------------
    # Backtracking / path assembly
    # -------------------------
    def _backtrack(self, V, P, idx):
        path = []
        cur = idx
        while cur != -1:
            path.append(V[cur])
            cur = P[cur]
        path.reverse()
        return np.array(path)

    def _assemble_path(self, a_idx, b_idx):
        """
        Build full path from start-tree node a_idx to goal-tree node b_idx.
        Note: Tree B is rooted at goal, so its backtrack gives goal->...->b_idx;
              we need b_idx->...->goal, so reverse it.
        """
        path_a = self._backtrack(self.Va, self.Pa, a_idx)          # start -> ... -> a_idx
        path_b_goal_to_b = self._backtrack(self.Vb, self.Pb, b_idx)  # goal -> ... -> b_idx
        path_b = path_b_goal_to_b[::-1]                            # b_idx -> ... -> goal

        # Avoid duplicating the connecting point if very close
        if len(path_b) > 0 and np.linalg.norm(path_a[-1] - path_b[0]) < 1e-9:
            path_b = path_b[1:]

        return np.vstack([path_a, path_b])

    # -------------------------
    # Public API
    # -------------------------
    def plan(self):
        """
        Returns:
          path: (N,3) numpy array, or None
        """
        for i in range(self.cfg.max_iter):
            q_rand = self._sample_point()

            # Extend Tree A toward random sample
            status_a, self.Va, self.Pa, a_new = self._extend(self.Va, self.Pa, q_rand)
            if status_a == "trapped":
                # swap trees anyway to keep exploration balanced
                self._swap_trees()
                continue

            # Try to connect Tree B to the new node in Tree A
            q_target = self.Va[a_new]
            status_b, self.Vb, self.Pb, b_last = self._connect(self.Vb, self.Pb, q_target)

            if status_b == "reached":
                # Trees are connected: build full path
                print(f"Connected in {i+1} iterations (RRT-Connect).")

                # Find nearest in B to q_target (b_last is last added, should be close)
                # We want the actual meeting indices: a_new in A, b_last in B.
                full_path = self._assemble_path(a_new, b_last)
                return full_path

            # Alternate roles of the trees to avoid bias
            self._swap_trees()

        print("No path found within iteration limit (RRT-Connect).")
        return None

    def _swap_trees(self):
        # swap A and B trees (including parents) so next iteration grows the other side first
        self.Va, self.Vb = self.Vb, self.Va
        self.Pa, self.Pb = self.Pb, self.Pa

        # also swap the notion of start/goal roots (not strictly necessary for correctness,
        # but it keeps assemble logic consistent if you only assemble on connect right away)
        self.start, self.goal = self.goal, self.start
