import numpy as np
import heapq
from dataclasses import dataclass

@dataclass
class BITStarConfig:
    # Sampling / batching
    batch_size: int = 400
    max_batches: int = 60

    # Neighborhood for implicit random geometric graph
    neighbor_radius: float = 1.0

    # Goal acceptance
    goal_threshold: float = 2.0

    # Collision checking resolution
    n_collision_samples: int = 10

    # Safety limits to prevent runaway runtimes in dense problems
    max_edge_evals: int = 200000  # total edge pops from the queue

    # If True: keep improving after first solution until budgets exhausted
    anytime: bool = True


class BITStar:
    """
    Practical BIT* (Batch Informed Trees) implementation for 3D Euclidean planning.

    Drop-in interface:
      - __init__(start, goal, env, cfg, draw_callback=None)
      - plan() -> (N,3) path or None

    Requires env:
      - env.bounds -> (mn, mx) where mn, mx are scalars (uniform cube bounds)
      - env.is_segment_free(p, q, n_samples=int) -> bool

    draw_callback (Option A): called ONLY for accepted tree edges:
      draw_callback(parent_point, child_point)
    """

    def __init__(self, start, goal, env, cfg: BITStarConfig, draw_callback=None):
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.env = env
        self.cfg = cfg
        self.draw_callback = draw_callback

        # Tree storage
        self.V = [self.start]       # list of np.array(3,)
        self.parents = [-1]         # parent index
        self.g = [0.0]              # cost-to-come

        # Goal bookkeeping (goal becomes a tree node when connected)
        self.goal_idx = None
        self.c_best = float("inf")  # best solution cost

        # Sample set (points not yet in the tree)
        self.X = []                 # list of np.array(3,)

        # Priority queue of candidate edges:
        # (f_est, g_u, u_idx, x_idx) where x_idx indexes into X
        self.edge_queue = []

        # Cache for basis used in informed sampling
        self._R = None  # rotation matrix aligning x-axis to start->goal

        # Counters
        self.edge_evals = 0

    # -------------------------
    # Basic geometry / costs
    # -------------------------
    @staticmethod
    def _dist(a, b) -> float:
        return float(np.linalg.norm(a - b))

    def _heuristic(self, x) -> float:
        # admissible heuristic in Euclidean space
        return self._dist(x, self.goal)

    def _edge_cost(self, a, b) -> float:
        return self._dist(a, b)

    # -------------------------
    # Sampling
    # -------------------------
    def _uniform_sample(self):
        if hasattr(self.env, "bounds_xyz"):
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.env.bounds_xyz
            return np.array([
                np.random.uniform(xmin, xmax),
                np.random.uniform(ymin, ymax),
                np.random.uniform(zmin, zmax)
            ])
        mn, mx = self.env.bounds
        return np.random.uniform(mn, mx, size=3)

    def _compute_rotation_start_to_goal(self) -> np.ndarray:
        """
        Build an orthonormal basis R such that R[:,0] points along (goal-start).
        """
        d = self.goal - self.start
        norm = np.linalg.norm(d)
        if norm < 1e-12:
            return np.eye(3)

        e1 = d / norm

        # pick an arbitrary vector not parallel to e1
        a = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, e1)) > 0.9:
            a = np.array([0.0, 1.0, 0.0])

        e2 = a - np.dot(a, e1) * e1
        e2 /= np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        e3 /= np.linalg.norm(e3)

        R = np.column_stack([e1, e2, e3])
        return R

    def _sample_unit_ball(self) -> np.ndarray:
        """
        Uniform sample in 3D unit ball.
        """
        v = np.random.normal(size=3)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            v = np.array([1.0, 0.0, 0.0])
            v_norm = 1.0
        v = v / v_norm
        r = np.random.rand() ** (1.0 / 3.0)
        return r * v

    def _informed_sample(self) -> np.ndarray:
        """
        Sample uniformly in the prolate spheroid defined by:
          - foci: start, goal
          - major axis length: c_best
        Only valid when c_best is finite and >= c_min.
        """
        c_min = self._dist(self.start, self.goal)
        if not np.isfinite(self.c_best) or self.c_best <= c_min + 1e-12:
            return self._uniform_sample()

        if self._R is None:
            self._R = self._compute_rotation_start_to_goal()

        # Ellipsoid radii
        a = self.c_best / 2.0
        b = np.sqrt(max(self.c_best**2 - c_min**2, 0.0)) / 2.0
        L = np.diag([a, b, b])

        # Sample in unit ball and transform
        x_ball = self._sample_unit_ball()
        center = (self.start + self.goal) / 2.0
        x_ell = center + self._R @ (L @ x_ball)

        return x_ell

    def _sample_batch(self, n: int):
        """
        Add n new samples into X.
        - Before first solution: uniform
        - After first solution: informed ellipsoid
        """
        for _ in range(n):
            x = self._uniform_sample() if not np.isfinite(self.c_best) else self._informed_sample()
            self.X.append(x)

    # -------------------------
    # Neighbor selection (naive)
    # -------------------------
    def _near_sample_indices(self, v_point):
        """
        Return indices into self.X that are within neighbor_radius of v_point.
        """
        if not self.X:
            return []

        r = self.cfg.neighbor_radius
        # naive scan (fine for moderate batch sizes)
        out = []
        for i, x in enumerate(self.X):
            if np.linalg.norm(x - v_point) <= r:
                out.append(i)
        return out

    # -------------------------
    # Edge queue management
    # -------------------------
    def _push_edges_from_vertex(self, u_idx: int):
        """
        For a tree vertex u, push candidate edges (u -> x) for nearby samples x,
        ordered by an A*-like key f = g(u) + c(u,x) + h(x).
        Apply pruning by current c_best.
        """
        u = self.V[u_idx]
        g_u = self.g[u_idx]

        for x_idx in self._near_sample_indices(u):
            x = self.X[x_idx]

            c = self._edge_cost(u, x)
            if c > self.cfg.neighbor_radius:
                continue

            f_est = g_u + c + self._heuristic(x)

            # If we already have a solution, prune anything that cannot beat it
            if np.isfinite(self.c_best) and f_est >= self.c_best:
                continue

            heapq.heappush(self.edge_queue, (f_est, g_u, u_idx, x_idx))

    def _rebuild_edge_queue(self):
        """
        Build queue from all tree vertices to nearby samples.
        """
        self.edge_queue = []
        for u_idx in range(len(self.V)):
            self._push_edges_from_vertex(u_idx)

    # -------------------------
    # Goal connection attempt
    # -------------------------
    def _try_connect_goal_from(self, u_idx: int):
        """
        Try to connect the goal from a given tree node.
        If successful, add goal as a tree node and update c_best.
        """
        u = self.V[u_idx]
        if self._dist(u, self.goal) > self.cfg.goal_threshold:
            return

        if not self.env.is_segment_free(u, self.goal, n_samples=self.cfg.n_collision_samples):
            return

        new_cost = self.g[u_idx] + self._edge_cost(u, self.goal)
        if new_cost >= self.c_best:
            return

        # Add goal as a node
        self.V.append(self.goal.copy())
        self.parents.append(u_idx)
        self.g.append(new_cost)
        self.goal_idx = len(self.V) - 1
        self.c_best = new_cost

        if self.draw_callback is not None:
            self.draw_callback(u, self.goal)

    # -------------------------
    # Path reconstruction
    # -------------------------
    def _backtrack_path(self):
        if self.goal_idx is None:
            return None

        idxs = []
        cur = self.goal_idx
        while cur != -1:
            idxs.append(cur)
            cur = self.parents[cur]
        idxs.reverse()

        return np.vstack([self.V[i] for i in idxs])

    # -------------------------
    # Public API
    # -------------------------
    def plan(self):
        # Initial batch (uniform)
        self._sample_batch(self.cfg.batch_size)
        # Also allow goal to be discovered via neighbor edges sooner
        # (Goal connection is handled via _try_connect_goal_from)

        self._rebuild_edge_queue()

        batches_done = 1

        print(
        f"[BIT*] Working on batch {batches_done}/{self.cfg.max_batches} | "
        f"|V|={len(self.V)} |X|={len(self.X)} |queue|={len(self.edge_queue)}")

        # Main loop: keep processing promising edges; add batches if queue empties
        while self.edge_evals < self.cfg.max_edge_evals and batches_done <= self.cfg.max_batches:
            # If no candidate edges left, add another batch and rebuild connectivity
            if not self.edge_queue:
                # If we already have a solution and we're not doing anytime improvement: stop
                if self.goal_idx is not None and not self.cfg.anytime:
                    break

                self._sample_batch(self.cfg.batch_size)
                batches_done += 1
                self._rebuild_edge_queue()
                continue

            f_est, g_u, u_idx, x_idx = heapq.heappop(self.edge_queue)
            self.edge_evals += 1

            # If this edge can't beat current best, prune it
            if np.isfinite(self.c_best) and f_est >= self.c_best:
                continue

            # x_idx refers into X; it may be stale if X changed in the meantime
            if x_idx < 0 or x_idx >= len(self.X):
                continue

            x = self.X[x_idx]
            u = self.V[u_idx]

            # Lazy collision check for (u -> x)
            if np.linalg.norm(u - x) > self.cfg.neighbor_radius + 1e-9:
                continue
            if not self.env.is_segment_free(u, x, n_samples=self.cfg.n_collision_samples):
                continue

            # Accept: add x into the tree
            new_g = self.g[u_idx] + self._edge_cost(u, x)

            # Add node
            self.V.append(x.copy())
            self.parents.append(u_idx)
            self.g.append(new_g)
            v_idx = len(self.V) - 1

            if self.draw_callback is not None:
                d = np.linalg.norm(u - x)
                if d > self.cfg.neighbor_radius + 1e-6:
                    print(f"[WARN] accepted edge longer than radius: d={d:.3f}  r={self.cfg.neighbor_radius:.3f}")
                self.draw_callback(u, x)

            # Remove sample x from X by swap-pop, to keep indices compact.
            # IMPORTANT: this invalidates some queued x_idx values, which we tolerate by staleness checks above.
            last = self.X[-1]
            self.X[x_idx] = last
            self.X.pop()

            # Try to connect to goal if we're close enough
            self._try_connect_goal_from(v_idx)

            # Add new candidate edges from the new vertex to nearby samples
            self._push_edges_from_vertex(v_idx)

            # If we found a solution and are not doing anytime improvement: return immediately
            if self.goal_idx is not None and not self.cfg.anytime:
                print(f"Goal reached (BIT*) with cost {self.c_best:.3f} after {self.edge_evals} edge evals, {batches_done} batches.")
                return self._backtrack_path()

            # If we found a first solution, informed sampling will kick in automatically for future batches.
            # We keep going if anytime=True, to attempt improvements.

        if self.goal_idx is not None:
            print(f"BIT* finished with best cost {self.c_best:.3f} after {self.edge_evals} edge evals, {batches_done} batches.")
            return self._backtrack_path()

        print(f"BIT* did not find a path after {self.edge_evals} edge evals, {batches_done} batches.")
        return None
