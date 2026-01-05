import cvxpy as cp
import numpy as np

# allocation matrix to convert the MPC output to RPM for drone
kf = 3.16e-10
km = 7.94e-12
L  = 0.0397
m  = 0.027

ALLOC_A = np.array([
    [ kf,   kf,   kf,   kf],        # thrust
    [ 0.,  -kf*L, 0.,   kf*L],      # roll
    [-kf*L, 0.,   kf*L, 0.],        # pitch
    [-km,   km,  -km,   km]         # yaw
])

class MPCPathController:
    def __init__(self, quadcopter, N):
        # quadcopter : dynamics
        # N : horizon value

        self.quadcopter = quadcopter
        self.N = N
        self.nx = 12
        self.nu = 4

        # params, updated each tick
        self.x0   = cp.Parameter(self.nx)
        self.xref = cp.Parameter((self.nx, N + 1))

        # decision variables
        x = cp.Variable((self.nx, N + 1))   # state
        u = cp.Variable((self.nu, N))       # output

        # init cost and constraints
        cost = 0
        constraints = []

        # initial condition
        constraints += [x[:, 0] == self.x0]

        # angle bounds constant
        ang_lim = np.deg2rad(30)

        # get quad copter dynamic matrices
        A = quadcopter.A
        B = quadcopter.B

        for k in range(N):
            # tracking + control effort
            cost += cp.quad_form(x[:, k] - self.xref[:, k], quadcopter.Q)
            cost += cp.quad_form(u[:, k], quadcopter.R)

            # dynamics
            constraints += [x[:, k+1] == A @ x[:, k] + B @ u[:, k]]

            # bounds
            phi   = x[3, k]
            theta = x[4, k]
            constraints += [phi <= ang_lim]
            constraints += [phi >= -ang_lim]
            constraints += [theta <= ang_lim]
            constraints += [theta >= -ang_lim]

        # Terminal cost
        xN = x[:, N]
        x_ref_N = self.xref[:, N-1]   # last reference point
        cost += cp.quad_form(xN - x_ref_N, quadcopter.P)

        # Terminal set
        eps_pos = 0.75      # m
        constraints += [cp.abs(xN[0:3] - x_ref_N[0:3]) <= eps_pos]

        self.x_var = x
        self.u_var = u
        self.problem = cp.Problem(cp.Minimize(cost), constraints)

        # keep last solution around for warm-start
        self._last_omega = None

    def solve(self, x_init, x_target):
        """
        x_init: shape (12,)
        x_target: shape (12, N+1)  (or bigger; but you should pass exactly N+1 here)
        returns: omega (4,) motor angular speeds (sqrt(omega^2))
        """
        self.x0.value = x_init
        self.xref.value = x_target

        # solve (warm_start for quickness)
        self.problem.solve(solver=cp.OSQP, warm_start=True)

        u0 = self.u_var[:, 0].value

        delta_T, tau_x, tau_y, tau_z = u0
        T_total = m * 9.81 + delta_T
        U = np.array([T_total, tau_x, tau_y, tau_z])

        omega_sq = np.linalg.solve(ALLOC_A, U)
        omega_sq = np.clip(omega_sq, 0.0, None)
        omega = np.sqrt(omega_sq)

        self._last_omega = omega
        return omega