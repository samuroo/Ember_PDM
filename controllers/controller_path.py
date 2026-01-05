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

        # weights
        Q = np.diag([
            50*2, 50*2, 50*2,   # x, y, z
            10, 10, 0.0,        # φ, θ, ψ
            3, 3, 3,            # vx, vy, vz
            1.5, 1.5, 0.0       # p, q, r
        ])
        R = np.diag([0.1, 5, 5, 0.0])

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
            cost += cp.quad_form(x[:, k] - self.xref[:, k], Q)
            cost += cp.quad_form(u[:, k], R)

            # dynamics
            constraints += [x[:, k+1] == A @ x[:, k] + B @ u[:, k]]

            # bounds
            phi   = x[3, k]
            theta = x[4, k]
            constraints += [phi <= ang_lim]
            constraints += [phi >= -ang_lim]
            constraints += [theta <= ang_lim]
            constraints += [theta >= -ang_lim]

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
        # if u0 is None:
        #     # Fallback: if solver fails, return last command or hover
        #     if self._last_omega is not None:
        #         return self._last_omega
        #     return np.zeros(4)

        delta_T, tau_x, tau_y, tau_z = u0
        T_total = m * 9.81 + delta_T
        U = np.array([T_total, tau_x, tau_y, tau_z])

        omega_sq = np.linalg.solve(ALLOC_A, U)
        omega_sq = np.clip(omega_sq, 0.0, None)
        omega = np.sqrt(omega_sq)

        self._last_omega = omega
        return omega


# # allocation matrix to convert the MPC output to RPM for drone
# kf=3.16e-10
# km=7.94e-12
# L=0.0397
# m = 0.027
# A= np.array([
#         [ kf,  kf,  kf,  kf],          # thrust
#         [ 0., -kf*L, 0.,  kf*L],       # roll
#         [-kf*L, 0.,  kf*L, 0.],        # pitch
#         [-km,   km, -km,  km]          # yaw
#     ])

# def mpc_control_path(quadcopter, N, x_init, x_target):
#     """
#     Inputs:
#         quadcopter : dynamical system defined
#         N : horizon
#         x_init : current state of quad
#         x_target : target trajectory

#     return : list of drone thrust values (rotiations per minute)
#     """
#     cost = 0.
#     constraints = []
    
#     x = cp.Variable((12, N + 1))
#     u = cp.Variable((4, N))

#     Q = np.diag([
#         50*2, 50*2, 50*2,   # x, y, z
#         10, 10, 0.0,        # φ, θ, ψ
#         3, 3, 3,            # vx, vy, vz
#         1.5, 1.5, 0.0       # p, q, r
#     ])
#     # input weight matrix
#     R = np.diag([0.1, 5, 5, 0.0])

#     for k in range(N):
#         # extract the next waypoint in 20 steps
#         x_ref_k = x_target[:,k]

#         # main cost function
#         cost += cp.quad_form(x[:,k] - x_ref_k, Q)
#         cost += cp.quad_form(u[:,k], R)

#         # dynamical model
#         constraints += [x[:, k+1] == quadcopter.A @ x[:, k] + quadcopter.B @ u[:, k]]
        
#         # for later use in constraints
#         phi, theta = x[3, k], x[4, k]
#         vx, vy, vz = x[6, k], x[7, k], x[8, k]
#         p, q, r    = x[9, k], x[10, k]  , x[11, k]

#         yaw_roll_ang_const = 30
#         constraints += [phi <= np.deg2rad(yaw_roll_ang_const)]
#         constraints += [phi >= -np.deg2rad(yaw_roll_ang_const)]
#         constraints += [theta >= -np.deg2rad(yaw_roll_ang_const)]
#         constraints += [theta <= np.deg2rad(yaw_roll_ang_const)]
        
#     # init contstraint
#     constraints += [x[:, 0] == x_init]
    
#     # solves the MPC problem
#     problem = cp.Problem(cp.Minimize(cost), constraints)
#     problem.solve(solver=cp.OSQP)

#     # unpack the MPC solution
#     delta_T, tau_x, tau_y, tau_z = u[:, 0].value
#     T_total = m * 9.81 + delta_T
#     U = np.array([T_total, tau_x, tau_y, tau_z])

#     # linearly solve the thurst (i.e RPM) for each motor
#     omega_sq = np.linalg.solve(A, U)
#     omega_sq = np.clip(omega_sq, 0.0, None)
#     omega = np.sqrt(omega_sq)
    
#     return omega