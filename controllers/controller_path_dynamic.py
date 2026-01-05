import cvxpy as cp
import numpy as np

# allocation matrix to convert the MPC output to RPM for drone
kf=3.16e-10
km=7.94e-12
L=0.0397
m = 0.027
A= np.array([
        [ kf,  kf,  kf,  kf],          # thrust
        [ 0., -kf*L, 0.,  kf*L],       # roll
        [-kf*L, 0.,  kf*L, 0.],        # pitch
        [-km,   km, -km,  km]          # yaw
    ])

def mpc_control_path(quadcopter, N, x_init, x_target, obj_points):
    """
    Inputs:
        quadcopter : dynamical system defined
        N : horizon
        x_init : current state of quad
        x_target : target postion (could be traj.)

    return : list of drone thrust values (rotiations per minute)
    """
    cost = 0.
    constraints = []
    
    x = cp.Variable((12, N + 1))
    u = cp.Variable((4, N))

    # cost weight matrix
    # Q = np.diag([
    #     50*2, 50*2, 50*4, # x, y, z
    #     10, 10, 20, # φ, θ, ψ
    #     3, 3, 5, # vx, vy, vz
    #     1, 1, 1 # p, q, r
    # ])
    # # input weight matrix
    # R = np.diag([0.01, 0.01, 0.01, 0.01])

    Q = np.diag([
        50*2, 50*2, 50*2, # x, y, z
        10, 10, 0.0, # φ, θ, ψ
        3, 3, 3, # vx, vy, vz
        1.5, 1.5, 0.0 # p, q, r
    ])
    # input weight matrix
    R = np.diag([0.1, 5, 5, 0.0])
    p_close = []
    s_list = []
    norm_list = []
    num_obj = 0
    filter_dist = 1
    unique_pairs = np.unique(obj_points[:, :2], axis=0) 
    for body_id, link_id in unique_pairs:

        current_points = obj_points[(obj_points[:, 0] == body_id) & (obj_points[:, 1] == link_id)][:, 2:5]

        vecs = x_init[:3]-current_points
        closest_idx = np.argmin(np.linalg.norm(vecs, axis=1))

        close_vec = vecs[closest_idx]

        if np.linalg.norm(close_vec) < filter_dist:
            normal = close_vec/np.linalg.norm(close_vec)
            p_close.append(current_points[closest_idx])
            norm_list.append(normal)
            num_obj += 1

    norm_list = np.array(norm_list)
    if num_obj > 0:
        s_list = cp.Variable((num_obj, N))
    

    for k in range(N):
        # extract the next waypoint in 20 steps
        x_ref_k = x_target[:,k]

        # main cost function
        cost += cp.quad_form(x[:,k] - x_ref_k, Q)
        cost += cp.quad_form(u[:,k], R)

        # dynamical model
        constraints += [x[:, k+1] == quadcopter.A @ x[:, k] + quadcopter.B @ u[:, k]]
        
        # for later use in constraints
        phi, theta = x[3, k], x[4, k]
        vx, vy, vz = x[6, k], x[7, k], x[8, k]
        p, q, r    = x[9, k], x[10, k]  , x[11, k]

        yaw_roll_ang_const = 30
        constraints += [phi <= np.deg2rad(yaw_roll_ang_const)]
        constraints += [phi >= -np.deg2rad(yaw_roll_ang_const)]
        constraints += [theta >= -np.deg2rad(yaw_roll_ang_const)]
        constraints += [theta <= np.deg2rad(yaw_roll_ang_const)]

        # Half space constraints
        safety_dist = 0.3
        soft_dist = 0.8

        if num_obj != 0:
            for i in range(num_obj):
                constraints += [norm_list[i, :].T @ x[:3, k+1] >= norm_list[i, :].T @ p_close[i]]

                
                # slack_weight = 1e2
                # constraints += [norm_list[i, :].T @ x[:3, k+1] >= norm_list[i, :].T @ (p_close[i] + soft_dist*norm_list[i, :]) - s_list[i, k]]  # soft
                # constraints += [s_list[i, k] >= 0]
                # cost += slack_weight * s_list[i, k]
        
        

    constraints += [x[:, 0] == x_init]
    
    # Solves the problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP)

    # unpack the MPC solution
    delta_T, tau_x, tau_y, tau_z = u[:, 0].value
    T_total = m * 9.81 + delta_T
    U = np.array([T_total, tau_x, tau_y, tau_z])

    # linearly solve the thurst (i.e RPM) for each motor
    omega_sq = np.linalg.solve(A, U)
    omega_sq = np.clip(omega_sq, 0.0, None)
    omega = np.sqrt(omega_sq)
    
    return omega