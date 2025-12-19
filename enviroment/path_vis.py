import pybullet as p
import numpy as np

# global list for MPC visualization spheres
horizon_sphere_ids = []

# draw a path in space for visual purposes
def draw_path(path):  
    for i in range(len(path)-1):
        p.addUserDebugLine(path[i], path[i+1], [0.1,0,0], lineWidth=2)

# add and remove red spheres for MPC visulaization purposes
def update_horizon_visualization(x_ref_traj, base_radius=0.025, end_radius=0.005):
    global horizon_sphere_ids
    # 1. Remove old spheres
    for sid in horizon_sphere_ids:
        try:
            p.removeBody(sid)
        except:
            pass
    horizon_sphere_ids = []

    # Number of points in the trajectory
    num_markers = max(2,  x_ref_traj.shape[1] // 3)
    # 2. Compute radii decreasing linearly
    radii = np.linspace(base_radius, end_radius, num_markers)

    # 3. Draw new spheres
    indices = np.linspace(0, x_ref_traj.shape[1]-1, num_markers, dtype=int)

    # 4. Radii decreasing along the sampled points
    radii = np.linspace(base_radius, end_radius, num_markers)

    # 5. Draw spheres at the sampled positions
    for r, idx in zip(radii, indices):
        # Assume x_ref_traj[0:3, idx] are x, y, z
        pos = np.array([x_ref_traj[0, idx],
                        x_ref_traj[1, idx],
                        x_ref_traj[2, idx]])

        visual_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=r,
            rgbaColor=[1, 0, 0, 1]   # red
        )

        body_id = p.createMultiBody(
            baseVisualShapeIndex=visual_id,
            basePosition=pos
        )

        horizon_sphere_ids.append(body_id)