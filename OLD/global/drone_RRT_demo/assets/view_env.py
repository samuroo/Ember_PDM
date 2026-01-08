import os
import time
import pybullet as p
import pybullet_data

URDF_NAME = "DunderMifflin_Scranton.urdf"

def draw_aabbs(body_id: int):
    num_joints = p.getNumJoints(body_id)
    for link_idx in range(-1, num_joints):
        aabb_min, aabb_max = p.getAABB(body_id, link_idx)

        corners = [
            [aabb_min[0], aabb_min[1], aabb_min[2]],
            [aabb_max[0], aabb_min[1], aabb_min[2]],
            [aabb_max[0], aabb_max[1], aabb_min[2]],
            [aabb_min[0], aabb_max[1], aabb_min[2]],
            [aabb_min[0], aabb_min[1], aabb_max[2]],
            [aabb_max[0], aabb_min[1], aabb_max[2]],
            [aabb_max[0], aabb_max[1], aabb_max[2]],
            [aabb_min[0], aabb_max[1], aabb_max[2]],
        ]
        edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)
        ]
        for i, j in edges:
            p.addUserDebugLine(corners[i], corners[j], [1, 0, 0], 1.5, 0)

def main():
    print("CWD:", os.getcwd())
    urdf_path = os.path.abspath(URDF_NAME)
    print("URDF:", urdf_path)
    print("URDF exists?", os.path.exists(urdf_path))

    cid = p.connect(p.GUI)
    print("connect() returned:", cid)
    if cid < 0:
        raise RuntimeError("PyBullet GUI connection failed (cid < 0). Likely headless/WSL/no display.")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load plane (PyBullet doesn't spawn a floor automatically unless you load one)
    plane_id = p.loadURDF("plane.urdf")
    print("Loaded plane id:", plane_id)

    # Load environment URDF
    env_id = p.loadURDF(urdf_path, useFixedBase=True)
    print("Loaded env id:", env_id)

    p.resetDebugVisualizerCamera(
        cameraDistance=55,
        cameraYaw=45,
        cameraPitch=-60,
        cameraTargetPosition=[22.5, -12.5, 1.0]
    )

    draw_aabbs(env_id)

    print("GUI running. Close the PyBullet window to exit.")
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1/240)

if __name__ == "__main__":
    main()
