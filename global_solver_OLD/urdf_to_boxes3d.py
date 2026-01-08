import pybullet as p
import pybullet_data
import numpy as np


def load_env_and_extract_boxes3d(urdf_path):
    """
    Load a URDF environment into pybullet and extract
    axis aligned 3D boxes for each box collision shape.

    Returns:
        boxes: list of (xmin, ymin, xmax, ymax)
        bounds: [min_xy, max_xy] for RRT sampling
    """
    physics_client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    env_id = p.loadURDF(urdf_path, useFixedBase=True)

    num_joints = p.getNumJoints(env_id)
    link_indices = [-1] + list(range(num_joints))

    boxes = []

    for link_idx in link_indices:

        shapes = p.getCollisionShapeData(env_id, link_idx)

        # World pose of the link
        if link_idx == -1:
            pos, orn = p.getBasePositionAndOrientation(env_id)
        else:
            ls = p.getLinkState(env_id, link_idx)
            pos, orn = ls[0], ls[1]

        cx, cy, cz = pos

        # Process each shape in this link
        for shape in shapes:
            geom_type = shape[2]
            dims = shape[3]  # half extents

            if geom_type != p.GEOM_BOX:
                continue

            hx, hy, hz = [d * 0.5 for d in dims]

            xmin = cx - hx
            xmax = cx + hx
            ymin = cy - hy
            ymax = cy + hy
            zmin = cz - hz
            zmax = cz + hz

            boxes.append((xmin, xmax, ymin, ymax, zmin, zmax))

    p.disconnect(physics_client)

    if not boxes:
        raise RuntimeError("No box collision shapes found in URDF")

    # simple global bounding cube to sampe in
    all_x = [b[0] for b in boxes] + [b[2] for b in boxes]
    all_y = [b[1] for b in boxes] + [b[3] for b in boxes]
    all_z = [b[4] for b in boxes] + [b[5] for b in boxes]

    min_xy = min(min(all_x), min(all_y), min(all_z)) - 1.0
    max_xy = max(max(all_x), max(all_y), max(all_z)) + 1.0

    bounds = [min_xy, max_xy]

    return boxes, bounds


if __name__ == "__main__":
    boxes, bounds = load_env_and_extract_boxes3d("assets/hallway_env1.urdf")
    print("3D boxes (xmin, xmax, ymin, ymax, zmin, zmax):")
    for i, b in enumerate(boxes):
        print(f"  box {i}: {b}")
    print("Bounds:", bounds)