import pybullet as p

def spawn_boxes_from_aabbs(boxes, client_id=None):
    """
    boxes: iterable of (xmin, xmax, ymin, ymax, zmin, zmax)
    returns: list of body_ids
    """
    body_ids = []
    for i, (xmin, xmax, ymin, ymax, zmin, zmax) in enumerate(boxes):
        # Half extents (box size / 2)
        half_extents = [
            (xmax - xmin) / 2.0,
            (ymax - ymin) / 2.0,
            (zmax - zmin) / 2.0,
        ]

        # Center of the box
        center = [
            (xmin + xmax) / 2.0,
            (ymin + ymax) / 2.0,
            (zmin + zmax) / 2.0,
        ]

        col_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=client_id,
        )

        vis_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.7, 0.7, 0.7, 1.0],  # light grey
            physicsClientId=client_id,
        )

        body_id = p.createMultiBody(
            baseMass=0.0,  # static
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=center,
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=client_id,
        )

        body_ids.append(body_id)

    return body_ids