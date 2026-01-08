import numpy as np

class BoxObstacle3D:
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax):
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.zmin, self.zmax = zmin, zmax

    def collides_point(self, x, y, z, margin=0.0):
        return((self.xmin - margin) <= x <= (self.xmax + margin) and
               (self.ymin - margin) <= y <= (self.ymax + margin) and
               (self.zmin - margin) <= z <= (self.zmax + margin))
    
class Environment3D:
    def __init__(self, box_list, bounds, robot_radius=0.1):
        """
        box_list: list of BoxObstacle3D
        bounds: [min_xyz, max_xyz] cube region to sample in
        robot_radius: safety inflation for collision checks
        """

        self.boxes = box_list
        self.bounds = bounds
        self.robot_radius = robot_radius

    def is_point_free(self, point):
        x, y, z = point
        for box in self.boxes:
            if box.collides_point(x, y, z, margin=self.robot_radius):
                return False
        return True
        
    def is_segment_free(self, p1, p2, n_samples = 10):
        """
        Check if straight segment p1 -> p2 is collision-free by sampling points along it
        """
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        for alpha in np.linspace(0.0, 1.0, n_samples):
            p = (1 - alpha) * p1 + alpha * p2
            if not self.is_point_free(p):
                return False
        return True