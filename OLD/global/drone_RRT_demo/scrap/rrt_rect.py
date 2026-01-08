import random
import math
import matplotlib.pyplot as plt

show_animation = True

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.path_x = []
        self.path_y = []

class RRT:
    """
    RRT supporting rectangular obstacles

    obstacle_list = [
        (xmin, ymin, xmax, ymax),
        ...
    ]
    """

    def __init__(
            self, start, goal, obstacle_list, rand_area, expand_dis=0.5, path_resolution=0.1, goal_sample_rate=5,
            max_iter=1000, robot_radius=0.1
    ):
        self.start = Node(start[0], start[1])
        self.end = Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.path_resolution = path_resolution
        self.obstacle_list = obstacle_list
        self.robot_radius = robot_radius
        self.node_list = [self.start]

    def planning(self, animation=True):

        for i in range(self.max_iter):
            rnd = self.get_random_node()

            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd, self.expand_dis)

            if self.check_collision(new_node):
                self.node_list.append(new_node)

            if animation and i % 20 == 0:
                self.draw_graph(rnd)

                if self.calc_dist_to_goal(self.node_list[-1]) <= self.expand_dis:
                    final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                    if self.check_collision(final_node):
                        return self.generate_final_course(len(self.node_list) - 1)

        return None  # failed

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dist = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)

        move_dist = min(extend_length, dist)
        steps = int(move_dist / self.path_resolution)

        new_node.path_x = []
        new_node.path_y = []

        for _ in range(steps):
            from_node.x += self.path_resolution * math.cos(angle)
            from_node.y += self.path_resolution * math.sin(angle)
            new_node.path_x.append(from_node.x)
            new_node.path_y.append(from_node.y)

        new_node.x = from_node.x
        new_node.y = from_node.y
        new_node.parent = from_node

        return new_node
    
    def check_collision(self, node):
        """Return True if node is collision-free."""
        if node is None:
            return False

        for (xmin, ymin, xmax, ymax) in self.obstacle_list:
            for x, y in zip(node.path_x, node.path_y):
                if (xmin - self.robot_radius) <= x <= (xmax + self.robot_radius) and \
                   (ymin - self.robot_radius) <= y <= (ymax + self.robot_radius):
                    return False
        return True

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path[::-1]

    def calc_dist_to_goal(self, node):
        dx = node.x - self.end.x
        dy = node.y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            return Node(random.uniform(self.min_rand, self.max_rand),
                        random.uniform(self.min_rand, self.max_rand))
        return Node(self.end.x, self.end.y)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        return dlist.index(min(dlist))

    def draw_graph(self, rnd=None):
        plt.clf()

        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")

        # RRT tree
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        # Draw rectangles
        ax = plt.gca()
        for (xmin, ymin, xmax, ymax) in self.obstacle_list:
            rect = plt.Rectangle((xmin, ymin),
                                 xmax - xmin,
                                 ymax - ymin,
                                 fill=False, edgecolor="red")
            ax.add_patch(rect)

        # Start/Goal
        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")

        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.01)