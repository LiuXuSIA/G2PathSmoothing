"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import math
import numpy as np
import matplotlib.pyplot as plt
import kCurves

show_animation = True


class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """
        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                ax1.scatter(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), marker = '*', c='g', s=50)
                if len(closed_set.keys()) % 1 == 0:
                    plt.pause(0.1)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


fig = plt.figure(figsize=[18, 8.3])
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
plt.axis([-20, 220,-20,220])

ax1.set_title('generated path by A*', fontsize = 15, pad = 10, c='yellow')
ax2.set_title('smoothed path', fontsize = 15, pad = 10, c='yellow')

ax1.spines['bottom'].set_linewidth(1.2)
ax1.spines['left'].set_linewidth(1.2)
ax1.spines['top'].set_linewidth(1.2)
ax1.spines['right'].set_linewidth(1.2)

ax2.spines['bottom'].set_linewidth(1.2)
ax2.spines['left'].set_linewidth(1.2)
ax2.spines['top'].set_linewidth(1.2)
ax2.spines['right'].set_linewidth(1.2)

ax2.spines['bottom'].set_linewidth(1.2)
ax2.spines['left'].set_linewidth(1.2)
ax2.spines['top'].set_linewidth(1.2)
ax2.spines['right'].set_linewidth(1.2)

ax1.spines['bottom'].set_color('grey')
ax1.spines['left'].set_color('grey')
ax1.spines['top'].set_color('grey')
ax1.spines['right'].set_color('grey')

ax2.spines['bottom'].set_color('grey')
ax2.spines['left'].set_color('grey')
ax2.spines['top'].set_color('grey')
ax2.spines['right'].set_color('grey')

fig.patch.set_facecolor('black')
ax1.patch.set_facecolor('black')
ax2.patch.set_facecolor('black')


def main():
    print(__file__ + " start!!")
    
    planning_area = [[-20,220], [-20,220]]

    # start and goal position
    sx = 0.0
    sy = 0.0
    gx = 170.0
    gy = 55.0
    grid_size = 9.0  # [m]
    robot_radius = 5.0  # [m]

    # set obstacle positions
    ox, oy = [], []
    for i in range(0, 10):
        for j in range(30, 200):
            ox.append(i)
            oy.append(j)

    for i in range(40, 50):
        for j in range(0, 130):
            ox.append(i)
            oy.append(j)

    for i in range(40, 50):
        for j in range(150, 200):
            ox.append(i)
            oy.append(j)

    for i in range(90, 120):
        for j in range(0, 30):
            ox.append(i)
            oy.append(j)

    for i in range(90, 120):
        for j in range(50, 140):
            ox.append(i)
            oy.append(j)

    for i in range(50, 160):
        for j in range(160, 170):
            ox.append(i)
            oy.append(j)


    for i in range(120, 180):
        for j in range(80, 90):
            ox.append(i)
            oy.append(j)

    for i in range(180, 190):
        for j in range(0, 60):
            ox.append(i)
            oy.append(j)

    for i in range(190, 200):
        for j in range(120, 200):
            ox.append(i)
            oy.append(j)

    if show_animation:  # pragma: no cover
        ax1.scatter(ox, oy, c='grey', s=3)
        ax2.scatter(ox, oy, c='grey', s=3)

        ax1.scatter(sx, sy, marker = '^', s=100, c='red')
        ax1.scatter(gx, gy, marker = 'v', s=100, c='deeppink')
        plt.axis("equal")

        ax2.scatter(sx, sy, marker = '^', s=100, c='red')
        ax2.scatter(gx, gy, marker = 'v', s=100, c='purple')
        # plt.grid(True)
        plt.axis("equal")
        plt.pause(15)

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)
    path_ctrls = kCurves.kCurveOpen(np.c_[rx, ry])
    path_smoothed = kCurves.bezier_curve(path_ctrls, n_points = 500)

    if show_animation:  # pragma: no cover
        ax1.scatter(rx, ry, color="orange", s=70, label='path point')
        ax1.plot(rx, ry, c="white", linewidth=4, label='path')
        plt.axis("equal")
        plt.pause(0.001)
        ax2.plot(path_smoothed[:,0], path_smoothed[:,1], "b", linewidth=4,label='trajectory')
        ax2.scatter(rx, ry, color="orange", s=70, label='path point')
        plt.axis("equal")
        plt.pause(0.001)
        # plt.legend()
        plt.show()

if __name__ == '__main__':
    main()
