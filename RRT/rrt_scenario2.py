"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

author: AtsushiSakai(@Atsushi_twi)

"""

import math
import random
import sys
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import kCurves
from sklearn.neighbors import KDTree

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../Analytical-CuBezier/")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../Spline/")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../POSQ/")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../Dubins/")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../Reeds-Shepp/")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../Clothoids/")

import cubic_bezier
import cubic_spline
import bspline
import posq
import dubins
import reeds_shepp
import clothoid

show_animation = True


class RRT:
    """
    Class for RRT planning
    """
    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    class AreaBounds:

        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])


    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 robot_radius,
                 rand_area,
                 expand_dis=4.0,
                 path_resolution=0.1,
                 goal_sample_rate=5,
                 max_iter=1000,
                 play_area=None
                 ):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        play_area:stay inside this area [xmin,xmax,ymin,ymax]

        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.robot_radius = robot_radius
        self.node_list = []

    def planning(self, animation=True):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_if_outside_play_area(new_node, self.play_area) and \
               self.check_collision(new_node, self.obstacle_list, self.robot_radius):
                self.node_list.append(new_node)

            # if animation:
            #     self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x,
                                      self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end,
                                        self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list, self.robot_radius):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None, None  # cannot find path

    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind):
        points = [[self.end.x, self.end.y]]
        path = np.c_[self.end.path_x, self.end.path_y].tolist()
        node = self.node_list[goal_ind]
        while node.parent is not None:
            points.append([node.x, node.y])
            path.extend(np.c_[node.path_x, node.path_y].tolist())
            node = node.parent
        points.append([node.x, node.y])

        return points, path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def draw_graph(self, rnd=None):
        plt.clf()
        # if rnd is not None:
        #     plt.plot(rnd.x, rnd.y, "^k")
        # for node in self.node_list:
        #     if node.parent:
        #         plt.plot(node.path_x, node.path_y, "-g")
        
        plt.scatter(self.obstacle_list[:,0], self.obstacle_list[:,1], c='grey', s=2)
        # plt.plot(self.start.x, self.start.y, "xr")
        # plt.plot(self.end.x, self.end.y, "xr")
        # plt.axis("equal")
        plt.xlim(-20, 220) 
        plt.ylim(-20, 220)
        # plt.axis([0, 30, 0, 30])
        # plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def plot_circle(x, y, size, color="black"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_if_outside_play_area(node, play_area):

        if play_area is None:
            return True  # no play_area was defined, every pos should be ok

        if node.x < play_area.xmin or node.x > play_area.xmax or \
           node.y < play_area.ymin or node.y > play_area.ymax:
            return False  # outside - bad
        else:
            return True  # inside - ok

    @staticmethod
    def check_collision(node, obstacleList, robot_radius):
        # print('1')
        if node is None:
            return False
        
        # print(node.x, node.y)
        # print(node.path_x, node.path_y)
        # print(obstacleList)
        tree = KDTree(obstacleList)
        dis, _ = tree.query(np.c_[node.path_x, node.path_y], k=1, return_distance=True)
        # print(dis)
        min_dis = np.min(dis[:,0])
        # print(min_dis)

        if min_dis <= robot_radius:
            return False  # collision

        return True  # safe

    @staticmethod
    def acute_angle_check(node):
        if not node.parent.parent:
            return False
        else:
            angle_cos = np.dot([node.x-node.parent.x, node.y-node.parent.y], \
                           [node.parent.parent.x-node.parent.x, node.parent.parent.y-node.parent.y]) / \
                           (np.linalg.norm([node.x-node.parent.x, node.y-node.parent.y], 2) * \
                            np.linalg.norm([node.parent.parent.x-node.parent.x, node.parent.parent.y-node.parent.y])) 

        return True  if angle_cos >=0 else False

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

def consistency_error(trajectory, path):
    trajectory = np.asarray(trajectory)
    path = np.asarray(path)
    tree = KDTree(path)
    dis, _ = tree.query(trajectory, k=1, return_distance=True)
    average_d = np.mean(dis[:,0])
    maximum_d = np.max(dis[:,0])
    return average_d, maximum_d

def path_length(path):
    length = 0
    for i in range(len(path)-1):
        length += np.sqrt((path[i+1][1] - path[i][1])**2 + (path[i+1][0] - path[i][0])**2)

    return length


def main():
    print("start " + __file__)
    now = datetime.now().strftime(r'%Y%m%d%H%M%S')
    rootPath = 'F:\\PC2Win10\\Study\\PHD\\Research\\paper_writting\\RAL2023\\'
    
    o_x, o_y = [], []
    for i in range(0, 10):
        for j in range(30, 200):
            o_x.append(i)
            o_y.append(j)

    for i in range(40, 50):
        for j in range(0, 130):
            o_x.append(i)
            o_y.append(j)

    for i in range(40, 50):
        for j in range(150, 200):
            o_x.append(i)
            o_y.append(j)

    for i in range(90, 120):
        for j in range(0, 30):
            o_x.append(i)
            o_y.append(j)

    for i in range(90, 120):
        for j in range(50, 140):
            o_x.append(i)
            o_y.append(j)

    for i in range(50, 160):
        for j in range(160, 170):
            o_x.append(i)
            o_y.append(j)


    for i in range(120, 180):
        for j in range(80, 90):
            o_x.append(i)
            o_y.append(j)

    for i in range(180, 190):
        for j in range(0, 60):
            o_x.append(i)
            o_y.append(j)

    for i in range(190, 200):
        for j in range(120, 200):
            o_x.append(i)
            o_y.append(j)


    obstacleList = np.c_[o_x, o_y]
    
    sx = 0.0
    sy = 0.0
    gx = 167.0
    gy = 50.0
    # Set Initial parameters
    robotRadius = 5  # [m]
    expandDis = 20 #6
    pathResolution = 0.5

    rrt = RRT(
        start=[sx, sy],
        goal=[gx, gy],
        rand_area=[0,200],
        obstacle_list=obstacleList,
        robot_radius = robotRadius,  # [m]
        expand_dis = expandDis,  #6
        path_resolution = pathResolution #0.01  #0.5
        # play_area=[0, 10, 0, 14]
        )
    '''
    log_file = rootPath + 'experiments\\'+'sce1_'+ now+'_rr'+str(robotRadius)+'_ed'+str(expandDis)+'_pr'+str(pathResolution)+'.txt'
    path_points_ours, averge_CE_ours, maximumCE_ours, averageD2PP_ours, maximumD2pp_ours, pathLength_ours, averageTime_ours = [],[],[],[],[],[],[]
    path_points_cubicBezier, averge_CE_cubicBezier, maximumCE_cubicBezier, averageD2PP_cubicBezier, maximumD2pp_cubicBezier, pathLength_cubicBezier, averageTime_cubicBezier = [],[],[],[],[],[],[]
    path_points_cubicSpline, averge_CE_cubicSpline, maximumCE_cubicSpline, averageD2PP_cubicSpline, maximumD2pp_cubicSpline, pathLength_cubicSpline, averageTime_cubicSpline = [],[],[],[],[],[],[]
    path_points_BsplineA, averge_CE_BsplineA, maximumCE_BsplineA, averageD2PP_BsplineA, maximumD2pp_BsplineA, pathLength_BsplineA, averageTime_BsplineA = [],[],[],[],[],[],[]
    path_points_BsplineI, averge_CE_BsplineI, maximumCE_BsplineI, averageD2PP_BsplineI, maximumD2pp_BsplineI, pathLength_BsplineI, averageTime_BsplineI = [],[],[],[],[],[],[]
    path_points_POSQ, averge_CE_POSQ, maximumCE_POSQ, averageD2PP_POSQ, maximumD2pp_POSQ, pathLength_POSQ, averageTime_POSQ = [],[],[],[],[],[],[]
    path_points_Dubins, averge_CE_Dubins, maximumCE_Dubins, averageD2PP_Dubins, maximumD2pp_Dubins, pathLength_Dubins, averageTime_Dubins = [],[],[],[],[],[],[]
    path_points_ReedsShepp, averge_CE_ReedsShepp, maximumCE_ReedsShepp, averageD2PP_ReedsShepp, maximumD2pp_ReedsShepp, pathLength_ReedsShepp, averageTime_ReedsShepp = [],[],[],[],[],[],[]
    path_points_Clothoids, averge_CE_Clothoids, maximumCE_Clothoids, averageD2PP_Clothoids, maximumD2pp_Clothoids, pathLength_Clothoids, averageTime_Clothoids = [],[],[],[],[],[],[]
    
    log = open(log_file, 'a')
    planning_time = 0
    while planning_time < 10:
        try:

            print('the %dth planning' % (planning_time))
            points, path = rrt.planning(animation=show_animation)
                
            smoothed_path = []
            # print(np.shape(path))
            start_time_ours = time.time()
            path_ctrls = kCurves.kCurveOpen(points,max_iter=20)
            # our method
            path_smoothed_ours = kCurves.bezier_curve_gap(path_ctrls, ds = 0.01)
            end_time_ours = time.time()
            time_ours = end_time_ours - start_time_ours
            average_D2PP, maximum_D2PP = consistency_error(points, path_smoothed_ours)
            average_CE, maximum_CE = consistency_error(path_smoothed_ours, path)
            length = path_length(path_smoothed_ours)
    
            path_points_ours.append(len(points))
            averge_CE_ours.append(average_CE)
            maximumCE_ours.append(maximum_CE)
            averageD2PP_ours.append(average_D2PP)
            maximumD2pp_ours.append(maximum_D2PP)
            pathLength_ours.append(length)
            averageTime_ours.append(time_ours)

            # cubic Bezier
            start_time_cubicBezie = time.time()
            path_smoothed_cubicBezie = cubic_bezier.pathsmoothbezier(np.array(points)[:,0],np.array(points)[:,1], 0.01, 0.5)[:,0:2]
            end_time_cubicBezie = time.time()
            time_cubicBezie = end_time_cubicBezie - start_time_cubicBezie
            average_D2PP, maximum_D2PP = consistency_error(points, path_smoothed_cubicBezie)
            average_CE, maximum_CE = consistency_error(path_smoothed_cubicBezie, path)
            length = path_length(path_smoothed_cubicBezie)

            path_points_cubicBezier.append(len(points))
            averge_CE_cubicBezier.append(average_CE)
            maximumCE_cubicBezier.append(maximum_CE)
            averageD2PP_cubicBezier.append(average_D2PP)
            maximumD2pp_cubicBezier.append(maximum_D2PP)
            pathLength_cubicBezier.append(length)
            averageTime_cubicBezier.append(time_cubicBezie)

            # cubic spline
            start_time_cubicSpline = time.time()
            path_smoothed_cubicSpline, _, _, _ = cubic_spline.calc_spline_course(np.array(points)[:,0],np.array(points)[:,1], ds=0.01)
            end_time_cubicSpline = time.time()
            time_cubicSpline = end_time_cubicSpline - start_time_cubicSpline
            average_D2PP, maximum_D2PP = consistency_error(points, path_smoothed_cubicSpline)
            average_CE, maximum_CE = consistency_error(path_smoothed_cubicSpline, path)
            length = path_length(path_smoothed_cubicSpline)
            
            path_points_cubicSpline.append(len(points))
            averge_CE_cubicSpline.append(average_CE)
            maximumCE_cubicSpline.append(maximum_CE)
            averageD2PP_cubicSpline.append(average_D2PP)
            maximumD2pp_cubicSpline.append(maximum_D2PP)
            pathLength_cubicSpline.append(length)
            averageTime_cubicSpline.append(time_cubicSpline)

            # Abspline
            start_time_Abspline = time.time()
            path_smoothed_Abspline = bspline.approximate_b_spline_path(np.array(points)[:,0].tolist(),np.array(points)[:,1].tolist(), 0.01)
            end_time_Abspline = time.time()
            time_Abspline = end_time_Abspline - start_time_Abspline
            average_D2PP, maximum_D2PP = consistency_error(points, path_smoothed_Abspline)
            average_CE, maximum_CE = consistency_error(path_smoothed_Abspline, path)
            length = path_length(path_smoothed_Abspline)

            path_points_BsplineA.append(len(points))
            averge_CE_BsplineA.append(average_CE)
            maximumCE_BsplineA.append(maximum_CE)
            averageD2PP_BsplineA.append(average_D2PP)
            maximumD2pp_BsplineA.append(maximum_D2PP)
            pathLength_BsplineA.append(length)
            averageTime_BsplineA.append(time_Abspline)

            # Ibspline
            start_time_Ibsplin = time.time()
            path_smoothed_Ibspline = bspline.interpolate_b_spline_path(np.array(points)[:,0].tolist(),np.array(points)[:,1].tolist(), 0.01)
            end_time_Ibsplin = time.time()
            time_Ibsplin = end_time_Ibsplin - start_time_Ibsplin
            average_D2PP, maximum_D2PP = consistency_error(points, path_smoothed_Ibspline)
            average_CE, maximum_CE = consistency_error(path_smoothed_Ibspline, path)
            length = path_length(path_smoothed_Ibspline)

            path_points_BsplineI.append(len(points))
            averge_CE_BsplineI.append(average_CE)
            maximumCE_BsplineI.append(maximum_CE)
            averageD2PP_BsplineI.append(average_D2PP)
            maximumD2pp_BsplineI.append(maximum_D2PP)
            pathLength_BsplineI.append(length)
            averageTime_BsplineI.append(time_Ibsplin)

            # posq
            start_time_posq = time.time()
            path_smoothed_posq = posq.generate_path(np.array(points)[:,0],np.array(points)[:,1], 0.01)[:,0:2]
            end_time_posq = time.time()
            time_posq = end_time_posq - start_time_posq
            average_D2PP, maximum_D2PP = consistency_error(points, path_smoothed_posq)
            average_CE, maximum_CE = consistency_error(path_smoothed_posq, path)
            length = path_length(path_smoothed_posq)

            path_points_POSQ.append(len(points))
            averge_CE_POSQ.append(average_CE)
            maximumCE_POSQ.append(maximum_CE)
            averageD2PP_POSQ.append(average_D2PP)
            maximumD2pp_POSQ.append(maximum_D2PP)
            pathLength_POSQ.append(length)
            averageTime_POSQ.append(time_posq)

            # dubins
            start_time_dubins = time.time()
            path_smoothed_dubins = dubins.generate_path(np.array(points)[:,0],np.array(points)[:,1], 0.01, 0.5)
            end_time_dubins = time.time()
            time_dubins = end_time_dubins - start_time_dubins
            average_D2PP, maximum_D2PP = consistency_error(points, path_smoothed_dubins)
            average_CE, maximum_CE = consistency_error(path_smoothed_dubins, path)
            length = path_length(path_smoothed_dubins)

            path_points_Dubins.append(len(points))
            averge_CE_Dubins.append(average_CE)
            maximumCE_Dubins.append(maximum_CE)
            averageD2PP_Dubins.append(average_D2PP)
            maximumD2pp_Dubins.append(maximum_D2PP)
            pathLength_Dubins.append(length)
            averageTime_Dubins.append(time_dubins)

            # reeds_shepp
            start_time_reeds_shepp = time.time()
            path_smoothed_reeds_shepp = reeds_shepp.generate_final_path(np.array(points)[:,0],np.array(points)[:,1], 0.01, 0.5)
            end_time_reeds_shepp = time.time()
            time_reeds_shepp = end_time_reeds_shepp - start_time_reeds_shepp
            average_D2PP, maximum_D2PP = consistency_error(points, path_smoothed_reeds_shepp)
            average_CE, maximum_CE = consistency_error(path_smoothed_reeds_shepp, path)
            length = path_length(path_smoothed_reeds_shepp)

            path_points_ReedsShepp.append(len(points))
            averge_CE_ReedsShepp.append(average_CE)
            maximumCE_ReedsShepp.append(maximum_CE)
            averageD2PP_ReedsShepp.append(average_D2PP)
            maximumD2pp_ReedsShepp.append(maximum_D2PP)
            pathLength_ReedsShepp.append(length)
            averageTime_ReedsShepp.append(time_reeds_shepp)

            # clothoid
            start_time_clothoid = time.time()
            path_smoothed_clothoid = clothoid.generate_final_path(np.array(points)[:,0],np.array(points)[:,1], 0.01)
            end_time_clothoid = time.time()
            time_clothoid = end_time_clothoid - start_time_clothoid
            average_D2PP, maximum_D2PP = consistency_error(points, path_smoothed_clothoid)
            average_CE, maximum_CE = consistency_error(path_smoothed_clothoid, path)
            length = path_length(path_smoothed_clothoid)

            path_points_Clothoids.append(len(points))
            averge_CE_Clothoids.append(average_CE)
            maximumCE_Clothoids.append(maximum_CE)
            averageD2PP_Clothoids.append(average_D2PP)
            maximumD2pp_Clothoids.append(maximum_D2PP)
            pathLength_Clothoids.append(length)
            averageTime_Clothoids.append(time_clothoid)
            
            log.write('******************* ' + str(planning_time)+'th smoothing *******************\n')
            log.write('path_points_ours:'+str(path_points_ours)+'\n')
            log.write('averge_CE_ours:'+str(averge_CE_ours)+'\n')
            log.write('maximumCE_ours:'+str(maximumCE_ours)+'\n')
            log.write('averageD2PP_ours:'+str(averageD2PP_ours)+'\n')
            log.write('maximumD2pp_ours:'+str(maximumD2pp_ours)+'\n')
            log.write('pathLength_ours:'+str(pathLength_ours)+'\n')
            log.write('averageTime_ours:'+str(averageTime_ours)+'\n\n')            
            
            log.write('path_points_cubicBezier:'+str(path_points_cubicBezier)+'\n')
            log.write('averge_CE_cubicBezier:'+str(averge_CE_cubicBezier)+'\n')
            log.write('maximumCE_cubicBezier:'+str(maximumCE_cubicBezier)+'\n')
            log.write('averageD2PP_cubicBezier:'+str(averageD2PP_cubicBezier)+'\n')
            log.write('maximumD2pp_cubicBezier:'+str(maximumD2pp_cubicBezier)+'\n')
            log.write('pathLength_cubicBezier:'+str(pathLength_cubicBezier)+'\n')
            log.write('averageTime_cubicBezier:'+str(averageTime_cubicBezier)+'\n\n')
            
            log.write('path_points_cubicSpline:'+str(path_points_cubicSpline)+'\n')
            log.write('averge_CE_cubicSpline:'+str(averge_CE_cubicSpline)+'\n')
            log.write('maximumCE_cubicSpline:'+str(maximumCE_cubicSpline)+'\n')
            log.write('averageD2PP_cubicSpline:'+str(averageD2PP_cubicSpline)+'\n')
            log.write('maximumD2pp_cubicSpline:'+str(maximumD2pp_cubicSpline)+'\n')
            log.write('pathLength_cubicSpline:'+str(pathLength_cubicSpline)+'\n')
            log.write('averageTime_cubicSpline:'+str(averageTime_cubicSpline)+'\n\n')
            
            log.write('path_points_BsplineA:'+str(path_points_BsplineA)+'\n')
            log.write('averge_CE_BsplineA:'+str(averge_CE_BsplineA)+'\n')
            log.write('maximumCE_BsplineA:'+str(maximumCE_BsplineA)+'\n')
            log.write('averageD2PP_BsplineA:'+str(averageD2PP_BsplineA)+'\n')
            log.write('maximumD2pp_BsplineA:'+str(maximumD2pp_BsplineA)+'\n')
            log.write('pathLength_BsplineA:'+str(pathLength_BsplineA)+'\n')
            log.write('averageTime_BsplineA:'+str(averageTime_BsplineA)+'\n\n')
            
            log.write('path_points_BsplineI:'+str(path_points_BsplineI)+'\n')
            log.write('averge_CE_BsplineI:'+str(averge_CE_BsplineI)+'\n')
            log.write('maximumCE_BsplineI:'+str(maximumCE_BsplineI)+'\n')
            log.write('averageD2PP_BsplineI:'+str(averageD2PP_BsplineI)+'\n')
            log.write('maximumD2pp_BsplineI:'+str(maximumD2pp_BsplineI)+'\n')
            log.write('pathLength_BsplineI:'+str(pathLength_BsplineI)+'\n')
            log.write('averageTime_BsplineI:'+str(averageTime_BsplineI)+'\n\n')
            
            log.write('path_points_POSQ:'+str(path_points_POSQ)+'\n')
            log.write('averge_CE_POSQ:'+str(averge_CE_POSQ)+'\n')
            log.write('maximumCE_POSQ:'+str(maximumCE_POSQ)+'\n')
            log.write('averageD2PP_POSQ:'+str(averageD2PP_POSQ)+'\n')
            log.write('maximumD2pp_POSQ:'+str(maximumD2pp_POSQ)+'\n')
            log.write('pathLength_POSQ:'+str(pathLength_POSQ)+'\n')
            log.write('averageTime_POSQ:'+str(averageTime_POSQ)+'\n\n')
            
            log.write('path_points_Dubins:'+str(path_points_Dubins)+'\n')
            log.write('averge_CE_Dubins:'+str(averge_CE_Dubins)+'\n')
            log.write('maximumCE_Dubins:'+str(maximumCE_Dubins)+'\n')
            log.write('averageD2PP_Dubins:'+str(averageD2PP_Dubins)+'\n')
            log.write('maximumD2pp_Dubins:'+str(maximumD2pp_Dubins)+'\n')
            log.write('pathLength_Dubins:'+str(pathLength_Dubins)+'\n')
            log.write('averageTime_Dubins:'+str(averageTime_Dubins)+'\n\n')
            
            log.write('path_points_ReedsShepp:'+str(path_points_ReedsShepp)+'\n')
            log.write('averge_CE_ReedsShepp:'+str(averge_CE_ReedsShepp)+'\n')
            log.write('maximumCE_ReedsShepp:'+str(maximumCE_ReedsShepp)+'\n')
            log.write('averageD2PP_ReedsShepp:'+str(averageD2PP_ReedsShepp)+'\n')
            log.write('maximumD2pp_ReedsShepp:'+str(maximumD2pp_ReedsShepp)+'\n')
            log.write('pathLength_ReedsShepp:'+str(pathLength_ReedsShepp)+'\n')
            log.write('averageTime_ReedsShepp:'+str(averageTime_ReedsShepp)+'\n\n')
            
            log.write('path_points_Clothoids:'+str(path_points_Clothoids)+'\n')
            log.write('averge_CE_Clothoids:'+str(averge_CE_Clothoids)+'\n')
            log.write('maximumCE_Clothoids:'+str(maximumCE_Clothoids)+'\n')
            log.write('averageD2PP_Clothoids:'+str(averageD2PP_Clothoids)+'\n')
            log.write('maximumD2pp_Clothoids:'+str(maximumD2pp_Clothoids)+'\n')
            log.write('pathLength_Clothoids:'+str(pathLength_Clothoids)+'\n')
            log.write('averageTime_Clothoids:'+str(averageTime_Clothoids)+'\n\n')

        except:
            continue
        planning_time += 1

    mean_path_points_ours ,std_path_points_ours = np.mean(path_points_ours), np.std(path_points_ours)
    mean_averge_CE_ours ,std_averge_CE_ours = np.mean(averge_CE_ours), np.std(averge_CE_ours)
    mean_maximumCE_ours ,std_maximumCE_ours = np.mean(maximumCE_ours), np.std(maximumCE_ours)
    mean_averageD2PP_ours ,std_averageD2PP_ours = np.mean(averageD2PP_ours), np.std(averageD2PP_ours)
    mean_maximumD2pp_ours ,std_maximumD2pp_ours = np.mean(maximumD2pp_ours), np.std(maximumD2pp_ours)
    mean_pathLength_ours ,std_pathLength_ours = np.mean(pathLength_ours), np.std(pathLength_ours)
    mean_averageTime_ours ,std_averageTime_ours = np.mean(averageTime_ours), np.std(averageTime_ours)

    mean_path_points_cubicBezier ,std_path_points_cubicBezier = np.mean(path_points_cubicBezier), np.std(path_points_cubicBezier)
    mean_averge_CE_cubicBezier ,std_averge_CE_cubicBezier = np.mean(averge_CE_cubicBezier), np.std(averge_CE_cubicBezier)
    mean_maximumCE_cubicBezier ,std_maximumCE_cubicBezier = np.mean(maximumCE_cubicBezier), np.std(maximumCE_cubicBezier)
    mean_averageD2PP_cubicBezier ,std_averageD2PP_cubicBezier = np.mean(averageD2PP_cubicBezier), np.std(averageD2PP_cubicBezier)
    mean_maximumD2pp_cubicBezier ,std_maximumD2pp_cubicBezier = np.mean(maximumD2pp_cubicBezier), np.std(maximumD2pp_cubicBezier)
    mean_pathLength_cubicBezier ,std_pathLength_cubicBezier = np.mean(pathLength_cubicBezier), np.std(pathLength_cubicBezier)
    mean_averageTime_cubicBezier ,std_averageTime_cubicBezier = np.mean(averageTime_cubicBezier), np.std(averageTime_cubicBezier)

    mean_path_points_cubicSpline ,std_path_points_cubicSpline = np.mean(path_points_cubicSpline), np.std(path_points_cubicSpline)
    mean_averge_CE_cubicSpline ,std_averge_CE_cubicSpline = np.mean(averge_CE_cubicSpline), np.std(averge_CE_cubicSpline)
    mean_maximumCE_cubicSpline ,std_maximumCE_cubicSpline = np.mean(maximumCE_cubicSpline), np.std(maximumCE_cubicSpline)
    mean_averageD2PP_cubicSpline ,std_averageD2PP_cubicSpline = np.mean(averageD2PP_cubicSpline), np.std(averageD2PP_cubicSpline)
    mean_maximumD2pp_cubicSpline ,std_maximumD2pp_cubicSpline = np.mean(maximumD2pp_cubicSpline), np.std(maximumD2pp_cubicSpline)
    mean_pathLength_cubicSpline ,std_pathLength_cubicSpline = np.mean(pathLength_cubicSpline), np.std(pathLength_cubicSpline)
    mean_averageTime_cubicSpline ,std_averageTime_cubicSpline = np.mean(averageTime_cubicSpline), np.std(averageTime_cubicSpline)

    mean_path_points_BsplineA ,std_path_points_BsplineA = np.mean(path_points_BsplineA), np.std(path_points_BsplineA)
    mean_averge_CE_BsplineA ,std_averge_CE_BsplineA = np.mean(averge_CE_BsplineA), np.std(averge_CE_BsplineA)
    mean_maximumCE_BsplineA ,std_maximumCE_BsplineA = np.mean(maximumCE_BsplineA), np.std(maximumCE_BsplineA)
    mean_averageD2PP_BsplineA ,std_averageD2PP_BsplineA = np.mean(averageD2PP_BsplineA), np.std(averageD2PP_BsplineA)
    mean_maximumD2pp_BsplineA ,std_maximumD2pp_BsplineA = np.mean(maximumD2pp_BsplineA), np.std(maximumD2pp_BsplineA)
    mean_pathLength_BsplineA ,std_pathLength_BsplineA = np.mean(pathLength_BsplineA), np.std(pathLength_BsplineA)
    mean_averageTime_BsplineA ,std_averageTime_BsplineA = np.mean(averageTime_BsplineA), np.std(averageTime_BsplineA)

    mean_path_points_BsplineI ,std_path_points_BsplineI = np.mean(path_points_BsplineI), np.std(path_points_BsplineI)
    mean_averge_CE_BsplineI ,std_averge_CE_BsplineI = np.mean(averge_CE_BsplineI), np.std(averge_CE_BsplineI)
    mean_maximumCE_BsplineI ,std_maximumCE_BsplineI = np.mean(maximumCE_BsplineI), np.std(maximumCE_BsplineI)
    mean_averageD2PP_BsplineI ,std_averageD2PP_BsplineI = np.mean(averageD2PP_BsplineI), np.std(averageD2PP_BsplineI)
    mean_maximumD2pp_BsplineI ,std_maximumD2pp_BsplineI = np.mean(maximumD2pp_BsplineI), np.std(maximumD2pp_BsplineI)
    mean_pathLength_BsplineI ,std_pathLength_BsplineI = np.mean(pathLength_BsplineI), np.std(pathLength_BsplineI)
    mean_averageTime_BsplineI ,std_averageTime_BsplineI = np.mean(averageTime_BsplineI), np.std(averageTime_BsplineI)

    mean_path_points_POSQ ,std_path_points_POSQ = np.mean(path_points_POSQ), np.std(path_points_POSQ)
    mean_averge_CE_POSQ ,std_averge_CE_POSQ = np.mean(averge_CE_POSQ), np.std(averge_CE_POSQ)
    mean_maximumCE_POSQ ,std_maximumCE_POSQ = np.mean(maximumCE_POSQ), np.std(maximumCE_POSQ)
    mean_averageD2PP_POSQ ,std_averageD2PP_POSQ = np.mean(averageD2PP_POSQ), np.std(averageD2PP_POSQ)
    mean_maximumD2pp_POSQ ,std_maximumD2pp_POSQ = np.mean(maximumD2pp_POSQ), np.std(maximumD2pp_POSQ)
    mean_pathLength_POSQ ,std_pathLength_POSQ = np.mean(pathLength_POSQ), np.std(pathLength_POSQ)
    mean_averageTime_POSQ ,std_averageTime_POSQ = np.mean(averageTime_POSQ), np.std(averageTime_POSQ)

    mean_path_points_Dubins ,std_path_points_Dubins = np.mean(path_points_Dubins), np.std(path_points_Dubins)
    mean_averge_CE_Dubins ,std_averge_CE_Dubins = np.mean(averge_CE_Dubins), np.std(averge_CE_Dubins)
    mean_maximumCE_Dubins ,std_maximumCE_Dubins = np.mean(maximumCE_Dubins), np.std(maximumCE_Dubins)
    mean_averageD2PP_Dubins ,std_averageD2PP_Dubins = np.mean(averageD2PP_Dubins), np.std(averageD2PP_Dubins)
    mean_maximumD2pp_Dubins ,std_maximumD2pp_Dubins = np.mean(maximumD2pp_Dubins), np.std(maximumD2pp_Dubins)
    mean_pathLength_Dubins ,std_pathLength_Dubins = np.mean(pathLength_Dubins), np.std(pathLength_Dubins)
    mean_averageTime_Dubins ,std_averageTime_Dubins = np.mean(averageTime_Dubins), np.std(averageTime_Dubins)

    mean_path_points_ReedsShepp ,std_path_points_ReedsShepp = np.mean(path_points_ReedsShepp), np.std(path_points_ReedsShepp)
    mean_averge_CE_ReedsShepp ,std_averge_CE_ReedsShepp = np.mean(averge_CE_ReedsShepp), np.std(averge_CE_ReedsShepp)
    mean_maximumCE_ReedsShepp ,std_maximumCE_ReedsShepp = np.mean(maximumCE_ReedsShepp), np.std(maximumCE_ReedsShepp)
    mean_averageD2PP_ReedsShepp ,std_averageD2PP_ReedsShepp = np.mean(averageD2PP_ReedsShepp), np.std(averageD2PP_ReedsShepp)
    mean_maximumD2pp_ReedsShepp ,std_maximumD2pp_ReedsShepp = np.mean(maximumD2pp_ReedsShepp), np.std(maximumD2pp_ReedsShepp)
    mean_pathLength_ReedsShepp ,std_pathLength_ReedsShepp = np.mean(pathLength_ReedsShepp), np.std(pathLength_ReedsShepp, )
    mean_averageTime_ReedsShepp ,std_averageTime_ReedsShepp = np.mean(averageTime_ReedsShepp), np.std(averageTime_ReedsShepp)

    mean_path_points_Clothoids ,std_path_points_Clothoids = np.mean(path_points_Clothoids), np.std(path_points_Clothoids)
    mean_averge_CE_Clothoids ,std_averge_CE_Clothoids = np.mean(averge_CE_Clothoids), np.std(averge_CE_Clothoids)
    mean_maximumCE_Clothoids ,std_maximumCE_Clothoids = np.mean(maximumCE_Clothoids), np.std(maximumCE_Clothoids)
    mean_averageD2PP_Clothoids ,std_averageD2PP_Clothoids = np.mean(averageD2PP_Clothoids), np.std(averageD2PP_Clothoids)
    mean_maximumD2pp_Clothoids ,std_maximumD2pp_Clothoids = np.mean(maximumD2pp_Clothoids), np.std(maximumD2pp_Clothoids)
    mean_pathLength_Clothoids ,std_pathLength_Clothoids = np.mean(pathLength_Clothoids), np.std(pathLength_Clothoids)
    mean_averageTime_Clothoids ,std_averageTime_Clothoids = np.mean(averageTime_Clothoids), np.std(averageTime_Clothoids)

    log.write('******************* total results *******************\n')
    log.write('mean_path_points_ours:'+str(mean_path_points_ours)+' +- '+str(std_path_points_ours)+'\n')
    log.write('mean_averge_CE_ours:'+str(mean_averge_CE_ours)+' +- '+str(std_averge_CE_ours)+'\n')
    log.write('mean_maximumCE_ours:'+str(mean_maximumCE_ours)+' +- '+str(std_maximumCE_ours)+'\n')
    log.write('mean_averageD2PP_ours:'+str(mean_averageD2PP_ours)+' +- '+str(std_averageD2PP_ours)+'\n')
    log.write('mean_maximumD2pp_ours:'+str(mean_maximumD2pp_ours)+' +- '+str(std_maximumD2pp_ours)+'\n')
    log.write('mean_pathLength_ours:'+str(mean_pathLength_ours)+' +- '+str(std_pathLength_ours)+'\n')
    log.write('mean_averageTime_ours:'+str(mean_averageTime_ours)+' +- '+str(std_averageTime_ours)+'\n\n')            
    
    log.write('mean_path_points_cubicBezier:'+str(mean_path_points_cubicBezier)+' +- '+str(std_path_points_cubicBezier)+'\n')
    log.write('mean_averge_CE_cubicBezier:'+str(mean_averge_CE_cubicBezier)+' +- '+str(std_averge_CE_cubicBezier)+'\n')
    log.write('mean_maximumCE_cubicBezier:'+str(mean_maximumCE_cubicBezier)+' +- '+str(std_maximumCE_cubicBezier)+'\n')
    log.write('mean_averageD2PP_cubicBezier:'+str(mean_averageD2PP_cubicBezier)+' +- '+str(std_averageD2PP_cubicBezier)+'\n')
    log.write('mean_maximumD2pp_cubicBezier:'+str(mean_maximumD2pp_cubicBezier)+' +- '+str(std_maximumD2pp_cubicBezier)+'\n')
    log.write('mean_pathLength_cubicBezier:'+str(mean_pathLength_cubicBezier)+' +- '+str(std_pathLength_cubicBezier)+'\n')
    log.write('mean_averageTime_cubicBezier:'+str(mean_averageTime_cubicBezier)+' +- '+str(std_averageTime_cubicBezier)+'\n\n')
    
    log.write('mean_path_points_cubicSpline:'+str(mean_path_points_cubicSpline)+' +- '+str(std_path_points_cubicSpline)+'\n')
    log.write('mean_averge_CE_cubicSpline:'+str(mean_averge_CE_cubicSpline)+' +- '+str(std_averge_CE_cubicSpline)+'\n')
    log.write('mean_maximumCE_cubicSpline:'+str(mean_maximumCE_cubicSpline)+' +- '+str(std_maximumCE_cubicSpline)+'\n')
    log.write('mean_averageD2PP_cubicSpline:'+str(mean_averageD2PP_cubicSpline)+' +- '+str(std_averageD2PP_cubicSpline)+'\n')
    log.write('mean_maximumD2pp_cubicSpline:'+str(mean_maximumD2pp_cubicSpline)+' +- '+str(std_maximumD2pp_cubicSpline)+'\n')
    log.write('mean_pathLength_cubicSpline:'+str(mean_pathLength_cubicSpline)+' +- '+str(std_pathLength_cubicSpline)+'\n')
    log.write('mean_averageTime_cubicSpline:'+str(mean_averageTime_cubicSpline)+' +- '+str(std_averageTime_cubicSpline)+'\n\n')
    
    log.write('mean_path_points_BsplineA:'+str(mean_path_points_BsplineA)+' +- '+str(std_path_points_BsplineA)+'\n')
    log.write('mean_averge_CE_BsplineA:'+str(mean_averge_CE_BsplineA)+' +- '+str(std_averge_CE_BsplineA)+'\n')
    log.write('mean_maximumCE_BsplineA:'+str(mean_maximumCE_BsplineA)+' +- '+str(std_maximumCE_BsplineA)+'\n')
    log.write('mean_averageD2PP_BsplineA:'+str(mean_averageD2PP_BsplineA)+' +- '+str(std_averageD2PP_BsplineA)+'\n')
    log.write('mean_maximumD2pp_BsplineA:'+str(mean_maximumD2pp_BsplineA)+' +- '+str(std_maximumD2pp_BsplineA)+'\n')
    log.write('mean_pathLength_BsplineA:'+str(mean_pathLength_BsplineA)+' +- '+str(std_pathLength_BsplineA)+'\n')
    log.write('mean_averageTime_BsplineA:'+str(mean_averageTime_BsplineA)+' +- '+str(std_averageTime_BsplineA)+'\n\n')
    
    log.write('mean_path_points_BsplineI:'+str(mean_path_points_BsplineI)+' +- '+str(std_path_points_BsplineI)+'\n')
    log.write('mean_averge_CE_BsplineI:'+str(mean_averge_CE_BsplineI)+' +- '+str(std_averge_CE_BsplineI)+'\n')
    log.write('mean_maximumCE_BsplineI:'+str(mean_maximumCE_BsplineI)+' +- '+str(std_maximumCE_BsplineI)+'\n')
    log.write('mean_averageD2PP_BsplineI:'+str(mean_averageD2PP_BsplineI)+' +- '+str(std_averageD2PP_BsplineI)+'\n')
    log.write('mean_maximumD2pp_BsplineI:'+str(mean_maximumD2pp_BsplineI)+' +- '+str(std_maximumD2pp_BsplineI)+'\n')
    log.write('mean_pathLength_BsplineI:'+str(mean_pathLength_BsplineI)+' +- '+str(std_pathLength_BsplineI)+'\n')
    log.write('mean_averageTime_BsplineI:'+str(mean_averageTime_BsplineI)+' +- '+str(std_averageTime_BsplineI)+'\n\n')
    
    log.write('mean_path_points_POSQ:'+str(mean_path_points_POSQ)+' +- '+str(std_path_points_POSQ)+'\n')
    log.write('mean_averge_CE_POSQ:'+str(mean_averge_CE_POSQ)+' +- '+str(std_averge_CE_POSQ)+'\n')
    log.write('mean_maximumCE_POSQ:'+str(mean_maximumCE_POSQ)+' +- '+str(std_maximumCE_POSQ)+'\n')
    log.write('mean_averageD2PP_POSQ:'+str(mean_averageD2PP_POSQ)+' +- '+str(std_averageD2PP_POSQ)+'\n')
    log.write('mean_maximumD2pp_POSQ:'+str(mean_maximumD2pp_POSQ)+' +- '+str(std_maximumD2pp_POSQ)+'\n')
    log.write('mean_pathLength_POSQ:'+str(mean_pathLength_POSQ)+' +- '+str(std_pathLength_POSQ)+'\n')
    log.write('mean_averageTime_POSQ:'+str(mean_averageTime_POSQ)+' +- '+str(std_averageTime_POSQ)+'\n\n')
    
    log.write('mean_path_points_Dubins:'+str(mean_path_points_Dubins)+' +- '+str(std_path_points_Dubins)+'\n')
    log.write('mean_averge_CE_Dubins:'+str(mean_averge_CE_Dubins)+' +- '+str(std_averge_CE_Dubins)+'\n')
    log.write('mean_maximumCE_Dubins:'+str(mean_maximumCE_Dubins)+' +- '+str(std_maximumCE_Dubins)+'\n')
    log.write('mean_averageD2PP_Dubins:'+str(mean_averageD2PP_Dubins)+' +- '+str(std_averageD2PP_Dubins)+'\n')
    log.write('mean_maximumD2pp_Dubins:'+str(mean_maximumD2pp_Dubins)+' +- '+str(std_maximumD2pp_Dubins)+'\n')
    log.write('mean_pathLength_Dubins:'+str(mean_pathLength_Dubins)+' +- '+str(std_pathLength_Dubins)+'\n')
    log.write('mean_averageTime_Dubins:'+str(mean_averageTime_Dubins)+' +- '+str(std_averageTime_Dubins)+'\n\n')
    
    log.write('mean_path_points_ReedsShepp:'+str(mean_path_points_ReedsShepp)+' +- '+str(std_path_points_ReedsShepp)+'\n')
    log.write('mean_averge_CE_ReedsShepp:'+str(mean_averge_CE_ReedsShepp)+' +- '+str(std_averge_CE_ReedsShepp)+'\n')
    log.write('mean_maximumCE_ReedsShepp:'+str(mean_maximumCE_ReedsShepp)+' +- '+str(std_maximumCE_ReedsShepp)+'\n')
    log.write('mean_averageD2PP_ReedsShepp:'+str(mean_averageD2PP_ReedsShepp)+' +- '+str(std_averageD2PP_ReedsShepp)+'\n')
    log.write('mean_maximumD2pp_ReedsShepp:'+str(mean_maximumD2pp_ReedsShepp)+' +- '+str(std_maximumD2pp_ReedsShepp)+'\n')
    log.write('mean_pathLength_ReedsShepp:'+str(mean_pathLength_ReedsShepp)+' +- '+str(std_pathLength_ReedsShepp)+'\n')
    log.write('mean_averageTime_ReedsShepp:'+str(mean_averageTime_ReedsShepp)+' +- '+str(std_averageTime_ReedsShepp)+'\n\n')
    
    log.write('mean_path_points_Clothoids:'+str(mean_path_points_Clothoids)+' +- '+str(std_path_points_Clothoids)+'\n')
    log.write('mean_averge_CE_Clothoids:'+str(mean_averge_CE_Clothoids)+' +- '+str(std_averge_CE_Clothoids)+'\n')
    log.write('mean_maximumCE_Clothoids:'+str(mean_maximumCE_Clothoids)+' +- '+str(std_maximumCE_Clothoids)+'\n')
    log.write('mean_averageD2PP_Clothoids:'+str(mean_averageD2PP_Clothoids)+' +- '+str(std_averageD2PP_Clothoids)+'\n')
    log.write('mean_maximumD2pp_Clothoids:'+str(mean_maximumD2pp_Clothoids)+' +- '+str(std_maximumD2pp_Clothoids)+'\n')
    log.write('mean_pathLength_Clothoids:'+str(mean_pathLength_Clothoids)+' +- '+str(std_pathLength_Clothoids)+'\n')
    log.write('mean_averageTime_Clothoids:'+str(mean_averageTime_Clothoids)+' +- '+str(std_averageTime_Clothoids))
    log.close()
    '''
    
    points, path = rrt.planning(animation=show_animation)
    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")
    path_ctrls = kCurves.kCurveOpen(points)
    path_smoothed = kCurves.bezier_curve_gap(path_ctrls, ds = 0.01)
    # Draw final path
    if show_animation:
        plt.figure()
        rrt.draw_graph()
        plt.scatter([x for (x, y) in points], [y for (x, y) in points], color="g", s=50, label='path point')
        plt.plot([x for (x, y) in points], [y for (x, y) in points], '-r', linewidth=4, label='path')
        # plt.grid(True)
        plt.axis("equal")
        plt.pause(0.01)  # Need for Mac
        
        plt.figure()
        rrt.draw_graph()
        plt.scatter([x for (x, y) in points], [y for (x, y) in points], color="g", s=50, label='path point')
        plt.plot(path_smoothed[:,0], path_smoothed[:,1], "b", linewidth=4,label='trajectory')
        plt.axis("equal")
        plt.pause(0.01)
        plt.show()

if __name__ == '__main__':
    main()
