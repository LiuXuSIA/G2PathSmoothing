
import numpy as np
import posq
import pcdProcess
import random
import matplotlib.pyplot as plt
import math

def generate_random_floats(lower_limit, upper_limit, size):
    random_floats = []
    for _ in range(size):
        random_float = random.uniform(lower_limit, upper_limit)
        random_floats.append(random_float)
    return random_floats

def generate_path(rx, ry):
    direction = 1
    resolution = 0.1
    base = 0.4
    init_t = 0.1
    max_speed = 1.0
    trajs = []

    angles = generate_random_floats(-math.pi, math.pi, len(rx))
    road_points = np.c_[rx, ry, angles]
    for i in range(len(road_points)-1):
        source = np.array([road_points[i][0], road_points[i][1], road_points[i][2]])
        target = np.array([road_points[i+1][0], road_points[i+1][1], road_points[i+1][1]])
        
        traj, speedvec, vel, inct = posq.integrate(source, target, direction,
                                                resolution, base,
                                                init_t, max_speed, nS=0)
        trajs.extend(traj.tolist())

    return trajs

if __name__ == "__main__":

    rootPath_TITS = 'F:\\PC2Win10\\Study\\PHD\\Research\\paper_writting\\TITS2023\\'
    result_hill = rootPath_TITS + 'results\\hill\\result_hill.txt'
    point_quarry_Astar = rootPath_TITS + 'results\\quarry\\point_quarry_Astar.txt'
    point_sia_hill_Astar = rootPath_TITS + 'results\\hill\\point_sia_hill_Astar.txt'
    
    points = pcdProcess.loadData(point_sia_hill_Astar)
    rx, ry = points[:, 0], points[:, 1]
    trajs = generate_path(rx, ry)
    plt.figure(figsize=(10, 10))
    plt.plot(np.array(trajs)[:,0], np.array(trajs)[:,1])
    plt.show()

# angles = []
# for i in range(len(road_points)-1):
#     current = road_points[i+1] - road_points[i]
#     angles.append(np.arccos(current[0] / np.sqrt(current[0]**2 + current[1]**2)))

# angles.append(angles[-1])

# plt.plot(traj[0, 0], traj[0, 0], ls='', marker='8', color='b',
#          markersize=15, label='start')
# plt.plot(traj[-1, 0], traj[-1, 0], ls='', marker='8', color='g',
#          markersize=15, label='goal')

# ax = plt.axes()
# for wp in traj:
#     vx, vy = np.cos(wp[2]), np.sin(wp[2])
#     ax.arrow(wp[0], wp[1], 0.1*vx, 0.1*vy, head_width=0.02,
#              head_length=0.02, lw=0.4, fc='brown', ec='brown')

# ax = plt.axes()
# for wp in traj:
# vx, vy = np.cos(wp[2]), np.sin(wp[2])
# plt.plot(traj[:,0], traj[:,1])

# plt.xlim((-0.2, 1.2))
# plt.ylim((-0.2, 1.2))
# plt.savefig('posq_demo.pdf')
# plt.show()
