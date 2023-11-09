"""
Path Planner with B-Spline
author: Atsushi Sakai (@Atsushi_twi)
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scipy_interpolate
import pcdProcess

def approximate_b_spline_path(x: list, y: list, ds: float,
                              degree: int = 3) -> tuple:
    """
    approximate points with a B-Spline path

    :param x: x position list of approximated points
    :param y: y position list of approximated points
    :param n_path_points: number of path points
    :param degree: (Optional) B Spline curve degree
    :return: x and y position list of the result path
    """
    t = range(len(x))
    x_tup = scipy_interpolate.splrep(t, x, k=degree)
    y_tup = scipy_interpolate.splrep(t, y, k=degree)

    x_list = list(x_tup)
    x_list[1] = x + [0.0, 0.0, 0.0, 0.0]

    y_list = list(y_tup)
    y_list[1] = y + [0.0, 0.0, 0.0, 0.0]
    
    ipl_t = np.arange(0.0, len(x) - 1, ds)
    rx = scipy_interpolate.splev(ipl_t, x_list)
    ry = scipy_interpolate.splev(ipl_t, y_list)

    return np.c_[rx, ry]


def interpolate_b_spline_path(x: list, y: list, ds: float,
                              degree: int = 3) -> tuple:
    """
    interpolate points with a B-Spline path

    :param x: x positions of interpolated points
    :param y: y positions of interpolated points
    :param n_path_points: number of path points
    :param degree: B-Spline degree
    :return: x and y position list of the result path
    """

    ipl_t = np.linspace(0.0, len(x) - 1, len(x))
    spl_i_x = scipy_interpolate.make_interp_spline(ipl_t, x, k=degree)
    spl_i_y = scipy_interpolate.make_interp_spline(ipl_t, y, k=degree)
    
    travel = np.arange(0.0, len(x) - 1, ds)

    x, y = spl_i_x(travel), spl_i_y(travel)

    # 计算一阶和二阶导数的样条
    dx = spl_i_x.derivative()
    ddx = spl_i_x.derivative(2)

    dy = spl_i_y.derivative()
    ddy = spl_i_y.derivative(2)

    # 计算一阶和二阶导数的值
    dx_dt = dx(travel)
    ddx_dt = ddx(travel)
    
    dy_dt = dy(travel)
    ddy_dt = ddy(travel)

    dy_dx = dy_dt / dx_dt
    ddy_dx = (ddy_dt*dx_dt - ddx_dt*dy_dt) / (dx_dt ** 2)

    # 计算曲率
    curvature = np.abs(ddy_dx) / (1 + dy_dx**2)**1.5
    
    dt = np.c_[dx_dt, dy_dt]
    dt /= np.linalg.norm(dt, 2)
    normal = (np.c_[x, y]) - (np.c_[-dt[:,1], dt[:, 0]] * np.tile(curvature.reshape(-1, 1), 2)*8)

    # curvatures = []
    # for t_i in t:
    #     point = bezier_point(t_i, control_points[i])
    #     dt = bezier_point(t_i, derivatives_cp[1][i])
    #     ddt = bezier_point(t_i, derivatives_cp[2][i])
    #     # Radius of curvature
    #     curvature = bezier_curvature(dt[0], dt[1], ddt[0], ddt[1])
    #     curvatures.append(curvature)
    #     # radius = 1 / curvature
    #     # Normalize derivative
    #     dt /= np.linalg.norm(dt, 2)
    #     # tangent = np.array([point, point + dt])
    #     normal = np.array([point, point - np.array([- dt[1], dt[0]]) * curvature * 3])
    #     # curvature_center = point + np.array([- dt[1], dt[0]]) * radius
    #     # circle = plt.Circle(tuple(curvature_center), radius, color=(0, 0.8, 0.8), fill=False, linewidth=1)
    #     # ax.plot(point[0], point[1])
    #     # ax.plot(tangent[:, 0], tangent[:, 1])
    #     ax.plot(normal[:, 0], normal[:, 1], color='green', linewidth=1.5)
    #     normals.append(point - np.array([- dt[1], dt[0]]) * curvature * 3)

    return np.c_[x, y], normal

rootPath_TITS = 'F:\\PC2Win10\\Study\\PHD\\Research\\paper_writting\\TITS2023\\'
result_hill = rootPath_TITS + 'results\\hill\\result_hill.txt'
point_quarry_Astar = rootPath_TITS + 'results\\quarry\\point_quarry_Astar.txt'
point_sia_hill_Astar = rootPath_TITS + 'results\\hill\\point_sia_hill_Astar.txt'
point_planet_Astar = rootPath_TITS + 'results\\planet\\point_planet_Astar.txt'

road_points = pcdProcess.loadData(point_sia_hill_Astar)

def main():
    print(__file__ + " start!!")
    # way points
    # way_point_x = [-1.0, 3.0, 4.0, 2.0, 1.0]
    # way_point_y = [0.0, -3.0, 1.0, 1.0, 3.0]
    way_point_x = road_points[:, 0].tolist()
    way_point_y = road_points[:, 1].tolist()
    
    n_course_point = 10  # sampling number

    apath = approximate_b_spline_path(way_point_x, way_point_y, 0.01, 3)

    ipath, normal = interpolate_b_spline_path(way_point_x, way_point_y, 0.01, 3)

    # show results
    # plt.plot(way_point_x, way_point_y, '-og', label="way points")
    # plt.plot(apath[:,0], apath[:,1], '-r', label="Approximated B-Spline path")
    # plt.plot(ipath[:,0], ipath[:,1], '-b', label="Interpolated B-Spline path")
    # plt.grid(True)
    # plt.legend()
    # plt.axis("equal")
    # plt.show()
    plt.figure()
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2.2)
    ax.spines['left'].set_linewidth(2.2)
    ax.spines['top'].set_linewidth(2.2)
    ax.spines['right'].set_linewidth(2.2)
    plt.title('approximated b-Spline', fontsize = 15)
    plt.plot(apath[:,0]*5, apath[:,1]*5,linewidth=4, c='darkorange',label='trajectory')
    plt.scatter(np.array(way_point_x)*5, np.array(way_point_y)*5,c='g',s=40, label='path points')
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.legend(fontsize = 18)

    plt.figure()
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2.2)
    ax.spines['left'].set_linewidth(2.2)
    ax.spines['top'].set_linewidth(2.2)
    ax.spines['right'].set_linewidth(2.2)
    plt.title('interpolated b-Spline', fontsize = 15)
    plt.plot(ipath[:,0]*5, ipath[:,1]*5,linewidth=4, c='darkorange',label='trajectory')
    plt.scatter(np.array(way_point_x)*5, np.array(way_point_y)*5,c='g',s=40, label='path points')
    # for i, j in zip(ipath, normal):
    #     plt.plot([i[0], j[0]], [i[1], j[1]], color='green', linewidth=1.5)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.legend(fontsize = 18)

    # plt.figure()
    # plt.plot(ipath[:,0], curvatures, color='green', linewidth=1.5)
    plt.show()


if __name__ == '__main__':
    main()
