import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scipy_interpolate

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

    return np.c_[x, y]

def loadData(filePath):
    Data = []
    fr = open(filePath)
    initialData = fr.readlines()
    fr.close()
    for element in initialData:
        lineArr = element.strip().split(' ')
        Data.append([float(x) for x in lineArr])
    return np.array(Data)

if __name__ == '__main__':

    point_hill = 'pathPoints\\point_hill.txt'

    road_points = loadData(point_hill)
    
    way_point_x = road_points[:, 0].tolist()
    way_point_y = road_points[:, 1].tolist()
    
    n_course_point = 10  # sampling number

    apath = approximate_b_spline_path(way_point_x, way_point_y, 0.01, 3)

    ipath = interpolate_b_spline_path(way_point_x, way_point_y, 0.01, 3)

    plt.figure()
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2.2)
    ax.spines['left'].set_linewidth(2.2)
    ax.spines['top'].set_linewidth(2.2)
    ax.spines['right'].set_linewidth(2.2)
    plt.title('approximated b-Spline', fontsize = 15)
    plt.plot(apath[:,0], apath[:,1],linewidth=4, c='darkorange',label='trajectory')
    plt.scatter(way_point_x, way_point_y,c='g',s=40, label='path points')
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
    plt.plot(ipath[:,0], ipath[:,1],linewidth=4, c='darkorange',label='trajectory')
    plt.scatter(way_point_x, way_point_y,c='g',s=40, label='path points')
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.legend(fontsize = 18)

    plt.show()
