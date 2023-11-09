import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, special
import pcdProcess
import keyboard
import time
from RRT.rrt import path_length

def maxT(control_points):
    t_max = np.dot(control_points[0]-control_points[1], control_points[0]-2*control_points[1]+control_points[2]) / \
        np.linalg.norm(control_points[0]-2*control_points[1]+control_points[2]) ** 2
    return t_max

def maxParam(c0, cp, c2):
    a = np.dot(c2 - c0, c2 - c0)
    b = 3 * np.dot(c2 - c0, c0 - cp)
    c = np.dot(3 * c0 - 2 * cp - c2, c0 - cp)
    d = - np.dot(c0 - cp, c0 - cp)
    # solve a t^3 + b t^2 + c t + d = 0 in [0, 1]
    # TODO: if c0==c2, solve a quadratic or linear equation.
    # https://en.wikipedia.org/wiki/Cubic_equation
    # u = t + b / 3a
    p = (3 * a * c - b * b) / 3 / a / a
    q = (2 * b * b * b - 9 * a * b * c + 27 * a * a * d) / 27 / a / a / a
    if (4 * p * p * p + 27 * q * q >= 0):
        # single real root
	    return np.cbrt(-q / 2 + np.sqrt(q * q / 4 + p * p * p / 27)) + np.cbrt(-q / 2 - np.sqrt(q * q / 4 + p * p * p / 27)) - b / 3 / a
    else:
        # three real roots
        for k in range(3):
            t = 2 * np.sqrt(-p / 3) * np.cos(1. / 3 * np.arccos(3 * q / 2. / p * np.sqrt(-3 / p)) - 2 * np.pi * k / 3.) - b / 3 / a
            if 0 <= t and t <= 1:
                return t
    # error
    return -1

def area(A, B, C):
    mat = np.r_[B-A, C-B].reshape(2,2)
    # print(mat)
    return np.abs(np.linalg.det(mat) / 2)
    # return (np.linalg.det(mat) / 2)

# def lamb(c0, c1, c2, c3):
#     a1 = np.linalg.det(np.c_[c2-c1, c0-c1]) - np.linalg.det(np.c_[c2-c1, c3-c2])
#     a2 = np.linalg.det(np.c_[c2-c1, c0-c1]) + np.linalg.det(np.c_[c2-c1, c3-c2])
#     b = 2*np.linalg.det(np.c_[c2-c1, c1-c0])
#     c = np.linalg.det(np.c_[c2-c1, c0-c1])

#     delta_1 = b**2 - 4*a1*c
#     delta_2 = b**2 - 4*a2*c

#     if delta_1 < 0:
#         print(a1, a2)
    
#     (delta, a) = (delta_1, a1) if delta_1>=0 else (delta_2, a2)

#     t1 = (-b + np.sqrt(delta))/(2*a)
#     t2 = (-b - np.sqrt(delta))/(2*a)

#     return t1 if t1>=0 and t1<=1 else t2

def lamb(c0, c1, c3, c4):
    return np.sqrt(area(c0, c1, c3)) / (np.sqrt(area(c0, c1, c3)+1e-15) + np.sqrt(area(c1, c3, c4)+1e-15))

def mod(i, n):
    return ((i + n) % n)

def bernstein_poly(n, i, t):
    return special.comb(n, i) * t ** i * (1 - t) ** (n - i)

def bezier_point(t, control_points):
    n = len(control_points) - 1
    return np.sum([bernstein_poly(n, i, t) * control_points[i] for i in range(n + 1)], axis=0)

def bezier_curve(control_points, n_points = 100):
    inters = []
    for control_point in control_points:
        for t in np.linspace(0, 1, n_points):
            inters.append(bezier_point(t, control_point))
    return np.array(inters)   ########## TO DO

def bezier_derivatives_control_points(control_points, n_derivatives):
    # Compute control points of the successive derivatives of a given bezier curve.
    # A derivative of a bezier curve is a bezier curve.
    w = {0: control_points}
    for i in range(n_derivatives):
        for k in range(len(w[i])):
            n = len(w[i][k])
            if k == 0:
                w[i + 1] = [[(n - 1) * (w[i][k][j + 1] - w[i][k][j]) for j in range(n - 1)]]
            else:
                w[i + 1].append([(n - 1) * (w[i][k][j + 1] - w[i][k][j]) for j in range(n - 1)])
    return w

def bezier_curvature(dx, dy, ddx, ddy):
    #Compute curvature at one point given first and second derivatives.
    return (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)

# calculate the curvature only using the control point
def curvature1(ctrls, t):
    return 0.5*np.linalg.det(np.c_[ctrls[1]-ctrls[0], ctrls[2]-ctrls[1]]) / \
        np.linalg.norm((1-t)*(ctrls[1]-ctrls[0])+t*(ctrls[2]-ctrls[1]))**3

def plot_figure(input_points, control_points, curve_inters):
    ax.scatter(np.array(input_points)[:,0], np.array(input_points)[:,1], marker='o', color = 'r', s = 100)
    ax.scatter(np.array(control_points)[:,0], np.array(control_points)[:,1], marker='o', color = 'white', s = 50)
    ax.plot(curve_inters[:,0], curve_inters[:,1], color = 'darkorange', linewidth = 5)
    # ax.plot(curve_inters[100:,0], curve_inters[100:,1], color = 'saddlebrown', linewidth = 3)

def plot_features(t, t_max, control_points, derivatives_cp):
    normals = []
    t = [t] if isinstance(t, int) or isinstance(t, float) else t
    for i in range(len(control_points)):
        curvatures = []
        for t_i in t:
            point = bezier_point(t_i, control_points[i])
            dt = bezier_point(t_i, derivatives_cp[1][i])
            ddt = bezier_point(t_i, derivatives_cp[2][i])
            # Radius of curvature
            curvature = bezier_curvature(dt[0], dt[1], ddt[0], ddt[1])
            curvatures.append(curvature)
            # radius = 1 / curvature
            # Normalize derivative
            dt /= np.linalg.norm(dt, 2)
            # tangent = np.array([point, point + dt])
            normal = np.array([point, point - np.array([- dt[1], dt[0]]) * curvature * 3])
            # curvature_center = point + np.array([- dt[1], dt[0]]) * radius
            # circle = plt.Circle(tuple(curvature_center), radius, color=(0, 0.8, 0.8), fill=False, linewidth=1)
            # ax.plot(point[0], point[1])
            # ax.plot(tangent[:, 0], tangent[:, 1])
            ax.plot(normal[:, 0], normal[:, 1], color='green', linewidth=1.5)
            normals.append(point - np.array([- dt[1], dt[0]]) * curvature * 3)
        maxCurIndex = np.argmax(np.array(curvatures))
        maxt_i = t_max[i]
        dt = bezier_point(maxt_i, derivatives_cp[1][i])
        ddt = bezier_point(maxt_i, derivatives_cp[2][i])
        curvature_max_i = bezier_curvature(dt[0], dt[1], ddt[0], ddt[1])
        # point_max = bezier_point(t[maxCurIndex], control_points[i])
        point_max = bezier_point(maxt_i, control_points[i])
        # maxt_i = maxT(control_points[i])
        # print(t[maxCurIndex], maxt_i)
        # dt = bezier_point(t[maxCurIndex], derivatives_cp[1][i])
        dt /= np.linalg.norm(dt, 2)
        normal = np.array([point_max, point_max - np.array([- dt[1], dt[0]]) * curvature_max_i * 3])
        ax.plot(normal[:, 0], normal[:, 1], color='red', linewidth=3.5)
    
    ax.plot(np.array(normals)[:, 0], np.array(normals)[:, 1], c='blue', linewidth=2)
    # ax.plot(np.array(normals)[60:, 0], np.array(normals)[60:, 1], c='g')
    # ax.add_artist(circle)
    # ax.axis("equal")
    # ax.grid(True)

ind = None
def get_ind_under_point(event):
    #get the index of the vertex under point if within epsilon tolerance
    xt,yt = np.array(x),np.array(y)
    d = np.sqrt((xt-event.xdata)**2 + (yt-event.ydata)**2)
    indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
    ind = indseq[0]
    if d[ind] >=0.1:
        ind = None
    return ind

#get the point by pressing
def on_mouse_press(event):
    global ind
    if event.inaxes == None:
        print("none")
        return 
    if event.button == 1:
        global input_points
        # input_points = pcdProcess.loadData(point_sia_hill_Astar).tolist()
        # ax.scatter(np.array(input_points)[:,0],np.array(input_points)[:,1], c='r')
        ax.scatter(event.xdata,event.ydata, c='r')
        point = []
        point.append(event.xdata)
        point.append(event.ydata)
        input_points.append(point)
        fig.canvas.draw()
        # print(input_points)
    elif event.button == 3:
        # print(input_points)
        xt, yt = np.array(input_points)[:,0], np.array(input_points)[:,1]
        d = np.sqrt((xt-event.xdata)**2 + (yt-event.ydata)**2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]
        if d[ind] >=0.1:
            ind = None

def mouse_motion_callback(event):
    global ind, curve_fun
    #on mouse movement
    if event.button != 3 or ind == None or event.inaxes == None or dynamic_plot == True: 
        return

    input_points.pop(ind)
    input_points.insert(ind, [event.xdata, event.ydata])

    control_points, t_max = curve_fun(input_points, max_iter = 50)
    if not dynamic_plot:
        # print(control_points, len(control_points))
        curve_inters = bezier_curve(control_points, n_points = 100)
        derivatives_cp = bezier_derivatives_control_points(control_points, 2)
        
        # print(derivatives_cp)
        # print(control_points)
        control_points_list = []
        # print(control_points)
        for items in control_points:
            for item in items:
                control_points_list.append(item.tolist())
        t = np.linspace(0, 1, 200)
        plt.cla()
        # plt.xlim(0, 15)
        # plt.ylim(0, 15)
        # plt.xlim(-7, 3)
        # plt.ylim(-1, 11)
        # plt.scatter(basemap[:,0],basemap[:,1],cmap='RdYlGn_r',c=0.3*basemap[:,2]+basemap[:,3],s=5,marker='o')
        plot_features(t, t_max, control_points, derivatives_cp)
        plot_figure(input_points, control_points_list, curve_inters)
        fig.canvas.draw()

def on_key_press(event):
    global curve_fun
    print('input points:', input_points)
    if event.key == ' ':
        control_points, t_max, _, _, _, _, _ = curve_fun(input_points, max_iter = 50)
        if not dynamic_plot:
            # print(control_points, len(control_points))
            curve_inters = bezier_curve(control_points, n_points = 100)
            derivatives_cp = bezier_derivatives_control_points(control_points, 2)
            
            # print(derivatives_cp)
            # print(control_points)
            control_points_list = []
            # print(control_points)
            for items in control_points:
                for item in items:
                    control_points_list.append(item.tolist())
            # plt.scatter(basemap[:,0],basemap[:,1],cmap='RdYlGn_r',c=0.3*basemap[:,2]+basemap[:,3],s=5,marker='o')
            t = np.linspace(0, 1, 80)
            plot_features(t, t_max, control_points, derivatives_cp)
            plot_figure(input_points, control_points_list, curve_inters)
            fig.canvas.draw()
            # plt.show()
    elif event.key == 'escape':
        plt.cla()
        plt.xlim(0, 15)
        plt.ylim(0, 15)
        # plt.xlim(-7, 3)
        # plt.ylim(-1, 11)
        fig.canvas.draw()
        input_points.clear()

def plot_curve_change(ctrls, t_max, iter_i, consumed_time, max_iter, iter_type, cecm=0, cecc=0):
    global curve_fun
    if iter_type == 'all':
        pass
    elif iter_type == 'even':
        if iter_i%2 == 1: return
    elif iter_type == 'odd':
        if iter_i%2 == 0: return
    curve_inters = bezier_curve(ctrls, n_points = 100)
    derivatives_cp = bezier_derivatives_control_points(ctrls, 2)
    control_points_list = []
    for items in ctrls:
        for item in items:
            control_points_list.append(item.tolist())
    t = np.linspace(0, 1, 80)
    plot_features(t, t_max, ctrls, derivatives_cp)
    plot_figure(input_points, control_points_list, curve_inters)
    # ax.set_title("curve generated by " + curve_fun.__name__ + "  iter number:" + str(iter_i), fontsize = 12, pad = 10)
    ax.set_title("iteration number:" + str(iter_i) + "  consumed time:" + str('%.3f'%(consumed_time*1000)) + 'ms\n' +\
        "current CECM:" + str('%.5f'%(cecm)) + "  current CECC:" + str('%.5f'%(cecc)), fontsize = 12, pad = 10, c='yellow')

    # plt.axis("tight")
    plt.pause(1)
    if (iter_type == 'all' or iter_type == 'odd') and iter_i < max_iter - 1:
        ax.cla()
        plt.xlim(0, 15)
        plt.ylim(0, 15)
    elif iter_type == 'even' and iter_i < max_iter - 2:
        ax.cla()
        plt.xlim(0, 15)
        plt.ylim(0, 15)
    else:
        plt.show() 

def kCurveClosed(pts, max_iter = 50, iter_type = 'all'):

    consumed_time = 0
    
    pts = np.array(pts)
    n = len(pts)
    # return bezier control points, a vector of triple of 2d points, {{q0,q1,q2}, {q0,q1,q2}, ...}
    ctrls = np.ones([n,3]).tolist()
    for i in range(n):
        ctrls[i][0] = (pts[i] + pts[mod(i - 1, n)]) / 2
        ctrls[i][1] = pts[i]
        ctrls[i][2] = (pts[i] + pts[mod(i + 1, n)]) / 2

    t, ld = [0 for i in range(n)], [0 for i in range(n)]
    matA = np.mat(np.zeros([n, n]))
    matB = np.mat(pts)

    for iter_i in range(max_iter):
        time_start = time.time()
        for i in range(n):
            t[i] = maxParam(ctrls[i][0], pts[i], ctrls[i][2])
            ld[i] = lamb(ctrls[i][0], ctrls[i][1], ctrls[mod(i + 1, n)][1], ctrls[mod(i + 1, n)][2])
    
        for i in range(n):
            matA[i, mod(i - 1, n)] = (1 - ld[mod(i - 1, n)]) * (1 - t[i]) * (1 - t[i])
            matA[i, i] = ld[mod(i - 1, n)] * (1 - t[i]) * (1 - t[i]) + 2 * (1 - t[i]) * t[i] + (1 - ld[i]) * t[i] * t[i]
            matA[i, mod(i + 1, n)] = ld[i] * t[i] * t[i]

        # linear solver of Ax=B
        # print(matA, matB)
        corners = linalg.solve(matA, matB)
        for i in range(n):
            ctrls[i][1] = corners[i]
        for i in range(n):
            ctrls[mod(i + 1, n)][0] = ctrls[i][2] = (1 - ld[i]) * ctrls[i][1] + ld[i] * ctrls[mod(i + 1, n)][1]

        time_end = time.time()
        consumed_time += (time_end - time_start)
        if dynamic_plot: plot_curve_change(ctrls, iter_i, consumed_time, max_iter, iter_type)

    t_all = t

    return ctrls

def kCurveOpen(pts, max_iter = 100, iter_type = 'all'):
    pts = np.array(pts)
    n = len(pts) - 2
    # return bezier control points, a vector of triple of 2d points, {{q0,q1,q2}, {q0,q1,q2}, ...}
    ctrls = np.ones([n,3]).tolist()
    for i in range(1, n + 1):
        ctrls[i - 1][0] = (pts[i] + pts[mod(i - 1, n)]) / 2
        ctrls[i - 1][1] = pts[i]
        ctrls[i - 1][2] = (pts[i] + pts[mod(i + 1, n)]) / 2

    ctrls[0][0] = pts[0]
    ctrls[n - 1][2] = pts[n + 1]

    t, ld = [0 for i in range(n)], [0 for i in range(n - 1)]
    matA = np.mat(np.zeros([n, n]))
    matB = pts[1:-1]*1.0
    
    max_dis, aver_dis, t_all, t_total, curvatures_bias, iters = [], [], [], 0, [], 0
    # point_maxs = []
    # curvature_bias = []
    # for j in range(len(t)):
    #     point_maxs.append(bezier_point(t[j] , ctrls[j]))
    # for j in range(len(t)-1):
    #     curvature_bias.append(np.abs(curvature1(ctrls[j], 1) - curvature1(ctrls[j+1], 0)))
    # curvatures_bias.append(np.max(curvature_bias))

    # error_maxima = pts[1:-1] - np.array(point_maxs)
    # dis = np.sqrt(error_maxima[:,0]**2 + error_maxima[:,1]**2)
    # max_dis.append(np.max(dis))
    # aver_dis.append(np.mean(dis))

    for iter_i in range(max_iter):
        point_maxs = []
        curvature_bias = []
        # step 1
        computing_start_time = time.time()
        for i in range(n):
            t[i] = maxParam(ctrls[i][0], pts[i + 1], ctrls[i][2])
        matA[0, 0] = 2 * (1 - t[0]) * t[0] + (1 - ld[0]) * t[0] * t[0]
        matA[0, 1] = ld[0] * t[0] * t[0]
        for i in range(1, n - 1):
            matA[i, mod(i - 1, n)] = (1 - ld[mod(i - 1, n)]) * (1 - t[i]) * (1 - t[i])
            matA[i, i] = ld[mod(i - 1, n)] * (1 - t[i]) * (1 - t[i]) + 2 * (1 - t[i]) * t[i] + (1 - ld[i]) * t[i] * t[i]
            matA[i, mod(i + 1, n)] = ld[i] * t[i] * t[i]
        matA[n - 1, n - 2] = (1 - ld[n - 2]) * (1 - t[n - 1]) * (1 - t[n - 1])
        matA[n - 1, n - 1] = ld[n - 2] * (1 - t[n - 1]) * (1 - t[n - 1]) + 2 * (1 - t[n - 1]) * t[n - 1]

        matB[0] = pts[1] - (1 - t[0]) * (1 - t[0]) * pts[0]
        matB[n - 1] = pts[n] - t[n - 1] * t[n - 1] * pts[n + 1]
        # linear solver of Ax=B
        corners = linalg.solve(matA, matB)
        for i in range(n): 
            ctrls[i][1] = corners[i]
        
        # step 2
        for i in range(n - 1):
            ld[i] = lamb(ctrls[i][0], ctrls[i][1], ctrls[mod(i + 1, n)][1], ctrls[mod(i + 1, n)][2])
            ctrls[mod(i + 1, n)][0] = ctrls[i][2] = (1 - ld[i]) * ctrls[i][1] + ld[i] * ctrls[mod(i + 1, n)][1]

        computing_end_time = time.time()
        computing_time = computing_end_time - computing_start_time
        t_total += computing_time
        t_all.append(t_total)
        
        for j in range(len(t)):
            point_maxs.append(bezier_point(t[j] , ctrls[j]))

        for j in range(len(t)-1):
            curvature_bias.append(np.abs(np.abs(curvature1(ctrls[j], 1)) - np.abs(curvature1(ctrls[j+1], 0))))
        
        current_curvature_bias_max = np.max(curvature_bias)
        curvatures_bias.append(current_curvature_bias_max)
        
        error_maxima = pts[1:-1] - np.array(point_maxs)
        dis = np.sqrt(error_maxima[:,0]**2 + error_maxima[:,1]**2)
        current_dis_max = np.max(dis)
        max_dis.append(current_dis_max)
        aver_dis.append(np.mean(dis))

        if dynamic_plot:
            fig.patch.set_facecolor('black')
            ax.patch.set_facecolor('black')
            plot_curve_change(ctrls, t, iter_i, t_total, max_iter, iter_type, current_curvature_bias_max, current_dis_max)


        if (current_curvature_bias_max < 1e-3 and current_dis_max < 1e-2):
            print("convergence achieved after %d iteration." % (iter_i))
            iters = iter_i
            break

    # print(dis, curvature_bias)
    return ctrls, t, max_dis, t_all, curvatures_bias, current_dis_max, current_curvature_bias_max

def kCurveClosed_M(pts, max_iter = 50, iter_type = 'all'):
    pts = np.array(pts)
    # print(pts)
    n = len(pts)
    # return bezier control points, a vector of triple of 2d points, {{q0,q1,q2}, {q0,q1,q2}, ...}
    t, ld = [0 for i in range(n)], [0.5 for i in range(n)]
    ctrls = np.ones([n,3]).tolist()
    for i in range(n):
        ctrls[i][1] = pts[i]
    for i in range(n):
        ctrls[mod(i + 1, n)][0] = ctrls[i][2] = (1 - ld[i]) * ctrls[i][1] + ld[i] * ctrls[mod(i + 1, n)][1]

    for iter_i in range(max_iter):
        for i in range(n):
            t[i] = maxParam(ctrls[i][0], pts[i], ctrls[i][2])
            ctrls[i][1] = (pts[i] - (1 - t[i]) * (1 - t[i]) * ctrls[i][0] - t[i] * t[i] * ctrls[i][2]) / (2 * t[i] * (1 - t[i]))
            
        for i in range(n):
            ld[i] = lamb(ctrls[i][0], ctrls[i][1], ctrls[mod(i + 1, n)][1], ctrls[mod(i + 1, n)][2])
            ctrls[mod(i + 1, n)][0] = ctrls[i][2] = (1 - ld[i]) * ctrls[i][1] + ld[i] * ctrls[mod(i + 1, n)][1]
            
        if dynamic_plot: plot_curve_change(ctrls, iter_i, max_iter, iter_type)

    # for iter_i in range(max_iter):
    #     for i in range(n):
    #         t[i] = maxParam(ctrls[i][0], pts[i], ctrls[i][2])
    #         ld[i] = lamb(ctrls[i][0], ctrls[i][1], ctrls[mod(i + 1, n)][1], ctrls[mod(i + 1, n)][2])
    #     for i in range(n):
    #         ctrls[mod(i + 1, n)][0] = ctrls[i][2] = (1 - ld[i]) * ctrls[i][1] + ld[i] * ctrls[mod(i + 1, n)][1]
    #     for i in range(n):
    #         ctrls[i][1] = (pts[i] - (1 - t[i]) * (1 - t[i]) * ctrls[i][0] - t[i] * t[i] * ctrls[i][2]) / (2 * t[i] * (1 - t[i]))
    #     for i in range(n):
    #         ctrls[mod(i + 1, n)][0] = ctrls[i][2] = (1 - ld[i]) * ctrls[i][1] + ld[i] * ctrls[mod(i + 1, n)][1]
            
    #     if dynamic_plot: plot_curve_change(ctrls, iter_i, max_iter, iter_type)

    return ctrls

def kCurveOpen_M(pts, max_iter = 50, iter_type = 'all'):
    pts = np.array(pts)
    n = len(pts) - 2
    # return bezier control points, a vector of triple of 2d points, {{q0,q1,q2}, {q0,q1,q2}, ...}
    ctrls = np.ones([n,3]).tolist()
    ctrls[0][0] = pts[0]
    ctrls[-1][2] = pts[-1]
    for i in range(n):
        ctrls[i][1] = pts[i + 1]

    t, ld = [0 for i in range(n)], [0.5 for i in range(n - 1)]
    
    for i in range(n - 1):
        ctrls[i][2] = ctrls[i + 1][0] = (1 - ld[i]) * ctrls[i][1] + ld[i] * ctrls[i + 1][1]
    
    for iter_i in range(max_iter):
        # step 1
        for _ in range(4):
            for i in range(n - 1):
                ld[i] = lamb(ctrls[i][0], ctrls[i][1], ctrls[i + 1][1], ctrls[i + 1][2])
                ctrls[i][2] = ctrls[i + 1][0] = (1 - ld[i]) * ctrls[i][1] + ld[i] * ctrls[i + 1][1] 

        # step 2
        for i in range(n):
            t[i] = maxParam(ctrls[i][0], pts[i + 1], ctrls[i][2])
            ctrls[i][1] = (pts[i + 1] - (1 - t[i]) ** 2 * ctrls[i][0] - t[i] ** 2 * ctrls[i][2]) / (2 * t[i] * (1 - t[i]))
        
        if dynamic_plot: plot_curve_change(ctrls, iter_i, max_iter, iter_type)

    return ctrls

def normalize_angle(angle):
    return angle + 2*np.pi if angle < 0 else angle

def plot_curvatre(control_points):
    curvatures = []
    for ctrl in control_points:
        for t in np.linspace(0, 1, 500):
            curvatures.append(curvature1(ctrl, t))
    return curvatures

def plot_slope(control_points):
    slopes = []
    derivatives_cp = bezier_derivatives_control_points(control_points, 1)
    for ctrl in derivatives_cp[1]:
        for t in np.linspace(0, 1, 100):
            dt = bezier_point(t, ctrl)
            slopes.append(normalize_angle(np.arctan2(dt[1], dt[0])))
    return slopes

curve_fun = kCurveClosed
def main_interactive(curve_func = kCurveClosed):
    global curve_fun
    curve_fun = curve_func
    fig.canvas.mpl_connect("button_press_event", on_mouse_press)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('motion_notify_event', mouse_motion_callback)
    ax.set_title("iteration number:" + str(0) + "  consumed time:" + str('%.3f'%(0*1000)) + 'ms\n' + \
         'current CECM:Nan current CECC:Nan', fontsize = 12, pad = 10, c='yellow')
    plt.show()

def main_static(curve_fun = kCurveClosed, iter_num = 50, curve_show = 'all', ctrl_file = None, base_map = None):

    global input_points

    if ctrl_file != None:
        input_points = pcdProcess.loadData(ctrl_file)
    else:
        # input_points = [[0,0], [0,1], [1,3], [2,2],[2,1], [1,0]]
        # input_points = [[0,0], [0,1], [0,2], [2,2],[2,1], [1,0]]
        # input_points = [[3.725, 1.613], [2.306, 7.012], [9.580, 8.597], [12.064, 4.765], [9.064, 1.632]]
        # input_points = [[2.4354838709677415, 2.330866807610994], [3.5967741935483866, 7.654032014497131], [8.451612903225806, 10.334491090305045], [11.548387096774192, 4.652672908486862], [9.483870967741934, 1.5758079130172153]]
        # input_points_1 = [[3.80, 0.78], [2.06, 2.46], [2.49, 6.20], [4.70, 8.54], [8.70, 8.97], [11.38, 6.93], \
        #     [11.95, 3.72], [10.96, 1.42], [9.35, 0.40], [6.59, 0.57]]
        # input_points = np.array([[0,10,10,20,20,30,30,40,40,-10,-10,  0], [0, 0,10, 0,10,10, 0, 0,20, 20,-20,-20]]).T
        input_points = pcdProcess.loadData(point_quarry_Astar)
        # iter_results, time_results = [], []
        # for i in range(4, 20):
        #     iter_nums, t_totals = [], []
        #     for j in range(10):
        #         input_points = np.c_[np.linspace(0, 15, i), np.random.rand(i,1)*15]
        #         control_points, t_max, max_dis_0, t_all_0, curvatures_bias_0, iters, t_total \
        #              = curve_fun(input_points, max_iter = iter_num, iter_type = curve_show)

        #         iter_nums.append(iters)
        #         t_totals.append(t_total)

        #     iter_nums.remove(np.max(iter_nums))
        #     iter_nums.remove(np.max(iter_nums))
        #     iter_nums.remove(np.min(iter_nums))
        #     iter_nums.remove(np.min(iter_nums))

        #     t_totals.remove(np.max(t_totals))
        #     t_totals.remove(np.max(t_totals))
        #     t_totals.remove(np.min(t_totals))
        #     t_totals.remove(np.min(t_totals))

        #     iter_mean = np.mean(iter_nums)
        #     iter_std = np.std(iter_nums)

        #     time_mean = np.mean(t_totals)
        #     time_std = np.std(t_totals)

        #     iter_results.append([iter_mean, iter_std])
        #     time_results.append([time_mean, time_std])

        # print('********************* results *********************')
        # print(iter_results, time_results)

        # plt.figure()
        # plt.gca().spines['bottom'].set_linewidth(2.2)
        # plt.gca().spines['left'].set_linewidth(2.2)
        # plt.gca().spines['top'].set_linewidth(2.2)
        # plt.gca().spines['right'].set_linewidth(2.2)
        # plt.plot(range(4, 20), np.array(iter_results)[:,0], color = 'fuchsia',label='iter num', marker = '^',  markersize = 12, markeredgecolor='black', linewidth=4.5)
        # plt.xticks(np.arange(4, 20, 2))
        # plt.xticks(fontsize = 24)
        # plt.yticks(fontsize = 24)
        # plt.legend(fontsize = 24)
        # plt.tight_layout()
        # plt.savefig('E:\\NutsroreSync\\PaperWriting\\RAL2023\\manuscript\\images\\iter_results.jpg', dpi=400)

        # plt.figure()
        # plt.gca().spines['bottom'].set_linewidth(2.2)
        # plt.gca().spines['left'].set_linewidth(2.2)
        # plt.gca().spines['top'].set_linewidth(2.2)
        # plt.gca().spines['right'].set_linewidth(2.2)
        # plt.plot(range(4, 20), np.array(time_results)[:,0], color = 'darkorange',label='conv time', marker = 'o',  markersize = 12, markeredgecolor='black', linewidth=4.5)
        # plt.xticks(np.arange(4, 20, 2))
        # plt.xticks(fontsize = 24)
        # plt.yticks(fontsize = 24)
        # plt.legend(fontsize = 24)
        # plt.tight_layout()
        # plt.savefig('E:\\NutsroreSync\\PaperWriting\\RAL2023\\manuscript\\images\\time_results.jpg', dpi=400)

        # plt.show()
        
    control_points, t_max, max_dis_0, t_all_0, curvatures_bias_0, iters, t_total = \
        curve_fun(input_points, max_iter = iter_num, iter_type = curve_show)
    # control_points, t_max, max_dis_1, t_all_1, curvatures_bias_1, iters, t_total = \
    #     curve_fun(input_points_1, max_iter = iter_num, iter_type = curve_show)
    
    path = bezier_curve(control_points, n_points = 100)
    plt.figure()
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2.2)
    ax.spines['left'].set_linewidth(2.2)
    ax.spines['top'].set_linewidth(2.2)
    ax.spines['right'].set_linewidth(2.2)
    plt.plot(path[:,0]*5, path[:,1]*5,linewidth=4, c='darkorange',label='trajectory')
    plt.scatter(input_points[:,0]*5, input_points[:,1]*5,c='g',s=40, label='path points')
    # plt.plot(input_points[:,0], input_points[:,1], c='gray',label='path', linewidth=3)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.legend(fontsize = 18)
    plt.show()
    
    curvatures = plot_curvatre(control_points)
    slopes = plot_slope(control_points)
    pathLength = path_length(input_points)
    curvatures_xy = np.c_[np.linspace(0, pathLength, len(curvatures)), curvatures].tolist()
    
    piece_curvatures = [[curvatures_xy[0]]]
    for i in range(len(curvatures_xy)-1):
        if curvatures_xy[i+1][1] - curvatures_xy[i][1] < 0.01:
            piece_curvatures[-1].append(curvatures_xy[i+1])
        else:
            piece_curvatures.append([curvatures_xy[i+1]])
    
    # # plt.figure()
    # # plt.gca().spines['bottom'].set_linewidth(2.2)
    # # plt.gca().spines['left'].set_linewidth(2.2)
    # # plt.gca().spines['top'].set_linewidth(2.2)
    # # plt.gca().spines['right'].set_linewidth(2.2)
    # # plt.plot(max_dis_0, color = 'fuchsia',label='point number = 5', marker = '^',  markersize = 12, markeredgecolor='black', linewidth=4.5)
    # # plt.xticks(np.arange(0, len(max_dis_0), 2))
    # # plt.xticks(fontsize = 24)
    # # plt.yticks(fontsize = 24)
    # # plt.legend(fontsize = 24)
    # # plt.tight_layout()
    # # plt.savefig('E:\\NutsroreSync\\PaperWriting\\RAL2023\\manuscript\\images\\distance_change.jpg', dpi=400)

    # # plt.figure()
    # # plt.gca().spines['bottom'].set_linewidth(2.2)
    # # plt.gca().spines['left'].set_linewidth(2.2)
    # # plt.gca().spines['top'].set_linewidth(2.2)
    # # plt.gca().spines['right'].set_linewidth(2.2)
    # # plt.plot(max_dis_1, color = 'darkorange',label='point number = 10', marker = 'o',  markersize = 12, markeredgecolor='black', linewidth=4.5)
    # # plt.xticks(np.arange(0, len(max_dis_0), 2))
    # # plt.xticks(fontsize = 24)
    # # plt.yticks(fontsize = 24)
    # # plt.legend(fontsize = 24)
    # # plt.tight_layout()
    # # plt.savefig('E:\\NutsroreSync\\PaperWriting\\RAL2023\\manuscript\\images\\distance_change1.jpg', dpi=400)

    # plt.figure()
    # ax=plt.gca()
    # ax.spines['bottom'].set_linewidth(2.2)
    # ax.spines['left'].set_linewidth(2.2)
    # ax.spines['top'].set_linewidth(2.2)
    # ax.spines['right'].set_linewidth(2.2)
    # plt.plot(t_all_0, color = 'fuchsia',label='point num = 5', marker = '^',  markersize = 12, markeredgecolor='black', linewidth=4.5)
    # # plt.plot(t_all_1, color = 'darkorange',label='point number = 10', marker = 'o',  markersize = 12, markeredgecolor='black', linewidth=3.5)
    # plt.xticks(np.arange(0, len(t_all_0), 2))
    # plt.xticks(fontsize = 24)
    # plt.yticks(fontsize = 24)
    # plt.legend(loc=2, fontsize = 24)
    # plt.tight_layout()
    # plt.savefig('E:\\NutsroreSync\\PaperWriting\\RAL2023\\manuscript\\images\\time_change_5.jpg', dpi=400)

    # plt.figure()
    # ax=plt.gca()
    # ax.spines['bottom'].set_linewidth(2.2)
    # ax.spines['left'].set_linewidth(2.2)
    # ax.spines['top'].set_linewidth(2.2)
    # ax.spines['right'].set_linewidth(2.2)
    # # plt.plot(t_all_0, color = 'fuchsia',label='point number = 5', marker = '^',  markersize = 12, markeredgecolor='black', linewidth=3.5)
    # plt.plot(t_all_1, color = 'darkorange',label='point num = 10', marker = 'o',  markersize = 12, markeredgecolor='black', linewidth=4.5)
    # plt.xticks(np.arange(0, len(t_all_1), 2))
    # plt.xticks(fontsize = 24)
    # plt.yticks(fontsize = 24)
    # plt.legend(loc=2, fontsize = 24)
    # plt.tight_layout()
    # plt.savefig('E:\\NutsroreSync\\PaperWriting\\RAL2023\\manuscript\\images\\time_change_10.jpg', dpi=400)
    # plt.show()

    # plt.figure()
    # ax=plt.gca()
    # ax.spines['bottom'].set_linewidth(2.2)
    # ax.spines['left'].set_linewidth(2.2)
    # ax.spines['top'].set_linewidth(2.2)
    # ax.spines['right'].set_linewidth(2.2)
    # plt.plot(curvatures_bias_0, color = 'fuchsia',label='point number = 5', marker = '^',  markersize = 12, markeredgecolor='black', linewidth=4.5)
    # # plt.plot(curvatures_bias_1, color = 'darkorange',label='point number = 10', marker = 'o',  markersize = 12, markeredgecolor='black', linewidth=3.5)
    # plt.xticks(np.arange(0, len(max_dis_0), 2))
    # plt.xticks(fontsize = 24)
    # plt.yticks(fontsize = 24)
    # plt.legend(fontsize = 24)
    # plt.tight_layout()
    # plt.savefig('E:\\NutsroreSync\\PaperWriting\\RAL2023\\manuscript\\images\\curvature_bias_change.jpg', dpi=400)
    # # plt.show()

    # plt.figure()
    # ax=plt.gca()
    # ax.spines['bottom'].set_linewidth(2.2)
    # ax.spines['left'].set_linewidth(2.2)
    # ax.spines['top'].set_linewidth(2.2)
    # ax.spines['right'].set_linewidth(2.2)
    # # plt.plot(curvatures_bias_0, color = 'fuchsia',label='point number = 5', marker = '^',  markersize = 12, markeredgecolor='black', linewidth=3.5)
    # plt.plot(curvatures_bias_1, color = 'darkorange',label='point number = 10', marker = 'o',  markersize = 12, markeredgecolor='black', linewidth=4.5)
    # plt.xticks(np.arange(0, len(max_dis_0), 2))
    # plt.xticks(fontsize = 24)
    # plt.yticks(fontsize = 24)
    # plt.legend(fontsize = 24)
    # plt.savefig('E:\\NutsroreSync\\PaperWriting\\RAL2023\\manuscript\\images\\curvature_bias_change1.jpg', dpi=400)
    # plt.tight_layout()
    # plt.show()
    
    if not dynamic_plot:
        # print(control_points, len(control_points))
        curve_inters = bezier_curve(control_points, n_points = 100)
        derivatives_cp = bezier_derivatives_control_points(control_points, 2)
        # print(derivatives_cp)
        # print(control_points)
        control_points_list = []
        for items in control_points:
            for item in items:
                control_points_list.append(item.tolist())
        if base_map != None:
            basemap = pcdProcess.loadData(base_map)
            plt.scatter(basemap[:,0],basemap[:,1],cmap='RdYlGn_r',c=0.3*basemap[:,2]+basemap[:,3],s=5,marker='o')

        t = np.linspace(0, 1, 200)
        plot_features(t, t_max, control_points, derivatives_cp)
        plot_figure(input_points, control_points_list, curve_inters)
        ax.set_title("curve generated by " + curve_fun.__name__, fontsize = 12, pad = 10)
        plt.axis("tight")
        # plt.show()
        fig = plt.figure(figsize=(20,4))
        ax1 = plt.gca()
        ax1.spines['bottom'].set_linewidth(3)
        ax1.spines['left'].set_linewidth(3)
        ax1.spines['top'].set_linewidth(3)
        ax1.spines['right'].set_linewidth(3)
        plt.plot(np.linspace(0, pathLength, len(curvatures)), np.abs((curvatures)), c='r', linewidth=4)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        # plt.legend(fontsize = 24)
        plt.tight_layout()
        # plt.savefig('E:\\NutsroreSync\\PaperWriting\\RAL2023\\manuscript\\images\\curvature_profile.jpg', dpi=400)
        # plt.plot(curvatures, linestyle=':')
        # plt.show()
        fig = plt.figure(figsize=(20,4))
        ax2 = plt.gca()
        ax2.spines['bottom'].set_linewidth(3)
        ax2.spines['left'].set_linewidth(3)
        ax2.spines['top'].set_linewidth(3)
        ax2.spines['right'].set_linewidth(3)
        plt.plot(np.linspace(0, pathLength, len(slopes)), slopes, c='g', linewidth=4)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        # plt.legend(fontsize = 24)
        plt.tight_layout()
        # plt.savefig('E:\\NutsroreSync\\PaperWriting\\RAL2023\\manuscript\\images\\slope_profile.jpg', dpi=400) 
        plt.figure(figsize=(20,4))
        ax3 = plt.gca()
        ax3.spines['bottom'].set_linewidth(3)
        ax3.spines['left'].set_linewidth(3)
        ax3.spines['top'].set_linewidth(3)
        ax3.spines['right'].set_linewidth(3)
        for (i, piece_curvature) in enumerate(piece_curvatures):
            plt.plot(np.array(piece_curvature)[:,0], np.array(piece_curvature)[:,1], c='r', linewidth=4)
            if i < len(piece_curvatures) - 1:
                plt.plot([piece_curvatures[i][-1][0], piece_curvatures[i+1][0][0]], \
                    [piece_curvatures[i][-1][1], piece_curvatures[i+1][0][1]], linestyle='--', c='r', linewidth=4)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        # plt.legend(fontsize = 24)
        plt.tight_layout()
        # plt.savefig('E:\\NutsroreSync\\PaperWriting\\RAL2023\\manuscript\\images\\curvature_true_profile.jpg', dpi=400) 
        plt.show()

dynamic_plot = False
# fig, ax = plt.subplots(1,1,figsize=(12,20))
# fig, ax = plt.subplots(1,1,figsize=[18, 8.3])
# fig.patch.set_facecolor('black')
# ax.patch.set_facecolor('black')
# ax.spines['bottom'].set_color('red')
# ax.spines['left'].set_color('red')
# ax.spines['top'].set_color('red')
# ax.spines['right'].set_color('red')
# plt.scatter(data[:,0],data[:,1],cmap='RdGy_r',c=data[:,2]+data[:,5]*3,s=4,marker='o')
input_points = []
# plt.xlim(0, 15) 
# plt.ylim(0, 15)

rootPath_TITS = 'F:\\PC2Win10\\Study\\PHD\\Research\\paper_writting\\TITS2023\\'

result_hill = rootPath_TITS + 'results\\hill\\result_hill.txt'
basemap = pcdProcess.loadData(result_hill)
point_quarry_Astar = rootPath_TITS + 'results\\quarry\\point_quarry_Astar.txt'
point_sia_hill_Astar = rootPath_TITS + 'results\\hill\\point_sia_hill_Astar.txt'
point_planet_Astar = rootPath_TITS + 'results\\planet\\point_planet_Astar.txt'

if __name__ == "__main__":
    # main_interactive(curve_func = kCurveOpen) #kCurveClosed
    main_static(curve_fun = kCurveOpen, iter_num = 100, curve_show = 'all', ctrl_file=point_sia_hill_Astar, base_map = None)   # even odd all